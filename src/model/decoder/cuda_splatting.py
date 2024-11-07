from math import isqrt
from typing import Literal
from dataclasses import dataclass

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points
from ..encoder.costvolume.conversions import depth_to_relative_disparity
from ...misc.sh_rotation import eval_sh

from ...global_cfg import get_cfg


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


@dataclass
class RenderOutput:
    color: Float[Tensor, "batch 3 height width"] | None
    feature: Float[Tensor, "batch channels height width"] | None
    mask: Float[Tensor, "batch height width"] | None
    depth: Float[Tensor, "batch height width"] | None


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_feature_sh_coefficients: (
        Float[Tensor, "batch gaussian channels d_feature_sh"] | None
    ) = None,
    scale_invariant: bool = True,
    use_sh: bool = True,
) -> RenderOutput:  # Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    # _, _, _, n = gaussian_sh_coefficients.shape
    # degree = isqrt(n) - 1
    # shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    degree = 0
    shs = None
    features = None
    colors_precomp = None
    if use_sh:
        if gaussian_sh_coefficients is not None:
            degree = isqrt(gaussian_sh_coefficients.shape[-1]) - 1
            shs = rearrange(
                gaussian_sh_coefficients, "b g xyz n -> b g n xyz"
            ).contiguous()
        if gaussian_feature_sh_coefficients is not None:
            # TODO implement general feature SH conversion in CUDA rasterizer
            campos = extrinsics[:, :3, 3]
            dir_pp = gaussian_means - campos.unsqueeze(1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
            features = 0.5 + eval_sh(
                isqrt(gaussian_feature_sh_coefficients.shape[-1]) - 1,
                gaussian_feature_sh_coefficients,
                dir_pp_normalized,
            )
    else:
        if gaussian_sh_coefficients is not None:
            colors_precomp = gaussian_sh_coefficients[..., 0]
        if gaussian_feature_sh_coefficients is not None:
            features = gaussian_feature_sh_coefficients[..., 0]

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_feature_maps = []
    all_alpha_masks = []
    all_depth_maps = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        extra_kwargs = {"features": features[i] if features is not None else None}
        if get_cfg().use_legacy_renderer:
            extra_kwargs = {}
        rasterizer_out = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            # colors_precomp=None if use_sh else shs[i, :, 0, :],
            colors_precomp=colors_precomp[i] if colors_precomp is not None else None,
            # features=features[i] if features is not None else None,
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
            **extra_kwargs,
        )
        if len(rasterizer_out) == 5:
            image, feature_map, alpha_mask, depth_map, radii = rasterizer_out
        else: 
            image, radii = rasterizer_out
            feature_map = None
            alpha_mask = None
            depth_map = None

        all_images.append(image)
        all_feature_maps.append(feature_map)
        all_alpha_masks.append(None if alpha_mask is None else alpha_mask.squeeze(0))
        all_depth_maps.append(None if depth_map is None else depth_map.squeeze(0))
        all_radii.append(radii)
    all_images = torch.stack(all_images) if all_images[0] is not None else None
    all_feature_maps = (
        torch.stack(all_feature_maps) if all_feature_maps[0] is not None else None
    )
    all_alpha_masks = (
        torch.stack(all_alpha_masks) if all_alpha_masks[0] is not None else None
    )
    all_depth_maps = (
        torch.stack(all_depth_maps) if all_depth_maps[0] is not None else None
    )
    return RenderOutput(all_images, all_feature_maps, all_alpha_masks, all_depth_maps)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_feature_sh_coefficients: (
        Float[Tensor, "batch gaussian channels d_feature_sh"] | None
    ) = None,
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # _, _, _, n = gaussian_sh_coefficients.shape
    # degree = isqrt(n) - 1
    # shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()
    degree = 0
    shs = None
    features = None
    colors_precomp = None
    if use_sh:
        if gaussian_sh_coefficients is not None:
            degree = isqrt(gaussian_sh_coefficients.shape[-1]) - 1
            shs = rearrange(
                gaussian_sh_coefficients, "b g xyz n -> b g n xyz"
            ).contiguous()
        if gaussian_feature_sh_coefficients is not None:
            # TODO implement general feature SH conversion in CUDA rasterizer
            campos = extrinsics[:, :3, 3]
            dir_pp = gaussian_means - campos.unsqueeze(1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
            features = 0.5 + eval_sh(
                isqrt(gaussian_feature_sh_coefficients.shape[-1]) - 1,
                gaussian_feature_sh_coefficients,
                dir_pp_normalized,
            )
    else:
        if gaussian_sh_coefficients is not None:
            colors_precomp = gaussian_sh_coefficients[..., 0]
        if gaussian_feature_sh_coefficients is not None:
            features = gaussian_feature_sh_coefficients[..., 0]

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        extra_kwargs = {"features": features[i] if features is not None else None}
        if get_cfg().use_legacy_renderer:
            extra_kwargs = {}
        rasterizer_out = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            # colors_precomp=None if use_sh else shs[i, :, 0, :],
            colors_precomp=colors_precomp[i] if colors_precomp is not None else None,
            # features=features[i] if features is not None else None,
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
            **extra_kwargs,
        )
        if len(rasterizer_out) == 5:
            image, feature_map, mask, depth_map, radii = rasterizer_out
        else:
            image, radii = rasterizer_out
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "relative_disparity":
        fake_color = depth_to_relative_disparity(
            fake_color, near[:, None], far[:, None]
        )
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    ).color
    return result.mean(dim=1)
