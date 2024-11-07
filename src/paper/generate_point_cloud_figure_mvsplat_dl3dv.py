from pathlib import Path

import hydra
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import default_collate
import json
from tqdm import tqdm

from ..visualization.vis_depth import viz_depth_tensor
from ..dataset.data_module import get_data_shim
import os
from PIL import Image

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.geometry.projection import homogenize_points, project
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import render_cuda_orthographic
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply
    from src.visualization.color_map import apply_color_map_to_image
    from src.visualization.drawing.cameras import unproject_frustum_corners
    from src.visualization.drawing.lines import draw_lines
    from src.visualization.drawing.points import draw_points


# with open("assets/evaluation_index_re10k.json") as f:
#     scene_cfgs = json.load(f)

# supple scenes
SCENES = (
    # scene, context 1, context 2, far plane
    # (
    #     "552646ce086e79d5a9e6296f5559149486be270535257e47291da5b7ada47f42",
    #     0,
    #     30,
    #     60,
    #     90,
    #     120,
    #     6.0,
    #     [110],
    #     1.4,
    #     19,
    # ),
    (
        "165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557",
        79, 122, 196, 271, 306,
        6.0,
        [110],
        1.4,
        19,
    ),
)


FIGURE_WIDTH = 500
MARGIN = 4
GAUSSIAN_TRIM = 8
LINE_WIDTH = 1.8
LINE_COLOR = [255, 0, 0]
POINT_DENSITY = 0.5


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def generate_point_cloud_figure(cfg_dict):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    device = torch.device("cuda:0")

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = ModelWrapper.load_from_checkpoint(
        checkpoint_path,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        refiner=None,
        losses=[],
        step_tracker=None,
    )
    model_wrapper.eval()
    data_shim = get_data_shim(encoder)

    for idx, (scene, *context_indices, far, angles, line_width, cam_div) in enumerate(
        tqdm(SCENES)
    ):
        LINE_WIDTH = line_width
        v_context = len(context_indices)
        # Create a dataset that always returns the desired scene.
        view_sampler_cfg = ViewSamplerArbitraryCfg(
            "arbitrary",
            v_context,
            2,
            context_views=list(context_indices),
            target_views=[0, 0],  # use [40, 80] for teaser
        )
        cfg.dataset.view_sampler = view_sampler_cfg
        cfg.dataset.overfit_to_scene = scene

        # Get the scene.
        dataset = get_dataset(cfg.dataset, "test", None)
        example = default_collate([next(iter(dataset))])
        example = apply_to_collection(example, Tensor, lambda x: x.to(device))
        example = data_shim(example)

        # Generate the Gaussians.
        visualization_dump = {}
        gaussians = encoder.forward(
            example["context"], False, visualization_dump=visualization_dump
        )

        # Figure out which Gaussians to mask off/throw away.
        _, _, _, h, w = example["context"]["image"].shape

        # Transform means into camera space.
        means = rearrange(
            gaussians.means,
            "() (v h w spp) xyz -> h w spp v xyz",
            v=v_context,
            h=h,
            w=w,
        )
        means = homogenize_points(means)
        w2c = example["context"]["extrinsics"].inverse()[0]
        means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3]

        # Create a mask to filter the Gaussians. First, throw away Gaussians at the
        # borders, since they're generally of lower quality.
        mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        # Then, drop Gaussians that are really far away.
        mask = mask & (means[..., 2] < far * 2)

        def trim(element):
            element = rearrange(
                element, "() (v h w spp) ... -> h w spp v ...", v=v_context, h=h, w=w
            )
            return element[mask][None]

        for angle in angles:
            # Define the pose we render from.
            # pose = torch.eye(4, dtype=torch.float32, device=device)
            # rotation = R.from_euler("xyz", [10, -15, 0], True).as_matrix()
            # pose[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
            # translation = torch.eye(4, dtype=torch.float32, device=device)
            # # visual balance, 0.5x pyramid/frustum volume
            # translation[2, 3] = far * (0.5 ** (1 / 3))
            # # translation[2, 3] = far * (0.5 ** (1 / 3))  # * 3.0
            # translation[1, 3] = -0.2
            # translation[0, 3] = -0.5
            # pose = translation @ pose

            pose = torch.eye(4, dtype=torch.float32, device=device)
            rotation = R.from_euler("xyz", [-15, angle - 90, 0], True).as_matrix()
            pose[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
            translation = torch.eye(4, dtype=torch.float32, device=device)
            # visual balance, 0.5x pyramid/frustum volume
            translation[2, 3] = far * (0.5 ** (1 / 3))
            pose = translation @ pose

            ones = torch.ones((1,), dtype=torch.float32, device=device)
            render_args = {
                "extrinsics": example["context"]["extrinsics"][0, :1] @ pose,
                "width": ones * far * 2,
                "height": ones * far * 2,
                "near": ones * 0,
                "far": ones * far,
                "image_shape": (1024, 1024),
                "background_color": torch.zeros(
                    (1, 3), dtype=torch.float32, device=device
                ),
                "gaussian_means": trim(gaussians.means),
                "gaussian_covariances": trim(gaussians.covariances),
                "gaussian_sh_coefficients": trim(gaussians.harmonics),
                "gaussian_opacities": trim(gaussians.opacities),
                # "fov_degrees": 1.5,
            }

            # Render alpha (opacity).
            dump = {}
            alpha_args = {
                **render_args,
                "gaussian_sh_coefficients": torch.ones_like(
                    render_args["gaussian_sh_coefficients"][..., :1]
                ),
                "use_sh": False,
            }
            alpha = render_cuda_orthographic(**alpha_args, dump=dump)[0]

            # Render (premultiplied) color.
            color = render_cuda_orthographic(**render_args)[0]

            # Render depths. Without modifying the renderer, we can only render
            # premultiplied depth, then hackily transform it into straight alpha depth,
            # which is needed for sorting.
            depth = render_args["gaussian_means"] - dump["extrinsics"][0, :3, 3]
            depth = depth.norm(dim=-1)
            depth_args = {
                **render_args,
                "gaussian_sh_coefficients": repeat(depth, "() g -> () g c ()", c=3),
                "use_sh": False,
            }
            depth_premultiplied = render_cuda_orthographic(**depth_args)
            depth = (depth_premultiplied / alpha).nan_to_num(posinf=1e10, nan=1e10)[0]

            # Save the rendering for later depth-based alpha compositing.
            layers = [(color, alpha, depth)]

            # Figure out the intrinsics from the FOV.
            fx = 0.5 / (0.5 * dump["fov_x"]).tan()
            fy = 0.5 / (0.5 * dump["fov_y"]).tan()
            dump_intrinsics = torch.eye(3, dtype=torch.float32, device=device)
            dump_intrinsics[0, 0] = fx
            dump_intrinsics[1, 1] = fy
            dump_intrinsics[:2, 2] = 0.5

            # Compute frustum corners for the context views.
            frustum_corners = unproject_frustum_corners(
                example["context"]["extrinsics"][0],
                example["context"]["intrinsics"][0],
                torch.ones((v_context,), dtype=torch.float32, device=device)
                * far
                / cam_div,
            )
            camera_origins = example["context"]["extrinsics"][0, :, :3, 3]
            # stack the rendered pose for debugging

            # Generate the 3D lines that have to be computed.
            lines = []
            for corners, origin in zip(frustum_corners, camera_origins):
                for i in range(4):
                    lines.append((corners[i], corners[i - 1]))
                    lines.append((corners[i], origin))

            # Generate an alpha compositing layer for each line.
            for line_idx, (a, b) in enumerate(lines):
                # Start with the point whose depth is further from the camera.
                a_depth = (dump["extrinsics"].inverse() @ homogenize_points(a))[..., 2]
                b_depth = (dump["extrinsics"].inverse() @ homogenize_points(b))[..., 2]
                start = a if (a_depth > b_depth).all() else b
                end = b if (a_depth > b_depth).all() else a

                # Create the alpha mask (this one is clean).
                start_2d = project(start, dump["extrinsics"], dump_intrinsics)[0][0]
                end_2d = project(end, dump["extrinsics"], dump_intrinsics)[0][0]
                alpha = draw_lines(
                    torch.zeros_like(color),
                    start_2d[None],
                    end_2d[None],
                    (1, 1, 1),
                    LINE_WIDTH,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                # if line_idx // 8 == 0:
                #     lcolor = [1.0, 0, 0]
                # elif line_idx // 8 == 1:
                #     lcolor = [0, 1.0, 0]
                # else:
                #     lcolor = [0, 0, 1.0]

                # Create the color.
                lc = torch.tensor(
                    LINE_COLOR,
                    dtype=torch.float32,
                    device=device,
                )
                color = draw_lines(
                    torch.zeros_like(color),
                    start_2d[None],
                    end_2d[None],
                    lc,
                    LINE_WIDTH,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                # Create the depth. We just individually render points.
                wh = torch.tensor((w, h), dtype=torch.float32, device=device)
                delta = (wh * (start_2d - end_2d)).norm()
                num_points = delta / POINT_DENSITY
                t = torch.linspace(0, 1, int(num_points) + 1, device=device)
                xyz = start[None] * t[:, None] + end[None] * (1 - t)[:, None]
                depth = (xyz - dump["extrinsics"][0, :3, 3]).norm(dim=-1)
                depth = repeat(depth, "p -> p c", c=3)
                xy = project(xyz, dump["extrinsics"], dump_intrinsics)[0]
                depth = draw_points(
                    torch.ones_like(color) * 1e10,
                    xy,
                    depth,
                    LINE_WIDTH,  # makes it 2x as wide as line
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                layers.append((color, alpha, depth))

            # Do the alpha compositing.
            canvas = torch.ones_like(color)
            colors = torch.stack([x for x, _, _ in layers])
            alphas = torch.stack([x for _, x, _ in layers])
            depths = torch.stack([x for _, _, x in layers])
            index = depths.argsort(dim=0)
            colors = colors.gather(index=index, dim=0)
            alphas = alphas.gather(index=index, dim=0)
            t = (1 - alphas).cumprod(dim=0)
            t = torch.cat([torch.ones_like(t[:1]), t[:-1]], dim=0)
            image = (t * colors).sum(dim=0)
            total_alpha = (t * alphas).sum(dim=0)
            image = total_alpha * image + (1 - total_alpha) * canvas

            base = Path(f"point_clouds/{cfg.wandb['name']}/{idx:0>6}_{scene[:10]}")
            save_image(image, f"{base}_angle_{angle:0>3}.png")

            # also save the premultiplied color for debugging
            # save_image(layers[0][0], f"{base}_angle_{angle:0>3}_raw.png")

            # Render depth.
            *_, h, w = example["context"]["image"].shape
            # rendered = decoder.forward(
            #     gaussians,
            #     example["context"]["extrinsics"],
            #     example["context"]["intrinsics"],
            #     example["context"]["near"],
            #     example["context"]["far"],
            #     (h, w),
            #     "depth",
            # )

            # convert the rotations from camera space to world space as required
            cam_rotations = trim(visualization_dump["rotations"])[0]
            c2w_mat = repeat(
                example["context"]["extrinsics"][0, :, :3, :3],
                "v a b -> h w spp v a b",
                h=h,
                w=w,
                spp=1,
            )
            c2w_mat = c2w_mat[mask]  # apply trim

            cam_rotations_np = R.from_quat(
                cam_rotations.detach().cpu().numpy()
            ).as_matrix()
            world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
            world_rotations = R.from_matrix(world_mat).as_quat()
            world_rotations = torch.from_numpy(world_rotations).to(
                visualization_dump["scales"]
            )

            export_ply(
                example["context"]["extrinsics"][0, 0],
                trim(gaussians.means)[0],
                trim(visualization_dump["scales"])[0],
                world_rotations,
                trim(gaussians.harmonics)[0],
                trim(gaussians.opacities)[0],
                base / "gaussians.ply",
            )

            # save encoder depth map
            depth_vis = (
                (visualization_dump["depth"].squeeze(-1).squeeze(-1)).cpu().detach()
            )
            for v_idx in range(depth_vis.shape[1]):
                vis_depth = viz_depth_tensor(
                    1.0 / depth_vis[0, v_idx], return_numpy=True
                )  # inverse depth
                # save_path = path / scene / f"color/input{v_idx}_depth.png"
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(vis_depth).save(f"{base}_depth_{v_idx}.png")

            # save context views
            for v_idx in range(example["context"]["image"].shape[1]):
                save_image(
                    example["context"]["image"][0, v_idx], f"{base}_input_{v_idx}.png"
                )

            # result = rendered.depth.cpu().detach()
            # print(result.shape)
            # assert False
            # for v_idx in range(result.shape[1]):
            # vis_depth = viz_depth_tensor(
            # 1.0 / result[0, v_idx], return_numpy=True
            # )  # inverse depth
            # save_path = path / scene / f"color/input{v_idx}_depth.png"
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Image.fromarray(vis_depth).save(f"{base}_depth_{v_idx}_gs.png")

            # depth_near = result[result > 0].quantile(0.01).log()
            # depth_far = result.quantile(0.99).log()
            # result = result.log()
            # result = 1 - (result - depth_near) / (depth_far - depth_near)
            # result = apply_color_map_to_image(result, "turbo")
            # save_image(result[0, 0], f"{base}_depth_0_gs.png")
            # save_image(result[0, 1], f"{base}_depth_1_gs.png")
            a = 1
        a = 1
    a = 1


if __name__ == "__main__":
    with torch.no_grad():
        generate_point_cloud_figure()
