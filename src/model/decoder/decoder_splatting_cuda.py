from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from typing import Optional

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput

from ...global_cfg import get_cfg


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    scale_factor: Optional[float] = None


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        if self.cfg.scale_factor is not None:
            render_image_shape = tuple(map(lambda x: int(x * self.cfg.scale_factor), image_shape))
        else:
            render_image_shape = image_shape

        feature_sh = (
            repeat(gaussians.feature_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v)
            if gaussians.feature_harmonics is not None
            else None
        )
        rendered_out = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            render_image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            gaussian_feature_sh_coefficients=feature_sh,
        )
        if self.cfg.scale_factor is not None:
            for cur_attr in ["color", "feature"]:
                if getattr(rendered_out, cur_attr, None) is not None:
                    setattr(
                        rendered_out,
                        cur_attr,
                        torch.nn.functional.interpolate(
                            getattr(rendered_out, cur_attr),
                            size=image_shape,
                            # scale_factor=(1. / self.cfg.scale_factor),
                            mode="bilinear",
                            align_corners=True,
                        ),
                    )
            for cur_attr in ["mask", "depth"]:
                if getattr(rendered_out, cur_attr, None) is not None:
                    setattr(
                        rendered_out,
                        cur_attr,
                        torch.nn.functional.interpolate(
                            getattr(rendered_out, cur_attr).unsqueeze(1),
                            size=image_shape,
                            # scale_factor=(1.0 / self.cfg.scale_factor),
                            mode="bilinear",
                            align_corners=True,
                        ).squeeze(1),
                    )

        rendered_out_color = rendered_out.color
        # rescale back if encoder is being downscaled
        if get_cfg().model.encoder.downscale_geo_input is not None:
            rendered_out_color = torch.nn.functional.interpolate(
                rendered_out_color,
                scale_factor=get_cfg().model.encoder.downscale_geo_input,
                mode="bilinear",
                align_corners=True,
            )
        color = rearrange(rendered_out_color, "(b v) c h w -> b v c h w", b=b, v=v)
        feature = (
            None
            if rendered_out.feature is None
            else rearrange(rendered_out.feature, "(b v) c h w -> b v c h w", b=b, v=v)
        )
        if depth_mode is not None and depth_mode != "depth":
            depth = self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            )
        else:
            if rendered_out.depth is not None:
                depth = rearrange(rendered_out.depth, "(b v) h w -> b v h w", b=b, v=v)
            else:
                depth = None

        if rendered_out.mask is not None:
            mask = rearrange(rendered_out.mask, "(b v) h w -> b v h w", b=b, v=v)
        else:
            mask = None

        return DecoderOutput(
            color=color,
            depth=depth,
            feature=feature,
            mask=mask,
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
