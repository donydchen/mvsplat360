from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict
import itertools

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import BackboneMultiview, BackboneUniMatch
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    legacy_2views: bool
    use_legacy_unimatch_backbone: bool
    grid_sample_disable_cudnn: bool
    costvolume_nearest_n_views: Optional[int] = None
    multiview_trans_nearest_n_views: Optional[int] = None
    fit_ckpt: Optional[bool] = False
    depth_upscale_factor: Optional[int] = None
    downscale_geo_input: Optional[int] = None


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        if cfg.use_legacy_unimatch_backbone:
            self.backbone = BackboneUniMatch(
                feature_channels=cfg.d_feature,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_backbone_cross_attn,
                use_epipolar_trans=cfg.use_epipolar_trans,
            )
        else:
            self.backbone = BackboneMultiview(
                feature_channels=cfg.d_feature,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_backbone_cross_attn,
                use_epipolar_trans=cfg.use_epipolar_trans,
            )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=(
                cfg.downscale_factor
                if cfg.depth_upscale_factor is None
                else cfg.depth_upscale_factor
            ),
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            legacy_2views=cfg.legacy_2views,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        ckpt_path = get_cfg().checkpointing.pretrained_model
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path: str) -> None:
        print("Loading encoder pretrained weight from", path)
        if path.endswith("ckpt"):
            enc_dict = torch.load(path, map_location="cpu")["state_dict"]
        else:
            raise NotImplementedError
        if "state_dict" in list(enc_dict.keys()):
            enc_dict = enc_dict["state_dict"]
        # remove the 'encoder.' prefix
        enc_dict = OrderedDict(
            {k[8:]: v for k, v in enc_dict.items() if k.startswith("encoder")}
        )

        # need to update the weight if predict features
        if self.cfg.fit_ckpt:
            print("Fitting pretrained mvsplat encoder weights to new models...")
            for name, param in self.state_dict().items():
                new_shape = param.shape
                if name not in enc_dict:
                    raise Exception(f"Found unknown new weight {name}")
                old_shape = enc_dict[name].shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if new_shape != old_shape:
                    print(
                        f"Manual init:{name} with new shape {new_shape} "
                        f"and old shape {old_shape}"
                    )
                    new_param = param.clone().zero_()
                    old_param = enc_dict[name]
                    if len(new_shape) == 1:
                        index_size = min(new_param.shape[0], old_param.shape[0])
                        new_param[:index_size] = old_param[:index_size]
                    elif len(new_shape) >= 2:
                        index_o_size = min(new_param.shape[0], old_param.shape[0])
                        index_i_size = min(new_param.shape[1], old_param.shape[1])
                        new_param[:index_o_size, :index_i_size] = old_param[
                            :index_o_size, :index_i_size
                        ]
                    enc_dict[name] = new_param

        missing, unexpected = self.load_state_dict(enc_dict, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys "
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        input_images = context["image"].clone().detach()
        if self.cfg.downscale_geo_input is not None:
            input_images = rearrange(
                torch.nn.functional.interpolate(
                    rearrange(input_images, "b v ... -> (b v) ..."),
                    scale_factor=(1.0 / self.cfg.downscale_geo_input),
                    mode="bilinear",
                    align_corners=True,
                ),
                "(b v) ... -> b v ...",
                b=input_images.shape[0]
            )

        device = input_images.device
        b, v, _, h, w = input_images.shape
        if (
            self.cfg.costvolume_nearest_n_views is not None
            or self.cfg.multiview_trans_nearest_n_views is not None
        ):
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)
        else:
            cameras_dist_index = None

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        trans_features, cnn_features = self.backbone(
            input_images,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
            nn_matrix=(
                None
                if cameras_dist_index is None
                else cameras_dist_index[
                    :, :, : self.cfg.multiview_trans_nearest_n_views
                ]
            ),
        )

        # Sample depths from the resulting features.
        # print("trans feat shape", trans_features.shape)
        # print("cnn feat shape", cnn_features.shape)

        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(input_images.clone().detach(), "b v c h w -> (v b) c h w")
        if self.cfg.depth_upscale_factor is not None and (
            self.cfg.depth_upscale_factor != self.cfg.downscale_factor):
            extra_info["images"] = torch.nn.functional.interpolate(
                extra_info["images"],
                scale_factor=(
                    float(self.cfg.depth_upscale_factor) / self.cfg.downscale_factor
                ),
                mode="bilinear",
                align_corners=True,
            )
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel
        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
            nn_matrix=(
                None
                if cameras_dist_index is None
                else cameras_dist_index[:, :, : self.cfg.costvolume_nearest_n_views]
            ),
        )

        # reset the h w if needed
        if self.cfg.depth_upscale_factor is not None and (
            self.cfg.depth_upscale_factor != self.cfg.downscale_factor):
            # print(h, w)
            s = float(self.cfg.downscale_factor) / self.cfg.depth_upscale_factor
            h = int(h / s)
            w = int(w / s)
            # print(h, w)

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            feature_harmonics=(
                None
                if gaussians.feature_harmonics is None
                else rearrange(
                    gaussians.feature_harmonics,
                    "b v r srf spp c d_f_sh -> b (v r srf spp) c d_f_sh",
                )
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
