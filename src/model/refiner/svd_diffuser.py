from dataclasses import dataclass
from typing import Literal, Optional, Dict
import time

import torch
from einops import rearrange, repeat
from contextlib import nullcontext
import torch.nn as nn
import itertools
from safetensors.torch import load_file as load_safetensors
from collections import OrderedDict
from sgm.models.diffusion import DiffusionEngine

from ...global_cfg import get_cfg
from .utils import HiddenPrints


@dataclass
class SVDDiffusionCfg:
    name: Literal["svd"]
    config_path: str
    ckpt_path: str | None
    use_same_noise_fwd: bool
    verbose: bool
    cond_aug: float
    motion_bucket_id: int | None
    fps_id: int | None
    fit_ckpt: bool
    svd_num_steps: int
    svd_num_frames: int
    svd_clip_cond_type: Literal["concate", "average"]
    test_first_stage: Optional[bool] = False
    gs_feat_scale_factor: Optional[float] = 1. / 8.
    en_and_decode_n_samples_a_time: Optional[int] = None
    test_time_attn_num_splits: Optional[int] = None  # apply local attention for time module in inference
    weight_train_gs_feat_via_enc: Optional[float] = 0.0
    train_gs_feat_enc_type: Literal["gs_rendered", "target_gt"] = "gs_rendered"


class SVDDiffusion(DiffusionEngine):
    """docstring for SVDDiffusion."""

    def __init__(
        self,
        cfg: SVDDiffusionCfg,
        **kwargs,
    ):
        scheduler_config = kwargs.pop("scheduler_config", None)
        # overwrite to have better control
        kwargs["ckpt_path"] = None
        kwargs["sampler_config"].params.verbose = False  # cfg.verbose
        kwargs["sampler_config"].params.num_steps = cfg.svd_num_steps  # 25
        kwargs["sampler_config"].params.guider_config.params.num_frames = cfg.svd_num_frames
        
        with nullcontext() if cfg.verbose else HiddenPrints():
            super().__init__(
                optimizer_config=None,
                scheduler_config=None,
                network_wrapper=None,
                use_ema=False,
                ema_decay_rate=0.9999,
                input_key="jpg",
                log_keys=None,
                no_cond_log=False,
                compile_model=False,
                en_and_decode_n_samples_a_time=cfg.en_and_decode_n_samples_a_time,
                **kwargs,
            )
        self.cfg = cfg
        self.cond_aug = cfg.cond_aug
        self.motion_bucket_id = cfg.motion_bucket_id
        self.fps_id = cfg.fps_id
        self.params_new = []
        self.scheduler_config = scheduler_config

        # use Gaussians rendered feature, no need to forward concate image
        self.use_gs_feat = (
            get_cfg().model.encoder.gaussian_adapter.n_feature_channels > 0
        )
        self.train_gs_feat_via_enc = self.cfg.weight_train_gs_feat_via_enc > 0

        if cfg.ckpt_path is not None:
            self.init_from_ckpt(cfg.ckpt_path)

        # override network config
        self.conditioner.embedders[0].set_clip_cond_type(cfg.svd_clip_cond_type)
        self.conditioner.embedders[0].n_copies = cfg.svd_num_frames

    def forward(self, x, batch):
        extra_kwargs = {}
        gs_feat = None
        if self.use_gs_feat:
            gs_feat = batch.pop("gs_rendered_features", None)
            assert gs_feat is not None, "must provide Gaussian rendered latent."
            extra_kwargs.update(
                {
                    "use_rendered_latent": True,
                    "gs_rendered_features": gs_feat,
                    "gs_feat_scale_factor": self.cfg.gs_feat_scale_factor,
                }
            )
        if self.train_gs_feat_via_enc:
            assert self.use_gs_feat, "must use with feat based branch only"
            extra_kwargs.update(
                {"weight_train_gs_feat_via_enc": self.cfg.weight_train_gs_feat_via_enc}
            )

        loss = self.loss_fn(
            self.model, self.denoiser, self.conditioner, x, batch, **extra_kwargs
        )
        if not self.train_gs_feat_via_enc:
            loss_mean = loss.mean()
            loss_dict = {"loss": loss_mean}
        else:
            loss_svd, loss_gs_feat_enc_mse = loss
            loss_svd = loss_svd.mean()
            loss_mean = loss_svd + loss_gs_feat_enc_mse
            loss_dict = {
                "loss": loss_mean,
                "loss_svd": loss_svd,
                "loss_gs_feat_enc_mse": loss_gs_feat_enc_mse,
            }

        return loss_mean, loss_dict

    def shared_step(self, batch: Dict):
        cond = self.get_input(batch, is_training=True)

        # scale the target view if needed
        tgt_views = rearrange(batch["target_views"], "b v ... -> (b v) ...")
        if self.use_gs_feat:
            tgt_scale_factor = self.cfg.gs_feat_scale_factor * 8.
            if tgt_scale_factor > 1.:
                tgt_views = nn.functional.interpolate(
                    tgt_views,
                    scale_factor=tgt_scale_factor,
                    mode="bilinear",
                    align_corners=True,
                )
        tgt_views = tgt_views * 2.0 - 1.0

        x = self.encode_first_stage(tgt_views)

        loss, loss_dict = self(x, cond)
        return loss, loss_dict

    @torch.no_grad()
    def get_input(self, batch, **kwargs):
        """
        batch["context_views"]: input views
        batch["gs_rendered_views"]: gsplat rendered target views
        batch["gs_rendered_features"]: gsplat rendered hdim features
        batch["target_views"]: ground truch target views; for training only
        """

        # load the data
        gs_img = batch["gs_rendered_views"]
        context_views = batch["context_views"]
        b, v = gs_img.size()[:2]
        assert (
            v == self.cfg.svd_num_frames
        ), f"must render {self.cfg.svd_num_frames} views from gsplat to apply SVD, got {self.v}"

        cond = {}

        # get conditional clip image
        # NOTE: hack to update params inside the conditioner CLIP branch
        self.conditioner.embedders[0].n_cond_frames = context_views.shape[1]
        cond_clip = rearrange(context_views, "b v ... -> (b v) ...")
        cond["cond_frames_without_noise"] = cond_clip.clone() * 2.0 - 1.0

        # get conditional concatanate image
        if self.use_gs_feat:
            assert batch["gs_rendered_features"] is not None, "must provide gaussians rendered features"
            cond["gs_rendered_features"] = batch["gs_rendered_features"]
            # hack a dummy input to match the network archi for cond
            cond_concat = torch.zeros_like(rearrange(gs_img, "b v ... -> (b v) ..."))[
                :1
            ]
        else:
            cond_concat = rearrange(gs_img, "b v ... -> (b v) ...")
            cond_concat = cond_concat.clone() * 2.0 - 1.0
        cond["cond_frames"] = cond_concat + self.cond_aug * torch.randn_like(
            cond_concat
        )

        # use gt target enc feat to tune the gs feat
        if self.train_gs_feat_via_enc:
            assert self.use_gs_feat, "`train_gs_feat_via_enc` only works with gs_feat based conditions"
            if kwargs.get("is_training", False):
                if self.cfg.train_gs_feat_enc_type == "gs_rendered":
                    cond_concat = rearrange(gs_img, "b v ... -> (b v) ...")
                elif self.cfg.train_gs_feat_enc_type == "target_gt":
                    cond_concat = rearrange(
                        batch["target_views"], "b v ... -> (b v) ..."
                    )
            else:  # get a dummy input to match the network archi for testing
                cond_concat = torch.zeros_like(
                    rearrange(gs_img, "b v ... -> (b v) ...")
                )[:1]

            tgt_scale_factor = self.cfg.gs_feat_scale_factor * 8.0
            if tgt_scale_factor > 1.0:
                cond_concat = nn.functional.interpolate(
                    cond_concat,
                    scale_factor=tgt_scale_factor,
                    mode="bilinear",
                    align_corners=True,
                )
            cond_concat = cond_concat.clone() * 2.0 - 1.0
            cond["cond_frames"] = cond_concat + self.cond_aug * torch.randn_like(
                cond_concat
            )

        # get other video conditions
        cond["cond_aug"] = (
            torch.tensor([self.cond_aug]).repeat(b * v).to(self.device)
        )

        if self.motion_bucket_id is None:
            cond["motion_bucket_id"] = None
        else:
            cond["motion_bucket_id"] = (
                torch.tensor([self.motion_bucket_id])
                .repeat(b * v)
                .to(self.device)
            )

        if self.fps_id is None:
            cond["fps_id"] = None
        else:
            cond["fps_id"] = torch.tensor([self.fps_id]).repeat(b * v).to(self.device)

        cond["num_video_frames"] = v
        cond["image_only_indicator"] = torch.zeros(b, v).to(self.device)

        return cond

    @torch.no_grad()
    def test_step(self, batch, batch_idx=None):
        if self.cfg.test_first_stage:
            target_views = batch["target_views"]
            target_views = rearrange(target_views, "b v ... -> (b v) ...")
            tgt_scale_factor = self.cfg.gs_feat_scale_factor * 8.0
            if tgt_scale_factor > 1.0:
                target_views = nn.functional.interpolate(
                    target_views,
                    scale_factor=tgt_scale_factor,
                    mode="bilinear",
                    align_corners=True,
                )
            x = self.encode_first_stage(target_views * 2.0 - 1.0)
            recon_x = self.decode_first_stage(x)
            if tgt_scale_factor > 1.0:
                recon_x = nn.functional.interpolate(
                    recon_x,
                    scale_factor=(1.0 / tgt_scale_factor),
                    mode="bilinear",
                    align_corners=True,
                )
            samples = torch.clamp((recon_x + 1.0) / 2.0, min=0.0, max=1.0)
            out_dict = {"samples_cfg": samples}
            return out_dict

        randn = batch.pop("randn", None)
        cond = self.get_input(batch)
        gs_feat = cond.pop("gs_rendered_features", None)

        # get conditional and unconditional inputs
        c, uc = self.conditioner.get_unconditional_conditioning(
            cond,
            force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
        )

        sampling_kwargs = {}

        if self.use_gs_feat:
            src_ctx = c.pop("concat", None)
            c["concat"] = nn.functional.interpolate(
                rearrange(gs_feat, "b v ... -> (b v) ..."),
                scale_factor=self.cfg.gs_feat_scale_factor,  # 1.0 / 8.0,
                mode="bilinear",
                align_corners=True,
            )
            uc["concat"] = torch.zeros_like(c["concat"])

        sampling_kwargs["image_only_indicator"] = repeat(cond["image_only_indicator"], "b ... -> (b 2) ...")
        sampling_kwargs["num_video_frames"] = cond["num_video_frames"]
        if self.cfg.test_time_attn_num_splits is not None:
            sampling_kwargs["num_splits"] = self.cfg.test_time_attn_num_splits

        if randn is None:
            b, v, _, h, w = batch["gs_rendered_views"].size()
            randn = torch.randn(
                (b * v, 4, int(h * self.cfg.gs_feat_scale_factor), 
                 int(w * self.cfg.gs_feat_scale_factor)), device=self.device
            )

        # forward sampling
        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **sampling_kwargs
        )
        samples_z = self.sampler(denoiser, randn, c, uc=uc)

        samples_x = self.decode_first_stage(samples_z)

        # rescale the output if needed
        if self.use_gs_feat:
            tgt_scale_factor = self.cfg.gs_feat_scale_factor * 8.0
            if tgt_scale_factor > 1.0:
                samples_x = nn.functional.interpolate(
                    samples_x,
                    scale_factor=(1.0 / tgt_scale_factor),
                    mode="bilinear",
                    align_corners=True,
                )

        out_dict = {}
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        out_dict.update({"samples_cfg": samples})
        return out_dict

    def init_from_ckpt(self, path: str) -> None:
        start_t = time.time()
        print("Loading SD pretrained weight from", path)
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError
        print(f"Loaded pretrained-weight in {(time.time() - start_t):.3f} seconds.")
        start_t = time.time()
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]

        # if removing motion bucket and fps input key
        # update conditioner embedders weight to match new order
        if self.cfg.fit_ckpt:
            print("Fitting pretrained SVD weights to new models...")
            updated_sd = OrderedDict()
            default_cond_keys = [
                "cond_frames_without_noise",
                "fps_id",
                "motion_bucket_id",
                "cond_frames",
                "cond_aug",
            ]
            svd_cond_keys = default_cond_keys.copy()
            if self.cfg.motion_bucket_id is None:
                svd_cond_keys.pop(svd_cond_keys.index("motion_bucket_id"))
            if self.cfg.fps_id is None:
                svd_cond_keys.pop(svd_cond_keys.index("fps_id"))
            for name, params in sd.items():
                if name.startswith("conditioner.embedders"):
                    old_index = int(name.split(".")[2])
                    if default_cond_keys[old_index] not in svd_cond_keys:
                        continue
                    new_index = svd_cond_keys.index(default_cond_keys[old_index])
                    if new_index == old_index:
                        updated_sd[name] = params
                    else:
                        name_list = name.split(".")
                        name_list[2] = str(new_index)
                        updated_sd[".".join(name_list)] = params
                else:  # keep as it
                    updated_sd[name] = params
            sd = updated_sd
            print("svd_cond_keys", svd_cond_keys)
            print("default_cond_keys", default_cond_keys)

            # match the shape
            for name, param in itertools.chain(
                self.named_parameters(), self.named_buffers()
            ):
                new_shape = param.shape
                if name not in sd:
                    if "diffusion" in name:
                        print(f"Manual zero init:{name} with new shape {new_shape} ")
                        new_param = param.clone().zero_()
                        sd[name] = new_param
                        self.params_new.append(name)
                    else:
                        continue
                old_shape = sd[name].shape
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
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        index_size = min(new_param.shape[0], old_param.shape[0])
                        new_param[:index_size] = old_param[:index_size]
                    elif len(new_shape) >= 2:
                        index_o_size = min(new_param.shape[0], old_param.shape[0])
                        index_i_size = min(new_param.shape[1], old_param.shape[1])
                        new_param[:index_o_size, :index_i_size] = old_param[
                            :index_o_size, :index_i_size
                        ]
                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys "
            f"in {(time.time() - start_t):.3f} seconds."
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_params_lrs(self, lr):
        params_old = []
        params_new = []
        for name, params in self.model.named_parameters():
            # params_new is tracked in `init_from_ckpt`
            if 'model.'+name in self.params_new or 'model_ema.'+name in self.params_new:
                params_new.append(params)
            else:
                params_old.append(params)

        # filter only trainable params
        params_old = list(filter(lambda p: p.requires_grad, params_old))
        params_new = list(filter(lambda p: p.requires_grad, params_new))

        params_lrs = []
        if len(params_old) > 0:
            params_lrs.append({"params": params_old, "lr": 1.0 * lr})
        if len(params_new) > 0:
            params_lrs.append({"params": params_new, "lr": 10.0 * lr})

        return params_lrs
