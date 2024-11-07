from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Literal
import math

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
from tqdm import tqdm


from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_psnr
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..misc.utils import split_into_chunks
from ..visualization.annotation import add_label

from ..visualization.camera_trajectory.stabilize import render_stabilization_path
from ..visualization.layout import add_border, hcat, vcat
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .refiner import SVDDiffusion


@dataclass
class OptimizerCfg:
    lr: float
    refiner_lr: float
    warm_up_steps: int
    sched_method: Literal["one_cycle", "linear_lr"]


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_context: bool
    save_combined: bool
    save_gt: bool
    eval_time_skip_steps: int
    dec_chunk_size: Optional[int] = None
    refiner_overlap: Optional[int] = 0
    refiner_loop_times: Optional[int] = 1
    save_video_pts_times: Optional[float] = 1.
    render_stable_poses: Optional[bool] = False
    stable_k_size: Optional[int] = 45
    save_nn_gt: Optional[bool] = False


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_save_model: bool
    eval_data_length: int
    eval_deterministic: bool
    tune_mvsplat: bool
    val_log_images_to_video: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    refiner: None | SVDDiffusion
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        refiner: None | SVDDiffusion,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        if not self.train_cfg.tune_mvsplat:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.refiner = refiner
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        # FIXME: gaussians splatting does not support float16...
        with (
            torch.set_grad_enabled(self.train_cfg.tune_mvsplat),
            torch.cuda.amp.autocast(enabled=False),
        ):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                False,
                scene_names=batch["scene"],
            )
            output = self.decoder(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )

        # use diffusion model to refine the output
        if self.refiner is not None:
            refiner_input = {
                "context_views": batch["context"]["image"],
                "target_views": batch["target"]["image"],
                "gs_rendered_views": output.color,
                "gs_rendered_features": output.feature,
            }
            if get_cfg().model.refiner.name == "svd":
                refiner_loss, refiner_loss_dict = self.refiner.shared_step(
                    refiner_input
                )
            else:
                raise Exception(
                    f"training for {get_cfg().model.refiner.name} "
                    f"has not yet implemented."
                )

        # Compute metrics.
        target_gt = batch["target"]["image"]
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        if self.train_cfg.tune_mvsplat:
            for loss_fn in self.losses:
                loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss
        if self.refiner is not None:
            total_loss = total_loss + refiner_loss
            for k, v in refiner_loss_dict.items():
                updated_k = f"loss/refiner_{k.split('/')[-1]}"
                self.log(updated_k, v)
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene'][:2]]}; "
                f"context = {batch['context']['index'][:2].tolist()}; "
                f"target = {batch['target']['index'][:2].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Render Gaussians.
        with torch.cuda.amp.autocast(enabled=False):
            with self.benchmarker.time("encoder"):
                gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.run_decoder_in_chunk(
                    batch, gaussians, self.test_cfg.dec_chunk_size,
                    stable_camera=self.test_cfg.render_stable_poses,
                )

        # NOTE: use diffusion net to refine the GS renderred outputs
        rgb_refined = None
        all_context = batch["context"]["image"]
        all_gs_color = None if output is None else output.color
        if self.refiner is not None:
            all_gs_feat = None if output is None else output.feature
            n_tgt_views = all_gs_color.shape[1]
            n_svd_frames = get_cfg().model.refiner.svd_num_frames
            # print(all_context.shape)
            n_chunk_frames = n_svd_frames
            gs_feat_scale_factor = get_cfg().model.refiner.gs_feat_scale_factor
            # print(n_tgt_views, n_svd_frames)
            rgb_refined_candidates = []

            for _ in tqdm(range(self.test_cfg.refiner_loop_times), "test refiner...", leave=False):
                rgb_refined_list = []
                chunk_ses = split_into_chunks(
                    n_tgt_views, n_chunk_frames, self.test_cfg.refiner_overlap
                )
                randn = torch.randn((b * n_svd_frames, 4,
                                     int(h * gs_feat_scale_factor),
                                     int(w * gs_feat_scale_factor)), device=self.device)

                for (s_idx, e_idx) in tqdm(
                    chunk_ses,
                    desc="looping refiner",
                    leave=False,
                ):
                    # print(s_idx, e_idx)
                    cur_gs_batch, n_border_patch = self.get_current_chunk(
                        all_gs_color, s_idx, e_idx, n_chunk_frames
                    )
                    cur_gsfeat_batch, _ = self.get_current_chunk(
                        all_gs_feat, s_idx, e_idx, n_chunk_frames
                    )
                    refiner_input = {
                        "context_views": all_context,
                        "gs_rendered_views": cur_gs_batch,
                        "gs_rendered_features": cur_gsfeat_batch,
                        "randn": randn.clone(),
                    }

                    if getattr(get_cfg().model.refiner, "test_first_stage", False):
                        cur_tgt_batch, _ = self.get_current_chunk(
                            batch["target"]["image"], s_idx, e_idx, n_chunk_frames
                        )
                        refiner_input.update({"target_views": cur_tgt_batch})

                    if get_cfg().model.refiner.name == "svd":
                        refiner_output = self.refiner.test_step(refiner_input)
                        rgb_refined = rearrange(
                            refiner_output["samples_cfg"], "(b v) ... -> b v ...", b=b
                        )[0]  # .detach().cpu()
                    else:
                        raise Exception(
                            f"testing for {get_cfg().model.refiner.name} "
                            f"has not yet implemented."
                        )
                    if n_border_patch == 0:
                        n_border_patch = self.test_cfg.refiner_overlap
                    rgb_refined_list.append(
                        rgb_refined[: n_chunk_frames - n_border_patch]
                    )
                rgb_refined_candidates.append(torch.cat(rgb_refined_list, dim=0))

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = all_gs_color[0]  # output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            # save gt for computing scores
            if self.test_cfg.save_gt:
                for index, color in zip(batch["target"]["index"][0], rgb_gt):
                    # out_name = f"{batch_idx:0>3}_{index:0>6}"
                    out_name = f"{index:0>6}"
                    if "index_quantify" in batch["target"] and (
                        index not in batch["target"]["index_quantify"][0]
                    ):
                        continue
                        # out_name = f"{out_name}_quan"
                    save_image(color, path / "ImagesGT" / f"{scene}_{out_name}.png")

            for index, color in zip(batch["target"]["index"][0], images_prob):
                # out_name = f"{batch_idx:0>3}_{index:0>6}"
                out_name = f"{index:0>6}"
                if "index_quantify" in batch["target"] and (index not in batch["target"]["index_quantify"][0]):
                    continue
                    # out_name = f"{out_name}_quan"
                save_image(
                    color, path / "ImagesGSplat" / f"{scene}_{out_name}.png"
                )

            if self.refiner is not None:
                for version_id, cur_rgb_refined in enumerate(rgb_refined_candidates):
                    for index, color in zip(
                        batch["target"]["index"][0], cur_rgb_refined
                    ):
                        # out_name = f"{batch_idx:0>3}_{index:0>6}"
                        out_name = f"{index:0>6}"
                        if (
                            "index_quantify" in batch["target"]
                            and (index not in batch["target"]["index_quantify"][0])
                        ):
                            continue
                            # out_name = f"{out_name}_quan"
                        save_image(
                            color,
                            path / f"ImagesRefined{version_id}" / f"{scene}_{out_name}.png",
                        )

        # Save context images
        if self.test_cfg.save_context:
            for c_idx in range(all_context.shape[1]):
                save_image(
                    all_context[0, c_idx],
                    path / "ImageContext"
                    / f"{scene}_input{c_idx}_{batch['context']['index'][0, c_idx]:0>6}.png",
                )

        # save video
        if self.test_cfg.save_video:
            rgb_refined = rgb_refined_candidates[0]
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "VideoGSplat" / f"{scene}_frame_{frame_str}.mp4",
                pts_times=self.test_cfg.save_video_pts_times,
            )
            if self.refiner is not None:
                save_video(
                    [a for a in rgb_refined],
                    path / "VideoRefined" / f"{scene}_frame_{frame_str}.mp4",
                    pts_times=self.test_cfg.save_video_pts_times
                )

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        with torch.cuda.amp.autocast(enabled=False):
            gaussians_softmax = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
            output_softmax = self.decoder(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
        rgb_softmax = output_softmax.color[0]
        scores_dict = {"gsplat": rgb_softmax}

        if self.refiner is not None:
            refiner_input = {
                "context_views": batch["context"]["image"],
                "gs_rendered_views": output_softmax.color,
                "gs_rendered_features": output_softmax.feature,
            }
            if get_cfg().model.refiner.name == "svd":
                refiner_output = self.refiner.test_step(refiner_input)
                rgb_refined = rearrange(
                    refiner_output["samples_cfg"], "(b v) ... -> b v ...", b=b
                )[0]
            else:
                raise Exception(
                    f"training for {get_cfg().model.refiner.name} "
                    f"has not yet implemented."
                )
            scores_dict.update({"refined": rgb_refined})

        # Construct comparison image.
        rgb_gt = batch["target"]["image"][0]
        comparison = [
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt[:4]), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax[:4]), "Target (GSplat)"),
        ]
        if self.refiner is not None:
            comparison.append(add_label(vcat(*rgb_refined[:4]), "Target (Refined)"))
        comparison = hcat(*comparison)
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Log comparison videos
        if self.train_cfg.val_log_images_to_video:
            video_frames = []
            for v_idx in range(rgb_gt.shape[0]):
                image_list = [add_label(rgb_softmax[v_idx], "GSplat")]
                if self.refiner is not None:
                    image_list.append(add_label(rgb_refined[v_idx], "Refined"))
                image_list.append(add_label(rgb_gt[v_idx], "Ground Truth"))
                video_frames.append(hcat(*image_list))
            self.log_images_to_video(video_frames, "comparison", loop_reverse=True)

    def run_decoder_in_chunk(self, batch, gaussians, chunk_size=None, depth_mode=None,
                             stable_camera=False):
        h, w = batch["context"]["image"].shape[-2:]
        v = batch["target"]["extrinsics"].shape[1]
        b = batch["target"]["extrinsics"].shape[0]
        assert b == 1, "only do batch_size=1 for now."
        camera_poses = batch["target"]["extrinsics"]
        if stable_camera:
            stable_poses = render_stabilization_path(
                camera_poses[0].detach().cpu().numpy(),
                k_size=self.test_cfg.stable_k_size,
            )
            stable_poses = list(map(lambda x: np.concatenate((x, np.array([[0., 0., 0., 1.]])), axis=0), stable_poses))
            stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(camera_poses)
            camera_poses = stable_poses.unsqueeze(0)

        if chunk_size is None:
            chunk_size = v
        chunk_size = min(v, chunk_size)
        for chunk_idx in tqdm(
            range(math.ceil(v / chunk_size)),
            desc="looping decoder",
            leave=False,
        ):
            s = int(chunk_idx * chunk_size)
            e = int((chunk_idx + 1) * chunk_size)
            output_chunk = self.decoder(
                gaussians,
                camera_poses[:, s:e],
                batch["target"]["intrinsics"][:, s:e],
                batch["target"]["near"][:, s:e],
                batch["target"]["far"][:, s:e],
                (h, w),
                depth_mode=depth_mode,
            )
            if chunk_idx == 0:
                output = output_chunk
            else:
                for attr in ["color", "depth", "feature", "mask"]:
                    if getattr(output_chunk, attr) is not None:
                        setattr(
                            output,
                            attr,
                            torch.cat(
                                (getattr(output, attr), getattr(output_chunk, attr)),
                                dim=1,
                            ),
                        )

        return output

    def get_current_chunk(self, batch_input, start_idx, end_idx, chunk_size):
        n_border_patch = 0
        if batch_input is None:
            return None, n_border_patch

        assert (
            batch_input.shape[0] == 1
        ), "currently only used for test time, use batch_size=1"

        cur_gs_batch = batch_input[0, start_idx:end_idx]

        if cur_gs_batch.shape[0] != chunk_size:
            n_border_patch = chunk_size - cur_gs_batch.shape[0]
            cur_gs_batch = torch.cat(
                (
                    cur_gs_batch,
                    repeat(
                        cur_gs_batch[-1:],
                        "b ... -> (b n) ...",
                        n=n_border_patch,
                    ),
                ),
                dim=0,
            )
        return cur_gs_batch.unsqueeze(0), n_border_patch

    def log_images_to_video(self, images, log_name, loop_reverse=True):
        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        try:
            fps = int(get_cfg().dataset.view_sampler.num_target_views)
        except:
            fps = 14
        fps = int(fps / 2.)
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{log_name}": wandb.Video(video[None], fps=fps, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        if self.refiner is None:
            optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
            cur_lr = self.optimizer_cfg.lr
        else:
            params_lrs = (
                [
                    {"params": self.encoder.parameters(), "lr": self.optimizer_cfg.lr},
                ]
                if self.train_cfg.tune_mvsplat
                else []
            )
            params_lrs.extend(
                self.refiner.get_params_lrs(lr=self.optimizer_cfg.refiner_lr)
            )
            optimizer = optim.Adam(params_lrs)
            cur_lr = [p["lr"] for p in params_lrs]

        sched_method = self.optimizer_cfg.sched_method
        if sched_method == "one_cycle":
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                cur_lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )
        elif sched_method == "linear_lr":
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        else:
            raise Exception(f"Unknown sched_method `{sched_method}`")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
