from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor
from typing import Optional

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    scale_factor: Optional[float] = None


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg", verbose=False)
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        pred_color = rearrange(prediction.color, "b v c h w -> (b v) c h w")
        gt_color = rearrange(image, "b v c h w -> (b v) c h w")
        if self.cfg.scale_factor is not None:
            pred_color = torch.nn.functional.interpolate(
                            pred_color,
                            scale_factor=self.cfg.scale_factor,
                            mode="bilinear",
                            align_corners=True,
                        )
            gt_color = torch.nn.functional.interpolate(
                            gt_color,
                            scale_factor=self.cfg.scale_factor,
                            mode="bilinear",
                            align_corners=True,
                        )

        loss = self.lpips.forward(
            pred_color,
            gt_color,
            normalize=True,
        )
        return self.cfg.weight * loss.mean()
