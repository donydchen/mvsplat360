from typing import Dict, Optional, Tuple
from einops import rearrange

import torch
import torch.nn as nn

from sgm.modules.encoders.modules import GeneralConditioner
from sgm.modules.diffusionmodules.denoiser import Denoiser
from sgm.modules.diffusionmodules.loss import StandardDiffusionLoss as SDLoss


class StandardDiffusionLoss(SDLoss):
    """docstring for StandardDiffusionLoss."""
    def __init__(self, **kwargs):
        super(StandardDiffusionLoss, self).__init__(**kwargs)

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
        use_rendered_latent: Optional[bool] = False,
        gs_rendered_features: Optional[torch.Tensor] = None,
        gs_feat_scale_factor: Optional[float] = 0.125,
        weight_train_gs_feat_via_enc: Optional[float] = 0.0,
    ) -> torch.Tensor | Tuple[torch.Tensor]:
        cond = conditioner(batch)

        tgt_enc_feat = None
        if weight_train_gs_feat_via_enc > 0:
            tgt_enc_feat = cond.pop("concat").clone().detach()

        if use_rendered_latent:
            # reshape the gaussian rendered features
            assert gs_rendered_features is not None, "must provide gs rendered features"
            cond["concat"] = nn.functional.interpolate(
                rearrange(gs_rendered_features, "b v ... -> (b v) ..."),
                scale_factor=gs_feat_scale_factor,  # 1.0 / 8.0,
                mode="bilinear",
                align_corners=True,
            )

        svd_loss = self._forward(network, denoiser, cond, input, batch)

        if weight_train_gs_feat_via_enc > 0:
            delta = cond["concat"][:, : tgt_enc_feat.shape[1]] - tgt_enc_feat
            loss_gs_feat_enc_mse = weight_train_gs_feat_via_enc * (delta**2).mean()
            return svd_loss, loss_gs_feat_enc_mse
        else:
            return svd_loss
