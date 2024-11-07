from typing import Dict
from einops import rearrange, repeat, reduce

from sgm.modules.encoders.modules import AbstractEmbModel
from sgm.util import instantiate_from_config


class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):
    """
    Modified from: sgm.modules.encoders.modules.FrozenOpenCLIPImagePredictionEmbedder
    """
    def __init__(
        self,
        open_clip_embedding_config: Dict,
        n_cond_frames: int,
        n_copies: int,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.open_clip = instantiate_from_config(open_clip_embedding_config)
        self.clip_cond_type = "concate"

    def forward(self, vid):
        vid = self.open_clip(vid)
        vid = rearrange(vid, "(b t) d -> b t d", t=self.n_cond_frames)
        if self.clip_cond_type == "average":
            vid = reduce(vid, "b t d -> b () d", "mean")
            # print("average clip", vid.shape)
        vid = repeat(vid, "b t d -> (b s) t d", s=self.n_copies)

        return vid

    def set_clip_cond_type(self, clip_cond_type):
        cond_type_list = ["concate", "average"]
        assert clip_cond_type in cond_type_list, f"Only support types {cond_type_list} for now."
        self.clip_cond_type = clip_cond_type
