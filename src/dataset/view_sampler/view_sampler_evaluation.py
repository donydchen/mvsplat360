import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union, Optional

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ...evaluation.evaluation_index_generator import IndexEntry
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from ...global_cfg import get_cfg


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int
    test_rules: Optional[Literal["fixed", "v1", "v2"]] = "fixed"


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None | list[IndexEntry]]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: (
                    None
                    if (v is None) or (isinstance(v, list) and len(v) == 0)
                    else [
                        from_dict(
                            IndexEntry,
                            v_item,
                            dacite_config,
                        )
                        for v_item in (v if isinstance(v, list) else [v])
                    ]
                )
                for k, v in json.load(f).items()
            }

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> Union[
        tuple[
            Int64[Tensor, " context_view"],  # indices for context views
            Int64[Tensor, " target_view"],  # indices for target views
        ],
        tuple[
            Int64[Tensor, " context_view"],  # indices for context views
            Int64[Tensor, " target_view"],  # indices for target views
            Int64[Tensor, " target_quantify_view"],  # indices for target quantify views
        ],
        list[IndexEntry],
    ]:
        entries = self.index.get(scene)
        if entries is None:
            raise ValueError(f"No indices available for scene {scene}.")
        out_lists = []
        for entry in entries:
            context_indices = torch.tensor(
                entry.context, dtype=torch.int64, device=device
            )
            target_indices = torch.tensor(
                entry.target, dtype=torch.int64, device=device
            )
            # NOTE: hack for 3-view test using the 2-view index files
            v = get_cfg()["dataset"]["view_sampler"]["num_context_views"]
            if v > len(context_indices) and v == 3:
                left, right = context_indices.unbind(dim=-1)
                context_indices = torch.stack(
                    (left, (left + right) // 2, right), dim=-1
                )

            out_item = IndexEntry(context_indices, target_indices)
            if entry.target_quantify is not None:
                target_quantify_indices = torch.tensor(
                    entry.target_quantify, dtype=torch.int64, device=device
                )
                out_item.target_quantify = target_quantify_indices

                # hack to update full target view here
                if self.cfg.test_rules == "v1":
                    scene_num_views = extrinsics.shape[0]
                    all_views = sorted(entry.context + entry.target_quantify)
                    cur_len = len(all_views)
                    for _ in range(14 - cur_len):
                        pad_item = min(all_views[-1] + 1, scene_num_views - 1)
                        all_views.append(pad_item)
                    out_item.target = torch.tensor(
                        all_views, dtype=torch.int64, device=device
                    )
                    # print(all_views)
                if self.cfg.test_rules == "v2":
                    all_views = sorted(entry.target_quantify * 4 + entry.context)
                    out_item.target = torch.tensor(
                        all_views, dtype=torch.int64, device=device
                    )

            out_lists.append(out_item)

        # support the initial cases
        if len(out_lists) == 1:
            a = out_lists[0]
            out_lists = [a.context, a.target]
            if a.target_quantify is not None:
                out_lists.append(a.target_quantify)
            out_lists = tuple(out_lists)

        return out_lists

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
