import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional
import os

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    ori_image_shape: list[int]
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    sort_target_index: Optional[bool] = False
    overfit_max_views: Optional[int] = None
    sort_context_index: Optional[bool] = False
    use_index_to_load_chunk: Optional[bool] = False
    chunk_idx_path: Optional[str] = None
    test_chunk_start: Optional[int] = None
    test_chunk_end: Optional[int] = None
    use_only_indexed_scenes: Optional[bool] = False


class DatasetDL3DV(IterableDataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetDL3DVCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        # assert view_sampler.cfg.name in [
        #     "nnfps",
        #     "evaluation",
        #     "arbitrary",
        # ], "Must use NN FPS or evaluation sampling for DL3DV dataset"
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            if self.cfg.chunk_idx_path is not None:
                print("use chunk idx", self.cfg.chunk_idx_path)
                with open(self.cfg.chunk_idx_path, "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(
                    [os.path.join(root, x) for x in list(set(json_dict.values()))]
                )
                print("chunk length", len(root_chunks))
            elif self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            if self.cfg.test_chunk_start is not None:
                if self.cfg.test_chunk_end is not None:
                    self.chunks = self.chunks[self.cfg.test_chunk_start:self.cfg.test_chunk_end]
                else:
                    self.chunks = self.chunks[self.cfg.test_chunk_start:]
            else:
                self.chunks = self.chunks[:: cfg.test_chunk_interval]

        if self.stage == "val":
            self.chunks = self.chunks * int(1e6 // len(self.chunks))

        # print(self.chunks)
        # print(self.index)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # print(chunk_path)
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                if self.stage == "test":
                    chunk = item
                else:
                    chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            # NOTE: each chunk inside contains roughly 1 scene, we loop over each scene
            # serveral times to avoid frequent disk I/O

            # for example in chunk:
            times_per_scene = (
                self.cfg.test_times_per_scene
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])

                scene = example["key"]

                if self.cfg.use_only_indexed_scenes and (scene not in self.index):
                    continue

                try:
                    extra_kwargs = {}
                    if self.cfg.overfit_to_scene is not None and self.stage != "test":
                        extra_kwargs.update(
                            {
                                "max_num_views": (
                                    148
                                    if self.cfg.overfit_max_views is None
                                    else self.cfg.overfit_max_views
                                )
                            }
                        )
                    # print("extra_kwargs", extra_kwargs)

                    out_data = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                        **extra_kwargs,
                    )
                    if isinstance(out_data, tuple):
                        context_indices, target_indices = out_data[:2]
                        c_list = [
                            (
                                context_indices.sort()[0]
                                if self.cfg.sort_context_index
                                else context_indices
                            )
                        ]
                        t_list = [
                            (
                                target_indices.sort()[0]
                                if self.cfg.sort_target_index
                                else target_indices
                            )
                        ]
                    if isinstance(out_data, list):
                        c_list = [
                            (
                                a.context.sort()[0]
                                if self.cfg.sort_context_index
                                else a.context
                            )
                            for a in out_data
                        ]
                        t_list = [
                            (
                                a.target.sort()[0]
                                if self.cfg.sort_target_index
                                else a.target
                            )
                            for a in out_data
                        ]

                    # reverse the context
                    # context_indices = torch.flip(context_indices, dims=[0])
                    # print(context_indices)
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    # print(scene)
                    # print(get_fov(intrinsics).rad2deg()[0])
                    # assert False
                    continue

                for context_indices, target_indices in zip(c_list, t_list):
                    # Load the images.
                    context_images = [
                        example["images"][index.item()] for index in context_indices
                    ]
                    context_images = self.convert_images(context_images)
                    target_images = [
                        example["images"][index.item()] for index in target_indices
                    ]
                    target_images = self.convert_images(target_images)

                    # Skip the example if the images don't have the right shape.
                    # context_image_invalid = context_images.shape[1:] != (3, 540, 960)
                    # target_image_invalid = target_images.shape[1:] != (3, 540, 960)
                    context_image_invalid = context_images.shape[1:] != tuple(
                        [3, *self.cfg.ori_image_shape]
                    )  # (3, 270, 480)
                    target_image_invalid = target_images.shape[1:] != tuple(
                        [3, *self.cfg.ori_image_shape]
                    )  # (3, 270, 480)

                    if self.cfg.skip_bad_shape and (
                        context_image_invalid or target_image_invalid
                    ):
                        print(
                            f"Skipped bad example {example['key']}. Context shape was "
                            f"{context_images.shape} and target shape was "
                            f"{target_images.shape}."
                        )
                        continue

                    # Resize the world to make the baseline 1.
                    context_extrinsics = extrinsics[context_indices]
                    if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                        a, b = context_extrinsics[:, :3, 3]
                        scale = (a - b).norm()
                        if scale < self.cfg.baseline_epsilon:
                            print(
                                f"Skipped {scene} because of insufficient baseline "
                                f"{scale:.6f}"
                            )
                            continue
                        extrinsics[:, :3, 3] /= scale
                    else:
                        scale = 1

                    nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                    example_out = {
                        "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            "near": self.get_bound("near", len(context_indices))
                            / nf_scale,
                            "far": self.get_bound("far", len(context_indices))
                            / nf_scale,
                            "index": context_indices,
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            "near": self.get_bound("near", len(target_indices))
                            / nf_scale,
                            "far": self.get_bound("far", len(target_indices))
                            / nf_scale,
                            "index": target_indices,
                        },
                        "scene": scene,
                    }
                    if self.stage == "train" and self.cfg.augment:
                        example_out = apply_augmentation_shim(example_out)
                    if self.cfg.image_shape == list(context_images.shape[2:]):
                        yield example_out
                    else:
                        yield apply_crop_shim(example_out, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                if not (root / data_stage).is_dir():
                    continue

                # Load the root's index.
                if self.cfg.chunk_idx_path is not None:
                    # print("use chunk idx", self.cfg.chunk_idx_path)
                    with open(self.cfg.chunk_idx_path, "r") as f:
                        index = json.load(f)
                else:
                    with (root / data_stage / "index.json").open("r") as f:
                        index = json.load(f)
                # print("loaded index", index)
                index = {
                    k: Path(root / data_stage / v)
                    for k, v in index.items()
                    if os.path.basename(v) in [os.path.basename(a) for a in self.chunks]
                }
                # print("chunk", self.chunks)
                # print("updated index", index)
                # print("self chunks", self.chunks)
                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        if self.stage in ['train', 'test']:
            return (
                min(
                    len(self.index.keys()) * self.cfg.test_times_per_scene,
                    self.cfg.test_len,
                )
                if self.stage == "test" and self.cfg.test_len > 0
                else len(self.index.keys()) * self.cfg.train_times_per_scene
            )
        else:
            # set a very large value here to ensure the validation keep going
            # and do not exhaust; it will be wrap to length 1 anyway.
            return int(1e10)
