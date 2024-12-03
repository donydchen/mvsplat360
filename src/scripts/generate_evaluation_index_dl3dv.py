"""
Generate the evaluation index for DL3DV-10K dataset.

python -m src.scripts.generate_evaluation_index_dl3dv --data_dir=datasets/dl3dv/test --num_target_views=56
"""

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import json
import argparse
import os
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

from ..dataset.view_sampler.view_sampler_bounded_v2 import farthest_point_sample


def convert_poses(
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


def partition_list(lst, n_bins):
    if n_bins <= 0:
        raise ValueError("Number of bins must be greater than 0")
    if len(lst) < n_bins:
        raise ValueError("Number of bins cannot exceed the length of the list")

    bin_size = len(lst) // n_bins
    borders = [lst[0]]  # First border is always the first index
    for i in range(1, n_bins):
        border_index = min(
            i * bin_size, len(lst) - 1
        )  # Ensure last bin doesn't exceed list length
        borders.append(lst[border_index])
    borders.append(lst[-1])  # Last border is always the last index
    return borders


def find_train_and_test_index(
    chunk_path,
    scene_name=None,
    num_context_views=5,
    num_target_skip=1,
    num_target_views=28,
    view_selection_ratio=1.0,
):
    chunk = torch.load(chunk_path)
    out_dict = OrderedDict()
    for example in chunk:
        cur_scene_name = example["key"]
        if scene_name is not None and cur_scene_name != scene_name:
            continue

        extrinsics, intrinsics = convert_poses(example["cameras"])
        n_views = extrinsics.shape[0]
        # choose only the first n views for evaluation
        n_views = int(n_views * view_selection_ratio)

        index_context = sorted(
            farthest_point_sample(
                extrinsics[:n_views, :3, -1].unsqueeze(0), num_context_views
            )
            .squeeze(0)
            .tolist()
        )

        index_target_all = [x for x in range(n_views) if x not in index_context]
        index_target_select = partition_list(index_target_all, num_target_views)
        assert (
            len(index_target_select) >= num_target_views
        ), f"double check {cur_scene_name} at {chunk_path}: target len: {len(index_target_select)} from {len(index_target_all)}"
        index_target = index_target_select[:num_target_views]

        out_dict[cur_scene_name] = {"context": index_context, "target": index_target}

    return out_dict


def generate_index_file(args):
    n_ctx = args.num_context_views
    n_tgt = args.num_target_views

    out_dir = f"assets/dl3dv_evaluation"
    os.makedirs(out_dir, exist_ok=True)
    # data_dir = "datasets/DL3DV-10K/dl3dv_benchmark/test"
    data_dir = args.data_dir
    chunk_paths = sorted(glob(os.path.join(data_dir, "*.torch")))  # [:2]
    out_dict_all = OrderedDict()
    for chunk_path in tqdm(chunk_paths):
        out_dict = find_train_and_test_index(
            chunk_path,
            scene_name=None,
            num_context_views=n_ctx,
            num_target_views=n_tgt,
            view_selection_ratio=args.view_selection_ratio,
        )
        out_dict_all.update(out_dict)

    out_name = f"dl3dv_ctx_{n_ctx}v_tgt_{n_tgt}v"
    if args.view_selection_ratio < 1:
        out_name = f"{out_name}_n{int(args.view_selection_ratio * 300)}"
    out_path = os.path.join(out_dir, f"{out_name}.json")

    with open(out_path, "w") as f:
        json.dump(out_dict_all, f)

    print(f"Save index to {out_path}.")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="root dir of the test data")
    parser.add_argument(
        "--num_target_views", type=int, default=56, help="num of target views"
    )
    parser.add_argument(
        "--num_context_views", type=int, default=5, help="num of context views"
    )
    parser.add_argument(
        "--view_selection_ratio",
        type=float,
        default=1.0,
        help="test ratio; set to 0.5 for N=150",
    )

    args = parser.parse_args()

    generate_index_file(args)
