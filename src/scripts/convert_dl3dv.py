import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict
from PIL import Image

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import argparse
import json
import os

from glob import glob
from scipy.spatial import distance_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="depth directory")
parser.add_argument("--output_dir", type=str, help="dataset directory")
parser.add_argument(
    "--img_subdir",
    type=str,
    default="images_8",
    help="image directory name",
    choices=["images_4", "images_8", "gaussian_splat/images_8", "gaussian_splat/images_4"],
)
parser.add_argument("--n_test", type=int, default=10, help="test skip")
parser.add_argument("--which_stage", type=str, default=None, help="dataset directory")
parser.add_argument("--detect_overlap", action="store_true")

args = parser.parse_args()


INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)


# Target 200 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(2e8)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    image_keys = set(
        example.name
        for example in tqdm(list((INPUT_DIR / stage).iterdir()), desc="Indexing scenes")
        if example.is_dir() and not example.name.startswith(".")
    )
    # keys = image_keys & metadata_keys
    keys = image_keys
    # print(keys)
    # assert False
    print(f"Found {len(keys)} keys.")
    return sorted(list(keys))


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {
        int(path.stem.split("_")[-1]): load_raw(path)
        for path in example_path.iterdir()
        if path.suffix.lower() not in [".npz"]
    }


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(example_path: Path) -> Metadata:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    url = str(example_path).split("/")[-3]
    with open(example_path, "r") as f:
        meta_data = json.load(f)

    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    saved_fx = float(fx) / float(store_w)
    saved_fy = float(fy) / float(store_h)
    saved_cx = float(cx) / float(store_w)
    saved_cy = float(cy) / float(store_h)

    timestamps = []
    cameras = []
    opencv_c2ws = []  # will be used to calculate camera distance

    for frame in meta_data["frames"]:
        timestamps.append(
            int(os.path.basename(frame["file_path"]).split(".")[0].split("_")[-1])
        )
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        # transform_matrix is in blender c2w, while we need to store opencv w2c matrix here
        opencv_c2w = np.array(frame["transform_matrix"]) @ blender2opencv
        opencv_c2ws.append(opencv_c2w)
        camera.extend(np.linalg.inv(opencv_c2w)[:3].flatten().tolist())
        cameras.append(np.array(camera))

    # timestamp should be the one that match the above images keys, use for indexing
    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    # calculate camera distance
    # opencv_c2ws = np.array(opencv_c2ws)
    # xyzs = opencv_c2ws[:, :3, -1]
    # cameras_dist_matrix = distance_matrix(xyzs, xyzs, p=2)
    # cameras_dist_index = np.argsort(cameras_dist_matrix, axis=1)
    # cameras_dist_matrix = torch.tensor(cameras_dist_matrix, dtype=torch.float32)
    # cameras_dist_index = torch.tensor(cameras_dist_index, dtype=torch.int64)

    return {"url": url, "timestamps": timestamps, "cameras": cameras}


def partition_train_test_splits(root_dir, n_test=10):
    sub_folders = sorted(glob(os.path.join(root_dir, "*/")))
    test_list = sub_folders[::n_test]
    train_list = [x for x in sub_folders if x not in test_list]
    out_dict = {"train": train_list, "test": test_list}
    return out_dict


def is_image_shape_matched(image_dir, target_shape):
    image_path = sorted(glob(str(image_dir / "*")))[0]
    # print(image_path)
    try:
        im = Image.open(image_path)
    except:
        return False
    w, h = im.size
    if (h, w) == target_shape:
        return True
    else:
        return False


def legal_check_for_all_scenes(root_dir, target_shape, image_dir):
    sub_folders = sorted(glob(os.path.join(root_dir, "*/")))
    for sub_folder in tqdm(sub_folders, desc="checking scenes..."):
        img_dir = sorted(glob(os.path.join(sub_folder, image_dir)))[0]
        assert is_image_shape_matched(Path(img_dir), target_shape), f"image shape does not match for {sub_folder}"
        if "gaussian" in image_dir:
            pose_file = os.path.join(sub_folder, "gaussian_splat", "transforms.json")
        else:
            pose_file = os.path.join(sub_folder, "transforms.json")
        assert os.path.isfile(pose_file), f"cannot find pose file for {sub_folder}"


if __name__ == "__main__":
    # subfold_dict = {"train": "gaussian_splat", "test": "colmap"}
    if "images_8" in args.img_subdir:
        target_shape = (270, 480)  # (h, w)
    elif "images_4" in args.img_subdir:
        target_shape = (540, 960)

    print("checking all scenes...")
    legal_check_for_all_scenes(INPUT_DIR, target_shape, args.img_subdir)
    print("all scenes pass the check.")

    out_dict = partition_train_test_splits(INPUT_DIR, n_test=args.n_test)

    # ignore scene
    if args.detect_overlap:
        bm_file = os.path.join(
            INPUT_DIR, os.pardir, "benchmark_scenes.json"
        )
        if not os.path.isfile(bm_file):
            benchmark_dir = os.path.join(
                INPUT_DIR, os.pardir, "Benchmark"
            )
            overlap_scenes = [x for x in os.listdir(benchmark_dir) if os.path.isdir(os.path.join(benchmark_dir, x))]
            with open(bm_file, 'w') as f:
                json.dump(overlap_scenes, f)
        else:
            with open(bm_file, 'r') as f:
                overlap_scenes = json.load(f)
        assert len(overlap_scenes) == 140, "test scenes should contain 140 scenes"

        # overlap_scenes_path = os.path.join(
        #     INPUT_DIR, os.pardir, "process_logs", "benchmark_scenes_in_3K_4K.json"
        # )
        # with open(overlap_scenes_path, 'r') as f:
        #     overlap_scenes = json.load(f)
    else:
        overlap_scenes = []

    # print(overlap_scenes)
    # exit()

    for stage in ("train", "test"):
        if args.which_stage is not None and stage != args.which_stage:
            print(f"Only process stage {args.which_stage}, skip [{stage}]")
            continue

        # print(f"{os.path.basename(INPUT_DIR)}_{stage}.json")
        # break

        error_logs = []
        # keys = get_example_keys(stage)
        image_dirs = out_dict[stage]

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            # print(
            #     f"Saving chunk {chunk_key} of {len(image_dirs)} ({chunk_size / 1e6:.2f} MB)."
            # )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for image_dir in tqdm(image_dirs, desc=f"Processing {stage}"):
            key = os.path.basename(image_dir.strip("/"))
            if key in overlap_scenes:
                print(f"scene {key} in benchmark, skip.")
                continue

            image_dir = Path(image_dir) / args.img_subdir

            # skip scenes that does not match the target shape
            if not is_image_shape_matched(image_dir, target_shape):
                error_msg = f"---------> [ERROR] image shape not match in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue

            # if key not in [
            #     "d9b6376623741313bf6da6bf4cdb9828be614a2ce9390ceb3f31cd535d661a75"
            # ]:
            #     continue

            # image_dir = INPUT_DIR / stage / key / subfold_dict[stage] / args.img_subdir
            # metadata_dir = INPUT_METADATA_DIR / stage / key / "colmap" / "transforms.json"
            num_bytes = get_size(image_dir)

            # Read images and metadata.
            images = load_images(image_dir)
            meta_path = image_dir.parent / "transforms.json"
            if not meta_path.is_file():
                error_msg = f"---------> [ERROR] no meta file in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue
            example = load_metadata(meta_path)

            # Merge the images into the example.
            try:
                example["images"] = [
                    images[timestamp.item()] for timestamp in example["timestamps"]
                ]
            except:
                error_msg = f"---------> [ERROR] Some images missing in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue
            # assert len(images) == len(example["timestamps"])
            # if len(images) != len(example["timestamps"]):
            # print(f"Something wrong happen in {key}")

            # Add the key to the example.
            example["key"] = key

            # print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()

        # generate index
        if len(image_dirs) > 0:
            print("Generate key:torch index...")
            index = {}
            stage_path = OUTPUT_DIR / stage
            for chunk_path in tqdm(
                list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"
            ):
                if chunk_path.suffix == ".torch":
                    chunk = torch.load(chunk_path)
                    for example in chunk:
                        index[example["key"]] = str(chunk_path.relative_to(stage_path))
            with (stage_path / "index.json").open("w") as f:
                json.dump(index, f)

        # dump error logs
        if len(error_logs) > 0:
            print(error_logs)
            error_path = os.path.join(
                INPUT_DIR, os.pardir, "process_logs", f"{os.path.basename(INPUT_DIR)}_{stage}.json"
            )
            with open(error_path, 'w') as f:
                json.dump(error_logs, f)
