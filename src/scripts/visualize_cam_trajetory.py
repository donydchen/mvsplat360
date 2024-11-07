from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from einops import einsum
from jaxtyping import Float
import json
import os
import torch
import argparse
from tqdm import tqdm
from glob import glob
from PIL import Image
import skvideo.io
import sys

# from scipy.spatial import procrustes
from sklearn.decomposition import PCA

# from flowmap.export.colmap import read_colmap_model

SOURCES = (
    # (name, path, color)
    # ("COLMAP", "/home/ghost/Datasets/nerf_llff_data", "#000000"),
    # ("FlowMap", "/nobackup/nvme1/charatan/flowmap_paper_outputs/v6", "#E6194B"),
    ("COLMAP", "datasets/DL3DV10K/1K", "#E6194B"),
)

SCENES = (
    # (name, whether to manually adjust orientation)
    # ("co3d_bench", True),
    # ("co3d_hydrant", False),
    # ("flower", False),
    # ("horns", False),
    # ("mipnerf360_bonsai", True),
    # ("mipnerf360_garden", False),
    # ("tandt_caterpillar", False),
    # ("tandt_horse", False),
    ("001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f", False),
)

MARGIN = 0
SQUASH = 0.6


def read_colmap_model(example_path):
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

    all_extrinsics = []
    all_intrinsics = []
    all_image_names = []

    for frame in meta_data["frames"]:
        timestamps.append(
            int(os.path.basename(frame["file_path"]).split(".")[0].split("_")[-1])
        )
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        # transform_matrix is in blender c2w, while we need to store opencv w2c matrix here
        opencv_c2w = np.array(frame["transform_matrix"]) @ blender2opencv

        all_extrinsics.append(torch.from_numpy(opencv_c2w).to(torch.float32))

    return torch.stack(all_extrinsics), None


def get_rotation(
    points: Float[np.ndarray, "point 3"],
    flip: bool,  # For when PCA doesn't give the desired orientation.
) -> Float[np.ndarray, "3 3"]:
    pca = PCA(n_components=3).fit(points)

    x, y, _ = pca.components_.T
    z = np.cross(x, y)
    y = np.cross(z, x)

    rotation = np.linalg.inv(np.stack([x, y, z]))
    return rotation[[0, 2, 1]] if flip else rotation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="depth directory")
    parser.add_argument(
        "--output_dir", type=str, default="figures", help="dataset directory"
    )
    parser.add_argument("--combine_video", action="store_true")
    args = parser.parse_args()

    if args.combine_video:
        # Path(f"{args.output_dir}/trajectories/videos").mkdir(exist_ok=True, parents=True)
        for scene in tqdm(sorted(glob(os.path.join(args.input_dir, "*/")))):
            path = f"{args.input_dir}/{os.path.basename(scene.strip('/'))}.mp4"
            if os.path.isfile(path):
                continue

            # print(path)
            writer = skvideo.io.FFmpegWriter(
                path,
                outputdict={
                    "-pix_fmt": "yuv420p",
                    "-crf": "21",
                    "-vf": f"setpts=1.*PTS",
                },
            )

            frames = sorted(glob(os.path.join(scene, "images_8", "*.png")))
            for frame in tqdm(frames, desc="adding frames", leave=False):
                writer.writeFrame(Image.open(frame))
            writer.close()
        sys.exit()

    # https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
    from mpl_toolkits.mplot3d.axis3d import Axis

    if not hasattr(Axis, "_get_coord_info_old"):

        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs

        Axis._get_coord_info_old = Axis._get_coord_info
        Axis._get_coord_info = _get_coord_info_new

    # load scenes
    SCENES = [
        (x, False)
        for x in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, x))
    ]

    for scene, flip in tqdm(SCENES, desc="visualizing scene trajetory..."):
        # Load the trajectories.
        # print(Path(SOURCES[0][1]) / scene / "sparse/0")
        trajectories = [
            read_colmap_model(Path(args.input_dir) / scene / "transforms.json")[0][
                :, :3, 3
            ]
            .detach()
            .cpu()
            .numpy()
            for _, path, _ in SOURCES
        ]

        # Align the trajectories to the first one.
        # trajectories[1:] = [procrustes(trajectories[0], t)[1] for t in trajectories[1:]]

        # Scale the first trajectory so it's consistent with the others.
        # trajectories[0] = procrustes(trajectories[0], trajectories[1])[0]

        # Figure out the transformation based on the first trajectory.
        rotation = get_rotation(trajectories[0], flip)
        trajectories = [einsum(rotation, t, "i j, p j -> p i") for t in trajectories]

        # Create the plot.
        fig = plt.figure(figsize=(1.18, 1.18), dpi=100)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_proj_type("ortho")
        ax.view_init(elev=30, azim=45)

        # Plot the trajectories.
        for i, trajectory in enumerate(trajectories):
            _, _, color = SOURCES[i]
            ax.plot3D(
                *trajectory.T,
                color=color,
                linewidth=0.5,
                linestyle="--" if i == 0 else "-",
            )

        # Set the axis limits.
        points = np.concatenate(trajectories)
        minima = points.min(axis=0)
        maxima = points.max(axis=0)
        span = (maxima - minima).max() * (1 + MARGIN) * np.array([1, 1, SQUASH])
        means = 0.5 * (maxima + minima)
        starts = means - 0.5 * span
        ends = means + 0.5 * span
        ax.set_xlim(starts[0], ends[0])
        ax.set_ylim(starts[1], ends[1])
        ax.set_zlim(starts[2], ends[2])
        ax.set_aspect("equal")

        # Style the plot.
        ax.set_xticks(np.linspace(starts[0], ends[0], 6))
        ax.set_yticks(np.linspace(starts[1], ends[1], 6))
        ax.set_zticks(np.linspace(starts[2], ends[2], 4))

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo["axisline"]["linewidth"] = 0.75
            axis._axinfo["axisline"]["color"] = (0, 0, 0)
            axis._axinfo["grid"]["linewidth"] = 0.25
            axis._axinfo["grid"]["linestyle"] = "-"
            axis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8)
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.0
            axis.set_pane_color((1, 1, 1))

        Path(f"{args.output_dir}/trajectories").mkdir(exist_ok=True, parents=True)
        fig.savefig(f"{args.output_dir}/trajectories/{scene}.svg")
        plt.close(fig)
