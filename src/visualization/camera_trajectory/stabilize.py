"""
https://github.com/google/dynibar/blob/main/ibrnet/data_loaders/llff_data_utils.py
"""

import numpy as np
import cv2


def render_stabilization_path(poses, k_size=45):
    """Rendering stablizaed camera path."""

    # hwf = poses[0, :, 4:5]
    num_frames = poses.shape[0]
    output_poses = []

    input_poses = []

    for i in range(num_frames):
        input_poses.append(
            np.concatenate(
                [poses[i, :3, 0:1], poses[i, :3, 1:2], poses[i, :3, 3:4]], axis=-1
            )
        )

    input_poses = np.array(input_poses)

    gaussian_kernel = cv2.getGaussianKernel(ksize=k_size, sigma=-1)
    output_r1 = cv2.filter2D(input_poses[:, :, 0], -1, gaussian_kernel)
    output_r2 = cv2.filter2D(input_poses[:, :, 1], -1, gaussian_kernel)

    output_r1 = output_r1 / np.linalg.norm(output_r1, axis=-1, keepdims=True)
    output_r2 = output_r2 / np.linalg.norm(output_r2, axis=-1, keepdims=True)

    output_t = cv2.filter2D(input_poses[:, :, 2], -1, gaussian_kernel)

    for i in range(num_frames):
        output_r3 = np.cross(output_r1[i], output_r2[i])

        render_pose = np.concatenate(
            [
                output_r1[i, :, None],
                output_r2[i, :, None],
                output_r3[:, None],
                output_t[i, :, None],
            ],
            axis=-1,
        )

        output_poses.append(render_pose[:3, :])

    return output_poses
