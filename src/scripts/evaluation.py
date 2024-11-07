import os
import lpips
import numpy as np
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch

# from image_folder import make_dataset
# from cleanfid import fid

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []

    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)

    return img_paths, len(img_paths)


def make_dataset(path_files):
    # if path_files.find(".txt") != -1:
    #     paths, size = make_dataset_txt(path_files)
    # else:
    paths, size = make_dataset_dir(path_files)

    return paths, size


parser = argparse.ArgumentParser(description="Image quality evaluations on the dataset")
parser.add_argument(
    "--gt_path", type=str, default="../results/", help="path to original gt data"
)
parser.add_argument(
    "--g_path", type=str, default="../results.", help="path to the generated data"
)
parser.add_argument(
    "--num_test", type=int, default=0, help="how many examples to load for testing"
)

args = parser.parse_args()
lpips_vgg = lpips.LPIPS(net="squeeze")


def calculate_score(img_gt, img_test):
    """
    function to calculate the image quality score
    :param img_gt: original image
    :param img_test: generated image
    :return: mae, ssim, psnr
    """

    l1loss = np.mean(np.abs(img_gt - img_test))

    psnr_score = psnr(img_gt, img_test, data_range=1)

    ssim_score = ssim(
        img_gt, img_test, multichannel=True, data_range=1, win_size=11, channel_axis=2
    )

    lpips_dis = lpips_vgg(
        torch.from_numpy(img_gt).permute(2, 0, 1),
        torch.from_numpy(img_test).permute(2, 0, 1),
        normalize=True,
    )

    return l1loss, ssim_score, psnr_score, lpips_dis.data.numpy().item()


if __name__ == "__main__":
    gt_paths, gt_size = make_dataset(args.gt_path)
    print("gt_size", gt_size)
    # assert False
    g_paths, g_size = make_dataset(args.g_path)
    print("g_size", g_size)
    # assert False
    # fid_score = fid.compute_fid(
    #     args.gt_path,
    #     args.g_path,
    #     mode="clean",
    #     dataset_res=256,
    #     batch_size=100,
    #     model_name="clip_vit_b_32",
    # )

    l1losses = []
    ssims = []
    psnrs = []
    lpipses = []

    size = args.num_test if args.num_test > 0 else min(gt_size, g_size)

    for i in range(size):
        gt_img = Image.open(gt_paths[i]).convert("RGB")
        gt_numpy = np.array(gt_img).astype(np.float32) / 255.0

        # print(gt_paths[i])
        # print(args.g_path + gt_paths[i].split("/")[-1])

        if os.path.exists(args.g_path + gt_paths[i].split("/")[-1]):
            g_img = Image.open(args.g_path + gt_paths[i].split("/")[-1]).convert("RGB")
            g_numpy = np.array(g_img).astype(np.float32) / 255.0
            l1loss, ssim_score, psnr_score, lpips_score = calculate_score(
                gt_numpy, g_numpy
            )
            print(os.path.basename(gt_paths[i]))
            print(l1loss, ssim_score, psnr_score, lpips_score)
            l1losses.append(l1loss)
            ssims.append(ssim_score)
            psnrs.append(psnr_score)
            lpipses.append(lpips_score)

    # assert False
    print(
        "{:>10},{:>10},{:>10},{:>10},{:>10}".format(
            "l1loss", "SSIM", "PSNR", "LPIPS", "FID-Score"
        )
    )
    print(
        "{:10.4f},{:10.4f},{:10.4f},{:10.4f}".format(
            np.mean(l1losses),
            np.mean(ssims),
            np.mean(psnrs),
            np.mean(lpipses),
            # np.mean(fid_score),
        )
    )
    print(
        "{:10.4f},{:10.4f},{:10.4f},{:10.4f}".format(
            np.var(l1losses), np.var(ssims), np.var(psnrs), np.var(lpipses)
        )
    )
