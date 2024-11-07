from glob import glob
import torch
import json
import math
from tqdm import tqdm
import argparse
import re

from ..evaluation.metrics import (
    compute_lpips,
    compute_psnr,
    compute_ssim,
    compute_dists,
)
from ..misc.image_io import load_image

import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image


class SimpleImageDataset(Dataset):
    def __init__(self, image_dict):
        self.image_dict = image_dict
        self.gt_dir = image_dict.pop("GT")
        self.image_names = self.list_images(self.gt_dir)

    def list_images(self, gt_dir):
        return sorted([x for x in os.listdir(gt_dir) if x.endswith("png")])  # [:10]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        cur_image_name = self.image_names[idx]
        out_dict = {"GT": load_image(os.path.join(self.gt_dir, cur_image_name))}
        for method_name, method_dir in self.image_dict.items():
            out_dict[method_name] = load_image(os.path.join(method_dir, cur_image_name))

        return out_dict


def print_average_scores(scene_dict, method_name):
    try:
        print_method_name = re.match(r"^(.*)_epoch", method_name).group(1)
    except:
        return
    msg = f"{print_method_name:<25}:"
    for k, v in scene_dict.items():
        msg = msg + f" {k.upper()}:{(sum(v)/len(v)):.3f},"
    print(msg)


def main(out_dir, use_postpro=False, vtype="ctx5"):
    device = "cuda"

    if vtype == "ctx5":
        methods_roots = {
            "GT": "outputs/test/dl3dv_480P_ctx5_tgt56_tsplit4/ImagesGT",
            "seqbyseq": "outputs/test/dl3dv_480P_ctx5_tgt56_seqbyseq/ImagesRefined0",
            "tsplit4": "outputs/test/dl3dv_480P_ctx5_tgt56_tsplit4/ImagesRefined0",
        }
    elif vtype == "n150":
        # update methods_roots path here
        pass
    elif vtype == "ctx3":
        # update methods_roots path here
        pass
    elif vtype == "ctx4":
        # update methods_roots path here
        pass
    elif vtype == "ctx6":
        # update methods_roots path here
        pass
    elif vtype == "ctx7":
        # update methods_roots path here
        pass
    else:
        raise Exception(f"Please set paths for {vtype}.")

    if use_postpro:
        updated_methods_roots = {}
        for k, v in methods_roots.items():
            if k == "GT":
                updated_methods_roots[k] = v
            else:
                v_list = v.split("/")
                updated_v = "/".join([*v_list[:-1], "ImagesPostprocessed0V2"])
                updated_k = f"{k}_pp"
                updated_methods_roots[updated_k] = updated_v
        methods_roots = updated_methods_roots
    # print(methods_roots)
    # return
    # check image length
    for k, v in methods_roots.items():
        img_len = len(glob(os.path.join(v, "*.png")))
        print(k, img_len)
        assert img_len == 140 * 56, f"Length err {img_len}; Double check {v}"

    dataset = SimpleImageDataset(image_dict=methods_roots)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)

    methods_metrics = {
        k: {m: [] for m in ["psnr", "ssim", "lpips", "dists"]}
        for k in methods_roots.keys()
    }

    for data_item in tqdm(dataloader, desc="looping data..."):
        gt_images = data_item.pop("GT").to(device)
        for method_name, method_images in data_item.items():
            method_images = method_images.to(device)

            methods_metrics[method_name]["psnr"].extend(
                compute_psnr(gt_images, method_images).detach().cpu().tolist()
            )
            methods_metrics[method_name]["ssim"].extend(
                compute_ssim(gt_images, method_images).detach().cpu().tolist()
            )
            methods_metrics[method_name]["lpips"].extend(
                compute_lpips(gt_images, method_images).detach().cpu().tolist()
            )
            methods_metrics[method_name]["dists"].extend(
                compute_dists(gt_images, method_images).detach().cpu().tolist()
            )

    # print final scores
    for method_name, scene_dict in methods_metrics.items():
        # dump the total scores
        out_name = methods_roots[method_name].strip("/").split("/")[-2]
        out_name = f"{method_name}_{out_name}"

        with open(os.path.join(out_dir, f"{out_name}.json"), "w") as f:
            json.dump(scene_dict, f)

        # print average scores
        print_average_scores(scene_dict, out_name)

    print("All Done! Compute the FID score via:")
    print(f"python -m pytorch_fid --device cuda:0 {methods_roots['GT']} {methods_roots['seqbyseq']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_json", action="store_true")
    parser.add_argument("--use_pp", action="store_true", dest="compute metrics with post-processed images")
    parser.add_argument(
        "--vtype",
        type=str,
        default="ctx5",
        choices=["ctx5", "ctx3", "ctx4", "ctx6", "ctx7", "n150"],
    )

    args = parser.parse_args()

    if args.vtype == "ctx5":
        out_dir = "outputs/test_scores/dl3dv_480P"
    else:
        out_dir = f"outputs/test_scores/dl3dv_480P_{args.vtype}"

    if args.against_encdec:
        out_dir = f"{out_dir}_encdec"
    os.makedirs(out_dir, exist_ok=True)

    if args.print_json:
        json_files = sorted(glob(os.path.join(out_dir, "*.json")))
        for json_file in json_files:
            with open(json_file, "r") as f:
                scene_dict = json.load(f)
                method_name = os.path.basename(json_file).split(".")[0]
                print_average_scores(scene_dict, method_name)
    else:
        main(
            out_dir,
            use_postpro=args.use_pp,
            vtype=args.vtype,
        )
