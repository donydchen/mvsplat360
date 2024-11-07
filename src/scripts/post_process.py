from skimage import exposure, io
import os
from glob import glob 
from tqdm import tqdm
import argparse


# Function to perform histogram matching
def match_histograms(source, reference):
    matched = exposure.match_histograms(source, reference, channel_axis=-1)
    return matched


def align_to_gsplat(root_dir, refined_version):
    # root_dir = "outputs/test/batch8x1_2a_dl3dv_5views_diff_feat_fps/epoch_102-step_100000_ctx5_tgt56"
    refined_dir = os.path.join(root_dir, f"ImagesRefined{refined_version}")
    gsplat_dir = os.path.join(root_dir, "ImagesGSplat")
    out_dir = os.path.join(root_dir, f"ImagesPostprocessed{refined_version}V2")
    os.makedirs(out_dir, exist_ok=True)
    image_paths = sorted(glob(os.path.join(refined_dir, "*.png")))  # [:10]
    # scene_name = "none"
    # input_images_dict = {}
    # in_id_lists = []
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path)

        src_image = io.imread(image_path)
        ref_image = io.imread(os.path.join(gsplat_dir, image_name))

        matched_image = match_histograms(src_image, ref_image)
        io.imsave(os.path.join(out_dir, os.path.basename(image_path)), matched_image)


if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="depth directory")
    parser.add_argument("--refined_version", type=int, default=0, help="depth directory")

    args = parser.parse_args()

    align_to_gsplat(args.root_dir, args.refined_version)
