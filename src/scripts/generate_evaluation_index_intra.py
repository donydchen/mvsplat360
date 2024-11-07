import json
from pathlib import Path
import argparse
import os
from tqdm import tqdm
import random


def partition_list(lst, n_bins):
    if n_bins <= 0:
        raise ValueError("Number of bins must be greater than 0")
    if len(lst) < n_bins:
        raise ValueError("Number of bins cannot exceed the length of the list")

    bin_size = len(lst) // n_bins
    borders = [lst[0]]  # First border is always the first index
    for i in range(1, n_bins):
        border_index = min(i * bin_size, len(lst) - 1)  # Ensure last bin doesn't exceed list length
        borders.append(lst[border_index])
    borders.append(lst[-1])  # Last border is always the last index
    return borders


parser = argparse.ArgumentParser()
parser.add_argument("--index_input", type=str, help="source evaluation file path")
# parser.add_argument(
#     "--index_output", type=str, help="output video render index file name"
# )
parser.add_argument(
    "--target_len", type=int, default=14, help="target index length"
)
parser.add_argument("--match_input", action="store_true")

args = parser.parse_args()


# INDEX_INPUT = Path("assets/evaluation_index_re10k.json")
# INDEX_OUTPUT = Path("assets/evaluation_index_re10k_video.json")
INDEX_INPUT = Path(args.index_input)
idx_outname = (os.path.basename(args.index_input).split(".")[0] + 
               f"_{args.target_len}_target_views")
if args.match_input:
    idx_outname = idx_outname + "_match"
idx_outname = idx_outname + ".json"

INDEX_OUTPUT = INDEX_INPUT.parent / idx_outname


if __name__ == "__main__":
    with INDEX_INPUT.open("r") as f:
        index_input = json.load(f)

    index_output = {}
    for scene, scene_index_input in tqdm(index_input.items()):
        # # Handle scenes for which there's no index.
        if scene_index_input is None:
            index_output[scene] = None
            continue
        # if len(scene_index_input) > 1:
        #     raise Exception("multiple evaluation index group not yet supported")

        # scene_index_input = scene_index_input[0]

        context_left = min(scene_index_input["context"])
        context_right = max(scene_index_input["context"])
        # target_left = min(scene_index_input["target"])
        # target_right = max(scene_index_input["target"])

        bound_left = context_left
        bound_right = context_right

        if bound_right < args.target_len:
            print(f"Not enough view for {scene}")
            continue

        # total_index = list(range(target_left, target_right + 1))
        # # remove context index
        # total_index = [x for x in total_index if x not in scene_index_input["context"]]

        # if len(total_index) < args.target_len:
        #     left_list = [
        #         x
        #         for x in range(min(total_index))
        #         if x not in scene_index_input["context"]
        #     ]
        #     right_list = [
        #         x
        #         for x in range(max(total_index) + 1, bound_right + 1)
        #         if x not in scene_index_input["context"]
        #     ]
        #     candi_list = [*left_list, *right_list]
        #     candi_tgt_len = args.target_len - len(total_index)
        #     if len(candi_list) < candi_tgt_len:
        #         print(f"Fail to find view candidates for {scene}")
        #         print(scene_index_input["context"], scene_index_input["target"])
        #         continue
        #     random.shuffle(candi_list)
        #     total_index.extend(candi_list[:candi_tgt_len])
        #     total_index = sorted(total_index)

        # target_index_list = total_index[::len(total_index) // args.target_len]
        # target_index_list = target_index_list[: args.target_len]

        total_index = list(range(bound_left, bound_right + 1))
        target_index_list = partition_list(total_index, args.target_len - 1)
        assert len(target_index_list) == args.target_len, f"double check {scene}"

        if args.match_input:
            for input_tgt in scene_index_input["target"]:
                if input_tgt in target_index_list:
                    continue
                # replace the selected target
                nn_index = target_index_list.index(min(target_index_list,
                                                       key=lambda x: abs(x - input_tgt)))
                target_ranges = list(range(nn_index, len(target_index_list)))
                target_ranges.extend(list(range(nn_index))[::-1])
                for cur_idx in target_ranges:
                    nn_value = target_index_list[cur_idx]
                    if nn_value in scene_index_input["target"]:  # do not replace init target
                        continue
                    # apply replace
                    target_index_list[cur_idx] = input_tgt
                    break

        target_index_list = sorted(target_index_list)
        index_output[scene] = {
            "context": scene_index_input["context"],
            "target": target_index_list,
        }

        if args.match_input:
            for input_idx in scene_index_input["target"]:
                if input_idx not in target_index_list:
                    print("Hit corner case", scene)
                    print(index_output[scene])
                    print(scene_index_input["target"])
                    assert False
            index_output[scene].update({"target_quantify": scene_index_input["target"]})

    with INDEX_OUTPUT.open("w") as f:
        json.dump(index_output, f)

    print(f"All Done! Save to {str(INDEX_OUTPUT)}")
