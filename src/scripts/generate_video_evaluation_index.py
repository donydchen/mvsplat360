import json
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--index_input", type=str, help="source evaluation file path")
parser.add_argument(
    "--index_output", type=str, help="output video render index file name"
)
parser.add_argument(
    "--use_target_bound",
    action="store_true",
    help="whether to use target bound; if false, use context bound",
)
args = parser.parse_args()


# INDEX_INPUT = Path("assets/evaluation_index_re10k.json")
# INDEX_OUTPUT = Path("assets/evaluation_index_re10k_video.json")
INDEX_INPUT = Path(args.index_input)
INDEX_OUTPUT = INDEX_INPUT.parent / args.index_output


if __name__ == "__main__":
    with INDEX_INPUT.open("r") as f:
        index_input = json.load(f)

    index_output = {}
    for scene, scene_index_input in index_input.items():
        if isinstance(scene_index_input, list):
            if len(scene_index_input) > 0:
                scene_index_input = scene_index_input[0]
            else:
                scene_index_input = None
        
        # Handle scenes for which there's no index.
        if scene_index_input is None:
            index_output[scene] = None
            continue

        # Add all intermediate frames as target frames.
        a, b = scene_index_input["context"]
        if args.use_target_bound:
            targets = scene_index_input["target"]
            target_index_list = list(range(min(targets), max(targets) + 1))
        else:
            target_index_list = list(range(a, b + 1))
        index_output[scene] = {
            "context": [a, b],
            "target": target_index_list,
        }

    with INDEX_OUTPUT.open("w") as f:
        json.dump(index_output, f)

    print("All Done!")
