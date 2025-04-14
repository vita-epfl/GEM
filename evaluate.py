import os
from eval_utils import EvaluationHelper
import argparse
from pytorch_lightning import seed_everything
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--condition_type", type=str, default="unconditional",
                        choices=['unconditional', 'object_manipulation', 'skeleton_manipulation', 'object_insertion', 'ego_motion'],
                        help="type of condition",)
    args = parser.parse_args()

    real_videos_path = sorted([os.path.join(args.dir, "real", "npy", path) for path in os.listdir(os.path.join(args.dir, "real", "npy")) if path.endswith(".npy")])
    generated_videos_path = sorted([os.path.join(args.dir, "virtual", "npy", path) for path in os.listdir(os.path.join(args.dir, "virtual", "npy")) if path.endswith(".npy")])

    dataset_length = len(real_videos_path)
    if args.condition_type != "unconditional":
        dataset_length = min(dataset_length, 1500)
    seed_everything(22)
    helper = EvaluationHelper(args.condition_type, None, after_generation=True)

    metrics = helper.get_evaluation_metrics(real_videos_path[:dataset_length], generated_videos_path[:dataset_length])
    # metrics = helper.get_evaluation_metrics(real_videos_path[start_idx:end_idx], real_videos_path[start_idx:end_idx])
    print(metrics)


if __name__ == "__main__":
    main()