from PIL import Image, ImageOps
import numpy as np
import sys
from p_tqdm import p_map
import json
import h5py
from pathlib import Path
from tqdm import tqdm
import os

base = Path("/store/swissai/a03/datasets/nuscenes")
dst_path = Path("/capstor/scratch/cscs/pmartell/datasets/nuscenes_h5")
dst_path_proc = Path("/capstor/scratch/cscs/pmartell/datasets/nuscenes_h5_proc")

dst_path.mkdir(exist_ok=True, parents=True)
dst_path_proc.mkdir(exist_ok=True, parents=True)

target_width = 1024
target_height = 576
FRAMES_PER_CHUNK = 25


def fast_load_img(image_name):
    try:
        img = Image.open(image_name)
        # Efficiently resize and crop the image
        img = ImageOps.fit(
            img,
            (target_width, target_height),
            method=Image.Resampling.LANCZOS,  # High-quality downsampling filter
            centering=(0.5, 0.5),  # Center cropping
        )
        # Convert to numpy array without unnecessary copy
        arr = np.array(img, copy=False).reshape(target_height, target_width, 3)
        return arr
    except Exception as e:
        # print to stderr
        print("Corrupt image", image_name, file=sys.stderr)
        return np.zeros((target_height, target_width, 3), dtype="uint8")


def prep_sample(idx, sample_dict):
    video_name = f"nu{idx:06d}"
    try:
        if os.path.exists(str(dst_path / video_name) + ".h5"):
            all_valid = True
            with h5py.File(str(dst_path / video_name) + ".h5", "r") as f:
                if f["num_written"][0] != len(sample_dict["frames"]):
                    all_valid = False

            with h5py.File(
                str(dst_path_proc / ("trajectory_" + video_name)) + ".h5", "r"
            ) as f:
                if f["num_written"][0] != 8:
                    all_valid = False

            if all_valid:
                return
    except Exception as e:
        print(e)

    try:
        image_seq = list()
        for frame_path in sample_dict["frames"]:
            image = fast_load_img(base / frame_path)
            image_seq.append(image)

        with h5py.File(str(dst_path / video_name) + ".h5", "w") as f:
            ds = f.create_dataset(
                "video",
                (len(image_seq), target_height, target_width, 3),
                chunks=(
                    min(FRAMES_PER_CHUNK, len(image_seq)),
                    target_height,
                    target_width,
                    3,
                ),
                dtype="uint8",
            )
            for i, image in enumerate(image_seq):
                ds[i] = image
            f.create_dataset("num_written", data=[len(image_seq)])

        with h5py.File(
            str(dst_path_proc / ("trajectory_" + video_name)) + ".h5", "w"
        ) as f:
            ds = f.create_dataset(
                "trajectory",
                data=sample_dict["traj"][2:],
                dtype="float32",
            )
            f.create_dataset("num_written", data=[8])
            f.create_dataset("is_nuscenes", data=[True])
    except Exception as e:
        print(f"Error processing {video_name}: {e}")


anno_file = "nuScenes.json"
with open("nuScenes.json", "r") as anno_json:
    samples = json.load(anno_json)

p_map(prep_sample, range(len(samples)), samples, num_cpus=72)
