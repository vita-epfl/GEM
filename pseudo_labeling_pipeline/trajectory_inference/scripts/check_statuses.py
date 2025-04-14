import os
import h5py
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from p_tqdm import p_map
import os
import numpy as np

BASE_DIR = Path("/store/swissai/a03/datasets")

H5_DIRS = [
    "CCD/h5_files",
    "D2City",
    "DAD/positive",
    "DAD/negative",
    "DoTA",
    "Drive360/h5_files",
    "OpenDV-YouTube/h5",
    "YouTubeCrash/h5_files/accident",
    "YouTubeCrash/h5_files/nonaccident",
    "bdd100k/h5",
    "kitti_h5/data_h5",
    "nuplan_h5",
    "HondaHDD/extracted",
]

def check_status_of_depth(file):
    try:
        with h5py.File(file, "r") as depth_file:
            num_written = depth_file["num_written"][0]
            tmp = depth_file["depth"][num_written - 1]
        assert not np.array_equal(tmp, np.zeros_like(tmp))
        with open("good_files.txt", "a") as f:
            f.write(f"{file}\n")
        return True
    except Exception as e:
        # print(e)
        # print(f"Error opening {file}. Moving on.")
        os.remove(file)
        print("REMOVED", file)
        with open("bad_files.txt", "a") as f:
            f.write(f"{file}\n")
        return False

def collect_frames():
    # if exists
    if os.path.exists("good_files.txt"):
        os.remove("good_files.txt")
    if os.path.exists("bad_files.txt"):
        os.remove("bad_files.txt")

    for dataset in tqdm(H5_DIRS):
        dirpath = BASE_DIR / (dataset + "_proc")
        paths = list(dirpath.rglob("*.h5"))
        # Filter out all paths whose filenames start with camera_, depth_ or trajectory_
        paths = [p for p in paths if any([p.name.startswith(x) for x in ["depth_"]])]
        statuses = p_map(check_status_of_depth, paths)
        if len(statuses) == 0:
            print(f"Dataset: {dataset} -> found nothing")
            continue
        true_statuses = [s for s in statuses if s]
        # print ratio of completion 
        print(f"Dataset: {dataset}, {float(len(true_statuses))/len(statuses)*100:.2f}%")

def main():
    collect_frames()

if __name__ == "__main__":
    main()
