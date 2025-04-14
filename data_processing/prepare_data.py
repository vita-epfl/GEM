import os
import queue
import threading

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from p_tqdm import p_map
import h5py
import numpy as np

print("Validating data...")
print("This script will merge all CSV files in the data folder and validate the paths of the pseudo labels.")

# Define folders to search and output file path
base_dir = Path("./data/")
output_file = "annotations.csv"

# Recursively find all folders in the base directory
# Make sure only the deepest folders are included
folders = [str(folder) for folder in base_dir.glob("**/*") if folder.is_dir() and not any(folder.is_dir() for folder in folder.iterdir())]

print(">>>> Folders found in the base directory:")
for folder in folders:
    print("\t>", folder)

def process_file(file_path, header_written, lock, outfile):
    """Processes a single file, replaces paths, and appends content to the output file."""
    try:
        i = 0
        for chunk in pd.read_csv(file_path, chunksize=10000):
            # Locking to ensure one thread writes at a time
            with lock:
                # override first row with:
                # rows[i][0] = Path(file_path).parent / Path(rows[i][0]).name
                chunk.iloc[:, 0] = chunk.iloc[:, 0].apply(
                    lambda x: Path(file_path).parent / Path(x).name
                )

                if not header_written[0]:
                    chunk.to_csv(outfile, index=False, header=True, mode="a")
                    header_written[0] = True
                else:
                    chunk.to_csv(outfile, index=False, header=False, mode="a")
            
            i += 1
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")


def merge_csv_files(folders, output_file):
    # Open the output file in append mode and write headers only once
    with open(output_file, "w") as outfile:
        header_written = [False]  # List to allow mutability for threads
        lock = threading.Lock()  # Lock to prevent concurrent writes

        # Queue to manage files for processing
        file_queue = queue.Queue()

        # Count total CSV files and add them to queue for progress bar
        total_files = 0
        for folder in folders:
            if os.path.exists(folder):
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith(".csv"):
                            file_queue.put(os.path.join(root, file))
                            total_files += 1

        # Initialize progress bar
        with tqdm(total=total_files, desc="Merging CSV files") as pbar:
            # Threaded worker function
            def worker():
                while not file_queue.empty():
                    file_path = file_queue.get()
                    process_file(file_path, header_written, lock, outfile)
                    pbar.update(1)
                    file_queue.task_done()

            # Create and start threads
            threads = [threading.Thread(target=worker) for _ in range(288)]
            for thread in threads:
                thread.start()

            # Wait for all threads to finish
            file_queue.join()
            for thread in threads:
                thread.join()
    
    print(f"All CSV files have been merged into {output_file}")


def validate_labels(total_anno_file):
    def get_labels(path):
        # validates and gets the paths of the labels
        path = Path(path)
        skeleton_filepath = path.parent / f"pose_{path.name}"
        depth_filepath = None
        traj_filepath = None

        try:
            depth_filepath = path.parent / f"depth_{path.name}"
            with h5py.File(depth_filepath, "r") as depth_file:
                num_written = depth_file["num_written"][0]
                assert "depth" in depth_file, f"depth not found in {depth_filepath}"
                tmp = depth_file["depth"][num_written - 1]
            assert not np.array_equal(tmp, np.zeros_like(tmp))
        except Exception as e:
            depth_filepath = None
            print("Depth error", e)

        try:
            traj_filepath = path.parent / f"trajectory_{path.name}"
            if not Path(traj_filepath).exists():
                # print("Doesn't exist", traj_filepath)
                traj_filepath = None
            with h5py.File(traj_filepath, "r") as trajectory_file:
                if "is_nuscenes" not in trajectory_file:
                    assert (
                        "num_written" in trajectory_file
                    ), f"num_written not found in {traj_filepath}"
                    assert (
                        "merged_trajectory" in trajectory_file
                    ), f"merged_trajectories not found in {traj_filepath}"
                    # assert trajectory_file["trajectory"].shape[0] * trajectory_file["trajectory"].shape[1] >= num_written-1, f"Only {trajectory_file['trajectory'].shape[0] * trajectory_file['trajectory'].shape[1]} frames written"
                    # num_written = trajectory_file["num_written"][0]
        except Exception as e:
            traj_filepath = None
            print("Trajectory error", e)

        return skeleton_filepath, depth_filepath, traj_filepath


    def load_csv_to_list(file_path, chunksize=200000):
        rows = []
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            rows.extend(chunk.values.tolist())
        return rows


    rows_list = load_csv_to_list(total_anno_file)

    path_list = set([row[0] for row in rows_list])
    skeletons_depths_and_trajs = p_map(get_labels, path_list)
    skeletons = [x[0] for x in skeletons_depths_and_trajs]
    depths = [x[1] for x in skeletons_depths_and_trajs]
    trajs = [x[2] for x in skeletons_depths_and_trajs]

    skeleton_cache = dict(zip(path_list, skeletons))
    depth_cache = dict(zip(path_list, depths))
    trajs_cache = dict(zip(path_list, trajs))

    # Now update the csv file
    # Now go through each row and add a new column for the depth and trajectory paths
    for row in rows_list:
        path = row[0]
        skeleton = skeleton_cache.get(path, None)
        depth = depth_cache.get(path, None)
        traj = trajs_cache.get(path, None)
        row.append(skeleton)
        row.append(depth)
        row.append(traj)

    num_nones_in_depth = len([depth for depth in depth_cache.values() if depth is None])
    num_nones_in_traj = len([traj for traj in trajs_cache.values() if traj is None])
    print(f"Depth invalid rate: {100*num_nones_in_depth/len(depth_cache):.2f}%")
    print(f"Traj invalid rate: {100*num_nones_in_traj/len(trajs_cache):.2f}%")

    annotations_old = pd.read_csv(total_anno_file)

    annotations = pd.DataFrame(rows_list, columns=annotations_old.columns.tolist() + ["pose_path", "depth_path", "trajectory_path"])
    annotations.to_csv(total_anno_file, index=False)


##### STEP 1 - Merge all CSV files into one
merge_csv_files(folders, output_file)

##### STEP 2 - Validate and gather the file paths of the pseudo labels
validate_labels(output_file)
