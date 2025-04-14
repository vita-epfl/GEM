import argparse
import os
import sys
from glob import glob
import numpy as np
from PIL import Image, ImageOps
from multiprocessing import Pool, Value
import h5py
from pathlib import Path
from time import time
import math

# 288 cores

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

progress = None
start_time = None
num_images = None

NUM_CPUS = 72 * 4
FRAMES_PER_CHUNK = 25
IMAGES_PER_WRITE = 200
target_width = 1024
target_height = 576


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


def proc_video(video_dir, dst_path: Path):
    global progress, start_time, num_images

    if not os.path.isdir(video_dir):
        print("Not a directory:", video_dir)
        return

    video_name = os.path.basename(video_dir)
    image_list = sorted(
        glob(os.path.join(video_dir, f"*.jpg"))
        + glob(os.path.join(video_dir, f"*.png"))
    )
    num_frames = len(image_list)

    if os.path.exists(str(dst_path / video_name) + ".h5"):
        try:
            f = h5py.File(str(dst_path / video_name) + ".h5", "r")
            if f["num_written"][0] >= num_frames:
                print("Video", video_name, "already processed. Skipping.")
                if f["num_written"][0] < num_frames:
                    print(
                        f"WARNING: Video {video_name} is missing frames! Actual frames: {num_frames}, written frames: {f['num_written'][0]}"
                    )
            else:
                f.close()
                raise Exception(
                    f"Video has {num_frames} frames, but only {f['num_written'][0]} were written."
                )
            f.close()
            with num_images.get_lock():
                num_images.value -= num_frames
            return
        except Exception as e:
            print("H5 of video", video_name, "seems to be corrupt. Will overwrite.")
            os.remove(str(dst_path / video_name) + ".h5")

    if num_frames == 0:
        print("No frames found for video", video_dir)
        return

    f = h5py.File(str(dst_path / video_name) + ".h5", "w")

    print("Processing video...", video_dir)

    ds = f.create_dataset(
        "video",
        (num_frames, target_height, target_width, 3),
        chunks=(min(FRAMES_PER_CHUNK, num_frames), target_height, target_width, 3),
        dtype="uint8",
    )

    num_written = f.create_dataset("num_written", (1,), dtype="int32")

    try:
        for i in range(int(math.ceil(num_frames / IMAGES_PER_WRITE))):
            read_imgs = [
                fast_load_img(image_list[j])
                for j in range(
                    i * IMAGES_PER_WRITE, min((i + 1) * IMAGES_PER_WRITE, num_frames)
                )
            ]
            if len(read_imgs) == 0:
                print(
                    f"WARNING: No images loaded for video {video_name} at frames {i*IMAGES_PER_WRITE} to {(i+1)*IMAGES_PER_WRITE}"
                )
                continue
            read_imgs = np.stack(read_imgs)

            ds[i * IMAGES_PER_WRITE : min((i + 1) * IMAGES_PER_WRITE, num_frames)] = (
                read_imgs
            )
            # Count total number of frames written
            num_written[0] = min((i + 1) * IMAGES_PER_WRITE, num_frames)

            with progress.get_lock():
                progress.value += len(read_imgs)

            p = progress.value
            N = num_images.value
            percentage = (p / N) * 100
            elapsed_time = time() - start_time
            time_per_image = elapsed_time / p if p > 0 else 0
            remaining_time = (N - p) * time_per_image if p > 0 else 0
            print(f"Progress: {p}/{N} ({percentage:.2f}%)")
            print(f"Estimated time remaining: {remaining_time/3600:.2f} hours")
    except Exception as e:
        print("!!!! Error processing video", video_dir)
        print(e)
        print("Will close file")

    f.close()
    return num_frames


if __name__ == "__main__":
    progress = Value("i", 0)  # 'i' indicates an integer
    num_images = Value("i", 58303552)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, help="Directory to save .hdf5 files")
    parser.add_argument("--data_dir", type=str, help="Directory with videos frames")
    args = parser.parse_args()
    print("Starting up...")

    start_time = time()

    video_dirs = []
    for group in os.listdir(args.data_dir):
        if os.path.isdir(os.path.join(args.data_dir, group)):
            video_dirs.extend(
                [
                    (Path(args.data_dir) / group) / video
                    for video in os.listdir(os.path.join(args.data_dir, group))
                    if os.path.isdir(os.path.join(args.data_dir, group, video))
                ]
            )

    print("Found", len(video_dirs), "videos")
    # print("Found video_dirs", video_dirs)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Prepare arguments for each video directory to be processed
    pool_args = [(video_dir, Path(args.out_dir)) for video_dir in video_dirs]

    # Use multiprocessing.Pool to parallelize the work
    with Pool(processes=NUM_CPUS) as pool:
        pool.starmap(proc_video, pool_args)

    print(">>> All done! <<<")
