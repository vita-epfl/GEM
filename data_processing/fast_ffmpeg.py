import argparse
import subprocess
import glob
import os
import shutil
import multiprocessing as mp
from pathlib import Path
import random
from time import time

NUM_CPUS = 288 // 4


def acquire_video_wrapper(args):
    """
    Wraps the call to the acquire video functions
    :param args: arguments to unpack
    :return:
    """

    try:
        return acquire_video(*args)
    except Exception as e:
        print(f"Error acquiring video: {e}")


def acquire_video(video_path, output_path, tmp_path, fps, extension, tgt_w, tgt_h):
    """
    Acquires a video and saves it to the specified output directory

    :param video_path: the video file path
    :param output_path: the directory in which to save the output video. The directory must not exist
    :param tmp_path: directory where to save temporary output
    :param fps: fps at which to acquire the video
    :param extension: extension in which to save files
    :return:
    """
    print(f" - Acquiring '{video_path}'")

    # Cleans the tmp_directory
    if os.path.isdir(tmp_path) and len(os.listdir(tmp_path)) > 0:
        # Use ffprobe to get video duration in minutes
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        num_frames = int(float(result.stdout) * fps)
        if num_frames > 0:
            # Check if the number of frames is the same as the number of files in the directory
            num_files = len(os.listdir(tmp_path))
            if float(num_files) >= float(num_frames) * 0.985:
                print(
                    f"Skipping video, already acquired: {num_files} files for {num_frames} frames"
                )
                return
            else:
                # Clean the directory
                print(
                    f"Cleaning up dir",
                    tmp_path,
                    f"because found {num_files} files BUT video has {num_frames} frames",
                )
                shutil.rmtree(tmp_path)

    os.makedirs(tmp_path, exist_ok=True)

    start_time = time()

    # print(f"Reading video '{video_path}'")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    dims = result.stdout.split("x")
    width = int(dims[0])
    height = int(dims[1])

    if height > width:
        # Swap width and height
        temp = width
        width = height
        height = temp

    if (
        width < 100
        or height < 100
        or width > 1000000
        or height > 1000000
        or height > width
    ):
        print(f"Error reading video dimensions: read {width}x{height}")
        return

    requested_aspect_ratio = tgt_w / tgt_h
    # Fit largest possible rectangle of aspect ratio requested_aspect_ratio in the frame
    w_to_crop = int(height * requested_aspect_ratio)
    assert w_to_crop <= width, f"Error calculating crop width: {w_to_crop} > {width}"
    h_to_crop = height

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            f"-filter_complex",
            f"fps={fps},crop={w_to_crop}:{h_to_crop},scale={tgt_w}:{tgt_h}",
            f"{tmp_path}/%06d.{extension}",
        ]
    )

    frame_paths = list(sorted(glob.glob(os.path.join(tmp_path, f"*.{extension}"))))
    frames_count = len(frame_paths)

    # Checks that frames were generated
    if frames_count <= 0:
        raise Exception(f"Reading video '{video_path}', but no frames were generated")

    end_time = time()
    print(f"Time taken to extract frames: {end_time - start_time}")


def first_stage_preprocess(arguments):
    output_extension = "png"
    root_directory = arguments.video_dir
    output_directory = Path(arguments.output_dir)

    # if output_directory.exists():
    #     # Count files in the output directory (non-recursive)
    #     num_files = len(os.listdir(output_directory))
    #     if num_files > 0:
    #         confirm = input(f"Output directory for first stage is not empty ({output_directory}). What would you like to do? (A - Abort, E - Use the existing frames and skip frame extraction, DELETE - delete): ")
    #         if confirm.lower() == "a":
    #             print("Aborting...")
    #             exit()
    #         elif confirm.lower() == "e":
    #             return
    #         elif confirm == "DELETE":
    #             shutil.rmtree(output_directory)
    #             print("Deleted existing frames")
    #         else:
    #             print("Invalid command. Aborting...")
    #             exit()

    acquisition_fps = arguments.fps
    processes = arguments.processes

    print("Finding videos in root dir", root_directory)

    # Searches the top level directories
    directories_to_process = [
        current_file
        for current_file in glob.glob(os.path.join(root_directory, "*"))
        if os.path.isdir(current_file)
    ]
    directories_to_process.append(root_directory)
    directories_to_process.sort()

    # Extracts the paths of all videos in the directories to process
    video_paths = []
    video_group_names = []
    for current_directory in directories_to_process:
        directory_contents = (
            glob.glob(os.path.join(current_directory, f"*.mp4"))
            + glob.glob(os.path.join(current_directory, "*.webm"))
            + glob.glob(os.path.join(current_directory, "*.mov"))
        )
        video_paths.extend(directory_contents)
        video_group_names.extend(
            [os.path.basename(current_directory)] * len(directory_contents)
        )
    video_paths.sort()

    video_names = [
        f"{video_group_names[index]}_{os.path.basename(video_path).split('.')[0]}"
        for index, video_path in enumerate(video_paths)
    ]

    # Creates the output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    output_paths = [
        os.path.join(output_directory, f"{index:09d}")
        for index in range(len(video_paths))
    ]
    frames_paths = [
        os.path.join(output_directory, f"{video_names[index]}")
        for index in range(len(video_paths))
    ]

    tgt_widths = [arguments.target_width] * len(video_paths)
    tgt_heights = [arguments.target_height] * len(video_paths)

    # List worker parameters
    work_items = list(
        zip(
            video_paths,
            output_paths,
            frames_paths,
            [acquisition_fps] * len(video_paths),
            [output_extension] * len(video_paths),
            tgt_widths,
            tgt_heights,
        )
    )

    print("== Video Acquisition ==")

    if NUM_CPUS == 1:
        # Processes all videos sequentially
        for work_item in work_items:
            acquire_video_wrapper(work_item)
    else:
        # Processes all videos
        pool = mp.Pool(processes)
        pool.map(acquire_video_wrapper, work_items)
        pool.close()


if __name__ == "__main__":
    print("== Video Search ==")

    # Loads arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_width", type=int, default=1024)
    parser.add_argument("--target_height", type=int, default=576)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--processes", type=int, default=NUM_CPUS)
    arguments = parser.parse_args()

    target_width = arguments.target_width
    target_height = arguments.target_height
    assert target_width > 0 and target_height > 0, "Invalid target width or height"
    first_stage_preprocess(arguments)
    print("== Done with first stage - Frame extraction (1/2) ==")
