import os
import argparse
from pathlib import Path
import h5py
from collections import deque


parser = argparse.ArgumentParser(prog="write_h5")
# paths
parser.add_argument("--base-dir", type=str)
parser.add_argument("--replace-from", type=str)
parser.add_argument("--replace-to", type=str)
# tuning parameters
parser.add_argument("--chunk-size", type=int, default=48)
parser.add_argument("--image-height", type=int, default=576)
parser.add_argument("--image-width", type=int, default=1024)
parser.add_argument("--h5-chunk-size", type=int, default=24)


def process_files(args, file_queue):

    while True:
        file_path = file_queue.popleft()
        if file_path == "DONE":
            break

        os.makedirs(
            os.path.dirname(file_path).replace(args.replace_from, args.replace_to),
            exist_ok=True,
        )

        with h5py.File(file_path, "r") as read_f:

            with h5py.File(
                os.path.join(
                    os.path.dirname(file_path).replace(
                        args.replace_from, args.replace_to
                    ),
                    os.path.basename(file_path),
                ),
                "w",
            ) as write_f:

                num_written = read_f["num_written"][0]

                video_ds = write_f.create_dataset(
                    "video",
                    (num_written, args.image_height, args.image_width, 3),
                    chunks=(
                        min(args.h5_chunk_size, num_written),
                        args.image_height,
                        args.image_width,
                        3,
                    ),
                    dtype="uint8",
                )
                write_f.create_dataset("num_written", data=[num_written], dtype="int32")

                read_idx = 0
                while read_idx < num_written:
                    start_idx = read_idx
                    stop_idx = min(read_idx + args.chunk_size, num_written)
                    data = read_f["video"][start_idx:stop_idx]
                    video_ds[start_idx : stop_idx] = data

                    read_idx = stop_idx

                assert read_idx == num_written

        print(f"Finished processing {file_path}")


if __name__ == "__main__":

    args = parser.parse_args()

    file_paths = sorted(Path(args.base_dir).rglob("*.h5"))
    file_queue = deque(file_paths)
    file_queue.append("DONE")

    process_files(args, file_queue)
