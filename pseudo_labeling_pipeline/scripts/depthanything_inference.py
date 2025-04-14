import os
import argparse
from functools import partial
import h5py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, get_worker_info, Sampler
from torch.profiler import profile, ProfilerActivity
from torchvision.transforms import Compose

from depth_anything.metric_depth.depth_anything_v2.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)
from depth_anything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


parser = argparse.ArgumentParser(prog="DepthAnything_inference")
# paths
parser.add_argument("--file-list", type=str, default="./h5_file_list.txt")
parser.add_argument("--weights-dir", type=str, default="./weights")
# tuning parameters
parser.add_argument("--num-gpus", type=int, default=4)
parser.add_argument("--num-proc-per-gpu", type=int, default=1)
parser.add_argument("--buffer-size", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument("--encoder", type=str, default="vits")
# constants
parser.add_argument("--image-height", type=int, default=576)
parser.add_argument("--image-width", type=int, default=1024)
parser.add_argument("--h5-chunk-size", type=int, default=24)
# profiling
parser.add_argument("--no-profiler", action="store_true")
parser.add_argument("--log-dir", type=str, default="./logs")
parser.add_argument("--exp-name", type=str, default="depthanything_exp")


class SequentialSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.dataset_len = None

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        return iter(range(self.dataset_len))


class H5Dataset(Dataset):

    def __init__(self, file_paths, input_size=518):
        self.file_paths = file_paths
        self.current_path = None
        self.transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __getitem__(self, idx):
        # why fill dataset at getitem rather than init?
        # each worker (which are forked after the init) need to have their own file handle
        if self.file_paths[self.file_idx.value] != self.current_path:
            self.current_path = self.file_paths[self.file_idx.value]

        with h5py.File(self.current_path, "r") as f:
            img = f.get("video")[idx] / 255.0
        return self.transform({"image": img})["image"]


def init_fn(v_file_idx, worker_id):
    info = get_worker_info()
    info.dataset.file_idx = v_file_idx


def process_files(rank, p_rank, args, file_queue, file_paths, model):

    print(f"Started process {p_rank} on GPU {rank}")

    # just in case DepthAnything internals use default
    torch.cuda.set_device(f"cuda:{rank}")

    # define global
    v_file_idx = mp.Value("i", 0)

    # run profiler on rank 0 GPU only
    if (not args.no_profiler) and (rank == 0):
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(args.log_dir, f"{args.exp_name}_rank_{rank}_{p_rank}")
            ),
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        prof.start()

    # push model to assigned GPU
    model = model.to(f"cuda:{rank}").eval()

    # each process assigns num_workers workers for data loading
    dataset = H5Dataset(file_paths)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=partial(init_fn, v_file_idx),
    )

    while True:
        file_idx = file_queue.get()
        if file_idx == "DONE":
            break

        try:  # catch all errors
            file_path = file_paths[file_idx]

            # check if file is already processed
            if os.path.exists(
                os.path.join(os.path.dirname(file_path), f"depth_{os.path.basename(file_path)}")
            ):
                try:
                    with h5py.File(
                        os.path.join(
                            os.path.dirname(file_path),
                            f"depth_{os.path.basename(file_path)}",
                        ),
                        "r",
                    ) as depth_file:
                        num_written = depth_file["num_written"][0]
                        tmp = depth_file["depth"][num_written - 1]
                    assert not np.array_equal(tmp, np.zeros_like(tmp))
                    print(
                        f"Depth H5 file for {file_path} already processed, skipping!"
                    )
                    continue
                except Exception as e:
                    print(
                        f"Depth H5 file for {file_path} seems to be corrupt. Will overwrite."
                    )
                    os.remove(
                        os.path.join(
                            os.path.dirname(file_path),
                            f"depth_{os.path.basename(file_path)}",
                        )
                    )

            with h5py.File(file_path, "r") as f:
                num_written = f["num_written"][0]

            # assign file to dataloader
            v_file_idx.value = file_idx
            data_loader._index_sampler.sampler.dataset_len = num_written

            # open target h5
            with h5py.File(
                os.path.join(os.path.dirname(file_path), f"depth_{os.path.basename(file_path)}"),
                "w",
            ) as depth_file:

                depth_ds = depth_file.create_dataset(
                    "depth",
                    (num_written, args.image_height, args.image_width),
                    chunks=(
                        min(args.h5_chunk_size, num_written),
                        args.image_height,
                        args.image_width,
                    ),
                    dtype="float16",
                )
                depth_file.create_dataset(
                    "num_written", data=[num_written], dtype="int32"
                )

                write_idx = 0
                depth_pred = []
                with torch.no_grad():
                    for data in data_loader:
                        predictions = model.forward(data.to(f"cuda:{rank}"))
                        predictions = nn.functional.interpolate(
                            predictions[:, None],
                            [args.image_height, args.image_width],
                            mode="bilinear",
                            align_corners=True,
                        )[:, 0]
                        # convert to float16 (normalize by maxdepth)
                        predictions /= 80
                        depth_pred.append(predictions)
                        if len(depth_pred) * args.batch_size >= args.buffer_size:
                            # dump buffer to file
                            depth_pred_np = torch.cat(depth_pred, dim=0).cpu().numpy()
                            depth_ds[write_idx : write_idx + len(depth_pred_np)] = (
                                depth_pred_np
                            )
                            write_idx += len(depth_pred_np)
                            depth_pred = []

                        if (not args.no_profiler) and (rank == 0):
                            prof.step()

                # dump the rest
                depth_pred_np = torch.cat(depth_pred, dim=0).cpu().numpy()
                depth_ds[write_idx:] = depth_pred_np

            print(f"Finished processing {file_path} on gpu {rank} process {p_rank}")

        except Exception as e:
            print(e)
            print(f"Error processing depth for {file_path}. Moving on.")
            with open(
                os.path.join(args.log_dir, f"{args.exp_name}_failed_depth.txt"), "a"
            ) as f:
                f.write(f"{file_path} REASON: {e}\n")

    if (not args.no_profiler) and (rank == 0):
        prof.stop()


if __name__ == "__main__":

    mp.set_start_method("spawn")

    args = parser.parse_args()

    assert args.num_workers > 0

    print("Starting DepthAnything with the following parameters:")
    print(f"exp-name: {args.exp_name}")
    print(f"num-gpus: {args.num_gpus}")
    print(f"num-proc-per-gpu: {args.num_proc_per_gpu}")
    print(f"buffer-size: {args.buffer_size}")
    print(f"batch-size: {args.batch_size}")
    print(f"num-workers: {args.num_workers}")
    print(f"encoder: {args.encoder}")

    if args.file_list.endswith(".h5"):
        file_paths = [args.file_list]
    else:
        with open(args.file_list, "r") as f:
            file_paths = f.read().splitlines()
    file_queue = mp.Queue()

    for file_idx in range(len(file_paths)):
        file_queue.put(file_idx)
    for file_idx in range(args.num_gpus):
        file_queue.put("DONE")

    # model config
    dataset = "vkitti"  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80  # 20 for indoor model, 80 for outdoor model
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    model = DepthAnythingV2(**{**model_configs[args.encoder], "max_depth": max_depth})
    model.load_state_dict(
        torch.load(
            os.path.join(
                args.weights_dir,
                f"depth_anything_v2_metric_{dataset}_{args.encoder}.pth",
            ),
            map_location="cpu",
        )
    )

    os.makedirs(args.log_dir, exist_ok=True)

    processes = []
    for rank in range(args.num_gpus):
        for p_rank in range(args.num_proc_per_gpu):
            p = mp.Process(
                target=process_files,
                args=(
                    rank,
                    p_rank,
                    args,
                    file_queue,
                    file_paths,
                    model,
                ),
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print("DONE - depth")
