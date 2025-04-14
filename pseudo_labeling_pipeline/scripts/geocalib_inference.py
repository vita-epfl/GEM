import os
import argparse
from functools import partial
from datetime import datetime
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info
from torch.profiler import profile, ProfilerActivity

from geocalib import GeoCalib


parser = argparse.ArgumentParser(prog="GeoCalib_inference")
# paths
parser.add_argument("--file-list", type=str, default="./h5_file_list.txt")
# tuning parameters
parser.add_argument("--num-gpus", type=int, default=4)
parser.add_argument("--num-proc-per-gpu", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=50)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--camera-model", type=str, default="pinhole")
# profiling
parser.add_argument("--no-profiler", action="store_true")
parser.add_argument("--log-dir", type=str, default="./logs")
parser.add_argument("--exp-name", type=str, default="geocalib_exp")


class RegularSampler(Sampler):

    def __init__(self, data_source, num_grid_points):
        self.data_source = data_source
        self.num_grid_points = num_grid_points  # fixed
        self.dataset_len = None

    def __len__(self):
        return self.num_grid_points

    def __iter__(self):
        grid_points = torch.round(
            torch.linspace(0, self.dataset_len - 1, steps=self.num_grid_points + 2)[
                1:-1
            ]
        ).to(int)
        return iter(grid_points)


class H5Dataset(Dataset):

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.current_path = None

    def __getitem__(self, idx):
        # why fill dataset at getitem rather than init?
        # each worker (which are forked after the init) need to have their own file handle
        if (
            self.current_path is None
            or self.file_paths[self.file_idx.value] != self.current_path
        ):
            self.current_path = self.file_paths[self.file_idx.value]

        with h5py.File(self.current_path, "r") as file:
            dataset = file.get("video")[idx]
            return (dataset.transpose(2, 0, 1) / 255.0).astype(np.float32)


def init_fn(v_file_idx, worker_id):
    info = get_worker_info()
    info.dataset.file_idx = v_file_idx


def process_files(rank, p_rank, args, file_queue, file_paths, model):

    print(f"Started process {p_rank} on GPU {rank}")

    # just in case GeoCalib internals use default
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
    model = model.to(f"cuda:{rank}")

    # each process assigns num_workers workers for data loading
    dataset = H5Dataset(file_paths)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=RegularSampler(dataset, args.batch_size),
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
                os.path.join(os.path.dirname(file_path), f"camera_{os.path.basename(file_path)}")
            ):
                try:
                    with h5py.File(
                        os.path.join(
                            os.path.dirname(file_path),
                            f"camera_{os.path.basename(file_path)}",
                        ),
                        "r",
                    ) as calib_file:
                        K = calib_file["camera"][:]
                    assert K.shape == (3, 3)
                    print(
                        f"Calib H5 file for {file_path} already processed, skipping!"
                    )
                    continue
                except Exception as e:
                    print(
                        f"Calib H5 file for {file_path} seems to be corrupt. Will overwrite."
                    )
                    os.remove(
                        os.path.join(
                            os.path.dirname(file_path),
                            f"camera_{os.path.basename(file_path)}",
                        )
                    )
            
            with h5py.File(file_path, "r") as f:
                num_written = f["num_written"][0]

            # assign file to dataloader
            v_file_idx.value = file_idx
            data_loader._index_sampler.sampler.dataset_len = num_written

            # open target h5
            with h5py.File(
                os.path.join(os.path.dirname(file_path), f"camera_{os.path.basename(file_path)}"),
                "w",
            ) as calib_file:

                # grab a single batch, randomly sampled from video
                data = next(iter(data_loader)).to(f"cuda:{rank}")

                if args.camera_model == "pinhole":
                    result = model.calibrate(
                        data,
                        camera_model="pinhole",
                        shared_intrinsics=True,
                    )
                    # no distortion parameters because we use pinhole model
                    K = result["camera"].K
                    # roll and pitch angles of the gravity vector
                    rp = result["gravity"].rp
                else:
                    # batched inference is unfortunately not possible
                    results = [
                        model.calibrate(
                            d, camera_model=args.camera_model, num_steps=200
                        )
                        for d in data
                    ]
                    K = torch.concatenate([r["camera"].K for r in results], dim=0)
                    rp = torch.concatenate([r["gravity"].rp for r in results], dim=0)
                    k1 = torch.concatenate([r["camera"].k1 for r in results], dim=0)

                # main field
                calib_file.create_dataset(
                    "camera",
                    data=K.mean(dim=0).cpu().numpy(),
                    dtype="float32",
                )

                # if we don't use pinhole model, save distortion
                if args.camera_model != "pinhole":
                    dist = np.zeros(5, dtype=np.float32)
                    dist[0] = k1.mean(dim=0).item()
                    calib_file.create_dataset(
                        "distortion",
                        data=dist,
                        dtype="float32",
                    )

                # save gravity
                calib_file.create_dataset(
                    "gravity",
                    data=rp.mean(dim=0).cpu().numpy(),
                    dtype="float32",
                )

            print(f"Finished processing {file_path} on gpu {rank} process {p_rank}")

            if (not args.no_profiler) and (rank == 0):
                prof.step()

        except Exception as e:
            print(e)
            print(f"Error processing calib for {file_path}. Moving on.")
            with open(
                os.path.join(args.log_dir, f"{args.exp_name}_failed_calib.txt"), "a"
            ) as f:
                f.write(f"{file_path} REASON: {e}\n")

    if (not args.no_profiler) and (rank == 0):
        prof.stop()


if __name__ == "__main__":

    mp.set_start_method("spawn")

    args = parser.parse_args()

    assert args.num_workers > 0    

    print(datetime.now())
    print("Starting GeoCalib with the following parameters:")
    print(f"exp-name: {args.exp_name}")
    print(f"num-gpus: {args.num_gpus}")
    print(f"num-proc-per-gpu: {args.num_proc_per_gpu}")
    print(f"batch-size: {args.batch_size}")
    print(f"num-workers: {args.num_workers}")
    print(f"camera-model: {args.camera_model}")

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

    os.makedirs(args.log_dir, exist_ok=True)

    model = GeoCalib()

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

    print("DONE - geocalib")