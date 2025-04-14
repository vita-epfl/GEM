import os
import argparse
from functools import partial
import h5py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info
from torch.profiler import profile, ProfilerActivity
from lietorch import SO3

from droid_trajectory.droid_core.droid import Droid


parser = argparse.ArgumentParser(prog="DroidSLAM_inference")
# paths
parser.add_argument("--file-list", type=str, default="./file_list.txt")
parser.add_argument("--weights", type=str, default="./droid.pth")
# tuning parameters
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--num-proc-per-gpu", type=int, default=1)
parser.add_argument("--trajectory-length", type=int, default=64)
parser.add_argument("--trajectory-overlap", type=int, default=5)
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument("--min-trajectory-length", type=int, default=30)
parser.add_argument("--file-list-idx", type=int, default=0)
# constants
parser.add_argument("--image-height", type=int, default=576)
parser.add_argument("--image-width", type=int, default=1024)
# profiling
parser.add_argument("--no-profiler", action="store_true")
parser.add_argument("--log-dir", type=str, default="./logs")
parser.add_argument("--exp-name", type=str, default="droidslam_exp")


class SequentialSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.dataset_len = None

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        return iter(range(self.dataset_len))


class H5Dataset(Dataset):

    def __init__(self, file_paths, resize_size, crop_size, image_size, args):
        self.file_paths = file_paths
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.image_size = image_size
        self.current_path = None
        self.args = args

    def __getitem__(self, idx):
        # why fill dataset at getitem rather than init?
        # each worker (which are forked after the init) need to have their own file handle
        if self.file_paths[self.file_idx.value] != self.current_path:
            self.current_path = self.file_paths[self.file_idx.value]
            # intrinsics
            with h5py.File(
                os.path.join(
                    os.path.dirname(self.current_path),
                    f"camera_{os.path.basename(self.current_path)}",
                ),
                "r",
            ) as calib_file:
                K = calib_file["camera"][:]
                if "distortion" in calib_file:
                    self.undistort = True
                    dist = calib_file["distortion"][:]
                    new_K, _ = cv2.getOptimalNewCameraMatrix(
                        K,
                        dist,
                        (self.image_size[1], self.image_size[0]),
                        0,
                        (self.image_size[1], self.image_size[0]),
                    )
                    self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                        K,
                        dist,
                        None,
                        new_K,
                        (self.image_size[1], self.image_size[0]),
                        5,  # CV_32FC1
                    )
                    K = new_K
                else:
                    self.undistort = False
                # rescale intrinsics
                fx = K[0, 0] * self.resize_size[1] / self.image_size[1]
                fy = K[1, 1] * self.resize_size[0] / self.image_size[0]
                cx = K[0, 2] * self.resize_size[1] / self.image_size[1]
                cy = K[1, 2] * self.resize_size[0] / self.image_size[0]
            self.intrinsics = np.array([fx, fy, cx, cy])

        with h5py.File(self.current_path, "r") as file:
            image = file.get("video")[idx + self.index_offset.value]

        with h5py.File(
            os.path.join(
                os.path.dirname(self.current_path), f"depth_{os.path.basename(self.current_path)}"
            ),
            "r",
        ) as file:
            depth = file.get("depth")[idx + self.index_offset.value]
            depth = depth.astype(float) * 80

        if self.undistort:
            image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self.mapx, self.mapy, cv2.INTER_NEAREST)

        # resize and crop
        image = cv2.resize(image, (self.resize_size[1], self.resize_size[0]))
        image = image[: self.crop_size[0], : self.crop_size[1]]
        image = image.transpose(2, 0, 1)

        depth = torch.as_tensor(depth)
        depth = nn.functional.interpolate(depth[None, None], self.resize_size).squeeze()
        depth = depth[: self.crop_size[0], : self.crop_size[1]]

        return idx, image[None], depth, self.intrinsics


def init_fn(v_file_idx, v_index_offset, worker_id):
    info = get_worker_info()
    info.dataset.file_idx = v_file_idx
    info.dataset.index_offset = v_index_offset


def quaternion_to_matrix(q):
    Q = SO3.InitFromVec(torch.Tensor(q))
    R = Q.matrix().detach().cpu().numpy().astype(np.float32)
    return R[:3, :3]


def get_pose_matrix(traj):
    Ts = []
    for i in range(len(traj)):
        pose = traj[i]
        t, q = pose[1:4], pose[4:]
        R = quaternion_to_matrix(q)
        T = np.eye(4)
        # Twc = [R | t]
        T[:3, :3] = R
        T[:3, 3] = t
        Ts.append(T)
    return np.stack(Ts, axis=0)


def process_files(rank, p_rank, args, file_queue, file_paths):

    # just in case DroidSLAM internals use default
    torch.cuda.set_device(f"cuda:{rank}")

    # define global
    v_file_idx = mp.Value("i", 0)
    v_index_offset = mp.Value("i", 0)

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

    # resize and crop as in DroidSLAM repo
    resize_height = int(
        args.image_height
        * np.sqrt((384 * 512) / (args.image_height * args.image_width))
    )
    resize_width = int(
        args.image_width * np.sqrt((384 * 512) / (args.image_height * args.image_width))
    )
    crop_height = resize_height - resize_height % 8
    crop_width = resize_width - resize_width % 8

    # each process assigns num_workers workers for data loading
    dataset = H5Dataset(
        file_paths,
        resize_size=[resize_height, resize_width],
        crop_size=[crop_height, crop_width],
        image_size=[args.image_height, args.image_width],
        args=args,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=partial(init_fn, v_file_idx, v_index_offset),
    )

    while True:
        file_idx = file_queue.get()
        if file_idx == "DONE":
            break

        try:  # catch all errors
            file_path = file_paths[file_idx]

            # check if file is already processed
            if os.path.exists(
                os.path.join(
                    os.path.dirname(file_path),
                    f"trajectory_{os.path.basename(file_path)}",
                )
            ):
                try:
                    with h5py.File(
                        os.path.join(
                            os.path.dirname(file_path),
                            f"trajectory_{os.path.basename(file_path)}",
                        ),
                        "r",
                    ) as trajectory_file:
                        tmp = trajectory_file["num_written"][0]
                        tmp2 = trajectory_file["stop_idx"][-1]
                        tmp3 = trajectory_file["trajectory"][-1]
                    assert np.array_equal(tmp, tmp2)
                    assert not np.array_equal(tmp3, np.zeros_like(tmp3))
                    print(
                        f"Trajectory H5 file for {file_path} already processed, skipping!"
                    )
                    continue
                except Exception as e:
                    print(
                        f"Trajectory H5 file for {file_path} seems to be corrupt. Will overwrite."
                    )
                    os.remove(
                        os.path.join(
                            os.path.dirname(file_path),
                            f"trajectory_{os.path.basename(file_path)}",
                        )
                    )

            with h5py.File(file_path, "r") as f:
                num_written = f["num_written"][0]

            # assign file to dataloader
            v_file_idx.value = file_idx

            # open target h5
            with h5py.File(
                os.path.join(
                    os.path.dirname(file_path),
                    f"trajectory_{os.path.basename(file_path)}",
                ),
                "w",
            ) as trajectory_file:

                trajectories = []
                start_idx = []
                end_idx = []
                new_p = 0
                while new_p < num_written:

                    # adjust index_offset for last trajectory
                    if (
                        num_written - new_p + args.trajectory_overlap
                        < args.min_trajectory_length
                    ):
                        index_offset = max(0, num_written - args.min_trajectory_length)
                    else:
                        index_offset = max(0, new_p - args.trajectory_overlap)

                    # create model and push to assigned GPU
                    droid = Droid(
                        weights=args.weights,
                        image_size=[crop_height, crop_width],
                        upsample=True,
                        buffer=args.trajectory_length,
                        device=f"cuda:{rank}",
                    )

                    # assign new trajectory
                    remaining_length = num_written - index_offset
                    segment_length = min(remaining_length, args.trajectory_length)
                    data_loader._index_sampler.dataset_len = segment_length
                    v_index_offset.value = index_offset

                    for data in data_loader:
                        droid.track(*data)

                    # do global bundle adjustment
                    traj_est = droid.terminate(data_loader)
                    traj_est = get_pose_matrix(traj_est)
                    trajectories.append(traj_est)
                    start_idx.append(index_offset)
                    end_idx.append(index_offset + len(traj_est))

                    new_p = index_offset + segment_length

                    del droid

                    if (not args.no_profiler) and (rank == 0):
                        prof.step()

                # pad trajectories and stack
                padded_trajectories = []
                for t in trajectories:
                    if len(t) != args.trajectory_length:
                        # pad trajectory with zeros
                        fl = np.zeros((args.trajectory_length, 4, 4))
                        fl[: len(t)] = t
                        padded_trajectories.append(fl)
                    else:
                        padded_trajectories.append(t)
                trajectory_file.create_dataset(
                    "trajectory",
                    data=np.stack(padded_trajectories, axis=0),
                    dtype="float32",
                )
                trajectory_file.create_dataset(
                    "start_idx",
                    data=start_idx,
                    dtype="int32",
                )
                trajectory_file.create_dataset(
                    "stop_idx",
                    data=end_idx,
                    dtype="int32",
                )
                trajectory_file.create_dataset(
                    "num_written", data=[num_written], dtype="int32"
                )

            print(f"Finished processing {file_path} on gpu {rank} process {p_rank}")

        except Exception as e:
            print(e)
            print(f"Error processing trajectory for {file_path}. Moving on.")
            with open(
                os.path.join(args.log_dir, f"{args.exp_name}_failed_trajectory.txt"),
                "a",
            ) as f:
                f.write(f"{file_path} REASON: {e}\n")

    if (not args.no_profiler) and (rank == 0):
        prof.stop()


if __name__ == "__main__":

    mp.set_start_method("spawn")

    args = parser.parse_args()

    assert args.num_workers > 0
    assert args.min_trajectory_length <= args.trajectory_length

    print("Starting DroidSLAM with the following parameters:")
    print(f"exp-name: {args.exp_name}")
    print(f"num-gpus: {args.num_gpus}")
    print(f"num-proc-per-gpu: {args.num_proc_per_gpu}")
    print(f"trajectory-length: {args.trajectory_length}")
    print(f"trajectory-overlap: {args.trajectory_overlap}")
    print(f"min-trajectory-length: {args.min_trajectory_length}")
    print(f"num-workers: {args.num_workers}")

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
                ),
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print("DONE - droidslam")
