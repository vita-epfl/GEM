import h5py
import numpy as np


OVERLAP = 100


# use this in datasets root dir to create list of all trajectory h5:
# find ~+ -type f -name "trajectory_*.h5" > $SCRATCH/trajectory_inference/traj_list.txt


def main(file_list):

    for file_name in file_list:

        print(f"do {file_name}")

        with h5py.File(file_name, "r+") as f:

            num_written = f["num_written"][0]
            padded_trajectories = f["trajectory"][:]
            start_idx = f["start_idx"][:]
            stop_idx = f["stop_idx"][:]

            if len(padded_trajectories) == 1:
                continue

            trajectories = [
                t[: (stop - start)]
                for t, start, stop in zip(padded_trajectories, start_idx, stop_idx)
            ]

            # merge trajectories
            while len(trajectories) > 1:
                # cut ends to yield overlap=1
                trajectories[0] = trajectories[0][
                    : len(trajectories[0]) - (OVERLAP - 1) // 2
                ]
                trajectories[1] = trajectories[1][OVERLAP // 2 :]
                T = np.linalg.solve(
                    trajectories[1][0].T,
                    trajectories[0][-1].T,
                ).T
                B = trajectories[1][0]
                B_inv = np.eye(4)
                B_inv[:3, :3] = B[:3, :3].T
                B_inv[:3, 3] = -B[:3, :3].T @ B[:3, 3]
                trans = np.einsum("AB,NBC->NAC", T, trajectories[1][1:])
                trajectories[0] = np.concatenate([trajectories[0], trans], axis=0)
                trajectories.pop(1)

            if len(trajectories[0]) != num_written:
                with open("./failed_trajectory_correction.txt", "a") as logf:
                    logf.write("{file_name}\n")
                continue

            try:
                del f["merged_trajectory"]
            except KeyError:
                pass
            f.create_dataset("merged_trajectory", data=trajectories[0], dtype="float32")


if __name__ == "__main__":

    with open("traj_list.txt", "r") as f:
        file_list = f.read().splitlines()

    main(file_list)
