import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

with h5py.File("/scratch/pm23r001/h5_dataset/depth_CzLRoZNXuIg.h5", "r") as f:
    depth = f["depth"][:]
    print(depth.max(), depth.min())

with h5py.File("/scratch/pm23r001/h5_controls/trajectory_scene.h5", "r") as f:
    poses = f["trajectory"][:]
    merged_poses = f["merged_trajectory"][:]
    num_written = f["num_written"][:]
    start_idx = f["start_idx"][:]
    stop_idx = f["stop_idx"][:]

assert poses.shape[0] == 1
trajectories = poses[0, 0:25]
this_pose = trajectories[:-5]
next_pose = trajectories[5:]
relative_pose = np.linalg.solve(this_pose, next_pose)
print(
    trajectories.shape,
)
print(poses.shape, relative_pose.shape)

xtr = relative_pose[:, 0, 3]
plt.plot(xtr)
plt.ylabel("~ steering angle right")
plt.savefig("steering.png")

ztr = relative_pose[:, 2, 3]
plt.plot(ztr)
plt.ylabel("speed")
plt.savefig("speed.png")

print(np.abs(relative_pose[:, :, -1]).mean(axis=-0))

plt.title("BEV trajectory")
plt.plot(trajectories[:, 0, 3], trajectories[:, 2, 3])
plt.gca().set_aspect("equal")
plt.savefig("trajectory.png")
