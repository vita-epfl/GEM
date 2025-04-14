from tqdm import tqdm
import decord
import numpy as np
import os
import cv2
import imageio
import time
import random
import torch
from dwpose.util import draw_pose
from dwpose.dwpose_detector import dwpose_detector as dwprocessor
from pose_net import PoseNet


# Loading frames
video_path = (
    "/mnt/vita/scratch/datasets/OpenDV-YouTube/full_images/4K_DRIVE/3E4HcpHpbZs"
)
video_frame_paths = sorted(
    [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".jpg")]
)[75:125]

width, height = 1024, 576
frames = []
for frame_path in video_frame_paths:
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

# Pose detection
detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]

# drawing all poses
output_pose = []
for detected_pose in detected_poses:
    im = draw_pose(detected_pose, height, width)
    output_pose.append(np.array(im))

# filtering poses with low confidence, and randomly removing some people, and drawing the filtered poses
filtered_poses = []
for i in range(len(detected_poses)):
    if len(detected_poses[i]["bodies"]["score"]) == 0:
        filtered_poses.append(np.zeros((height, width, 3), dtype=np.uint8))
        continue
    for j in range(len(detected_poses[i]["bodies"]["score"])):
        rand = random.random()
        if (detected_poses[i]["bodies"]["score"][j] < 0.5).sum() > len(
            detected_poses[i]["bodies"]["score"][0]
        ) / 2 or rand < 0.5:
            # if more than half of the keypoints are invisible, or the random number is less than 0.5, then remove the person
            detected_poses[i]["bodies"]["subset"][j] = -1
            detected_poses[i]["bodies"]["score"][j] = 0
            detected_poses[i]["faces"][j] = 0
    filtered_poses.append(draw_pose(detected_poses[i], height, width))

# saving the video
writer = imageio.get_writer("rand_pose_video.mp4", fps=10)
for i in range(len(output_pose)):
    writer.append_data(
        np.vstack(
            [
                frames[i],
                output_pose[i].transpose((1, 2, 0)),
                filtered_poses[i].transpose((1, 2, 0)),
            ]
        )
    )
writer.close()

# PoseNet
posenet = PoseNet(noise_latent_channels=320)  # 200k parameters
out = posenet(
    torch.tensor(np.array(filtered_poses)).float() / 255.0
)  # shape 50, *320*, 72, 128 for num_frames, noise_latent_channels, height//8, width//8
