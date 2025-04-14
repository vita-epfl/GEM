from tqdm import tqdm
import numpy as np
import os
import cv2
import time
import random
import torch
from dwpose.util import draw_pose
from p_tqdm import p_map
from dwpose.dwpose_detector import DWposeDetector
from time import time
import h5py

dwpose_detector = DWposeDetector(
    device="cuda:0",
    model_det="models/yolox_l.onnx",
    model_pose="models/dw-ll_ucoco_384.onnx",
)

# Loading frames
video_path = "/store/swissai/a03/datasets/OpenDV-YouTube/h5/NTPTVtSzXPo.h5"
with h5py.File(video_path, "r") as f:
    frames = f["video"][:1000]
# Pose detection
print("Detecting poses...")
start = time()
detected_poses = p_map(dwpose_detector, frames, num_cpus=8)
end = time()

print(f"Time taken: {end-start}")
