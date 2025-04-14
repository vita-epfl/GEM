import time
import pandas as pd
import random
import ast
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path
import numpy as np
import os
import json
from dwpose.util import draw_pose

import h5py
import numpy as np
import torch
from p_tqdm import p_map
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from numpy.linalg import inv
from gem.data.image_cropper import center_resize
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


def normalize_trajectory(trajectory):
    trajectory = trajectory - trajectory[0]

    direction_vector = trajectory[1]
    angle_to_y_axis = np.arctan2(direction_vector[0], direction_vector[1])
    rotation_matrix = np.array(
        [
            [np.cos(angle_to_y_axis), -np.sin(angle_to_y_axis)],
            [np.sin(angle_to_y_axis), np.cos(angle_to_y_axis)],
        ]
    )
    normalized_points = trajectory.dot(rotation_matrix.T)
    return normalized_points


def process_file(h5_path):
    try:
        with h5py.File(h5_path, "r") as f:
            return f["video"].shape[0]
    except Exception as e:
        return None


def load_csv_to_list(file_path, chunksize=200000):
    rows = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        rows.extend(chunk.values.tolist())
    return rows


def get_rendered_poses(file_path, idx, num_frames, height, width):
    detected_poses = []
    with h5py.File(file_path, "r") as f:
        bodies = f["bodies"][idx : idx + num_frames]
        bodies_score = f["bodies_score"][idx : idx + num_frames]
        bodies_subset = f["bodies_subset"][idx : idx + num_frames]
        faces = f["faces"][idx : idx + num_frames]
        faces_score = f["faces_score"][idx : idx + num_frames]
        validity = f["validity"][idx : idx + num_frames]

        rendered_poses = []

        for i in range(len(bodies)):
            detected_pose = {
                "bodies": {
                    "candidate": bodies[i],  # (6, 18)
                    "score": bodies_score[i],  # (108, 2)
                    "subset": bodies_subset[i],
                },
                "hands": None,
                "faces": faces[i],
                "faces_score": faces_score[i],
            }

            for j in range(6):
                if validity[i, j] == 0:
                    continue
                if random.random() < 0.3:
                    detected_pose["bodies"]["subset"][j] = -1
                    detected_pose["bodies"]["score"][j] = 0
                    detected_pose["faces"][j] = 0

            detected_poses.append(detected_pose)

    for detected_pose in detected_poses:
        rendered_poses.append(draw_pose(detected_pose, height, width))

    rendered_poses = torch.tensor(np.array(rendered_poses)).float()
    return rendered_poses


class CuratedDataset(Dataset):
    def __init__(
        self,
        data_dict,
        total_anno_file="annotations.csv",
        # Image settings
        target_width=1024,
        target_height=576,
        num_frames=25,
        # Curation settings
        step_size=1,
        filter=False,
        aesthetic_threshold=0,
        piqe_threshold=100,
        blur_threshold=10000,
        patch_similarity_50=0.0,
        average_cosine_similarity_threshold=0.0,
        environment_score_threshold=[0, 0, 0, 0],
        motion_score_threshold=0.05,
        diversity_setting=None,
        sscd_setting=None,
        reference_frame_horizon=0,
    ):
        super().__init__()
        self.data_dict = data_dict
        self.resize_quality = "quality"  # "fast" or "quality"
        self.total_anno_file = total_anno_file
        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames
        self.resize_transform = transforms.Resize(
            (target_height, target_width),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.filter = filter

        self.step_size = step_size
        self.samples = None

        self.aesthetic_threshold = aesthetic_threshold
        self.piqe_threshold = piqe_threshold
        self.blur_threshold = blur_threshold
        self.patch_similarity_50 = patch_similarity_50
        self.average_cosine_similarity_threshold = average_cosine_similarity_threshold
        self.environment_score_threshold = environment_score_threshold
        self.motion_score_threshold = motion_score_threshold
        self.diversity_setting = diversity_setting
        self.sscd_setting = sscd_setting
        self.diversity_keys = [
            "in_similarity_98",
            "in_similarity_95",
            "in_similarity_90",
            "in_similarity_85",
            "in_similarity_75",
        ]
        self.reference_frame_horizon = reference_frame_horizon
        self.similarity_keys = [
            "sscd_similarity_95",
            "sscd_similarity_75",
            "sscd_similarity_50",
        ]

        self.process_metadata(total_anno_file)

    def should_skip_sample(
        self,
        idx,
        aesthetic_score,
        piqe_score,
        blur_score,
        motion_score,
        average_cosine_similarity,
        patch_similarity_50,
        in_dino,
        similarity,
        environment_score,
    ):
        if not self.filter:
            return False

        skip_reasons = {
            "aesthetic": aesthetic_score < self.aesthetic_threshold,
            "piqe": piqe_score > self.piqe_threshold,
            "blur": blur_score > self.blur_threshold,
            "patch_similarity_50": patch_similarity_50 < self.patch_similarity_50,
            "average_cosine_similarity": average_cosine_similarity
            < self.average_cosine_similarity_threshold,
            "diversity": in_dino == 0,
            "similarity": similarity == 0,
            "environment_score": any(
                [
                    environment_score[i] < self.environment_score_threshold[i]
                    for i in range(4)
                ]
            ),
            "motion": motion_score < self.motion_score_threshold,
        }
        for reason, condition in skip_reasons.items():
            if condition:
                self.skip_statistic[reason] += 1
                return True
        return False

    def proc_row(self, row):
        path = Path(row[0])

        skeleton_path = row[-3]
        depth_file = row[-2]
        traj_file = row[-1]

        if depth_file is None:
            return None

        sample_data = {
            "path": path,
            "depth_path": depth_file,
            "trajectory_path": traj_file,
            "skeleton_path": skeleton_path,
            "idx": int(row[1]),
            "aesthetic_score": float(row[3]),
            "piqe_score": float(row[4]),
            "blur_score": float(row[5]),
            "environment_score": [0, 0, 0, 0],  # eval(row[5]),
            # **{key: int(row[7 + i]) for i, key in enumerate(self.diversity_keys)},
            **{key: int(row[7 + i]) if pd.notna(row[7 + i]) else 0 for i, key in enumerate(self.diversity_keys)},
            # **{key: float(row[key]) for key in enumerate(self.similarity_keys)},
            "patch_similarity_50": float(row[15]),
            "average_cosine_similarity": float(row[16]),
            "motion_score": float(row[17]),
            "patchwise_motion_vectors": [0, 0, 0, 0],  # eval(row[7]),
        }
        # Apply skip logic and collect samples
        self.skip_statistic["total"] += 1

        if not self.should_skip_sample(
            idx=sample_data["idx"],
            aesthetic_score=sample_data["aesthetic_score"],
            piqe_score=sample_data["piqe_score"],
            blur_score=sample_data["blur_score"],
            motion_score=sample_data["motion_score"],
            average_cosine_similarity=sample_data["average_cosine_similarity"],
            patch_similarity_50=sample_data["patch_similarity_50"],
            in_dino=(
                sample_data[self.diversity_setting] if self.diversity_setting else 1
            ),
            similarity=(sample_data[self.sscd_setting] if self.sscd_setting else 1),
            environment_score=sample_data["environment_score"],
        ):
            return sample_data

        return None

    def proc_meta_file_whole(self, row_list):
        path_list = set([row[0] for row in row_list])
        print(f"Total paths: {len(path_list)}")
        samples = [self.proc_row(row) for row in tqdm(row_list)]
        # filter out nones
        total_samples = len(samples)
        samples = [sample for sample in samples if sample is not None]

        filtered_samples = len(samples)
        print(f"Total samples: {total_samples}, Filtered samples: {filtered_samples}")
        print(f"Filtering rate: {100*(1 - filtered_samples / total_samples):.2f}%")

        return samples

    def process_metadata(self, total_anno_file=None):
        # Initialize sample list and skip statistics
        self.samples = []
        self.skip_statistic = {
            key: 0
            for key in [
                "aesthetic",
                "piqe",
                "blur",
                "motion",
                "diversity",
                "environment_score",
                "similarity",
                "patch_similarity_50",
                "average_cosine_similarity",
                "total",
            ]
        }
        
        assert total_anno_file is not None, "Please provide a total annotation file"
        self.total_anno_file = total_anno_file
        rows_list = load_csv_to_list(total_anno_file)
        self.samples = self.proc_meta_file_whole(rows_list)
        self.samples = [sample for sample in self.samples]
 
        # self.samples = [ sample for path in self.h5_paths for sample in self.proc_meta_file(path) ]
        # self.samples = [sample for sample in self.samples if sample is not None]
        print(len(self.samples))
        assert len(self.samples) > 0, "No samples found after filtering!"

        # Calculate sample count based on frame step size
        samples_per_clip = (
            1 + self.num_frames // self.step_size
            if self.num_frames != self.step_size
            else 1
        )
        self.num_samples = (
            (len(self.samples) * samples_per_clip) - (samples_per_clip - 1)
            if self.num_frames != self.step_size
            else len(self.samples)
        )

        with open("skip_statistics.csv", "a+") as f:
            writer = csv.DictWriter(f, fieldnames=self.skip_statistic.keys())
            writer.writeheader()
            writer.writerow(self.skip_statistic)

        with open("skip_statistics.csv", "a") as f:
            settings = {
                "aesthetic_threshold": self.aesthetic_threshold,
                "piqe_threshold": self.piqe_threshold,
                "blur_threshold": self.blur_threshold,
                "motion_score_threshold": self.motion_score_threshold,
                "patch_similarity_50": self.patch_similarity_50,
                "average_cosine_similarity_threshold": self.average_cosine_similarity_threshold,
                "environment_score_threshold": self.environment_score_threshold,
                "diversity_setting": self.diversity_setting,
                "sscd_setting": self.sscd_setting,
            }
            writer = csv.DictWriter(f, fieldnames=settings.keys())
            writer.writeheader()
            writer.writerow(settings)

        return 0

    def build_data_dict(self, image_seq, sample_dict, reference_frame_idx=0):
        # cond_aug = torch.rand(1) * 0.8
        reference_frame = image_seq[0]
        image_seq = image_seq[1:] if self.reference_frame_horizon > 0 else image_seq
        random = torch.rand(1).item()
        # cond_aug = [torch.rand(1).item()*5 if random < 0.5 else torch.tensor([0.0])][0]
        cond_aug = torch.rand(1) * 5 if random < 0.5 else torch.tensor([0.0])

        data_dict = {
            "img_seq": image_seq,
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([10]),
            "cond_frames_without_noise": reference_frame,
            "cond_frames": reference_frame + cond_aug * torch.randn_like(image_seq[0]),
            "cond_aug": cond_aug,
            "reference_frame_idx": torch.tensor([reference_frame_idx]),
        }
        if self.reference_frame_horizon > 0:
            data_dict["first_frame"] = image_seq[0]

        if sample_dict is not None:
            data_dict.update(sample_dict)
        return data_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, frame_idx):
        # Find out which video this index belongs to
        reference_frame_idx = 0
        # print(f"frame_idx: {frame_idx}")
        samples_per_clip = (
            1 + self.num_frames // self.step_size
            if self.num_frames != self.step_size
            else 1
        )
        clip_idx = frame_idx // samples_per_clip
        h5_path = self.samples[clip_idx]["path"]
        h5_depth_path = self.samples[clip_idx]["depth_path"]
        h5_trajectory_path = self.samples[clip_idx]["trajectory_path"]
        h5_skeleton_path = self.samples[clip_idx]["skeleton_path"]

        idx = self.samples[clip_idx]["idx"]
        # given the clip we are in, and the start index of the clip, we can calculate the index of the frame
        idx = idx + self.step_size * (frame_idx % samples_per_clip)

        sample_dict = {}

        # check if the idx is within the bounds of the video
        with h5py.File(h5_path, "r") as f:
            if idx + self.num_frames > f["video"].shape[0]:
                idx = f["video"].shape[0] - self.num_frames
            video = f["video"][idx : idx + self.num_frames]
            if self.reference_frame_horizon > 0:
                reference_frame_idx = random.randint(
                    # 0, self.reference_frame_horizon
                    max(0, idx - self.reference_frame_horizon),
                    idx,
                )
                # if reference_frame_idx > 0:
                reference_frame = f["video"][reference_frame_idx]
                reference_frame_idx = -reference_frame_idx + idx
                video = np.concatenate([reference_frame[np.newaxis], video], axis=0)

        if "depth_img" in self.data_dict:
            try:
                with h5py.File(h5_depth_path, "r") as d:
                    depth = d["depth"][idx : idx + self.num_frames]

                    if "is_float16" in d and d["is_float16"][0] == 1:
                        # float16 files are already normalized
                        depth = depth.astype(np.float32)
                    else:
                        # float32 files are not normalized
                        depth = depth / 80.0

                depth = torch.from_numpy(depth).float()
                depth = depth.unsqueeze(1).repeat(1, 3, 1, 1)
                # Crop center and resize
                depth = center_resize(depth, self.target_width, self.target_height, self.resize_quality)

                depth = (depth - 0.5) * 2.0  # [-1, 1] range
            except Exception as e:
                print(f"Error loading depth: {h5_depth_path} with: {e}")
                depth = torch.zeros(self.num_frames, 3, self.target_height, self.target_width)
        else:
            depth = torch.zeros(self.num_frames, 3, self.target_height, self.target_width)

        if "trajectory" in self.data_dict:
            try:
                with h5py.File(h5_trajectory_path, "r") as t:
                    if "is_nuscenes" in t and t["is_nuscenes"][0] == True:
                        pos_2d = t["trajectory"][:]
                        pos_2d = np.concatenate([np.zeros((1, 2)), pos_2d], axis=0)
                        # interpolate the trajectory from 4 to 25 frames
                        subset_indices = np.arange(0, 25, 5)
                        full_indices = np.arange(25)
                        traj_full = np.zeros((25, 2))
                        for i in range(2):
                            f = interp1d(
                                subset_indices,
                                pos_2d[:, i],
                                kind="linear",
                                fill_value="extrapolate",
                            )
                            traj_full[:, i] = f(full_indices)
                        gt = traj_full
                        gt = gaussian_filter1d(gt, sigma=10, axis=0, mode="nearest")
                        pos_2d = normalize_trajectory(gt)[::5][1:]
                        pos_2d = pos_2d.reshape(-1)
                        pos_2d = torch.from_numpy(pos_2d).float()
                    else:
                        start_idx = t["start_idx"][:]
                        traj_idx = np.argmin(start_idx <= idx) - 1
                        traj_frame_idx = idx - start_idx[traj_idx]
                        trajectory = t["trajectory"][
                            traj_idx, traj_frame_idx : traj_frame_idx + self.num_frames
                        ]

                        trajectory_2d_david = np.array(
                            [trajectory[:, 0, 3], trajectory[:, 2, 3]]
                        ).T
                        trajectory_2d_david = gaussian_filter1d(
                            trajectory_2d_david, sigma=10, axis=0, mode="nearest"
                        )
                        trajectory_2d_david = normalize_trajectory(trajectory_2d_david)
                        pos_2d = (trajectory_2d_david[::5][1:]).reshape(-1)
                        pos_2d = torch.from_numpy(pos_2d).float()
            except Exception as e:
                print(f"Error loading trajectory: {h5_trajectory_path} with: {e}")
                pos_2d = torch.zeros(8).float()
        else:
            pos_2d = torch.zeros(8).float()

        if "rendered_poses" in self.data_dict:
            try:
                rendered_poses = get_rendered_poses(
                    h5_skeleton_path,
                    idx,
                    self.num_frames,
                    self.target_height,
                    self.target_width,
                )
            except Exception as e:
                print("Error loading skeletons: ", e)
                rendered_poses = torch.zeros(self.num_frames, 3, self.target_height, self.target_width)
        else:
            rendered_poses = torch.zeros(self.num_frames, 3, self.target_height, self.target_width)

        # check for any nans in pos_2d or depth
        if torch.isnan(pos_2d).any():
            pos_2d = torch.zeros(8).float()
            print("WARN: Found NaNs in pos_2d... Switching to zeros")
        if torch.isnan(depth).any():
            depth = torch.zeros(
                self.num_frames, 3, self.target_height, self.target_width
            )
            print("WARN: Found NaNs in depth... Switching to zeros")
        if torch.isnan(rendered_poses).any():
            rendered_poses = torch.zeros(
                self.num_frames, 3, self.target_height, self.target_width
            )
            print("WARN: Found NaNs in rendered_poses... Switching to zeros")

        # preprocess
        video = torch.from_numpy(video).float().permute(0, 3, 1, 2)
        video = video / 255.0  # [0, 1] range
        video = center_resize(
            video, self.target_width, self.target_height, self.resize_quality
        )  # Crop center and resize
        video = torch.clamp(video, 0, 1)  # Clamp to [0, 1] just to be sure
        video = video * 2.0 - 1.0

        if depth is None:
            depth = torch.zeros_like(video)

        if len(pos_2d.shape) != 1:
            print("Found a weird pos_2d shape: ", pos_2d.shape)
            pos_2d = pos_2d.reshape(-1)

        if pos_2d.shape[0] != 8:
            print("Found a very weird pos_2d shape: ", pos_2d.shape)
            pos_2d = torch.zeros(8).float()

        data_dict = self.build_data_dict(video, sample_dict, reference_frame_idx)

        if "depth_img" in self.data_dict:
            data_dict["depth_img"] = depth
        
        if "trajectory" in self.data_dict:
            data_dict["trajectory"] = pos_2d
        
        if "rendered_poses" in self.data_dict:
            data_dict["rendered_poses"] = rendered_poses

        return data_dict


class CuratedSampler(LightningDataModule):
    def __init__(
        self,
        data_dict,
        batch_size,
        total_anno_file="annotations.csv",
        num_workers=0,
        prefetch_factor=2,
        shuffle=True,
        target_height=320,
        target_width=576,
        num_frames=25,
        step_size=1,
        filter=False,
        aesthetic_threshold=0,
        piqe_threshold=100,
        motion_score_threshold=0.0,
        blur_threshold=10000,
        patch_similarity_50=0.0,
        average_cosine_similarity_threshold=0.0,
        sscd_setting=None,
        environment_score_threshold=[0, 0, 0, 0],
        diversity_setting=None,
        reference_frame_horizon=0,
        **kwargs,
    ):
        super().__init__()
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.shuffle = shuffle
        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames
        self.step_size = step_size
        self.filter = filter
        self.aesthetic_threshold = aesthetic_threshold
        self.piqe_threshold = piqe_threshold
        self.motion_score_threshold = motion_score_threshold
        self.blur_threshold = blur_threshold
        self.environment_score_threshold = environment_score_threshold
        self.diversity_setting = diversity_setting
        self.reference_frame_horizon = reference_frame_horizon
        self.kwargs = kwargs
        self.sscd_setting = sscd_setting
        self.patch_similarity_50 = patch_similarity_50
        self.average_cosine_similarity_threshold = average_cosine_similarity_threshold
        self.total_anno_file = total_anno_file

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = CuratedDataset(
                data_dict=self.data_dict,
                total_anno_file=self.total_anno_file,
                target_height=self.target_height,
                target_width=self.target_width,
                num_frames=self.num_frames,
                step_size=self.step_size,
                filter=self.filter,
                aesthetic_threshold=self.aesthetic_threshold,
                piqe_threshold=self.piqe_threshold,
                motion_score_threshold=self.motion_score_threshold,
                blur_threshold=self.blur_threshold,
                environment_score_threshold=self.environment_score_threshold,
                diversity_setting=self.diversity_setting,
                reference_frame_horizon=self.reference_frame_horizon,
                patch_similarity_50=self.patch_similarity_50,
                average_cosine_similarity_threshold=self.average_cosine_similarity_threshold,
                sscd_setting=self.sscd_setting,
                **self.kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            # sampler=sampler,
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,  # we disable online testing to improve training efficiency
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
