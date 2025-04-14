import os
import torch
import numpy as np
from ultralytics import YOLO
from gem.utils.roi_utils import roi_to_rectangle_region

# from mimicmotion.dwpose.util import draw_pose
from tqdm import tqdm
import cv2
import imageio
import json
from torchmetrics.image.fid import FrechetInceptionDistance
from fvd_utils import load_fvd_model, get_fvd_logits, frechet_distance
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random


class FVDVideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video = self.process_video_to_numpy(video_path)
        return video

    def process_video_to_numpy(
        self, video_path, target_size=(224, 224), intermediate_size=(448, 256)
    ):
        processed_frames = []
        frames = np.load(video_path)
        for frame in frames:
            # Convert the frame (as numpy array) to PIL Image (RGB)
            image = Image.fromarray(frame)

            # Center crop and resize to intermediate size
            ori_w, ori_h = image.size
            if ori_w / ori_h > intermediate_size[0] / intermediate_size[1]:
                tmp_w = int(intermediate_size[0] / intermediate_size[1] * ori_h)
                left = (ori_w - tmp_w) // 2
                right = (ori_w + tmp_w) // 2
                image = image.crop((left, 0, right, ori_h))
            elif ori_w / ori_h < intermediate_size[0] / intermediate_size[1]:
                tmp_h = int(intermediate_size[1] / intermediate_size[0] * ori_w)
                top = (ori_h - tmp_h) // 2
                bottom = (ori_h + tmp_h) // 2
                image = image.crop((0, top, ori_w, bottom))

            image = image.resize(intermediate_size, resample=Image.LANCZOS)

            # Convert image to tensor
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()

            # Final resize to target size (224, 224)
            processed_frames.append(image_tensor)
        processed_frames = torch.stack(processed_frames)
        resized_image_tensor = torch.nn.functional.interpolate(
            processed_frames, size=target_size, mode="bilinear", align_corners=False
        )

        # Convert tensor back to numpy array
        return (
            resized_image_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(int)
        )  # BHWC format


class FIDVideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        return np.load(video_path)


class EvaluationHelper:
    # ========================================== INITIALIZATION ==========================================
    def __init__(self, eval_mode, model):
        """
        eval_mode: str, evaluation mode, e.g. 'object_manupulation'
        options: 'unconditional', 'object_manipulation', 'skeleton_manipulation', 'object_insertion', 'ego_motion'
        """
        self.eval_mode = eval_mode
        if eval_mode == "object_manipulation":
            self.control_frames = [4, 9, 14, 19]
        elif eval_mode == "skeleton_manipulation":
            self.control_frames = list(range(25))
        self.load_models()
        self.model = model

    def load_models(self):
        if self.eval_mode == "object_manipulation":
            self.tracker_model = YOLO("yolo11n.pt")
        elif self.eval_mode == "skeleton_manipulation":
            from mimicmotion.dwpose.dwpose_detector import (
                dwpose_detector as dwprocessor,
            )

            self.pose_model = dwprocessor
        # TODO: add other models

    # ========================================== COMPATIBILITY CHECK ==========================================
    def is_compatible_video(self, video_frames):
        """
        returns True if the video is compatible with the evaluation mode, e.g. if there is a car in the first frame
        when trying to evaluate car moving control
        video_frames: list of frames in the original video
        return (bool, optional results)
        """
        if self.eval_mode in ["unconditional", "ego_motion"]:
            return True, None
        elif self.eval_mode == "object_manupulation":
            return self.check_car_exist(video_frames)
        elif self.eval_mode == "skeleton_manupulation":
            return self.check_human_exist(video_frames)
        elif self.eval_mode == "object_insertion":
            return self.check_empty_road_exist(video_frames)
        else:
            raise ValueError(f"Invalid evaluation mode: {self.eval_mode}")

    def check_car_exist(self, video_frames):
        """
        returns True if there is a car in the first frame
        video_frames: list of frames in the original video of shape (T, H, W, C)
        returns: (bool, tracking_results)
        """
        self.load_models()
        tracking_results = self.tracker_model.track(
            video_frames[0], persist=True, verbose=False
        )[0]
        boxes = tracking_results[-1].boxes
        if boxes.id is None:
            return False, tracking_results
        boxes = boxes.xyxy[boxes.cls == 2]  # only track cars
        width, height = video_frames[0].shape[1], video_frames[0].shape[0]
        best_box_id = self.select_best_box(boxes, width, height)
        if best_box_id is None:
            return False, tracking_results
        return True, (tracking_results, best_box_id)

    def select_best_box(self, boxes, width, height):
        """
        Select the best box from the list of boxes based on a heuristic
        :param boxes: list of boxes in the format [x_min, y_min, x_max, y_max]
        :param width: width of the image
        :param height: height of the image
        :return: the best box id or None if no box is selected
        """
        if len(boxes) == 0:
            return None
        boxes = np.array([box.cpu().numpy() for box in boxes])
        box_centers = np.array(
            [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes]
        )

        # filter boxes whose center is in 20% margin around the image
        margin = 0.2
        margin_width = width * margin
        margin_height = height * margin
        center_boxes_mask = (
            (box_centers[:, 0] > margin_width)
            & (box_centers[:, 0] < width - margin_width)
            & (box_centers[:, 1] > margin_height)
            & (box_centers[:, 1] < height - margin_height)
        )
        if not np.any(center_boxes_mask):
            return None

        # filter boxes whose area is less than 0.5% of the image
        boxes = boxes[center_boxes_mask]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_box_idx = np.argmax(areas)
        if areas[largest_box_idx] < 0.005 * width * height:
            return None

        return np.where(center_boxes_mask)[0][largest_box_idx]

    def check_human_exist(self, video_frames):
        """
        returns True if there is a human in the first frame
        video_frames: list of frames in the original video
        """
        detected_poses = [self.pose_model(frm) for frm in video_frames]
        mx_people, num_frames = 6, len(detected_poses)
        bodies, bodies_score, bodies_subset, faces, faces_score, validity = (
            np.zeros((num_frames, mx_people * 18, 2)),
            np.zeros((num_frames, mx_people, 18)),
            np.ones((num_frames, mx_people, 18)) * -1,
            np.zeros((num_frames, mx_people, 68, 2)),
            np.zeros((num_frames, mx_people, 68)),
            np.zeros((num_frames, mx_people)),
        )
        # filtering poses with low confidence
        for i in range(len(detected_poses)):
            if len(detected_poses[i]["bodies"]["score"]) == 0:
                continue
            valids, valids_score = [], []
            for j in range(len(detected_poses[i]["bodies"]["score"])):
                if (detected_poses[i]["bodies"]["score"][j] < 0.5).sum() < len(
                    detected_poses[i]["bodies"]["score"][0]
                ) / 2:
                    # if less than half of the keypoints are invisible, then the person is valid
                    valids.append(j)
                    valids_score.append(detected_poses[i]["bodies"]["score"][j].mean())
            if len(valids) == 0:
                continue
            valids_score = np.array(valids_score)
            valids = np.array(valids)
            valids = valids[np.argsort(valids_score)[::-1]]
            valids = valids[:mx_people]
            for j, idx in enumerate(valids):
                bodies[i, j * 18 : (j + 1) * 18] = detected_poses[i]["bodies"][
                    "candidate"
                ][idx * 18 : (idx + 1) * 18]
                bodies_score[i, j] = detected_poses[i]["bodies"]["score"][idx]
                bodies_subset[i, j] = detected_poses[i]["bodies"]["subset"][idx]
                faces[i, j] = detected_poses[i]["faces"][idx]
                faces_score[i, j] = detected_poses[i]["faces_score"][idx]
                validity[i, j] = 1
        if np.any(validity):
            return True, (
                bodies,
                bodies_score,
                bodies_subset,
                faces,
                faces_score,
                validity,
            )

    def check_empty_road_exist(self, video_frames):
        """
        returns True if there is no car in the first frame
        video_frames: list of frames in the original video
        """
        pass

    # ========================================== CONTROL GENERATION ==========================================
    def get_controls(self, video_frames):
        """
        returns the controls for the video if the video is compatible with the evaluation mode, otherwise returns None
        video_frames: list of frames in the original video
        """
        is_comp, res = self.is_compatible_video(video_frames)
        if not is_comp:
            return None
        if self.eval_mode == "unconditional":
            return {}
        elif self.eval_mode == "object_manupulation":
            return self.get_manipulation_control(video_frames, res)
        elif self.eval_mode == "skeleton_manipulation":
            return self.get_skeleton_control(video_frames, res)
        elif self.eval_mode == "object_insertion":
            return self.get_insertion_control(video_frames, res)
        elif self.eval_mode == "ego_motion":
            return self.get_ego_motion_control(video_frames)
        else:
            raise ValueError(f"Invalid evaluation mode: {self.eval_mode}")

    def get_manipulation_control(self, video_frames, first_frame_detection_results):
        """
        returns the manipulation control for the video
        video_frames: list of frames in the original video
        first_frame_detection_results: the detection results of the first frame
        return: dict containing the rectangle regions of the object to manipulate with [-2, -2, -2, -2] if the object is not found
        """
        width, height = video_frames[0].shape[1], video_frames[0].shape[0]
        tracking_result, best_box_id = first_frame_detection_results
        tracking_results = [tracking_result]
        for frame in video_frames[1:]:
            tracking_results.append(
                self.tracker_model.track(frame, persist=True, verbose=False)[0]
            )

        boxes = tracking_results[0].boxes
        selected_id = int(boxes.id[boxes.cls == 2][best_box_id])
        tracked_boxes = []
        for result in tracking_results:
            if result.boxes.id is not None and selected_id in result.boxes.id:
                tracked_boxes.append(
                    result.boxes.xyxy[result.boxes.id == selected_id][0]
                )
            else:
                tracked_boxes.append(None)

        all_rectangle_regions = [
            (roi_to_rectangle_region(roi, width, height) if roi is not None else None)
            for roi in tracked_boxes
        ]
        all_rectangle_regions_tensor = torch.tensor(
            [
                (
                    [region.w, region.h, region.x, region.y]
                    if region is not None
                    else [-2, -2, -2, -2]
                )
                for region in all_rectangle_regions
            ]
        )

        at_when = torch.tensor(
            [i for i in self.control_frames if tracked_boxes[i] is not None]
        )
        at_location = all_rectangle_regions_tensor[at_when]
        to_location = at_location
        to_when = at_when
        assert torch.all(
            at_location != -2
        ), "The at_when should have filtered out the -2 values"

        video_frames_tensor = (
            torch.stack(
                [torch.tensor(frame, dtype=torch.float) for frame in video_frames]
            )
            / 255.0
            * 2
            - 1
        )
        video_frames_tensor = video_frames_tensor.cuda()
        # TODO change the dino encoder
        dino_features = self.model.conditioner.embedders[-1].get_demo_input(
            # use get_demo_input2 from demo3 instead, with masks and for highres use 1-4 dino tokens. Masks should be of size out_res
            video_frames_tensor,
            at_location,
            at_when,
            to_location,
            to_when,
            num_total_frames=25,
            num_tokens=random.randint(
                0, 4
            ),  # for low res 8-16 tokens, for high res 1-4 tokens
        )
        return {"DINO_features": dino_features}

    def get_skeleton_control(self, video_frames, skeleton_detection_results):
        """
        returns the skeleton control for the video
        video_frames: list of frames in the original video
        skeleton_detection_results: the detection results of the whole video
        return: dict, the skeleton control
        """
        bodies, bodies_score, bodies_subset, faces, faces_score, validity = (
            skeleton_detection_results
        )
        # drawing the poses
        poses_img, width, height = (
            [],
            video_frames[0].shape[1],
            video_frames[0].shape[0],
        )
        for i in range(len(video_frames)):
            poses_img.append(
                draw_pose(
                    {
                        "bodies": {
                            "candidate": bodies[i],
                            "score": bodies_score[i],
                            "subset": bodies_subset[i],
                        },
                        "faces": faces[i],
                        "faces_score": faces_score[i],
                    },
                    height,
                    width,
                )
            )
        return {"pose_img": poses_img}

    def get_insertion_control(self, video_frames, first_frame_detection_results):
        """
        returns the insertion control for the video
        video_frames: list of frames in the original video
        first_frame_detection_results: the detection results of the first frame
        return: dict, the insertion control
        """
        pass

    def get_ego_motion_control(self, video_frames):
        """
        returns the ego motion control for the video
        video_frames: list of frames in the original video
        return: dict, the ego motion control
        """
        pass

    # ========================================== EVALUATION METRICS ==========================================
    def get_evaluation_metrics(self, gt_video_paths, gen_video_paths):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        if self.eval_mode == "unconditional":
            return self.get_quality_metrics(gt_video_paths, gen_video_paths)
        if self.eval_mode == "object_manipulation":
            return self.get_evaluation_metric_object_manipulation(
                gt_video_paths, gen_video_paths
            )
        elif self.eval_mode == "skeleton_manipulation":
            return self.get_evaluation_metric_skeleton_manipulation(
                gt_video_paths, gen_video_paths
            )
        elif self.eval_mode == "object_insertion":
            return self.get_evaluation_metric_object_insertion(
                gt_video_paths, gen_video_paths
            )
        elif self.eval_mode == "ego_motion":
            return self.get_evaluation_metric_ego_motion(
                gt_video_paths, gen_video_paths
            )
        else:
            raise ValueError(f"Invalid evaluation mode: {self.eval_mode}")

    def get_quality_metrics(self, gt_video_paths, gen_video_paths):
        """
        returns the quality metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        assert len(gt_video_paths) == len(
            gen_video_paths
        ), "Number of real and generated videos must be the same"

        def extract_embeddings(
            video_paths, i3d, batch_size=4, num_workers=128, device="cuda"
        ):
            # Create dataset and dataloader
            dataset = FVDVideoDataset(video_paths)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
            )
            embeddings = []
            for videos in tqdm(dataloader, desc="Extract Embeddings"):
                videos = videos.numpy()  # Convert to numpy array
                embeddings.append(
                    get_fvd_logits(
                        videos, i3d=i3d, device=device, batch_size=batch_size
                    )
                )
            embeddings = torch.cat(embeddings, dim=0)
            return embeddings

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        i3d = load_fvd_model(device)

        real_embeddings = extract_embeddings(gt_video_paths, i3d, device=device)
        fake_embeddings = extract_embeddings(gen_video_paths, i3d, device=device)

        print("calculating fvd ...")
        fvd = frechet_distance(fake_embeddings, real_embeddings)
        fvd = fvd.cpu().numpy()  # np float32
        print(f"FVD Score: {fvd}")

        fid_total = FrechetInceptionDistance().cuda()
        real_dataloader = DataLoader(
            FIDVideoDataset(gt_video_paths), batch_size=4, num_workers=64, shuffle=False
        )
        fake_dataloader = DataLoader(
            FIDVideoDataset(gen_video_paths),
            batch_size=4,
            num_workers=64,
            shuffle=False,
        )
        for real_batch, fake_batch in tqdm(
            zip(real_dataloader, fake_dataloader),
            desc="Calculating FID",
            total=len(real_dataloader),
        ):
            B, T, H, W, C = real_batch.shape
            fid_total.update(
                real_batch.reshape(B * T, H, W, C)
                .permute(0, 3, 1, 2)
                .type(torch.uint8)
                .cuda(),
                real=True,
            )
            fid_total.update(
                fake_batch.reshape(B * T, H, W, C)
                .permute(0, 3, 1, 2)
                .type(torch.uint8)
                .cuda(),
                real=False,
            )

        fid = fid_total.compute().item()
        print(f"FID Score: {fid}")
        return {"FVD": fvd, "FID": fid}

    def get_evaluation_metric_object_manipulation(
        self, gt_video_paths, gen_video_paths, visualize=False
    ):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        box_misplacement = {}
        for i in self.control_frames:
            box_misplacement[i] = []
        for video_real, video_fake in tqdm(
            zip(gt_video_paths, gen_video_paths), desc="Evaluating object manipulation"
        ):
            if not video_real.endswith(".mp4"):
                continue
            # load the GT video and track the best box
            tracker_model = YOLO("yolo11n.pt")
            cap_gt = cv2.VideoCapture(video_real)
            gt_tracking_results = []
            first_frame_gt = None
            while cap_gt.isOpened():
                ret_gt, frame_gt = cap_gt.read()
                if not ret_gt:
                    break
                if first_frame_gt is None:
                    first_frame_gt = frame_gt
                gt_tracking_results.append(
                    tracker_model.track(frame_gt, persist=True, verbose=False)[0]
                )
            cap_gt.release()

            # if no boxes are detected in the first frame, skip the video (this should not happen)
            boxes = gt_tracking_results[0].boxes
            if boxes.id is None:
                continue
            boxes = boxes.xyxy[boxes.cls == 2]  # only track cars
            best_box_id = self.select_best_box(
                boxes, first_frame_gt.shape[1], first_frame_gt.shape[0]
            )
            if best_box_id is None:
                continue

            # load the generated video and track the same box
            tracker_model = YOLO("yolo11n.pt")
            cap_val = cv2.VideoCapture(video_fake)
            val_tracking_results = []
            while cap_val.isOpened():
                ret_val, frame_val = cap_val.read()
                if not ret_val:
                    break
                if first_frame_gt is not None:
                    frame_val = first_frame_gt
                    first_frame_gt = None
                val_tracking_results.append(
                    tracker_model.track(frame_val, persist=True, verbose=False)[0]
                )
            cap_val.release()

            if visualize:
                writer = imageio.get_writer(
                    video_real.replace(".mp4", "") + "_gt_track.mp4", fps=10
                )
                for i in range(len(gt_tracking_results)):
                    annotated_frame = gt_tracking_results[i].plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(annotated_frame)
                writer.close()
                writer = imageio.get_writer(
                    video_fake.replace(".mp4", "") + "_gen_track.mp4", fps=10
                )
                for i in range(len(val_tracking_results)):
                    annotated_frame = val_tracking_results[i].plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(annotated_frame)
                writer.close()

            # get the tracked boxes for the selected box
            selected_id = int(
                gt_tracking_results[0].boxes.id[gt_tracking_results[0].boxes.cls == 2][
                    best_box_id
                ]
            )
            gt_tracked_boxes = []
            for result in gt_tracking_results:
                if result.boxes.id is not None and selected_id in result.boxes.id:
                    gt_tracked_boxes.append(
                        result.boxes.xyxy[result.boxes.id == selected_id][0]
                    )
                else:
                    gt_tracked_boxes.append(None)
            eval_tracked_boxes = []
            for result in val_tracking_results:
                if result.boxes.id is not None and selected_id in result.boxes.id:
                    eval_tracked_boxes.append(
                        result.boxes.xyxy[result.boxes.id == selected_id][0]
                    )
                else:
                    eval_tracked_boxes.append(None)

            # compare the tracked boxes
            for i in self.control_frames:
                if (
                    gt_tracked_boxes[i] is not None
                    and eval_tracked_boxes[i] is not None
                ):
                    gt_box_center = np.array(
                        [
                            (gt_tracked_boxes[i][0] + gt_tracked_boxes[i][2]) / 2,
                            (gt_tracked_boxes[i][1] + gt_tracked_boxes[i][3]) / 2,
                        ]
                    )
                    eval_box_center = np.array(
                        [
                            (eval_tracked_boxes[i][0] + eval_tracked_boxes[i][2]) / 2,
                            (eval_tracked_boxes[i][1] + eval_tracked_boxes[i][3]) / 2,
                        ]
                    )
                    box_misplacement[i].append(
                        np.linalg.norm(gt_box_center - eval_box_center)
                    )
        print(f"Average box misplacement: ", end="")
        for i in self.control_frames:
            print(f"{np.mean(box_misplacement[i]):.2f} ", end="")
        print(
            "Average: ",
            np.mean([np.mean(box_misplacement[i]) for i in self.control_frames[1:]]),
        )

    def get_evaluation_metric_skeleton_manipulation(
        self, gt_video_paths, gen_video_paths
    ):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        pass

    def save_temp_annotation_files(
        self, real_poses, fake_poses, filename_prefix="temp"
    ):
        """
        save the real and fake poses to temporary files for visualization
        real_poses: list of real poses
        fake_poses: list of fake poses
        filename_prefix: str, the prefix for the temporary files
        """
        images = []
        gt_annotations = []
        dt_annotations = []
        image_id = 0
        annotation_id = 0

        # Define categories (COCO requires this to be predefined, here we assume one category: 'person')
        categories = [
            {
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": [
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                    "neck",
                ],
                "skeleton": [],
            }
        ]

        # Process each frame for real and fake poses
        for real_frame, fake_frame in zip(real_poses, fake_poses):
            n_real, n_fake = real_frame.shape[0], fake_frame.shape[0]

            # Define the image (frame)
            images.append(
                {
                    "id": image_id,
                    "width": 1920,  # assuming a default width
                    "height": 1080,  # assuming a default height
                    "file_name": f"{filename_prefix}_{image_id}.jpg",
                }
            )

            # Annotations for ground truth
            for i in range(n_real):
                keypoints = real_frame[i].reshape(-1).tolist()
                gt_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "keypoints": keypoints,
                        "num_keypoints": int(sum(k > 0 for k in keypoints[2::3])),
                        "bbox": [
                            min(keypoints[0::3]),
                            min(keypoints[1::3]),
                            max(keypoints[0::3]) - min(keypoints[0::3]),
                            max(keypoints[1::3]) - min(keypoints[1::3]),
                        ],  # simple bbox
                        "iscrowd": 0,
                        "area": 0,  # could be calculated based on the bbox
                    }
                )
                annotation_id += 1

            # Annotations for detections
            for i in range(n_fake):
                keypoints = fake_frame[i].reshape(-1).tolist()
                dt_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "keypoints": keypoints,
                        "num_keypoints": int(sum(k > 0 for k in keypoints[2::3])),
                        "bbox": [
                            min(keypoints[0::3]),
                            min(keypoints[1::3]),
                            max(keypoints[0::3]) - min(keypoints[0::3]),
                            max(keypoints[1::3]) - min(keypoints[1::3]),
                        ],  # simple bbox
                        "score": 1.0,  # example fixed score for detections
                        "iscrowd": 0,
                        "area": 0,  # could be calculated based on the bbox
                    }
                )
                annotation_id += 1

            image_id += 1

        # Write to files
        with open(f"{filename_prefix}_gt_annotations.json", "w") as f:
            json.dump(
                {
                    "images": images,
                    "annotations": gt_annotations,
                    "categories": categories,
                },
                f,
            )
        with open(f"{filename_prefix}_dt_annotations.json", "w") as f:
            json.dump(
                {
                    "images": images,
                    "annotations": dt_annotations,
                    "categories": categories,
                },
                f,
            )

    def get_evaluation_metric_object_insertion(self, gt_video_paths, gen_video_paths):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        pass

    def get_evaluation_metric_ego_motion(self, gt_video_paths, gen_video_paths):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        pass
