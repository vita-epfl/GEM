import os
import torch
import numpy as np
from ultralytics import YOLO
from gem.utils.roi_utils import roi_to_rectangle_region
from tqdm import tqdm
import cv2
import imageio
import json
from torchmetrics.image.fid import FrechetInceptionDistance
from fvd_utils import load_fvd_model, get_fvd_logits, frechet_distance
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dwpose.util import draw_pose
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import math



class FVDVideoDataset(Dataset):
    def __init__(self, video_paths, num_frames=25):
        self.video_paths = video_paths
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video = self.process_video_to_numpy(video_path)
        return video

    def process_video_to_numpy(self, video_path, target_size=(224, 224), intermediate_size=(448, 256)):
        processed_frames = []
        frames = np.load(video_path)[:self.num_frames]
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
        resized_image_tensor = torch.nn.functional.interpolate(processed_frames, size=target_size, mode="bilinear", align_corners=False)

        # Convert tensor back to numpy array
        return resized_image_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(int)   # BHWC format


class FIDVideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        return np.load(video_path)[1:]  # skip the first frame


class EvaluationHelper:
    # ========================================== INITIALIZATION ==========================================
    def __init__(self, eval_mode, model, after_generation=False):
        """
        eval_mode: str, evaluation mode, e.g. 'object_manipulation'
        options: 'unconditional', 'object_manipulation', 'skeleton_manipulation', 'object_insertion', 'ego_motion'
        """
        self.eval_mode = eval_mode
        if eval_mode == 'object_manipulation':
            self.control_frames = [5, 15, 20]
        elif eval_mode == 'skeleton_manipulation':
            self.control_frames = list(range(25))
        elif eval_mode == 'ego_motion':
            self.control_frames = [4, 9, 14, 19]
        self.after_generation = after_generation
        self.load_models()
        self.model = model

    def load_models(self):
        if self.eval_mode == "object_manipulation":
            self.tracker_model = YOLO("ckpts/yolo11n.pt")
        elif self.eval_mode == "skeleton_manipulation":
            if self.after_generation:  # get the poses for evaluation
                from dwpose.wholebody import Wholebody
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                self.pose_model = Wholebody(model_det="ckpts/DWPose/yolox_l.onnx", model_pose="ckpts/DWPose/dw-ll_ucoco_384.onnx", device=device)
            else:  # get the poses for control generation
                from dwpose.dwpose_detector import dwpose_detector as dwprocessor
                self.pose_model = dwprocessor
        elif self.eval_mode == "ego_motion":
            from pseudo_labeling_pipeline.trajectory_inference.extraction.generate_trajectory import TrajectoryExtractor
            self.trajectory_extractor = TrajectoryExtractor(weights_dir="ckpts")

    # ========================================== COMPATIBILITY CHECK ==========================================
    def is_compatible_video(self, video_frames):
        """
        returns True if the video is compatible with the evaluation mode, e.g. if there is a car in the first frame
        when trying to evaluate car moving control
        video_frames: list of frames in the original video
        return (bool, optional results)
        """
        if self.eval_mode in ['unconditional']:
            return True, None
        elif self.eval_mode == 'object_manipulation':
            return self.check_car_exist(video_frames)
        elif self.eval_mode == 'skeleton_manipulation':
            return self.check_human_exist(video_frames)
        elif self.eval_mode == 'ego_motion':
            return self.get_ego_motion(video_frames)
        else:
            raise ValueError(f'Invalid evaluation mode: {self.eval_mode}')

    def check_car_exist(self, video_frames):
        """
        returns True if there is a car in the first frame
        video_frames: list of frames in the original video of shape (T, H, W, C)
        returns: (bool, tracking_results)
        """
        tracking_results = self.tracker_model.track(video_frames[0], persist=True, verbose=False)[0]
        boxes = tracking_results.boxes
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
        box_centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes])

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
        if self.after_generation:
            detected_poses = [self.pose_model(frm, do_postprocess=False) for frm in video_frames]
        else:
            detected_poses = [self.pose_model(frm) for frm in video_frames]
        num_valids = 0
        # filtering poses with low confidence
        for i in range(len(detected_poses)):
            if self.after_generation:
                scores = detected_poses[i][1][:, :17]
            else:
                scores = detected_poses[i]['bodies']['score']
            if len(scores) == 0:
                continue
            for j in range(len(scores)):
                if (scores[j] < 0.5).sum() < len(scores[0]) / 2:
                    # if less than half of the keypoints are invisible, then the person is valid
                    num_valids += 1
        if num_valids > 5:
            return True, detected_poses
        return False, None

    def normalize_trajectory(self, trajectory):
        trajectory = trajectory - trajectory[0]
        direction_vector = trajectory[1]
        angle_to_y_axis = np.arctan2(direction_vector[0], direction_vector[1])
        rotation_matrix = np.array([
            [np.cos(angle_to_y_axis), -np.sin(angle_to_y_axis)],
            [np.sin(angle_to_y_axis), np.cos(angle_to_y_axis)]
        ])
        normalized_points = trajectory.dot(rotation_matrix.T)
        return normalized_points

    def get_ego_motion(self, video_frames):
        """
        returns True and the trajectory of the car if droid slam is able to extract the trajectory
        video_frames: list of frames in the original video
        """
        ret = self.trajectory_extractor(np.array(video_frames)[..., ::-1])
        if ret is None:
            return False, None
        trajectory_2d, all_traj = ret
        trajectory_2d = gaussian_filter1d(trajectory_2d, sigma=10, axis=0, mode="nearest")
        trajectory_2d = self.normalize_trajectory(trajectory_2d)
        return True, trajectory_2d



    # ========================================== CONTROL GENERATION ==========================================
    def get_controls(self, video_frames):
        """
        returns the controls for the video if the video is compatible with the evaluation mode, otherwise returns None
        video_frames: list of frames in the original video
        """
        is_comp, res = self.is_compatible_video(video_frames)
        if not is_comp:
            return None
        if self.eval_mode == 'unconditional':
            return {}
        elif self.eval_mode == 'object_manipulation':
            return self.get_manipulation_control(video_frames, res)
        elif self.eval_mode == 'skeleton_manipulation':
            return self.get_skeleton_control(video_frames, res)
        elif self.eval_mode == 'ego_motion':
            return self.get_ego_motion_control(res)
        else:
            raise ValueError(f'Invalid evaluation mode: {self.eval_mode}')

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
            tracking_results.append(self.tracker_model.track(frame, persist=True, verbose=False)[0])

        boxes = tracking_results[0].boxes
        selected_id = int(boxes.id[boxes.cls == 2][best_box_id])
        tracked_boxes = []
        for result in tracking_results:
            if result.boxes.id is not None and selected_id in result.boxes.id:
                tracked_boxes.append(result.boxes.xyxy[result.boxes.id == selected_id][0])
            else:
                tracked_boxes.append(None)

        all_rectangle_regions_tensor = torch.tensor([(roi_to_rectangle_region(roi, width, height) if roi is not None else [-2, -2, -2, -2]) for roi in tracked_boxes])

        at_when = torch.tensor([i for i in self.control_frames if tracked_boxes[i] is not None])
        if len(at_when) == 0:
            return None
        at_location = all_rectangle_regions_tensor[at_when]
        to_when = torch.tensor([i for i in self.control_frames if tracked_boxes[i] is not None])
        to_location = all_rectangle_regions_tensor[to_when]
        assert torch.all(at_location != -2), "The at_when should have filtered out the -2 values"

        video_frames_tensor = torch.stack([torch.tensor(frame[..., ::-1].copy(), dtype=torch.float) for frame in video_frames]) / 255.0 * 2 - 1
        video_frames_tensor = video_frames_tensor.permute(0, 3, 1, 2).to(self.model.device)

        encoder = None
        for embedder in self.model.conditioner.embedders:
            if hasattr(embedder, "dino_channels"):
                encoder = embedder
        assert encoder is not None, "DINO embedder not found"

        dino_features = encoder.get_demo_input(
            video_frames_tensor,
            at_location,
            at_when,
            to_location,
            to_when,
            # ids=torch.tensor([1 for _ in range(len(at_location))]).to(self.model.device),
            num_total_frames=25,
            num_tokens=3,
        )
        return {"fd_crossattn": dino_features}

    def get_skeleton_control(self, video_frames, skeleton_detection_results):
        """
        returns the skeleton control for the video
        video_frames: list of frames in the original video
        skeleton_detection_results: the detection results of the whole video
        return: dict, the skeleton control
        """
        # drawing the poses
        poses_img, width, height = [], video_frames[0].shape[1], video_frames[0].shape[0]
        for i in range(len(video_frames)):
            poses_img.append(draw_pose(skeleton_detection_results[i], height, width))
        poses_img = np.array(poses_img)
        poses_img = torch.tensor(poses_img).to(self.model.device)

        encoder = None
        for embedder in self.model.conditioner.embedders:
            if hasattr(embedder, "posenet"):
                encoder = embedder
        assert encoder is not None, "Posenet embedder not found"
        return {"skeletons_context": encoder(poses_img)}

    def get_ego_motion_control(self, trajectory):
        """
        returns the ego motion control for the video
        trajectory: trajectory of all frames in the original video
        return: dict, the ego motion control which is the 2d position of the car in timesteps 5, 10, 15, 20
        """
        return {"trajectory": torch.tensor(trajectory[::5][1:].reshape(-1)).float().to(self.model.device)}

    # ========================================== EVALUATION METRICS ==========================================
    def get_evaluation_metrics(self, gt_video_paths, gen_video_paths):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        if self.eval_mode == 'unconditional':
            return self.get_quality_metrics(gt_video_paths, gen_video_paths)
        if self.eval_mode == 'object_manipulation':
            return self.get_evaluation_metric_object_manipulation(gt_video_paths, gen_video_paths)
        elif self.eval_mode == 'skeleton_manipulation':
            return self.get_evaluation_metric_skeleton_manipulation(gt_video_paths, gen_video_paths)
        elif self.eval_mode == 'ego_motion':
            return self.get_evaluation_metric_ego_motion(gt_video_paths, gen_video_paths)
        else:
            raise ValueError(f'Invalid evaluation mode: {self.eval_mode}')

    def get_quality_metrics(self, gt_video_paths, gen_video_paths, do_long=False):
        """
        returns the quality metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        assert len(gt_video_paths) == len(gen_video_paths), "Number of real and generated videos must be the same"

        def extract_embeddings(video_paths, i3d, batch_size=4, num_workers=128, device="cuda", num_frames=25):
            # Create dataset and dataloader
            dataset = FVDVideoDataset(video_paths, num_frames=num_frames)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            embeddings = []
            for videos in tqdm(dataloader, desc="Extract Embeddings"):
                videos = videos.numpy()  # Convert to numpy array
                embeddings.append(get_fvd_logits(videos, i3d=i3d, device=device, batch_size=batch_size))
            embeddings = torch.cat(embeddings, dim=0)
            return embeddings

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        i3d = load_fvd_model(device)

        if do_long:
            for video_length in range(25, 151, 25):
                real_embeddings = extract_embeddings(gt_video_paths, i3d, device=device, num_frames=video_length)
                fake_embeddings = extract_embeddings(gen_video_paths, i3d, device=device, num_frames=video_length)

                print('calculating fvd ...')
                fvd = frechet_distance(fake_embeddings, real_embeddings)
                fvd = fvd.cpu().numpy()  # np float32
                print(f"FVD Score at {video_length} frames: {fvd}")
        else:
            real_embeddings = extract_embeddings(gt_video_paths, i3d, device=device)
            fake_embeddings = extract_embeddings(gen_video_paths, i3d, device=device)

            print('calculating fvd ...')
            fvd = frechet_distance(fake_embeddings, real_embeddings)
            fvd = fvd.cpu().numpy()  # np float32
            print(f"FVD Score: {fvd}")

        fid_total = FrechetInceptionDistance().cuda()

        if do_long:
            real_dataloader = DataLoader(FIDVideoDataset(gt_video_paths), batch_size=4, num_workers=64, shuffle=False)
            fake_dataloader = DataLoader(FIDVideoDataset(gen_video_paths), batch_size=4, num_workers=64, shuffle=False)
            # for video_length in range(25, 151, 25):

            for real_batch, fake_batch in tqdm(zip(real_dataloader, fake_dataloader), desc="Calculating FID", total=len(real_dataloader)):
                # real_batch = real_batch[:, video_length-25:video_length]
                # fake_batch = fake_batch[:, video_length-25:video_length]
                real_batch = real_batch[:, :149]
                fake_batch = fake_batch[:, :149]
                print(real_batch.shape, fake_batch.shape)
                B, T, H, W, C = real_batch.shape
                fid_total.update(real_batch.reshape(B*T, H, W, C).permute(0, 3, 1, 2).type(torch.uint8).cuda(), real=True)
                fid_total.update(fake_batch.reshape(B*T, H, W, C).permute(0, 3, 1, 2).type(torch.uint8).cuda(), real=False)
            fid = fid_total.compute().item()
            print(f"FID Score at {150} frames: {fid}")
        else:
            real_dataloader = DataLoader(FIDVideoDataset(gt_video_paths), batch_size=4, num_workers=64, shuffle=False)
            fake_dataloader = DataLoader(FIDVideoDataset(gen_video_paths), batch_size=4, num_workers=64, shuffle=False)
            for real_batch, fake_batch in tqdm(zip(real_dataloader, fake_dataloader), desc="Calculating FID", total=len(real_dataloader)):
                B, T, H, W, C = real_batch.shape
                fid_total.update(real_batch.reshape(B * T, H, W, C).permute(0, 3, 1, 2).type(torch.uint8).cuda(), real=True)
                fid_total.update(fake_batch.reshape(B * T, H, W, C).permute(0, 3, 1, 2).type(torch.uint8).cuda(), real=False)
            fid = fid_total.compute().item()
            print(f"FID Score: {fid}")
        return {"FVD": fvd, "FID": fid}

    def get_evaluation_metric_object_manipulation(self, gt_video_paths, gen_video_paths, visualize=True, n_samples=500):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        box_misplacement = {}
        for i in self.control_frames:
            box_misplacement[i] = []
        num_evaluated, avg_metric = 0, 0
        pbar = tqdm(desc=f"Evaluating object manipulation {avg_metric}", total=min(len(gt_video_paths), len(gen_video_paths), n_samples))
        for video_real, video_fake in zip(gt_video_paths, gen_video_paths):
            if not video_real.endswith(".npy"):
                continue
            # load the GT video and track the best box
            tracker_model = YOLO("ckpts/yolo11n.pt")
            gt_tracking_results = []
            try:
                frames_gt = np.load(video_real)[..., ::-1]
            except Exception as e:
                print(f"Error loading video {video_real}: {e}")
                continue
            best_box_id = None
            for i, frame_gt in enumerate(frames_gt):
                gt_tracking_results.append(tracker_model.track(frame_gt, persist=True, verbose=False)[0])
                if i==0:
                    # if no boxes are detected in the first frame, skip the video (this should not happen)
                    boxes = gt_tracking_results[0].boxes
                    if boxes.id is None:
                        continue
                    boxes = boxes.xyxy[boxes.cls == 2]  # only track cars
                    best_box_id = self.select_best_box(boxes, frame_gt.shape[1], frame_gt.shape[0])
                    if best_box_id is None:
                        continue
            if best_box_id is None:
                continue

            # load the generated video and track the same box
            tracker_model = YOLO("ckpts/yolo11n.pt")
            frames_gen = np.load(video_fake)[..., ::-1]
            frames_gen[0] = frames_gt[0]  # make sure the first frame is the same for tracking
            val_tracking_results = []
            for frame_val in frames_gen:
                val_tracking_results.append(tracker_model.track(frame_val, persist=True, verbose=False)[0])

            # get the tracked boxes for the selected box
            selected_id = int(gt_tracking_results[0].boxes.id[gt_tracking_results[0].boxes.cls == 2][best_box_id])
            gt_tracked_boxes = []
            for result in gt_tracking_results:
                if result.boxes.id is not None and selected_id in result.boxes.id:
                    gt_tracked_boxes.append(result.boxes.xyxy[result.boxes.id == selected_id][0])
                else:
                    gt_tracked_boxes.append(None)
            eval_tracked_boxes = []
            for result in val_tracking_results:
                if result.boxes.id is not None and selected_id in result.boxes.id:
                    eval_tracked_boxes.append(result.boxes.xyxy[result.boxes.id == selected_id][0])
                else:
                    eval_tracked_boxes.append(None)

            if visualize:
                # Visualizes the tracked boxes in both GT and Gen videos
                video_name = video_real.split("/")[-1][:-4]
                os.makedirs(os.path.dirname(video_real.replace(f"real/npy/{video_name}.npy", f"object_manipulation_compare/{video_name}.mp4")), exist_ok=True)
                writer = imageio.get_writer(video_real.replace(f"real/npy/{video_name}.npy", f"object_manipulation_compare/{video_name}.mp4"), fps=10)
                for i in range(len(frames_gt)):
                    gt_frame = frames_gt[i]
                    eval_frame = frames_gen[i]
                    if gt_tracked_boxes[i] is not None:
                        x1, y1, x2, y2 = gt_tracked_boxes[i]
                        gt_frame = cv2.rectangle(np.array(gt_frame, np.uint8), (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        if eval_tracked_boxes[i] is not None:
                            x1, y1, x2, y2 = eval_tracked_boxes[i]
                            gt_frame = cv2.rectangle(np.array(gt_frame, np.uint8), (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    if eval_tracked_boxes[i] is not None:
                        x1, y1, x2, y2 = eval_tracked_boxes[i]
                        eval_frame = cv2.rectangle(np.array(eval_frame, np.uint8), (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        if gt_tracked_boxes[i] is not None:
                            x1, y1, x2, y2 = gt_tracked_boxes[i]
                            eval_frame = cv2.rectangle(np.array(eval_frame, np.uint8), (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    writer.append_data(np.concatenate([gt_frame, eval_frame], axis=0)[..., ::-1])
                writer.close()
                if len(os.listdir(os.path.dirname(video_real.replace(f"real/npy/{video_name}.npy", f"object_manipulation_compare/{video_name}.mp4")))) == 20:
                    visualize = False
            # compare the tracked boxes
            for i in self.control_frames:
                if gt_tracked_boxes[i] is not None and eval_tracked_boxes[i] is not None:
                    gt_box_center = np.array([(gt_tracked_boxes[i][0] + gt_tracked_boxes[i][2]) / 2,
                                              (gt_tracked_boxes[i][1] + gt_tracked_boxes[i][3]) / 2,])
                    eval_box_center = np.array([(eval_tracked_boxes[i][0] + eval_tracked_boxes[i][2]) / 2,
                                                (eval_tracked_boxes[i][1] + eval_tracked_boxes[i][3]) / 2,])
                    box_misplacement[i].append(np.linalg.norm(gt_box_center - eval_box_center))
            num_evaluated += 1
            pbar.update(1)
            avg_metric = np.mean([np.mean(box_misplacement[i]) for i in self.control_frames])
            pbar.set_description(f"Evaluating object manipulation {avg_metric:.2f}")
            if num_evaluated >= n_samples:
                break
        print(f"Average box misplacement: ", end="")
        for i in self.control_frames:
            print(f"{np.mean(box_misplacement[i]):.2f} ", end="")
        print("Average: ", np.mean([np.mean(box_misplacement[i]) for i in self.control_frames]),)
        return {"box_misplacement": np.mean([np.mean(box_misplacement[i]) for i in self.control_frames])}

    def get_evaluation_metric_skeleton_manipulation(self, gt_video_paths, gen_video_paths, n_samples=500):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        assert self.after_generation, "Skeleton manipulation evaluation is only available in after generation mode"
        pbar = tqdm(desc=f"Evaluating object manipulation {0}", total=min(len(gt_video_paths), len(gen_video_paths), n_samples))
        real_poses, fake_poses = [], []
        for video_real, video_fake in zip(gt_video_paths, gen_video_paths):
            if not video_real.endswith(".npy"):
                continue
            assert video_real.split("/")[-1] == video_fake.split("/")[-1], "The real and fake videos should have the same name"
            # get the ground truth poses
            try:
                video_frames = np.load(video_real)
            except:
                continue
            has_people, gt_detected_poses = self.check_human_exist(video_frames)
            if not has_people:  # if there are no people in the video, skip, this should not happen unless for unconditional video evaluation
                continue
            # get the generated poses
            try:
                video_frames = np.load(video_fake)
            except:
                continue
            gen_detected_poses = [self.pose_model(frm, do_postprocess=False) for frm in video_frames]
            # compare the poses
            real_poses += [np.concatenate([gt_detected_poses[i][0][:, :17], gt_detected_poses[i][1][:, :17, None]], axis=-1) for i in range(len(gt_detected_poses))]
            fake_poses += [np.concatenate([gen_detected_poses[i][0][:, :17], gen_detected_poses[i][1][:, :17, None]], axis=-1) for i in range(len(gen_detected_poses))]

            pbar.update(1)
            if pbar.n >= n_samples:
                break

        self.save_temp_annotation_files(real_poses, fake_poses, filename_prefix="temp")
        coco = COCO(f"temp_gt_annotations.json")
        coco_dt = coco.loadRes(f"temp_dt_annotations.json")
        coco_eval = COCOeval(coco, coco_dt, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def save_temp_annotation_files(self, real_poses, fake_poses, filename_prefix="temp"):
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
        categories = [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person',
            'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                          'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'neck'],
            'skeleton': []
        }]

        # Process each frame for real and fake poses
        for real_frame, fake_frame in zip(real_poses, fake_poses):
            n_real, n_fake = real_frame.shape[0], fake_frame.shape[0]

            # Define the image (frame)
            images.append({
                'id': image_id,
                'width': 1920,  # assuming a default width
                'height': 1080,  # assuming a default height
                'file_name': f'{filename_prefix}_{image_id}.jpg'
            })

            # Annotations for ground truth
            for i in range(n_real):
                real_frame[i][:, 2][real_frame[i][:, 2] < 0.5] = 0  # filter out low confidence keypoints
                real_frame[i][:, 2][real_frame[i][:, 2] >= 0.5] = 2  # set the confidence to 1
                num_keypoints = real_frame[i][:, 2].sum() / 2
                keypoints = real_frame[i].reshape(-1).tolist()
                gt_annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': keypoints,
                    'num_keypoints': int(num_keypoints),
                    'bbox': [min(keypoints[0::3]), min(keypoints[1::3]),
                             max(keypoints[0::3]) - min(keypoints[0::3]),
                             max(keypoints[1::3]) - min(keypoints[1::3])],  # simple bbox
                    'iscrowd': 0,
                    'area': (max(keypoints[0::3]) - min(keypoints[0::3])) * (max(keypoints[1::3]) - min(keypoints[1::3]))
                })
                annotation_id += 1

            # Annotations for detections
            for i in range(n_fake):
                fake_frame[i][:, 2][fake_frame[i][:, 2] < 0.5] = 0  # filter out low confidence keypoints
                fake_frame[i][:, 2][fake_frame[i][:, 2] >= 0.5] = 2  # set the confidence to 1
                num_keypoints = fake_frame[i][:, 2].sum() / 2
                keypoints = fake_frame[i].reshape(-1).tolist()
                dt_annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': keypoints,
                    'num_keypoints': int(num_keypoints),
                    'bbox': [min(keypoints[0::3]), min(keypoints[1::3]),
                             max(keypoints[0::3]) - min(keypoints[0::3]),
                             max(keypoints[1::3]) - min(keypoints[1::3])],  # simple bbox
                    'score': 1.0,  # example fixed score for detections
                    'iscrowd': 0,
                    'area': (max(keypoints[0::3]) - min(keypoints[0::3])) * (max(keypoints[1::3]) - min(keypoints[1::3]))
                })
                annotation_id += 1

            image_id += 1

        # Write to files
        with open(f'{filename_prefix}_gt_annotations.json', 'w') as f:
            json.dump({'images': images, 'annotations': gt_annotations, 'categories': categories}, f)
        with open(f'{filename_prefix}_dt_annotations.json', 'w') as f:
            json.dump(dt_annotations, f)

    def get_evaluation_metric_ego_motion(self, gt_video_paths, gen_video_paths, n_samples=500, scale_compensate=False):
        """
        returns the evaluation metrics for the generated videos
        gt_video_paths: list of paths to the ground truth videos
        gen_video_paths: list of paths to the generated videos
        """
        ades = []
        pbar = tqdm(desc=f"Calculating ADE {0}", total=min(len(gt_video_paths), n_samples))
        for video_real, video_fake in zip(gt_video_paths, gen_video_paths):
            if not video_real.endswith(".npy"):
                continue
            assert video_real.split("/")[-1] == video_fake.split("/")[-1], "The real and fake videos should have the same name"
            # get the ground truth trajectory
            # if "NUSCENES" in video_real and False:
            #     with open("annotations/nuScenes_val.json", "r") as f:
            #         nuScenes_val = json.load(f)
            #     video_name = video_real.split("/")[-1].replace(".npy", "")
            #     video_id = int(video_name.split("_")[-1])
            #     scene = nuScenes_val[video_id]
            #     traj = np.array(scene["traj"][2:]).reshape(-1, 2)
            #     traj = np.concatenate([np.zeros((1, 2)), traj], axis=0)
            #     # interpolate the trajectory from 4 to 25 frames
            #     subset_indices = np.arange(0, 25, 5)
            #     full_indices = np.arange(25)
            #     traj_full = np.zeros((25, 2))
            #     for i in range(2):
            #         f = interp1d(subset_indices, traj[:, i], kind="linear", fill_value='extrapolate')
            #         traj_full[:, i] = f(full_indices)
            #     ret_gt = (traj_full, None)
            # else:
            frames_gt = np.load(video_real)
            ret_gt = self.trajectory_extractor(frames_gt)
            if ret_gt is None:
                continue

            frames_gen = np.load(video_fake)
            ret_gen = self.trajectory_extractor(frames_gen)

            gt_traj, _ = ret_gt
            if ret_gen is None:
                gen_traj = np.zeros_like(gt_traj)
            else:
                gen_traj, _ = ret_gen
                # compensate for scale of the generated trajectory for nuscenes
                if "NUSCENES" in video_real and scale_compensate:
                    numerator = np.sum(np.sum(gt_traj * gen_traj, axis=1))  # dot product for each timestep
                    denominator = np.sum(np.sum(gen_traj ** 2, axis=1))  # norm squared for each timestep
                    # Compute the scaling factor
                    s = numerator / denominator
                    gen_traj *= s

            # calculate the average displacement error
            ade = np.mean(np.linalg.norm(gt_traj - gen_traj, axis=1))
            ades.append(ade)
            pbar.update(1)
            pbar.set_description(f"Calculating ADE {np.mean(ades):.2f}")
            if len(ades) >= n_samples:
                break
        print(f"Average displacement error: {np.mean(ades):.2f}")
        return {"ADE": np.mean(ades)}
