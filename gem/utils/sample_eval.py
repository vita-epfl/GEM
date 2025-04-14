import argparse

# from torcheval.metrics import FrechetInceptionDistance
import csv
import json
import os
import random
from typing import Tuple

import cv2
import imageio
import init_proj_path
import torch.nn.functional as Func

# from cdfvd import fvd
# from ultralytics import YOLO
from eval_utils import EvaluationHelper
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from sample_utils import *
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

from gem.flow.flow.flow_network import FlowNetwork
from gem.modules.encoders.dino_v2_features_id import DinoEncoder
from gem.utils.roi_utils import (
    ROI_CLASS_CAR,
    create_controls_overlay,
    create_roi_model,
    detect_roi_objects,
    rectangle_region_to_roi,
    roi_to_rectangle_region,
)

VERSION2SPECS = {
    "gem": {
        # "config": "/store/swissai/a03/ckpts/s2.yaml",
        # "ckpt": "/store/swissai/a03/ckpts/stage2.safetensors",
        # "config": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/checkpoints/2024-10-19T19-31-58_example-fs0/configs/2024-10-19T19-31-58-project.yaml",
        "config": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/checkpoints/stage0_n128/configs/2024-10-18T05-41-16-project.yaml",
        # "config": "configs/example/best_so_far.yaml",
        # "ckpt": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/checkpoints/high_res_last.safetensors",
        "ckpt": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/checkpoints/stage0_n128.safetensors",
    }
    # "gem": {"config": "configs/inference/vista.yaml", "ckpt": "ckpts/vista.safetensors"}
}

DATASET2SOURCES = {
    "NUSCENES": {"data_root": "data/nuscenes", "anno_file": "annos/nuScenes_val.json"},
    "IMG": {
        "data_root": "/mnt/vita/scratch/datasets/OpenDV-YouTube/val_images/"
    },  # "image_folder"},
}


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--version", type=str, default="gem", help="model version")
    parser.add_argument("--dataset", type=str, default="NUSCENES", help="dataset name")
    parser.add_argument("--save", type=str, default="outputs", help="directory to save samples")
    parser.add_argument(
        "--action",
        type=str,
        default="free",
        help="action mode for control, such as traj, cmd, steer, goal",
    )
    parser.add_argument("--n_rounds", type=int, default=1, help="number of sampling rounds")
    parser.add_argument("--n_frames", type=int, default=25, help="number of frames for each round")
    parser.add_argument(
        "--n_conds",
        type=int,
        default=1,
        help="number of initial condition frames for the first round",
    )
    parser.add_argument("--seed", type=int, default=23, help="random seed for seed_everything")
    parser.add_argument(
        "--height", type=int, default=576, help="target height of the generated video"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="target width of the generated video"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.5,
        help="scale of the classifier-free guidance",
    )
    parser.add_argument(
        "--cond_aug", type=float, default=0.0, help="strength of the noise augmentation"
    )
    parser.add_argument("--n_steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument(
        "--rand_gen",
        action="store_false",
        help="whether to generate samples randomly or sequentially",
    )
    parser.add_argument("--low_vram", action="store_true", help="whether to save memory or not")
    parser.add_argument(
        "--num_evals", type=int, default=10, help="number of samples to evaluate on"
    )
    parser.add_argument(
        "--condition_type",
        type=str,
        default="unconditional",
        choices=[
            "unconditional",
            "object_manipulation",
            "skeleton_manipulation",
            "object_insertion",
            "ego_motion",
        ],
        help="type of condition",
    )
    parser.add_argument(
        "--load_videos",
        action="store_true",
        help="whether to load videos or generate them. Only works for unconditional samples",
    )
    parser.add_argument(
        "--control-frames",
        nargs="+",
        type=int,
        default=[0, 5, 11, 18, 24],
        help="list of frame numbers to put control on",
    )
    return parser


def get_sample(selected_index=0, dataset_name="NUSCENES", num_frames=25, action_mode="free"):
    dataset_dict = DATASET2SOURCES[dataset_name]
    action_dict = None
    if dataset_name == "IMG":
        image_list = os.listdir(dataset_dict["data_root"])
        total_length = len(image_list)
        while selected_index >= total_length:
            selected_index -= total_length
        image_file = image_list[selected_index]

        path_list = [os.path.join(dataset_dict["data_root"], image_file)] * num_frames
    else:
        with open(dataset_dict["anno_file"], "r") as anno_json:
            all_samples = json.load(anno_json)
        total_length = len(all_samples)
        while selected_index >= total_length:
            selected_index -= total_length
        sample_dict = all_samples[selected_index]

        path_list = list()
        if dataset_name == "NUSCENES":
            for index in range(num_frames):
                image_path = os.path.join(dataset_dict["data_root"], sample_dict["frames"][index])
                assert os.path.exists(image_path), image_path
                path_list.append(image_path)
            if action_mode != "free":
                action_dict = dict()
                if action_mode == "traj" or action_mode == "trajectory":
                    action_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
                elif action_mode == "cmd" or action_mode == "command":
                    action_dict["command"] = torch.tensor(sample_dict["cmd"])
                elif action_mode == "steer":
                    # scene might be empty
                    if sample_dict["speed"]:
                        action_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
                    # scene might be empty
                    if sample_dict["angle"]:
                        action_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
                elif action_mode == "goal":
                    # point might be invalid
                    if (
                        sample_dict["z"] > 0
                        and 0 < sample_dict["goal"][0] < 1600
                        and 0 < sample_dict["goal"][1] < 900
                    ):
                        action_dict["goal"] = torch.tensor(
                            [
                                sample_dict["goal"][0] / 1600,
                                sample_dict["goal"][1] / 900,
                            ]
                        )
                else:
                    raise ValueError(f"Unsupported action mode {action_mode}")
        else:
            raise ValueError(f"Invalid dataset {dataset_name}")
    return path_list, selected_index, total_length, action_dict


def load_img(file_name, target_height=320, target_width=576, device="cuda"):
    if file_name is not None:
        image = Image.open(file_name)
        if not image.mode == "RGB":
            image = image.convert("RGB")
    else:
        raise ValueError(f"Invalid image file {file_name}")
    ori_w, ori_h = image.size
    # print(f"Loaded input image of size ({ori_w}, {ori_h})")

    if ori_w / ori_h > target_width / target_height:
        tmp_w = int(target_width / target_height * ori_h)
        left = (ori_w - tmp_w) // 2
        right = (ori_w + tmp_w) // 2
        image = image.crop((left, 0, right, ori_h))
    elif ori_w / ori_h < target_width / target_height:
        tmp_h = int(target_height / target_width * ori_w)
        top = (ori_h - tmp_h) // 2
        bottom = (ori_h + tmp_h) // 2
        image = image.crop((0, top, ori_w, bottom))
    image = image.resize((target_width, target_height), resample=Image.LANCZOS)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
    )(image)
    return image.to(device)


def main():
    fid_per_video = FrechetInceptionDistance(feature=2048)
    fid_total = FrechetInceptionDistance(feature=2048)

    parser = parse_args()
    opt, unknown = parser.parse_known_args()

    if opt.load_videos:
        helper = EvaluationHelper(opt.condition_type, None)
        real_videos_path = [
            os.path.join(opt.save, "real", "npy", path)
            for path in os.listdir(os.path.join(opt.save, "real", "npy"))
            if path.endswith(".npy")
        ]
        generated_videos_path = [
            os.path.join(opt.save, "virtual", "npy", path)
            for path in os.listdir(os.path.join(opt.save, "virtual", "npy"))
            if path.endswith(".npy")
        ]
        metrics = helper.get_evaluation_metrics(real_videos_path, generated_videos_path)
        print(metrics)
        breakpoint()
        # if opt.condition_type in ["Unconditional", "GT-Moving-Control"]:
        #     # evaluate_tracking_from_videos(opt)
        #
        #     evaluate_quality_from_videos(opt, fid_total)
        # else:
        #     raise ValueError(f"Invalid condition type {opt.condition_type} for loading videos")
        # return

    base_val_path = DATASET2SOURCES[opt.dataset]["data_root"]
    seed_everything(opt.seed)

    youtubers = [os.path.join(base_val_path, name) for name in os.listdir(base_val_path)]
    video_paths = []
    for youtuber in youtubers:
        video_paths.extend([os.path.join(youtuber, name) for name in os.listdir(youtuber)])

    # Create and load models
    set_lowvram_mode(opt.low_vram)
    version_dict = VERSION2SPECS[opt.version]
    model = init_model(version_dict)
    encoder = model.conditioner.embedders[-1]
    unique_keys = set([x.input_key for x in model.conditioner.embedders])
    num_frames = opt.n_frames

    if opt.n_rounds > 1:
        guider = "TrianglePredictionGuider"
    else:
        guider = "VanillaCFG"
    sampler = init_sampling(
        guider=guider,
        steps=opt.n_steps,
        cfg_scale=opt.cfg_scale,
        num_frames=opt.n_frames,
    )
    uc_keys = [
        "img_seq",
        "cond_frames",
        "cond_frames_without_noise",
    ]

    metrics = {
        "box_misplacement": {},
    }
    for metric in metrics.keys():
        for i in opt.control_frames:
            metrics[metric][i] = []
    metrics["fid"] = []
    visualize = True
    val_img_idx = 0
    while val_img_idx < opt.num_evals:
        # Load video frames
        chosen_video = random.choice(video_paths)
        chosen_video_frames_path = sorted(
            [os.path.join(chosen_video, name) for name in os.listdir(chosen_video)]
        )
        start_frame = random.randint(0, len(chosen_video_frames_path) - num_frames)
        print("Evaluating on video: ", chosen_video, "starting from frame: ", start_frame)
        ground_truth_images = []
        ground_truth_images_cond = []
        tracker_model = YOLO("yolo11n.pt")
        tracking_results = []
        for i in range(num_frames):
            gt_path = chosen_video_frames_path[start_frame + i]

            gt_img = load_img(gt_path, opt.height, opt.width)
            ground_truth_images_cond.append(gt_img)

            img_gt = cv2.imread(gt_path)
            img_gt = cv2.resize(img_gt, (opt.width, opt.height))
            tracking_results.append(tracker_model.track(img_gt, persist=True, verbose=False)[0])
            if i == 0:  # select the best box in the first frame and break if no good box is found
                boxes = tracking_results[-1].boxes
                if boxes.id is None:
                    best_box_id = None
                    break
                boxes = boxes.xyxy[boxes.cls == 2]  # only track cars
                best_box_id = select_best_box(boxes, opt.width, opt.height)
                if best_box_id is None:
                    break
            img_gt = cv2.cvtColor(img_gt.astype(np.float32), cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_gt).permute(2, 0, 1) / 255.0
            ground_truth_images.append(img_tensor)
        if best_box_id is None:
            print("No good car detected in the first frame.")
            continue
        ground_truth_images_cond = torch.stack(ground_truth_images_cond)
        ground_truth_images = torch.stack(ground_truth_images)

        boxes = tracking_results[0].boxes
        selected_id = int(boxes.id[boxes.cls == 2][best_box_id])
        tracked_boxes = []
        for result in tracking_results:
            if result.boxes.id is not None and selected_id in result.boxes.id:
                tracked_boxes.append(result.boxes.xyxy[result.boxes.id == selected_id][0])
            else:
                tracked_boxes.append(None)
        # create mp4 video with roi overlay
        if visualize:
            os.makedirs(os.path.join(opt.save), exist_ok=True)
            writer = imageio.get_writer(
                os.path.join(opt.save, f"{opt.dataset}_{val_img_idx:06}_roi_overlay.mp4"),
                fps=10,
            )
            for i in range(num_frames):
                img = (ground_truth_images[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(
                    np.uint8
                )
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if tracked_boxes[i] is not None:
                    img = cv2.rectangle(
                        img,
                        (int(tracked_boxes[i][0]), int(tracked_boxes[i][1])),
                        (int(tracked_boxes[i][2]), int(tracked_boxes[i][3])),
                        (0, 255, 0),
                        2,
                    )
                writer.append_data(img)
            writer.close()

        # Process model inputs
        value_dict = init_embedder_options(unique_keys)
        cond_img = ground_truth_images_cond[0][None]
        value_dict["img_seq"] = ground_truth_images_cond
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = opt.cond_aug
        value_dict["cond_frames"] = cond_img + opt.cond_aug * torch.randn_like(cond_img)

        if opt.condition_type == "GT-Moving-Control":
            # get dino features
            all_rectangle_regions = [
                (roi_to_rectangle_region(roi, opt.width, opt.height) if roi is not None else None)
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

            at_when = torch.tensor([i for i in opt.control_frames if tracked_boxes[i] is not None])
            at_location = all_rectangle_regions_tensor[at_when]
            to_location = at_location
            to_when = at_when
            assert torch.all(
                at_location != -2
            ), "The at_when should have filtered out the -2 values"
            dino_features = encoder.get_demo_input(  # use get_demo_input2 from demo3 instead, with masks and for highres use 1-4 dino tokens. Masks should be of size out_res
                ground_truth_images_cond,
                at_location,
                at_when,
                to_location,
                to_when,
                num_total_frames=25,
                num_tokens=random.randint(
                    0, 4
                ),  # for low res 8-16 tokens, for high res 1-4 tokens
            )
            value_dict["fd_crossattn"] = dino_features
        elif opt.condition_type == "Random-Moving-Control":
            raise NotImplementedError("Random-Moving-Control not implemented yet")
        else:
            value_dict["fd_crossattn"] = torch.zeros(
                25, 768, opt.height // 8, opt.width // 8, 1, 1
            )

        # generate the video
        out = do_sample(
            ground_truth_images_cond,
            model,
            sampler,
            value_dict,
            num_rounds=opt.n_rounds,
            num_frames=opt.n_frames,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[index for index in range(opt.n_conds)],
        )
        samples, samples_z, inputs = out
        if visualize:
            virtual_path = os.path.join(opt.save, "virtual")
            real_path = os.path.join(opt.save, "real")
            perform_save_locally(virtual_path, samples, "videos", opt.dataset, val_img_idx)
            # perform_save_locally(virtual_path, samples, "grids", opt.dataset, val_img_idx)
            # perform_save_locally(virtual_path, samples, "images", opt.dataset, val_img_idx)
            perform_save_locally(real_path, inputs, "videos", opt.dataset, val_img_idx)
            # perform_save_locally(real_path, inputs, "grids", opt.dataset, val_img_idx)
            # perform_save_locally(real_path, inputs, "images", opt.dataset, val_img_idx)

        tracker_model = YOLO("yolo11n.pt")
        sampled_video = (samples.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
        sampled_video[0] = (ground_truth_images[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(
            np.uint8
        )  # replace the first frame with the ground truth for consistency with the tracking model
        sampled_video = sampled_video[..., ::-1]
        eval_tracking_results = [
            tracker_model.track(
                np.ascontiguousarray(sampled_video_frame), persist=True, verbose=False
            )[0]
            for sampled_video_frame in sampled_video
        ]
        if visualize:
            writer = imageio.get_writer(
                os.path.join(opt.save, f"{opt.dataset}_{val_img_idx:06}_gen_track.mp4"),
                fps=10,
            )
            for i in range(num_frames):
                annotated_frame = eval_tracking_results[i].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                writer.append_data(annotated_frame)
            writer.close()
        eval_tracked_boxes = []
        for result in eval_tracking_results:
            if result.boxes.id is not None and selected_id in result.boxes.id:
                eval_tracked_boxes.append(result.boxes.xyxy[result.boxes.id == selected_id][0])
            else:
                eval_tracked_boxes.append(None)
        # get metrics
        for i in opt.control_frames:
            if tracked_boxes[i] is not None and eval_tracked_boxes[i] is not None:
                gt_box_center = np.array(
                    [
                        (tracked_boxes[i][0] + tracked_boxes[i][2]) / 2,
                        (tracked_boxes[i][1] + tracked_boxes[i][3]) / 2,
                    ]
                )
                eval_box_center = np.array(
                    [
                        (eval_tracked_boxes[i][0] + eval_tracked_boxes[i][2]) / 2,
                        (eval_tracked_boxes[i][1] + eval_tracked_boxes[i][3]) / 2,
                    ]
                )
                metrics["box_misplacement"][i].append(
                    np.linalg.norm(gt_box_center - eval_box_center)
                )

        ground_truth_images = ground_truth_images * 255
        ground_truth_images = ground_truth_images.type(torch.uint8)

        eval_frames = samples * 255
        eval_frames = eval_frames.type(torch.uint8).squeeze(0)

        fid_per_video.update(ground_truth_images, real=True)
        fid_per_video.update(eval_frames.cpu(), real=False)
        fid_score = fid_per_video.compute()
        fid_per_video.reset()

        fid_total.update(ground_truth_images, real=True)
        fid_total.update(eval_frames.cpu(), real=False)

        metrics["fid"].append(fid_score.item())
        print(f"Video {val_img_idx} completed.")
        val_img_idx += 1
        for i in opt.control_frames:
            if tracked_boxes[i] is not None and eval_tracked_boxes[i] is not None:
                print(f"{i}:", metrics["box_misplacement"][i][-1])
        print(metrics["box_misplacement"])

    fvd_evaluator = fvd.cdfvd(
        "i3d",
        n_real=opt.num_evals,
        n_fake=opt.num_evals,
        compute_feats=True,
        device="cuda",
        ckpt_path="ckpts/i3d_pretrained_400.pt",
    )
    fvd_evaluator.compute_real_stats(
        fvd_evaluator.load_videos(
            os.path.join(opt.save, "real/videos/"),
            data_type="video_folder",
            resolution=opt.height,
            sequence_length=25,
            num_workers=0,
        )
    )
    fvd_evaluator.compute_fake_stats(
        fvd_evaluator.load_videos(
            os.path.join(opt.save, "virtual/videos/"),
            data_type="video_folder",
            resolution=opt.height,
            sequence_length=25,
            num_workers=0,
        )
    )
    fvd_score = fvd_evaluator.compute_fvd_from_stats()

    for metric in metrics.keys():
        if metric != "fid":
            print(f"Average {metric}: ", end="")
            for i in opt.control_frames[1:]:
                print(f"{np.mean(metrics[metric][i]):.2f} ", end="")
            print(
                "Average: ",
                np.mean([np.mean(metrics[metric][i]) for i in opt.control_frames[1:]]),
            )
        else:
            print(f"Average FID Score: {np.mean(metrics['fid'])}")
    print("Total FID Score: ", fid_total.compute())

    print(f"FVD Score: {fvd_score}")
    # csv_file_path = os.path.join("csv_results", f"val_results_{video_counter}_videos.csv")
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)
    #     writer.writerows(rows)


def select_best_box(boxes, width, height):
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


def evaluate_quality_from_videos(opt, fid_total):
    """
    Evaluate the quality of the pre-generated videos using FVD and FID scores
    """
    real_videos_path, fake_videos_path = "real/videos/", "virtual/videos/"
    video_names = [
        x for x in os.listdir(os.path.join(opt.save, real_videos_path)) if x.endswith(".mp4")
    ]
    video_names_fake = [
        x for x in os.listdir(os.path.join(opt.save, fake_videos_path)) if x.endswith(".mp4")
    ]
    assert len(video_names) == len(
        video_names_fake
    ), "Number of real and generated videos must be the same"

    fvd_evaluator = fvd.cdfvd(
        "i3d",
        n_real=len(video_names),
        n_fake=len(video_names_fake),
        compute_feats=True,
        device="cuda",
        ckpt_path="ckpts/i3d_pretrained_400.pt",
    )
    fvd_evaluator.compute_real_stats(
        fvd_evaluator.load_videos(
            os.path.join(opt.save, real_videos_path),
            data_type="video_folder",
            resolution=576,
            sequence_length=25,
            num_workers=10,
        )
    )
    fvd_evaluator.compute_fake_stats(
        fvd_evaluator.load_videos(
            os.path.join(opt.save, fake_videos_path),
            data_type="video_folder",
            resolution=576,
            sequence_length=25,
            num_workers=10,
        )
    )
    fvd_score = fvd_evaluator.compute_fvd_from_stats()
    print(f"FVD Score: {fvd_score}")

    batch_size, real_frames, fake_frames = 128, [], []
    fid_total = fid_total.cuda()
    for video_name in tqdm(video_names):
        if not video_name.endswith(".mp4"):
            continue
        real_video_path = os.path.join(opt.save, real_videos_path, video_name)
        fake_video_path = os.path.join(opt.save, fake_videos_path, video_name)
        real_video = imageio.get_reader(real_video_path)
        fake_video = imageio.get_reader(fake_video_path)
        for i in range(opt.n_frames):
            real_frames.append(real_video.get_data(i))
            fake_frames.append(fake_video.get_data(i))
        if len(real_frames) >= batch_size:
            real_frames_tensor = (
                torch.tensor(np.array(real_frames)).permute(0, 3, 1, 2).type(torch.uint8).cuda()
            )
            fake_frames_tensor = (
                torch.tensor(np.array(fake_frames)).permute(0, 3, 1, 2).type(torch.uint8).cuda()
            )
            fid_total.update(real_frames_tensor, real=True)
            fid_total.update(fake_frames_tensor, real=False)
            real_frames, fake_frames = (
                real_frames[batch_size:],
                fake_frames[batch_size:],
            )

    fid_score = fid_total.compute().item()
    print(f"FID Score: {fid_score}")


def evaluate_tracking_from_videos(opt):
    """
    Evaluate the tracking quality of the pre-generated videos using the tracking model
    It selects the best box in the first frame of GT, and tracks it through both videos.
    It then compares the tracked boxes in the GT and the generated video.
    """
    gt_videos_dir, val_videos_dir = os.path.join(opt.save, "real/videos/"), os.path.join(
        opt.save, "virtual/videos/"
    )
    videos_names = os.listdir(gt_videos_dir)
    box_misplacement = {}
    for i in opt.control_frames:
        box_misplacement[i] = []
    for video_name in tqdm(videos_names):
        if not video_name.endswith(".mp4"):
            continue
        # load the GT video and track the best box
        tracker_model = YOLO("yolo11n.pt")
        cap_gt = cv2.VideoCapture(os.path.join(gt_videos_dir, video_name))
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
        best_box_id = select_best_box(boxes, opt.width, opt.height)
        if best_box_id is None:
            continue

        # load the generated video and track the same box
        tracker_model = YOLO("yolo11n.pt")
        cap_val = cv2.VideoCapture(os.path.join(val_videos_dir, video_name))
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

        # writer = imageio.get_writer(os.path.join(opt.save, video_name.replace(".mp4", "") + "_gt_track2.mp4"), fps=10)
        # for i in range(opt.n_frames):
        #     annotated_frame = gt_tracking_results[i].plot()
        #     annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        #     writer.append_data(annotated_frame)
        # writer.close()
        # writer = imageio.get_writer(os.path.join(opt.save, video_name.replace(".mp4", "") + "_gen_track2.mp4"), fps=10)
        # for i in range(opt.n_frames):
        #     annotated_frame = val_tracking_results[i].plot()
        #     annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        #     writer.append_data(annotated_frame)
        # writer.close()

        # get the tracked boxes for the selected box
        selected_id = int(
            gt_tracking_results[0].boxes.id[gt_tracking_results[0].boxes.cls == 2][best_box_id]
        )
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

        # compare the tracked boxes
        for i in opt.control_frames:
            if gt_tracked_boxes[i] is not None and eval_tracked_boxes[i] is not None:
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
                box_misplacement[i].append(np.linalg.norm(gt_box_center - eval_box_center))
    print(f"Average box misplacement: ", end="")
    for i in opt.control_frames[1:]:
        print(f"{np.mean(box_misplacement[i]):.2f} ", end="")
    print(
        "Average: ",
        np.mean([np.mean(box_misplacement[i]) for i in opt.control_frames[1:]]),
    )


if __name__ == "__main__":
    main()
