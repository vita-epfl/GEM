import argparse
import json
import os
import pickle
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision import transforms

# import init_proj_path
from gem.utils.sample_utils import *

VERSION2SPECS = {
    "final_lr": {
        "config": "ckpts/gem_n128_stage2_2100.yaml",
        "ckpt": "ckpts/gem_n128_stage2_2100.safetensors",
        # "config": "ckpts2/gem_opendv_n128.yaml",
        # "ckpt": "ckpts2/gem_n128_stage2_300.safetensors"
        # "ckpt": "ckpts2/gem_opendv_n128.safetensors"
    },
    "gem": {
        "config": "logs/2024-10-03T09-52-03_example-stage2_debug/configs/2024-10-03T09-52-03-project.yaml",  # configs/2024-09-23T09-23-08-project.yaml",
        "ckpt": "/home/ss24m050/Documents/worldm/logs/2024-10-03T09-52-03_example-stage2_debug/checkpoints/last.ckpt",
    },
    "svd_full_temp": {
        "config": "logs/2024-10-02T22-20-07_s2_n16_full_temp_svd/configs/2024-10-02T22-20-07-project.yaml",
        "ckpt": "ckpts/half_res_svd_full_temp.safetensors",
    },
    "add": {
        "config": "logs/2024-10-07T22-14-37_s1_n16/configs/2024-10-08T12-16-03-project.yaml",
        "ckpt": "ckpts/add.safetensors",
    },
    "add_hr": {
        "config": "logs/2024-10-08T22-03-31_high_res_add/configs/2024-10-09T11-52-43-project.yaml",
        "ckpt": "ckpts/add_hr.safetensors",
    },
    "cross": {
        "config": "logs/2024-10-07T22-14-37_s0_n16/configs/2024-10-08T13-42-52-project.yaml",
        "ckpt": "ckpts/cross.safetensors",
    },
    "add_id": {
        "config": "logs/2024-10-14T11-30-42_adding_id/configs/2024-10-14T22-42-28-project.yaml",
        "ckpt": "ckpts/adding_id.safetensors",
    },
    "n": {
        "config": "stage0_n128/configs/2024-10-18T05-41-16-project.yaml",
        "ckpt": "ckpts/stage0_n128.safetensors",
    },
    "local": {
        "config": "checkpoints/stage0_n128/configs/2024-10-18T05-41-16-project.yaml",
        "ckpt": "checkpoints/stage0_n128.safetensors",
    },
    "local2": {
        "config": "checkpoints/2024-10-19T19-31-58_example-fs0/configs/2024-10-19T19-31-58-project.yaml",
        "ckpt": "checkpoints/high_res_last.safetensors",
    },
    "latest":
    { 
        "config": "configs/inference/gem_new_stage2.yaml",
        "ckpt": "gem_new_stage2.safetensors"
    }
}

# driving = "/var/tmp/europe_videos_converted/val/J_Utah_paVB7zNvb0E"
# driving = "/var/tmp/europe_videos_converted/val/Gezeyenti_Y9vRHf-TI5U"
driving = "/store/swissai/a03/datasets/OpenDV-YouTube/val_images/Driving_Experience/SJsssmcq8U4"
driving = "/store/swissai/a03/datasets/OpenDV-YouTube/val_images/KenoVelicanstveni/94EDSmtNCxM"
#driving = "/store/swissai/a03/datasets/OpenDV-YouTube/val_images/Driving_Experience/SJsssmcq8U4"
driving = "val_images/"
driving = "/capstor/store/cscs/swissai/a03/datasets/OpenDV-YouTube/val_images/Driving_Experience/nOP1blfMCTg"

DATASET2SOURCES = {
    "IMG": {"data_root": driving},  # "image_folder"},
    "LOCAL": {"data_root": "val_images"},
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
        help="action mode for control, such as traj, cmd, steer, goal, dino",
    )
    parser.add_argument("--n_rounds", type=int, default=1, help="number of sampling rounds")
    parser.add_argument("--n_frames", type=int, default=25, help="number of frames for each round")
    parser.add_argument(
        "--n_conds",
        type=int,
        default=0,
        help="number of initial condition frames for the first round",
    )
    parser.add_argument("--seed", type=int, default=22, help="random seed for seed_everything")
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
    parser.add_argument("--sampler", type=str, default="EulerEDMSamplerDynamicPyramid")
    parser.add_argument("--guider", type=str, default="VanillaCFG")
    parser.add_argument("--dino_data", type=str, default="action.pkl")
    return parser


def get_sample(
    selected_index=0,
    dataset_name="NUSCENES",
    num_frames=25,
    action_mode="free",
    demo=False,
):
    dataset_dict = DATASET2SOURCES[dataset_name]
    action_dict = None
    if dataset_name in ["IMG", "LOCAL"]:
        # full_image_list = os.listdir(dataset_dict["data_root"])
        # print(len(full_image_list))
        # image_list = list()
        # for path in full_image_list:
        #     end_number = path.split("_")[-1].split(".")[0]
        #     if end_number.isdigit() and int(end_number) < 1000:
        #         image_list.append(path)
        image_list = sorted(os.listdir(dataset_dict["data_root"]))
        total_length = len(image_list)
        while selected_index >= total_length:
            selected_index -= total_length
        image_file = image_list[selected_index]

        # path_list = [os.path.join(dataset_dict["data_root"], image_file)] * num_frames

        path_list = list()
        for index in range(num_frames):
            image_file = image_list[(selected_index + index) % total_length]
            path_list.append(os.path.join(dataset_dict["data_root"], image_file))
    else:
        with open(dataset_dict["anno_file"], "r") as anno_json:
            all_samples = json.load(anno_json)
        total_length = len(all_samples)
        while selected_index >= total_length:
            selected_index -= total_length
        sample_dict = all_samples[selected_index]

        path_list = list()
        if demo:
            action_dict["fd_crossattn"] = torch.randn(25, 256, 764, 1, 1)

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


def annotate_masks(images, source_masks, target_masks, target_when=None):
    """
    Annotate source and target masks on the input images.
    Args:
        images (torch.Tensor): Input image tensor of shape (N, C, H, W).
        source_masks (torch.Tensor): Source masks of shape (N, H, W).
        target_masks (torch.Tensor): Target masks of shape (N, H, W).
        target_when (Optional): List or int for target identification, specifying the frame or condition.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)

    images_out = []
    for nr, image in enumerate(images):
        image_np = image.cpu().numpy()

        # If the image tensor is in the range [0, 1], scale it to [0, 255]
        if image_np.max() <= 1.0 and image_np.min() >= 0.0:
            image_np = (image_np * 255).astype(np.uint8)
        elif image_np.min() < 0.0:
            image_np = ((image_np + 1) * 127.5).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        # Convert (C, H, W) to (H, W, C) if necessary
        if image_np.shape[0] == 3 and image_np.ndim == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        # Ensure the image is in H, W, 3 format
        if image_np.ndim == 2:  # Convert grayscale to color
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 1:  # Convert single channel to 3 channels
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        # Convert the image back to uint8 (in case it's still float32)
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        H, W, _ = image_np.shape

        # Resize all masks to the image size
        source_mask_resized = (
            F.interpolate(source_masks.unsqueeze(1).float(), size=(H, W), mode="nearest")
            .squeeze(1)
            .cpu()
            .numpy()
        )
        target_mask_resized = (
            F.interpolate(target_masks.unsqueeze(1).float(), size=(H, W), mode="nearest")
            .squeeze(1)
            .cpu()
            .numpy()
        )

        # Overlay all source masks (transparent green)
        for i in range(source_mask_resized.shape[0]):
            mask_area = source_mask_resized[i] > 0
            green_overlay = np.zeros_like(image_np, dtype=np.uint8)
            green_overlay[:, :, 1] = 255  # Green channel
            alpha = 0.4
            image_np[mask_area] = cv2.addWeighted(image_np, 1 - alpha, green_overlay, alpha, 0)[
                mask_area
            ]

        # Overlay all target masks (transparent red, slightly different shades per target_when)
        for i in range(target_mask_resized.shape[0]):
            mask_area_target = target_mask_resized[i] > 0
            red_overlay = np.zeros_like(image_np, dtype=np.uint8)
            red_overlay[:, :, 2] = 255  # Red channel

            idx = (
                target_when
                if isinstance(target_when, int)
                else target_when[i] if target_when is not None else i
            )
            idx = idx.cpu()
            red_shade_variation = (idx % 5) * 20  # Slight variation in red intensity
            red_overlay[:, :, 2] = 255 - red_shade_variation

            image_np[mask_area_target] = cv2.addWeighted(
                image_np, 1 - alpha, red_overlay, alpha, 0
            )[mask_area_target]

        # Draw frame count in the top left corner
        frame_text = (
            f"Frame: {nr} Target: t={target_when.tolist() if target_when is not None else nr}"
        )

        # Convert the image to correct format before using cv2.putText
        image_np = np.ascontiguousarray(image_np)

        cv2.putText(
            image_np,
            frame_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Normalize the image back to the range [-1, 1]
        image_np = image_np.astype(np.float32) / 255
        image_np = (image_np * 2) - 1
        images_out.append(torch.from_numpy(image_np))

    return torch.stack(images_out, dim=0)


def get_demo_conditions(
    embedder,
    #images,
    filename
):
    #image = images[0]
    #image = image[None].to("cuda")

    # filename = "selection_data2.pkl"
    #filename = "selection_data2.pkl"
    #filename = "highway_mv_left_insert_right.pkl"
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)

    # Access the loaded data
    target_images = loaded_data["target_images"]
    loaded_source_images = loaded_data["source_images"]
    loaded_bounding_boxes = loaded_data["bounding_boxes"]
    loaded_identities = loaded_data["identities"]
    loaded_source_masks = loaded_data["source_masks"]
    loaded_target_masks = loaded_data["target_masks"]
    loaded_source_indices = loaded_data["source_indices"]
    loaded_target_indices = loaded_data["target_indices"]
    loaded_condition_numbers = loaded_data["condition_numbers"]

    pixel_values = (
        torch.from_numpy(loaded_source_images).cuda().to("cuda") / 255
    ) * 2 - 1  # .repeat(2, 1, 1, 1)
    target_images = (
        torch.from_numpy(target_images).cuda().to("cuda") / 255
    ) * 2 - 1  # .repeat(2, 1, 1, 1)
    bounding_boxes = torch.from_numpy(loaded_bounding_boxes).to("cuda")  # .repeat(2, 1)
    identities = torch.from_numpy(loaded_identities).to("cuda")  # .repeat(2)
    source_masks = torch.from_numpy(loaded_source_masks).to("cuda").bool()  # .repeat(2, 1, 1)
    target_masks = torch.from_numpy(loaded_target_masks).to("cuda").bool()  # .repeat(2, 1, 1)
    source_indices = torch.from_numpy(loaded_source_indices).to("cuda")  # .repeat(2)
    target_indices = torch.from_numpy(loaded_target_indices).to("cuda")  # .repeat(2)
    print(pixel_values.shape)

    #source_indices = torch.tensor([0])
    if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
        print(
            "Warning: z_where contains NaN or Inf values",
        )  # source_indices = torch.tensor([0, 0]).to("cuda")

    if torch.isnan(bounding_boxes).any() or torch.isinf(bounding_boxes).any():
        print(
            "Warning: z_where contains NaN or Inf values"
        )  # source_indices = torch.tensor([0, 0]).to("cuda")
    # target_indices = torch.tensor([4, 16]).to("cuda")

    condition_numbers = torch.from_numpy(loaded_condition_numbers).to("cuda")  # .repeat(2, 1)
    # target_indices[-4:] = torch.tensor([15, 15, 15, 15]).to("cuda")
    print(pixel_values.shape)
    print(bounding_boxes.shape)
    print(source_masks.shape)
    print(source_indices.shape)
    (
        pixel_values,
        bounding_boxes,
        identities,
        source_masks,
        target_masks,
        source_indices,
        target_indices,
        condition_numbers,
    ) = (
        [pixel_values],
        [bounding_boxes],
        [identities],
        [source_masks],
        [target_masks],
        [source_indices],
        [target_indices],
        [condition_numbers],
    )

    dino_conds = []

    embedder.to("cuda")
    with autocast("cuda", dtype=torch.float32):
        # if True:
        for i in range(len(source_masks)):
            dino_cond = embedder.get_demo_input2(
                pixel_values[i],
                source_idxs=source_indices[i],
                source_crops=bounding_boxes[i],
                source_masks=source_masks[i],
                target_idxs=target_indices[i],
                target_masks=target_masks[i],
                num_total_frames=25,
                ids=None,  # identities[i],
            )
            dino_conds.append(dino_cond)
    return dino_conds, source_masks, target_masks, target_indices, target_images


if __name__ == "__main__":
    parser = parse_args()
    opt, unknown = parser.parse_known_args()

    set_lowvram_mode(opt.low_vram)
    version_dict = VERSION2SPECS[opt.version]
    model = init_model(version_dict)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])
    unique_keys.add("fd_crossattn")
    unique_keys.add("skeletons_context")
    unique_keys.add("rendered_poses")

    if True:
        seed_everything(opt.seed)

        # use same frames
        frame_list, sample_index, dataset_length, action_dict = get_sample(
            1281,  # 50, #5, #3453,  # 5,
            opt.dataset,
            25,
            opt.action,
            # sample_index, opt.dataset, 25, opt.action
        )

        img_seq = list()
        for each_path in frame_list:
            img = load_img(each_path, opt.height, opt.width)
            img_seq.append(img)
        images = torch.stack(img_seq)

        dino_cond, gt_box, control_box, target_when, target_images = get_demo_conditions(
            model.conditioner.embedders[-3], opt.dino_data
        )

        for i in range(len(dino_cond)):
            value_dict = init_embedder_options(unique_keys)
            #cond_img = img_seq[0][None]
            cond_img = target_images[0][None]
            images = torch.zeros_like(images)
            images[0] = cond_img
            value_dict["img_seq"] = images
            #value_dict["img_seq"] = images  # g_seq
            value_dict["cond_frames_without_noise"] = cond_img
            value_dict["cond_aug"] = opt.cond_aug
            value_dict["cond_frames"] = cond_img + opt.cond_aug * torch.randn_like(cond_img)
            value_dict["fd_crossattn"] = dino_cond[i]
            value_dict["skeletons_context"] = torch.zeros(
                opt.n_frames, 320, opt.height // 8, opt.width // 8, 1, 1, 1
            )  
            value_dict["rendered_poses"] = torch.zeros_like(images)
            print(dino_cond[i].shape)

            # images = images[: opt.n_frames]
            if action_dict is not None:
                for key, value in action_dict.items():
                    value_dict[key] = value

            #if opt.n_rounds > 1:
            #    guider = "TrianglePredictionGuider"
            #else:
            #    guider = "VanillaCFG"

            # sampler="EulerEDMSamplerPyramid",
            # guider="LinearPredictionGuider",
            # discretization="EDMDiscretization",
            # steps=50,
            # num_frames=25,
            sampler = init_sampling(
                sampler=opt.sampler,
                guider=opt.guider,
                steps=opt.n_steps,
                num_frames=opt.n_frames,
                cfg_max_scale=opt.cfg_scale
            )

            uc_keys = [
                "img_seq",
                "cond_frames",
                "cond_frames_without_noise",
                "command",
                "trajectory",
                "speed",
                "angle",
                "goal",
                "skeletons_context",
                "fd_crossattn"
            ]

            out = do_sample(
                images,
                model,
                sampler,
                value_dict,
                num_rounds=opt.n_rounds,
                num_frames=opt.n_frames,
                force_uc_zero_embeddings=uc_keys,
                initial_cond_indices=[index for index in range(opt.n_conds)],
            )
            # out = images, images, images

            if isinstance(out, (tuple, list)):
                samples, samples_z, inputs = out
                virtual_path = os.path.join(opt.save, "virtual")
                real_path = os.path.join(opt.save, "real")
                perform_save_locally(
                    virtual_path, samples, "videos", opt.dataset, sample_index
                )
                perform_save_locally(
                    virtual_path, samples, "grids", opt.dataset, sample_index
                )
                perform_save_locally(
                    virtual_path, samples, "images", opt.dataset, sample_index
                )
                # perform_save_locally(real_path, inputs, "videos", opt.dataset, sample_index, action=0)
                perform_save_locally(
                    real_path, images, "videos", opt.dataset, sample_index
                )
                perform_save_locally(
                    real_path, inputs, "grids", opt.dataset, sample_index
                )
                perform_save_locally(
                    real_path, inputs, "images", opt.dataset, sample_index
                )

                # print(gt_box.shape)
                annotated_images = annotate_masks(
                    inputs, gt_box[i], control_box[i], target_when[i]
                )
                perform_save_locally(
                    real_path,
                    (annotated_images.permute(0, 3, 1, 2) + 1) / 2,
                    "videos_bbox",
                    opt.dataset,
                    sample_index,
                )

                annotated_samples = annotate_masks(
                    samples, gt_box[i], control_box[i], target_when[i]
                )
                perform_save_locally(
                    virtual_path,
                    (annotated_samples.permute(0, 3, 1, 2) + 1) / 2,
                    "videos_bbox",
                    opt.dataset,
                    sample_index,
                )
            else:
                raise TypeError

            if opt.rand_gen:
                sample_index += random.randint(1, dataset_length - 1)
            else:
                sample_index += 1
                if dataset_length <= sample_index:
                    sample_index = -1
