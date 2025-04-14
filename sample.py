import os
import argparse
import json
import random

import torch
from pytorch_lightning import seed_everything
from torchvision import transforms
from tqdm import tqdm
from eval_utils import EvaluationHelper
import cv2

from gem.utils.sample_utils import *

VERSION2SPECS = {
    "gem": {
       
        "ckpt": "checkpoints/gem_new_stage2.safetensors",
        "config": "configs/training/stage2.yaml",
       
    }
   
}
data_root = "datasets/nuscenes/"

driving = "datasets/OpenDV-YouTube/val_images/Driving_Experience/nOP1blfMCTg"

zero_shot = "datasets/zero-samples"


DATASET2SOURCES = {
    "NUSCENES": {"data_root": data_root, "anno_file": "annotations/nuScenes_val.json"},
    "IMG": {"data_root": driving},  
    "ZeroShot": {"data_root": zero_shot},
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
    parser.add_argument("--seed", type=int, default=50, help="random seed for seed_everything")
    parser.add_argument(
        "--height", type=int, default=576, help="target height of the generated video"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="target width of the generated video"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="scale of the classifier-free guidance",
    )
    parser.add_argument("--n_steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument(
        "--rand_gen",
        action="store_false",
        help="whether to generate samples randomly or sequentially",
    )
    parser.add_argument("--low_vram", action="store_true", help="whether to save memory or not")

    parser.add_argument("--demo", action="store_true", help="whether to run the demo mode")
    parser.add_argument("--sampler", type=str, default="EulerEDMSamplerDynamicPyramid")
    parser.add_argument("--guider", type=str, default="VanillaCFG")
    parser.add_argument("--cfg_min_scale", type=float, default=1.0)
    parser.add_argument("--cfg_max_scale", type=float, default=1.5)
    parser.add_argument(
        "--cond_aug", type=float, default=0.0, help="strength of the noise augmentation"
    )
    parser.add_argument("--sigma_max", type=float, default=150.0)
    parser.add_argument("--rho", type=float, default=7)

    parser.add_argument(
        "--condition_type",
        type=str,
        default="unconditional",
        choices=['unconditional', 'object_manipulation', 'skeleton_manipulation', 'ego_motion'],
        help="type of condition",
    )
    return parser


def get_sample(
    selected_index=0,
    dataset_name="NUSCENES",
    num_frames=25,
    action_mode="free",
):
    dataset_dict = DATASET2SOURCES[dataset_name]
    action_dict = None
    if dataset_name == "IMG":
        image_list = os.listdir(dataset_dict["data_root"])
        total_length = len(image_list)
        while selected_index >= total_length:
            selected_index -= total_length
        path_list = list()
        for index in range(num_frames):
            image_file = image_list[(selected_index + index) % total_length]
            path_list.append(os.path.join(dataset_dict["data_root"], image_file))
    elif dataset_name in ["OpenDV", "OpenDV-250"]:
        with open(dataset_dict["anno_file"], "r") as anno_json:
            all_samples = json.load(anno_json)
        total_length = len(all_samples)
        while selected_index >= total_length:
            print(f"Warning!!!!! Selected index {selected_index} is out of range, total length is {total_length}", "*" * 10)
            selected_index -= total_length
        path_list = all_samples[selected_index]
        path_list = [os.path.join(dataset_dict["data_root"], path) for path in path_list]
    elif dataset_name == "ZeroShot":
        image_list = os.listdir(dataset_dict["data_root"])
        total_length = len(image_list)
        while selected_index >= total_length:
                selected_index -= total_length
        image_file = image_list[selected_index]
        if total_length < num_frames:
            path_list = [os.path.join(dataset_dict["data_root"], image_file)] * num_frames
        else:
            path_list = list()
            for index in range(num_frames):
                image_file = image_list[(selected_index + index) % total_length]
                path_list.append(os.path.join(dataset_dict["data_root"], image_file))
    elif dataset_name == "NUSCENES":
        with open(dataset_dict["anno_file"], "r") as anno_json:
            all_samples = json.load(anno_json)
        total_length = len(all_samples)
        while selected_index >= total_length:
            selected_index -= total_length
        sample_dict = all_samples[selected_index]

        path_list = list()
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
    image = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])(
        image
    )
    return image.to(device)




if __name__ == "__main__":
    parser = parse_args()
    opt, unknown = parser.parse_known_args()
    set_lowvram_mode(opt.low_vram)
    version_dict = VERSION2SPECS[opt.version]
    model = init_model(version_dict)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])

    config = OmegaConf.load(version_dict["config"])
    _, _, dataset_length, _ = get_sample(0, opt.dataset, opt.n_frames, opt.action)

    helper = EvaluationHelper(opt.condition_type, model)
    if opt.condition_type == "ego_motion" and opt.dataset == "NUSCENES":  # taking GT ego motion for nuscenes
        opt.action = "traj"

    for sample_index in tqdm(range(dataset_length), desc="Sampling"):
        seed_everything(opt.seed)
        frame_list, sample_index, dataset_length, action_dict = get_sample(sample_index, opt.dataset, opt.n_frames, opt.action)
        img_seq, control_img_seq = list(), list()
        for each_path in frame_list:
            img = load_img(each_path, opt.height, opt.width)
            img_seq.append(img)
            if opt.condition_type != "unconditional":
                img_gt = cv2.imread(each_path)
                img_gt = cv2.resize(img_gt, (opt.width, opt.height))
                control_img_seq.append(img_gt)
        images = torch.stack(img_seq)

        if action_dict is None:
            controls = helper.get_controls(control_img_seq)
        else:
            controls = {"trajectory": action_dict["trajectory"]}
        if controls is None:
            print(f"Sample id {sample_index} has no control information, skipping.")
            continue

        value_dict = init_embedder_options(unique_keys)
        cond_img = img_seq[0][None]
        value_dict["img_seq"] = torch.zeros_like(images)
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] =  opt.cond_aug
        value_dict["cond_frames"] = cond_img
        value_dict["fd_crossattn"] = torch.zeros(opt.n_frames, 768, opt.height//16, opt.width//16, 1, 1)
        value_dict["skeletons_context"] = torch.zeros(opt.n_frames, 768, opt.height//16, opt.width//16,1, 1, 1)
        value_dict["rendered_poses"] = torch.zeros_like(images)
        for key, value in controls.items():
            value_dict[key] = value

        sampler = init_sampling(
            sampler=opt.sampler,
            guider=opt.guider,
            steps=opt.n_steps,
            cfg_max_scale=opt.cfg_max_scale,
            cfg_min_scale=opt.cfg_min_scale,
            num_frames=opt.n_frames,
            sigma_max=opt.sigma_max,
            rho=opt.rho
        )
        uc_keys = [
            "img_seq",
            "cond_frames",
            "cond_frames_without_noise",
            "skeletons_context",
            "rendered_poses",
            "trajectory",
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

        save_dir = f"{opt.save}/n_frames_{opt.n_frames}_steps_{opt.n_steps}_sampler_{opt.sampler}_guider_{opt.guider}_cfg_scale_min_{opt.cfg_min_scale}_cfg_scale_max_{opt.cfg_max_scale}_cond_aug_{opt.cond_aug}_sigma_max_{opt.sigma_max}_rho_{opt.rho}"
        if isinstance(out, (tuple, list)):
            samples, samples_z, inputs = out
            virtual_path = os.path.join(save_dir, "virtual")
            real_path = os.path.join(save_dir, "real")
            perform_save_locally(virtual_path, samples, "videos", opt.dataset, sample_index)
            perform_save_locally(virtual_path, samples, "npy", opt.dataset, sample_index)

            perform_save_locally(real_path, images, "videos", opt.dataset, sample_index)
            perform_save_locally(real_path, images, "npy", opt.dataset, sample_index)
            print(f"Sample id {sample_index} generation finished and saved.")
        else:
            raise TypeError