import argparse
import json
import os
import random

import torch
from pytorch_lightning import seed_everything
from torchvision import transforms

# import init_proj_path
from gem.utils.sample_utils import *

# VERSION2SPECS = {
#    #"gem": {"config": "logs/2024-09-13T08-28-53_1_nodes/configs/2024-09-13T08-28-53-project.yaml", "ckpt": "logs/2024-09-13T08-28-53_1_nodes/checkpoints/last.ckpt"}
#    #"gem": {"config": "logs/2024-09-15T14-31-38_1_nodes/configs/2024-09-15T14-31-38-project.yaml", "ckpt": "/capstor/scratch/cscs/sstapf/AutonomousDriving/ckpts/ckpt_15_09.safetensors"}
#    #"gem": {"config": "logs/2024-09-13T08-41-52_1_nodes/configs/2024-09-13T08-41-52-project.yaml", "ckpt": "ckpts/ckpt.safetensors"}
#   #"gem": {"config": "logs/2024-09-11T21-36-11_16_nodes/configs/2024-09-11T21-36-11-project.yaml", "ckpt": "ckpts/ckpt.safetensors"}
#    "gem": {"config": "logs/2024-09-11T21-36-11_16_nodes/configs/2024-09-11T21-36-11-project.yaml", "ckpt": "ckpts/fp32_model.safetensors"}
#    #"gem": {"config": "configs/inference/vista.yaml", "ckpt": "ckpts/vista.safetensors"}
# }
## fr9s3jqYBvw
# data = "/store/swissai/a03/datasets/OpenDV-YouTube/val_images/KenoVelicanstveni/fr9s3jqYBvw"
##data = "/store/swissai/a03/datasets/OpenDV-YouTube/val_images/KenoVelicanstveni/giViRewuugA"
# DATASET2SOURCES = {
#    "NUSCENES": {"data_root": "data/nuscenes", "anno_file": "annos/nuScenes_val.json"},
#    "IMG": {"data_root": data},
#    #"IMG": {"data_root": "image_folder"},
# }

VERSION2SPECS = {
    #    "gem": {"config": "configs/inference/test.yaml", "ckpt": "ckpts/last.ckpt"}
    # "gem": {"config": "logs/2024-09-15T20-15-10_s1_n16/configs/2024-09-15T20-15-10-project.yaml", "ckpt": "ckpts/stage1.safetensors"}
    "gem": {
        # "config": "/capstor/scratch/cscs/mhasan/AutonomousDriving/logs/2024-10-10T11-08-52_s2_n16/configs/2024-10-10T11-08-52-project.yaml", #configs/2024-09-23T09-23-08-project.yaml",
        # "config": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/checkpoints/stage0_n128/configs/2024-10-18T05-41-16-project.yaml",
        # "ckpt": "/capstor/store/cscs/swissai/a03/long_training/gem_n128_stage2_final.safetensors",
        # "config": "/capstor/store/cscs/swissai/a03/long_training/gem_n128_stage2_final.yaml",
        "ckpt": "/capstor/store/cscs/swissai/a03/long_training/gem_n128_stage2_final.safetensors",
        "config": "/capstor/store/cscs/swissai/a03/long_training/gem_n128_stage2_final.yaml",
        # "ckpt": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/checkpoints/high_res_last.safetensors",
        # "config": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/checkpoints/2024-10-19T19-31-58_example-fs0/configs/2024-10-19T19-31-58-project.yaml",
        # "ckpt": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/custom_sampler_64tokens/custom_3000step_64tokens_ckpt.safetensors",
        # "config": "/mnt/vita/scratch/datasets/OpenDV-YouTube/ckpts/custom_sampler_64tokens/configs/2024-10-20T12-37-28-project.yaml",
        # "ckpt": "/capstor/scratch/cscs/mhasan/AutonomousDriving/ckpts/noisy_rf.safetensors",
    }
    # "gem": {"config": "logs/2024-09-15T20-13-08_s2_n16/configs/2024-09-15T20-13-08-project.yaml", "ckpt": "ckpts/stage2.safetensors"}
    # "gem": {"config": "configs/inference/vista.yaml", "ckpt": "ckpts/vista.safetensors"}
}
data_root = "/store/swissai/a03/datasets/nuscenes/"
# driving = "/mnt/vita/scratch/datasets/OpenDV-YouTube/full_images/Sunset_Drive_California/1Gexr4rtawI"
# driving = (
#     "/mnt/vita/scratch/datasets/OpenDV-YouTube/val_images/KenoVelicanstveni/G45uV97Bkcc"
# )
driving = "/capstor/store/cscs/swissai/a03/datasets/OpenDV-YouTube/val_images/Driving_Experience/nOP1blfMCTg"
# data_root = "/store/swissai/a03/datasets/nuscenes_reid/nuscenes_new"
# driving = "/var/tmp/europe_videos_converted/val/J_Utah_paVB7zNvb0E"

DATASET2SOURCES = {
    "NUSCENES": {"data_root": data_root, "anno_file": "annotations/nuScenes_val.json"},
    "IMG": {"data_root": driving},  # "image_folder"},
    "OpenDV": {"data_root": "/mnt/vita/scratch/datasets/OpenDV-YouTube/val_images/"},
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

    parser.add_argument("--demo", action="store_true", help="whether to run the demo mode")
    parser.add_argument("--sampler", type=str, default="EulerEDMSamplerPyramid2")
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
    if dataset_name == "IMG":
        image_list = os.listdir(dataset_dict["data_root"])
        total_length = len(image_list)
        while selected_index >= total_length:
            selected_index -= total_length
        image_file = image_list[selected_index]

        # path_list = [os.path.join(dataset_dict["data_root"], image_file)] * num_frames

        path_list = list()
        for index in range(num_frames):
            image_file = image_list[(selected_index + index) % total_length]
            path_list.append(os.path.join(dataset_dict["data_root"], image_file))
    elif dataset_name == "OpenDV":
        chosen_video = random.choice(video_paths)
        chosen_video_frames_path = sorted(
            [os.path.join(chosen_video, name) for name in os.listdir(chosen_video)]
        )
        start_frame = random.randint(0, len(chosen_video_frames_path) - num_frames)
        print("Evaluating on video: ", chosen_video, "starting from frame: ", start_frame)
        path_list = chosen_video_frames_path[start_frame : start_frame + num_frames]
        total_length = len(chosen_video_frames_path)
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


def get_gt_sample(
    selected_index=0,
    dataset_name="NUSCENES",
    num_frames=25,
    action_mode="free",
    demo=False,
):
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


def get_demo_condition(
    images,
    at_where: torch.Tensor,
    at_when: torch.Tensor,
    to_where: torch.Tensor,
    to_when: torch.Tensor,
    num_total_frames: int,
):
    return 0


if __name__ == "__main__":
    parser = parse_args()
    opt, unknown = parser.parse_known_args()
    set_lowvram_mode(opt.low_vram)
    version_dict = VERSION2SPECS[opt.version]
    model = init_model(version_dict)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])
    sample_index = 0
    config = OmegaConf.load(version_dict["config"])
    guider = config.model.params.sampler_config.params.guider_config.target
    guider = guider.split(".")[-1]
    guider_max_scale = config.model.params.sampler_config.params.guider_config.params.max_scale
    guider_min_scale = config.model.params.sampler_config.params.guider_config.params.min_scale
    while sample_index >= 0:
        seed_everything(opt.seed)
        frame_list, sample_index, dataset_length, action_dict = get_sample(
            sample_index, opt.dataset, opt.n_frames, opt.action
        )
        img_seq = list()
        for each_path in frame_list:
            img = load_img(each_path, opt.height, opt.width)
            img_seq.append(img)
        images = torch.stack(img_seq)
        value_dict = init_embedder_options(unique_keys)
        cond_img = img_seq[0][None]
        value_dict["img_seq"] = torch.zeros_like(images)
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = torch.tensor([0])  # opt.cond_aug
        value_dict["cond_frames"] = cond_img + opt.cond_aug * torch.randn_like(cond_img)
        value_dict["fd_crossattn"] = torch.zeros(
            opt.n_frames, 776, opt.height // 8, opt.width // 8, 1, 1
        )
        value_dict["skeletons_context"] = torch.zeros(
            opt.n_frames, 320, opt.height // 8, opt.width // 8, 1, 1, 1
        )  # torch.zeros_like(images).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        value_dict["rendered_poses"] = torch.zeros_like(images)
        # value_dict["trajectory"] = torch.zeros([8])
        # images = images[: opt.n_frames]
        if action_dict is not None:
            for key, value in action_dict.items():
                value_dict[key] = value
        sampler = init_sampling(
            sampler=opt.sampler,
            guider=guider,
            steps=opt.n_steps,
            cfg_max_scale=guider_max_scale,
            cfg_min_scale=guider_min_scale,
            num_frames=opt.n_frames,
        )
        uc_keys = [
            "img_seq",
            "cond_frames",
            "cond_frames_without_noise",
            "skeletons_context",
            # "trajectory"
            # "skeletons_context"
            # "command",
            # "speed",
            # "angle",
            # "goal",
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

        if isinstance(out, (tuple, list)):
            samples, samples_z, inputs = out
            virtual_path = os.path.join(opt.save, "virtual")
            real_path = os.path.join(opt.save, "real")
            perform_save_locally(virtual_path, samples, "videos", opt.dataset, sample_index)
            # perform_save_locally(virtual_path, samples, "grids", opt.dataset, sample_index )
            # perform_save_locally(virtual_path, samples, "images", opt.dataset, sample_index )
            # print("changed inputs to images")
            # perform_save_locally(real_path, inputs, "videos", opt.dataset, sample_index)
            perform_save_locally(real_path, images, "videos", opt.dataset, sample_index)
            # perform_save_locally(real_path, inputs, "grids", opt.dataset, sample_index)
            # perform_save_locally(real_path, inputs, "images", opt.dataset, sample_index)
            print(f"Sample id {sample_index} generation finished and saved.")
        else:
            raise TypeError

        if opt.rand_gen:
            sample_index += random.randint(1, dataset_length - 1)
        else:
            sample_index += 1
            if dataset_length <= sample_index:
                break
