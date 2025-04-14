import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision
from einops import rearrange, repeat
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import autocast
from tqdm import tqdm

from gem.modules.diffusionmodules.sampling import (  # EulerEDMSamplerDynamicPyramid2,
    EulerEDMSampler,
    EulerEDMSamplerDynamicPyramid,
    EulerEDMSamplerPyramid,
)
from gem.util import default, instantiate_from_config
from train import save_img_seq_to_video


def init_model(version_dict, load_ckpt=True):
    config = OmegaConf.load(version_dict["config"])
    model = load_model_from_config(config, version_dict["ckpt"] if load_ckpt else None)
    return model


lowvram_mode = False


def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    return model


def load_model(model):
    model.cuda()


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()


def load_model_from_config(config, ckpt=None):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_svd = torch.load(ckpt, map_location="cpu")
            # dict contains:
            # "epoch", "global_step", "pytorch-lightning_version",
            # "state_dict", "loops", "callbacks", "optimizer_states", "lr_schedulers"
            if "global_step" in pl_svd:
                print(f"Global step: {pl_svd['global_step']}")
            svd = pl_svd["state_dict"]
        elif ckpt.endswith("safetensors"):
            svd = load_safetensors(ckpt)
        else:
            raise NotImplementedError("Please convert the checkpoint to safetensors first")

        missing, unexpected = model.load_state_dict(svd, strict=False)
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    model = initial_model_load(model)
    model.eval()
    return model


def init_embedder_options(keys):
    # hardcoded demo settings, might undergo some changes in the future
    value_dict = dict()
    for key in keys:
        if key in ["fps_id", "fps"]:
            fps = 10
            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1
        elif key == "motion_bucket_id":
            value_dict["motion_bucket_id"] = 127  # [0, 511]
    return value_dict


def perform_save_locally(save_path, samples, mode, dataset_name, sample_index):
    assert mode in ["images", "grids", "videos", "videos_bbox"]
    merged_path = os.path.join(save_path, mode)
    # if exists increment
    os.makedirs(merged_path, exist_ok=True)
    samples = samples.cpu()

    if mode == "images":
        frame_count = 0
        for sample in samples:
            sample = rearrange(sample.numpy(), "c h w -> h w c")
            if "real" in save_path:
                sample = 255.0 * (sample + 1.0) / 2.0
            else:
                sample = 255.0 * sample
            image_save_path = os.path.join(
                merged_path, f"{dataset_name}_{sample_index:06}_{frame_count:04}.png"
            )
            # if os.path.exists(image_save_path):
            #     return
            Image.fromarray(sample.astype(np.uint8)).save(image_save_path)
            frame_count += 1
    elif mode == "grids":
        grid = torchvision.utils.make_grid(samples, nrow=int(samples.shape[0] ** 0.5))
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
        if "real" in save_path:
            grid = 255.0 * (grid + 1.0) / 2.0
        else:
            grid = 255.0 * grid
        grid_save_path = os.path.join(merged_path, f"{dataset_name}_{sample_index:06}.png")
        # if os.path.exists(grid_save_path):
        #     return
        Image.fromarray(grid.astype(np.uint8)).save(grid_save_path)
    elif mode == "videos":
        img_seq = rearrange(samples.numpy(), "t c h w -> t h w c")
        if "real" in save_path:
            img_seq = 255.0 * (img_seq + 1.0) / 2.0
        else:
            img_seq = 255.0 * img_seq
        video_save_path = os.path.join(merged_path, f"{dataset_name}_{sample_index:06}.mp4")
        # if os.path.exists(video_save_path):
        #     return
        save_img_seq_to_video(video_save_path, img_seq.astype(np.uint8), 10)
    elif mode == "videos_bbox":
        img_seq = rearrange(samples.numpy(), "t c h w -> t h w c")
        if "real" in save_path:
            img_seq = 255.0 * (img_seq + 1.0) / 2.0
        else:
            img_seq = 255.0 * img_seq
        video_save_path = os.path.join(merged_path, f"{dataset_name}_{sample_index:06}.mp4")
        # if os.path.exists(video_save_path):
        #     return
        save_img_seq_to_video(video_save_path, img_seq.astype(np.uint8), 10)
    else:
        raise NotImplementedError


def init_sampling(
    sampler="EulerEDMSamplerPyramid",
    guider="LinearPredictionGuider",
    discretization="EDMDiscretization",
    steps=50,
    cfg_max_scale=2.5,
    cfg_min_scale=1.0,
    num_frames=25,
):
    print(f"Using sampler: {sampler}")
    discretization_config = get_discretization(discretization)
    guider_config = get_guider(guider, cfg_max_scale, cfg_min_scale, num_frames)
    print("guider_config", guider_config)
    sampler = get_sampler(sampler, steps, discretization_config, guider_config)
    return sampler


def get_discretization(discretization):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "gem.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        }
    elif discretization == "EDMDiscretization":
        discretization_config = {
            "target": "gem.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {"sigma_min": 0.002, "sigma_max": 700.0, "rho": 7.0},
        }
    else:
        raise NotImplementedError
    return discretization_config


def get_guider(
    guider="LinearPredictionGuider", cfg_max_scale=2.5, cfg_min_scale=1.0, num_frames=25
):
    if guider == "IdentityGuider":
        guider_config = {"target": "gem.modules.diffusionmodules.guiders.IdentityGuider"}
    elif guider == "VanillaCFG":
        # scale = cfg_max_scale
        scale = 1.0

        guider_config = {
            "target": "gem.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale},
        }
    elif guider == "LinearPredictionGuider":
        max_scale = cfg_max_scale
        min_scale = cfg_min_scale

        guider_config = {
            "target": "gem.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {
                # "max_scale": max_scale,
                # "min_scale": min_scale,
                "max_scale": 1.0,
                "min_scale": 1.0,
                "num_frames": num_frames,
            },
        }
    elif guider == "TrianglePredictionGuider":
        # max_scale = cfg_max_scale
        # min_scale = 1.0

        guider_config = {
            "target": "gem.modules.diffusionmodules.guiders.TrianglePredictionGuider",
            "params": {
                "max_scale": 2.5,
                "min_scale": 1.0,
                "num_frames": num_frames,
            },
        }
    else:
        raise NotImplementedError
    return guider_config


def get_sampler(sampler, steps, discretization_config, guider_config):
    if sampler == "EulerEDMSampler":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        sampler = EulerEDMSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            verbose=False,
        )
    elif sampler == "EulerEDMSamplerPyramid":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        sampler = EulerEDMSamplerPyramid(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            verbose=False,
        )
    elif sampler == "EulerEDMSamplerDynamicPyramid":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        sampler = EulerEDMSamplerDynamicPyramid(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown sampler {sampler}")
    return sampler


def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    # hardcoded demo setups, might undergo some changes in the future
    batch = dict()
    batch_uc = dict()

    for key in keys:
        if key in value_dict:
            if key in ["fps", "fps_id", "motion_bucket_id", "cond_aug"]:
                batch[key] = repeat(
                    torch.tensor([value_dict[key]]).to(device), "1 -> b", b=math.prod(N)
                )
            elif key in ["command", "trajectory", "speed", "angle", "goal"]:
                batch[key] = repeat(value_dict[key][None].to(device), "1 ... -> b ...", b=N[0])
            elif key in ["cond_frames", "cond_frames_without_noise"]:
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
            elif key in ["img_seq", "fd_crossattn"]:
                batch[key] = value_dict[key].to(device)
            else:
                # batch[key] = value_dict[key]
                raise NotImplementedError

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_condition(model, value_dict, num_samples, force_uc_zero_embeddings, device, chunk_size=25):
    load_model(model.conditioner)
    c_total = {}
    uc_total = {}
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        batch, batch_uc = get_batch(
            list(set([x.input_key for x in model.conditioner.embedders])),
            value_dict,
            [end - start],
        )
        if value_dict.get("fd_crossattn") is not None:
            batch["fd_crossattn"] = value_dict["fd_crossattn"]
            batch_uc["fd_crossattn"] = value_dict["fd_crossattn"]
            c_chunk, uc_chunk = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                # force_cond_zero_embeddings="cond_frames_without_noise",
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
        for k in c_chunk:
            if isinstance(c_chunk[k], torch.Tensor):
                c_chunk[k], uc_chunk[k] = map(
                    lambda y: y[k][: end - start].to(device), (c_chunk, uc_chunk)
                )
                if c_chunk[k].shape[0] < end - start:
                    c_chunk[k] = c_chunk[k][[0]]
                if uc_chunk[k].shape[0] < end - start:
                    uc_chunk[k] = uc_chunk[k][[0]]
            if k not in c_total:
                c_total[k] = c_chunk[k]
                uc_total[k] = uc_chunk[k]
            else:
                c_total[k] = torch.cat([c_total[k], c_chunk[k]], dim=0)
                uc_total[k] = torch.cat([uc_total[k], uc_chunk[k]], dim=0)
    unload_model(model.conditioner)
    for k in c_total:
        c_total[k] = c_total[k][:num_samples]
        uc_total[k] = uc_total[k][:num_samples]
    return c_total, uc_total


def fill_latent(cond, length, cond_indices, device):
    latent = torch.zeros(length, *cond.shape[1:]).to(device)
    latent[cond_indices] = cond
    return latent


@torch.no_grad()
def do_sample(
    images,
    model,
    sampler,
    value_dict,
    num_rounds,
    num_frames,
    force_uc_zero_embeddings: Optional[List] = None,
    initial_cond_indices: Optional[List] = None,
    device="cuda",
):
    if initial_cond_indices is None:
        initial_cond_indices = [0]

    force_uc_zero_embeddings = default(force_uc_zero_embeddings, list())
    precision_scope = autocast

    with torch.no_grad(), precision_scope(device), model.ema_scope("Sampling"):
        c, uc = get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)

        load_model(model.first_stage_model)
        z = model.encode_first_stage(images)
        unload_model(model.first_stage_model)

        # samples_z = torch.zeros((num_rounds * (num_frames - 3) + 3, *z.shape[1:])).to(
        #     device
        # )
        samples_z = torch.zeros((num_rounds * (num_frames - 3) + 3, *z.shape[1:])).to(device)

        sampling_progress = tqdm(total=num_rounds, desc="Dreaming")

        def denoiser(x, sigma, cond, cond_mask):
            return model.denoiser(model.model, x, sigma, cond, cond_mask)

        load_model(model.denoiser)
        load_model(model.model)

        initial_cond_mask = torch.zeros(num_frames).to(device)
        prediction_cond_mask = torch.zeros(num_frames).to(device)
        initial_cond_mask[initial_cond_indices] = 1
        prediction_cond_mask[[0, 1, 2]] = 1

        noise = torch.randn_like(z)
        sample = sampler(
            denoiser,
            noise,
            cond=c,
            uc=uc,
            cond_frame=z,  # cond_frame will be rescaled when calling the sampler
            cond_mask=initial_cond_mask,
        )
        sampling_progress.update(1)
        sample[0] = z[0]
        samples_z[:num_frames] = sample

        for n in range(num_rounds - 1):
            load_model(model.first_stage_model)
            samples_x_for_guidance = model.decode_first_stage(sample[-14:])
            unload_model(model.first_stage_model)
            value_dict["cond_frames_without_noise"] = samples_x_for_guidance[[-3]]
            value_dict["cond_frames"] = sample[[-3]] / model.scale_factor

            for embedder in model.conditioner.embedders:
                if hasattr(embedder, "skip_encode"):
                    embedder.skip_encode = True
            c, uc = get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)
            for embedder in model.conditioner.embedders:
                if hasattr(embedder, "skip_encode"):
                    embedder.skip_encode = False

            filled_latent = fill_latent(sample[-3:], num_frames, [0, 1, 2], device)

            noise = torch.randn_like(filled_latent)
            sample = sampler(
                denoiser,
                noise,
                cond=c,
                uc=uc,
                cond_frame=filled_latent,  # cond_frame will be rescaled when calling the sampler
                cond_mask=prediction_cond_mask,
            )
            sampling_progress.update(1)
            samples_z[(n + 1) * (num_frames - 3) + 3 : (n + 1) * (num_frames - 3) + num_frames] = (
                sample[3:]
            )

        unload_model(model.model)
        unload_model(model.denoiser)

        load_model(model.first_stage_model)
        samples_x = model.decode_first_stage(samples_z)
        unload_model(model.first_stage_model)

        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        return samples, samples_z, images
