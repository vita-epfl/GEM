"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from typing import Dict, Union

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from gem.modules.diffusionmodules.sampling_utils import to_d
from gem.util import append_dims, default, instantiate_from_config


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(guider_config)
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, cond_mask, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, cond_mask, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling Setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EulerEDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(
        self, sigma, next_sigma, denoiser, x, cond, cond_mask=None, uc=None, gamma=0.0
    ):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, cond_mask, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def __call__(
        self,
        denoiser,
        x,  # x is randn
        cond,
        uc=None,
        cond_frame=None,
        cond_mask=None,
        num_steps=None,
    ):

        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        replace_cond_frames = cond_mask is not None and cond_mask.any()

        for i in self.get_sigma_gen(num_sigmas):
            if replace_cond_frames:
                x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(
                    cond_mask, cond_frame.ndim
                )
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                cond_mask,
                uc,
                gamma,
            )
        if replace_cond_frames:
            x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(
                cond_mask, cond_frame.ndim
            )
        return x


class EulerSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(
        self, sigma, next_sigma, denoiser, x, cond, cond_mask=None, uc=None, gamma=0.0
    ):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, cond_mask, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def __call__(
        self,
        denoiser,
        x,  # x is randn
        cond,
        uc=None,
        cond_frame=None,
        cond_mask=None,
        num_steps=None,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        replace_cond_frames = cond_mask is not None and cond_mask.any()

        for i in self.get_sigma_gen(num_sigmas):
            if replace_cond_frames:
                x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(
                    cond_mask, cond_frame.ndim
                )
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                cond_mask,
                uc,
                gamma,
            )
        if replace_cond_frames:
            x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(
                cond_mask, cond_frame.ndim
            )
        return x


class EulerEDMSamplerPyramid(SingleStepDiffusionSampler):
    """
    EulerEDMSamplerPyramid is a diffusion sampler that uses a pyramid scheduling matrix
    """

    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.sampling_timesteps = 100

    def _generate_pyramid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float, sigmas
    ):
        # sechduling matrix used for auto-regressive sampling
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1

        min_sigma = sigmas[-1].cpu().numpy()
        scheduling_matrix = (
            np.ones((height, horizon), dtype=np.float32) * min_sigma + 0.02
        )
        for t in range(horizon):
            for m in range(height):
                if t > m:
                    scheduling_matrix[m, t] = sigmas[0]
                elif (m - t) >= sigmas.shape[0]:
                    scheduling_matrix[m, t] = sigmas[-1]
                else:
                    scheduling_matrix[m, t] = sigmas[m - t]

        return scheduling_matrix

    def sampler_step(
        self, sigma, next_sigma, denoiser, x, cond, cond_mask=None, uc=None, gamma=0.0
    ):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, cond_mask, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def __call__(
        self,
        denoiser,
        x,  # x is randn
        cond,
        uc=None,
        cond_frame=None,
        cond_mask=None,
        num_steps=None,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        self.sampling_timesteps = sigmas.shape[0] - 1
        replace_cond_frames = cond_mask is not None and cond_mask.any()
        scheduling_matrix = (
            self._generate_pyramid_scheduling_matrix(x.shape[0], 1, sigmas) + 0.001
        )
        scheduling_matrix = torch.from_numpy(scheduling_matrix).to(self.device)

        for i in tqdm(range(scheduling_matrix.shape[0] - 1)):
            if replace_cond_frames:
                x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(
                    cond_mask, cond_frame.ndim
                )

            gamma = 0.0
            x = self.sampler_step(
                s_in * (scheduling_matrix[i]),
                s_in * (scheduling_matrix[i + 1]),
                denoiser,
                x,
                cond,
                cond_mask,
                uc,
                gamma,
            )
        if replace_cond_frames:
            x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(
                cond_mask, cond_frame.ndim
            )

        del scheduling_matrix
        torch.cuda.empty_cache()

        return x


class EulerEDMSamplerDynamicPyramid(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.chunk_size = 25
        self.clip_noise = 20.0
        self.scaling_factor = 0.18215

    def _generate_dybamic_pyramid_scheduling_matrix(
        self,
        horizon: int,
        uncertainty_scale: float,
        sigmas,
        current_mat=None,
        current_m=None,
    ):
        min_sigma = sigmas[-1].cpu().numpy()

        if current_mat is not None and current_m is not None:
            height = current_m + sigmas.shape[0] + 1
            scheduling_matrix = current_mat
            extra_rows = (
                torch.ones((height, horizon), dtype=torch.float32) * min_sigma
            ).to(self.device)
            scheduling_matrix = torch.cat((scheduling_matrix, extra_rows), dim=0)
            new_col = (
                np.ones((scheduling_matrix.shape[0]), dtype=np.float32) * min_sigma
            )
            scheduling_matrix = scheduling_matrix[:, 1:]
            new_col = torch.from_numpy(new_col).to(self.device)
            new_col[current_m : current_m + sigmas.shape[0]] = sigmas
            scheduling_matrix = torch.cat([scheduling_matrix, new_col.unsqueeze(1)], 1)
            return scheduling_matrix

        scale = self.sampling_timesteps // horizon
        height = (
            self.sampling_timesteps + int((scale * horizon - 1) * uncertainty_scale) + 1
        )
        scheduling_matrix = np.ones((height, horizon), dtype=np.float32) * min_sigma

        for m in range(height):
            for t in range(horizon):
                if scale * t > m:
                    scheduling_matrix[m, t] = sigmas[0]
                elif (m - scale * t) >= scale * horizon:
                    scheduling_matrix[m, t] = sigmas[-1]
                else:
                    scheduling_matrix[m, t] = sigmas[m - scale * t]


        return scheduling_matrix

    def sampler_step(
        self, sigma, next_sigma, denoiser, x, cond, cond_mask=None, uc=None, gamma=0.0
    ):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, cond_mask, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def __call__(
        self,
        denoiser,
        x,  # x is randn
        cond,
        uc=None,
        cond_frame=None,
        cond_mask=None,
        num_steps=None,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        sigmas = sigmas + 0.002
        self.sampling_timesteps = sigmas.shape[0] - 1

        replace_cond_frames = cond_mask is not None and cond_mask.any()

        scheduling_matrix = self._generate_dybamic_pyramid_scheduling_matrix(
            self.chunk_size, 1, sigmas
        )
        scheduling_matrix = torch.from_numpy(scheduling_matrix).to(self.device)

        og_x = x
        x_pred = torch.zeros_like(og_x)

        start_frame = 0
        num_frames = og_x.shape[0]
        x = og_x[start_frame : start_frame + self.chunk_size]
        i = 0
        new_ref = None
        with tqdm(desc="Sampling") as pbar:
            while scheduling_matrix[i, -1] != sigmas[-1]:
                end_frame = start_frame + self.chunk_size

                current_cond_frame = cond_frame[start_frame:end_frame]
                current_mask = cond_mask[start_frame:end_frame]

                current_s_in = s_in[start_frame:end_frame]
                current_cond = {}
                for key, value in cond.items():
                    if value.shape[0] >= x.shape[0]:
                        current_cond[key] = value[start_frame:end_frame]
                    else:
                        current_cond[key] = value

                    if start_frame > 0 and key == "concat" and new_ref is not None:
                        cond_x = new_ref
                        cond_x = cond_x.repeat(current_cond[key].shape[0], 1, 1, 1)
                        current_cond[key] = cond_x

                current_uc = {}
                for key, value in uc.items():
                    if value.shape[0] >= x.shape[0]:
                        current_uc[key] = value[start_frame:end_frame]
                    else:
                        current_uc[key] = value

                if replace_cond_frames:
                    x = x * append_dims(
                        1 - current_mask, x.ndim
                    ) + current_cond_frame * append_dims(
                        current_mask, current_cond_frame.ndim
                    )

                gamma = 0.0

                x = self.sampler_step(
                    current_s_in * (scheduling_matrix[i]),
                    current_s_in * (scheduling_matrix[i + 1]),
                    denoiser,
                    x,
                    current_cond,
                    current_mask,
                    current_uc,
                    gamma,
                )
                if scheduling_matrix[i, 0] == sigmas[-1]:
                    x_pred[start_frame:end_frame] = x
                    new_ref = x_pred[start_frame][:4].unsqueeze(0)
                    new_ref = new_ref / self.scaling_factor
                    if start_frame + self.chunk_size < num_frames:
                        x = torch.cat(
                            [x[1:], og_x[start_frame + self.chunk_size].unsqueeze(0)],
                            dim=0,
                        )
                        scheduling_matrix = (
                            self._generate_dybamic_pyramid_scheduling_matrix(
                                x.shape[0], 1, sigmas, scheduling_matrix, i + 1
                            )
                        )
                        start_frame += 1
                i += 1
                pbar.update(1)
        x_pred[start_frame:] = x
        x = x_pred
        if replace_cond_frames:
            x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(
                cond_mask, cond_frame.ndim
            )

        del scheduling_matrix
        torch.cuda.empty_cache()

        return x


class EulerEDMSamplerDynamicPyramidFM(EulerEDMSamplerDynamicPyramid):
    """
    EulerEDMSamplerDynamicPyramidFM is a flow matching sampler that uses a dynamic pyramid scheduling matrix
    """

    def _generate_dynamic_pyramid_scheduling_matrix(
        self,
        horizon: int,
        uncertainty_scale: float,
        sigmas,
        current_mat=None,
        current_m=None,
    ):
        min_sigma = sigmas[-1].cpu().numpy()

        if current_mat is not None and current_m is not None:
            height = current_m + sigmas.shape[0] + 1
            scheduling_matrix = current_mat
            extra_rows = (
                torch.ones((height, horizon), dtype=torch.float32) * min_sigma
            ).to(self.device)
            scheduling_matrix = torch.cat((scheduling_matrix, extra_rows), dim=0)
            new_col = (
                np.ones((scheduling_matrix.shape[0]), dtype=np.float32) * min_sigma
            )
            scheduling_matrix = scheduling_matrix[:, 1:]
            new_col = torch.from_numpy(new_col).to(self.device)
            new_col[current_m : current_m + sigmas.shape[0]] = sigmas
            scheduling_matrix = torch.cat([scheduling_matrix, new_col.unsqueeze(1)], 1)
            return scheduling_matrix

        scale = self.sampling_timesteps // horizon
        height = self.sampling_timesteps + int(
            (scale * horizon - 1) * uncertainty_scale
        )  # + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.float32) * min_sigma

        for m in range(height):
            for t in range(horizon):
                if scale * t > m:
                    scheduling_matrix[m, t] = sigmas[0]
                elif (m - scale * t) >= scale * horizon:
                    scheduling_matrix[m, t] = sigmas[-1]
                else:
                    scheduling_matrix[m, t] = sigmas[m - scale * t]

        return scheduling_matrix

    def sampler_step(
        self, sigma, next_sigma, denoiser, x, cond, cond_mask=None, uc=None, gamma=0.0
    ):
        sigma_hat = sigma  # * (gamma + 1.0)

        v = self.denoise(x, denoiser, sigma_hat, cond, cond_mask, uc)

        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, v, dt)
        return euler_step

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            do_append_zero=False,
            device=self.device,
        )
        uc = default(uc, cond)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def __call__(
        self,
        denoiser,
        x,  # x is randn
        cond,
        uc=None,
        cond_frame=None,
        cond_mask=None,
        num_steps=None,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        sigmas = sigmas
        self.sampling_timesteps = sigmas.shape[0]  # - 1

        replace_cond_frames = cond_mask is not None and cond_mask.any()

        scheduling_matrix = self._generate_dynamic_pyramid_scheduling_matrix(
            self.chunk_size, 1, sigmas
        )
        scheduling_matrix = torch.from_numpy(scheduling_matrix).to(self.device)

        og_x = x
        x_pred = torch.zeros_like(og_x)

        start_frame = 0
        num_frames = og_x.shape[0]
        x = og_x[start_frame : start_frame + self.chunk_size]
        i = 0
        new_ref = None
        with tqdm(desc="Sampling") as pbar:
            while scheduling_matrix[i, -1] != sigmas[-1]:
                end_frame = start_frame + self.chunk_size

                current_cond_frame = cond_frame[start_frame:end_frame]
                current_mask = cond_mask[start_frame:end_frame]

                current_s_in = s_in[start_frame:end_frame]
                current_cond = {}
                for key, value in cond.items():
                    if value.shape[0] >= x.shape[0]:
                        current_cond[key] = value  # [start_frame:end_frame]
                    else:
                        current_cond[key] = value

                    if start_frame > 0 and key == "concat" and new_ref is not None:
                        cond_x = new_ref
                        cond_x = cond_x.repeat(current_cond[key].shape[0], 1, 1, 1)
                        current_cond[key] = cond_x

                current_uc = {}
                for key, value in uc.items():
                    if value.shape[0] >= x.shape[0]:
                        current_uc[key] = value  # [start_frame:end_frame]
                    else:
                        current_uc[key] = value

                if replace_cond_frames:
                    x = x * append_dims(
                        1 - current_mask, x.ndim
                    ) + current_cond_frame * append_dims(
                        current_mask, current_cond_frame.ndim
                    )

                gamma = 0.0

                x = self.sampler_step(
                    current_s_in * (scheduling_matrix[i]),
                    current_s_in * (scheduling_matrix[i + 1]),
                    denoiser,
                    x,
                    current_cond,
                    current_mask,
                    current_uc,
                    gamma,
                )
                if scheduling_matrix[i, 0] == sigmas[-1]:
                    x_pred[start_frame] = x[0]
                    new_ref = x_pred[start_frame][:4].unsqueeze(0)
                    new_ref = new_ref / self.scaling_factor
                    
                    if start_frame + self.chunk_size < num_frames:
                        x = torch.cat(
                            [x[1:], og_x[start_frame + self.chunk_size].unsqueeze(0)],
                            dim=0,
                        )
                        scheduling_matrix = (
                            self._generate_dynamic_pyramid_scheduling_matrix(
                                x.shape[0], 1, sigmas, scheduling_matrix, i + 1
                            )
                        )
                        start_frame += 1
                i += 1
                pbar.update(1)
        x_pred[start_frame:] = x
        x = x_pred
        if replace_cond_frames:
            x = x * append_dims(1 - cond_mask, x.ndim) + cond_frame * append_dims(
                cond_mask, cond_frame.ndim
            )

        del scheduling_matrix
        torch.cuda.empty_cache()

        return x
