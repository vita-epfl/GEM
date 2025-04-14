import torch
from einops import rearrange, repeat

from gem.util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2, num_frames=25):
        self.p_mean = p_mean
        self.p_std = p_std
        self.num_frames = num_frames

    def __call__(self, n_samples, rand=None):
        bs = n_samples // self.num_frames
        rand_init = torch.randn((bs,))[..., None]
        rand_init = repeat(rand_init, "b 1 -> (b t)", t=self.num_frames)
        rand = default(rand, rand_init)
        log_sigma = self.p_mean + self.p_std * rand
        return log_sigma.exp()


class IndependantEDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2, num_frames=25):
        self.p_mean = p_mean
        self.p_std = p_std
        self.num_frames = num_frames

    def __call__(self, n_samples, rand=None):
        bs = n_samples // self.num_frames

        rand_init = torch.randn((bs, self.num_frames))
        rand_init = rearrange(rand_init, "b t -> (b t)")
        rand = default(rand, rand_init)
        log_sigma = self.p_mean + self.p_std * rand
        return log_sigma.exp()


class CustomEDMSampling:
    def __init__(
        self,
        p_mean=-1.2,
        p_std=1.2,
        num_frames=25,
        sigma_min=0.002,
        sigma_max=700,
        rho=7.0,
    ):
        self.p_mean = p_mean
        self.p_std = p_std
        self.num_frames = num_frames
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        alpha = 1.0
        beta = 1.5
        self.shift_dist = torch.distributions.Beta(alpha, beta)

    def sigma_to_t(self, sigma, sigma_min=None, sigma_max=None, rho=None):
        if sigma_min is None:
            sigma_min = self.sigma_min
        if sigma_max is None:
            sigma_max = self.sigma_max
        if rho is None:
            rho = self.rho
        t = (sigma ** (1 / rho) - sigma_max ** (1 / rho)) / (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
        )
        return t

    def t_to_sigma(self, t, sigma_min=None, sigma_max=None, rho=None):
        if sigma_min is None:
            sigma_min = self.sigma_min
        if sigma_max is None:
            sigma_max = self.sigma_max
        if rho is None:
            rho = self.rho
        sigma = (
            sigma_max ** (1 / rho)
            + t * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        return sigma

    def __call__(self, n_samples, rand=None):
        bs = n_samples // self.num_frames
        rand_init = torch.randn((bs, 1)).repeat(1, self.num_frames)
        rand_init = rearrange(rand_init, "b t -> (b t)")
        rand = default(rand, rand_init)
        log_sigma = self.p_mean + self.p_std * rand
        sigma = log_sigma.exp()
        rand_num = torch.rand((1)).item()
        rho = (torch.rand(1) * 6 + 1).clamp(min=1.0, max=7.0)
        sigma_max = 100 + torch.rand(1) * 600

        if rand_num < 0.95:
            if rand_num < 0.15:
                chunk_size = torch.randint(1, 4, (1,)).item()
                n_chunks = self.num_frames // chunk_size
               
                t = torch.linspace(0.95, 0, n_chunks).unsqueeze(0).repeat(bs, 1)
                t = t.repeat_interleave(chunk_size, dim=1)
               
                t = torch.cat(
                    [t, torch.zeros((bs, self.num_frames - t.size(1)))], dim=1
                ).to(sigma.device)
            else:
                first_frame_sigma = rearrange(
                    sigma, "(b t) -> b t", b=bs, t=self.num_frames
                )[:, 0]
                start_t = (
                    self.sigma_to_t(first_frame_sigma, sigma_max=sigma_max, rho=rho)
                    .unsqueeze(1)
                    .clamp(min=0, max=1)
                )
                random_slope = (
                    -1 + torch.randn((bs, 1)) * (1 / self.num_frames)
                ).clamp(min=-1, max=0)
                if True:  # True:
                    shift = self.shift_dist.sample((bs, 1)).to(first_frame_sigma.device)
                else:
                    shift = torch.zeros((bs, 1)).to(first_frame_sigma.device)
                t_linspace = (
                    torch.linspace(0, 1, self.num_frames)
                    .to(first_frame_sigma.device)
                    .unsqueeze(0)
                )  # Shape: (1, num_frames)
                t = (start_t + (t_linspace - shift) * random_slope).clamp(min=0, max=1)
            # Add randomness to t
            if True:
                time_noise = torch.randn((bs, self.num_frames)) * (3 / self.num_frames)
                t = (t + time_noise).clamp(min=0, max=1)
                t = rearrange(t, "b t -> (b t)")
            # Compute sigma based on t
            sigma = self.t_to_sigma(t, rho=rho, sigma_max=sigma_max)
        return sigma

class DiscreteSampling:
    def __init__(
        self,
        discretization_config,
        num_idx,
        do_append_zero=False,
        flip=True,
        num_frames=25,
    ):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.num_frames = num_frames

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        bs = n_samples // self.num_frames
        rand_init = torch.randint(0, self.num_idx, (bs,))[..., None]
        rand_init = repeat(rand_init, "b 1 -> (b t)", t=self.num_frames)
        idx = default(rand, rand_init)
        return self.idx_to_sigma(idx)


class UniformTimeSampler:
    def __init__(self, t_min: float = 0.0, t_max: float = 1.0):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, n_samples) -> torch.Tensor:
        # Sample t uniformly from [t_min, t_max)
        t = torch.rand(1) * (self.t_max - self.t_min) + self.t_min
        t = t.repeat(n_samples)
        return t


class FMForcingSampling:
    def __init__(self, num_frames=25, rho=1):
        self.num_frames = num_frames
        self.rho = rho
        alpha = 1.0
        beta = 1.0
        self.shift_dist = torch.distributions.Beta(alpha, beta)

    def __call__(self, n_samples, rand=None):
        bs = n_samples // self.num_frames
        # rand_init = torch.randn((bs, self.num_frames))
        rand_init = torch.rand((bs, 1)).repeat(1, self.num_frames)
        rand_init = rearrange(rand_init, "b t -> (b t)")
        rand = default(rand, rand_init)
        t = rand

        random_slope = -1 + torch.randn((bs, 1)) * (1 / self.num_frames)
        shift = self.shift_dist.sample((bs, 1)).to(t.device)

        t_linspace = (
            torch.linspace(0, 1, self.num_frames).to(t.device).unsqueeze(0)
        )  # Shape: (1, num_frames)

        t = (t[0] + (t_linspace - shift) * random_slope).clamp(min=0, max=1)

        # Add randomness to t
        if True:
            time_noise = torch.randn((bs, self.num_frames)) * (2 / self.num_frames)
            t = (t + time_noise).clamp(min=0, max=1)
            t = rearrange(t, "b t -> (b t)")

        return t 


class DiffusionForcingSampling:
    def __init__(
        self,
        discretization_config,
        num_idx,
        do_append_zero=False,
        flip=True,
        num_frames=25,
    ):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.num_frames = num_frames

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        bs = n_samples // self.num_frames
        rand_init = torch.randint(0, self.num_idx, (self.num_frames, bs))
        rand_init = rearrange(rand_init, "b t -> (b t)")
        idx = default(rand, rand_init)
        return self.idx_to_sigma(idx)
