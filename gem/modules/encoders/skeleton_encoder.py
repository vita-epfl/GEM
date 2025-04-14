from gem.modules.encoders.modules import AbstractEmbModel
import torch
from typing import Optional
from tqdm import tqdm
import random
import numpy as np
from .pose_net import PoseNet
import torch.nn.functional as F
from einops import rearrange
import imageio


class SkeletonEncoder(AbstractEmbModel):
    def __init__(self, target_width, target_height):
        super(SkeletonEncoder, self).__init__()

        self.tokens_width = target_width // 8
        self.tokens_height = target_height // 8
        self.posenet = PoseNet(noise_latent_channels=320)  # 200k parameters

    def get_uc(self, emb):
        bs_t, l_c, c_c = emb.shape
        return torch.zeros(
            bs_t,
            l_c,
            320,
            self.tokens_height,
            self.tokens_width,
            1,
            1,
            device=emb.device,
        )

    def forward(
        self,
        rendered_poses: torch.Tensor,
        time_idxs: Optional[torch.Tensor] = None,
        force_cond_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # pixel_values: (bs, 3, h, w)
        rendered_poses = torch.nan_to_num(
            rendered_poses, nan=0.0, posinf=0.0, neginf=0.0
        )
        out = self.posenet(rendered_poses / 255.0)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # interpolate the output to the target size
        out = F.interpolate(
            out,
            (self.tokens_height, self.tokens_width),
            mode="bilinear",
            align_corners=False,
        )

        # add 3 dimensions of 1s to make sure output has 7 dimensions
        out = out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        assert out.ndim == 7
        return out
