import torch
from omegaconf import DictConfig
from gem.modules.encoders.modules import AbstractEmbModel
from typing import Optional
from dataclasses import dataclass
from genie.latent_action_model import LatentActionModel
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
from typing import Tuple, Dict
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms as T
import math
import copy
import numpy as np

ImageNet_Norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def crop_image_for_dino_input(x: torch.Tensor):
    """Crop the image to be divisible by 14 :param x: [(bl) c h w] :return: [(bl) c h w]"""
    h, w = x.size(-2), x.size(-1)
    new_h = h - (h % 14)
    new_w = w - (w % 14)
    return x[..., :new_h, :new_w]


@torch.no_grad()
def get_dino_features(dino, x: torch.Tensor, n: int = 1):
    """Autocrops the image to be divisible by 14, and then gets the features from the DINO model.

    Args:
        x: [(bl) c h w]
        n: number of layers from the last

    Returns:
        [(bl) d h w]
    """
    global ImageNet_Norm
    x_dino = crop_image_for_dino_input(x)
    assert x_dino.size(1) == 3, "DINOv2 model expects 3 channels"
    ImageNet_Norm = ImageNet_Norm.to(x_dino.device)
    x_dino = ImageNet_Norm((x_dino + 1) / 2.0)
    dino_feats = dino.get_intermediate_layers(x_dino, n=n, reshape=True)
    dino_feats = torch.cat(dino_feats, dim=1)
    return dino_feats, x_dino


def create_blueprints(n_embd, n_head, d_head, n_rep):
    ENC_BLUEPRINT = (
        (
            "space-time_attn",
            {
                "n_rep": n_rep,
                "n_embd": n_embd,
                "n_head": n_head,
                "d_head": d_head,
            },
        ),
        (
            "spacetime_downsample",
            {
                "in_channels": n_embd,  # Adjusted to match n_embd
                "kernel_size": 3,
                "time_factor": 1,
                "space_factor": 2,
            },
        ),
        (
            "space-time_attn",
            {
                "n_rep": n_rep,
                "n_embd": n_embd,
                "n_head": n_head,
                "d_head": d_head,
            },
        ),
        (
            "spacetime_downsample",
            {
                "in_channels": n_embd,  # Adjusted to match n_embd
                "kernel_size": 3,
                "time_factor": 1,
                "space_factor": 2,
            },
        ),
        (
            "space-time_attn",
            {
                "n_rep": n_rep,
                "n_embd": n_embd,
                "n_head": n_head,
                "d_head": d_head,
            },
        ),
    )

    DEC_BLUEPRINT = (
        (
            "space-time_attn",
            {
                "n_rep": n_rep,
                "n_embd": n_embd,
                "n_head": n_head,
                "d_head": d_head,
                "has_ext": (True, True),
                # 'time_attn_kw'  : {'key_dim' : 8},
            },
        ),
        (
            "spacetime_upsample",
            {
                "in_channels": n_embd,
                "kernel_size": 3,
                "time_factor": 1,
                "space_factor": 2,
            },
        ),
        (
            "space-time_attn",
            {
                "n_rep": n_rep,
                "n_embd": n_embd,
                "n_head": n_head,
                "d_head": d_head,
                "has_ext": (True, False),
            },
        ),
        (
            "spacetime_upsample",
            {
                "in_channels": n_embd,
                "kernel_size": 3,
                "time_factor": 1,
                "space_factor": 2,
            },
        ),
        (
            "space-time_attn",
            {
                "n_rep": n_rep,
                "n_embd": n_embd,
                "n_head": n_head,
                "d_head": d_head,
                "has_ext": (True, False),
            },
        ),
    )

    return ENC_BLUEPRINT, DEC_BLUEPRINT


class ActionEncoder(AbstractEmbModel):
    def __init__(
        self,
        dino_version: str,
        num_frames: int,
        target_width: int,
        target_height: int,
        downsample_factor: int,  # Number of times tokens are downsampled (spatially, per dimension)
        inp_channels: int,
        n_rep: int,
        n_embd: int,  # Basically model width
        n_head: int,  # Number of heads in the attention mechanism
        d_head: int,  # Dimension of the head
        n_codebook: int,  # Number of entries in the codebook
        d_codebook: int,  # Codebook dimension
        ucg_prob: float,  # Probability of unconditional generation
        token_mask_prob: float,  # Probability of each individual token being masked
    ):
        super().__init__()

        ENC_BLUEPRINT, DEC_BLUEPRINT = create_blueprints(
            n_embd, n_head=n_head, d_head=d_head, n_rep=n_rep
        )

        self.enc_desc = copy.deepcopy(ENC_BLUEPRINT)
        self.dec_desc = copy.deepcopy(DEC_BLUEPRINT)

        self.d_codebook = d_codebook
        self.n_embd = n_embd
        self.ucg_prob = ucg_prob
        self.token_mask_prob = token_mask_prob
        self.inp_channels = inp_channels
        self.n_codebook = n_codebook

        self.dino_version = dino_version
        self.target_width = target_width
        self.target_height = target_height

        self.downsample_factor = downsample_factor
        PATCH_SIZE = 8
        self.tokens_width = target_width // PATCH_SIZE
        self.tokens_height = target_height // PATCH_SIZE
        self.num_frames = num_frames

        self.model = LatentActionModel(
            self.enc_desc,
            self.dec_desc,
            d_codebook=self.d_codebook,
            inp_channels=self.inp_channels,
            inp_shape=(self.tokens_height, self.tokens_width),
            ker_size=3,
            n_embd=self.n_embd,
            n_codebook=self.n_codebook,
            max_timesteps=num_frames,
        )

        # We output T-1 actions, so we have fixed embeddings for the first frame
        self.frame0_actions = nn.Parameter(
            torch.randn(
                1,
                self.d_codebook,
                1,
                self.tokens_height // self.downsample_factor,
                self.tokens_width // self.downsample_factor,
            )
            * 0.01
        )

        self.unconditional_condition = nn.Parameter(
            torch.randn(
                1,
                self.d_codebook,
                1,
                self.tokens_height // self.downsample_factor,
                self.tokens_width // self.downsample_factor,
            )
            * 0.01
        )

        print(
            f"ActionEncoder parameter count: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M"
        )
        assert (
            int(round(math.sqrt(self.model.dec_fact))) == downsample_factor
        ), "Downsample factor must be the square root of the decoder factor"

    def get_unconditional_condition(self, BT: int):
        B = BT // self.num_frames
        # Repeat batch dimension and time dimension (B, C, T, H, W)
        unconditional_condition = self.unconditional_condition.repeat(
            B, 1, self.num_frames, 1, 1
        )
        return unconditional_condition

    def forward(
        self,
        encoded_frames: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """
        Encodes the action information from the input pixel values.
        BT = Batch size * Time size (number of frames per sample)

        Args:
            encoded_frames (torch.Tensor): Shape (BT, 3, H, W)

        Returns:
            torch.Tensor: _description_
        """
        if np.random.rand() < self.ucg_prob:
            extra_cond = self.get_unconditional_condition(encoded_frames.size(0))
            extra_cond = rearrange(extra_cond, "b c t h w -> (b t) c h w")
            return extra_cond, None, None, None

        b = encoded_frames.size(0) // self.num_frames
        input_tokens = encoded_frames.detach()  # just to be sure
        input_tokens = rearrange(
            input_tokens, "(b t) c h w -> b c t h w", t=self.num_frames, b=b
        )
        _, act_id, act_loss, act_losses, bottleneck_act_quantized = self.model(
            input_tokens
        )
        b = bottleneck_act_quantized.size(0)
        frame0_actions = self.frame0_actions.repeat(b, 1, 1, 1, 1)
        # Apply random masking to act_emb
        bottleneck_act_quantized = torch.cat(
            [frame0_actions, bottleneck_act_quantized], dim=2
        )
        mask = torch.rand_like(bottleneck_act_quantized) < self.token_mask_prob
        bottleneck_act_quantized = self.get_unconditional_condition(
            encoded_frames.size(0)
        ) * mask + bottleneck_act_quantized * (~mask)
        bottleneck_act_quantized = rearrange(
            bottleneck_act_quantized, "b c t h w -> (b t) c h w"
        )
        return bottleneck_act_quantized, act_id, act_loss, act_losses
