from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# from utils.torch_utils import spatial_transform
import torch.nn.functional as F
from einops import einsum, rearrange
from omegaconf import DictConfig
from torch import nn
from torchvision import transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

from gem.modules.diffusionmodules.openaimodel import Timestep
from gem.modules.encoders.modules import AbstractEmbModel
from gem.util import instantiate_from_config, visualize_dino_features

from .dino_utils import create_crop_region, drop_from_masks, spatial_transform


class DinoEncoder(AbstractEmbModel):
    def __init__(
        self,
        num_frames,
        dino_version: str,
        dino_channels: int,
        proj_channels: int,
        image_width: int,
        image_height: int,
        mode: str,  # 3d, 2d, 1d
        num_condition_tokens: int,
        num_random_condition_frames: int,
        condition_frames: List[int],
        random_crop_per_frame: bool,
        cage_crop: DictConfig,
        out_res: List[int] = None,
        mask_prob: float = 0.15,
        ucg_prob: float = 0.0,
        learned_mask: bool = True,
        num_dino_layers: int = 1,
        encoder_cfg: Optional[Dict[str, Any]] = None,
        out_patch_size: Optional[int] = None,
    ):
        super().__init__()
        self.dino_version = dino_version
        self.dino_channels = dino_channels * num_dino_layers
        self.num_dino_layers = num_dino_layers
        self.cage_crop = cage_crop
        self.num_random_condition_frames = num_random_condition_frames
        self.condition_frames = condition_frames
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_version)
        self.dino.requires_grad_(False)
        self.raft = raft_large(pretrained=True, progress=False)
        self.raft.requires_grad_(False)
        self.mask_prob = mask_prob
        self.proj_channels = proj_channels

        self.num_condition_frames = (
            len(condition_frames)
            if num_random_condition_frames == 0
            else num_random_condition_frames
        )

        self.image_width = image_width
        self.image_height = image_height
        self.mode = mode
        self.random_crop_per_frame = random_crop_per_frame
        self.num_condition_tokens = num_condition_tokens
        if out_patch_size is not None:
            assert out_res is None, "Cannot provide both out_res and out_patch_size"
            self.out_res = [
                image_height // out_patch_size,
                image_width // out_patch_size,
            ]
        else:
            assert (
                out_res is not None
            ), "Either out_res or out_patch_size must be provided"
            self.out_res = out_res
        self.imagenet_norm = T.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        assert self.num_condition_tokens <= 128, "Number of tokens should be <= 256"
        self.identity = torch.nn.Embedding(128 + 1, dino_channels)

        self.num_frames = num_frames
        self.ucg_prob = ucg_prob

        if self.proj_channels > 0:
            self.cond_projection = nn.Sequential(
                nn.Linear(
                    self.dino_channels,
                    self.proj_channels,
                ),
                nn.LayerNorm(self.proj_channels),
            )

        mask_dim = (
            self.dino_channels
        )  # if self.proj_channels == 0 else self.proj_channels
        if learned_mask:
            self.cond_masked_token = nn.Parameter(
                torch.randn(mask_dim),
                requires_grad=True,
            )
        else:
            self.cond_masked_token = torch.zeros(mask_dim)

        # self.cond_pos_emb = posemb_sincos_1d(self.num_pos_emb_tokens, self.dino_channels)
        if mode not in ["3d", "3d_id"]:
            self.cond_pos_emb = nn.Embedding(
                (320 // 14) * (576 // 14), self.dino_channels
            )
        if encoder_cfg is not None:
            self.encoder = instantiate_from_config(encoder_cfg)

        # self.final_pos_emb = posemb_sincos_1d(self.num_condition_tokens, self.dino_channels)
        if mode == "1d":
            self.time_emb = nn.Embedding(self.num_frames, self.dino_channels)

    @torch.no_grad()
    def get_features(
        self,
        target_frames: torch.Tensor,
        condition_frames: torch.Tensor,
        force_roi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Extracts features from the given target frames using DINO, optionally applying a region of interest (ROI).

        This function processes the `target_frames` of shape [b, l, c, h, w] and `condition_frames` indices
        of shape [b, 1], where `b` is the batch size, `l` is the number of frames, `c` is the number of channels,
        `h` is the height, and `w` is the width. It optionally takes a `force_roi` parameter.

        - If no `force_roi` is provided, a random crop is applied to the selected target frames, and the features
          are extracted using the DINO model.
        - If `force_roi` is provided, the features are extracted from the specified region.

        The function returns the extracted features and a corresponding mask indicating the areas where features
        were extracted. The features are zoomed out based on the context of the extraction.

        Args:
            target_frames (torch.Tensor): Input frames of shape [b, l, c, h, w].
            condition_frames (torch.Tensor): Conditioning frame indices of shape [b, 1].
            force_roi (Optional[RectangleRegion]): Region of interest for feature extraction. If None, a random crop is used.

        Returns:
            Tuple[torch.Tensor]: A tuple containing:
                - Extracted features: A tensor of shape [b, l, c, h, w].
                - Mask: A binary mask indicating where the features were extracted.
        """

        # save image
        # for i in range(target_frames.size(0)):
        #     for j in range(target_frames.size(1)):
        #         T.ToPILImage()(target_frames[i, j].cpu()).save(f"dino_features/target_frames_{i}_{j}.png")

        batch_size = target_frames.size(0)
        selected_frames = target_frames[:, condition_frames, :, :]
        num_target_frames = selected_frames.size(1)
        H, W = target_frames.size(-2), target_frames.size(-1)

        h_out, w_out = self.out_res

        if force_roi is None:
            if self.random_crop_per_frame:
                z_where = create_crop_region(
                    **self.cage_crop,
                    num_target_frames=num_target_frames * batch_size,
                    batched_lengths=True,
                    height=H,
                    width=W,
                    scale_aspect_ratio=True,
                )
            else:
                z_where = create_crop_region(
                    **self.cage_crop,
                    num_target_frames=batch_size,
                    batched_lengths=True,
                    height=H,
                    width=W,
                    scale_aspect_ratio=True,
                ).repeat(
                    num_target_frames,
                    1,
                )
                zw = int(z_where[0, 0] * W)
                zh = int(z_where[0, 1] * H)
        else:
            z_where = force_roi
        # print(z_where)
        # output is [b*t, w, h, x, y] !!!
        zw = int(z_where[0, 0] * W)
        zh = int(z_where[0, 1] * H)

        flat_transformed_target_frames = spatial_transform(
            rearrange(selected_frames, "b l c h w -> (b l) c h w"),
            z_where,
            # [batch_size * num_target_frames, 3, H, W],
            [batch_size * num_target_frames, 3, zh, zw],
            inverse=False,
            padding_mode="border",
            mode="bilinear",
        )

        flat_allowed_masks = spatial_transform(
            torch.ones(batch_size * num_target_frames, 1, h_out, w_out).to(
                selected_frames.device
            ),
            z_where,
            [batch_size * num_target_frames, 1, h_out, w_out],
            inverse=True,
            padding_mode="zeros",
            mode="nearest",
        )

        # transformed_target_frames = rearrange(
        #     flat_transformed_target_frames, "(b l) c h w -> b l c h w", b=batch_size
        # )
        allowed_masks = rearrange(
            flat_allowed_masks, "(b l) c h w -> b l c h w", b=batch_size
        ).to(torch.bool)

        transformed_target_frames = F.interpolate(
            flat_transformed_target_frames, size=(224, 224), mode="bicubic"
        )

        transformed_target_frames = self.imagenet_norm(
            transformed_target_frames / 2 + 0.5
        )

        dino_feats = self.dino.get_intermediate_layers(
            transformed_target_frames,
            # rearrange(transformed_target_frames, "b l c h w -> (b l) c h w"),
            n=self.num_dino_layers,
            reshape=True,
        )

        # visualize_dino_features(dino_feats[-1], output_dir="dino_features")
        dino_feats = torch.cat(dino_feats, dim=1)
        # print(f"num_dino_feats: {dino_feats.size(-1) * dino_feats.size(-2)}")

        dino_feats_zoomed_in = rearrange(
            dino_feats, "(b l) c h w -> b l c h w", b=batch_size
        )

        flat_feats = spatial_transform(
            rearrange(dino_feats_zoomed_in, "b l c h w -> (b l) c h w"),
            z_where,
            [batch_size * num_target_frames, dino_feats.size(2), h_out, w_out],
            inverse=True,
            padding_mode="zeros",
            mode="nearest",
        )

        # visualize_dino_features(flat_feats, output_dir="dino_features_total")
        feats = rearrange(flat_feats, "(b l) c h w -> b l c h w", b=batch_size)

        return feats, allowed_masks, dino_feats_zoomed_in

    def _add_embeddings(
        self, x: torch.Tensor, time_idxs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Adds positional and temporal embeddings to the input tensor `x`.

        Args:
            x (torch.Tensor): Input tensor of shape [b, l, c, h, w].
            time_idxs (torch.Tensor): Time indices of shape [b, 1].

        Returns:
            torch.Tensor: Output tensor with added positional and temporal embeddings.
        """
        if time_idxs is None:
            time_idxs = torch.arange(x.size(1), device=x.device)

        x = rearrange(x, "b l h w c -> b l c h w")
        b, l, c, h, w = x.size()
        x = rearrange(x, "b l c h w -> (b l) (h w) c")
        batch_size, seq_len, _ = x.shape
        pos_indices = torch.arange(
            seq_len, device=x.device
        )  # [0, 1, 2, ..., seq_len-1]
        x = x + self.cond_pos_emb(pos_indices).unsqueeze(0)

        x = rearrange(x, "(b l) (h w) c -> b l (h w) c", b=b, l=l, h=h, w=w)

        if hasattr(self, "time_emb"):
            time_idxs = time_idxs.unsqueeze(0).repeat(b, 1)
            temp_emb = self.time_emb(time_idxs)
            x = x + temp_emb.unsqueeze(2)

        x = rearrange(x, "b l (h w) c -> b l h w c", h=h, w=w)

        return x

    def pad_to_num_frames(
        self, x: torch.Tensor, time_idxs: torch.Tensor
    ) -> torch.Tensor:
        """
        Pads the input tensor `x` to have `self.num_frames` frames by adding masked tokens.

        Args:
            x (torch.Tensor): Input tensor of shape [b, l, c, h, w].
            time_idxs (torch.Tensor): Time indices of shape [b, 1].

        Returns:
            torch.Tensor: Padded tensor with masked tokens. [b, self.num_frames,]
        """
        bs_t, l_c, h_c, w_c, _ = x.shape
        x = rearrange(x, "b l h w c -> b l (h w) c", h=h_c, w=w_c)
        out_feats = (
            self.cond_masked_token.to(x.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(bs_t, self.num_frames, h_c * w_c, 1)
        ).to(x.device)
        batch_indices = (
            torch.arange(bs_t).unsqueeze(1).expand(-1, l_c).to(time_idxs.device)
        )
        out_feats[batch_indices, time_idxs, :, :] = x
        out_feats = rearrange(out_feats, "b l (h w) c -> b l h w c", h=h_c, w=w_c)

        return out_feats

    @torch.no_grad()
    def get_demo_input(
        self,
        pixel_values: torch.Tensor,
        at_where: torch.Tensor,
        at_when: torch.Tensor,
        to_where: torch.Tensor,
        to_when: torch.Tensor,
        num_total_frames: int,
        num_tokens: int = 0,
        ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        this creates only one conditioning, not batched

        pixel_values: [num_frames, c, h, w], images to extract conditioning from

        at_when: [num_frames] from which frame of pixel_values to extract conditioning
        at_location: [num_frames, 4]  w, h, x, y of the region to extract conditioning from

        to_when: [num_frames] to which time index to put the tokens
        to_location: [num_frames, 4] w, h, x, y of the region to put the tokens

        num_total_frames: total number of frames in generated video
        """

        assert (
            at_when.ndim == 1 and to_when.ndim == 1
        ), "at_when and to_when should be 1D"
        assert (
            at_where.ndim == 2 and to_where.ndim == 2
        ), "at_where and to_where should be 2D"
        assert (to_when >= 0).all() and (
            to_when < num_total_frames
        ).all(), "to_when should be in the range of num_total_frames"
        assert (at_when >= 0).all() and (
            at_when < pixel_values.size(0)
        ).all(), "at_when should be in the range of provided frames"

        if (
            pixel_values.size(-1) != self.image_width
            or pixel_values.size(-2) != self.image_height
        ):
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.image_height, self.image_width),
                mode="bilinear",
            )

        pixel_values = pixel_values.unsqueeze(0)  # [1, num_frames, c, h, w]

        b, l, c, height, width = pixel_values.size()

        at_where[:, 0] = (
            (at_where[:, 0] * width).int() - ((at_where[:, 0] * width).int() % 14)
        ) / width
        at_where[:, 1] = (
            (at_where[:, 1] * height).int() - ((at_where[:, 1] * height).int() % 14)
        ) / height

        _, _, features_zoomed_in = self.get_features(
            pixel_values,
            condition_frames=at_when,
            force_roi=at_where,
        )
        b, l, c, h, w = features_zoomed_in.size()

        # print("shared cond_feats")
        # features_zoomed_in = features_zoomed_in[:, 0].repeat(1, l, 1, 1, 1)

        flat_feats = spatial_transform(
            rearrange(features_zoomed_in, "b l c h w -> (b l) c h w"),
            to_where,
            [b * l, c, self.out_res[0], self.out_res[1]],
            inverse=True,
            padding_mode="zeros",
            mode="nearest",
        )

        # jfeatures = rearrange(flat_feats, "(b l) c h w -> b l h w c", b=b)
        cond_feats = rearrange(flat_feats, "(b l) c h w -> b l h w c", b=b)

        if self.mode not in ["3d", "3d_id"]:
            features = self._add_embeddings(cond_feats, to_when)

        flat_valid_masks = spatial_transform(
            torch.ones(b * l, 1, self.out_res[0], self.out_res[1]).to(
                cond_feats.device
            ),
            to_where,
            [b * l, 1, self.out_res[0], self.out_res[1]],
            inverse=True,
            padding_mode="zeros",
            mode="nearest",
        ).to(torch.bool)

        b, l, h, w, _ = cond_feats.shape

        cond_feats = rearrange(cond_feats, "b l h w c -> b l (h w) c")
        valid_masks = rearrange(flat_valid_masks, "b l h w -> (b l) (h w)")

        valid_masks = drop_from_masks(valid_masks, num_tokens)
        valid_masks = rearrange(valid_masks, "(b l) n -> b l n", b=b, l=l)
        valid_masks = valid_masks.unsqueeze(-1)  # b, l, n, 1
        cond_feats = cond_feats * valid_masks
        cond_feats = rearrange(cond_feats, "b l n c -> b l n 1 c")

        if ids is not None:
            identities = {}
            for id in torch.unique(ids):
                # identity = self.identity(torch.randperm(self.num_condition_tokens).to(flat_feats.device)) # or arange
                if id == 0:
                    identity = self.identity(
                        torch.zeros_like(
                            # torch.randperm(self.num_condition_tokens).to(flat_feats.device)
                            torch.arange(self.num_condition_tokens).to(
                                flat_feats.device
                            )
                        )
                    )
                else:
                    identity = self.identity(
                        torch.randperm(self.num_condition_tokens).to(flat_feats.device)
                    )  # or arange
                # identity = self.identity(torch.zeros_like(torch.randperm(self.num_condition_tokens).to(flat_feats.device)))
                # identity = self.identity(
                #        torch.arange(self.num_condition_tokens).to(flat_feats.device) + 1
                #    )  # or arange
                identities[id.item()] = identity

            # cond_feats = rearrange(cond_feats, "1 l (h w) 1 c -> l h w c", h=h, w=w)
            cond_feats = rearrange(
                cond_feats,
                " 1 t n 1 c -> t n c",
            )  # Shape: [b, c, h, w]
            if type(ids) == int:
                ids = [ids]
            for j in range(len(ids)):
                non_zero_mask = (cond_feats[j] != 0).all(dim=1)
                selected_feats = cond_feats[j][non_zero_mask]
                # identities = self.identity(torch.randperm(selected_feats.size(0)).to(selected_feats.device)) # or arange
                temp = (
                    selected_feats + identities[ids[j].item()][: selected_feats.size(0)]
                )  # #[]
                cond_feats[j][non_zero_mask] = temp

            cond_feats = rearrange(cond_feats, "t n c -> 1 t n 1 c")

        cond_feats = self.pad_to_num_frames(
            cond_feats, to_when
        )  # [b, self.num_frames, h, w, c]

        cond_feats = rearrange(cond_feats, "b l (h w) 1 c -> b l h w c", h=h, w=w)
        # cond_feats_ = rearrange(cond_feats, "b l h w c -> (b l) c h w")
        # visualize_dino_features(cond_feats_.detach().cpu(), output_dir="dino_features")
        cond_feats = rearrange(cond_feats, "b l h w c -> (b l) c h w", h=h, w=w)

        return cond_feats.view(*cond_feats.shape, 1, 1)

    @torch.no_grad()
    def get_demo_input2(
        self,
        pixel_values: torch.Tensor,
        source_idxs: torch.Tensor,
        source_crops: torch.Tensor,
        source_masks: torch.Tensor,
        target_idxs: torch.Tensor,
        target_masks: torch.Tensor,
        num_total_frames: int,
        ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pixel_values = pixel_values.unsqueeze(0)  # [1, num_frames, c, h, w]
        # saafe images
        # for i in range(pixel_values.size(0)):
        #     for j in range(pixel_values.size(1)):
        #         T.ToPILImage()(pixel_values[i, j].cpu()).save(f"dino_features/pixel_values_{i}_{j}.png")

        b, l, c, height, width = pixel_values.size()

        features, _, _ = self.get_features(
            pixel_values,
            condition_frames=source_idxs,
            force_roi=source_crops,
        )
        b, l, c, h, w = features.size()

        if ids is not None:
            identities = {}
            for id in torch.unique(ids):
                # identity = self.identity(torch.randperm(self.num_condition_tokens).to(flat_feats.device)) # or arange
                if id == 0:
                    identity = self.identity(
                        torch.zeros_like(
                            # torch.randperm(self.num_condition_tokens).to(flat_feats.device)
                            torch.arange(self.num_condition_tokens).to(
                                pixel_values.device
                            )
                        )
                    )
                else:
                    identity = self.identity(
                        torch.randperm(self.num_condition_tokens).to(
                            pixel_values.device
                        )
                    )  # or arange
                # identity = self.identity(torch.zeros_like(torch.randperm(self.num_condition_tokens).to(flat_feats.device)))
                # identity = self.identity(
                #        torch.arange(self.num_condition_tokens).to(flat_feats.device) + 1
                #    )  # or arange
                identities[id.item()] = identity

            if isinstance(ids, int):
                ids = [ids]

        features_ = rearrange(features, "1 l c h w ->  l h w c")
        placeholder = torch.zeros((num_total_frames, h, w, c)).to(pixel_values.device)

        for j in range(len(target_idxs)):
            selected_feats = features_[j, source_masks[j]]
            if ids is not None:
                selected_feats = (
                    selected_feats + identities[ids[j].item()][: selected_feats.size(0)]
                )
            # features_[j] = torch.zeros_like(features_[j])
            placeholder[target_idxs[j], target_masks[j]] = selected_feats
            # features_[j, target_masks[j]] = selected_feats + identities[ids[j].item()][: selected_feats.size(0)]

        cond_feats = rearrange(placeholder, "l h w c -> 1 l h w c")

        # cond_feats = self.pad_to_num_frames(cond_feats, target_idxs)  # [b, self.num_frames, h, w, c]
        cond_feats = rearrange(cond_feats, "b l h w c -> (b l) c h w", h=h, w=w)

        return cond_feats.view(*cond_feats.shape, 1, 1)

    def get_uc(self, emb):
        bs_t, l_c, c_c = emb.shape
        return (
            self.cond_masked_token.unsqueeze(0)
            .unsqueeze(0)
            .repeat(bs_t, l_c, 1)
            .to(emb.dtype)
        )

    def _get_masked_token(self, batch_size):
        """Return a tensor filled with the masked token."""
        cond_masked_token = self.cond_masked_token.view(1, 1, -1, 1, 1, 1)
        cond_masked_token = cond_masked_token.expand(
            batch_size,
            self.num_condition_tokens,
            -1,
            -1,
            -1,
            -1,
        )
        return cond_masked_token

    def _resize_images(self, pixel_values):
        """Resize images to the required height and width."""
        if (pixel_values.size(-2) != self.image_height) or (
            pixel_values.size(-1) != self.image_width
        ):
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.image_height, self.image_width),
                mode="bilinear",
            )
        return pixel_values

    @torch.no_grad()
    def _extract_cond_feats(self, pixel_values, time_idxs, force_cond_feats):
        """Extract conditional features from pixel values or use forced features."""
        cond_feats = force_cond_feats
        if force_cond_feats is None:
            bs, c, h, w = pixel_values.size()
            bs_t = bs // self.num_frames
            pixel_values = pixel_values.reshape(bs_t, self.num_frames, c, h, w)

            if time_idxs is None and self.num_random_condition_frames != 0:
                # get random amount of condition frames between 1 and self.num_frames // 2
                if self.num_random_condition_frames < 1:
                    num_condition_frames = torch.randint(
                        # 1, self.num_frames, (1,), device=pixel_values.device
                        1,
                        int(self.num_frames // 2.5),
                        (1,),
                        device=pixel_values.device,
                    )
                else:
                    num_condition_frames = self.num_condition_frames
                time_idxs = (
                    torch.randperm(self.num_frames, device=pixel_values.device)[
                        :num_condition_frames
                    ]
                    .sort()
                    .values
                )

            else:
                time_idxs = torch.tensor(self.condition_frames).to(pixel_values.device)

            cond_feats, valid_masks, _ = self.get_features(
                pixel_values, condition_frames=time_idxs
            )

        cond_feats = rearrange(cond_feats, "b l c h w -> b l h w c").clone()
        valid_masks = rearrange(
            valid_masks, "b l 1 h w -> b l h w", b=bs_t, l=len(time_idxs)
        )
        return cond_feats, valid_masks, time_idxs

    def _insert_masked_tokens(self, cond_feats, valid_masks):
        b, l, h, w, c = cond_feats.shape
        cond_feats = rearrange(cond_feats, "b l h w c -> b l (h w) c")
        valid_masks = rearrange(valid_masks, "b l h w -> b l (h w)")
        cond_feats[~valid_masks] = (
            self.cond_masked_token.to(cond_feats.dtype)
            .unsqueeze(0)
            .to(cond_feats.device)
        )
        valid_masks = rearrange(valid_masks, "b l n -> (b l) n", b=b, l=l)
        cond_feats = rearrange(cond_feats, "b l (h w) c -> b l h w c", h=h, w=w)

        return cond_feats

    def _process_1d_mode(self, cond_feats, valid_masks):
        """Process conditional features in 1D mode."""
        cond_feats = cond_feats[valid_masks]
        cond_feats = cond_feats.view(cond_feats.size(0), -1, cond_feats.size(-1))

        # Randomly sample condition tokens
        indices = torch.randperm(cond_feats.size(1))[: self.num_condition_tokens]
        cond_feats = cond_feats[:, indices, :]

        # Apply masking probability
        if self.mask_prob > 0:
            mask = torch.bernoulli(
                torch.full(
                    cond_feats.size()[:2], self.mask_prob, device=cond_feats.device
                )
            ).bool()
            cond_feats[mask] = 0

        # cond_feats = self.cond_projection(cond_feats)
        return cond_feats.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def _process_2d_mode(
        self, cond_feats, valid_masks, time_idxs, num_condition_tokens=None
    ):
        """Process conditional features in 2D mode."""
        b, l, h, w, c = cond_feats.shape
        cond_feats = rearrange(cond_feats, "b l h w c -> (b l) (h w) c")
        valid_masks = rearrange(valid_masks, "b l h w -> (b l) (h w)")

        num_condition_tokens = (
            self.num_condition_tokens
            if num_condition_tokens is None
            else num_condition_tokens
        )
        valid_masks = drop_from_masks(valid_masks, num_condition_tokens)
        cond_feats = cond_feats[valid_masks]

        cond_feats = rearrange(
            cond_feats, "(b t s) c -> b t s 1 c", b=b, t=l, s=num_condition_tokens
        )
        cond_feats = self.pad_to_num_frames(cond_feats, time_idxs)
        cond_feats = rearrange(cond_feats, "b t s 1 c -> (b t) s c")

        # Apply masking probability
        if not num_condition_tokens:
            if self.mask_prob > 0:
                mask = torch.bernoulli(
                    torch.full(
                        cond_feats.size()[:2], self.mask_prob, device=cond_feats.device
                    )
                ).bool()
                cond_feats[mask] = (
                    self.cond_masked_token.to(cond_feats.dtype)
                    .unsqueeze(0)
                    .to(cond_feats.device)
                )
        else:
            cond_feats_placeholder = torch.zeros_like(cond_feats).repeat(
                1, self.num_condition_tokens, 1
            )
            cond_feats_placeholder[:, :num_condition_tokens, :] = cond_feats
            cond_feats = cond_feats_placeholder

        return cond_feats.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def _process_3d_mode(
        self,
        cond_feats,
        valid_masks,
        time_idxs,
        num_condition_tokens=None,
    ):
        """Process conditional features in 3D mode."""
        b, l, h, w, _ = cond_feats.shape

        cond_feats = rearrange(cond_feats, "b l h w c -> b l (h w) c")
        valid_masks = rearrange(valid_masks, "b l h w -> (b l) (h w)")

        if num_condition_tokens is None:
            num_condition_tokens = torch.randint(
                1,
                self.num_condition_tokens,
                (valid_masks.size(0),),
                device=cond_feats.device,
            )
        valid_masks = drop_from_masks(valid_masks, num_condition_tokens)
        for i in range(valid_masks.size(0)):
            token_in_mask = valid_masks[i].sum()
            if token_in_mask < num_condition_tokens[i]:
                raise ValueError(
                    f"Number of tokens to condition on is greater than the number of tokens in the mask. "
                    f"Number of tokens to condition on: {num_condition_tokens[i]}, "
                    f"Number of tokens in the mask: {token_in_mask}"
                )

        valid_masks = rearrange(valid_masks, "(b l) n -> b l n", b=b, l=l)
        valid_masks = valid_masks.unsqueeze(-1)  # b, l, n, 1
        cond_feats = cond_feats * valid_masks

        cond_feats = rearrange(cond_feats, "b l n c -> b l n 1 c")
        cond_feats = self.pad_to_num_frames(
            cond_feats, time_idxs
        )  # [b, self.num_frames, h, w, c]

        cond_feats = rearrange(cond_feats, "b l (h w) 1 c -> b l h w c", h=h, w=w)
        # cond_feats_ = rearrange(cond_feats, "b l h w c -> (b l) c h w")
        # visualize_dino_features(cond_feats_.detach().cpu(), output_dir="dino_features")
        cond_feats = rearrange(cond_feats, "b l h w c -> (b l) c h w", h=h, w=w)

        return cond_feats.view(*cond_feats.shape, 1, 1)

    def _process_3d_id_mode(
        self,
        pixel_values,
        cond_feats,
        valid_masks,
        time_idxs,
        num_condition_tokens=None,
    ):
        cond_feats = (
            self._process_3d_mode(
                cond_feats,
                valid_masks,
                time_idxs,
                num_condition_tokens=num_condition_tokens,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        cond_feats = rearrange(
            cond_feats,
            "(b l) c h w -> b l h w c",
            b=pixel_values.size(0) // self.num_frames,
            l=self.num_frames,
        )

        b, l, h, w, _ = cond_feats.shape

        ## randomly sample a few of the cond_feats
        pixel_values = F.interpolate(
            pixel_values,
            size=(320, 576),
            mode="bilinear",
        )
        pixel_values = rearrange(
            pixel_values, "(b l) c h w -> b l h w c", b=b, l=self.num_frames
        )
        num_of_source_frames = torch.randint(0, 2, (1,)).item()
        source_frame_idx = torch.randperm(len(time_idxs))[:num_of_source_frames].to(
            cond_feats.device
        )
        source_frame_idx = time_idxs[source_frame_idx]

        indices = torch.randperm(self.num_condition_tokens * num_of_source_frames).to(
            cond_feats.device
        )

        # add zero id
        # if len(time_idxs) > 1:
        #     # remove source_frame_idx from time_idxs
        #     remaining_idx = time_idxs[time_idxs != source_frame_idx]
        #     remaining_idxs = time_idxs != source_frame_idx
        #     mask = valid_masks[:, remaining_idxs]
        #     cond_feats[:, remaining_idx, :, :][mask] = cond_feats[:, remaining_idx, :, :][mask] + self.identity(
        #         torch.zeros(cond_feats[:, remaining_idx, :, :][mask].size(0)).to(dtype=torch.int64, device=cond_feats.device)
        #     )

        for i in range(len(source_frame_idx)):
            num_of_target_frames_for_each_source_frames = torch.randint(
                1, 2, (1,)
            ).item()
            if (
                num_of_target_frames_for_each_source_frames == 0
                or num_of_source_frames == 0
            ):
                continue
            # sample target frames (excluding the source frame)
            # limit distance between source and target frames
            # available_frames = torch.arange(start=max(0, source_frame_idx[i] - 10), end=min(self.num_frames, source_frame_idx[i] + 10))
            available_frames = torch.arange(self.num_frames, device=cond_feats.device)
            available_frames = available_frames[
                ~torch.isin(available_frames, source_frame_idx)
            ]
            # available_frames = available_frames[available_frames != source_frame_idx[i]]
            target_frame_idx = available_frames[
                torch.randperm(len(available_frames))[
                    :num_of_target_frames_for_each_source_frames
                ]
            ]

            target_frames = pixel_values[:, target_frame_idx, :, :, :]
            source_frame = pixel_values[:, source_frame_idx[i], :, :, :]
            source_frames = source_frame.unsqueeze(1).repeat(
                1, num_of_target_frames_for_each_source_frames, 1, 1, 1
            )

            source_frames = rearrange(source_frames, "b l h w c -> (b l) c h w")
            target_frames = rearrange(target_frames, "b l h w c -> (b l) c h w")
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                    self.raft = self.raft.to(torch.float32)
                    optical_flow = self.raft(
                        target_frames.float(), source_frames.float()
                    )[-1].float()

                optical_flow = optical_flow.to(target_frames.dtype)

            # flow_imgs = flow_to_image(optical_flow)
            # for i in range(flow_imgs.size(0)):
            #    T.ToPILImage()(flow_imgs[i].cpu()).save(f"dino_features/flow_{i}.png")

            optical_flow_feature_size = F.interpolate(
                optical_flow,
                size=(h, w),
                mode="bilinear",
            )
            optical_flow_feature_size[:, 0, :, :] = optical_flow_feature_size[
                :, 0, :, :
            ] / (pixel_values.size(3) / 2)
            optical_flow_feature_size[:, 1, :, :] = optical_flow_feature_size[
                :, 1, :, :
            ] / (pixel_values.size(2) / 2)

            optical_flow_feature_size = rearrange(
                optical_flow_feature_size,
                "(b l) c h w -> b l h w c",
                b=b,
                l=num_of_target_frames_for_each_source_frames,
            )

            source_cond_feats = cond_feats[
                :, source_frame_idx[i], :, :, :
            ]  # Shape: [b, h, w, c]
            source_cond_feats = rearrange(
                source_cond_feats, "b h w c -> b c h w"
            )  # Shape: [b, c, h, w]

            # add identiy encoding to the nonzero values
            source_cond_feats = rearrange(
                source_cond_feats, "b c h w -> b (h w) c"
            )  # Shape: [b, c, h, w]
            for j in range(source_cond_feats.size(0)):
                non_zero_mask = (source_cond_feats[j] != 0).any(dim=1)
                selected_feats = source_cond_feats[j][non_zero_mask]
                identities = self.identity(
                    # torch.arange(selected_feats.size(0)).to(selected_feats.device) + 1
                    indices[
                        i * self.num_condition_tokens : i * self.num_condition_tokens
                        + selected_feats.size(0)
                    ]
                    # torch.randperm(selected_feats.size(0)).to(selected_feats.device)
                )  # or arange
                temp = source_cond_feats[j][non_zero_mask] + identities
                source_cond_feats[j][non_zero_mask] = temp
                # source_cond_feats[i][non_zero_mask] = source_cond_feats[i][non_zero_mask] + identities

            source_cond_feats = rearrange(
                source_cond_feats, "b (h w) c -> b h w c", h=h, w=w
            )  # Shape: [b, c, h, w]

            optical_flow = (
                optical_flow_feature_size  # Already rearranged to [b, l, h, w, 2]
            )
            optical_flow = rearrange(
                optical_flow, "b l h w c -> (b l) h w c"
            )  # Shape: [b*l, h, w, 2]
            num_targets = optical_flow.size(0)  # b * l
            source_cond_feats_expanded = source_cond_feats.unsqueeze(1).expand(
                -1, num_of_target_frames_for_each_source_frames, -1, -1, -1
            )
            source_cond_feats_expanded = rearrange(
                source_cond_feats_expanded, "b l c h w -> (b l) c h w"
            )  # Shape: [b*l, c, h, w]

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, h, device=optical_flow.device),
                torch.linspace(-1, 1, w, device=optical_flow.device),
            )
            grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: [h, w, 2]
            grid = grid.unsqueeze(0).expand(
                num_targets, -1, -1, -1
            )  # Shape: [b*l, h, w, 2]

            flow_x = optical_flow[..., 0]
            flow_y = optical_flow[..., 1]
            flow = torch.stack((flow_x, flow_y), dim=-1)  # Shape: [b*l, h, w, 2]

            sampling_grid = grid + flow  # Shape: [b*l, h, w, 2]
            source_cond_feats_expanded = source_cond_feats_expanded.permute(0, 3, 1, 2)
            warped_feats = F.grid_sample(
                source_cond_feats_expanded.float(),
                sampling_grid.float(),
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            )  # Shape: [b*l, c, h, w]
            warped_feats = warped_feats.to(source_cond_feats.dtype)

            warped_feats = rearrange(
                warped_feats,
                "(b l) c h w -> b l h w c",
                b=b,
                l=num_of_target_frames_for_each_source_frames,
            )

            cond_feats = cond_feats.clone()
            cond_feats[:, source_frame_idx[i], :, :, :] = source_cond_feats.clone()
            cond_feats[:, target_frame_idx, :, :, :] = warped_feats.clone()

        # cond_feats_ = rearrange(cond_feats[:1,...], "b l h w c -> (b l) c h w")
        # visualize_dino_features(cond_feats_.detach().cpu(), output_dir="dino_features")
        cond_feats = rearrange(cond_feats, "b l h w c -> (b l) c h w", h=h, w=w)

        return cond_feats.view(*cond_feats.shape, 1, 1)

    # def _process_3d_id_mode(
    #    self,
    #    pixel_values,
    #    cond_feats,
    #    valid_masks,
    #    time_idxs,
    #    num_condition_tokens=None,
    # ):
    #    cond_feats = (
    #        self._process_3d_mode(
    #            cond_feats,
    #            valid_masks,
    #            time_idxs,
    #            num_condition_tokens=num_condition_tokens,
    #        )
    #        .squeeze(-1)
    #        .squeeze(-1)
    #    )
    #    cond_feats = rearrange(
    #        cond_feats,
    #        "(b l) c h w -> b l h w c",
    #        b=pixel_values.size(0) // self.num_frames,
    #        l=self.num_frames,
    #    )

    #    b, l, h, w, _ = cond_feats.shape

    #    ###
    #    # randomly sample a few of the cond_feats
    #    # pixel_values = F.interpolate(
    #    #     pixel_values,
    #    #     size=(320, 576),
    #    #     mode="bilinear",
    #    # )
    #    pixel_values = rearrange(
    #        pixel_values, "(b l) c h w -> b l h w c", b=b, l=self.num_frames
    #    )
    #    num_of_source_frames = torch.randint(0, 3, (1,)).item()
    #    source_frame_idx = torch.randperm(len(time_idxs))[:num_of_source_frames].to(
    #        cond_feats.device
    #    )
    #    source_frame_idx = time_idxs[source_frame_idx].sort()

    #    # add zero id
    #    # if len(time_idxs) > 1:
    #    #     # remove source_frame_idx from time_idxs
    #    #     remaining_idx = time_idxs[time_idxs != source_frame_idx]
    #    #     remaining_idxs = time_idxs != source_frame_idx
    #    #     mask = valid_masks[:, remaining_idxs]
    #    #     cond_feats[:, remaining_idx, :, :][mask] = cond_feats[:, remaining_idx, :, :][mask] + self.identity(
    #    #         torch.zeros(cond_feats[:, remaining_idx, :, :][mask].size(0)).to(dtype=torch.int64, device=cond_feats.device)
    #    #     )

    #    for i in range(len(source_frame_idx)):
    #        num_of_target_frames_for_each_source_frames = torch.randint(
    #            0, 3, (1,)
    #        ).item()
    #        if (
    #            num_of_target_frames_for_each_source_frames == 0
    #            or num_of_source_frames == 0
    #        ):
    #            continue
    #        # sample target frames (excluding the source frame)
    #        # limit distance between source and target frames
    #        # available_frames = torch.arange(start=max(0, source_frame_idx[i] - 10), end=min(self.num_frames, source_frame_idx[i] + 10))
    #        available_frames = torch.arange(self.num_frames, device=cond_feats.device)
    #        available_frames = available_frames[available_frames != source_frame_idx[i]]
    #        target_frame_idx = available_frames[
    #            torch.randperm(len(available_frames))[
    #                :num_of_target_frames_for_each_source_frames
    #            ]
    #        ]

    #        target_frames = pixel_values[:, target_frame_idx, :, :, :]
    #        source_frame = pixel_values[:, source_frame_idx[i], :, :, :]
    #        source_frames = source_frame.repeat(
    #            1, num_of_target_frames_for_each_source_frames, 1, 1, 1
    #        )

    #        source_frames = rearrange(source_frames, "b l h w c -> (b l) c h w")
    #        target_frames = rearrange(target_frames, "b l h w c -> (b l) c h w")
    #        with torch.no_grad():
    #            optical_flow = self.raft(target_frames, source_frames)[-1]
    #        # flow_imgs = flow_to_image(optical_flow)
    #        optical_flow_feature_size = F.interpolate(
    #            optical_flow,
    #            size=(h, w),
    #            mode="bilinear",
    #        )
    #        optical_flow_feature_size[:, 0, :, :] = optical_flow_feature_size[
    #            :, 0, :, :
    #        ] / (pixel_values.size(3) / 2)
    #        optical_flow_feature_size[:, 1, :, :] = optical_flow_feature_size[
    #            :, 1, :, :
    #        ] / (pixel_values.size(2) / 2)

    #        optical_flow_feature_size = rearrange(
    #            optical_flow_feature_size,
    #            "(b l) c h w -> b l h w c",
    #            b=b,
    #            l=num_of_target_frames_for_each_source_frames,
    #        )

    #        source_cond_feats = cond_feats[
    #            :, source_frame_idx[i], :, :, :
    #        ]  # Shape: [b, h, w, c]
    #        source_cond_feats = rearrange(
    #            source_cond_feats, "b h w c -> b c h w"
    #        )  # Shape: [b, c, h, w]

    #        # add identiy encoding to the nonzero values
    #        source_cond_feats = rearrange(
    #            source_cond_feats, "b c h w -> b (h w) c"
    #        )  # Shape: [b, c, h, w]
    #        for j in range(source_cond_feats.size(0)):
    #            non_zero_mask = (source_cond_feats[j] != 0).any(dim=1)
    #            selected_feats = source_cond_feats[j][non_zero_mask]
    #            identities = self.identity(
    #                torch.randperm(selected_feats.size(0)).to(selected_feats.device) + 1
    #            )  # or arange
    #            temp = source_cond_feats[j][non_zero_mask] + identities
    #            source_cond_feats[j][non_zero_mask] = temp
    #            # source_cond_feats[i][non_zero_mask] = source_cond_feats[i][non_zero_mask] + identities

    #        source_cond_feats = rearrange(
    #            source_cond_feats, "b (h w) c -> b h w c", h=h, w=w
    #        )  # Shape: [b, c, h, w]

    #        optical_flow = (
    #            optical_flow_feature_size  # Already rearranged to [b, l, h, w, 2]
    #        )
    #        optical_flow = rearrange(
    #            optical_flow, "b l h w c -> (b l) h w c"
    #        )  # Shape: [b*l, h, w, 2]
    #        num_targets = optical_flow.size(0)  # b * l
    #        source_cond_feats_expanded = source_cond_feats.unsqueeze(1).expand(
    #            -1, num_of_target_frames_for_each_source_frames, -1, -1, -1
    #        )
    #        source_cond_feats_expanded = rearrange(
    #            source_cond_feats_expanded, "b l c h w -> (b l) c h w"
    #        )  # Shape: [b*l, c, h, w]

    #        grid_y, grid_x = torch.meshgrid(
    #            torch.linspace(-1, 1, h, device=optical_flow.device),
    #            torch.linspace(-1, 1, w, device=optical_flow.device),
    #        )
    #        grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: [h, w, 2]
    #        grid = grid.unsqueeze(0).expand(
    #            num_targets, -1, -1, -1
    #        )  # Shape: [b*l, h, w, 2]

    #        flow_x = optical_flow[..., 0]
    #        flow_y = optical_flow[..., 1]
    #        flow = torch.stack((flow_x, flow_y), dim=-1)  # Shape: [b*l, h, w, 2]

    #        sampling_grid = grid + flow  # Shape: [b*l, h, w, 2]
    #        source_cond_feats_expanded = source_cond_feats_expanded.permute(0, 3, 1, 2)
    #        warped_feats = F.grid_sample(
    #            source_cond_feats_expanded,
    #            sampling_grid,
    #            mode="nearest",
    #            padding_mode="zeros",
    #            align_corners=True,
    #        )  # Shape: [b*l, c, h, w]

    #        warped_feats = rearrange(
    #            warped_feats,
    #            "(b l) c h w -> b l h w c",
    #            b=b,
    #            l=num_of_target_frames_for_each_source_frames,
    #        )

    #        cond_feats = cond_feats.clone()
    #        cond_feats[:, source_frame_idx[i], :, :, :] = source_cond_feats.clone()
    #        cond_feats[:, target_frame_idx, :, :, :] = warped_feats.clone()
    #        # zero out cond feats between source and target frames
    #        for j in range(source_frame_idx[i] + 1, target_frame_idx):
    #            cond_feats[:, j, :, :, :] = 0

    #    # cond_feats_ = rearrange(cond_feats[:1,...], "b l h w c -> (b l) c h w")
    #    # visualize_dino_features(cond_feats_.detach().cpu(), output_dir="dino_features")
    #    cond_feats = rearrange(cond_feats, "b l h w c -> (b l) c h w", h=h, w=w)

    #    return cond_feats.view(*cond_feats.shape, 1, 1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        time_idxs: Optional[torch.Tensor] = None,
        force_cond_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, channels, height, width] input images
            time_idxs: [batch_size, 1] time indices for each batch element
            force_cond_feats: [batch_size, length, channels, height, width] features to use instead of extracting from pixel_values
        """
        batch_size = pixel_values.size(0)

        # Early return if random number is less than ucg_prob
        if torch.rand(1).item() < self.ucg_prob:
            return self._get_masked_token(batch_size).to(pixel_values.dtype)

        if force_cond_feats is not None and time_idxs is None:
            raise ValueError(
                "time_idxs should be provided when force_cond_feats is given"
            )

        # Resize pixel_values if necessary
        pixel_values = self._resize_images(
            pixel_values
        )  # [b * num_frames, c, image_height, image_width]

        # Extract conditional features
        cond_feats, valid_masks, time_idxs = self._extract_cond_feats(
            pixel_values, time_idxs, force_cond_feats
        )  # [b, l, h, w, c], [b, l, h, w], [b,]

        # Add positional embeddings if required
        if self.mode not in ["3d", "3d_id"]:
            cond_feats = self._add_embeddings(cond_feats, time_idxs)

        cond_feats = self._insert_masked_tokens(cond_feats, valid_masks)

        # Process conditional features based on the mode
        if self.mode == "1d":
            return self._process_1d_mode(cond_feats, valid_masks)
        elif self.mode == "2d":
            return self._process_2d_mode(cond_feats, valid_masks, time_idxs)
        elif self.mode == "3d":
            return self._process_3d_mode(cond_feats, valid_masks, time_idxs)
        elif self.mode == "3d_encoded":
            return self.encoder(
                self._process_3d_mode(cond_feats, valid_masks, time_idxs),
                cond_mask=None,
                num_frames=self.num_frames,
            )
        elif self.mode == "3d_id":
            return self._process_3d_id_mode(
                pixel_values, cond_feats, valid_masks, time_idxs
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
