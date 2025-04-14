# The model that takes in dino tokens, transforms them a bit, downsamples, then adds the resulting features to the dino_video_model.py

from typing import List, Union

from gem.modules.diffusionmodules.openaimodel import *
from gem.modules.video_attention import SpatialVideoTransformer
from gem.util import default, repeat_as_img_seq
from gem.modules.diffusionmodules.util import AlphaBlender
from gem.modules.diffusionmodules.video_model import VideoResBlock


class DINOCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        # use_spatial_context: bool = False,
        merge_strategy: str = "learned_with_images",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
        add_lora: bool = False,
        action_control: bool = False,
        src_h_condition_idxs: List[int] = [1, 4, 7],
        dst_h_condition_idxs: List[int] = [2, 5, 8],
    ):
        super().__init__()

        self.src_h_condition_idxs = src_h_condition_idxs
        self.dst_h_condition_idxs = dst_h_condition_idxs
        assert len(self.src_h_condition_idxs) == len(
            self.dst_h_condition_idxs
        ), "src_h_condition_idxs and dst_h_condition_idxs must have the same length"

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.cond_time_stack_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
            add_lora=False,
            action_control=False,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=False,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                use_checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
                add_lora=add_lora,
                action_control=action_control,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                            add_lora=add_lora,
                            action_control=action_control,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        self.output_convs = nn.ModuleList([])

        for src_idx, out_channel in zip(src_h_condition_idxs, self.out_channels):
            ch = input_block_chans[src_idx]
            self.output_convs.append(
                nn.Sequential(
                    normalization(ch),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, ch, out_channel, 1, padding=0)),
                )
            )
        # print self number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters in DINOCNN: {num_params}")

    def forward(
        self,
        image_seq: torch.Tensor,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond_mask: torch.Tensor,
        num_frames: int,
    ) -> List[torch.Tensor]:
        hs = list()
        if image_seq is not None:
            x = torch.cat([image_seq, x], dim=1)
        h = x
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if cond_mask is not None and cond_mask.any():
            cond_mask_ = cond_mask[..., None].float()
            emb = self.cond_time_stack_embed(t_emb) * cond_mask_ + self.time_embed(
                t_emb
            ) * (1 - cond_mask_)
        else:
            emb = self.time_embed(t_emb)

        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, num_frames=num_frames)
            if i in self.src_h_condition_idxs:
                hs.append(h)

        final_hs = [None] * (max(self.dst_h_condition_idxs) + 1)

        for i in range(len(self.src_h_condition_idxs)):
            final_hs[self.dst_h_condition_idxs[i]] = self.output_convs[i](hs[i])

        return final_hs
