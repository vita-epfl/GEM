# video_model.py but with DINO conditioning
from typing import List, Union

from gem.modules.diffusionmodules.openaimodel import *
from gem.modules.video_attention import SpatialVideoTransformer
from gem.util import default, repeat_as_img_seq
from .util import AlphaBlender
from omegaconf import DictConfig
from gem.util import instantiate_from_config



class VideoResBlock(ResBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        video_kernel_size: Union[int, List[int]] = 3,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down,
        )
        self.time_stack = ResBlock(
            default(out_channels, channels),
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,
            causal=False,
        )
        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            rearrange_pattern="b t -> b 1 t 1 1",
        )

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        x = super().forward(x, emb)

        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_frames)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_frames)

        x = self.time_stack(x, rearrange(emb, "(b t) ... -> b t ...", t=num_frames))
        x = self.time_mixer(x_spatial=x_mix, x_temporal=x)
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x



class VideoUNet(nn.Module):
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
        use_spatial_context: bool = False,
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
        angle_control: bool = False,
        dino_cnn_cfg: Optional[DictConfig] = None,
        additional_ref_time_embed=False,
    ):
        super().__init__()

        assert context_dim is not None

        self.dino_cnn_cfg = dino_cnn_cfg
        self.dino_cnn = (
            instantiate_from_config(dino_cnn_cfg) if dino_cnn_cfg is not None else None
        )

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
        self.additional_ref_time_embed = additional_ref_time_embed

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

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":  # this way
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError

        self.adm_in_channels = adm_in_channels
        if self.additional_ref_time_embed:
            self.ref_time_embed = nn.Sequential(
                nn.Sequential(
                    linear(256, time_embed_dim),
                    nn.SiLU(),
                    # zero_module(linear(time_embed_dim, context_dim)),
                    zero_module(linear(time_embed_dim, time_embed_dim)),
                ),
            )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self.skeleton_context_proj = nn.Conv2d(320, model_channels, 1)

        self.depth_in = zero_module(
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
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
            angle_control=False,
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
                use_spatial_context=use_spatial_context,
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
                angle_control=angle_control,
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
                            angle_control=angle_control,
                        )
                    )

                layers_input = layers
                self.input_blocks.append(TimestepEmbedSequential(*layers_input))
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

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
                add_lora=add_lora,
                action_control=action_control,
                angle_control=angle_control,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList(list())
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
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
                            angle_control=angle_control,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
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
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_up=time_downup,
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # Not used if data_dict doesn't contain depth_img
        self.out_depth = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # Print param count in millions
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model has {num_params / 1e6:.2f}M parameters")

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        fd_context: Optional[torch.Tensor] = None,
        skeletons_context: Optional[torch.Tensor] = None,
    ):
        # if weights are in half precision, inputs must be half precision
        dtype = self.time_embed[0].weight.dtype
        if dtype == torch.float16 or dtype == torch.bfloat16:
            x = x.type(dtype)
            timesteps = timesteps.type(dtype)
            if context is not None:
                context = context.type(dtype)
            if y is not None:
                y = y.type(dtype)
            if time_context is not None:
                time_context = time_context.type(dtype)
            if cond_mask is not None:
                cond_mask = cond_mask.type(dtype)
            if fd_context is not None:
                fd_context = fd_context.type(dtype)

        assert (y is not None) == (
            self.num_classes is not None
        ), "Must specify y if and only if the model is class-conditional"
        hs = list()
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if dtype == torch.bfloat16 or dtype == torch.float16:
            t_emb = t_emb.type(dtype)

        if cond_mask is not None and cond_mask.any():
            cond_mask_ = cond_mask[..., None].float()
            emb = self.cond_time_stack_embed(t_emb) * cond_mask_ + self.time_embed(
                t_emb
            ) * (1 - cond_mask_)
        else:
            emb = self.time_embed(t_emb)

        if num_frames > 1 and context.shape[0] != x.shape[0]:
            assert (
                context.shape[0] == x.shape[0] // num_frames
            ), f"{context.shape} {x.shape}"
            context = repeat_as_img_seq(context, num_frames)

        if y.size(1) != self.adm_in_channels and self.additional_ref_time_embed:
            ref_t = y[:, self.adm_in_channels :]
            ref_t = self.ref_time_embed(ref_t)
            ref_t = repeat_as_img_seq(ref_t, num_frames)
            emb = emb + ref_t
            # context = torch.cat((context, ref_t), dim=1)
            y = y[:, : self.adm_in_channels]

        if self.num_classes is not None:
            if num_frames > 1 and y.shape[0] != x.shape[0]:
                assert y.shape[0] == x.shape[0] // num_frames, f"{y.shape} {x.shape}"
                y = repeat_as_img_seq(y, num_frames)
            emb = emb + self.label_emb(y)

        h_depth = None
        if x.shape[1] == 12: # If there's RGB + depth to be denoised...
            h = x[:, [0, 1, 2, 3, 8, 9, 10, 11]]
            h_depth = x[:, [4, 5, 6, 7, 8, 9, 10, 11]]
        else: # Otherwise we expect 8 channels
            assert x.shape[1] == 8, f"Expected 8 channels, got {x.shape[1]}"
            h = x

        if self.dino_cnn is not None:
            h_condition = self.dino_cnn(
                None, fd_context, timesteps, cond_mask, num_frames
            )
        else:
            h_condition = None

        for i, module in enumerate(self.input_blocks):
            h = module(
                h,
                emb,
                context=context,
                time_context=time_context,
                num_frames=num_frames,
            )

            if i == 0 and h_depth is not None:
                h = h + self.depth_in(
                    h_depth,
                    emb,
                    context=context,
                    time_context=time_context,
                    num_frames=num_frames,
                )

            if i == 0 and skeletons_context is not None:
                h = h + self.skeleton_context_proj(
                    skeletons_context.squeeze(-1).squeeze(-1).squeeze(-1)
                )

            if (
                h_condition is not None
                and i < len(h_condition)
                and h_condition[i] is not None
            ):
                h = h + h_condition[i]
            hs.append(h)

        h = self.middle_block(
            h, emb, context=context, time_context=time_context, num_frames=num_frames
        )

        for module in self.output_blocks:
            h = torch.cat((h, hs.pop()), dim=1)
            h = module(
                h,
                emb,
                context=context,
                time_context=time_context,
                num_frames=num_frames,
            )

        h = h.type(x.dtype)
        out1 = self.out(h)
        if h_depth is not None:
            out2 = self.out_depth(h)  # [B 4 H W]
            # Concat on channel dim
            out1 = torch.cat([out1, out2], dim=1)

        return out1
