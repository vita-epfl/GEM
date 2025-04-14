import math
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from einops import rearrange
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from gem.modules import UNCONDITIONAL_CONFIG
from gem.modules.autoencoding.temporal_ae import VideoDecoder
from gem.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from gem.modules.ema import LitEma
from gem.util import default, disabled_train, get_obj_from_str, instantiate_from_config
import cv2


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
    return wandb_logger


def normalize_and_convert_to_uint8(tensor):
    return (
        ((tensor.clamp(-1, 1).detach().cpu().float().numpy() + 1) / 2.0) * 255
    ).astype(np.uint8)


class DiffusionEngine(LightningModule):
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Ensure batch is a dictionary
        if isinstance(batch, dict):
            # Transfer all items except 'img_seq_CPU' to the specified device
            for key, value in batch.items():
                if key != "img_seq_CPU":  # Keep this key on the CPU
                    batch[key] = super().transfer_batch_to_device(
                        value, device, dataloader_idx
                    )
        else:
            # If batch is not a dictionary, use the default transfer behavior
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "img",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: int = 14,
        num_frames: int = 25,
        high_fd_control: bool = False,
        slow_spatial_layers: bool = False,
        slow_temporal_layers: bool = False,
        train_peft_adapters: bool = False,
        train_lora_and_depth: bool = False,
        replace_cond_frames: bool = False,
        enable_bf16: bool = False,
        fixed_cond_frames: Union[List, None] = None,
        enable_online_depth: bool = False,
        pixel_space_post_training: bool = False,
        fm_pretraining: bool = False,
    ):
        super().__init__()

        self.fm_pretraining = fm_pretraining
        self.pixel_space_post_training = pixel_space_post_training
        self.enable_bf16 = enable_bf16
        self.enable_online_depth = enable_online_depth
        self.log_keys = log_keys
        self.input_key = input_key
        
        self.optimizer_config = default(
            optimizer_config,
            {"target": "deepspeed.ops.adam.FusedAdam"},
        )
        self.train_lora_and_depth = train_lora_and_depth
        self.save_hyperparameters()
        model = instantiate_from_config(network_config)
        
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        self.ema_decay_rate = ema_decay_rate
        if use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.num_frames = num_frames
        self.high_fd_control = high_fd_control
        self.slow_spatial_layers = slow_spatial_layers
        self.slow_temporal_layers = slow_temporal_layers
        self.train_peft_adapters = train_peft_adapters
        self.replace_cond_frames = replace_cond_frames
        self.fixed_cond_frames = fixed_cond_frames

    def reinit_ema(self):
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=self.ema_decay_rate)
            print(f"Reinitializing EMAs of {len(list(self.model_ema.buffers()))}")

    def init_from_ckpt(self, path: str) -> None:
        if path.endswith("ckpt"):
            svd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("bin"):  # for deepspeed merged checkpoints
            svd = torch.load(path, map_location="cpu")
            for k in list(svd.keys()):  # remove the prefix
                if "_forward_module" in k:
                    svd[k.replace("_forward_module.", "")] = svd[k]
                del svd[k]
        elif path.endswith("safetensors"):
            svd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(svd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    @torch.no_grad()
    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict
        # image tensors should be scaled to -1 ... 1 and in bchw format
        input_shape = batch[self.input_key].shape
        if len(input_shape) != 4:  # is an image sequence
            assert (
                input_shape[1] == self.num_frames
            ), f"{input_shape} vs {self.num_frames}"
            batch[self.input_key] = rearrange(batch[self.input_key], "b t c h w -> (b t) c h w")

        if "depth_img" in batch and len(batch["depth_img"].shape) != 4:  # is an image sequence
            assert batch["depth_img"].shape[1] == self.num_frames
            batch["depth_img"] = rearrange(batch["depth_img"], "b t c h w -> (b t) c h w")

        if "rendered_poses" in batch and len(batch["rendered_poses"].shape) != 4:  # is an image sequence
            assert (
                batch["rendered_poses"].shape[1] == self.num_frames
            ), f"{batch['rendered_poses'].shape} vs {self.num_frames}"
            batch["rendered_poses"] = rearrange(batch["rendered_poses"], "b t c h w -> (b t) c h w")

        return batch[self.input_key], batch

    @torch.no_grad()
    def decode_first_stage(self, z, overlap=3):
        z = z / self.scale_factor
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        all_out = list()
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            if overlap < n_samples:
                previous_z = z[:overlap]
                for current_z in z[overlap:].split(n_samples - overlap, dim=0):
                    if isinstance(self.first_stage_model.decoder, VideoDecoder):
                        kwargs = {"timesteps": current_z.shape[0] + overlap}
                    else:
                        kwargs = dict()
                    context_z = torch.cat((previous_z, current_z), dim=0)
                    previous_z = current_z[-overlap:]
                    out = self.first_stage_model.decode(context_z, **kwargs)

                    if not all_out:
                        all_out.append(out)
                    else:
                        all_out[-1][-overlap:] = (
                            all_out[-1][-overlap:] + out[:overlap]
                        ) / 2
                        all_out.append(out[overlap:])
            else:
                for current_z in z.split(n_samples, dim=0):
                    if isinstance(self.first_stage_model.decoder, VideoDecoder):
                        kwargs = {"timesteps": current_z.shape[0]}
                    else:
                        kwargs = dict()
                    out = self.first_stage_model.decode(current_z, **kwargs)
                    all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = list()
        c_out = list()
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = z * self.scale_factor
        return z

    # @def forward(self, x, batch):
    def forward(self, x, batch):
        if self.pixel_space_post_training:
            loss, latent_loss, pixel_loss, depth_loss = self.loss_fn(
                self.model,
                self.denoiser,
                self.conditioner,
                x,
                batch,
                self.first_stage_model,
            )  # go to StandardDiffusionLoss
            loss_mean = loss.mean()
            latent_loss_mean = latent_loss.mean()
            pixel_loss_mean = pixel_loss.mean()
            depth_loss_mean = depth_loss.mean()
            loss_dict = {
                "loss": loss_mean,
                "latent_loss": latent_loss_mean,
                "pixel_loss": pixel_loss_mean,
                "depth_loss": depth_loss_mean,
            }
        else:
            loss, latent_loss, depth_loss = self.loss_fn(
                self.model, self.denoiser, self.conditioner, x, batch
            )
            loss_mean = loss.mean()
            latent_loss_mean = latent_loss.mean()
            depth_loss_mean = depth_loss.mean()
            loss_dict = {
                "loss": loss_mean,
                "latent_loss": latent_loss_mean,
                "depth_loss": depth_loss_mean,
            }
        
        if "depth_img" not in batch:
            loss_dict.pop("depth_loss")
        
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x, batch = self.get_input(batch)
        x = self.encode_first_stage(x)
        if "depth_img" in batch:
            x_depth = self.encode_first_stage(batch["depth_img"])
            x = torch.cat((x, x_depth), dim=1)

        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.high_fd_control:
            # double checked this to contain all correct parameters
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if (
                            "norm21" in n
                            or "attn21" in n
                            or "input_fd_blocks" in n
                            or "input_fd_zero_convs" in n
                        )
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "input_blocks" not in n
                        and "input_fd_blocks" not in n
                        and "input_fd_zero_convs" not in n
                        and "norm21" not in n
                        and "attn21" not in n
                    ],
                    "lr": lr * 0.1,
                },
            ]
        elif self.fm_pretraining:
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if (
                            # Input projection in the first input block
                            "input_blocks.0.0" in n
                            or
                            # Output projection in the last output block
                            "output_blocks" in n
                            and n.endswith("0.0.weight")
                            or
                            # Depth input layer
                            "depth_in" in n
                            or
                            # Time embeddings
                            "time_embed" in n
                            or "cond_time_stack_embed" in n
                            or
                            # Label embedding
                            "label_emb" in n
                        )
                    ]
                },
            ]

        elif self.slow_spatial_layers:
            param_dicts = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if "time_stack" in n
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "time_stack" not in n
                    ],
                    "lr": lr * 0.01,
                },
            ]
        elif self.slow_temporal_layers:
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "time_stack" not in n
                    ],
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if "time_stack" in n
                    ],
                    "lr": lr * 0.01,
                },
            ]
        elif self.train_peft_adapters:
            param_dicts = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if "adapter" in n
                    ]
                }
            ]
            for n, p in self.model.named_parameters():
                if "adapter" not in n:
                    p.requires_grad = False
        elif self.train_lora_and_depth:
            param_dicts = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "adapter" in n or "depth" in n
                    ]
                }
            ]

            frozen_count = 0
            for n, p in self.model.named_parameters():
                if "adapter" not in n and "depth" not in n:
                    p.requires_grad = False
                    frozen_count += 1

            print(f"!!!!! Froze {frozen_count} parameters")
        else:
            param_dicts = [{"params": list(self.model.parameters())}]
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                param_dicts.append({"params": list(embedder.parameters())})
        opt = self.instantiate_optimizer_from_config(
            param_dicts, lr, self.optimizer_config
        )
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        else:
            return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        cond_frame=None,
        uc: Union[Dict, None] = None,
        N: int = 25,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.enable_bf16):
            randn = torch.randn(N, *shape).to(self.device)
            cond_mask = torch.zeros(N).to(self.device)
            if self.replace_cond_frames:
                assert self.fixed_cond_frames
                cond_indices = self.fixed_cond_frames
                cond_mask = rearrange(cond_mask, "(b t) -> b t", t=self.num_frames)
                cond_mask[:, cond_indices] = 1
                cond_mask = rearrange(cond_mask, "b t -> (b t)")

            denoiser = lambda input, sigma, c, cond_mask: self.denoiser(
                self.model, input, sigma, c, cond_mask, **kwargs
            )
            samples = self.sampler(  # go to EulerEDMSampler
                denoiser, randn, cond, uc=uc, cond_frame=cond_frame, cond_mask=cond_mask
            )
        return samples

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 25,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [
            e.input_key for e in self.conditioner.embedders if e.ucg_rate > 0.0
        ]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys, "
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x, batch = self.get_input(batch)
        if "depth_img" in batch:
            x_depth = batch["depth_img"]

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=(
                ucg_keys if len(self.conditioner.embedders) > 0 else list()
            ),
        )

        sampling_kwargs = dict()

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        if "depth_img" in batch:
            x_depth = x_depth.to(self.device)[:N]

        z = self.encode_first_stage(x)
        if "depth_img" in batch:
            z_depth = self.encode_first_stage(x_depth)
            z = torch.cat((z, z_depth), dim=1)

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
                if c[k].shape[0] < N:
                    c[k] = c[k][[0]]
                if uc[k].shape[0] < N:
                    uc[k] = uc[k][[0]]

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, cond_frame=z, shape=z.shape[1:], uc=uc, N=N, **sampling_kwargs
                )
            img_samples = self.decode_first_stage(samples[:, :4, :, :])
            log["samples"] = log["samples_mp4"] = img_samples

            if "depth_img" in batch:
                depth_samples = self.decode_first_stage(samples[:, 4:, :, :])
                depth_samples = torch.mean(depth_samples, dim=1, keepdim=True)
                depth_samples = depth_samples.repeat(1, 3, 1, 1)
                log["depth_samples"] = log["depth_samples_mp4"] = depth_samples

        # Log inputs and reconstructions to WandB
        log["inputs"] = log["inputs_mp4"] = x
    
        if "depth_img" in batch:
            log["depth_inputs"] = log["depth_inputs_mp4"] = x_depth

        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return log

        if sample:
            samples_wb = [normalize_and_convert_to_uint8(img_samples)]
            if "depth_img" in batch:
                samples_depth_wb = [normalize_and_convert_to_uint8(depth_samples)]

        x_wb = [normalize_and_convert_to_uint8(x)]
        if "depth_img" in batch:
            x_depth_wb = [normalize_and_convert_to_uint8(x_depth)]

        wandb_logger.log_video("samples", samples_wb, fps=[10], step=self.global_step)
        if "depth_img" in batch:
            wandb_logger.log_video(
                "depth_samples", samples_depth_wb, fps=[10], step=self.global_step
            )

        wandb_logger.log_video("inputs", x_wb, fps=[10], step=self.global_step)

        if "depth_img" in batch:
            wandb_logger.log_video(
                "depth_inputs", x_depth_wb, fps=[10], step=self.global_step
            )

        if "trajectory" in batch:
            try:
                x_trajs = (
                    draw_trajectories(x, batch).permute(0, 3, 1, 2).detach().cpu().numpy()
                )
                x_trajs_wb = [normalize_and_convert_to_uint8(torch.from_numpy(x_trajs))]
                wandb_logger.log_video(
                    "trajs_inputs", x_trajs_wb, fps=[10], step=self.global_step
                )
            except Exception as e:
                print(f"Failed to draw trajectories: {e}")

        if "rendered_poses" in batch:
            try:
                skeletons_inputs = (
                    batch["rendered_poses"].detach().cpu().numpy().astype(np.uint8)
                )
                wandb_logger.log_video(
                    "rendered_poses",
                    [skeletons_inputs],
                    fps=[10],
                    step=self.global_step,
                )
            except Exception as e:
                print(f"Failed to draw skeletons: {e}")

        return log


def draw_trajectories(images, action_dict: dict = None):
    """
    Annotate bounding boxes on the input images with frame count and labels.
    Args:
        images (torch.Tensor): Input image tensor.
        action_dict (dict)
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)

    images_out = []
    traj = action_dict["trajectory"].reshape(-1, 2).detach().cpu().numpy()
    for nr, image in enumerate(images):
        image_np = image.cpu().numpy()

        # If the image tensor is in the range [0, 1], scale it to [0, 255]
        if image_np.max() <= 1.0 and image_np.min() >= 0.0:
            image_np = (image_np * 255).astype(np.uint8)
        elif image_np.min() < 0.0:
            image_np = ((image_np + 1) * 127.5).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        # If the image is in (C, H, W) format, transpose it to (H, W, C)
        if image_np.shape[0] == 3 and image_np.ndim == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        # Ensure that the image is in H, W, 3 format and is of type uint8
        if image_np.ndim == 2:  # Convert grayscale to color
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 1:  # Convert single channel to 3 channels
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] != 3:  # Handle any unexpected number of channels
            raise ValueError(
                "Expected image to have 3 channels, but got shape {}".format(
                    image_np.shape
                )
            )

        # Ensure the image is of type uint8
        image_np = image_np.astype(np.uint8)

        # Get image dimensions
        H, W, _ = image_np.shape
        traj_x = W // 2
        traj_y = int(H * 0.8)
        traj_scale = 3.0
        # take the l2 norm of all vectors
        traj_lens = np.linalg.norm(traj, axis=-1, keepdims=True)
        max_length = traj_lens.max()

        if max_length > 1e-4:
            # take max len
            traj_scale = 48.0 / (traj_lens.max() + 1e-5)

            for traj_idx in range(traj.shape[0]):
                tx = int(traj[traj_idx, 0] * traj_scale)
                ty = -int(traj[traj_idx, 1] * traj_scale)

                dst = (max(min(traj_x + tx, W), 0), max(min(traj_y + ty, H), 0))
                image_np = np.ascontiguousarray(image_np, dtype=np.uint8)
                cv2.circle(image_np, dst, radius=5, color=(255, 0, 0), thickness=-1)

        # Normalize the image back to the range [-1, 1]
        image_np = image_np.astype(np.float32) / 255
        image_np = (image_np * 2) - 1
        images_out.append(image_np)

    return torch.tensor(np.array(images_out))
