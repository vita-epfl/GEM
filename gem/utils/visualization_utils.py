from typing_extensions import Optional
import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA
import torch
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

# from utils.image_utils import fig2img
import numpy as np
import os
import wandb
from einops import rearrange
from typing import Union


def fig2img(fig) -> Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", transparent="True", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return img


def draw_roi(image: torch.Tensor, roi: torch.Tensor, color: str = "red") -> Image:
    """
    Draw a region of interest (ROI) on an image.

    Args:
        image (torch.Tensor): Image tensor of shape [c h w], normalized to [-1, 1].
        roi (torch.Tensor): ROI tensor of shape [4], containing the PIXEL coordinates of the ROI [ x, y, w, h ].
        color (str, optional): Color of the ROI. Defaults to 'red'.

    Returns:
        Image: Image with the ROI drawn.
    """
    assert len(image.shape) == 3, "Image must have shape [c h w]"
    assert len(roi) == 4, "ROI must have shape [4]"

    image = image.detach().cpu().numpy()
    image = (image + 1) / 2 * 255
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)

    # Draw the ROI
    x1, y1, w, h = roi
    x2 = x1 + w
    y2 = y1 + h
    image = np.ascontiguousarray(image.transpose(1, 2, 0))
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
    ax.axis("off")
    img = fig2img(fig)
    plt.close(fig)
    return img


def pil_image_to_torch(image: Image) -> torch.Tensor:
    """
    Convert a PIL image to a torch.Tensor.

    Args:
        image (Image): Image to convert.
    Returns:
        torch.Tensor: Converted image of shape [c h w], normalized to [-1, 1].
    """
    image_npy = np.array(image.convert("RGB"))  # [ h w c ]
    image_torch = torch.from_numpy(image_npy).permute(2, 0, 1).float() / 127.5 - 1
    return image_torch


def create_dino_video(
    dino_tokens: torch.Tensor, dino_frames: torch.Tensor, pca_mask: torch.Tensor
) -> np.ndarray:
    """Creates a video with the dino tokens and the dino_frames.

    Args:
        dino_tokens (torch.Tensor): [T, C, H, W] tensor with the dino tokens.
        target_frames (torch.Tensor): [T, 3, H, W] tensor with the target frames.
        pca_mask (torch.Tensor): [T, C, H, W] tensor with the PCA mask.

    Returns:
        np.ndarray: [L', 3, H, W] tensor with the video.
    """
    assert dino_tokens.size(0) == dino_frames.size(
        0
    ), "dino_tokens and dino_frames must have the same length"
    assert len(dino_frames.size()) == 4, "dino_frames must have 4 dimensions"
    assert dino_frames.size(1) == 3, "dino_frames must have 3 channels"

    L = dino_tokens.size(0)
    result_frames = []

    for i in range(L):
        frame_pil = visualize_dino(None, dino_tokens[i], dino_frames[i], pca_mask[i])
        result_frames.append(pil_image_to_torch(frame_pil))

    result_frames = np.stack(result_frames, axis=0)
    return result_frames


def load_video_frames(
    video_path: str, start_frame: float, num_frames: int, w=512, h=320, device="cuda:0"
) -> np.ndarray:
    """
    Load video frames from a directory containing images named 00000.png, 00001.png, etc.

    Args:
        video_path (str): Path to the video frames.
        start_frame (float): Starting frame (percentage of the video) e.g. 0.1, 0.5, 0.9...
        num_frames (int): Number of frames to load.

    Returns:
        torch.Tensor: Loaded frames with shape [b, num_frames, c, h, w]. NOTE that b=1.
    """
    frames = []
    # To compute zfill, get the length of a filename in the directory (that ends with .png)
    files = os.listdir(video_path)
    files = [f for f in files if f.endswith(".png")]
    zfill = len(files[0].split(".")[0])
    start_frame_num = int(start_frame * len(files))
    pil_frame = None
    for i in range(num_frames):
        frame_num = str(start_frame_num + i).zfill(zfill)
        frame = Image.open(os.path.join(video_path, f"{frame_num}.png")).resize((w, h))
        if i == 0:
            pil_frame = frame
        # Convert to torch.Tensor, normalize, permute and add to the list
        frame = (
            torch.from_numpy(np.array(frame))
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 127.5
            - 1
        )
        frames.append(frame)
    frames = torch.cat(frames, dim=0).to(device).unsqueeze(0)
    return frames, pil_frame


def log_image_to_wandb(vis_name: str, image: Union[torch.Tensor, np.ndarray]):
    """Logs an image to wandb.

    Args:
        vis_name (str): Name of the image to log.
        image (Union[torch.Tensor, np.ndarray]): Image tensor of shape [c, h, w], normalized to [-1, 1].
    """
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    wandb.log(
        {
            vis_name: wandb.Image(
                ((image + 1) / 2 * 255).astype("uint8").transpose(1, 2, 0)
            )
        }
    )


def log_video_to_wandb(
    vis_name: str, video_frames: Union[torch.Tensor, np.ndarray], fps: int = 7
):
    """
    Visualizes a video input in wandb.

    Args:
        vis_name (str): Name of the video to log.
        video_frames (Union[torch.Tensor, np.ndarray]): Tensor of shape [time c h w] or [b time c h w], normalized to [-1, 1].
    """
    assert (
        len(video_frames.shape) == 4 or len(video_frames.shape) == 5
    ), "video_frames must have shape [time c h w] or [b time c h w]"
    assert (
        video_frames.shape[-3] == 3
    ), "video_frames must have 3 channels. Got shape: {}".format(video_frames.shape)

    if torch.is_tensor(video_frames):
        video_frames = video_frames.detach().cpu().numpy()

    if video_frames.dtype != np.uint8:
        video_frames = video_frames.astype(np.float32)
        video_frames = (video_frames + 1) / 2
        video_frames = video_frames * 255
        video_frames = np.clip(video_frames, 0, 255)
        video_frames = video_frames.astype(np.uint8)

    wandb.log(
        {
            vis_name: wandb.Video(
                video_frames,
                fps=fps,
            )
        }
    )


def visualize_dino(
    vis_path: Optional[str],
    dino_feats: torch.Tensor,
    original_img: torch.Tensor,
    pca_mask: torch.Tensor,
) -> Image:
    """
    Visualize the DINO features overlaid on the original image.

    Args:
        vis_path (str): Path to save the visualization. If None, returns the PIL image.
        dino_feats (torch.Tensor): Features to visualize of shape [c, h, w].
        original_img (torch.Tensor): Original image to overlay the features on, of shape [c, h, w].
        pca_mask (torch.Tensor): Mask for PCA of shape [h, w]; True where features should be included in PCA.

    Returns:
        PIL.Image: The visualization of the DINO features overlaid on the original image if vis_path is None.
    """
    assert len(original_img.shape) == 3, "Original image must have shape [c, h, w]"
    assert len(dino_feats.shape) == 3, "DINO features must have shape [c, h, w]"
    assert original_img.size(0) == 3, "Original image must have 3 channels"

    new_h = original_img.size(1)
    new_w = original_img.size(2)

    # Convert dino_feats to numpy and reshape to [h*w, c]
    dino_feats_np = dino_feats.detach().cpu().numpy()  # Shape: [c, h, w]
    c, h, w = dino_feats_np.shape
    dino_feats_np = dino_feats_np.reshape(c, -1).T  # Shape: [h*w, c]

    # Flatten pca_mask to match the features dimension
    pca_mask_flat = pca_mask.flatten()  # Shape: [h*w], dtype: bool

    # Select features where pca_mask is True
    dino_feats_masked = dino_feats_np[
        pca_mask_flat > 0.01
    ]  # Shape: [num_masked_pixels, c]

    # Perform PCA on masked features
    pca_feats = None
    if dino_feats_masked.shape[0] > 0:
        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(
            dino_feats_masked
        )  # Shape: [num_masked_pixels, 3]

        # Min-max scaling of PCA features
        for i in range(3):
            pca_feats[:, i] = (pca_feats[:, i] - pca_feats[:, i].min()) / (
                pca_feats[:, i].max() - pca_feats[:, i].min() + 1e-9
            )

    # Initialize array for all PCA features
    pca_feats_all = np.zeros((h * w, 3))

    if pca_feats is not None:
        # Assign PCA-transformed features back to their positions
        pca_feats_all[pca_mask_flat > 0.01] = pca_feats

    # Reshape to [h, w, 3]
    pca_feats_all = pca_feats_all.reshape(h, w, 3)

    # Resize pca_features_rgb to the original image size using nearest neighbor
    pca_features_rgb = (
        torch.from_numpy(pca_feats_all).permute(2, 0, 1).unsqueeze(0)
    )  # Shape: [1, 3, h, w]
    pca_features_rgb = torch.nn.functional.interpolate(
        pca_features_rgb, (new_h, new_w), mode="nearest"
    )
    pca_features_rgb = (
        pca_features_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    )  # Shape: [new_h, new_w, 3]

    # Plot the overlay
    plt.clf()
    plt.axis("off")
    if original_img is not None:
        plt.imshow((original_img.permute(1, 2, 0).cpu().numpy() + 1) / 2)
    plt.imshow(pca_features_rgb, alpha=0.4)

    if vis_path is not None:
        plt.savefig(vis_path, bbox_inches="tight", transparent=True, pad_inches=0)
        plt.close()
        return None
    else:
        fig = plt.gcf()
        img = fig2img(fig)
        plt.close()
        return img
