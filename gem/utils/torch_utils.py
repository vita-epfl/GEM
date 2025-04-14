import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision


def torch_to_pil_image(image: torch.Tensor) -> Image:
    """
    Converts a torch tensor image to a PIL image
    Args:
        image: torch tensor image of shape (C, H, W)
    Returns:
        PIL image
    """
    image_bytes = ((image + 1) / 2 * 255).clamp(0, 255).byte()
    return torchvision.transforms.ToPILImage()(image_bytes)


def posemb_sincos_1d(n, dim, temperature=10000, dtype=torch.float32):
    """
    Creates positional embeddings for 1D patches using sin-cos positional embeddings
    Args:
        patches: 1D tensor of shape (B, N, D)
        temperature: temperature for positional embeddings
        dtype: dtype of the positional embeddings
    """

    n = torch.arange(n)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe.type(dtype)


def vis_torch_image(image, batch_idx, path):
    """
    Visualizes a torch tensor image of shape (B, C, H, W)
    """
    img_npy = image[batch_idx].cpu().detach().numpy()
    img_npy = img_npy.transpose(1, 2, 0)
    img_npy = (127.5 * (np.clip(img_npy, -1, 1) + 1)).astype(np.uint8)
    img = Image.fromarray(img_npy)
    img.save(path)


def spatial_transform(
    image, z_where, out_dims, inverse=False, padding_mode="border", mode="bilinear"
):
    """
    spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = (
        torch.zeros(2, 3, dtype=image.dtype)
        .repeat(image.shape[0], 1, 1)
        .to(image.device)
    )
    # set scaling
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-9)

    # set translation
    theta[:, 0, -1] = (
        z_where[:, 2] if not inverse else -z_where[:, 2] / (z_where[:, 0] + 1e-9)
    )
    theta[:, 1, -1] = (
        z_where[:, 3] if not inverse else -z_where[:, 3] / (z_where[:, 1] + 1e-9)
    )
    # 2. construct sampling grid
    grid = F.affine_grid(theta, out_dims, align_corners=False)
    grid = grid.to(image.dtype)
    # 3. sample image from grid
    return F.grid_sample(
        image, grid, align_corners=False, padding_mode=padding_mode, mode=mode
    )
