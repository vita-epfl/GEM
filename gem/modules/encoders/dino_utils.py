import torch
import torch.nn.functional as F


def drop_from_masks(valid_masks, num_conditioning_tokens):
    """
    Randomly adjust the valid_masks such that each row in the batch has exactly num_conditioning_tokens 1s.
    If a row has less than num_conditioning_tokens 1s, randomly add 1s to the row.

    Args:
    valid_masks (torch.Tensor): Tensor of shape (b, n) where each element is 0 or 1.
    num_conditioning_tokens (int): The desired number of 1s in each batch row.

    Returns:
    torch.Tensor: Adjusted valid_masks with exactly num_conditioning_tokens 1s in each row.
    """
    b, n = valid_masks.shape

    if type(num_conditioning_tokens) == int:
        num_conditioning_tokens = torch.tensor([num_conditioning_tokens])

    if torch.any(num_conditioning_tokens > n):
        raise ValueError(
            f"num_conditioning_tokens {num_conditioning_tokens} cannot be greater than the number of tokens {n}"
        )
    if num_conditioning_tokens.size(0) != b:
        num_conditioning_tokens = num_conditioning_tokens.repeat(b)

    for i in range(b):
        # Get indices of the 1s in the current mask
        ones_indices = torch.where(valid_masks[i] == 1)[0]

        # Randomly drop 1s if there are more than needed
        if len(ones_indices) > num_conditioning_tokens[i]:
            # Randomly select excess 1s to turn off
            drop_indices = torch.randperm(len(ones_indices))[
                : len(ones_indices) - num_conditioning_tokens[i]
            ]
            valid_masks[i, ones_indices[drop_indices]] = 0
        elif len(ones_indices) == num_conditioning_tokens[i]:
            continue
        else:
            # Randomly add 1s if there are less than needed
            zero_indices = torch.where(valid_masks[i] == 0)[0]
            num_zeros_to_flip = num_conditioning_tokens[i] - len(ones_indices)
            if num_zeros_to_flip > len(zero_indices):
                raise ValueError(
                    f"Not enough zeros to flip in row {i} to reach {num_conditioning_tokens[i]} ones."
                )
            flip_indices = zero_indices[
                torch.randperm(len(zero_indices))[:num_zeros_to_flip]
            ]
            valid_masks[i, flip_indices] = 1

    return valid_masks


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


# ImageNet_Norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


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


def create_crop_region(
    min_w: float = 0.3,
    min_h: float = 0.3,
    max_w: float = 0.8,
    max_h: float = 0.5,
    num_target_frames: int = 4,
    batched_lengths: bool = False,
    height: int = 224,
    width: int = 224,
    scale_aspect_ratio: bool = False,
):
    # Pre-compute differences
    # if scale_aspect_ratio:
    #     aspect_ratio = width / height
    #     min_w = min_w / aspect_ratio
    #     max_w = max_w / aspect_ratio

    w_range = max_w - min_w
    h_range = max_h - min_h

    # Create tensors to store random values all at once
    if batched_lengths:
        crop_w = torch.rand(1) * w_range + min_w
        crop_w = crop_w.repeat(num_target_frames)
        crop_h = torch.rand(1) * h_range + min_h
        crop_h = crop_h.repeat(num_target_frames)
    else:
        crop_w = torch.rand(num_target_frames) * w_range + min_w
        crop_h = torch.rand(num_target_frames) * h_range + min_h

    # scale crop_w to match aspect ratio
    # if scale_aspect_ratio:
    #     crop_w = crop_w * (height / width)

    # Compute min_x, max_x, min_y, max_y for all frames
    min_x = -1 + crop_w / 2
    max_x = 1 - crop_w / 2
    min_y = -1 + crop_h / 2
    max_y = 1 - crop_h / 2

    # Generate random x and y positions
    x = torch.rand(num_target_frames) * (max_x - min_x) + min_x
    y = torch.rand(num_target_frames) * (max_y - min_y) + min_y

    # Stack into a tensor of shape (num_target_frames, 1, 4)
    z_where = torch.stack([crop_w, crop_h, x, y], dim=1)

    # make h, w in z_where times H, W divisible by 14
    z_where[:, 0] = (
        (z_where[:, 0] * width).int() - ((z_where[:, 0] * width).int() % 14)
    ) / width
    z_where[:, 1] = (
        (z_where[:, 1] * height).int() - ((z_where[:, 1] * height).int() % 14)
    ) / height

    return z_where
