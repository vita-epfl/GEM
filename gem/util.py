import functools
import importlib
import os
from functools import partial
from inspect import isfunction

import fsspec
import torch
from einops import repeat


def visualize_dino_features(features, output_dir=".", threshold_percentile=0):
    """
    Visualizes DINO features by reducing them to RGB images using PCA and removing the background
    via thresholding the first PCA component for each image individually.

    Parameters:
    - features: Tensor
        The features output from the DINO model with shape [B, D, H, W].
    - output_dir: str
        Directory where the images will be saved.
    - threshold_percentile: float
        The percentile to use for thresholding the first PCA component to remove the background.

    Returns:
    - None
    """

    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # features shape: [B, D, H, W]
    B, D, H, W = features.shape
    N = H * W  # Total number of patches per image

    # Reshape features to [B, D, N]
    features = features.view(B, D, N)
    # Transpose to [B, N, D]
    features = features.permute(0, 2, 1)  # Now features.shape is [B, N, D]

    for idx in range(B):
        # Extract features for the current image
        image_features = features[idx]  # Shape: [N, D]

        # Convert to numpy array
        image_features_np = image_features.cpu().numpy()

        # Apply PCA to reduce features to 3 components
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(image_features_np)  # Shape: [N, 3]

        # Extract the first PCA component for thresholding
        first_component = features_pca[:, 0]  # Shape: [N]

        # # Determine threshold for background removal
        threshold = np.percentile(first_component, threshold_percentile)
        # # Create mask
        mask = first_component > threshold  # Shape: [N]

        # Normalize the PCA components to [0, 1] for visualization
        features_min = features_pca.min(axis=0, keepdims=True)
        features_max = features_pca.max(axis=0, keepdims=True)
        features_norm = (features_pca - features_min) / (
            features_max - features_min + 1e-5
        )

        # Apply mask to features
        features_masked = features_norm * mask[:, np.newaxis]

        # Reshape to image grid: [H, W, 3]
        features_image = features_masked.reshape(H, W, 3)

        # Save image
        plt.imshow(features_image)
        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, f"dino_feature_{idx}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


def disabled_train(self, mode=True):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore.
    """

    return self


def get_string_from_tuple(s):
    try:
        # check if the string starts and ends with parentheses
        if s[0] == "(" and s[-1] == ")":
            # convert the string to a tuple
            t = eval(s)
            # check if the type of t is tuple
            if isinstance(t, tuple):
                return t[0]
            else:
                pass
    except:
        pass
    return s


def is_power_of_two(n):
    """
    Return True if n is a power of 2, otherwise return False.
    """

    if n <= 0:
        return False
    else:
        return (n & (n - 1)) == 0


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


def repeat_as_img_seq(x, num_frames):
    if x is not None:
        if isinstance(x, list):
            new_x = list()
            for item_x in x:
                new_x += [item_x] * num_frames
            return new_x
        else:
            x = x.unsqueeze(1)
            x = repeat(x, "b 1 ... -> (b t) ...", t=num_frames)
            return x
    else:
        return None


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    else:
        return path


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def isheatmap(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return x.ndim == 2


def isneighbors(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


def exists(x):
    return x is not None


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """

    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        else:
            raise KeyError("Expected key `target` to instantiate")
    else:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return torch.cat((x, x.new_zeros([1])))


def append_dims(x, target_dims):
    """
    Appends dimensions to the end of a tensor until it has target_dims dimensions.
    """

    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"Input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_configs_path() -> str:
    """
    Get the `configs` directory.
    """

    this_dir = os.path.dirname(__file__)
    candidates = (
        os.path.join(this_dir, "configs"),
        os.path.join(this_dir, "..", "configs"),
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find configs in {candidates}")
