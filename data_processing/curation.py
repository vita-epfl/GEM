import argparse
import csv
import logging
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Process, Lock
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

import clip
import cv2
import h5py
import numpy as np
import open_clip
import torch
import torch.functional as F
import torch.nn as nn
from PIL import Image
from pypiqe import piqe
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import ToPILImage
from tqdm import tqdm

lock = Lock()
progress_log_path = "processed_files.log"

warnings.filterwarnings(
    "ignore",
    message=".*Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="xFormers is available (SwiGLU)"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="xFormers is available (Attention)"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="xFormers is available (Block)"
)


logging.basicConfig(level=logging.INFO)

to_pil = ToPILImage()

small_288 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(288),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

raft_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((520, 960)),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

dino_transform = transforms.Compose(
    [
        # center crop to smallest size
        # transforms.Lambda(lambda img: img.crop((0, 0, min(img.size), min(img.size)))),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize as required by ViT
    ]
)

raft_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((520, 960)),
        transforms.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
    ]
)


def get_sscd_model():
    model = torch.jit.load("ckpts/sscd_disc_advanced.torchscript.pt")
    return model


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


@torch.no_grad()
def get_aesthetic_score(
    model, amodel, images: np.ndarray, preprocess, device
) -> np.ndarray:
    """Get aesthetic score for a batch of images
    Args:
        images (np.ndarray) B, 3, H, W
    Returns:
        np.ndarray: B,
    """
    # preprocessed_image = preprocess([to_pil_image(image) for image in images])
    # preprocessed_image = torch.stack([preprocess(to_pil(image)) for image in images])
    preprocessed_image = preprocess(images).unsqueeze(0).to(device)
    image_features = model.encode_image(preprocessed_image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    prediction = amodel(image_features).detach().cpu().item()
    return prediction


def get_piqe_score(images: np.ndarray) -> np.ndarray:
    img = np.array(images)
    score, _, _, _ = piqe(img)
    return score


def measure_blur_in_roi(image, mask):
    image = np.array(image)  # /255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_image = gray * mask
    # save mask
    masked_image = masked_image.astype(np.uint8)
    # masked_image = cv2.bitwise_and(gray, gray, mask=mask)
    laplacian_var = cv2.Laplacian(masked_image, cv2.CV_64F).var()
    # num_pixels = np.count_nonzero(mask)
    # normalized_blur = (laplacian_var / num_pixels) ** 0.5 if num_pixels > 0 else 0

    return laplacian_var


def center_crop_to_smaller_side_PIL(img):
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    img = img.crop((left, top, right, bottom))
    return img


def center_crop_to_smaller_side(img):
    height, width, _ = img.shape
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2
    img = img[top:bottom, left:right]
    return img


def compute_average_flow(flow):
    """Compute the average optical flow vector magnitude."""
    # normalize the flow map
    if flow.ndim == 4:
        flow = flow.squeeze(0)
    magnitude = torch.sqrt(
        flow[0] ** 2 + flow[1] ** 2
    )  # Compute the magnitude of flow vectors
    return magnitude.mean().item()  # Return the mean magnitude


def compute_patchwise_average_flow(flow, grid_rows=1, grid_cols=3):
    """Compute the average optical flow vector magnitude over 2x4 grid patches."""
    if flow.ndim == 4:
        flow = flow.squeeze(0)
    h, w = flow.shape[1:]
    patch_height = h // grid_rows
    patch_width = w // grid_cols

    patch_averages = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            # Define the patch boundaries
            start_h = i * patch_height
            end_h = (i + 1) * patch_height
            start_w = j * patch_width
            end_w = (j + 1) * patch_width

            # Extract the patch from the flow map
            patch_flow = flow[:, start_h:end_h, start_w:end_w]

            # Compute the average of the flow vectors in the patch
            average_x = patch_flow[0].mean().item()
            average_y = patch_flow[1].mean().item()
            patch_average = [average_x, average_y]
            # Compute the magnitude of the flow vectors in the patch
            patch_averages.extend(patch_average)

            # magnitude = torch.sqrt(patch_flow[0] ** 2 + patch_flow[1] ** 2)
            # Compute the average magnitude for this patch
            # patch_avg = magnitude.mean().item()
            # patch_averages.append(patch_avg)

    return patch_averages  # Return the 8-dimensional vector (one value per patch)


@torch.no_grad()
def get_environment_scores(image, text_embeddings, clip_preprocess, clip_model, device):
    img_preprocessed = clip_preprocess(image).unsqueeze(0).to(device)
    clip_embedding = clip_model.encode_image(img_preprocessed)
    image_embedding = (
        clip_embedding.cpu().numpy()
    )  # Move to CPU and convert to numpy array

    similarities = np.dot(image_embedding, text_embeddings.T)
    similarities = similarities / (
        np.linalg.norm(image_embedding) * np.linalg.norm(text_embeddings, axis=1)
    )
    return similarities.squeeze()


@torch.no_grad()
def compute_optical_flow(
    raft_model, start_img, middle_img, end_img, raft_transform, device
):
    # if start_img.ndim == 3:
    #    start_imd, middle_img, end_img = start_img.unsqueeze(0), middle_img.unsqueeze(0), end_img.unsqueeze(0)

    start_img_tensor = raft_transform(start_img).to(device).unsqueeze(0)
    middle_img_tensor = raft_transform(middle_img).to(device).unsqueeze(0)
    end_img_tensor = raft_transform(end_img).to(device).unsqueeze(0)

    bs, _, h, w = start_img_tensor.size()

    source = torch.cat((start_img_tensor, middle_img_tensor), 0)
    target = torch.cat((middle_img_tensor, end_img_tensor), 0)

    flows = raft_model(source, target)[-1]
    h, w = flows.shape[-2:]
    flows = flows / torch.tensor([w / 2, h / 2]).view(1, 2, 1, 1).to(flows[0].device)

    # resize to start_img
    flows = torch.nn.functional.interpolate(
        flows, (start_img.size[1], start_img.size[0]), mode="bilinear"
    )

    return flows[:bs], flows[bs:]


def get_similarity_scores(
    similarity_features: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Returns the indices of images that have a cosine similarity lower than the threshold,
    i.e., those that are considered unique or not duplicates.

    Parameters:
    similarity_features (np.ndarray): A 2D numpy array of shape (num_images, feature_dim),
                                      where each row represents the feature vector of an image.
    threshold (float): The cosine similarity threshold above which images are considered duplicates.

    Returns:
    np.ndarray: An array containing the indices of images that are not duplicates.
    """
    # Step 1: Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(similarity_features)

    # Step 2: Initialize a set to keep track of indices to keep (non-duplicates)
    to_keep = set(range(similarity_features.shape[0]))
    num_images = similarity_matrix.shape[0]

    # Step 3: Check pairwise similarities and remove indices of images that are duplicates
    for i in range(num_images):
        for j in range(i + 1, num_images):
            if similarity_matrix[i, j] > threshold:
                # Mark the j-th image as a duplicate
                if j in to_keep:
                    to_keep.remove(j)

    # Step 4: Convert the set to a sorted numpy array and return it
    return np.array(sorted(to_keep))


# Define function to process a single video
def process_video(
    video_path,
    aesthetic_model,
    sscd_model,
    amodel,
    clip_model,
    dino_model,
    raft_model,
    text_embeddings,
    device,
    preprocess,
    clip_preprocess,
):
    torch.cuda.empty_cache()

    # image_paths = sorted(os.listdir(video_path))
    aesthetic_scores = []
    piqe_scores = []
    laplace_variances = []
    environment_scores = []
    average_flows = []
    dino_embeddings = []
    min_cosine_similarities = []
    average_cosine_similarities = []
    sscd_emmbeddings = []
    patchwise_flows = []

    n = 25
    results = []
    with h5py.File(video_path, "r") as f:
        num_frames = len(f["video"])
        timings = {
            "image_load_time": [],
            "aesthetic_time": [],
            "environment_time": [],
            "piqe_time": [],
            "optical_flow_time": [],
            "laplacian_time": [],
            "dino_time": [],
        }

        with torch.no_grad():
            # for i in range(0, num_frames - 2 * n, n):
            for i in tqdm(range(0, num_frames - 2 * n, n)):
                start = i
                middle = i + n // 2
                end = i + n - 1
                if end >= num_frames:
                    break
                # if i > 100:
                #    break

                # Start timing image load
                start_time = time.time()

                start_img = to_pil(f["video"][start])
                middle_img = to_pil(f["video"][middle])
                end_img = to_pil(f["video"][end])

                middle_img_crop = center_crop_to_smaller_side_PIL(middle_img)
                start_img_crop = center_crop_to_smaller_side_PIL(start_img)
                end_img_crop = center_crop_to_smaller_side_PIL(end_img)

                # End timing image load
                timings["image_load_time"].append(time.time() - start_time)

                # Timing aesthetic score computation
                start_time = time.time()

                middle_aesthetic = get_aesthetic_score(
                    aesthetic_model, amodel, middle_img_crop, preprocess, device
                )
                aesthetic_scores.append(middle_aesthetic)

                timings["aesthetic_time"].append(time.time() - start_time)

                # Timing environment score computation
                start_time = time.time()

                environment_score = get_environment_scores(
                    middle_img, text_embeddings, clip_preprocess, clip_model, device
                )
                environment_scores.append(environment_score.tolist())

                timings["environment_time"].append(time.time() - start_time)

                # Timing PIQE score computation
                start_time = time.time()

                middle_piqe = get_piqe_score(middle_img_crop)
                piqe_scores.append(middle_piqe)

                timings["piqe_time"].append(time.time() - start_time)

                # Timing optical flow computation
                start_time = time.time()

                flow_first_middle, flow_middle_end = compute_optical_flow(
                    raft_model, start_img, middle_img, end_img, raft_transform, device
                )

                average_flow_first_middle = compute_average_flow(flow_first_middle)
                average_flow_middle_end = compute_average_flow(flow_middle_end)
                average_flow = (average_flow_first_middle + average_flow_middle_end) / 2
                average_flows.append(average_flow)

                patchwise_flow_first_middle = compute_patchwise_average_flow(
                    flow_first_middle
                )
                patchwise_flow_middle_end = compute_patchwise_average_flow(
                    flow_middle_end
                )
                patchwise_flow = [
                    a + b
                    for a, b in zip(
                        patchwise_flow_first_middle, patchwise_flow_middle_end
                    )
                ]
                patchwise_flows.append(patchwise_flow)

                timings["optical_flow_time"].append(time.time() - start_time)

                # Timing Laplacian variance (blur measure)
                start_time = time.time()

                flow_magnitude = torch.sqrt(
                    (flow_middle_end[0, 0] ** 2) + (flow_middle_end[0, 1] ** 2)
                )
                motion_mask = (flow_magnitude > 0.20).float().cpu().squeeze().numpy()
                middle_laplacian = measure_blur_in_roi(middle_img, motion_mask)
                laplace_variances.append(middle_laplacian)

                timings["laplacian_time"].append(time.time() - start_time)

                # Timing DINO embedding extraction
                start_time = time.time()

                middle_img_dino = (
                    dino_transform(middle_img_crop).unsqueeze(0).to(device)
                )
                middle_dino_embedding = dino_model(middle_img_dino)
                dino_embeddings.append(middle_dino_embedding.cpu().numpy())
                timings["dino_time"].append(time.time() - start_time)

                ### patchwise dino
                first_last_dino = torch.stack(
                    [dino_transform(start_img_crop), dino_transform(end_img_crop)]
                ).to(device)
                dino_feats = dino_model.get_intermediate_layers(
                    first_last_dino, n=1, reshape=True
                )
                dino_feats = torch.cat(dino_feats, dim=0)
                # interpolate to 4x4
                # dino_feats = nn.functional.interpolate(dino_feats, size=(4, 4), mode='bilinear', align_corners=False)
                dino_feats = dino_feats.view(2, 384, -1).permute(0, 2, 1)
                cosine_similarity = nn.CosineSimilarity(dim=1)(
                    dino_feats[0], dino_feats[1]
                )
                # min_cosine_similarity = cosine_similarity.min().item()
                average_cosine_similarity = cosine_similarity.mean().item()
                min_cosine_similarity = (
                    cosine_similarity < 0.50
                ).sum().item() / cosine_similarity.size(0)
                min_cosine_similarities.append(min_cosine_similarity)
                average_cosine_similarities.append(average_cosine_similarity)

                # Compute SSCD embeddings
                middle_img_sscd = small_288(middle_img_crop).unsqueeze(0).to(device)
                middle_sscd_embedding = sscd_model(middle_img_sscd)
                sscd_emmbeddings.append(middle_sscd_embedding.cpu().numpy())

        # dino_idxs_98 = get_similarity_scores(np.squeeze(np.array(dino_embeddings), 1), threshold=0.98)
        # dino_idxs_95 = get_similarity_scores(np.squeeze(np.array(dino_embeddings), 1), threshold=0.95)
        # dino_idxs_90 = get_similarity_scores(np.squeeze(np.array(dino_embeddings), 1), threshold=0.90)
        # dino_idxs_85 = get_similarity_scores(np.squeeze(np.array(dino_embeddings), 1), threshold=0.85)
        # dino_idxs_75 = get_similarity_scores(np.squeeze(np.array(dino_embeddings), 1), threshold=0.75)

        # sscd_idxs_95 = get_similarity_scores(np.squeeze(np.array(sscd_emmbeddings), 1), threshold=0.95)
        # sscd_idxs_75 = get_similarity_scores(np.squeeze(np.array(sscd_emmbeddings), 1), threshold=0.75)
        # sscd_idxs_50 = get_similarity_scores(np.squeeze(np.array(sscd_emmbeddings), 1), threshold=0.50)

        for j in range(len(aesthetic_scores)):
            # start_image = os.path.join(video_path, image_paths[j * n])
            row = [
                # start_image,
                video_path,
                (j * n),
                n,
                aesthetic_scores[j],
                piqe_scores[j],
                laplace_variances[j],
                environment_scores[j],
                1,  # if j in dino_idxs_98 else 0,
                1,  # if j in dino_idxs_95 else 0,
                1,  # if j in dino_idxs_90 else 0,
                1,  # if j in dino_idxs_85 else 0,
                1,  # if j in dino_idxs_75 else 0,
                1,  # if j in sscd_idxs_95 else 0,
                1,  # if j in sscd_idxs_75 else 0,
                1,  # if j in sscd_idxs_50 else 0,
                min_cosine_similarities[j],
                average_cosine_similarities[j],
                average_flows[j],
                patchwise_flows[j],
            ]
            results.append(row)
    return results


#!/usr/bin/env python3


progress_log_path = "progress.log"


def log_progress(video_path):
    with open(progress_log_path, "a") as log_file:
        log_file.write(f"{video_path}\n")


def worker(device_id, video_paths, text_embeddings, annotation_file_folder):
    import os

    # Get the list of visible GPUs assigned by Slurm
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        devices = visible_devices.split(",")
        device = f"cuda:{device_id}" if len(devices) > device_id else f"cuda:0"
    else:
        device = f"cuda:0"
    torch.cuda.set_device(device)
    print(f"Process {os.environ.get('SLURM_PROCID')} using device {device}")
    # Load models once per process
    sscd_model = get_sscd_model().eval().to(device)
    amodel = get_aesthetic_model(clip_model="vit_l_14").eval().to(device)
    aesthetic_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    aesthetic_model.eval().to(device)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    dino_model = (
        torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
    )
    raft_model = raft_large(pretrained=True).eval().to(device)

    # Process each video in the assigned chunk
    # for video_path in tqdm(video_paths, desc="Processing videos"):
    for video_path in video_paths:
        try:
            results = process_video(
                video_path,
                aesthetic_model,
                sscd_model,
                amodel,
                clip_model,
                dino_model,
                raft_model,
                text_embeddings,
                device,
                preprocess,
                clip_preprocess,
            )

            # Save results as before
            base_folder = os.path.dirname(video_path)
            video_name = "curation_" + os.path.basename(video_path).replace(
                ".h5", ".csv"
            )
            csv_file_path = os.path.join(base_folder, video_name)

            headers = [
                "h5_path",
                "idx",
                "num_frames",
                "aesthetic_score",
                "PIQE_score",
                "blur_score",
                "environment_score",
                "in_similarity_98",
                "in_similarity_95",
                "in_similarity_90",
                "in_similarity_85",
                "in_similarity_75",
                "sscd_similarity_95",
                "sscd_similarity_75",
                "sscd_similarity_50",
                "patch_similarity_50",
                "average_cosine_similarity",
                "motion_score",
                "patchwise_motion_vectors",
            ]

            with open(csv_file_path, mode="w+", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headers)
                writer.writerows(results)

            print(f"Data for {video_name} saved to {csv_file_path}")
            log_progress(video_path)  # Log the processed video path

        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")


def parallel_video_processing(
    video_base_paths,
    start_index,
    end_index,
    text_embeddings,
    annotation_file_folder=None,
):
    # Collect all video paths
    video_paths = []
    for video_base_path in video_base_paths:
        video_paths.extend(
            [os.path.join(video_base_path, vp) for vp in os.listdir(video_base_path)]
        )

    # Slice the video paths list to process only the specified range
    video_paths = sorted(video_paths)[start_index:end_index]

    print(f"Total videos to process: {len(video_paths)}")

    # Get the rank and world size from environment variables
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

    # Debugging prints
    print(f"SLURM_PROCID: {rank}")
    print(f"SLURM_NTASKS: {world_size}")
    print(f"SLURM_LOCALID: {local_rank}")
    print(f"SLURM_NODEID: {os.environ.get('SLURM_NODEID')}")

    # Each process will process a subset of video_paths
    # Split video_paths among world_size processes
    video_paths_chunk = [
        video_paths[i] for i in range(rank, len(video_paths), world_size)
    ]

    print(f"Process {rank}/{world_size} processing {len(video_paths_chunk)} videos")

    # Set device_id to local_rank
    device_id = local_rank

    # Now call worker with the assigned video_paths_chunk
    worker(device_id, video_paths_chunk, text_embeddings, annotation_file_folder)


if __name__ == "__main__":
    # Define base directory and available H5 subdirectories
    BASE_DIR = Path("/capstor/store/cscs/swissai/a03/datasets")
    # BASE_DIR = Path("/capstor/scratch/cscs/pmartell/datasets/")
    H5_DIRS = [
    ]

    # Load text embeddings
    text_embeddings = torch.load("outputs/text_embeddings.pt")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process H5 video files.")
    parser.add_argument(
        "--h5_dirs",
        nargs="+",
        default=H5_DIRS,
        help="List of H5 video directories within the base directory (defaults to all in H5_DIRS)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for processing (default: 0)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="Ending index for processing (default: -1, to process all)",
    )

    args = parser.parse_args()

    # Construct full paths by adding the base directory to each specified folder name
    video_base_paths = [str(BASE_DIR / h5_dir) for h5_dir in args.h5_dirs]

    # Initialize the log file
    with open(progress_log_path, "a+") as log_file:
        log_file.write("Processed H5 files:\n")

    # Pass command-line arguments to processing function
    parallel_video_processing(
        video_base_paths,
        args.start_idx,
        args.end_idx if args.end_idx != -1 else None,
        text_embeddings,
    )
