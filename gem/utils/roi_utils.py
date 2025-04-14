import torch
import torchvision
from dataclasses import dataclass
import matplotlib.pyplot as plt
from .visualization_utils import fig2img
from .visualization_utils import pil_image_to_torch
import matplotlib.patches as patches
import numpy as np


@dataclass
class RectangleRegion:
    x: float
    y: float
    w: float
    h: float


ROI_CLASS_CAR = 3

ROI_CLASSES = [
    # 1,  # person
    # 2,  # bicycle
    ROI_CLASS_CAR,  # car
    # 4,  # motorcycle
    # 6,  # bus
    # 8,  # truck
    # 10,  # traffic light
]


def create_controls_overlay(box_to_move, dino_box, vec_x, vec_y, frames):
    """
    Adds an overlay to the input frames that contains:
     1. the region being moved
     2. the region that is being conditioned on
     3. an arrow indicating the direction of movement

    Args:
        box_to_move (list): bounding box coordinates of the region being moved
        dino_box (list): bounding box coordinates of the region being conditioned on
        vec_x (float): Vec_x is the x component of the movement vector
        vec_y (float): Vec_y is the y component of the movement vector
        frames (np.ndarray): [T, C, H, W] tensor containing the input frames
    Returns:
        frames_with_overlay (torch.Tensor): [T, C, H, W] tensor containing the input frames with the overlay
    """
    result = []
    T, C, H, W = frames.shape
    box = box_to_move
    for t in range(T):
        fig, ax = plt.subplots(1)
        ax.imshow(np.transpose(frames[t], (1, 2, 0)))
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            (dino_box[0], dino_box[1]),
            dino_box[2] - dino_box[0],
            dino_box[3] - dino_box[1],
            linewidth=1,
            edgecolor="b",
            facecolor="none",
            linestyle="dashed",
        )
        ax.add_patch(rect)
        # draw arrow from center of box to center of box + vec
        ax.arrow(
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
            vec_x * 14,
            vec_y * 14,
            head_width=6,
            head_length=6,
            fc="k",
            ec="k",
        )
        plt.axis("off")
        img = fig2img(fig)
        plt.close(fig)
        result.append(pil_image_to_torch(img))

    result = torch.stack(result, dim=0)
    return result


def create_roi_model(device: str = "cuda:0"):
    bbox_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True
    ).to(device)
    bbox_model.requires_grad_(False)
    bbox_model.eval()
    return bbox_model


def roi_to_rectangle_region(roi, frame_width, frame_height):
    """
    Converts an ROI to a normalized rectangle region.

    Parameters:
        roi: A tensor of the bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        frame_width: The width of the video frame.
        frame_height: The height of the video frame.

    Returns:
        A list [w, h, x, y] where:
            - w, h are normalized width and height of the bounding box in the range [0, 1]
            - x, y are normalized center coordinates of the bounding box in the range [-1, 1]
    """
    x_min, y_min, x_max, y_max = roi

    # Calculate width and height of the bounding box
    box_width = (x_max - x_min) / frame_width
    box_height = (y_max - y_min) / frame_height

    # Calculate center of the bounding box
    center_x = ((x_min + x_max) / 2) / frame_width * 2 - 1
    center_y = ((y_min + y_max) / 2) / frame_height * 2 - 1

    return [box_width, box_height, center_x, center_y]

def rectangle_region_to_roi(region, pil_frame):
    # roi is x1, y1, x2, y2 in pixels
    # RectangleRegion is w, h as percentage of image size
    # and x, y go from -1 to 1, where 0 is the center
    w, h = pil_frame.size
    x1 = (region.x + 1) / 2 * w - region.w * w / 2
    y1 = (region.y + 1) / 2 * h - region.h * h / 2
    x2 = x1 + region.w * w
    y2 = y1 + region.h * h
    return (x1, y1, x2, y2)


@torch.no_grad()
def detect_roi_objects(model, image_tensor, threshold=0.5, roi_classes=ROI_CLASSES):
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    boxes = predictions["boxes"][predictions["scores"] > threshold]
    labels = predictions["labels"][predictions["scores"] > threshold]
    scores = predictions["scores"][predictions["scores"] > threshold]
    if len(boxes) == 0:
        return [], [], []
    else:
        boxes, labels, scores = filter_boxes(boxes, labels, scores, roi_classes)
        return boxes, labels, scores


@torch.no_grad()
def filter_boxes(boxes, labels, scores, roi_classes=ROI_CLASSES):
    allowed_boxes = []
    allowed_labels = []
    allowed_scores = []

    for i, label in enumerate(labels):
        if label in roi_classes:
            allowed_boxes.append(boxes[i])
            allowed_labels.append(labels[i])
            allowed_scores.append(scores[i])

    return allowed_boxes, allowed_labels, allowed_scores
