import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from geocalib import GeoCalib
from depth_anything_v2.metric_depth.depth_anything_v2.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)


class BBoxExtractor:

    def __init__(self,
                 image_height=576,
                 image_width=1024,
                 encoder="vitl",
                 weights_dir=".",
                 seg_batch_size=8):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_height = image_height
        self.image_width = image_width
        self.weights_dir = weights_dir
        self.seg_batch_size = seg_batch_size

        # load calib model
        self.calib_model = GeoCalib()
        self.calib_model = self.calib_model.to(self.device)

        # load depth model
        dataset = "vkitti"  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 80  # 20 for indoor model, 80 for outdoor model
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }
        self.depth_model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
        self.depth_model.load_state_dict(
            torch.load(
                os.path.join(
                    weights_dir,
                    f"depth_anything_v2_metric_{dataset}_{encoder}.pth",
                ),
                map_location="cpu",
            )
        )
        self.depth_model = self.depth_model.to(self.device).eval()
        self.depth_transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        # load segment model
        self.image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.seg_model.to(self.device)

    def __call__(self, video):
        calib_matrix = self.do_calib(video)
        depth_video = self.do_depth(video)
        segmentation_video = self.do_segmentation(video)

        road_mask = segmentation_video == 0
        depth_video[~road_mask] = torch.nan

        ## add the bbox stuff
        car_width = 1.7  # in meters, randomize?
        avg_distance_per_h = torch.nanmean(depth_video, dim=1)  # N x H
        object_pixel_width_per_h = calib_matrix[0, 0] * car_width / avg_distance_per_h  # N x H
        # etc...

    @torch.no_grad()
    def do_calib(self, video):
        video = (video.transpose(0, 3, 1, 2) / 255.0).astype(np.float32)
        video = torch.from_numpy(video).to(self.device)
        result = self.calib_model.calibrate(
            video,
            camera_model="pinhole",
            shared_intrinsics=True,
        )
        intrinsics_np = result["camera"].K.mean(dim=0).cpu().numpy()
        return intrinsics_np

    @torch.no_grad()
    def do_depth(self, video):
        video = video / 255.0
        video = np.array([self.depth_transform({"image": d})["image"] for d in video])
        video = torch.from_numpy(video).to(self.device)
        predictions = self.depth_model.forward(video)
        predictions = nn.functional.interpolate(
            predictions[:, None],
            [self.image_height, self.image_width],
            mode="bilinear",
            align_corners=True,
        )[:, 0]
        predictions_np = predictions.cpu().numpy()
        return predictions_np

    @torch.no_grad()
    def do_segmentation(self, video):
        n_frames = len(video)
        i = 0
        segs = []
        while i < n_frames:
            batch = video[i:min(i+self.seg_batch_size, n_frames)]

            # make image quadratic
            left = batch[:, :, :self.image_height]
            right = batch[:, :, -self.image_height:]

            with torch.no_grad():
                inputs_left = self.image_processor(images=left, return_tensors="pt").pixel_values.to(self.device)
                outputs_left = self.seg_model(inputs_left)
                logits_left = nn.functional.interpolate(outputs_left.logits, [self.image_height, self.image_height], mode='bilinear', align_corners=False)  # B x 19 x H x H
                logits_left = nn.functional.pad(logits_left, (0, self.image_width - self.image_height, 0, 0), value=torch.nan)  # B x 19 x H x W

                inputs_right = self.image_processor(images=right, return_tensors="pt").pixel_values.to(self.device)
                outputs_right = self.seg_model(inputs_right)
                logits_right = nn.functional.interpolate(outputs_right.logits, [self.image_height, self.image_height], mode='bilinear', align_corners=False)  # B x 19 x H x H
                logits_right = nn.functional.pad(logits_right, (self.image_width - self.image_height, 0, 0, 0), value=torch.nan)  # B x 19 x H x W

                logits = torch.nanmean(torch.stack([logits_left, logits_right], dim=0), dim=0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            segs.append(preds)
            i += batch_size
        return np.stack(segs, axis=0)


if __name__ == "__main__":

    import h5py
    import matplotlib.pyplot as plt

    file_paths = [
        "$SCRATCH/generated/chunk_0.h5",
        "$SCRATCH/generated/chunk_1.h5",
        "$SCRATCH/generated/chunk_2.h5",
        "$SCRATCH/generated/chunk_3.h5",
        "$SCRATCH/generated/chunk_4.h5",
    ]

    extractor = BBoxExtractor(encoder="vitl", weights_dir="/capstor/scratch/cscs/pmartell/trajectory_inference/weights")

    for file_path in file_paths:
        print(f"doing {file_path}...")
        with h5py.File(file_path, "r") as f:
            video = f["video"][:]

        # feed in video as N x H x W x 3
        ret = extractor(video)

