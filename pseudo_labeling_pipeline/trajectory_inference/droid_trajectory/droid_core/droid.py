import torch
import numpy as np

from .droid_net import DroidNet
from .depth_video import DepthVideo
from .motion_filter import MotionFilter
from .droid_frontend import DroidFrontend
from .droid_backend import DroidBackend
from .trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict


class Droid:
    def __init__(
        self,
        weights,
        image_size,
        device,
        upsample=False,
        buffer=512,
        stereo=False,
        filter_thresh=2.4,
        frontend_thresh=16.0,
        backend_thresh=22.0,
        keyframe_thresh=4.0,
        vis_save=False,
    ):
        super(Droid, self).__init__()
        self.device = device
        self.load_weights(weights)

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(
            image_size=image_size, buffer=buffer, stereo=stereo, device=device
        )

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(
            self.net, self.video, device=device, thresh=filter_thresh
        )

        # frontend process
        self.frontend = DroidFrontend(
            self.net, self.video, device=device, upsample=upsample, keyframe_thresh=keyframe_thresh, frontend_thresh=frontend_thresh
        )

        # backend process
        self.backend = DroidBackend(
            self.net, self.video, device=device, upsample=upsample, backend_thresh=backend_thresh
        )

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video, device=device)

    def load_weights(self, weights):
        """load trained model weights"""

        # print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict(
            [(k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()]
        )

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to(self.device).eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """main thread - update map"""

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None):
        """terminate the visualization process, return timestamp and poses [t, q]"""

        del self.frontend

        # torch.cuda.empty_cache()
        # print("#" * 32)
        self.backend(7)

        # torch.cuda.empty_cache()
        # print("#" * 32)
        self.backend(20)

        camera_trajectory = self.traj_filler(stream)
        camera_trajectory = camera_trajectory.inv().data.cpu().numpy()

        # fill timestamp
        timestamps = np.arange(len(camera_trajectory)).reshape(-1, 1)
        traj_tum = np.concatenate([timestamps, camera_trajectory], axis=1)

        return traj_tum
