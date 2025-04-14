import torch

from .factor_graph import FactorGraph


class DroidBackend:
    def __init__(
        self,
        net,
        video,
        device,
        upsample=False,
        beta=0.3,
        backend_thresh=22.0,
        backend_radius=2,
        backend_nms=3,
    ):
        self.video = video
        self.device = device
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = upsample
        self.beta = beta
        self.backend_thresh = backend_thresh
        self.backend_radius = backend_radius
        self.backend_nms = backend_nms

    @torch.no_grad()
    def __call__(self, steps=12):
        """main update"""

        t = self.video.counter
        if not self.video.stereo and not torch.any(self.video.disps_sens):
            self.video.normalize()

        graph = FactorGraph(
            self.video,
            self.update_op,
            corr_impl="alt",
            max_factors=16 * t,
            upsample=self.upsample,
            device=self.device,
        )

        graph.add_proximity_factors(
            rad=self.backend_radius,
            nms=self.backend_nms,
            thresh=self.backend_thresh,
            beta=self.beta,
        )

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True
