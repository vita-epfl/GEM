import torch
import lietorch
import droid_backends

from .droid_net import cvx_upsample
from .geom import projective_ops as pops


class DepthVideo:
    def __init__(self, device, image_size=[480, 640], buffer=1024, stereo=False):

        self.device = device

        # current keyframe count
        self.counter = 0  # Value("i", 0)
        self.ready = 0  # Value("i", 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(
            buffer, device=device, dtype=torch.float
        )  # .share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.uint8)
        self.dirty = torch.zeros(
            buffer, device=device, dtype=torch.bool
        )  # .share_memory_()
        self.red = torch.zeros(
            buffer, device=device, dtype=torch.bool
        )  # .share_memory_()
        self.poses = torch.zeros(
            buffer, 7, device=device, dtype=torch.float
        )  # .share_memory_()
        self.disps = torch.ones(
            buffer, ht // 8, wd // 8, device=device, dtype=torch.float
        )  # .share_memory_()
        self.disps_sens = torch.zeros(
            buffer, ht // 8, wd // 8, device=device, dtype=torch.float
        )  # .share_memory_()
        self.disps_up = torch.zeros(
            buffer, ht, wd, device=device, dtype=torch.float
        )  # .share_memory_()
        self.intrinsics = torch.zeros(
            buffer, 4, device=device, dtype=torch.float
        )  # .share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(
            buffer, c, 128, ht // 8, wd // 8, dtype=torch.half, device=device
        )  # .share_memory_()
        self.nets = torch.zeros(
            buffer, 128, ht // 8, wd // 8, dtype=torch.half, device=device
        )  # .share_memory_()
        self.inps = torch.zeros(
            buffer, 128, ht // 8, wd // 8, dtype=torch.half, device=device
        )  # .share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor(
            [0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device
        )

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter:
            self.counter = index + 1

        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter:
            self.counter = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8, 3::8]
            self.disps_sens[index] = torch.where(depth > 0, 1.0 / depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        self.__item_setter(index, item)

    def __getitem__(self, index):
        """index the depth video"""

        # support negative indexing
        if isinstance(index, int) and index < 0:
            index = self.counter + index

        item = (
            self.poses[index],
            self.disps[index],
            self.intrinsics[index],
            self.fmaps[index],
            self.nets[index],
            self.inps[index],
        )

        return item

    def append(self, *item):
        self.__item_setter(self.counter, item)

    ### geometric operations ###

    def format_indicies(self, ii, jj):
        """to device, long, {-1}"""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=self.device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=self.device, dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """upsample disparity"""

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """normalize depth and poses"""

        s = self.disps[: self.counter].mean()
        self.disps[: self.counter] /= s
        self.poses[: self.counter, :3] *= s
        self.dirty[: self.counter] = True

    def reproject(self, ii, jj):
        """project points from ii -> jj"""
        ii, jj = self.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = pops.projective_transform(
            Gs, self.disps[None], self.intrinsics[None], ii, jj
        )

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """frame distance metric"""

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")

        ii, jj = self.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[: self.counter].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta
            )

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta
            )

            d = 0.5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta
            )

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(
        self,
        target,
        weight,
        eta,
        ii,
        jj,
        t0=1,
        t1=None,
        itrs=2,
        lm=1e-4,
        ep=0.1,
        motion_only=False,
    ):
        """dense bundle adjustment (DBA)"""

        # with self.get_lock():

        # [t0, t1] window of bundle adjustment optimization
        if t1 is None:
            t1 = max(ii.max().item(), jj.max().item()) + 1

        droid_backends.ba(
            self.poses,
            self.disps,
            self.intrinsics[0],
            self.disps_sens,
            target,
            weight,
            eta,
            ii,
            jj,
            t0,
            t1,
            itrs,
            lm,
            ep,
            motion_only,
        )

        self.disps.clamp_(min=0.001)
