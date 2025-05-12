import torch.nn as nn
import torch
import numpy as np
import os
import argparse
import torch.nn.functional as F
from utils.model_utils import *
from RAFT.core.raft import RAFT
from second.pytorch.models.pointpillars import PillarFeatureNet, PointPillarsScatter
from utils.model_utils.slim_utils import output_decoder
from second.core.voxel_generator import VoxelGenerator


class SLIM(nn.Module):

    def __init__(self, args):
        super(SLIM, self).__init__()

        raft_args = self.add_raft_args()
        pfn_raft_channels = 16

        self.fpn0 = PillarFeatureNet(num_input_features=6, use_norm=True, num_filters=(pfn_raft_channels,))
        self.fpn1 = PillarFeatureNet(num_input_features=6, use_norm=True, num_filters=(pfn_raft_channels,))

        self.pps1 = PointPillarsScatter(torch.tensor([160, 160, 64]), num_input_features=pfn_raft_channels)
        self.pps2 = PointPillarsScatter(torch.tensor([160, 160, 64]), num_input_features=pfn_raft_channels)

        self.raft = RAFT(raft_args, pfn_raft_channels)

    def forward(self, pc1, pc2, feature1, feature2, p_stat=0.31):
        """
        SLIM model class

        :param pc1: (B * 3 * N)
        :param pc2: (B * 3 * M)
        :param feature1: (B * 3 * N)
        :param feature2: (B * 3 * M)
        :param p_stat: threshold const
        :param batch_size: batch size
        """

        I0b = []
        I1b = []

        voxels0b = []
        voxels1b = []
        coors0b = []
        coors1b = []

        for batch_i in range(pc1.size(0)):

            pc1_ = pc1[batch_i]
            pc2_ = pc2[batch_i]
            feature1_ = feature1[batch_i]
            feature2_ = feature2[batch_i]

            pc1_ = torch.transpose(pc1_, 0, 1)
            pc2_ = torch.transpose(pc2_, 0, 1)
            feature1_ = torch.transpose(feature1_, 0, 1)
            feature2_ = torch.transpose(feature2_, 0, 1)

            # Concat to a (6 * N) feature tensor
            input0 = torch.cat([pc1_, feature1_], dim=1)
            input1 = torch.cat([pc2_, feature2_], dim=1)
            st_time = time()
            vg = VoxelGenerator(list([0.15625*4, 0.15625*4, 0.15625*4]), [0, -50, -10, 100, 50, 10], 10, 12000)
            voxels0, coors0, num_points_per_voxel0 = vg.generate(input0, 12000)
            voxels1, coors1, num_points_per_voxel1 = vg.generate(input1, 12000)
            end_time = time()
            voxels0b.append(voxels0)
            voxels1b.append(voxels1)
            coors0b.append(coors0)
            coors1b.append(coors1)

            # into the edited PFN
            I0 = self.fpn0(voxels0, num_points_per_voxel0, coors0)
            I1 = self.fpn1(voxels1, num_points_per_voxel1, coors1)

            I0 = I0.squeeze(0)
            I1 = I1.squeeze(0)
            # pc1 = pc1.squeeze(0)
            # pc2 = pc2.squeeze(0)
            # into the edited pillar scatter
            I0 = self.pps1(I0, torch.tensor(coors0).cuda(), 1)
            I1 = self.pps1(I1, torch.tensor(coors1).cuda(), 1)

            I0b.append(I0)
            I1b.append(I1)

        I0b = torch.cat(I0b, dim=0)
        I1b = torch.cat(I1b, dim=0)

        # Into the edited RAFT
        f1, L_wgt, L_cls = self.raft(I0b, I1b, 6)
        f2, L_wgt2, L_cls2 = self.raft(I1b, I0b, 6)
        L_cls = L_cls.squeeze(1) #[B,160,160]
        L_cls2 = L_cls2.squeeze(1)

        Tr, L_cls, L_wgt, F1, pc1m = output_decoder(pc1, voxels0b, coors0b, f1, L_cls, L_wgt, p_stat)
        Tr2, L_cls2, L_wgt2, F2, pc2m = output_decoder(pc2, voxels1b, coors1b, f2, L_cls2, L_wgt2, p_stat)

        return Tr, Tr2, L_cls, L_wgt, F1, pc1m, pc2m

    def add_raft_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', default='raft', help="name your experiment")
        parser.add_argument('--stage', help="determines which dataset to use for training")
        parser.add_argument('--restore_ckpt', help="restore checkpoint")
        parser.add_argument('--basic', action='store_false', help='use small model')
        parser.add_argument('--validation', type=str, nargs='+')

        parser.add_argument('--lr', type=float, default=0.00002)
        parser.add_argument('--num_steps', type=int, default=100000)
        parser.add_argument('--batch_size', type=int, default=6)
        parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
        parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

        parser.add_argument('--iters', type=int, default=12)
        parser.add_argument('--wdecay', type=float, default=.00005)
        parser.add_argument('--epsilon', type=float, default=1e-8)
        parser.add_argument('--clip', type=float, default=1.0)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
        parser.add_argument('--add_noise', action='store_true')
        raft_args = parser.parse_args()

        return raft_args
