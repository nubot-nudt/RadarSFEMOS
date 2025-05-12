"""
FLOT model from RigidFlow.
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import time
from model.util import PointNetSetAbstractionFLOT, knn_point, se3_transform, batch_mat2xyzrpy
from model.ot import sinkhorn
from model.pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from model.transformer import TransformerBlock


def compute_rmse(source,target,T):
    source_to_target = se3_transform(T, source)
    error = source_to_target - target
    rmse = torch.linalg.norm(error, dim=2)
    rmse = torch.sum(rmse, dim=1)
    return rmse


def compute_rigid_transform_with_ransac(source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        source (torch.Tensor): (B, M, 3) points
        target (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from source to target, i.e. T * source = target
    """
    # [B,N,1]
    batch_weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-8)
    # add RANSAC
    assert source.shape == target.shape
    max_iteration = 50
    sample_num = 10
    points_num = source.shape[1]
    optimal_rmse = torch.inf
    for i in range(max_iteration):
        random_index = torch.randperm(points_num)[:sample_num]
        a = source[:,random_index,:]
        b = target[:,random_index,:]
        weights_normalized = batch_weights_normalized[:,random_index,:]
        centroid_a = torch.sum(a * weights_normalized, dim=1)
        centroid_b = torch.sum(b * weights_normalized, dim=1)
        # [B,N,3]
        a_centered = a - centroid_a[:, None, :]
        b_centered = b - centroid_b[:, None, :]
        # [B,3,3]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

        # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # and choose based on determinant to avoid flips
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
        rot_mat_pos = v @ u.transpose(-1, -2)
        v_neg = v.clone()
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2)
        # [B,3,3]
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        # [B,3,1]
        translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]
        T = torch.cat((rot_mat, translation), dim=2)
        rmse = compute_rmse(source, target, T)
        rmse = torch.mean(rmse)
        if not i:
            optimal_rmse = rmse
            optimal_R = rot_mat
            optimal_T = translation
        else:
            if rmse < optimal_rmse:
                optimal_rmse = rmse
                optimal_R = rot_mat
                optimal_T = translation

    transform = torch.cat((optimal_R, optimal_T), dim=2)
    return transform


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-8)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform



class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_points, cfg.nblocks, cfg.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.transformer_dim, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_points, cfg.nblocks, cfg.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, cfg.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, cfg.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, pc1, pc2, flow):
        x = torch.cat((pc1, pc1 - pc2, flow), dim=2)
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
            
        return self.fc3(points)


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 32, 1)
        self.conv3 = torch.nn.Conv1d(32, 16, 1)
        self.conv4 = torch.nn.Conv1d(16, self.k, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        # self.mos_sa1 = PointNetSetAbstraction(nsample=32, in_channel=3, mlp=[32, 32, 32])
        # self.mos_sa2 = PointNetSetAbstraction(nsample=32, in_channel=32, mlp=[64, 64, 64])
        # self.mos_sa3 = PointNetSetAbstraction(nsample=32, in_channel=64, mlp=[128, 128, 128])

    def forward(self, x, flow, rigid_transform):
        transformed_x = se3_transform(rigid_transform, x)
        ego_motion = x - transformed_x
        feat_3 = flow - ego_motion
        # x = x.transpose(2, 1).contiguous()
        feat_3 = feat_3.transpose(2, 1).contiguous()
        # _, feat_1, index = self.mos_sa1(x, moving_flow, idx=None)
        # _, feat_2, _ = self.mos_sa2(x, feat_1, idx=index)
        # _, feat_3, _ = self.mos_sa3(x, feat_2, idx=index)

        feat_3 = F.relu(self.bn1(self.conv1(feat_3)))
        feat_3 = F.relu(self.bn2(self.conv2(feat_3)))
        feat_3 = F.relu(self.bn3(self.conv3(feat_3)))
        feat_3 = self.conv4(feat_3)
        return feat_3.transpose(2, 1).contiguous()


class FLOT(nn.Module):
    def __init__(self):
        super(FLOT,self).__init__()

        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        self.nb_iter = 1

        self.sa1 = PointNetSetAbstractionFLOT(nsample=32, in_channel=3, mlp=[32, 32, 32])
        self.sa2 = PointNetSetAbstractionFLOT(nsample=32, in_channel=32, mlp=[64, 64, 64])
        self.sa3 = PointNetSetAbstractionFLOT(nsample=32, in_channel=64, mlp=[128, 128, 128])

        self.rf1 = PointNetSetAbstractionFLOT(nsample=32, in_channel=3, mlp=[32, 32, 32])
        self.rf2 = PointNetSetAbstractionFLOT(nsample=32, in_channel=32, mlp=[64, 64, 64])
        self.rf3 = PointNetSetAbstractionFLOT(nsample=32, in_channel=64, mlp=[128, 128, 128])

        self.fc = nn.Conv1d(128, 3, kernel_size=1, bias=True)


    def forward(self, pc1, pc2):
        # B,N,3 ------> B,3,N
        # for infer purpose, dont use cuda, then we can increase the points
        pc1 = pc1.transpose(2, 1).contiguous()
        pc2 = pc2.transpose(2, 1).contiguous()
        # index always use the neighboring in the euclidean space
        # B, F, N
        start = time.time()
        _, l1_feature1, idx1 = self.sa1(pc1, pc1, idx = None)
        _, l2_feature1, _ = self.sa2(pc1, l1_feature1, idx1)
        _, l3_feature1, _ = self.sa3(pc1, l2_feature1, idx1)
        end = time.time()
        # print("encoder1 time: ", 1000*(end-start))

        start = time.time()
        _, l1_feature2, idx2 = self.sa1(pc2, pc2, idx = None)
        _, l2_feature2, _ = self.sa2(pc2, l1_feature2, idx2)
        _, l3_feature2, _ = self.sa3(pc2, l2_feature2, idx2)
        end = time.time()
        # print("encoder2 time: ", 1000*(end-start))
        # cross feature PC1 <--> PC2
        # idx_1_2 = knn_point(32, pc2.transpose(2, 1).contiguous(), pc1.transpose(2, 1).contiguous())
        # _, l1_cross_feature, _ = self.sa1(pc2, pc2, idx = idx_1_2)
        # _, l2_cross_feature, _ = self.sa2(pc2, l1_cross_feature, idx = idx_1_2)
        # _, l3_cross_feature, _ = self.sa3(pc2, l2_cross_feature, idx = idx_1_2)

        # Optimal transport
        # B,N,3
        ot_pc1_t = pc1.transpose(2, 1).contiguous()
        ot_pc2_t = pc2.transpose(2, 1).contiguous()

        start = time.time()
        transport = sinkhorn(
            l3_feature1.transpose(2, 1).contiguous(),
            l3_feature2.transpose(2, 1).contiguous(),
            ot_pc1_t,
            ot_pc2_t,
            epsilon=torch.exp(self.epsilon) + 0.03,
            gamma=torch.exp(self.gamma),
            max_iter=self.nb_iter,
        )
        # n^(-1)
        row_sum = transport.sum(-1, keepdim=True)

        # Estimate flow with transport plan
        # new version in HAO-MOS, our flow is from current to last
        pc1_in_pc2_coord_t = (transport @ ot_pc2_t) / (row_sum + 1e-8)
        ot_flow_t = ot_pc1_t - pc1_in_pc2_coord_t
        ot_flow = ot_flow_t.transpose(2, 1).contiguous()
        end = time.time()
        # print("OT time: ", 1000*(end-start))

        start = time.time()
        # # refine
        _, l1_ot_flow, _ = self.rf1(pc1, ot_flow, idx = idx1)
        _, l2_ot_flow, _ = self.rf2(pc1, l1_ot_flow, idx1)
        _, l3_ot_flow, _ = self.rf3(pc1, l2_ot_flow, idx1)
        residual_flow = self.fc(l3_ot_flow)
        sf = ot_flow + residual_flow    # [B, 3, N]

        # # B,3,N -------> B,N,3
        sf = sf.transpose(2, 1).contiguous()
        end = time.time()
        # print("residual time: ", 1000*(end-start))
        # calculate Rigid Transformation from pc1 to pc2
        # rigid_transform = compute_rigid_transform(ot_pc1_t, pc1_in_pc2_ref_t, weights=torch.sum(transport, dim=-1))
        # rigid_ot_pc1_t = se3_transform(rigid_transform, ot_pc1_t)
        # ego_motion = ot_pc1_t - rigid_ot_pc1_t
        return sf
        # return sf, idx1


class HAOMOS(nn.Module):
    def __init__(self):
    # def __init__(self, config):
        super(HAOMOS, self).__init__()
        self.flot = FLOT()
        # if config.pretrain:
        for p in self.flot.parameters():
            p.requires_grad = False
        self.mos = MOS()

    def forward(self, pc1, pc2):
        ego_motion, idx1 = self.flot(pc1, pc2)
        mos = self.mos(pc1, pc2, ego_motion, idx1)
        return mos



class MOS(nn.Module):
    def __init__(self):
        super(MOS, self).__init__()
        self.drf1 = PointNetSetAbstractionFLOT(nsample=32, in_channel=3, mlp=[32, 32, 32])
        self.drf2 = PointNetSetAbstractionFLOT(nsample=32, in_channel=32, mlp=[64, 64, 64])
        self.drf3 = PointNetSetAbstractionFLOT(nsample=32, in_channel=64, mlp=[128, 128, 128])

        self.dfc = nn.Conv1d(128, 2, kernel_size=1, bias=True)

    def forward(self, pc1, pc2, gt_ego_motion, idx1):
        dynamic_residual_flow = pc1 - gt_ego_motion - pc2
        dynamic_residual_flow = dynamic_residual_flow.transpose(2, 1).contiguous()

        _, dynamic_l1_ot_flow, _ = self.drf1(pc1.transpose(2, 1).contiguous(), dynamic_residual_flow, idx1)
        _, dynamic_l2_ot_flow, _ = self.drf2(pc1.transpose(2, 1).contiguous(), dynamic_l1_ot_flow, idx1)
        _, dynamic_l3_ot_flow, _ = self.drf3(pc1.transpose(2, 1).contiguous(), dynamic_l2_ot_flow, idx1)
        mos = self.dfc(dynamic_l3_ot_flow)
        return mos.transpose(2, 1).contiguous()


