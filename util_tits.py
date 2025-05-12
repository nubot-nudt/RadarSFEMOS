"""
References:
flownet3d_pytorch: https://github.com/hyangwinter/flownet3d_pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import sys
sys.path.append("..")



def index_points_flot(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    points = points.permute(0, 2, 1)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.detach().long(), :]
    return new_points.permute(0, 3, 1, 2).contiguous()


def se3_transform(g, a, normals=None):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
        a: Points to be transformed (N, 3) or (B, N, 3)
        normals: (Optional). If provided, normals will be transformed

    Returns:
        transformed points of size (N, 3) or (B, N, 3)

    """
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b


def batch_mat2xyzrpy(batch_rotation_matrix):
    '''
    Input: batch_rotation_matrix(torch.Tensor): [batch, 4, 4]
    Returns: batch_transform(torch.Tensor): [batch, 6], contains [x,y,z,roll,pitch,yaw]
    '''
    roll = torch.atan2(- batch_rotation_matrix[:,1,2], batch_rotation_matrix[:,2,2]).unsqueeze(1)
    pitch = torch.asin(batch_rotation_matrix[:,0,2]).unsqueeze(1)
    yaw = torch.atan2(- batch_rotation_matrix[:,0,1], batch_rotation_matrix[:,0,0]).unsqueeze(1)
    x = batch_rotation_matrix[:,0,3].unsqueeze(1)
    y = batch_rotation_matrix[:,1,3].unsqueeze(1)
    z = batch_rotation_matrix[:,2,3].unsqueeze(1)
    batch_transform = torch.cat((x,y,z,roll,pitch,yaw),dim=1)
    return batch_transform

def index_points(points, idx):
    """
    Input:
        points: input point cloud [B, N, C] tensor
        idx: point index [B, npoints] tensor
    output:
        indexed points: [B, npoints, C] tensor
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1]*(len(view_shape)-1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # [B, S]
    new_points = points[batch_indices, idx, :]
    return new_points

def query_knn_point(k, query, pc):
    """
    Input:
        k: number of neighbor points
        query: query points [B, S, 3]
        pc: point cloud [B, N, 3]
        points: point features
    Output:
        normed_knn_points [B, S, k, 3]
        knn_ids: index of knn points 
    """
    query = query.permute(0,2,1).unsqueeze(3)
    database = pc.permute(0,2,1).unsqueeze(2)
    norm = torch.norm(query-database, dim=1, keepdim=False)
    knn_d, knn_ids = torch.topk(norm, k=k, dim=2, largest=False, sorted=True)
    knn_points = index_points(pc, knn_ids)

    return knn_points, knn_ids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def group_query(nsample, s_xyz, xyz, s_points, idx = None):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    if idx is None:
        idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_flot(s_xyz.permute(0, 2, 1).contiguous(), idx.int()).permute(0, 2, 3, 1).contiguous()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)   # relative coordinates of neighbor points

    if s_points is not None:    # points feature grouping
        grouped_points = index_points_flot(s_points.permute(0, 2, 1).contiguous(), idx.int()).permute(0, 2, 3, 1).contiguous()
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm, idx


class PointNetSetAbstractionFLOT(nn.Module):
    def __init__(self, nsample, in_channel, mlp):
        super(PointNetSetAbstractionFLOT, self).__init__()
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))

            last_channel = out_channel


    def forward(self, xyz, points, idx = None):
        """
        In this pointnet++ like convolution, we do not downsample input points.
        ----------
        Input:
            xyz: input points position data, [B, C, N]
            points: input point features, [B, D, N]
        Return:
            new_xyz: input points position data, [B, C, N]
            new_points: output point features, [B, D', N]
            idx: index of neighboring points
        """

        xyz_t = xyz.permute(0, 2, 1).contiguous()

        new_xyz = xyz
        new_xyz_t = new_xyz.permute(0, 2, 1).contiguous()

        points_t = points.permute(0, 2, 1).contiguous()
        new_points, grouped_xyz_norm, idx = group_query(self.nsample, xyz_t, new_xyz_t, points_t, idx)

        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.lrelu(bn(conv(new_points)))
        new_points = torch.max(new_points, -1)[0]

        return new_xyz, new_points, idx
