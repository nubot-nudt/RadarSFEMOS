# def output_decoder():
import numpy as np
import torch
from torch.nn import functional as F

import utils


@torch.no_grad()
def voxelize(self, points):
    """Apply hard voxelization to points."""
    voxels, coors, num_points = [], [], []
    for res in points:
        res_voxels, res_coors, res_num_points = self.voxel_layer(res)
        voxels.append(res_voxels)
        coors.append(res_coors)
        num_points.append(res_num_points)
    voxels = torch.cat(voxels, dim=0)
    num_points = torch.cat(num_points, dim=0)
    coors_batch = []
    for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coors_batch.append(coor_pad)
    coors_batch = torch.cat(coors_batch, dim=0)
    return voxels, num_points, coors_batch


def output_decoder(pc1, voxels0, coors0, F1, L_cls, L_wgt, p_stat):
    """
    Output decoder

    pc1: (3 * N)
    F: (B * 2 * H * W)
    L_cls: (B * H * W)
    L_wgt: (B * H * W)
    p_stat: threshold
    """
    mask = torch.zeros(256)
    # map voxels to cloud points
    # see voxel_cloud_mapping for details
    L_cls = L_cls.unsqueeze(1)
    L_cls, L_wgt, F1, pc1m = voxel_cloud_mapping(pc1, voxels0, coors0, L_cls, L_wgt, F1)
   
    Tr = kabsch(pc1m, F1, L_cls, L_wgt, p_stat)
  
    # calculate aggregated flow
    for batch_i in range(F1.size(0)):
        for i in range(0, F1.size(1)):
            if L_cls[batch_i, :, i][0] >= p_stat:
                pc1m_ = torch.cat((pc1m, torch.ones(pc1m.size(dim=0), 1, pc1m.size(dim=2)).cuda()), 1)
                F1[batch_i, :, i] = torch.matmul((Tr[batch_i] - torch.eye(4).cuda()), pc1m_[batch_i, :, i])[:3]           
    return Tr, L_cls, L_wgt, F1, pc1m  


def rigid_f(pc1, Tr, I4=torch.eye(4).cuda()):
    pc1_ = torch.cat((pc1, torch.ones(pc1.size(dim=0), 1, pc1.size(dim=2)).cuda()), 1)
    Fr = torch.matmul((Tr - I4), pc1_)
    # pc1 = pc1[:, :3, :]
    Fr = Fr[:, :3, :]
    return Fr


def voxel_cloud_mapping(pc1, voxels0, coors0, L_cls, L_wgt, F1):
    """
    NOT IMPLEMENTED YET

    Map voxels to cloud points
    !!! Extra arguments like pc1/pc2 or voxels needed to find corresponding index
    !!! Add extra args and returns
    Ether create new reduced pc or F

    in:
        F: (B * 2 * H * W)
        L_cls: (B * H * W)
        L_wgt: (B * H * W)
    out:
        F: (B * 3 * N)
        L_cls: (B * 1 * N)
        L_wgt: (B * 1 * N)
    """
    # F = torch.cat((F, torch.zeros(4, 1, 1100).cuda()), dim=1)
    pc1m = []

    Fm1 = []

    L_clsm = []
    L_wgtm = []

    voxels0 = [torch.tensor(voxels).cuda() for voxels in voxels0]
    coors0 = [torch.tensor(coors).cuda() for coors in coors0]

    for batch_i in range(F1.size(0)):
        # pc1m_ = []
        # pc2m_ = []
        Fm1_ = []
        Fm2_ = []
        L_clsm_ = []
        L_wgtm_ = []
        L_clsm2_ = []
        L_wgtm2_ = []

        # Construct new corresponding F to pc1
        for i in range(pc1.size(2)):

            # Prepare for distance calculation
            p = pc1[batch_i, :,  i].unsqueeze(0).unsqueeze(0)
            # Find centroid of voxels
            voxels = torch.mean(voxels0[batch_i][:, :, :3], dim=1).unsqueeze(0)
            # Get closest voxel index
            closest_i = torch.argmin(utils.square_distance(p, voxels))

            f = F1[batch_i, :, coors0[batch_i][closest_i, 0], coors0[batch_i][closest_i, 1]]
            cls = L_cls[batch_i, :, coors0[batch_i][closest_i, 0], coors0[batch_i][closest_i, 1]]
            wgt = L_wgt[batch_i, :, coors0[batch_i][closest_i, 0], coors0[batch_i][closest_i, 1]]

            Fm1_.append(f.unsqueeze(1))
            L_clsm_.append(cls.unsqueeze(1))
            L_wgtm_.append(wgt.unsqueeze(1))

    
        Fm1.append(torch.cat(Fm1_, dim=1).unsqueeze(0))
        L_clsm.append(torch.cat(L_clsm_, dim=1).unsqueeze(0))
        L_wgtm.append(torch.cat(L_wgtm_, dim=1).unsqueeze(0))

    Fm1 = torch.cat(Fm1, dim=0).cuda()
 
    L_clsm = torch.cat(L_clsm, dim=0).cuda()
    L_wgtm = torch.cat(L_wgtm, dim=0).cuda()
    Fm1 = torch.cat((Fm1, torch.zeros(Fm1.size(0), 1, Fm1.size(2)).cuda()), dim=1).cuda()
   

    return L_clsm, L_wgtm, Fm1, pc1


def kabsch(pc1, F, L_cls, L_wgt, p_stat):

    """
    kabsch algorithm

    pc1: (B * 3 * N)
    F: (B * 3 * N)
    L_cls: (B * 1 * N)
    L_wgt: (B * 1 * N)
    p_stat: threshold
    """

    A = pc1
    B = pc1 + F

    assert A.size() == B.size()

    batch_size, num_rows, num_cols = A.size()

    ## mask to 0/1 weights for motive/static points
    # W = M.type(torch.bool).unsqueeze(2)
    # W = torch.zeros(A.size(dim=2))

    mask = (torch.sigmoid(L_cls) >= p_stat) * torch.ones(L_wgt.size()).cuda()
    # lwgt_ = torch.min(L_wgt, torch.zeros(L_wgt.size()).cuda())
    # s = torch.max(lwgt_ * mask)
    # exps = torch.exp(lwgt_ - s)
    # deno = torch.sum(torch.sigmoid(torch.abs(L_wgt)) * exps * mask)
    # numer = mask * torch.sigmoid(torch.abs(L_wgt)) * exps
    numer = torch.sigmoid(L_wgt) * mask
    deno = torch.sum(torch.sigmoid(L_wgt) * mask)
    numer += 10**(-4)
    deno += 10**(-4)

    W = numer / deno
    W = W.transpose(2, 1).contiguous()

    # find mean column wise
    centroid_A = torch.mean(A.transpose(2, 1).contiguous() * W, axis=1)
    centroid_B = torch.mean(B.transpose(2, 1).contiguous() * W, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(batch_size, num_rows, 1)
    centroid_B = centroid_B.reshape(batch_size, num_rows, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = torch.matmul(Am, Bm.transpose(2, 1).contiguous()*W)
    
    if torch.isnan(H).any():
        Tr = torch.eye(4).cuda()
        Tr.unsqueeze(0).repeat(F.size(0), 1, 1)
        return Tr 
    # find rotation
    U, S, V = torch.svd(H)
    Z = torch.matmul(V, U.transpose(2, 1).contiguous())
    # special reflection case
    d = (torch.linalg.det(Z) < 0).type(torch.int8)
    # -1/1
    d = d * 2 - 1
    Vc = V.clone()
    Vc[:, 2, :] *= -d.view(batch_size, 1)
    R = torch.matmul(Vc, U.transpose(2, 1).contiguous())

    t = torch.matmul(-R, centroid_A) + centroid_B

    Trans = torch.cat(
        (torch.cat((R, t), axis=2), torch.tensor([0, 0, 0, 1]).repeat(batch_size, 1).cuda().view(batch_size, 1, 4)),
        axis=1)

    return Trans