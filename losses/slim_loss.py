import torch
from utils.model_utils.slim_utils import kabsch, rigid_f
from utils.model_utils import *


def total_loss(pc1, pc2, F1, L_cls, L_wgt, Tr, Tr2, p_stat=0.31, m_thresh=0.05):
    """
    Calculates total loss

    Expected in:
        pc1: (B * 3 * N)
        pc2: (B * 3 * N)
        F: (B * 3 * N)
        L_cls: (B * 1 * N)
        L_wgt: (B * 1 * N)
        p_stat: threshold const
        m_thresh: threshold const
        Tr: (4 * 4)
    """
    f_nn = 2.0
    f_cycle = 1.0
    f_art = 1.0

    # Tr = kabsch(pc1, F, L_cls, L_wgt, p_stat)
    err_i, err_ri = nn_errors(pc1, pc2, F1, Tr)
    nn_loss = total_nn_loss(err_i, err_ri)
    cycle_loss = rigid_cycle_loss(pc1, pc2, Tr, Tr2)
    art_loss = artificial_label_loss(err_i, err_ri, L_cls)
    # total += supervised_loss(pc1, pc2, L_cls, F, Tr, m_thresh)

    total = f_nn * nn_loss + f_cycle * cycle_loss + f_art * art_loss

    items = {
        'Loss': total.item(),
    }

    return total, items


def computeNN(pc1, pc2):
    """
    pc1: B 3 N
    pc2: B 3 M
    """
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    npoints = pc1.size(1)
    sqrdist12 = square_distance(pc1, pc2)  # B N M

    # NN Dist
    dist1, _ = torch.topk(sqrdist12, 1, dim=-1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    return dist1


def nn_errors(pc1, pc2, F, Tr):
    pc1_warp = pc1 + F
    err_i = computeNN(pc1_warp, pc2)

    Fr = rigid_f(pc1, Tr)
    pc1_warp_r = pc1 + Fr
    err_ri = computeNN(pc1_warp_r, pc2)

    err_i = torch.maximum(err_i, torch.zeros_like(err_i)) + 10**-4
    err_ri = torch.maximum(err_ri, torch.zeros_like(err_ri)) + 10**-4
    # err_i = torch.abs(err_i)
    # err_ri = torch.abs(err_ri)

    return err_i.sqrt(), err_ri.sqrt()


def total_nn_loss(err_i, err_ri):
    loss = torch.mean(err_i + err_ri)
    return loss


def rigid_cycle_loss(pc1, pc2, Tr, Tr2):
    batch_size = pc1.size()[0]

    trans_diff = torch.matmul(Tr, Tr2) - torch.eye(4).cuda()
    pos_diff = torch.matmul(trans_diff, torch.cat((pc1, torch.ones(batch_size, 1, pc1.size()[2]).cuda()), dim=1))

    loss = torch.mean(torch.norm(pos_diff[:, :-1], dim=1))
    # diffs = torch.sqrt(torch.sum(pos_diff ** 2, dim=1))
    # diffs, _ = torch.topk(diffs, int(3 * npoints / 4), dim=1, largest=False, sorted=False)

    return loss


def artificial_label_loss(err_i, err_ri, L_cls):
    log1 = (err_i < err_ri) * torch.log(torch.sigmoid(L_cls.squeeze(1)))
    log2 = (err_i >= err_ri) * torch.log(1 - torch.sigmoid(L_cls.squeeze(1)))
    loss = - torch.sum(log1 + log2,dim=1)
    return loss.mean()


def supervised_loss(pc1, pc2, L_cls, F, Tr, m_thresh):
    F_gt = pc2 - pc1
    l_flow = torch.mean(F_gt - F)

    Trm = []
    for batch_i in range(Tr.size(0)):
        Trm.append(torch.inverse(Tr[batch_i]).unsqueeze(0))
    Odo = torch.cat(Trm, 0).cuda()
    ri = torch.abs(F_gt - rigid_f(pc1, Odo)) <= m_thresh
    ri = ri * torch.zeros(ri.size()).cuda()

    log1 = ri * torch.log(torch.sigmoid(L_cls))
    log2 = (1 - ri) * torch.log(1 - torch.sigmoid(L_cls))
    l_cls = - torch.mean(log1 + log2)

    l_r = torch.mean(rigid_f(pc1, Tr, Odo))

    total_loss = l_flow + l_cls + l_r
    return total_loss