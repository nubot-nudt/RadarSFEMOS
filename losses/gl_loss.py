import gc
import argparse
import sys
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
# from tensorboardX import SummaryWriter
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from tqdm import tqdm
import cv2
from utils import *
from utils.model_utils import *

def ChamferDistance(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M

    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    npoints = pc1.size(1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)
    
    return dist1, dist2

def GraphLaplacian(pc1, num_nb=50):
    
    '''
    pc1: B 3 N
    
    '''
    pc1 = pc1.permute(0, 2, 1)
    npoints = pc1.size()[1]
    B = pc1.size()[0]
    sqrdist = square_distance(pc1, pc1) # B N N
    dists, idx = torch.topk(sqrdist, num_nb+1, dim = -1, largest=False, sorted=True)
    W = torch.eye((npoints)).expand(B,npoints,npoints).cuda()
    W.scatter_(dim=2, index=idx[:,:,1:], src=torch.exp(-dists[:,:,1:]))
    
    D = torch.diag_embed(W.sum(dim=1),offset=0, dim1=-2, dim2=-1)
    
    L = torch.inverse(D)**1/2 @ (D-W) @ torch.inverse(D)**1/2
    
    return L
    
    
    
def ChamferGL(pc1, pc2, pred_f):
    
    f_gl = 10.0
    f_chamfer = 1.0
    N = pc1.size()[2]

    pc1_warp=pc1+pred_f

    # chamfer 
    dist1, dist2 = ChamferDistance(pc1_warp, pc2)
    chamferLoss = (dist1.sum(dim=1) + dist2.sum(dim=1)).mean()/N
    
    # graph laplacian
    L = GraphLaplacian(pc1)
    L_mat = pred_f @ L @ pred_f.permute(0,2,1)
    glLoss = L_mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).mean()
    
    
    total_loss = f_chamfer * chamferLoss + f_gl * glLoss
    
    items={
        'Loss': total_loss.item(),
        'glLoss': glLoss.item(),
        'chamferLoss': chamferLoss.item(),
        }

    return total_loss, items
    
def GL_optimize(pc1, pc2):
    
    nepochs = 50
    lr = 0.1
    pred_f = torch.zeros(pc1.size(), requires_grad=True, device="cuda")
    opt = optim.Adam([pred_f], lr=lr)
    for i in range(nepochs):
        loss, loss_item = ChamferGL(pc1,pc2,pred_f)
        if loss_item['glLoss']<=0:
            break
        opt.zero_grad() 
        loss.backward()
        opt.step()
    
    return pred_f
    
    