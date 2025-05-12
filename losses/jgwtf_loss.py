import torch.nn as nn
import torch
import numpy as np
import time
from utils import *
from utils.model_utils import *

def computeNN(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    npoints = pc1.size(1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #NN Dist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    return dist1

def computeCycle(pred_f,pred_b):
    
    '''
    pred_f: B 3 N
    pred_b: B 3 N
    
    '''
    diff_cycle = torch.sum((pred_f+pred_b)**2,1)
    
    return diff_cycle

def NNCycle(pc1,pc2,pred_f,pred_b):
    
    f_nn = 1.0
    f_cycle = 1.0
    
    
    # cycle loss  
    diff_cycle = computeCycle(pred_f,pred_b)
    cycleLoss = torch.mean(diff_cycle)
    
    
    # NN
    pc1_warp=pc1+pred_f
    dist = computeNN(pc1_warp,pc2)
    nnLoss=torch.mean(dist)
    
    total_loss = f_nn * nnLoss + f_cycle * cycleLoss
    
    items={
        'Loss': total_loss.item(),
        'nnLoss': nnLoss.item(),
        'cycleLoss': cycleLoss.item(),
        }
    
    return total_loss, items