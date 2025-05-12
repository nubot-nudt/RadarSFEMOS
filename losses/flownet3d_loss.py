from multiprocessing import reduction
import torch.nn as nn
import torch
import numpy as np
import time


def CycleL1Loss(pc1, pred_f, pred_b, labels):
    
    f_l1 = 1.0
    f_cycle = 0.3

    B, N = pc1.shape[0], pc1.shape[2]
    
    cycle_loss_obj = torch.nn.SmoothL1Loss(reduction='sum')
    l1_loss_obj = torch.nn.SmoothL1Loss(reduction='sum')
    # cycle loss 
    cycleLoss = cycle_loss_obj(pred_f,pred_b)/(B*N)
    
    # L1
    L1Loss=l1_loss_obj(pred_f, labels.transpose(2,1))/(B*N)
    
    total_loss = f_l1 * L1Loss + f_cycle * cycleLoss
    
    items={
        'Loss': total_loss.item(),
        'L1Loss': L1Loss.item(),
        'cycleLoss': cycleLoss.item(),
        }
    
    return total_loss, items