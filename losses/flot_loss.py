import torch.nn as nn
import torch
import numpy as np
import time


def FLOTLoss(pred_f, labels):

    error = pred_f - labels
    loss = torch.mean(torch.abs(error))

    items={
        'Loss': loss.item(),
        }
    
    return loss, items
