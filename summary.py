import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from models.arfnet import RaFlow
from pytorch_model_summary import summary


args = parse_args_from_yaml("configs.yaml")

print(summary(RaFlow(args).cuda(), torch.zeros((8,3,256)).cuda(), torch.zeros((8,3,256)).cuda(),torch.zeros((8,3,256)).cuda(),\
              torch.zeros((8,3,256)).cuda(),torch.tensor([0.1]).cuda(),show_input=True,show_hierarchical=False))
print(summary(RaFlow(args).cuda(), torch.zeros((8,3,256)).cuda(), torch.zeros((8,3,256)).cuda(),torch.zeros((8,3,256)).cuda(),\
              torch.zeros((8,3,256)).cuda(),torch.tensor([0.1]).cuda(),show_input=False,show_hierarchical=True))
