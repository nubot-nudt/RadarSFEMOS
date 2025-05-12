import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.model_utils import *


class JGWTF(nn.Module):
    def __init__(self,args):
        super(JGWTF,self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=int(args.num_points/2), radius=0.5, nsample=16, in_channel=3, mlp=[32,32,64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=int(args.num_points/8), radius=1.0, nsample=16, in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=int(args.num_points/32), radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=int(args.num_points/128), radius=4.0, nsample=8, in_channel=256, mlp=[256,256,512], group_all=False)
        
        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel = 128, mlp=[128, 128, 128], pooling='max', corr_func='concat')
        
        self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel = 256, f2_channel = 512, mlp=[], mlp2=[256, 256])
        self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel = 128+128, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.su3 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel = 64, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.fp = PointNetFeaturePropogation(in_channel = 256+3, mlp = [256, 256])
        
        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2=nn.Conv1d(128, 3, kernel_size=1, bias=True)
    
    def sceneflowestimate(self,pc1,pc2,feature1,feature2):
        
        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)
        
        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        
        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)
        
        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        
        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)
        
        return sf
    
    def batched_index_select(self, t, dim, inds):
        
        dummy = inds.expand(inds.size(0), inds.size(1), t.size(2))
        out = t.gather(dim, dummy) # b x e x f
        
        return out
    
    def forward(self, pc1, pc2, feature1, feature2, state):
        
        
        sf_f =  self.sceneflowestimate(pc1, pc2, feature1, feature2)
        if state=='test':
            return sf_f
        else:
            ## inverse flow
            pc1_warp=pc1+sf_f
            pc1_warp=pc1_warp.permute(0, 2, 1)
            pc2=pc2.permute(0, 2, 1)
            feature1=feature1.permute(0, 2, 1)
            feature2=feature2.permute(0, 2, 1)
            idx = knn_point(1, pc1_warp, pc2)
            pc1_anchor = (pc1_warp + self.batched_index_select(pc2,1,idx))/2
            ft1_anchor = (feature1 + self.batched_index_select(feature2,1,idx))/2
            #ft1_anchor = torch.cat((-ft1_anchor[:,:,0].unsqueeze(2),ft1_anchor[:,:,1:]),2)
            #ft1 = torch.cat((-feature1[:,:,0].unsqueeze(2),feature1[:,:,1:]),2)
            ft1 = feature1
            pc1_anchor=pc1_anchor.permute(0, 2, 1).contiguous()
            ft1_anchor=ft1_anchor.permute(0, 2, 1).contiguous()
            ft1=ft1.permute(0, 2, 1).contiguous()
            sf_b = self.sceneflowestimate(pc1_anchor, pc1, ft1_anchor, ft1)
            
            return sf_f, sf_b
