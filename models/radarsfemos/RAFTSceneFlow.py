import torch
import torch.nn as nn
import sys
from .extractor import FlotEncoder
from .corr import CorrBlock
from .update import UpdateBlock
from .refine import *
from .RadarTransformer import *
from .transformer import *
from utils.model_utils.radarsfemos_util import *
import time
from .pointtransformer_seg import *
from .corr import *
class RSF(nn.Module):
    def __init__(self, args):
        super(RSF, self).__init__()
        self.hidden_dim = 64
        self.context_dim = 64
        self.rigid_pcs = 0.25
        self.rigid_thres = 0.15
        self.feature_extractor = FlotEncoder()
        self.context_extractor = FlotEncoder()

        # self.context_extractor = TransformerBlock1(128)
        # self.pt = TransformerBlock1(128)
        # self.context_extractor = TransformerBlock(128,256,32)
        # self.pt = TransformerBlock(128,256,32)
        args.corr_levels = 3
        args.base_scales = 0.25
        args.truncate_k = int(args.num_points/4)
        self.corr_block = CorrBlock(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k)
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)
        self.num_neighbors = 32
        # self.fc1 = nn.Sequential(
        #     nn.Conv1d(3, 128, kernel_size=1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, kernel_size=1, bias=False)
        # )        
        # self.refine_block = FlotRefine()
        #PointNetFeaturePropogation(in_channel=128, mlp=[3])
        self.refine = FlotRefine()
    def forward(self, p, feature1 ,feature2,interval):
        num_iters = 8
        # feature extraction
        [xyz1, xyz2] = p  #torch.Size([32, 256, 3])
        
        fmap1, graph = self.feature_extractor(p[0])
        fmap2, _ = self.feature_extractor(p[1])  
        
        
        start_time = time.time()
        # correlation matrix
        self.corr_block.init_module(fmap1, fmap2, xyz2)
        fct1, graph_context = self.context_extractor(p[0])
        end = time.time() - start_time
        print(end)
        
        
        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)


        coords1, coords2 = xyz1, xyz1
        flow_predictions = []

        for itr in range(num_iters):
            coords2 = coords2.detach()
            corr = self.corr_block(coords=coords2)
            flow = coords2 - coords1
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
            
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)
        
        flow = self.refine(flow_predictions[-1],graph)
        
        
        return flow_predictions


def B_N_xyz2pxo(input,max_points):
    b, n, xyz = input.size()
    if b>1:
        n_o, count = [ max_points ],max_points
        tmp = [input[0]]
        for i in range(1, b):
            count += max_points
            n_o.append(count)
            tmp.append(input[i])
        n_o = torch.cuda.IntTensor(n_o)
        coord = torch.cat(tmp, 0)  
    else:
        n_o = [ max_points ]
        n_o = torch.cuda.IntTensor(n_o)
        coord = input[0]
    feat = coord
    label =  n_o   
    return coord, feat, label # (n, 3), (n, c), (b)

def pt_fmap_trans(input, max_points):
    p_all, x = input.size()
    b_num = p_all // max_points
    f = input.transpose(0,1)
    f = f.view( x, b_num, p_all // b_num)
    f = f.transpose(0,1)
    return f 

def pt_fmap_trans_ot(input, max_points):
    p_all, x = input.size()
    b_num = p_all // max_points
    f = input.transpose(0,1)
    f = f.view( x, b_num, p_all // b_num)
    f = f.permute(1,2,0)
    return f 
    


class RSF1(nn.Module):
    def __init__(self, args):
        super(RSF1, self).__init__()
        self.args = args
        self.num_neighbors = 32
        self.hidden_dim = 64 
        self.context_dim = 64
        self.feature_extractor = PointTransformerSeg()
        self.context_extractor = PointTransformerSeg()      
        args.corr_levels = 3
        args.base_scales = 0.25
        args.truncate_k = int(args.num_points/4) 
        self.corr_block = new_CorrBlock(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k,knn = self.num_neighbors)                             
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)
        self.args.max_points = 256

    def forward(self, p, feature1 ,feature2,interval):
        [xyz1, xyz2] = p

        p1, x1, o1 = B_N_xyz2pxo(p[0],self.args.max_points)
        p2, x2, o2 = B_N_xyz2pxo(p[1],self.args.max_points)
        
        fmap1_origin = self.feature_extractor([p1, x1, o1])
        fmap2_origin = self.feature_extractor([p2, x2, o2])
        
        fmap1 = pt_fmap_trans(fmap1_origin,self.args.max_points)
        fmap2 = pt_fmap_trans(fmap2_origin,self.args.max_points)

        # correlation matrix
        self.corr_block.init_module(fmap1, fmap2, xyz2)
        fct1_origin = self.context_extractor([p1, x1, o1])
        fct1 = pt_fmap_trans(fct1_origin,self.args.max_points)
        graph_context = Graph.construct_graph(p[0], self.num_neighbors)

        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp) 

        coords1, coords2 = xyz1, xyz1
        flow_predictions = []
        num_iters = 8
        for itr in range(num_iters):
            coords2 = coords2.detach()
            corr = self.corr_block(coords=coords2)
            flow = coords2 - coords1
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)

        return flow_predictions



