import gc
import argparse
import sys
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from utils import *
from utils.model_utils import *
import torch.nn.functional as F


def computeSoftChamfer(pc1, pc1_warp, pc2, zeta=0.005):
    
    '''
    pc1: B 3 N
    pc2: B 3 N
    pc1_warp: B 3 N

    '''
    pc1 = pc1.permute(0, 2, 1)
    pc1_warp = pc1_warp.permute(0,2,1)
    pc2 = pc2.permute(0, 2, 1)
    npoints = pc1.size(1)
    batch_size = pc1.size(0)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    ## use the kernel density estimation to obatin perpoint density
    dens12 = compute_density_loss(pc1, pc2, 1)
    dens21 = compute_density_loss(pc2, pc1, 1)

    mask1 = (dens12>zeta).type(torch.int32)
    mask2 = (dens21>zeta).type(torch.int32)

    sqrdist12w = square_distance(pc1_warp, pc2) # B N M
    
    dist1_w, _ = torch.topk(sqrdist12w, 1, dim = -1, largest=False, sorted=False)
    dist2_w, _ = torch.topk(sqrdist12w, 1, dim = 1, largest=False, sorted=False)
    dist1_w = dist1_w.squeeze(2)
    dist2_w = dist2_w.squeeze(1)
    
    dist1_w = F.relu(dist1_w-0.01)
    dist2_w = F.relu(dist2_w-0.01)
    
    dist1_w = dist1_w * mask1 
    dist2_w = dist2_w * mask2 

    
    return dist1_w, dist2_w


def computeWeightedSmooth(pc1, pred_flow, alpha=0.5):
    
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''
    
    B = pc1.size()[0] 
    N = pc1.size()[2]
    num_nb = 8
    pc1 = pc1.permute(0, 2, 1)
    npoints = pc1.size(1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    ## compute the neighbour distances in the point cloud
    dists, kidx = torch.topk(sqrdist, num_nb+1, dim = -1, largest=False, sorted=True)
    dists = dists[:,:,1:]
    kidx = kidx[:,:,1:]
    dists = torch.maximum(dists,torch.zeros(dists.size()).cuda())
    ## compute the weights according to the distances
    weights = torch.softmax(torch.exp(-dists/alpha).view(B,N*num_nb),dim=1)
    weights = weights.view(B,N,num_nb)
  
    grouped_flow = index_points_group(pred_flow, kidx) 
    diff_flow = (npoints*weights*torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3)).sum(dim = 2) 
    #diff_flow = (torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3)).sum(dim = 2) /num_nb
    return diff_flow


    
def computeloss(pc1,pc2, agg_f, vel1,interval, args):
    
    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    
    pc1_warp_a = pc1+agg_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp_a, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, agg_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(agg_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 
    
    
    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss + f_velo*veloLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item()*f_velo,
        }
    
    return total_loss, items

def computeloss_o(pc1,pc2, pred_f, vel1, pre_trans, stat_cls, gt_trans, prob_m, interval, args):
    
    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    f_ego = 1.0
    f_mask = 1.0
    
    
    pc1_warp = pc1+pred_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, pred_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(pred_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 
    
    # ego-motion
    pc1_pre = torch.matmul(pre_trans[:,:3,:3], pc1)+pre_trans[:,:3,3].unsqueeze(2)
    pc1_gt = torch.matmul(gt_trans[:,:3,:3], pc1)+gt_trans[:,:3,3].unsqueeze(2)
    egoLoss = torch.mean(torch.norm(pc1_pre-pc1_gt,dim=1))
    
    # mask
    #maskLoss = torch.mean(-(pse_m * torch.log(stat_cls+1e-6) + (1-pse_m) * torch.log(1-stat_cls-1e-6)))
    BCEloss = torch.nn.BCELoss()
    #KLloss = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
    #maskLoss = KLloss(torch.log(stat_cls.squeeze(1)+1e-8),torch.log(prob_m+1e-8))/N
    maskLoss = BCEloss(stat_cls.squeeze(1),prob_m)
    
    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss \
        + f_velo * veloLoss + f_ego * egoLoss + f_mask * maskLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item(),
        'egoLoss': egoLoss.item(),
        'maskLoss': maskLoss.item(),
        }
    
    return total_loss, items




def computeloss_ol(pc1,pc2, pred_f, gt, vel1, pre_trans, stat_cls, gt_trans, prob_m, mask, interval, args):
    
    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    f_ego = 1.0
    f_mask = 1.0
    f_super = 1.0
    
    
    pc1_warp = pc1+pred_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, pred_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(pred_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 
    
    # ego-motion
    pc1_pre = torch.matmul(pre_trans[:,:3,:3], pc1)+pre_trans[:,:3,3].unsqueeze(2)
    pc1_gt = torch.matmul(gt_trans[:,:3,:3], pc1)+gt_trans[:,:3,3].unsqueeze(2)
    egoLoss = torch.mean(torch.norm(pc1_pre-pc1_gt,dim=1))
    
    # mask
    BCEloss = torch.nn.BCELoss()
    maskLoss = BCEloss(stat_cls.squeeze(1),prob_m)
    
    # noisy supervise 
    superviseLoss = torch.mean((1-mask)*torch.norm(gt-pred_f,dim=1)+1e-10)
    
    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss \
        + f_velo * veloLoss + f_ego * egoLoss + f_mask * maskLoss + f_super * superviseLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item(),
        'egoLoss': egoLoss.item(),
        'maskLoss': maskLoss.item(),
        'superviseLoss': superviseLoss.item(),
        }
    
    return total_loss, items


def computeloss_l(pc1,pc2, pred_f, gt, vel1, mask, stat_cls, interval, args):
    
    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    f_mask = 1.0
    f_super = 1.0
    
    pc1_warp = pc1+pred_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, pred_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(pred_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 
    
    # mask
    BCEloss = torch.nn.BCELoss()
    maskLoss = BCEloss(stat_cls.squeeze(1),mask)

    # noisy supervise 
    superviseLoss = torch.mean((1-mask)*torch.norm(gt-pred_f,dim=1)+1e-10)
    
    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss \
        + f_velo * veloLoss  + f_super * superviseLoss + f_mask * maskLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item(),
        'maskLoss': maskLoss.item(),
        'superviseLoss': superviseLoss.item(),
        }
    
    return total_loss, items

def computeloss_oc(pc1,pc2, pred_f, vel1, pre_trans, stat_cls, gt_trans, prob_m, \
                    interval, radar_u, radar_v, padding_opt, args):

    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    f_ego = 1.0
    f_mask = 1.0
    f_opt = 0.1
    
    pc1_warp = pc1+pred_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, pred_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # ego-motion
    pc1_pre = torch.matmul(pre_trans[:,:3,:3], pc1)+pre_trans[:,:3,3].unsqueeze(2)
    pc1_gt = torch.matmul(gt_trans[:,:3,:3], pc1)+gt_trans[:,:3,3].unsqueeze(2)
    egoLoss = torch.mean(torch.norm(pc1_pre-pc1_gt,dim=1))
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(pred_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 

    # mask
    BCEloss = torch.nn.BCELoss()
    maskLoss = BCEloss(stat_cls.squeeze(1),prob_m)

    # optical flow
    # project the warped points on the image plane
    valid_opt = (torch.norm(padding_opt,dim=2)!=0)
    # measure the distance from warped 3D points to camera rays
    end_pixels = torch.cat((radar_u.unsqueeze(2), radar_v.unsqueeze(2)),dim=2) + padding_opt
    opt_div = point_ray_distance(pc1_warp, end_pixels, args)
    opticalLoss = torch.mean(valid_opt*opt_div)
    
    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss \
        + f_velo * veloLoss + f_ego * egoLoss + f_mask * maskLoss + f_opt * opticalLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item(),
        'egoLoss': egoLoss.item(),
        'maskLoss': maskLoss.item(),
        'opticalLoss': opticalLoss.item(),
        }
    
    return total_loss, items

def computeloss_c(pc1,pc2, agg_f, vel1, interval, radar_u, radar_v, \
                                         padding_opt, args):

    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    f_opt = 1.0
    
    pc1_warp_a = pc1+agg_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp_a, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, agg_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(agg_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 
    
    # optical flow
    # project the warped points on the image plane
    # w_u, w_v = project_radar_to_image(pc1_warp_a, args)
    valid_opt = (torch.norm(padding_opt,dim=2)!=0)
    # measure the distance from warped 3D points to camera rays
    end_pixels = torch.cat((radar_u.unsqueeze(2), radar_v.unsqueeze(2)),dim=2) + padding_opt
    opt_div = point_ray_distance(pc1_warp_a, end_pixels, args)
    opticalLoss = torch.mean(valid_opt*(opt_div))
    #opt_div_u = valid_opt * (radar_u + padding_opt[:,:,0] - w_u)
    #opt_div_v = valid_opt * (radar_v + padding_opt[:,:,1] - w_v)
    #opticalLoss = torch.mean(torch.sqrt(opt_div_u**2+opt_div_v**2 + 1e-10))
    
    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss + f_velo * veloLoss \
                    + f_opt * opticalLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item(),
        'opticalLoss': opticalLoss.item(),
        }
    
    return total_loss, items


def computeloss_olc(pc1,pc2, pred_f, gt, vel1, pre_trans, stat_cls, gt_trans, prob_m, mask,\
                     interval, radar_u, radar_v, padding_opt, args):
    
    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    f_ego = 1.0
    f_mask = 1.0
    f_super = 1.0
    f_opt = 0.1
    
    
    pc1_warp = pc1+pred_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, pred_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(pred_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 
    
    # ego-motion
    pc1_pre = torch.matmul(pre_trans[:,:3,:3], pc1)+pre_trans[:,:3,3].unsqueeze(2)
    pc1_gt = torch.matmul(gt_trans[:,:3,:3], pc1)+gt_trans[:,:3,3].unsqueeze(2)
    egoLoss = torch.mean(torch.norm(pc1_pre-pc1_gt,dim=1))
    
    # mask
    BCEloss = torch.nn.BCELoss()
    maskLoss = BCEloss(stat_cls.squeeze(1),prob_m)
    
    # noisy supervise 
    superviseLoss = torch.mean((1-mask)*torch.norm(gt-pred_f,dim=1)+1e-10)
    
    # optical loss
     # optical flow
    # project the warped points on the image plane
    # w_u, w_v = project_radar_to_image(pc1_warp_a, args)
    valid_opt = (torch.norm(padding_opt,dim=2)!=0)
    # measure the distance from warped 3D points to camera rays
    end_pixels = torch.cat((radar_u.unsqueeze(2), radar_v.unsqueeze(2)),dim=2) + padding_opt
    opt_div = point_ray_distance(pc1_warp, end_pixels, args)
    opticalLoss = torch.mean(valid_opt*(opt_div))


    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss \
        + f_velo * veloLoss + f_ego * egoLoss + f_mask * maskLoss + f_super * superviseLoss\
        + f_opt * opticalLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item(),
        'egoLoss': egoLoss.item(),
        'maskLoss': maskLoss.item(),
        'superviseLoss': superviseLoss.item(),
        'opticalLoss': opticalLoss.item(),
        }
    
    return total_loss, items


def computeloss_l(pc1,pc2, pred_f, gt, vel1, pre_trans, \
                                        stat_cls, prob_m, interval, args):
    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    
    pc1_warp_a = pc1+pred_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp_a, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, pred_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(pred_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 

    # mask
    BCEloss = torch.nn.BCELoss()
    maskLoss = BCEloss(stat_cls.squeeze(1), prob_m)

    # noisy supervise 
    superviseLoss = torch.mean((1-prob_m)*torch.norm(gt-pred_f,dim=1)+1e-10)
    
    
    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss + f_velo*veloLoss + maskLoss + superviseLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item()*f_velo,
        'maskLoss': maskLoss.item(),
        'superviseLoss': superviseLoss.item(),
        }
    
    return total_loss, items

def RaFlow_optimize(pc1,pc2,vel1,interval,args):
    
    nepochs = 50
    lr = 0.1
    pred_f = torch.zeros(pc1.size(), requires_grad=True, device="cuda")
    opt = optim.Adam([pred_f], lr=lr)
    for i in range(nepochs):
        
        loss, loss_item = computeloss(pc1,pc2,pred_f,vel1,interval,args)
        opt.zero_grad() 
        loss.backward()
        opt.step()
    
    return pred_f
    
    
