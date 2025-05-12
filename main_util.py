import os
import argparse
import sys
import copy
import torch
import ujson
from time import clock
from tqdm import tqdm
import cv2

import numpy as np
from losses.flot_loss import FLOTLoss
from losses.pvraft_loss import PVRAFTLoss
from utils import *
from vis_util import *
from visualize.radar_vis_util import *
from models import *
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from losses import *


def extract_data_info(data):

    pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow = data
    pc1 = pc1.cuda().transpose(2,1).contiguous()
    pc2 = pc2.cuda().transpose(2,1).contiguous()
    ft1 = ft1.cuda().transpose(2,1).contiguous()
    ft2 = ft2.cuda().transpose(2,1).contiguous()
    radar_v = radar_v.cuda().float()
    radar_u = radar_u.cuda().float()
    opt_flow = opt_flow.cuda().float()
    mask = mask.cuda().float()
    trans = trans.cuda().float()
    interval = interval.cuda().float()
    gt = gt.cuda().float()

    return pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow


def train_one_epoch(args, net, train_loader, opt):
    
    num_examples = 0
    total_loss = 0
    mode = 'train'
    loss_items =  copy.deepcopy(loss_dict[args.model])
    ## for debug
    #res_ls = torch.zeros(100, 16, 256)
    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        
        ## reading data from dataloader and transform their format
        pc1, pc2, ft1, ft2, gt_trans, flow_label, \
            fg_mask, interval, radar_u, radar_v, opt_flow = extract_data_info(data)
        vel1 = ft1[:,0]

        batch_size = pc1.size(0)
        num_examples += batch_size

    
        ## feed data into the model and compute loss
        
        # supervised learning
        if args.model=='flownet3d':
            pred_f = net(pc1, pc2, ft1, ft2)
            pc1_warp = pred_f + pc1
            pred_b = net(pc1_warp, pc1, ft1, ft1)
            loss, items  = CycleL1Loss(pc1, pred_f, pred_b, flow_label)
        if args.model=='pointpwcnet_full':
            pred_f, fps_pc1_idxs, _, pc1, pc2 = net(pc1, pc2, ft1, ft2)
            loss, items = multiScaleLoss(pred_f, flow_label, fps_pc1_idxs)
        if args.model=='flot':
            pred_f = net([pc1.transpose(2,1), pc2.transpose(2,1)])
            loss, items = FLOTLoss(pred_f, flow_label)
        if args.model == 'flowstep3d_full':
            pred_f = net(pc1,pc2,ft1,ft2)
            loss, items = FlowStep3D_sv_loss(pc1, pc2, pred_f, flow_label)




        if args.model == 'radarsfemos':
            # pred_f,L_cls,trans = net([pc1.transpose(2,1), pc2.transpose(2,1)],ft1, ft2,interval)
            pred_f = net([pc1.transpose(2,1), pc2.transpose(2,1)],ft1, ft2,interval)
            pred_f = pred_f[-1].transpose(2,1).contiguous()
            #loss, items = PVRAFTLoss(pred_f, flow_label)
            loss_obj = RadarFlowLoss()
            #loss, items = loss_obj(args, pc1, pc2, pred_f[-1].transpose(2,1), vel1,L_cls,trans)
            loss, items = loss_obj(args, pc1, pc2, pred_f, vel1)





        # self-supervised or cross-modal supervised learning
        if args.model=='pointpwcnet':
            pred_f, _, _, pc1, pc2 = net(pc1, pc2, ft1, ft2)
            loss, items = multiScaleChamferSmoothCurvature(pc1, pc2, pred_f)
            pred_f = pred_f[0]
        if args.model=='jgwtf':
            pred_f,pred_b = net(pc1, pc2, ft1, ft2,'train')
            loss, items = NNCycle(pc1, pc2, pred_f,pred_b)
        if args.model=='gl':
            pred_f = net(pc1, pc2, ft1, ft2)
            loss, items = ChamferGL(pc1, pc2, pred_f)
        if args.model =='flowstep3d':
            pred_f = net(pc1,pc2,ft1,ft2)
            loss, items = FlowStep3D_self_loss(pc1, pc2, pred_f)
        if args.model == 'slim':
            Tr, Tr2, L_cls, L_wgt, F1, pc1, pc2 = net(pc1, pc2, ft1, ft2)
            loss, items = slim_loss.total_loss(pc1, pc2, F1, L_cls, L_wgt, Tr, Tr2)
            print(L_cls.shape) #torch.Size([B, 1, 256])
            
        if args.model=='raflow':
            _, pred_f, _,_ = net(pc1, pc2, ft1, ft2, interval)
            #pred_f = net(pc1, pc2, ft1, ft2, interval)
            loss_obj = RadarFlowLoss()
            loss, items = loss_obj(args, pc1, pc2, pred_f, vel1)
        opt.zero_grad() 
        loss.backward()
        opt.step()
        
        total_loss += loss.item() * batch_size
        

        for l in loss_items:
            loss_items[l].append(items[l]) 

  
    total_loss=total_loss*1.0/num_examples
    
    for l in loss_items:
        loss_items[l]=np.mean(np.array(loss_items[l]))
    
    return total_loss, loss_items


def eval_one_epoch(args, net, eval_loader, textio):

    if not args.model in ['gl_wo','icp', 'arfnet_o']:
        net.eval()
    
    if args.save_res: 
        args.save_res_path ='checkpoints/'+args.exp_name+"/results/"
        num_seq = 0
        clip_info = args.clips_info[num_seq]
        seq_res_path = os.path.join(args.save_res_path, clip_info['clip_name'])
        if not os.path.exists(seq_res_path):
            os.makedirs(seq_res_path)

    num_pcs=0 
    if args.vis:
        vis_path_2D='checkpoints/'+args.exp_name+"/test_vis_2d/"
        if not os.path.exists(vis_path_2D):
            os.makedirs(vis_path_2D)    
    sf_metric = {'rne':0, '50-50 rne': 0, 'mov_rne': 0, 'stat_rne': 0,\
                 'sas': 0, 'ras': 0, 'epe': 0, 'accs': 0, 'accr': 0}

    seg_metric = {'acc': 0, 'miou': 0, 'sen': 0}
    pose_metric = {'RRE': 0, 'RTE': 0}
    
    gt_trans_all = torch.zeros((len(eval_loader)*eval_loader.batch_size,4,4)).cuda()
    pre_trans_all = torch.zeros((len(eval_loader)*eval_loader.batch_size,4,4)).cuda()

    epe_xyz = {'x': [], 'y':[], 'z':[]}


    # start point for inference
    start_point = time.time()

    for i, data in tqdm(enumerate(eval_loader), total = len(eval_loader)):

    
        pc1, pc2, ft1, ft2, trans, pgt , mask, interval, radar_u, radar_v, padding_opt = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        ft1 = ft1.cuda().transpose(2,1).contiguous()
        ft2 = ft2.cuda().transpose(2,1).contiguous()
        mask = mask.cuda()

        interval = interval.cuda().float()
        gt = gt.cuda().float()

        batch_size = pc1.size(0)
        vel1 = ft1[:,0]
       
        with torch.no_grad():
            
            pred_t = None
            if args.model == 'flownet3d':
                pred_f = net(pc1,pc2,ft1,ft2)
            if args.model == 'flowstep3d_full':
                pred_f = net(pc1,pc2,ft1,ft2)
                pred_f = pred_f[-1].transpose(2,1).contiguous()
            if args.model == 'pointpwcnet_full':
                pred, _, _, _,_, = net(pc1, pc2, ft1, ft2)
                pred_f=pred[0]
            if args.model == 'flot':
                pred_f = net([pc1.transpose(2,1), pc2.transpose(2,1)])
                pred_f = pred_f.transpose(2,1)

            if args.model == 'radarsfemos':
                pred_f= net([pc1.transpose(2,1), pc2.transpose(2,1)],ft1, ft2,interval)
                pred_f = pred_f[-1].transpose(2,1).contiguous()

            if args.model == 'gl_wo':
                pred_f = GL_optimize(pc1,pc2)
            if args.model == 'gl':
                pred_f = net(pc1,pc2,ft1,ft2)
            if args.model=='pointpwcnet':
                pred, _, _, _,_, = net(pc1, pc2, ft1, ft2)
                pred_f=pred[0]
            if args.model=='jgwtf':
                pred_f = net(pc1, pc2, ft1, ft2,'test')
            if args.model =='flowstep3d':
                pred_f = net(pc1,pc2,ft1,ft2)
                pred_f = pred_f[-1].transpose(2,1).contiguous()
            if args.model =='slim':
                _, _, _, _, pred_f, _, _ = net(pc1, pc2, ft1, ft2)
            if args.model == 'icp':
                pred_f, pred_t = icp_flow(pc1,pc2)
            
            if args.model=='raflow':
                _, pred_f, pred_t, pred_m = net(pc1, pc2, ft1, ft2, interval)

            # use estimated scene to warp point cloud 1 
            pc1_warp=pc1 + pred_f


            # ## TODO
            # pse_mseg, _ = mseg_label_RRV(pc1, trans.cuda(), ft1[:,0], interval, args)
            # seg_res = eval_motion_seg(pse_mseg, mask)
            # pred_m = pse_mseg
            # for metric in seg_res:
            #     seg_metric[metric] += batch_size * seg_res[metric]

            if args.save_res:
                res = {
                    'pc1': pc1[0].cpu().numpy().tolist(),
                    'pc2': pc2[0].cpu().numpy().tolist(),
                    'pred_f': pred_f[0].cpu().detach().numpy().tolist(),
                    'gt_f': gt[0].transpose(0,1).contiguous().cpu().detach().numpy().tolist(),
                    'mask': mask[0].cpu().numpy().astype(float).tolist(),
                    'pred_m': pred_m[0].cpu().detach().numpy().astype(float).tolist(),
                }
                
                if num_pcs < clip_info['index'][1]:
                    res_path = os.path.join(seq_res_path, '{}.json'.format(num_pcs))
                else:
                    num_seq += 1
                    clip_info = args.clips_info[num_seq]
                    seq_res_path = os.path.join(args.save_res_path, clip_info['clip_name'])
                    if not os.path.exists(seq_res_path):
                        os.makedirs(seq_res_path)
                    res_path = os.path.join(seq_res_path, '{}.json'.format(num_pcs))
                
                ujson.dump(res,open(res_path, "w"))

            if args.vis:
                radarsfemos_visulize_result_2D(pc1,pc2,pc1_warp,num_pcs,vis_path_2D)#(pc1,pc2,wps,num_pcs,path):
            # evaluate the estimated results using ground truth
            batch_res = eval_scene_flow(pc1, pred_f.transpose(2,1).contiguous(), gt, mask, args)
            for metric in sf_metric:
                sf_metric[metric] += batch_size * batch_res[metric]

            epe_xyz['x'].append(batch_res['epe_x'])
            epe_xyz['y'].append(batch_res['epe_y'])
            epe_xyz['z'].append(batch_res['epe_z'])

            ## evaluate the foreground segmentation precision and recall
            if args.model in ['raflow','cmflow_l', 'cmflow_lc', 'cmflow_c','cmflow_o','cmflow_oc','cmflow_ol','cmflow_olc']:
                seg_res = eval_motion_seg(pred_m, mask)
                for metric in seg_res:
                    seg_metric[metric] += batch_size * seg_res[metric]
            
            ## Use scene flow correspondence to estimate rigid 3D transformation
            if pred_t is not None:
                pred_trans = pred_t
            else:
                pred_trans = rigid_transform_torch(pc1, pc1_warp)
            
            gt_trans_all[num_pcs:(num_pcs+batch_size)] = trans
            pre_trans_all[num_pcs:(num_pcs+batch_size)] = pred_trans   

            pose_res = eval_trans_RPE(trans, pred_trans)
            for metric in pose_res:
                pose_metric[metric] += batch_size * pose_res[metric]
            
            num_pcs+=batch_size

    # end point for inference
    infer_time = time.time()-start_point

    for metric in sf_metric:
        sf_metric[metric] = sf_metric[metric]/num_pcs
    for metric in seg_metric:
        seg_metric[metric] = seg_metric[metric]/num_pcs
    for metric in pose_metric:
        pose_metric[metric] = pose_metric[metric]/num_pcs

    textio.cprint('###The inference speed is %.3fms per frame###'%(infer_time*1000/num_pcs))

    return sf_metric, seg_metric, pose_metric, gt_trans_all, pre_trans_all, epe_xyz



def extract_dynamic_from_fg(mask, pc1, trans, gt):
    
    # get rigid flow labels for all points
    gt_sf_rg = rigid_to_flow(pc1,trans)
    
    gt = gt.transpose(2,1)
    gt_sf_rg = gt_sf_rg.transpose(2,1)
    # get non-rigid components for points
    flow_nr = gt_sf_rg - gt
    
    # obtain the motion segmentation mask 
    fg_mask = (mask!=1)
    mask[torch.norm(flow_nr*fg_mask.unsqueeze(2),dim=2)<0.05]=1
    mask[mask!=1] = 0

    return mask
    
    
def probabilistic_label_opt(pc1, trans, radar_u, radar_v, opt_flow, args):

    batch_size = pc1.size(0)
    npoints = pc1.size(2)
    
    gt_sf_rg = rigid_to_flow(pc1,trans)
    gt_wp_rg = gt_sf_rg + pc1
    end_pixels = torch.cat((radar_u.unsqueeze(2), radar_v.unsqueeze(2)),dim=2) + opt_flow
    rg_proj = project_radar_to_image(gt_wp_rg, args)
    residual = torch.norm(rg_proj - end_pixels, dim=2)
    prob_m = torch.exp(-(residual**2)/(2*args.sigma_opt**2))

    return prob_m


def probabilistic_label_RRV(pc1,trans,vel1,interval,args):
    
    batch_size = pc1.size(0)
    npoints = pc1.size(2)
    gt_sf_rg = rigid_to_flow(pc1,trans)
    gt_sf_rg_proj=torch.sum(gt_sf_rg*pc1,dim=1)/(torch.norm(pc1,dim=1))
    residual=(vel1*interval.unsqueeze(1)-gt_sf_rg_proj)
    prob_m = torch.exp(-(residual**2)/(2*args.sigma_rrv**2))

    return prob_m

def mseg_label_RRV(pc1, trans, vel1, interval, args):

    gt_sf_rg = rigid_to_flow(pc1,trans)
    gt_sf_rg_proj=torch.sum(gt_sf_rg*pc1,dim=1)/(torch.norm(pc1,dim=1))
    residual=abs(vel1-gt_sf_rg_proj/interval.unsqueeze(1))
    N = pc1.shape[2]
    #low_residual, _ = torch.topk(residual, np.int(args.bs_ratio*N), dim=1, largest=False)
    bs_residual = torch.mean(residual, dim=1).unsqueeze(1)
    #bs_residual = 0
    # 1 denotes static, 0 denotes moving
    mseg_label = ((residual-bs_residual)<args.vr_thres).type(torch.float32)

    return mseg_label, residual

def mseg_label_opt(pc1, trans, radar_u, radar_v, opt_flow, args):

    gt_sf_rg = rigid_to_flow(pc1,trans)
    gt_wp_rg = gt_sf_rg + pc1
    end_pixels = torch.cat((radar_u.unsqueeze(2), radar_v.unsqueeze(2)),dim=2) + opt_flow
    rg_proj = project_radar_to_image(gt_wp_rg, args)
    residual = torch.norm(rg_proj - end_pixels, dim=2)

    mseg_label = ((residual)<args.opt_thres).type(torch.float32)
    #prob_m = torch.exp(-(residual**2)/(2*args.sigma_opt**2))

    return mseg_label

def plot_loss_epoch(train_items_iter, args, epoch):
    
    plt.clf()
    plt.plot(np.array(train_items_iter['Loss']).T, 'b')
    plt.plot(np.array(train_items_iter['nnLoss']).T, 'r')
    plt.plot(np.array(train_items_iter['chamferLoss']).T, 'k')
    plt.plot(np.array(train_items_iter['veloLoss']).T, 'g')
    plt.plot(np.array(train_items_iter['smoothnessLoss']).T, 'c')
    plt.plot(np.array(train_items_iter['cycleLoss']).T, 'y')
    plt.plot(np.array(train_items_iter['egoLoss']).T, 'm')
    plt.plot(np.array(train_items_iter['maskLoss']).T, 'r')
    plt.plot(np.array(train_items_iter['opticalLoss']).T, 'y')
    plt.plot(np.array(train_items_iter['superviseLoss']).T, 'r')
    plt.plot(np.array(train_items_iter['L1Loss']).T, 'm')
    plt.legend(['Total','nnLoss','chamferLoss','veloLoss','Smoothness', 'Cycle','egoLoss', 'maskLoss',\
        'opticalLoss', 'superviseLoss', 'L1Loss', ],loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/loss_train/loss_train_%s.png' %(args.exp_name,epoch),dpi=500)
 
    

