#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from utils import *
from dataset import *
from models import *
import numpy as np
from losses import *
from matplotlib import pyplot as plt
from main_util import train_one_epoch, plot_loss_epoch, eval_one_epoch
from clip_util import test_one_epoch_seq, train_one_epoch_seq, eval_one_epoch_seq
from vis_util import *

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'loss_train'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'loss_train')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/dataset/vod.py')


def test(args, net, test_loader, textio):

    if args.dataset == 'vodClipDataset':
        sf_metric, seg_metric, pose_metric, gt_trans, pre_trans, epe_xyz = test_one_epoch_seq(args, net, test_loader, textio)
    else:
        sf_metric, seg_metric, pose_metric, gt_trans, pre_trans, epe_xyz = eval_one_epoch(args, net, test_loader, textio)
    ## print scene flow evaluation results
    for metric in sf_metric:
        textio.cprint('###The mean {}: {}###'.format(metric, sf_metric[metric]))

    ## print motion seg evaluation results
    for metric in seg_metric:
        textio.cprint('###The mean {}: {}###'.format(metric, seg_metric[metric]))
    ## print ego_motion evaluation results
    for metric in pose_metric:
        textio.cprint('###The mean {}: {}###'.format(metric, pose_metric[metric]))

    ## plot the vehicle trajectory 
    plot_trajectory(gt_trans,pre_trans,args)
    plot_comparison(gt_trans,pre_trans,args)
    plot_epe_component(epe_xyz,args)


# ## TODO
    
def test_vis(args, net, test_loader, textio):

    args.vis_path_2D='checkpoints/'+args.exp_name+"/test_vis_2d/"
    args.vis_path_3D='checkpoints/'+args.exp_name+"/test_vis_3d_input/"
    args.vis_path_seg='checkpoints/'+args.exp_name+"/test_vis_seg/"
    args.vis_path_seg_pse = 'checkpoints/'+args.exp_name+"/test_vis_seg_pse/"
    args.vis_path_reg='checkpoints/'+args.exp_name+"/test_vis_reg/"

    if not os.path.exists(args.vis_path_2D):
        os.makedirs(args.vis_path_2D)
    if not os.path.exists(args.vis_path_3D):
        os.makedirs(args.vis_path_3D)
    if not os.path.exists(args.vis_path_seg):
        os.makedirs(args.vis_path_seg)
    if not os.path.exists(args.vis_path_seg_pse):
        os.makedirs(args.vis_path_seg_pse)
    if not os.path.exists(args.vis_path_reg):
        os.makedirs(args.vis_path_reg)


    sf_metric, seg_metric, pose_metric, gt_trans, pre_trans, epe_xyz = eval_one_epoch(args, net, test_loader, textio)
    ## print scene flow evaluation results
    for metric in sf_metric:
        textio.cprint('###The mean {}: {}###'.format(metric, sf_metric[metric]))

    ## print motion seg evaluation results
    for metric in seg_metric:
        textio.cprint('###The mean {}: {}###'.format(metric, seg_metric[metric]))
    ## print ego_motion evaluation results
    for metric in pose_metric:
        textio.cprint('###The mean {}: {}###'.format(metric, pose_metric[metric]))

    ## plot the vehicle trajectory 
    plot_trajectory(gt_trans,pre_trans,args)
    plot_comparison(gt_trans,pre_trans,args)
    plot_epe_component(epe_xyz,args)

  
def train(args, net, train_loader, val_loader, textio):
    
    
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(opt, args.decay_epochs, gamma = args.decay_rate)

    best_val_res = 0
    train_loss_ls = np.zeros((args.epochs))
    val_score_ls = np.zeros(args.epochs)
    train_items_iter = {
                    'Loss': [],'nnLoss': [],'smoothnessLoss': [],'veloLoss': [],
                    'cycleLoss': [],'curvatureLoss':[],'chamferLoss': [],'L2Loss': [], 'glLoss': [],
                    'egoLoss':[], 'maskLoss': [], 'superviseLoss': [], 'opticalLoss': [], 'L1Loss': [],
                    }
    
    for epoch in range(args.epochs):
        
        textio.cprint('====epoch: %d, learning rate: %f===='%(epoch, opt.param_groups[0]['lr']))

        textio.cprint('==starting training on the training set==')

        total_loss, loss_items = train_one_epoch(args, net, train_loader, opt)

        train_loss_ls[epoch] = total_loss
        for it in loss_items:
            train_items_iter[it].extend([loss_items[it]])
        textio.cprint('mean train loss: %f'%total_loss)

        textio.cprint('==starting evaluation on the validation set==')

        sf_metric, _, _, _, _,_ = eval_one_epoch(args, net, val_loader, textio)
        eval_score = sf_metric['accs']
        val_score_ls[epoch] = eval_score
        textio.cprint('mean ACCS score: %f'%eval_score)
        
        if best_val_res <= eval_score:
            best_val_res = eval_score
            textio.cprint('best val score till now: %f'%best_val_res)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
        
        scheduler.step()
        
        plot_loss_epoch(train_items_iter, args, epoch)

    textio.cprint('====best ACCS score after %d epochs: %f===='%(args.epochs, best_val_res)) 
    plt.clf()
    plt.plot(train_loss_ls[0:int(args.epochs)], 'b')
    plt.legend(['train_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/loss_train/train_loss.png' % args.exp_name,dpi=500)
    
    plt.clf()
    plt.plot(val_score_ls[0:int(args.epochs)], 'r')
    plt.legend(['val_score'])
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.savefig('checkpoints/%s/val_score.png' % args.exp_name,dpi=500)

    return best_val_res
   

def main(param_name='eval', param_value=False):
    
    args = parse_args_from_yaml("configs.yaml")
    
    #args[param_name] = param_value
    #args['exp_name'] = args['model'] + '_' + param_name + '_' + str(param_value)

    # CUDA settings
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    # deterministic results
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    
    # init checkpoint records 
    _init_(args)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    
    # init dataset and dataloader
    if args.eval:
        test_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition=args.eval_split,textio=textio)
        test_loader = DataLoader(test_set,num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False)
    else:
        train_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition=args.train_set,textio=textio)
        val_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition='val', textio=textio)
        train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, num_workers=args.num_workers, batch_size=args.val_batch_size, shuffle=False, drop_last=False)
            
    
    # update dataset extrisic and intrinsic to args
    if not args.eval:
        args.camera_projection_matrix = train_set.camera_projection_matrix
        args.t_camera_radar = train_set.t_camera_radar
        args.radar_res = train_set.res
    else:
        args.camera_projection_matrix = test_set.camera_projection_matrix
        args.t_camera_radar = test_set.t_camera_radar
        args.radar_res = test_set.res
    if args.eval:
        args.clips_info = test_set.clips_info
    

    # init the network (load or from scratch)
    net = init_model(args)
    
    
    if args.eval:
        best_val_res = None
        if args.vis:
            textio.cprint('==Enable Visulization==')
            test_vis(args, net, test_loader,textio)
        else:
            test(args, net, test_loader,textio)
    else:
        best_val_res = train(args, net, train_loader, val_loader,textio)

    textio.cprint('Max memory alocation: {}MB'.format(torch.cuda.max_memory_allocated(device=0)/1e6)) 
    print('FINISH')

    if best_val_res is not None:
        return best_val_res

if __name__ == '__main__':
    main()