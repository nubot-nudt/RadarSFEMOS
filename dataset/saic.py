#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import ujson
from utils import *
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class saicDataset(Dataset):
    
    def __init__(self, args, root='/home/toytiny/SAIC_radar/radar_pcs/', partition='train'):
        
    
        self.npoints = args.num_points
        self.get_radar_calib()
        self.aug = args.aug
        self.res = {'r_res': 0.2, # m
                    'theta_res': 1 * np.pi/180, # radian
                    'phi_res': 1.6 *np.pi/180  # radian
            }
        self.partition = partition
        self.root = root+partition+'/'
        self.pc_ls=sorted(os.listdir(self.root),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))
        self.scene_nbr=int(self.pc_ls[-1].split("-")[1].split("_")[0])
        self.datapath={'sample':[]}
        for idx in range(0,len(self.pc_ls)):
            self.datapath['sample'].append(self.root+self.pc_ls[idx])

        print(self.partition, ': ', len(self.datapath['sample']))
        
    def __getitem__(self, index):
        
    
        sample = self.datapath['sample'][index]
 
   
        with open(sample, 'rb') as fp:
            data = ujson.load(fp)
 
        data_1 = data["pc1"]
        data_2 = data["pc2"]
        
        ## obtain groundtruth for multiple tasks during test
        if self.partition =='test':
            trans = np.linalg.inv(np.array(data["trans"]))
            gt = np.array(data["gt"])
            mask = np.array(data["mask"])
            mask[mask <1] = 0
            mask[mask==1] = 1 
            radar_u = np.zeros(mask.shape[0])
            radar_v = np.zeros(mask.shape[0])
            padding_opt = np.zeros((mask.shape[0],2))

            
        else:
            trans = np.linalg.inv(np.array(data["trans"]))
            gt = np.array(data["gt"])
            mask = np.array(data["mask"])
            opt_info = data["opt_info"]
            radar_u = np.array(opt_info["radar_u"])
            radar_v = np.array(opt_info["radar_v"])
            # fill the index with zero where points have no cooresponding optical flow 
            filt = np.array(opt_info["filt"])
            opt_flow = np.array(opt_info["opt_flow"])
            padding_opt = np.zeros((radar_u.shape[0],2))
            padding_opt[filt] = opt_flow


        # read input data and features
        interval = data["interval"]
        pos1=np.vstack((data_1['car_loc_x'],data_1['car_loc_y'],data_1['car_loc_z'])).T.astype('float32')
        pos2=np.vstack((data_2['car_loc_x'],data_2['car_loc_y'],data_2['car_loc_z'])).T.astype('float32')
        vel1=np.array(data_1['car_vel_r']).astype('float32')
        vel2=np.array(data_2['car_vel_r']).astype('float32')
        rcs1=np.array(data_1['rcs']).astype('float32')
        rcs2=np.array(data_2['rcs']).astype('float32')
        power1=np.array(data_1['power']).astype('float32')
        power2=np.array(data_2['power']).astype('float32')
        feature1 = np.vstack((vel1,rcs1,power1)).T
        feature2 = np.vstack((vel2,rcs2,power2)).T
        
        ## downsample to npoints to enable fast batch processing (not in test)
        if self.partition!='test':

            sample_idx1 = np.random.choice(pos1.shape[0], self.npoints, replace=False)
            sample_idx2 = np.random.choice(pos2.shape[0], self.npoints, replace=False)
            
            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            feature1 = feature1[sample_idx1, :]
            feature2 = feature2[sample_idx2, :]
            radar_u = radar_u[sample_idx1]
            radar_v = radar_v[sample_idx1]
            padding_opt = padding_opt[sample_idx1,:]

            gt = gt[sample_idx1,:]
            mask = mask[sample_idx1]
            
         
        ## data augmentation
        if self.aug and self.partition not in ['test', 'val']  :
            
            t_1 = np.eye(4).astype(np.float32)
            t_2 = np.eye(4).astype(np.float32)
            
            # rotation
            yaw_1,pitch_1,roll_1 = np.random.uniform(-0.01,0.01,size=3)
            yaw_2,pitch_2,roll_2 = np.random.uniform(-0.01,0.01,size=3)
            angles_1 = [yaw_1, pitch_1,roll_1]
            angles_2 = [yaw_2, pitch_2,roll_2]
            rot1 = R.from_euler('ZYX', angles_1 , degrees=True)
            rot_m1 = rot1.as_matrix()
            rot2 = R.from_euler('ZYX', angles_2 , degrees=True)
            rot_m2 = rot2.as_matrix()
            
            # translation 
            shift_x1, shift_x2 = np.random.uniform(-0.01,0.01,size=2)
            shift_y1, shift_y2 = np.random.uniform(-0.01,0.01,size=2)
            shift_z1, shift_z2 = np.random.uniform(-0.01,0.01,size=2)
            shift_1 = np.array([shift_x1,shift_y1,shift_z1])
            shift_2 = np.array([shift_x2,shift_y2,shift_z2])
            t_1[0:3,0:3] = rot_m1.astype(np.float32)
            t_2[0:3,0:3] = rot_m2.astype(np.float32)
            t_1[0:3,3] = shift_1.astype(np.float32)
            t_2[0:3,3] = shift_2.astype(np.float32)
    
            # apply the random transformation to points
            pos1 = (np.matmul(t_1[0:3, 0:3], pos1.transpose()) + t_1[0:3,3:4]).transpose()
            pos2 = (np.matmul(t_2[0:3, 0:3], pos2.transpose()) + t_2[0:3,3:4]).transpose()
            # update the groundtruth transformation and groundtruth flow
            trans = t_2 @ trans @ np.linalg.inv(t_1)
            #gt = (np.matmul(trans[0:3, 0:3], pos1.transpose()) + trans[0:3,3:4]).transpose() - pos1
            
            
        return pos1, pos2, feature1, feature2, trans, gt, mask, interval, radar_u, radar_v, padding_opt
                
    def get_radar_calib(self):

        self.radar_ext = [0.06, -0.2, 0.7,-3.5, 2, 180]
        self.cam_ext = [-1.793, -0.036, 1.520, -1.66, -0.09, -0.9]
        self.cam_ins = {'fx': 1146.501, 'fy': 1146.589, 'cx': 971.982, 'cy': 647.093}

    def __len__(self):
        return len(self.datapath['sample'])