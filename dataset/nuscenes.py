#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import ujson
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R



class nuScenesDataset(Dataset):
    
    def __init__(self, npoints=256, root='/home/toytiny/nuscenes/radar_pcs/', partition='train'):
        self.npoints = npoints
        self.partition = partition
        self.root = root+partition+'/'
        self.pc_ls=sorted(os.listdir(self.root),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))
        self.scene_nbr=int(self.pc_ls[-1].split("-")[1].split("_")[0])
        
        self.cache = {}
        self.cache_size = 30000
        self.datapath={'pc1':[],'pc2':[]}
        for idx in range(0,len(self.pc_ls)-1):
            # the last point cloud of each scene do not have the next consecutive sweep
            if not int(self.pc_ls[idx].split("-")[1].split("_")[0])-int(self.pc_ls[idx+1].split("-")[1].split("_")[0]):
                self.datapath['pc1'].append(self.root+self.pc_ls[idx])
                self.datapath['pc2'].append(self.root+self.pc_ls[idx+1])
    
        print(self.partition, ': ', len(self.datapath['pc1']))
        
    def __getitem__(self, index):
        
        if index in self.cache:
            pos1, pos2, vel1, vel2, rcs1, rcs2, prop1, prop2 = self.cache[index]
            
        else:
            dp1 = self.datapath['pc1'][index]
            dp2 = self.datapath['pc2'][index]
            
            with open(dp1, 'rb') as fp1:
                data_1 = ujson.load(fp1)
                
            with open(dp2, 'rb') as fp2:
                data_2 = ujson.load(fp2)
                
            pos1=np.vstack((data_1['car_loc_x'],data_1['car_loc_y'],np.zeros(len(data_1['car_loc_z'])))).T.astype('float32')
            pos2=np.vstack((data_2['car_loc_x'],data_2['car_loc_y'],np.zeros(len(data_2['car_loc_z'])))).T.astype('float32')
            vel1=np.array(data_1['car_vel_r']).astype('float32')
            vel2=np.array(data_2['car_vel_r']).astype('float32')
            rcs1=np.array(data_1['rcs']).astype('float32')
            rcs2=np.array(data_2['rcs']).astype('float32')
            prop1=np.array(data_1['prop'])
            prop2=np.array(data_2['prop'])
            
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, vel1, vel2, prop1, rcs1, rcs2)
            
        if self.partition == 'train' or 'mini_train' or 'mini_val':
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
                
            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            vel1 = vel1[sample_idx1]
            vel2 = vel2[sample_idx2]
            prop1 = prop1[sample_idx1]
            prop2 = prop2[sample_idx2]
            rcs1 = rcs1[sample_idx1]
            rcs2 = rcs2[sample_idx2]
            
        else:
                
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            vel1 = vel1[:self.npoints]
            vel2 = vel2[:self.npoints]
            prop1 = prop1[:self.npoints]
            prop2 = prop2[:self.npoints]
            rcs1 = rcs1[:self.npoints]
            rcs2 = rcs2[:self.npoints]
                
                
        return pos1, pos2, vel1, vel2, rcs1, rcs2, prop1, prop2
                
        
    def __len__(self):
        return len(self.datapath['pc1'])