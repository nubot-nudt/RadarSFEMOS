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


class carlaDataset(Dataset):
    
    def __init__(self, args, root='/home/toytiny/carla_radar/radar_pcs/', partition='train'):
        
    
        self.npoints = args.num_points
        self.partition = partition
        self.root = root+partition+'/'
        self.interval = args.interval
        self.pc_ls=sorted(os.listdir(self.root),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))
        self.scene_nbr=int(self.pc_ls[-1].split("-")[1].split("_")[0])
        self.datapath={'pc1':[],'pc2':[],'rev': []}
        for idx in range(0,len(self.pc_ls)-1):
            # the last point cloud of each scene do not have the next consecutive sweep
            if not int(self.pc_ls[idx].split("-")[1].split("_")[0])-int(self.pc_ls[idx+1].split("-")[1].split("_")[0]):
                self.datapath['pc1'].append(self.root+self.pc_ls[idx])
                self.datapath['pc2'].append(self.root+self.pc_ls[idx+1])
                self.datapath['rev'].append(1)
                ## Temporal flip augmentation
                if self.partition=='t':
                    self.datapath['pc1'].append(self.root+self.pc_ls[idx+1])
                    self.datapath['pc2'].append(self.root+self.pc_ls[idx])
                    self.datapath['rev'].append(-1)
    
        print(self.partition, ': ', len(self.datapath['pc1']))
        
    def __getitem__(self, index):
        
    
        dp1 = self.datapath['pc1'][index]
        dp2 = self.datapath['pc2'][index]
        rev = self.datapath['rev'][index]
   
        with open(dp1, 'rb') as fp1:
            data_1 = ujson.load(fp1)
                
        with open(dp2, 'rb') as fp2:
            data_2 = ujson.load(fp2)
                
        if rev==-1:    
            trans=np.array(data_1['transform']).astype('float32')
        else:
            trans=np.linalg.inv(np.array(data_1['transform']).astype('float32'))
            
        pos1=np.vstack((data_1['car_loc_x'],data_1['car_loc_y'],data_1['car_loc_z'])).T.astype('float32')
        pos2=np.vstack((data_2['car_loc_x'],data_2['car_loc_y'],data_2['car_loc_z'])).T.astype('float32')
        vel1=rev*np.array(data_1['car_vel_r']).astype('float32')
        vel2=rev*np.array(data_2['car_vel_r']).astype('float32')
        rcs1=np.zeros(len(data_1['car_vel_r'])).astype('float32')
        rcs2=np.zeros(len(data_2['car_vel_r'])).astype('float32')
        prop1=np.zeros(len(data_1['car_vel_r'])).astype('float32')
        prop2=np.zeros(len(data_2['car_vel_r'])).astype('float32')
        gt=rev*np.array(data_1['sf']).astype('float32')
        mask = np.array(data_1['mask'])
       
            
        
        n1 = pos1.shape[0]
        if n1>=self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.arange(0,n1)
            sample_idx1=np.append(sample_idx1,np.random.choice(n1,self.npoints-n1,replace=True))
        n2 = pos2.shape[0]
        if n2>=self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.arange(0,n2)
            sample_idx2=np.append(sample_idx2,np.random.choice(n2,self.npoints-n2,replace=True))
                
        pos1 = pos1[sample_idx1, :]
        pos2 = pos2[sample_idx2, :]
        vel1 = vel1[sample_idx1]
        vel2 = vel2[sample_idx2]
        prop1 = prop1[sample_idx1]
        prop2 = prop2[sample_idx2]
        rcs1 = rcs1[sample_idx1]
        rcs2 = rcs2[sample_idx2]
        gt = gt[sample_idx1,:]
        mask = mask[sample_idx1]
        interval = self.interval
            

        return pos1, pos2, vel1, vel2, rcs1, rcs2, prop1, prop2, trans, gt, mask, interval
                
        
    def __len__(self):
        return len(self.datapath['pc1'])