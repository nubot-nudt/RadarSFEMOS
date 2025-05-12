import os
import json
import ujson
import numpy as np
import copy
from time import *
import sys
import shutil
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pose_extract import get_trans_from_gnssimu, get_matrix_from_ext, get_interpolate_pose

ROOT_PATH = "/home/toytiny/SAIC_radar/scene_flow_data/"

SAM_PATH = "/home/toytiny/SAIC_radar/" + "samples_5/"
IMG_PATH = "/home/toytiny/SAIC_radar/" + "sample_imgs/"

SIDE_RANGE = (-50, 50)
FWD_RANGE = (0, 100)
HEIGHT_RANGE = (-10,10) # (-10, 10)
RES = 0.15625
RADAR_EXT = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
# utc local
BASE_TS_LS = {'seq_1': [1642484600284,1642484600826],
              'seq_2': [1643179942119,1643179944003],
              'seq_3': [1643180543397,1643180545286],
              'seq_4': [1643181144484,1643181146376],
              'seq_5': [1643181745461,1643181747357],
              'seq_6': [1643182346486,1643182348386],
              'seq_7': [1643182947438,1643182949343]
              }

def get_rotation(arr):
    x,y,_ = arr
    yaw = np.arctan(y/x)
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

def get_bbx_param(obj_info):
    
    center = obj_info[2:5] #+ np.array([-2.5, 0, 0])
    # enlarge the box field to include points with meansure errors
    extent = obj_info[5:8] + 1.0
    angle = obj_info[8:11]
    rot_m = get_rotation(angle)
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    
    return obbx

def transform_bbx(obj_bbx,trans):
    
    # eight corner points 
    bbx_pnts = obj_bbx.get_box_points()
    o3d_pnts = o3d.geometry.PointCloud()
    o3d_pnts.points = bbx_pnts
    # transform eight points
    o3d_pnts.transform(trans)
    obj_bbx = o3d.geometry.OrientedBoundingBox.create_from_points(o3d_pnts.points)
    
    return obj_bbx
    
def get_flow_label(target1,tran,gt1,gt2,radar_to_ego,mode):
    
    pc1 = np.vstack((target1["car_loc_x"], target1["car_loc_y"], target1["car_loc_z"]))
    num_pnts = np.size(pc1,1)
    pc1 = o3d.utility.Vector3dVector(pc1.T)
    num_obj = np.size(gt1,0)
    labels = np.zeros((num_pnts,3),dtype=np.float32)
    mask = np.zeros(num_pnts,dtype=np.float32)
    in_idx_ls = []
    in_confs = np.zeros(num_pnts,dtype=np.float32)
    in_labels = np.zeros((num_pnts,3),dtype=np.float32)
    
    # get flow labels for points within objects 
    for i in range(num_obj):
        if gt1.ndim==2 and gt2.ndim==2:
            track_id1 = gt1[i,-2]
            next_idx = np.where(gt2[:,-2] == track_id1)[0]
            if len(next_idx)!=0 and not (gt1[i,5:8]<0.1).any(): # avoid too small boxes 
                # object in the first frame
                obj1 = gt1[i,:]
                obj_bbx1 = get_bbx_param(obj1)
                bbx1 = transform_bbx(obj_bbx1,radar_to_ego)
                # object in the second frame
                obj2 = gt2[next_idx[0],:]
                obj_bbx2 = get_bbx_param(obj2)
                bbx2 = transform_bbx(obj_bbx2,radar_to_ego)
                # select radar points within the bounding box in the first frame
                in_idx = bbx1.get_point_indices_within_bounding_box(pc1)
                if len(in_idx)>0:
                    in_labels[in_idx] = bbx2.center-bbx1.center   
                    in_confs[in_idx] =  obj1[-1]
                    in_idx_ls.extend(in_idx)
            else: 
                continue
        else:
            continue
        
    if mode=='test':
        # get rigid flow labels for all points
        pc1_rg = o3d.utility.Vector3dVector(np.asarray(pc1))
        pc1_geo = o3d.geometry.PointCloud()
        pc1_geo.points = pc1_rg
        pc1_tran = pc1_geo.transform(np.linalg.inv(tran)).points
        flow_r = np.asarray(pc1_tran)-np.asarray(pc1_rg)
        
        # get non-rigid components for inbox points
        flow_nr = in_labels[in_idx_ls] - flow_r[in_idx_ls]
    
        # obtain the index for foreground (dynamic) points 
        fg_idx = np.array(in_idx_ls)[np.linalg.norm(flow_nr,axis=1)>0.05]
        
    else:
        fg_idx = in_idx_ls
        flow_r = np.zeros((np.size(np.asarray(pc1),0),np.size(np.asarray(pc1),1)))
        
        
    if len(fg_idx)>0:
        bg_idx = np.delete(np.arange(0,num_pnts),fg_idx)
    else:
        bg_idx = np.arange(0,num_pnts)

    # fill the labels of foreground and background, obtain the mask
    mask[bg_idx] = 1
    labels[bg_idx] = flow_r[bg_idx]
    if len(fg_idx)>0:
        labels[fg_idx] = in_labels[fg_idx]
        mask[fg_idx] = 1-in_confs[fg_idx]
    
 
   
    return labels, mask
        
    
def get_radar_target(data,ts,trans,pose_ts):

    ## use the original right-hand coordinate systen, front is x, left is y, up is z
    x_points = data[0, :]
    y_points = data[1, :]
    z_points = data[2, :]
    vel_r = data[7, :]
    rcs = data[6,:]
    power = data[5,:]
    
    # dis = np.sqrt(x_points**2+y_points**2+z_points**2)
    # if dis.max()<100:
    #     state = 'near'
    # else:
    #     state = 'mid'
    f_filt = np.logical_and((x_points > FWD_RANGE[0]), (x_points < FWD_RANGE[1]))
    s_filt = np.logical_and((y_points > SIDE_RANGE[0]), (y_points < SIDE_RANGE[1]))
    h_filt = np.logical_and((z_points > HEIGHT_RANGE[0]), (z_points < HEIGHT_RANGE[1]))
    filt = np.logical_and(np.logical_and(f_filt, s_filt), h_filt)
    indices = np.argwhere(filt).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    points = np.vstack((x_points, y_points, z_points)).T
    N = points.shape[0]
    vel_r = vel_r[indices]
    rcs = rcs[indices]
    power = power[indices]

    # SHIFT to the BEV view

    x_img = np.floor((x_points) / RES)
    y_img = np.floor(-(y_points + SIDE_RANGE[0])/RES)
    bev_vel_r = vel_r / RES
    bev_vel_r[np.isnan(bev_vel_r)] = 0
    vel_r = bev_vel_r * RES

    # match radar and pose timestamps
    diff = abs(ts - pose_ts)
    idx = diff.argmin()
    pose = trans[idx]
    
    targets = {
        "car_loc_x": x_points,
        "car_loc_y": y_points,
        "car_loc_z": z_points,
        "car_vel_r": vel_r,
        "bev_loc_x": x_img,
        "bev_loc_y": y_img,
        "bev_vel_r": bev_vel_r,
        "rcs": rcs,
        "power": power,
        "pose": pose,
    }

    return targets

def mask_show(mask,target,num_pcs,img_path):
    
    img_path = img_path + "/" + "mask/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    x_img = np.array(target["bev_loc_x"]).astype(np.int32)
    y_img = np.array(target["bev_loc_y"]).astype(np.int32)
    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max,3], dtype=np.uint8)+255
    for i in range(len(x_img)):
        if mask[i]==1:
            im=cv2.circle(im,(x_img[i],y_img[i]),3,(255,0,0))
        else:
            im=cv2.circle(im,(x_img[i],y_img[i]),3,(0,0,255))
            
    im=cv2.putText(im, 'Static', (300,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    im=cv2.putText(im, 'Moving', (300,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    path = img_path + "{}.jpg".format(num_pcs)
    cv2.imwrite(path, im)

def align_show(radar1, radar2, tran, num_pcs,img_path):
    
    tran = np.eye(4)
    img_path = img_path + "/" + "align/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    nps_2 = len(radar2['car_loc_x'])
    nps_1 = len(radar1['car_loc_x'])
    pnts_2 = np.vstack((radar2['car_loc_x'],radar2['car_loc_y'],radar2['car_loc_z'])).T
    h_2 = np.hstack((pnts_2, np.ones((nps_2, 1))))
    a_2 = np.dot(tran, h_2.T)[:3].T 
    x_img_2 = np.floor((a_2[:,0]) / RES).astype(np.int32)
    y_img_2 = np.floor(-(a_2[:,1] + SIDE_RANGE[0])/RES).astype(np.int32)
    x_img_1 = np.array(radar1["bev_loc_x"]).astype(np.int32)
    y_img_1 = np.array(radar1["bev_loc_y"]).astype(np.int32)
    
    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max,3], dtype=np.uint8)+255
    for i in range(nps_1):
        im=cv2.circle(im,(x_img_1[i],y_img_1[i]),2,(255,0,0))
    for i in range(nps_2):
        im=cv2.circle(im,(x_img_2[i],y_img_2[i]),2,(0,255,0))
    im=cv2.putText(im, 'PC1', (300,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    im=cv2.putText(im, 'PC2_Aligned', (280,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    path = img_path + "{}.jpg".format(num_pcs)
    cv2.imwrite(path, im)
    

def route_plot(poses,seq):
    
    poses = np.array(poses)
    #end = -1
    x_ego = poses[:,0,3]
    y_ego = poses[:,1,3]
    z_ego = poses[:,2,3]
    #delta_x = x_ego[1:]-x_ego[:-1]
    #delta_y = y_ego[1:]-y_ego[:-1]
    #speed = np.hstack((0,np.sqrt(delta_x**2+delta_y**2)))*100
    
    plt.figure()
    plt.winter()
    plt.scatter(-y_ego,x_ego, s=0.5, c=z_ego)
    #plt.plot(x_ego,'r')
    #plt.plot(y_ego,'g')
    #plt.plot(z_ego[:1000],'b')
    
    # x_range = (-2000,2000)
    # y_range = (-2000,2000)
    # res = 5
    # x_img = np.floor((x_ego-x_range[0])/res).astype(np.int32)
    # y_img = np.floor(-(y_ego+y_range[0])/res).astype(np.int32)
    # x_max = int((x_range[1] - x_range[0]) / res)
    # y_max = int((y_range[1] - y_range[0]) / res)
    # im = np.zeros([y_max, x_max,3], dtype=np.uint8)+255
    # for i in range(len(x_ego)):
        # im=cv2.circle(im,(x_img[i],y_img[i]),2,(255,0,0))
        # cv2.imshow('route',im)
        # cv2.waitKey(1)
    plt.xlabel('X [m]',fontdict={'fontsize': 12, 'fontweight': 'medium'})
    plt.ylabel('Y [m]',fontdict={'fontsize': 12, 'fontweight': 'medium'})
    cb=plt.colorbar()
    plt.tight_layout()
    plt.xlim([-4000,4000])
    plt.ylim([-4000,4000])
    plt.tick_params(labelsize=12)
    cb.set_label('Z [m]',fontdict={'fontsize': 12, 'fontweight': 'medium'})
    cb.ax.tick_params(labelsize=12)
    # cv2.imwrite(path, im)
    path = seq + "route.jpg"
    #plt.savefig(path, dpi=600)
    delta_x = x_ego[1:]-x_ego[:-1]
    delta_y = y_ego[1:]-y_ego[:-1]
    drive_dis = np.sum(np.sqrt(delta_x**2+delta_y**2))
    
    print(drive_dis)
    
    return drive_dis

def main():
    
    if not os.path.exists(SAM_PATH):
        os.makedirs(SAM_PATH)
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
        
    seqs = sorted(os.listdir(ROOT_PATH))
    #splits = {'val': [seqs[6]]}
    #splits = {'val_non': [seqs[6]]}
    splits = {'test': [seqs[0]], 'train' : seqs[1:6], 'val': [seqs[6]]}
    #splits = {'test': [seqs[0]]}
    ## extrinsic parameters of radar
    ego_to_radar = get_matrix_from_ext(RADAR_EXT)
    radar_to_ego = np.linalg.inv(ego_to_radar)
    dis_all = 0
    ## Read, process and save samples 
    for split in splits:
        
        num_pcs = 0 
        num_seq = 0
        ## Save path for current split
        sam_path = SAM_PATH + "/"+ split
        if not os.path.exists(sam_path):
            os.makedirs(sam_path)
        img_path = IMG_PATH+ "/"+ split
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        
        for seq in splits[split]:
            
            base_ts = BASE_TS_LS[seq][0] ## utc base timestamp, for lidar and robosense
            base_ts_local = BASE_TS_LS[seq][1] ## local base timestamp, for radar and pose
            
            ## some used paths
            pose_path = ROOT_PATH + seq + "/" + "gnssimu-sample-v6@2.csv"
            gt_path = ROOT_PATH + seq + "/" + "sync_gt/"
            data_path = ROOT_PATH + seq + "/" + "sync_radar/"
            
            # filter uncorrect groundtruth output
            if split == 'test':
                filt_path = ROOT_PATH + seq + "/" + "test_filt2/"
                filt_ls = sorted(os.listdir(filt_path))
                filt_idx = []
                for filt in filt_ls:
                    filt_idx.append(int(filt.split('.')[0]))
                
            ## extract the pose data
            ego_poses, pose_ts = get_interpolate_pose(pose_path,scale=1)
            radar_poses =  ego_poses @ ego_to_radar 
            pose_ts = (pose_ts-base_ts_local)/1e3
            dis = route_plot(ego_poses,seq)
            dis_all+=dis
            ## Getting radar raw data and gt data    
            pcs_ls = sorted(os.listdir(data_path))
            gts_ls = sorted(os.listdir(gt_path))
            pcs_len = len(pcs_ls)
    
            print('Starting Aggregating radar pcs for {}: seq-{}'.format(split,num_seq))    
            for i in tqdm(range(pcs_len-1)):
                
                pc_path1 = pcs_ls[i]
                pc_path2 = pcs_ls[i+1]
                gt_path1 = gts_ls[i]
                gt_path2 = gts_ls[i+1]
                ts1 = (int(pc_path1.split('.')[0])-base_ts)/1e3
                ts2 = (int(pc_path2.split('.')[0])-base_ts)/1e3
                
                pd_data1 = pd.read_table(data_path+pc_path1, sep=",", header=None)
                data1 = pd_data1.values[1:,1:].T.astype(np.float32)
                pd_data2 = pd.read_table(data_path+pc_path2, sep=",", header=None)
                data2 = pd_data2.values[1:,1:].T.astype(np.float32)
                gt1 = np.loadtxt(gt_path+gt_path1)
                gt2 = np.loadtxt(gt_path+gt_path2)
                
                target1= get_radar_target(data1,ts1,radar_poses,pose_ts)
                target2= get_radar_target(data2,ts2,radar_poses,pose_ts)
                
            
                if np.size(target1['car_loc_x'],0)>256 and np.size(target2['car_loc_x'],0)>256:
                    
                    ## obtain groundtruth for the test set (only use reliable robosense output)
                    if split == 'test':
                        if (i in filt_idx) and ((i+1) in filt_idx):
                            ## transformation from coordinate 1 to coordinate 2
                            tran = np.dot(np.linalg.inv(target1['pose']), target2['pose'])
                            ## obtain the scene flow labels from rigid transform and tracking object bounding boxes
                            labels, mask = get_flow_label(target1,tran,gt1,gt2,radar_to_ego,'test')
                            mask_show(mask,target1,num_pcs,img_path)
                            # show aligned two point clouds
                            align_show(target1,target2,tran,num_pcs,img_path)
                        else: 
                            continue
                    ## do not obtain groundtruth for train and val    
                    else:
                        tran = np.dot(np.linalg.inv(target1['pose']), target2['pose'])
                        #labels = np.array([])
                        #mask = np.array([])
                        labels, mask = get_flow_label(target1,tran,gt1,gt2,radar_to_ego,'train')
                        mask_show(mask,target1,num_pcs,img_path)
                        align_show(target1,target2,tran,num_pcs,img_path)
                        
                    num_pcs+=1
                    out_path_cur = sam_path + '/' + "radar_seqs-{}_samples-{}.json".format(num_seq, num_pcs)
                    for r_key in target1:
                          target1[r_key] = target1[r_key].tolist()
                          target2[r_key] = target2[r_key].tolist()
                         
                    sample = {"pc1": target1,
                              "pc2": target2,
                              "interval": ts2-ts1,
                              "trans": tran.tolist(),
                              "gt": labels.tolist(),
                              "mask": mask.tolist(),
                              }
                    ujson.dump(sample, open(out_path_cur, "w"))
                    
            num_seq+=1
    print(dis_all)          
    # avg_pcs = np.mean(num_targets)        
    # plt.figure()
    # plt.hist(num_targets, bins=50)
    # plt.show()
    
    # plt.figure()
    # plt.plot(timestamps)
    # plt.xlabel('frame')
    # plt.ylabel('timestamp')
    #plt.savefig('timestamps.png',dpi=500)
if __name__ == "__main__":
    main()
