import os
import json
import ujson
import numpy as np
import copy
from time import *
from pathos.multiprocessing import ProcessingPool as Pool
import sys
import shutil
import pandas as pd


if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from tqdm import tqdm
from radar_mask import Static_Seg

# files operation upon raw files provided from the simulation code
# path = '/home/toytiny/carla_radar/'
# data_path=os.listdir(path)
# for d in data_path:
#     real_path=path+d+'/exp33/process_slow/'
#     comp=os.listdir(real_path)
#     for c in comp:
#         shutil.move(real_path+c,path+d+'/')
#     shutil.rmtree(path+d+'/exp33')

DATA_PATH = "/home/toytiny/carla_radar/raw/"
OUT_PATH_PC = "/home/toytiny/carla_radar/" + "radar_pcs/"
IMG_PATH = "/home/toytiny/carla_radar/" + "radar_img/"

SIDE_RANGE = (-50, 50)
FWD_RANGE = (0, 100)
HEIGHT_RANGE = (-2.75, 3.25)
RES = 0.15625
USE_MP = False
import numpy as np


def inside_test(points, cube3d):
    """
    cube3d  =  array of the shape (8,3) with coordinates
    points = array of points with shape (N, 3).
    Returns the indices of the points array which are inside the cube3d
    """
    N = points.shape[0]
    ## clock-wise, bottom first and then top
    b1, t1, b4, t4, b2, t2, b3, t3 = cube3d

    dir1 = t1 - b1 +1e-5
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / (size1 + 1e-5)

    dir2 = b2 - b1 + 1e-5
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / (size2 + 1e-5)

    dir3 = b4 - b1 + 1e-5
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / (size3 + 1e-5)
    

    cube3d_center = (b1 + t3) / 2.0

    dir_vec = points - cube3d_center

    res1 = np.where((np.absolute(np.dot(dir_vec, dir1)) * 2) > size1)[0]
    res2 = np.where((np.absolute(np.dot(dir_vec, dir2)) * 2) > size2)[0]
    res3 = np.where((np.absolute(np.dot(dir_vec, dir3)) * 2) > size3)[0]

    out_idx = list(set().union(res1, res2, res3))
    idx = np.arange(0, N, 1)
    in_idx = np.delete(idx, out_idx, axis=0)
    return in_idx


def get_radar_target(data, gt_1, gt_2, srd_1, srd_2, segmenter):

    ## use the original left-hand coordinate system in Carla, front is x, right is y
    x_points = data[0, :]
    y_points = data[1, :]
    z_points = data[2, :]
    vel_r = data[6, :]/2

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

    # SHIFT to the BEV view

    x_img = np.floor((x_points) / RES)
    y_img = np.floor((y_points - SIDE_RANGE[0]) / RES)

    bev_vel_r = vel_r / RES

    bev_vel_r[np.isnan(bev_vel_r)] = 0

    vel_r = bev_vel_r * RES
    
    ## Segment static points mask
    mask, _, _ = segmenter.ransac_carla(points,vel_r)
    nmotive = N-np.sum(mask)
    
    ## Transform gt to odometry between current scan to the next scan
    trans = np.dot(np.linalg.inv(gt_1), gt_2)

    ## Get the ground truth for scene flow
    sf = np.zeros((N, 3))
    srd1 = srd_1[1:, 1:].astype(np.float32)
    srd2 = srd_2[1:, 1:].astype(np.float32)
    
    ## transform to the right-hand certasian coordinate, the same as nuscenes
    #for i in range(8):
    #    srd1[:, 3 * (i + 1) - 1] = -srd1[:, 3 * (i + 1) - 1]
    #    srd2[:, 3 * (i + 1) - 1] = -srd2[:, 3 * (i + 1) - 1]

    nacts_1 = srd1.shape[0]
    nacts_2 = srd2.shape[0]

    ncors = 0
    for i in range(nacts_1):
        cur_id = srd1[i, 0]
        next_idx = np.where(srd2[:, 0] == cur_id)[0]
        if len(next_idx) == 0:
            continue
        else:
            cube1 = np.reshape(srd1[i, 1:-3], (8, 3))
            cube2 = np.reshape(srd2[next_idx[0], 1:-3], (8, 3))
            cube1_c = (cube1[0] + cube1[7]) / 2.0
            cube2_c = (cube2[0] + cube2[7]) / 2.0
            cube_sf = cube2_c - cube1_c
            cor_idx = inside_test(points, cube1)
            sf[cor_idx, :] = cube_sf
            ncors += len(cor_idx)
    
    bg_idx = np.where(sf[:, 0] == 0)[0]
  
    if ncors < N:
        h_points = np.hstack((points[bg_idx], np.ones((len(bg_idx), 1))))
        sf[bg_idx] = np.dot(np.linalg.inv(trans), h_points.T)[:3].T - points[bg_idx]

    targets = {
        "car_loc_x": x_points,
        "car_loc_y": y_points,
        "car_loc_z": z_points,
        "car_vel_r": vel_r,
        "bev_loc_x": x_img,
        "bev_loc_y": y_img,
        "bev_vel_r": bev_vel_r,
        "transform": trans,
        "sf": sf,
        "mask": mask,
    }
    
    sf_r = np.sum(sf*points,1)/(np.linalg.norm(points,axis=1))
    sf_vel = vel_r * 0.1
    return targets


def BEV_show(radar, num_pcs, img_path):

    x_img = np.array(radar["bev_loc_x"]).astype(np.int32)
    y_img = np.array(radar["bev_loc_y"]).astype(np.int32)
    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] += 255
    path = img_path + "/" + "{}.jpg".format(num_pcs)
    cv2.imwrite(path, im)

def process_one_sample(sc,gt1,gt2,srd1, srd2, pc_path, gt_path,srd_path,\
                       out_path_pc,out_path_im,num_pcs,num_scenes):
    

    ## Read radar data, odometry and surroundings info
    pd_data = pd.read_table(pc_path + sc, sep=",", header=None)
    gt_1 = pd.read_table(gt_path + gt1, sep=",", header=None)
    gt_2 = pd.read_table(gt_path + gt2, sep=",", header=None)
    srd_1 = pd.read_table(srd_path + srd1, sep=",", header=None)
    srd_2 = pd.read_table(srd_path + srd2, sep=",", header=None)
    data = pd_data.values.T
    gt_1 = gt_1.values
    gt_2 = gt_2.values
    srd_1 = srd_1.values
    srd_2 = srd_2.values

    ## Process the raw data to radar target package
    ## Initial static points segment
    segmenter = Static_Seg(threshold=0.05, max_iter=20)
    cur_target = get_radar_target(data, gt_1, gt_2, srd_1, srd_2, segmenter)
    for r_key in cur_target:
        if type(cur_target[r_key])==list:
            print('aaaa')
        cur_target[r_key] = cur_target[r_key].tolist()

    BEV_show(cur_target, num_pcs, out_path_im)
    out_path_cur = (
        out_path_pc
        + "/"
        + "radar_scenes-{}_pcs-{}.json".format(num_scenes, num_pcs)
    )
    ujson.dump(cur_target, open(out_path_cur, "w"))
    
def main():

 
    # inside_test(points , cube)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(OUT_PATH_PC):
        os.makedirs(OUT_PATH_PC)
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    out_ls = sorted(os.listdir(DATA_PATH), key=lambda x: eval(x.split("_")[-1]))

    ## Split all data to train and val set
    # splits = {"train": ['_out_1','_out_5','_out_7','_out_8','_out_10','_out_11','_out_12',
    #                      '_out_14','_out_16','_out_17','_out_18','_out_22','_out_23'], 
    #            "val": ['_out_2','_out_4','_out_9','_out_13','_out_21'], 
    #            "test": ['_out_3','_out_6','_out_15','_out_19','_out_20'],
    #            }
    splits = {#"train": ['_out_20'], 
              # "val": ['_out_18'], 
               "test": ['_out_19'],
                 }

    for split in splits:

        out_path_pc = OUT_PATH_PC + split
        out_path_im = IMG_PATH + split
        if not os.path.exists(out_path_pc):
            os.makedirs(out_path_pc)
        if not os.path.exists(out_path_im):
            os.makedirs(out_path_im)

        num_scenes = 0
        num_pcs = 0
        num_secs = 0
        
        
        for out in splits[split]:

            pc_path = DATA_PATH + out + "/pcl/"
            gt_path = DATA_PATH + out + "/gt/"
            srd_path = DATA_PATH + out + "/srd/"
            scan_ls = sorted(
                os.listdir(pc_path), key=lambda x: eval(x.split("_")[-1].split(".")[0])
            )
            gt_ls = sorted(
                os.listdir(gt_path), key=lambda x: eval(x.split("_")[-1].split(".")[0])
            )
            srd_ls = sorted(
                os.listdir(srd_path), key=lambda x: eval(x.split("_")[-1].split(".")[0])
            )
            scan_ls = scan_ls[1:-1]
            gt1_ls = gt_ls[1:-1]
            gt2_ls = gt_ls[2:]
            srd1_ls = srd_ls[1:-1]
            srd2_ls = srd_ls[2:]
            if len(scan_ls) > 0:
                num_scenes += 1
                num_secs+=1
            else:
                continue

            print(
                "Aggregating data for section {} of the {} set".format(
                    num_scenes, split
                )
            )
            
            pool = Pool(8) 
            zip_pc_path = [pc_path for i in range(len(scan_ls))]
            zip_gt_path = [gt_path for i in range(len(scan_ls)) ]
            zip_srd_path = [srd_path for i in range(len(scan_ls))]
            zip_out_pc = [out_path_pc for i in range(len(scan_ls)) ]
            zip_out_im = [out_path_im for i in range(len(scan_ls)) ]
            pc_idx = np.arange(num_pcs,num_pcs+len(scan_ls)).tolist()
            scene_idx = [num_scenes for i in range(len(scan_ls)) ]
            with tqdm(total=len(scan_ls)) as t:
                if USE_MP:
                    for i in enumerate(pool.imap(process_one_sample,scan_ls,gt1_ls,\
                                gt2_ls,srd1_ls,srd2_ls,zip_pc_path, zip_gt_path, zip_srd_path,\
                                zip_out_pc,zip_out_im,pc_idx,scene_idx)):
                        t.update()
 
        
                else:
                    for i in range(len(scan_ls)):
                        process_one_sample(scan_ls[i],gt1_ls[i],gt2_ls[i],srd1_ls[i],srd2_ls[i],\
                                    zip_pc_path[i], zip_gt_path[i], zip_srd_path[i],\
                                    zip_out_pc[i],zip_out_im[i],pc_idx[i],scene_idx[i])
                        t.update()
                        
            num_pcs+=len(scan_ls)
                

            print(
                "Finish data aggregating for section {} of the {} set".format(
                    num_secs, split
                )
            )


if __name__ == "__main__":
    main()
