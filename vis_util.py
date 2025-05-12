import os
import argparse
import sys
import torch
from tqdm import tqdm
import cv2
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.spatial.transform import Rotation as R
import matplotlib.ticker as ticker
import constants as cst
from vis_ops import flow_xy_to_colors


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]], 
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat

def caculate_align_mat(pVec_Arr):
    
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr/ (scale+1e-6)
    # must ensure pVec_Arr is also a unit vec. 
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,\
        z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat

def generate_arrow_mesh(begin,end,color):
    
    z_unit_Arr = np.array([0,0,1])
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height= 0.2 * vec_len + 1e-4, 
        cone_radius= 0.08 + 1e-4, 
        cylinder_height= 0.8 * vec_len + 1e-4, 
        cylinder_radius=  0.04 + 1e-4
        )
    mesh_arrow.paint_uniform_color(color)
    mesh_arrow.compute_vertex_normals()
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center = False)
    mesh_arrow.translate(np.array(begin))
    
    return mesh_arrow

def view_save_pc(pcd_1,pcd_2,sf_geo,gt_geo, frame,num_pcs,path):
    
    
    draw_gt = False
    save_view= False
    read_view= False
    torus = o3d.geometry.TriangleMesh.create_torus()
    sphere = o3d.geometry.TriangleMesh.create_sphere()
    # create shortcut for draw
    draw = o3d.visualization.EV.draw
    draw([ {'geometry': sphere, 'name': 'sphere'},
           {'geometry': torus, 'name': 'torus'} ])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=1280,height=720)
    # for sphere in pcd_1:
    #     vis.add_geometry(sphere)
    # for sphere in pcd_2:
    #     vis.add_geometry(sphere)
    # if not draw_gt:
    #     for arrow in sf_geo:
    #         vis.add_geometry(arrow)
    # else:
    #     for arrow in gt_geo:
    #         vis.add_geometry(arrow)
        
    # # vis.add_geometry(frame)
    # ctr = vis.get_view_control()
    # if read_view:
    #     param = o3d.io.read_pinhole_camera_parameters('{}.json'.format(3292))
    #     ctr.convert_from_pinhole_camera_parameters(param)
    # vis.run()
    # fname=path+'/'+'{}.png'.format(num_pcs)
    # vis.capture_screen_image(fname)
    # if save_view:
    #     param = ctr.convert_to_pinhole_camera_parameters()
    #     o3d.io.write_pinhole_camera_parameters('{}.json'.format(num_pcs), param)
    
def transform_to_ego(pc,T):
    
    pos = (np.matmul(T[0:3, 0:3], pc) + T[0:3,3:4])
    
    return pos

def get_matrix_from_ext(ext):
    
    N = np.size(ext,0)
    if ext.ndim==2:
        rot = R.from_euler('ZYX', ext[:,3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((N,4,4))
        tr[:,:3,:3] = rot_m
        tr[:,:3, 3] = ext[:,:3]
        tr[:, 3, 3] = 1
    if ext.ndim==1:
        rot = R.from_euler('ZYX', ext[3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((4,4))
        tr[:3,:3] = rot_m
        tr[:3, 3] = ext[:3]
        tr[ 3, 3] = 1
    return tr

def visulize_result_3D(pc1,pc2,sfs,wps,gts,num_pcs,path):
    
    npcs1=pc1.size()[2]
    npcs2=pc2.size()[2]
    
    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    sf=sfs[0].cpu().detach().numpy()
    wp=wps[0].cpu().detach().numpy()
    gt=gts[0].transpose(0,1).contiguous().cpu().detach().numpy()
    wp_gt=pc_1+gt
    
    radar_ext = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
    ego_to_radar = get_matrix_from_ext(radar_ext)
    pc_1 = transform_to_ego(pc_1,ego_to_radar)
    pc_2 = transform_to_ego(pc_2,ego_to_radar)
    wp = transform_to_ego(wp,ego_to_radar)
    wp_gt = transform_to_ego(wp_gt,ego_to_radar)
    
    ## source and target point cloud
    # pcd_1 = o3d.geometry.PointCloud()
    # pcd_1.points = o3d.utility.Vector3dVector(pc_1.T)
    # pcd_1.paint_uniform_color([0, 0, 1])
    # pcd_2 = o3d.geometry.PointCloud()
    # pcd_2.points = o3d.utility.Vector3dVector(pc_2.T)
    # pcd_2.paint_uniform_color([1, 0, 0])
    pcd_1 = []
    for i in range(npcs1):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0/255, 0/255, 255/255])
        mesh_sphere.translate(pc_1[:,i])
        pcd_1.append(mesh_sphere)
        
    pcd_2 = []
    for i in range(npcs2):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([255/255, 0/255, 255/255])
        mesh_sphere.translate(pc_2[:,i])
        pcd_2.append(mesh_sphere)

    sf_geo = []
    for i in range(npcs1):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0/255, 255/255, 0/255])
        mesh_sphere.translate(wp[:,i])
        sf_geo.append(mesh_sphere)
    gt_geo = []
    for i in range(npcs1):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0/255, 255/255, 0/255])
        mesh_sphere.translate(wp_gt[:,i])
        gt_geo.append(mesh_sphere)
        
    # sf_geo = []
    # for i in range(npcs1):
    #     begin = pc_1[:,i]
    #     end = wp[:,i]
    #     arrow = generate_arrow_mesh(begin,end,color=[0/255,200/255,0/255])
    #     sf_geo.append(arrow)    
    # gt_geo = []
    # for i in range(npcs1):
    #     begin = pc_1[:,i]
    #     end = wp_gt[:,i]
    #     arrow = generate_arrow_mesh(begin,end,color=[255/255,0,0])
    #     gt_geo.append(arrow)
    
    
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    
    ## view and control window for each frame
    view_save_pc(pcd_1,pcd_2,sf_geo,gt_geo, frame, num_pcs,path)

def view_save_seg(pcd_1,num_pcs,frame,path):
        
        
    read_view = False
    save_view = False
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920*2,height=1080*2)
    for sphere in pcd_1:
        vis.add_geometry(sphere)
    vis.add_geometry(frame)
    # vis.add_geometry(frame)
    ctr = vis.get_view_control()
    if read_view:
        param = o3d.io.read_pinhole_camera_parameters('{}.json'.format(num_pcs))
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    fname=path+'/'+'{}.png'.format(num_pcs)
    vis.capture_screen_image(fname)
    if save_view:
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('{}.json'.format(num_pcs), param)
        
        
def visulize_result_seg(pc1,pred_m,gt_m,num_pcs,path):
    
    npcs1=pc1.size()[2]
    
    pc1=pc1[0].cpu().numpy()
    pred_m=pred_m[0].cpu().detach().numpy()
    gt_m=gt_m[0].cpu().numpy()
    
    radar_ext = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
    ego_to_radar = get_matrix_from_ext(radar_ext)
    pc1 = transform_to_ego(pc1,ego_to_radar)
    
    pcd_1 = []
    for i in range(npcs1):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
        mesh_sphere.compute_vertex_normals()
        if pred_m[i]==1:
            mesh_sphere.paint_uniform_color([0/255,128/255,128/255])
        else:
            mesh_sphere.paint_uniform_color([255/255,20/255,147/255])
        mesh_sphere.translate(pc1[:,i])
        pcd_1.append(mesh_sphere)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    view_save_seg(pcd_1,num_pcs,frame,path)
    
    
def view_save_reg(pcd_1,pcd_2, warp, frame, num_pcs, path):
        
        
    read_view = False
    save_view = False
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920*2,height=1080*2)
    # for sphere in pcd_1:
    #     vis.add_geometry(sphere)
    for sphere in pcd_2:
        vis.add_geometry(sphere)
    for sphere in warp:
        vis.add_geometry(sphere)
    vis.add_geometry(frame)
    # vis.add_geometry(frame)
    ctr = vis.get_view_control()
    if read_view:
        param = o3d.io.read_pinhole_camera_parameters('{}.json'.format(num_pcs))
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    fname=path+'/'+'{}.png'.format(num_pcs)
    vis.capture_screen_image(fname)
    if save_view:
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('{}.json'.format(num_pcs), param)
        
        
def visualize_result_reg(pc1,pred_t,pc2,mask,num_pcs,path):
    
    npcs1=pc1.size()[2]
    npcs2=pc2.size()[2]
    
    pc1=pc1[0].cpu().numpy()
    pc2=pc2[0].cpu().numpy()
    pred_t = pred_t[0].cpu().detach().numpy()
    pc1_tran = transform_to_ego(pc1,pred_t)
    mask = mask[0].cpu().numpy()
    
    # if (mask==0).any():
    #     raise Exception('Exist moving point')
        
    radar_ext = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
    ego_to_radar = get_matrix_from_ext(radar_ext)
    pc1 = transform_to_ego(pc1,ego_to_radar)
    pc2 = transform_to_ego(pc2,ego_to_radar)
    pc1_tran = transform_to_ego(pc1_tran,ego_to_radar)
    
    pcd_1 = []
    for i in range(npcs1):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0/255, 0/255, 255/255])
        mesh_sphere.translate(pc1[:,i])
        pcd_1.append(mesh_sphere)
        
    pcd_2 = []
    for i in range(npcs2):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([255/255, 0/255, 255/255])
        mesh_sphere.translate(pc2[:,i])
        pcd_2.append(mesh_sphere)
    
    warp = []
    for i in range(npcs1):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0/255, 250/255, 154/255])
        mesh_sphere.translate(pc1_tran[:,i])
        warp.append(mesh_sphere)
    
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    
    view_save_reg(pcd_1,pcd_2, warp, frame, num_pcs, path)
    


def visualize_result_ft(pc1,pc2,ft1,ft2,num_pcs):
    
    
    pc1=pc1[0].cpu().numpy()
    ft1=ft1[0].cpu().numpy()
    pc2=pc2[0].cpu().numpy()
    ft2=ft2[0].cpu().numpy()

    # radar_ext = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
    # ego_to_radar = get_matrix_from_ext(radar_ext)
    # pc1 = transform_to_ego(pc1,ego_to_radar)
    # pc2 = transform_to_ego(pc2,ego_to_radar)
    
    x_img_1 = pc1[0]
    y_img_1 = -pc1[1]

    plt.figure()
    plt.winter()
    plt.scatter(x_img_1,y_img_1, s=1.5, c=ft1[2,:], marker='o')
    plt.xlim(0,70)
    plt.ylim(-35, 35)
    x_locator = MultipleLocator(40)
    y_locator = MultipleLocator(40)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)
    plt.tick_params(labelsize=16)
    #plt.xlabel('X [m]',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    #plt.ylabel('Y [m]',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    cb=plt.colorbar()
    cb.locator=ticker.MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=16)
    #cb.set_label('Z [m]',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    #plt.show()
    plt.savefig('power.png', dpi=600)
    print('show colorbar and figure')
    
    
def visualize_result_pos(pc1,pc2,num_pcs):

    npcs1=pc1.size()[2]
    npcs2=pc2.size()[2]
    
    pc1=pc1[0].cpu().numpy()
    pc2=pc2[0].cpu().numpy()
    
    fig=plt.figure()
    
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(pc1[0,:],-pc1[1,:],pc1[2,:], s=2.5, marker='o')
    x_locator = MultipleLocator(40)
    y_locator = MultipleLocator(40)
    z_locator = MultipleLocator(6)
    ax1.set_xlim(0,80)
    ax1.set_ylim(-50,50)
    ax1.tick_params(labelsize=16)
    ax1.xaxis.set_major_locator(x_locator)
    ax1.yaxis.set_major_locator(y_locator)
    ax1.zaxis.set_major_locator(z_locator)
    ax1.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax1.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax1.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax1.set_xlabel('X [m]',labelpad=15,fontsize=16)
    ax1.set_ylabel('Y [m]',labelpad=15,fontsize=16)
    ax1.set_zlabel('Z [m]',labelpad=10,fontsize=16)
    ax1.azim = -45
    ax1.dist = 10
    ax1.elev = 40
    ax1.grid(False)
    #plt.show()
    plt.savefig('pos.png', dpi=600)
    print('show')
    
def visulize_result_2D_pre(pc1, pc2, pred_f, pc1_warp, gt, num_pcs, mask, args):

    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    pred_f=pred_f[0].cpu().detach().numpy()
    pc1_warp=pc1_warp[0].cpu().detach().numpy()
    gt=gt[0].transpose(0,1).contiguous().cpu().detach().numpy()
    pc1_warp_gt=pc_1+gt
    error = np.linalg.norm(pc1_warp - pc1_warp_gt, axis = 0)
    mask = mask[0].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))
    #fig, ax = plt.subplots(1,2, figsize=(15,5))
    #fig, ax = plt.subplots(1,1, figsize = (10,5))

    x_locator = MultipleLocator(10)
    y_locator = MultipleLocator(10)

    x_flow, y_flow = pred_f[0], pred_f[1]
    rad = np.sqrt(np.square(x_flow) + np.square(y_flow))
    # x_gt, y_gt = gt[0], gt[1]
    # rad_gt = np.sqrt(np.square(x_gt) + np.square(y_gt))
    rad_max = np.max(rad)
    epsilon = 1e-5
    x_flow = x_flow / (rad_max + epsilon)
    y_flow = y_flow / (rad_max + epsilon)
 
    # x_gt = x_gt / (rad_max + epsilon)
    # y_gt = y_gt / (rad_max + epsilon)
    yy = np.linspace(-12.5, 12.5, 1000)
    yy1 = np.linspace(-10, 10, 1000)
    xx1 = np.sqrt(10**2-yy1**2)
    xx2 = np.sqrt(20**2-yy**2)
    xx3 = np.sqrt(30**2-yy**2)
    xx4 = np.sqrt(40**2-yy**2)
    xx5 = np.sqrt(50**2-yy**2)

    xx = np.linspace(0, 60, 1000)
    yy2 = np.zeros(xx.shape)
    yy3 = xx * np.tan(5*np.pi/180)
    yy4 = xx * np.tan(-5*np.pi/180)
    yy5 = xx * np.tan(10*np.pi/180)
    yy6 = xx * np.tan(-10*np.pi/180)
    yy7 = xx * np.tan(15*np.pi/180)
    yy8 = xx * np.tan(-15*np.pi/180)
    # theta = np.arctan(pc_1[1]/pc_1[0])
    # radius = np.sqrt(pc_1[1]**2 + pc_1[0]**2)
    # ax1 = plt.gca(projection='polar')
    # ax1 = ax[0]
    ax1 = plt.gca()
    # ax1.set_thetagrids(np.arange(-30, 30, 5.0))
    # ax1.set_thetamin(-20.0)  
    # ax1.set_thetamax(20.0)  
    # ax1.set_rgrids(np.arange(0, 60, 10))
    # ax1.set_rlabel_position(0.0)  
    # ax1.set_rlim(0.0, 5000.0)  
    # ax1.set_yticklabels(['0', '10', '20', '30', '40', '50'])
    # ax1.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    # ax1.set_axisbelow('True')
    colors = flow_xy_to_colors(x_flow, -y_flow)
    ax1.scatter(pc_1[0], pc_1[1], c = colors/255, marker='o', s=5)
    ax1.plot(xx1, yy1, linewidth=0.5, color='gray')
    ax1.plot(xx2, yy, linewidth=0.5, color='gray')
    ax1.plot(xx3, yy, linewidth=0.5, color='gray')
    ax1.plot(xx4, yy, linewidth=0.5, color='gray')
    ax1.plot(xx5, yy, linewidth=0.5, color='gray')
    ax1.plot(xx, yy2, linewidth=0.5, color='gray')
    ax1.plot(xx, yy3, linewidth=0.5, color='gray')
    ax1.plot(xx, yy4, linewidth=0.5, color='gray')
    ax1.plot(xx, yy5, linewidth=0.5, color='gray')
    ax1.plot(xx, yy6, linewidth=0.5, color='gray')
    ax1.plot(xx, yy7, linewidth=0.5, color='gray')
    ax1.plot(xx, yy8, linewidth=0.5, color='gray')
  
    ax1.text(10-0.55, -0.3, '10', fontsize=10, ma= 'center')
    ax1.text(20-0.55, -0.3, '20', fontsize=10, ma = 'center')
    ax1.text(30-0.55, -0.3, '30', fontsize=10, ma = 'center')
    ax1.text(40-0.55, -0.3, '40', fontsize=10, ma = 'center')
    ax1.text(50-0.55, -0.3, '50', fontsize=10, ma = 'center')

    ax1.set_xlim([0, 60])
    ax1.set_ylim([-15, 15])
    ax1.set_box_aspect(0.5)
    #ax1.title.set_text('Scene Flow')
    # ax1.xaxis.set_major_locator(x_locator)
    # ax1.yaxis.set_major_locator(y_locator)
    # ax1.tick_params(labelsize=10)
    [ax1.spines[loc_axis].set_visible(False) for loc_axis in ['top','right','bottom','left']]
    ax1.set_xticks([])
    ax1.set_yticks([])

    
    # ax2 = ax[1]
    # # rad_max = np.max(rad)
    # # colors = flow_xy_to_colors(x_gt, -y_gt)
    # ax2.scatter(pc_1[0], pc_1[1], c = error, marker='o', s=3, vmin = 0, vmax = 0.5, cmap='binary')
    # ax2.set_xlim([0, 60])
    # ax2.set_ylim([-12.5, 12.5])
    # ax2.set_box_aspect(5/12)
    # ax2.title.set_text('Error')
    # ax2.xaxis.set_major_locator(x_locator)
    # ax2.yaxis.set_major_locator(y_locator)
    # ax2.tick_params(labelsize=10)
    # cb=plt.colorbar()
    # cb.locator=ticker.MaxNLocator(nbins=8)
    # cb.update_ticks()
    # cb.ax.tick_params(labelsize=10)
    #ax.legend(loc='upper right')

    fig.tight_layout()
    path_im=args.vis_path_2D+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=200)
    fig.clf
    plt.cla
    plt.close('all')

def visulize_result_2D(pc1, pc2, pred_f, pc1_warp, gt, num_pcs, mask, args):
          
    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    pred_f=pred_f[0].cpu().detach().numpy()
    pc1_warp=pc1_warp[0].cpu().detach().numpy()
    gt=gt[0].transpose(0,1).contiguous().cpu().detach().numpy()
    pc1_warp_gt=pc_1+gt
    mask = mask[0].cpu().numpy()
    
    plt.figure()
    fig, ax = plt.subplots(2,2, figsize=(15,9))
    x_locator = MultipleLocator(10)
    y_locator = MultipleLocator(10)

    ax1 = ax[0,0]
    ax1.scatter(pc_1[0],pc_1[1], s=0.5, c='blue', marker=',', label='PC1')
    ax1.scatter(pc_2[0],pc_2[1], s=0.5, c='magenta', marker=',', label='PC2')
    ax1.set_xlim([0, 60])
    ax1.set_ylim([-15, 15])
    ax1.title.set_text('Input')
    ax1.xaxis.set_major_locator(x_locator)
    ax1.yaxis.set_major_locator(y_locator)
    ax1.tick_params(labelsize=10)
    ax1.legend(loc='upper right')

    ax2 = ax[0,1]
    ax2.scatter(pc1_warp[0],pc1_warp[1], s=0.5, c='green', marker=',', label='PC1_W')
    ax2.scatter(pc_2[0],pc_2[1], s=0.5, c='magenta', marker=',', label='PC2')
    ax2.set_xlim([0, 60])
    ax2.set_ylim([-15, 15])
    ax2.title.set_text('Prediction')
    ax2.xaxis.set_major_locator(x_locator)
    ax2.yaxis.set_major_locator(y_locator)
    ax2.tick_params(labelsize=10)
    ax2.legend(loc='upper right')

    ax3 = ax[1,0]
    ax3.scatter(pc1_warp_gt[0],pc1_warp_gt[1], s=0.5, c='green', marker=',', label='PC1_W_GT')
    ax3.scatter(pc_2[0],pc_2[1], s=0.5, c='magenta', marker=',', label='PC2')
    ax3.set_xlim([0, 60])
    ax3.set_ylim([-15, 15])
    
    ax3.title.set_text('GT')
    ax3.xaxis.set_major_locator(x_locator)
    ax3.yaxis.set_major_locator(y_locator)
    ax3.tick_params(labelsize=10)
    ax3.legend(loc='upper right')

    ax4 = ax[1,1]
    ax4.scatter(pc_1[0, mask==1], pc_1[1, mask==1], s=0.5, c='blue', marker=',', label='Stat PC1')
    ax4.scatter(pc_1[0, mask==0], pc_1[1, mask==0], s=0.5, c='orange', marker=',', label='Mov PC1')
    ax4.set_ylim([-15, 15])
    
    ax4.title.set_text('GT Mask')
    ax4.xaxis.set_major_locator(x_locator)
    ax4.yaxis.set_major_locator(y_locator)
    ax4.tick_params(labelsize=10)
    ax4.legend(loc='upper right')

    fig.tight_layout()
    path_im=args.vis_path_2D+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=200)
    fig.clf
    plt.cla
    plt.close('all')

def adjust_per_fig(ax):

    ax.set_xlim(0, 60)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-3, 3)
    ax.set_box_aspect([60,30,6])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.azim = -180
    ax.dist = 5
    ax.elev = 15
    ax.grid(False)

    return ax

def visulize_result_3D_plt(pc1, pc2, pred_f, pc1_warp, gt, num_pcs, mask, args):

    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    pred_f=pred_f[0].cpu().detach().numpy()
    pc1_warp=pc1_warp[0].cpu().detach().numpy()
    gt=gt[0].transpose(0,1).contiguous().cpu().detach().numpy()
    pc1_warp_gt=pc_1+gt
    mask = mask.cpu().numpy()[0]

    fig = plt.figure()
    ax1= plt.axes(projection='3d')
    #ax1 = fig.add_subplot(1, 3, 1, projection='3d')

    ax1.scatter3D(pc_1[0], pc_1[1], pc_1[2], c='b',marker='o', s=5, linewidth=0, alpha=1, cmap='spectral')
    ax1.scatter3D(pc_2[0], pc_2[1], pc_2[2], c='r',marker='o', s=5, linewidth=0, alpha=1, cmap='spectral')
    ax1 = adjust_per_fig(ax1)
    plt.axis('off')
    
    # ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # ax2.scatter(pc1_warp[0], pc1_warp[1], pc1_warp[2], c='b',marker='o', s=5, linewidth=0, alpha=1, cmap='spectral')
    # ax2.scatter(pc_2[0], pc_2[1], pc_2[2], c='r',marker='o', s=5, linewidth=0, alpha=1, cmap='spectral')
    # ax2 = adjust_per_fig(ax2)

    # ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    # ax3.scatter(pc1_warp_gt[0], pc1_warp_gt[1], pc1_warp_gt[2], c='b',marker='o', s=5, linewidth=0, alpha=1, cmap='spectral')
    # ax3.scatter(pc_2[0], pc_2[1], pc_2[2], c='r',marker='o', s=5, linewidth=0, alpha=1, cmap='spectral')
    # ax3 = adjust_per_fig(ax3)
    
    #fig.tight_layout()
    path_im=args.vis_path_3D+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=300)
    fig.clf
    plt.cla
    plt.close('all')

def visulize_result_2D_seg_pre(pc1, pc2, pred_m, num_pcs, args):


    npcs1=pc1.size()[2]
    npcs2=pc2.size()[2]
    
    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    pred_m = pred_m.cpu().numpy()[0]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    x_locator = MultipleLocator(10)
    y_locator = MultipleLocator(10)

    ax.scatter(pc_1[0, pred_m==0],-pc_1[1,pred_m==0], s=2, c='blue', marker=',', label='Moving')
    ax.scatter(pc_1[0, pred_m==1],-pc_1[1,pred_m==1], s=2, c='magenta', marker=',', label='Static')
    ax.set_xlim([0, 60])
    ax.set_ylim([-15, 15])
    ax.set_box_aspect(0.5)
    #ax1.title.set_text('Prediction')
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)
    ax.tick_params(labelsize=10)
    ax.legend(loc='upper right')

    
    #plt.xlabel('X [m]',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    #plt.ylabel('Y [m]',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    #cb=plt.colorbar()
    #cb.locator=ticker.MaxNLocator(nbins=4)
    #cb.update_ticks()
    #cb.ax.tick_params(labelsize=16)
    fig.tight_layout()
    path_im=args.vis_path_seg_pse+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=300)
    fig.clf
    plt.cla
    plt.close('all')

def visulize_result_2D_seg(pc1, pc2, mask, pred_m, num_pcs, args):
      
    npcs1=pc1.size()[2]
    npcs2=pc2.size()[2]
    
    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    mask = mask.cpu().numpy()[0]
    pred_m = pred_m.cpu().numpy()[0]
    
    plt.figure()
    fig, ax = plt.subplots(1,2, figsize=(15,4))
    x_locator = MultipleLocator(10)
    y_locator = MultipleLocator(10)

    ax1 = ax[0]
    ax1.scatter(pc_1[0, pred_m==0],-pc_1[1,pred_m==0], s=1, c='blue', marker=',', label='Moving')
    ax1.scatter(pc_1[0, pred_m==1],-pc_1[1,pred_m==1], s=0.5, c='magenta', marker=',', label='Static')
    ax1.set_xlim([0, 75])
    ax1.set_ylim([-20, 20])
    ax1.title.set_text('Prediction')
    ax1.xaxis.set_major_locator(x_locator)
    ax1.yaxis.set_major_locator(y_locator)
    ax1.tick_params(labelsize=10)
    ax1.legend(loc='upper right')

    ax2 = ax[1]
    ax2.scatter(pc_1[0, mask==0],-pc_1[1,mask==0], s=1, c='blue', marker=',', label='Moving')
    ax2.scatter(pc_1[0, mask==1],-pc_1[1,mask==1], s=0.5, c='magenta', marker=',', label='Static')
    ax2.set_xlim([0, 75])
    ax2.set_ylim([-20, 20])
    ax2.title.set_text('Ground truth')
    ax2.xaxis.set_major_locator(x_locator)
    ax2.yaxis.set_major_locator(y_locator)
    ax2.tick_params(labelsize=10)
    ax2.legend(loc='upper right')
    #plt.xlabel('X [m]',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    #plt.ylabel('Y [m]',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    #cb=plt.colorbar()
    #cb.locator=ticker.MaxNLocator(nbins=4)
    #cb.update_ticks()
    #cb.ax.tick_params(labelsize=16)
    fig.tight_layout()
    path_im=args.vis_path_seg_pse+'/'+'{}.png'.format(num_pcs)
    fig.savefig(path_im, dpi=300)
    fig.clf
    plt.cla
    plt.close('all')

def plot_epe_component(epe_xyz, args):

    epe_x = np.array(epe_xyz['x'])
    epe_y = np.array(epe_xyz['y'])
    epe_z = np.array(epe_xyz['z'])
    epe = np.sqrt(epe_x**2 + epe_y**2 + epe_z**2) 

    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(15,5))

    ax1 = ax[0,0]
    data_len = epe_x.shape[0]
    avg = np.mean(epe_x)
    std = np.std(epe_x)
    r1 = avg - std
    r2 = avg + std
    ax1.plot(epe_x, c='m',linestyle='-')
    ax1.plot([0, data_len], [avg, avg], c = 'b', label='Mean')
    ax1.fill_between(np.arange(1, data_len+1, 1), np.tile(r1,data_len), np.tile(r2,data_len), color = 'b', alpha=0.25)
    ax1.set_title('EPE_X (m)')
    ax1.set_ylim(0,1)
    ax1.legend(loc='upper right')

    ax2 = ax[0,1]
    avg = np.mean(epe_y)
    std = np.std(epe_y)
    r1 = avg - std
    r2 = avg + std
    ax2.plot(epe_y, c='m',linestyle='-')
    ax2.plot([0, data_len], [avg, avg], c = 'b', label='Mean')
    ax2.fill_between(np.arange(1, data_len+1, 1), np.tile(r1,data_len), np.tile(r2,data_len), color = 'b', alpha=0.25)
    ax2.set_title('EPE_Y (m)')
    ax2.set_ylim(0,1)
    ax2.legend(loc='upper right')

    ax3 = ax[1,0]
    avg = np.mean(epe_z)
    std = np.std(epe_z)
    r1 = avg - std
    r2 = avg + std
    ax3.plot(epe_z, c='m',linestyle='-')
    ax3.plot([0, data_len], [avg, avg], c = 'b', label='Mean')
    ax3.fill_between(np.arange(1, data_len+1, 1), np.tile(r1,data_len), np.tile(r2,data_len), color = 'b', alpha=0.25)
    ax3.set_title('EPE_Z (m)')
    ax3.set_ylim(0,0.1)
    ax3.legend(loc='upper right')

    ax4 = ax[1,1]
    avg = np.mean(epe)
    std = np.std(epe)
    r1 = avg - std
    r2 = avg + std
    ax4.plot(epe, c='m',linestyle='-')
    ax4.plot([0, data_len], [avg, avg], c = 'b', label='Mean')
    ax4.fill_between(np.arange(1, data_len+1, 1), np.tile(r1,data_len), np.tile(r2,data_len), color = 'b', alpha=0.25)
    ax4.set_title('EPE (m)')
    ax4.set_ylim(0,1)
    ax4.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig('checkpoints/' + args.exp_name + '/' + 'epe_xyz.png', dpi=500)
    print('----save epe components figure----')

def plot_trajectory(gt_trans,pre_trans,args):
    
    gt_trans = gt_trans.cpu().numpy()
    pre_trans = pre_trans.cpu().detach().numpy()
    
    save_path = 'checkpoints/' + args.exp_name + '/' + 'trajectory/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for clip_info in args.clips_info:
        clip_name = clip_info['clip_name']
        clip_idxs = clip_info['index']
        gt = gt_trans[clip_idxs[0]:clip_idxs[1]]
        pre = pre_trans[clip_idxs[0]:clip_idxs[1]]
        plot_one_clip_trajectory(gt, pre, clip_name, save_path, args)


def plot_one_clip_trajectory(gt, pre, clip_name, save_path, args):

    # calculate accumulated transformation matrix along the sequence
    num_frames = gt.shape[0]
    gt_a = np.zeros((num_frames,4,4))
    pre_a = np.zeros((num_frames,4,4))
    gt_a[0] = np.linalg.inv(gt[0])
    pre_a[0] = np.linalg.inv(pre[0])
    for i in range(1, num_frames):
        gt_a[i] = gt_a[i-1] @ np.linalg.inv(gt[i])
        pre_a[i] = pre_a[i-1] @ np.linalg.inv(pre[i])
    # calculate ego-vehicle coordinates w.r.t. the initial coordinates frame
    p_gt = np.zeros((num_frames+1,3))
    p_pre = np.zeros((num_frames+1,3))
    for i in range(num_frames):
        p_gt[i+1] = gt_a[i][:3,3]
        p_pre[i+1] = pre_a[i][:3,3]
    x_gt, y_gt, z_gt = p_gt[:,0], p_gt[:,1], p_gt[:,2]
    x_pre, y_pre, z_pre = p_pre[:,0], p_pre[:,1], p_pre[:,2]

    # new a figure and set it into 3d
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_locator = MultipleLocator(50)
    y_locator = MultipleLocator(50)
    z_locator = MultipleLocator(2)
    #ax.set_xlim(-80,80)
    #ax.set_ylim(-50,50)
    ax.tick_params(labelsize=5)
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)
    ax.zaxis.set_major_locator(z_locator)
    
    # set figure information
    ax.set_title("Trajectory Result")
    ax.set_xlabel('X [m]',labelpad=5,fontsize=6)
    ax.set_ylabel('Y [m]',labelpad=5,fontsize=6)
    ax.set_zlabel('Z [m]',labelpad=0,fontsize=6)
    
    # draw the figure
    ax.plot(x_gt, y_gt, z_gt, c='r',linestyle='-')
    ax.plot(x_pre, y_pre, z_pre, c='b',linestyle='-')
    ax.legend(['GT','Pre'])
    ax.azim = -40
    ax.dist = 15
    ax.elev = 40
    ax.grid(False)

    plt.savefig(save_path + clip_name + '.png', dpi=500)
    print('----save trajectory figure for {}----'.format(clip_name))
 

def get_euler_angle(trans):

    num_frames = trans.shape[0]
    angles = np.zeros((num_frames,3))
    for i in range(trans.shape[0]):
        r = R.from_matrix(trans[i,:3,:3])
        angles[i] = r.as_euler('XYZ') * 180/(2*np.pi)

    return angles


def plot_comparison(gt_trans,pre_trans,args):

    num_frames = gt_trans.shape[0]
    gt_trans = gt_trans.cpu().numpy()
    pre_trans = pre_trans.cpu().detach().numpy()

    save_path = 'checkpoints/' + args.exp_name + '/' + 'ego_motion_plt/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for clip_info in args.clips_info:
        clip_name = clip_info['clip_name']
        clip_idxs = clip_info['index']
        gt = gt_trans[clip_idxs[0]:clip_idxs[1]]
        pre = pre_trans[clip_idxs[0]:clip_idxs[1]]
        plot_one_clip_comparison(gt, pre, clip_name, save_path, args)


def plot_one_clip_comparison(gt, pre, clip_name, save_path, args):

    num_frames = gt.shape[0]
    for i in range(num_frames):
        gt[i] = np.linalg.inv(gt[i])
        pre[i] = np.linalg.inv(pre[i])
    delta_x_gt, delta_y_gt, delta_z_gt = gt[:,0,3], gt[:,1,3], gt[:,2,3]
    delta_x_pre, delta_y_pre, delta_z_pre = pre[:,0,3], pre[:,1,3], pre[:,2,3]
    delta_euler_gt = get_euler_angle(gt)
    delta_euler_pre = get_euler_angle(pre)

    fig, ax = plt.subplots(2, 3, figsize=(15,5))

    ax1 = ax[0,0]
    ax1.plot(delta_x_gt, c='r',linestyle='-')
    ax1.plot(delta_x_pre,c='b',linestyle='-')
    ax1.set_title('$\Delta$X(m)')
    ax1.set_ylim(-2,2)

    ax2 = ax[0,1]
    ax2.plot(delta_y_gt, c='r',linestyle='-')
    ax2.plot(delta_y_pre,c='b',linestyle='-')
    ax2.set_title('$\Delta$Y(m)')
    ax2.set_ylim(-2,2)

    ax3 = ax[0,2]
    ax3.plot(delta_z_gt, c='r',linestyle='-')
    ax3.plot(delta_z_pre,c='b',linestyle='-')
    ax3.set_title('$\Delta$Z(m)')
    ax3.set_ylim(-2,2)

    ax4 = ax[1,0]
    ax4.plot(delta_euler_gt[:,0], c='r',linestyle='-')
    ax4.plot(delta_euler_pre[:,0],c='b',linestyle='-')
    ax4.set_title('$\Delta$'+ u'\u03B1'+'($^\circ$)')
    ax4.set_ylim(-2,2)

    ax5 = ax[1,1]
    ax5.plot(delta_euler_gt[:,1], c='r',linestyle='-')
    ax5.plot(delta_euler_pre[:,1],c='b',linestyle='-')
    ax5.set_title('$\Delta$'+ u'\u03B2'+'($^\circ$)')
    ax5.set_ylim(-2,2)

    ax6 = ax[1,2]
    ax6.plot(delta_euler_gt[:,2], c='r',linestyle='-')
    ax6.plot(delta_euler_pre[:,2],c='b',linestyle='-')
    ax6.set_title('$\Delta$'+ u'\u03B3'+'($^\circ$)')
    ax6.set_ylim(-2,2)

    fig.tight_layout()
    plt.savefig(save_path +  clip_name + '.png', dpi=500)
    print('----save pose comparison figure----')
