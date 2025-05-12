import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append("..")
from tqdm import tqdm
import time
import copy
import yaml
import torch 
import random
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import torch.nn.functional as F
from collections import OrderedDict
sys.path.append("..")
#from model.FLOT import FLOT
from utils.util import load_oss_pcs, load_oss_labels, load_calib, load_lidar_poses



def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)

def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args:
        pose_path: (Complete) filename for the pose file
        Returns:
        A numpy array of size nx4x4 with n poses as 4x4 transformation
        matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']
    
    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')
    return np.array(poses)


def load_files(folder):
    """Load all files in a folder and sort."""
    root = os.path.realpath(os.path.expanduser(folder))
    all_paths = sorted(os.listdir(root))
    all_paths = [os.path.join(root, path) for path in all_paths]
    return all_paths


def load_oss_pcs(seq, folder):
    """Load all files in a folder and sort."""
    seq_num = [4541,1101,4661,801,271,2761,1101,1101,4071,1591,1201]
    num = seq_num[seq]
    all_paths = list(range(num))
    all_paths = [os.path.join(folder, "{0:06d}".format(path)+".bin") for path in all_paths]
    return all_paths


def load_oss_labels(seq, folder):
    """Load all files in a folder and sort."""
    seq_num = [4541,1101,4661,801,271,2761,1101,1101,4071,1591,1201]
    num = seq_num[seq]
    all_paths = list(range(num))
    all_paths = [os.path.join(folder, "{0:06d}".format(path)+".label") for path in all_paths]
    return all_paths

def load_lidar_poses(pose_path, T_cam_velo):
    """ load poses in lidar coordinate system """
    # load poses in camera system
    print(pose_path)
    poses = np.array(load_poses(pose_path))
    T_cam0_w = np.linalg.inv(poses[0])

    # load calibrations: camera0 to velodyne
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)
    # convert poses in LiDAR coordinate system
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(T_cam0_w).dot(pose).dot(T_cam_velo))
    new_poses = np.array(new_poses)
    poses = new_poses

    return poses

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()
    
def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def save_scene_flow(f_id, motionflow_xyz, motionflow_dir=None):
    if not os.path.exists(motionflow_dir):
        os.makedirs(motionflow_dir)
    frame_path = os.path.join(motionflow_dir, str(f_id).zfill(6) + '.ego')
    np.save(frame_path, motionflow_xyz)


def save_pred_index(f_id, motionflow_xyz, motionflow_dir=None):
    if not os.path.exists(motionflow_dir):
        os.makedirs(motionflow_dir)
    frame_path = os.path.join(motionflow_dir, str(f_id).zfill(6) + '.index')
    np.save(frame_path, motionflow_xyz)

def save_pred_ego_motion(config):
    seq = str(config['seq']).zfill(2)
    dataset_root = config['dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    label_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_labels')
    correspondence_root = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'correspondence_gt')
    ego_pred_folder = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'ego_pred')
    ego_index_folder = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'ego_index')
    file_paths = load_oss_pcs(int(seq), scan_folder)
    label_paths = load_oss_labels(int(seq), label_folder)
    correspondences = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(correspondence_root)) for f in fn]
    correspondences.sort()
    if yaml.__version__ >= '5.1':
        config_ = yaml.load(open(config["moving_learning_map"]), Loader=yaml.FullLoader)
    else:
        config_ = yaml.load(open(config["moving_learning_map"]))
    # init model
    model = FLOT().cuda()
    pretrained_dict = torch.load(config["pretrain"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    with torch.no_grad():
        for i in range(len(file_paths)):
            print(i)
            pc1 = np.fromfile(file_paths[i], dtype=np.float32)
            pc1 = pc1.reshape(-1, 4)[:,:3]

            loc1 = np.logical_and(abs(pc1[:,0]) < 50, abs(pc1[:,1]) < 50)
            pc1 = pc1[loc1, :]
            correspondence = np.load(correspondences[i])
            correspondence = correspondence[loc1]
            if i == 0:
                ego_motion_first = np.zeros((16384, 3))
                ego_index = np.zeros((16384))
                save_scene_flow(i, ego_motion_first, ego_pred_folder)
                save_pred_index(i, ego_index, ego_index_folder)
                continue
            # label_path = label_paths[i]
            # labels = np.fromfile(label_path, dtype=np.uint32)
            # labels = labels.reshape((-1))
            # semantic_labels = labels & 0xFFFF
            # instance_labels = labels >> 16
            # mos_labels = copy.deepcopy(semantic_labels)
            # for k, v in config_["moving_learning_map"].items():
            #     mos_labels[semantic_labels == k] = v
            # mos_labels = mos_labels[loc1]
            # moving_indices = np.where(mos_labels > 0)[0]
            # static_indices = np.where(mos_labels == 0)[0]
            # assert (len(moving_indices) + len(static_indices)) == len(mos_labels)



            # sample as many points as possible
            # if pc1.shape[0] > config["num_points"]:
            #     sample_idx1 = np.random.choice(pc1.shape[0], config["num_points"], replace=False)
            #     moving_sample = []
            #     for i in sample_idx1:
            #         if i in moving_indices:
            #             moving_sample.append(i)
            #     moving_pc = pc1[moving_sample, :]
            #     pc1 = pc1[sample_idx1, :]
            #     print("moving points: ", moving_pc.shape[0], " and pc: ", pc1.shape[0])




            # if moving_indices.shape[0] > 8192:
            #     moving_sample_index = np.random.choice(moving_indices.shape[0], 8192, replace=False)
            #     static_sample_index = np.random.choice(static_indices.shape[0], 8192, replace=False)
            #     moving_indices = moving_indices[moving_sample_index]
            #     static_indices = static_indices[static_sample_index]
            # else:
            #     if static_indices.shape[0] < 16384 - moving_indices.shape[0]:
            #         static_sample_index = np.random.choice(static_indices.shape[0], 16384 - moving_indices.shape[0], replace=True)
            #     else:
            #         static_sample_index = np.random.choice(static_indices.shape[0], 16384 - moving_indices.shape[0], replace=False)
            #     static_indices = static_indices[static_sample_index]
            # ego_index = np.concatenate((moving_indices, static_indices))
            # moving_pc = pc1[moving_indices, :]
            # pc1 = pc1[ego_index, :]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc1)
            data = pcd.voxel_down_sample_and_trace(0.3, pcd.get_min_bound(), pcd.get_max_bound())
            _, indices, _ = data
            indices = np.max(indices, axis=1)
            if indices.shape[0] > 16384:
                sample_index = np.random.choice(indices.shape[0], 16384, replace=False)
            else:
                sample_index = np.concatenate((np.arange(indices.shape[0]), np.random.choice(indices.shape[0], 16384 - indices.shape[0], replace=True)),axis=-1)
            sample_indices = indices[sample_index]
            assert len(sample_indices) == 16384
            pc1 = pc1[sample_indices, :]
            pc2 = np.fromfile(file_paths[i-1], dtype=np.float32)
            pc2 = pc2.reshape(-1, 4)[:,:3]
            pc2 = pc2[correspondence[sample_indices], :]

            pc1 = torch.tensor(pc1, dtype=torch.float).unsqueeze(0)
            pc2 = torch.tensor(pc2, dtype=torch.float).unsqueeze(0)

            ego_motion = model(pc1.cuda(), pc2.cuda())
            ego_motion = ego_motion.squeeze().detach().cpu().numpy()
            save_scene_flow(i, ego_motion, ego_pred_folder)
            save_pred_index(i, sample_index, ego_index_folder)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pc1)
            # color = [0.5, 0.5, 0.5]
            # color = np.tile(color, (len(pc1), 1))
            # pcd.colors = o3d.utility.Vector3dVector(color)

            # pcd2 = o3d.geometry.PointCloud()
            # pcd2.points = o3d.utility.Vector3dVector(moving_pc)
            # color2 = [0, 1, 0]
            # color2 = np.tile(color2, (len(moving_pc), 1))
            # pcd2.colors = o3d.utility.Vector3dVector(color2)
            # o3d.visualization.draw_geometries([pcd, pcd2])


def visual_scene_flow_from_origin_flot(config):
    seq = str(config['seq']).zfill(2)
    dataset_root = config['dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    gt_folder = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'scene_flow_gt')
    correspondence_root = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'correspondence_gt')
    file_paths = load_oss_pcs(int(seq), scan_folder)
    label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(gt_folder)) for f in fn]
    label_paths.sort()
    correspondences = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(correspondence_root)) for f in fn]
    correspondences.sort()
    # init model
    # model = FLOT()
    # pretrained_dict = torch.load(config["pretrain"])
    # new_state_dict = OrderedDict()
    # for k,v in pretrained_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    # model.cuda()
    # model.eval()
    for i in range(1,len(file_paths)):
        # if i > 3:
        #     break
        if i != 43 and i != 85 and i != 3245:
        # if i != 43 and i != 4019 and i != 1644 and i != 85 and i != 3245:
            continue
        pc1 = np.fromfile(file_paths[i], dtype=np.float32)
        pc1 = pc1.reshape(-1, 4)[:,:3]
        gt_flow = np.load(label_paths[i])
        correspondence = np.load(correspondences[i])
        pc2 = np.fromfile(file_paths[i-1], dtype=np.float32)
        pc2 = pc2.reshape(-1, 4)[:,:3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc1)
        data = pcd.voxel_down_sample_and_trace(0.3, pcd.get_min_bound(), pcd.get_max_bound(), False)
        color = [0.7,0.7,0.7]
        color = np.tile(color, (len(pc1), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
        color = [1,0,0]
        color = np.tile(color, (len(data[0].points), 1))
        data[0].colors = o3d.utility.Vector3dVector(color)
        # o3d.visualization.draw_geometries([data[0]])






        # loc1 = np.logical_and(abs(pc1[:,0]) < 35, abs(pc1[:,1]) < 35)
        # pc1 = pc1[loc1, :]
        # correspondence = np.load(correspondences[i])
        # correspondence = correspondence[loc1]
        # n1 = pc1.shape[0]
        # if n1 > config["num_points"]:
        #     sample_idx1 = np.random.choice(n1, config["num_points"], replace=False)
        # else:
        #     sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, config["num_points"] - n1, replace=True)), axis=-1)

        # source = pc1[sample_idx1,:]
        # gt_flow = gt_flow[sample_idx1,:]
        # source2 = pc2[correspondence[sample_idx1],:]

        # pc1 = torch.tensor(source, dtype=torch.float).unsqueeze(0)
        # pc2 = torch.tensor(source2, dtype=torch.float).unsqueeze(0)

        # scene_flow = model(pc1.cuda(), pc2.cuda())
        # scene_flow = scene_flow.squeeze().detach().cpu().numpy()
        # pred_target = source - scene_flow
        # gt_target = source - gt_flow
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(source)
        # color = [1,0,0]
        # color = np.tile(color, (len(source), 1))
        # pcd.colors = o3d.utility.Vector3dVector(color)

        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(source2)
        # color2 = [0.5,0.5,0.5]
        # color2 = np.tile(color2, (len(source2), 1))
        # pcd2.colors = o3d.utility.Vector3dVector(color2)

        # corrds = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        # points_pred = np.concatenate((source, pred_target), axis=0)
        # points_gt = np.concatenate((source, gt_target), axis=0)
        # lines = np.arange(source.shape[0] * 2).reshape(-1, 2, order='F')
        # # blue is pred
        # color_pred = np.tile([0,0,1], (len(lines), 1))
        # # green is gt
        # color_gt = np.tile([0,1,0], (len(lines), 1))
        # line_set_pred = o3d.geometry.LineSet()
        # line_set_pred.points = o3d.utility.Vector3dVector(points_pred) # shape: (num_points, 3)
        # line_set_pred.lines = o3d.utility.Vector2iVector(lines)   # shape: (num_lines, 2)
        # line_set_pred.colors = o3d.utility.Vector3dVector(color_pred) # shape: (num_lines, 3)

        # line_set_gt = o3d.geometry.LineSet()
        # line_set_gt.points = o3d.utility.Vector3dVector(points_gt) # shape: (num_points, 3)
        # line_set_gt.lines = o3d.utility.Vector2iVector(lines)   # shape: (num_lines, 2)
        # line_set_gt.colors = o3d.utility.Vector3dVector(color_gt) # shape: (num_lines, 3)

        # o3d.visualization.draw_geometries([pcd, pcd2, line_set_pred])
        # # o3d.visualization.draw_geometries([pcd, pcd2, line_set_pred, line_set_gt])


def visual_scene_flow_from_our_net(config):
    seq = str(config['seq']).zfill(2)
    dataset_root = config['kitti_dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    gt_folder = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'scene_flow_gt')
    correspondence_root = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'correspondence_gt')
    file_paths = load_oss_pcs(int(seq), scan_folder)
    label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(gt_folder)) for f in fn]
    label_paths.sort()
    correspondences = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(correspondence_root)) for f in fn]
    correspondences.sort()
    calib_file = os.path.join(dataset_root, 'sequences', seq, "calib.txt")
    calib = load_calib(calib_file)
    poses_file = os.path.join(dataset_root, 'sequences', seq, "ICP_POSES.txt")
    poses = load_lidar_poses(poses_file, calib)


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    to_reset = False
    # 读取viewpoint参数
    vis.get_render_option().load_from_json('RenderOption.json')
    param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
    ctr = vis.get_view_control()
    # 转换视角
    ctr.convert_from_pinhole_camera_parameters(param)
    pcd = o3d.geometry.PointCloud()
    i = 43
    pc1 = np.fromfile(file_paths[i], dtype=np.float32)
    pc1 = pc1.reshape(-1, 4)[:,:3]
    gt_flow = np.load(label_paths[i])
    correspondence = np.load(correspondences[i])
    pc2 = np.fromfile(file_paths[i-1], dtype=np.float32)
    pc2 = pc2.reshape(-1, 4)[:,:3]
    current_pose = poses[i]
    last_pose = poses[i-1]
    pc1_to_pc2 = (np.linalg.inv(last_pose) @ \
                (current_pose@ np.hstack((pc1,np.ones((pc1.shape[0],1)))).T)).T[:,:3]
    ego_motion = pc1 - pc1_to_pc2
    n1 = pc1.shape[0]
    n2 = pc2.shape[0]
    if n1 > config["num_points"]:
        sample_idx1 = np.random.choice(n1, config["num_points"], replace=False)
    else:
        # sample_idx1 = np.arange(n1)
        sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, config["num_points"] - n1, replace=True)),axis=-1)

    source = pc1[sample_idx1,:]
    gt_flow = gt_flow[sample_idx1,:]
    ego_motion = ego_motion[sample_idx1,:]
    source2 = pc2[correspondence[sample_idx1],:]
    pc1 = torch.tensor(source, dtype=torch.float).unsqueeze(0).cuda()
    pc2 = torch.tensor(source2, dtype=torch.float).unsqueeze(0).cuda()
    # init model
    model = FLOT().cuda()
    pretrained_dict = torch.load(config["pretrain"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    our_flow = model(pc1, pc2)
    our_flow = our_flow.squeeze().detach().cpu().numpy()

    pretrained_dict = torch.load(config["flot_pretrain"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    flot_flow = model(pc1, pc2)
    flot_flow = flot_flow.squeeze().detach().cpu().numpy()

    pretrained_dict = torch.load(config["pwc_pretrain"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    pwc_flow = model(pc1, pc2)
    pwc_flow = pwc_flow.squeeze().detach().cpu().numpy()
    our_flow[np.linalg.norm(our_flow, axis=1) > 2] = np.array([0,0,0])
    flot_flow[np.linalg.norm(flot_flow, axis=1) > 2] = np.array([0,0,0])
    pwc_flow[np.linalg.norm(pwc_flow, axis=1) > 2] = np.array([0,0,0])
    # gt_flow = gt_flow - ego_motion
    # our_flow = our_flow - ego_motion
    # flot_flow = flot_flow - ego_motion
    # pwc_flow = pwc_flow - ego_motion

    # 向量可视化
    pcd.points = o3d.utility.Vector3dVector(source)
    color = [0.8,0.8,0.8]
    color = np.tile(color, (len(source), 1))
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(source2)
    color2 = [0.5,0.5,0.5]
    color2 = np.tile(color2, (len(source2), 1))
    pcd2.colors = o3d.utility.Vector3dVector(color2)

    points_gt = np.concatenate((source, source - pwc_flow), axis=0)
    lines = np.arange(source.shape[0] * 2).reshape(-1, 2, order='F')
    # flow color
    color_gt = np.tile([205/255,154/255,0], (len(lines), 1))

    line_set_gt = o3d.geometry.LineSet()
    line_set_gt.points = o3d.utility.Vector3dVector(points_gt) # shape: (num_points, 3)
    line_set_gt.lines = o3d.utility.Vector2iVector(lines)   # shape: (num_lines, 2)
    line_set_gt.colors = o3d.utility.Vector3dVector(color_gt) # shape: (num_lines, 3)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.add_geometry(pcd2)
    vis.update_geometry(pcd2)
    vis.add_geometry(line_set_gt)
    vis.update_geometry(line_set_gt)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(5)

    # # flow模长可视化
    # color_map = cm.get_cmap('autumn')
    # gt_color = np.linalg.norm((gt_flow), axis=1)
    # our_color = np.linalg.norm((our_flow), axis=1)
    # flot_color = np.linalg.norm((flot_flow), axis=1)
    # pwc_color = np.linalg.norm((pwc_flow), axis=1)
    # color = color_map(1 - gt_color/gt_color.max())[:, :3]
    # pcd.points = o3d.utility.Vector3dVector(source)
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(10)
    # color_map = cm.get_cmap('autumn')
    # pcd.points = o3d.utility.Vector3dVector(source[our_color < gt_color.max()])
    # our_color = our_color[our_color < gt_color.max()]
    # color = color_map(1 - our_color/our_color.max())[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(10)
    # color_map = cm.get_cmap('autumn')
    # pcd.points = o3d.utility.Vector3dVector(source[flot_color < gt_color.max()])
    # flot_color = flot_color[flot_color < gt_color.max()]
    # color = color_map(1 - flot_color/flot_color.max())[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(10)
    # color_map = cm.get_cmap('autumn')
    # pcd.points = o3d.utility.Vector3dVector(source[pwc_color < gt_color.max()])
    # pwc_color = pwc_color[pwc_color < gt_color.max()]
    # color = color_map(1 - pwc_color/pwc_color.max())[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(10)


def visual_scene_flow_on_haomo_data(config):
    index = 6
    seq = str(index).zfill(3)
    print("visualization: ", seq)
    dataset_root = config['haomo_dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    gt_folder = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/haomo_gt", seq, 'scene_flow_gt')
    correspondence_root = os.path.join(dataset_root, 'sequences', seq, 'correspondence_gt')
    file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_folder)) for f in fn]
    file_paths.sort()
    label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(gt_folder)) for f in fn]
    label_paths.sort()
    correspondences = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(correspondence_root)) for f in fn]
    correspondences.sort()
    calib_file = os.path.join(dataset_root, 'sequences', seq, "calib.txt")
    calib = load_calib(calib_file)
    poses_file = os.path.join(dataset_root, 'sequences', seq, "poses.txt")
    poses = load_lidar_poses(poses_file, calib)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=800, height=600)
    # vis.get_render_option().load_from_json('RenderOption.json')
    # # 读取viewpoint参数
    # param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
    # ctr = vis.get_view_control()
    # # 转换视角
    # ctr.convert_from_pinhole_camera_parameters(param)
    i = 25
    pcd = o3d.geometry.PointCloud()

    pc1 = np.fromfile(file_paths[i], dtype=np.float32)
    pc1 = pc1.reshape(-1, 3)[:,:3]
    gt_flow = np.load(label_paths[i])
    correspondence = np.load(correspondences[i])
    pc2 = np.fromfile(file_paths[i-1], dtype=np.float32)
    pc2 = pc2.reshape(-1, 3)[:,:3]
    current_pose = poses[i]
    last_pose = poses[i-1]
    pc1_to_pc2 = (np.linalg.inv(last_pose) @ \
                (current_pose@ np.hstack((pc1,np.ones((pc1.shape[0],1)))).T)).T[:,:3]
    ego_motion = pc1 - pc1_to_pc2
    n1 = pc1.shape[0]
    n2 = pc2.shape[0]
    if n1 > config["num_points"]:
        sample_idx1 = np.random.choice(n1, config["num_points"], replace=False)
    else:
        # sample_idx1 = np.arange(n1)
        sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, config["num_points"] - n1, replace=True)),axis=-1)

    source = pc1[sample_idx1,:]
    gt_flow = gt_flow[sample_idx1,:]
    ego_motion = ego_motion[sample_idx1,:]
    source2 = pc2[correspondence[sample_idx1],:]
    pc1 = torch.tensor(source, dtype=torch.float).unsqueeze(0).cuda()
    pc2 = torch.tensor(source2, dtype=torch.float).unsqueeze(0).cuda()
    # init model
    model = FLOT().cuda()
    pretrained_dict = torch.load(config["pretrain"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    our_flow = model(pc1, pc2)
    our_flow = our_flow.squeeze().detach().cpu().numpy()

    pretrained_dict = torch.load(config["flot_pretrain"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    flot_flow = model(pc1, pc2)
    flot_flow = flot_flow.squeeze().detach().cpu().numpy()

    pretrained_dict = torch.load(config["pwc_pretrain"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    pwc_flow = model(pc1, pc2)
    pwc_flow = pwc_flow.squeeze().detach().cpu().numpy()
    our_flow[np.linalg.norm(our_flow, axis=1) > 2] = np.array([0,0,0])
    flot_flow[np.linalg.norm(flot_flow, axis=1) > 2] = np.array([0,0,0])
    pwc_flow[np.linalg.norm(pwc_flow, axis=1) > 2] = np.array([0,0,0])
    # our_flow /= np.linalg.norm(our_flow, axis=1).max()
    # flot_flow /= np.linalg.norm(flot_flow, axis=1).max()
    # pwc_flow /= np.linalg.norm(pwc_flow, axis=1).max()
    # gt_flow = gt_flow - ego_motion
    # our_flow = our_flow - ego_motion
    # flot_flow = flot_flow - ego_motion
    # pwc_flow = pwc_flow - ego_motion

    # 向量可视化
    pcd.points = o3d.utility.Vector3dVector(source)
    color = [0.8,0.8,0.8]
    color = np.tile(color, (len(source), 1))
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(source2)
    color2 = [0.5,0.5,0.5]
    color2 = np.tile(color2, (len(source2), 1))
    pcd2.colors = o3d.utility.Vector3dVector(color2)
    points_gt = np.concatenate((source, source - gt_flow), axis=0)
    lines = np.arange(source.shape[0] * 2).reshape(-1, 2, order='F')
    # flow color
    color_gt = np.tile([205/255,154/255,0], (len(lines), 1))

    line_set_gt = o3d.geometry.LineSet()
    line_set_gt.points = o3d.utility.Vector3dVector(points_gt) # shape: (num_points, 3)
    line_set_gt.lines = o3d.utility.Vector2iVector(lines)   # shape: (num_lines, 2)
    line_set_gt.colors = o3d.utility.Vector3dVector(color_gt) # shape: (num_lines, 3)
    o3d.visualization.draw_geometries([pcd, pcd2, line_set_gt])
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # vis.add_geometry(pcd2)
    # vis.update_geometry(pcd2)
    # vis.add_geometry(line_set_gt)
    # vis.update_geometry(line_set_gt)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    time.sleep(5)
    # color_map = cm.get_cmap('autumn')
    # gt_color = np.linalg.norm((gt_flow), axis=1)
    # our_color = np.linalg.norm((our_flow), axis=1)
    # flot_color = np.linalg.norm((flot_flow), axis=1)
    # pwc_color = np.linalg.norm((pwc_flow), axis=1)
    # color = color_map(gt_color/gt_color.max())[:, :3]
    # pcd.points = o3d.utility.Vector3dVector(source)
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(10)
    # color_map = cm.get_cmap('autumn')
    # pcd.points = o3d.utility.Vector3dVector(source[our_color < gt_color.max()])
    # our_color = our_color[our_color < gt_color.max()]
    # color = color_map(our_color/our_color.max())[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(color)

    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(10)
    # color_map = cm.get_cmap('autumn')
    # pcd.points = o3d.utility.Vector3dVector(source[flot_color < gt_color.max()])
    # flot_color = flot_color[flot_color < gt_color.max()]
    # color = color_map(flot_color/flot_color.max())[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(10)
    # color_map = cm.get_cmap('autumn')
    # pcd.points = o3d.utility.Vector3dVector(source[pwc_color < gt_color.max()])
    # pwc_color = pwc_color[pwc_color < gt_color.max()]
    # color = color_map(pwc_color/pwc_color.max())[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(color)
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(10)



def visual_scene_flow_from_flot_with_new_residual_flow(config):
    seq = str(config['seq']).zfill(2)
    dataset_root = config['dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    gt_folder = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'scene_flow_gt')
    correspondence_root = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'correspondence_gt')
    file_paths = load_oss_pcs(int(seq), scan_folder)
    label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(gt_folder)) for f in fn]
    label_paths.sort()
    correspondences = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(correspondence_root)) for f in fn]
    correspondences.sort()
    # init model
    model = FLOT_NEW_RESIDUAL()
    pretrained_dict = torch.load(config["pretrain"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    for i in range(1,len(file_paths)):
        if i < 43:
            continue
        if i > 43:
            break
        pc1 = np.fromfile(file_paths[i], dtype=np.float32)
        pc1 = pc1.reshape(-1, 4)[:,:3]
        gt_flow = np.load(label_paths[i])
        correspondence = np.load(correspondences[i])
        pc2 = np.fromfile(file_paths[i-1], dtype=np.float32)
        pc2 = pc2.reshape(-1, 4)[:,:3]
        n1 = pc1.shape[0]
        if n1 > config["num_points"]:
            sample_idx1 = np.random.choice(n1, config["num_points"], replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, config["num_points"] - n1, replace=True)),
                                            axis=-1)

        source = pc1[sample_idx1,:]
        gt_flow = gt_flow[sample_idx1,:]
        source2 = pc2[correspondence[sample_idx1],:]
        pc1 = torch.tensor(source, dtype=torch.float).unsqueeze(0)
        pc2 = torch.tensor(source2, dtype=torch.float).unsqueeze(0)

        scene_flow = model(pc1, pc2)
        scene_flow = scene_flow.squeeze().detach().cpu().numpy()
        pred_target = source - scene_flow
        gt_target = source - gt_flow
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(source)
        color = [1,0,0]
        color = np.tile(color, (len(source), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(source2)
        color2 = [0.5,0,0]
        color2 = np.tile(color2, (len(source2), 1))
        pcd2.colors = o3d.utility.Vector3dVector(color2)

        corrds = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        points_pred = np.concatenate((source, pred_target), axis=0)
        points_gt = np.concatenate((source, gt_target), axis=0)
        lines = np.arange(source.shape[0] * 2).reshape(-1, 2, order='F')
        # blue is pred
        color_pred = np.tile([0,0,1], (len(lines), 1))
        # green is gt
        color_gt = np.tile([0,1,0], (len(lines), 1))
        line_set_pred = o3d.geometry.LineSet()
        line_set_pred.points = o3d.utility.Vector3dVector(points_pred) # shape: (num_points, 3)
        line_set_pred.lines = o3d.utility.Vector2iVector(lines)   # shape: (num_lines, 2)
        line_set_pred.colors = o3d.utility.Vector3dVector(color_pred) # shape: (num_lines, 3)

        line_set_gt = o3d.geometry.LineSet()
        line_set_gt.points = o3d.utility.Vector3dVector(points_gt) # shape: (num_points, 3)
        line_set_gt.lines = o3d.utility.Vector2iVector(lines)   # shape: (num_lines, 2)
        line_set_gt.colors = o3d.utility.Vector3dVector(color_gt) # shape: (num_lines, 3)

        o3d.visualization.draw_geometries([pcd, pcd2, line_set_pred, line_set_gt])




if __name__ == "__main__":
    config = dict()
    config["seq"] = "08"
    # config["batch"] = 5
    config["num_points"] = 16384*2
    config["moving_learning_map"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/config/semantic-kitti-mos.yaml"
    config["haomo_dataset_root"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/LabelData2829_format"
    config["kitti_dataset_root"] = "/oss://haomo-algorithms/release/algorithms/manual_created_cards/semantickitti/semantic_kitti/dataset"
    config["gt_flow"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt"
    config["pred_flow"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/prediction"

    config["pwc_pretrain"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/debug_train_origin_flot_300_random_sample/300_model.pth"
    # config["pretrain"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/debug_train_dynamic_1_static_1_flow_keep_all_dynamic_points_correspondence/300_model.pth"

    config["flot_pretrain"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/debug_train_origin_flot_300_correspondence/300_model.pth"
    config["pretrain"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/debug_train_best_ego_motion_flow/400_model.pth"
    # visual_scene_flow_from_origin_flot(config)
    # config["pretrain"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/debug_train_dynamic_1_static_1_flow_keep_all_dynamic_points_new_residual_flow_correspondence/300_model.pth"

    # save render and view
    # scan = os.path.join(config["kitti_dataset_root"], 'sequences', "08", 'non_ground_velodyne', "000043.bin")
    scan = os.path.join(config["haomo_dataset_root"], 'sequences', "006", 'non_ground_velodyne', "000017.bin")
    pcd = o3d.geometry.PointCloud()
    pc1 = np.fromfile(scan, dtype=np.float32)
    # pc1 = pc1.reshape(-1, 4)[:,:3]
    pc1 = pc1.reshape(-1, 3)[:,:3]
    pcd.points = o3d.utility.Vector3dVector(pc1)
    color = [0.5,0.5,0.5]
    color = np.tile(color, (len(pc1), 1))
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])
    # save_view_point(pcd, "viewpoint.json")
    # visual_scene_flow_on_haomo_data(config)
    # visual_scene_flow_from_our_net(config)
    # save_pred_ego_motion(config)

