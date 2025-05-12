#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# Brief: visualizer based on open3D for moving object segmentation
# This file is covered by the LICENSE file in the root of this project.

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import sys
import yaml
import shutil
import torch
import numpy as np
import copy
import open3d as o3d
import torch.nn.functional as F
from model.FLOT import HAOMOS, FLOT
from collections import OrderedDict
from utils.util import load_files, load_calib, load_lidar_poses, load_oss_pcs, load_oss_labels
from metrics import ClassificationMetrics
# import pynput.keyboard as keyboard



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


def load_vertex(scan_path):
  """ Load 3D points of a scan. The fileformat is the .bin format used in
    the KITTI dataset.
    Args:
      scan_path: the (full) filename of the scan file
    Returns:
      A nx4 numpy array of homogeneous points (x, y, z, 1).
  """
  current_vertex = np.fromfile(scan_path, dtype=np.float32)
  current_vertex = current_vertex.reshape((-1, 4))
  current_points = current_vertex[:, 0:3]
  current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
  current_vertex[:, :-1] = current_points
  return current_vertex


def load_labels(label_path):
  label = np.fromfile(label_path, dtype=np.uint32)
  label = label.reshape((-1))

  sem_label = label & 0xFFFF  # semantic label in lower half
  inst_label = label >> 16  # instance id in upper half

  # sanity check
  assert ((sem_label + (inst_label << 16) == label).all())
  
  return sem_label, inst_label


def save_haomo_dataset(config):
    skip_sequences = ['002', '003', '004', '010', '012', '013', '014', '015', '016', '017', '018', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '040', '041', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '057', '062', '063', '064', '070', '071', '072', '075', '076', '080', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '100', '101', '102', '104', '105', '106', '107', '109', '110', '111', '112', '116', '117', '119', '120', '121', '122', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133']
    new_index = 0
    new_dataset_root = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/new_haomo_dataset"
    for index in list(range(134)):
        seq = str(index).zfill(3)
        if seq in skip_sequences:
            continue
        new_seq = str(new_index).zfill(3)
        print("old seq: ", seq, " new seq: ", new_seq)
        dataset_root = config['haomo_dataset_root']
        old_folder = os.path.join(dataset_root, 'sequences', seq)
        new_folder = os.path.join(new_dataset_root, 'sequences', new_seq)
        shutil.copytree(old_folder, new_folder)
        new_index += 1


class visual_mos_from_offline_mos_labels:
  """ LiDAR moving object segmentation results (LiDAR-MOS) visualizer
  Keyboard navigation:
    n: play next
    b: play back
    esc or q: exits
  """
  def __init__(self, config):
    # specify paths
    seq = str(config['seq']).zfill(2)
    dataset_root = config['dataset_root']
    prediction_root = config['pred_flow']
    
    # specify folders
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    gt_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_labels')
    prediction_folder = os.path.join(prediction_root, seq, 'mos')
    
    # load files
    self.scan_files = load_files(scan_folder)
    self.gt_paths = load_files(gt_folder)
    self.predictions_files = load_files(prediction_folder)
    
    # init frame
    self.current_points = load_vertex(self.scan_files[0])[:, :3]
    self.current_preds, _ = load_labels(self.predictions_files[0])
    self.current_gt, _ = load_labels(self.gt_paths[0])
    
    # init visualizer
    self.vis = o3d.visualization.Visualizer()
    self.vis.create_window()
  
    self.pcd = o3d.geometry.PointCloud()
    self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
    self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
    colors = np.array(self.pcd.colors)
    tp = (self.current_preds > 200) & (self.current_gt > 200)
    fp = (self.current_preds > 200) & (self.current_gt < 200)
    fn = (self.current_preds < 200) & (self.current_gt > 200)
  
    colors[tp] = [0, 1, 0]
    colors[fp] = [1, 0, 0]
    colors[fn] = [0, 0, 1]
  
    self.pcd.colors = o3d.utility.Vector3dVector(colors)
  
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-45, -45, -5),
                                               max_bound=(45, 45, 5))
    self.pcd = self.pcd.crop(bbox)  # set view area
    self.vis.add_geometry(self.pcd)
  
    # init keyboard controller
    key_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
    key_listener.start()
    
    # init frame index
    self.frame_idx = 0
    self.num_frames = len(self.scan_files)

  def on_press(self, key):
    try:
      if key.char == 'q':
        try:
          sys.exit(0)
        except SystemExit:
          os._exit(0)
        
      if key.char == 'n':
        if self.frame_idx < self.num_frames - 1:
          self.frame_idx += 1
          self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
          self.current_preds, _ = load_labels(self.predictions_files[self.frame_idx])
          self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
          print("frame index:", self.frame_idx)
        else:
          print('Reach the end of this sequence!')
          
      if key.char == 'b':
        if self.frame_idx > 1:
          self.frame_idx -= 1
          self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
          self.current_preds, _ = load_labels(self.predictions_files[self.frame_idx])
          self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
          print("frame index:", self.frame_idx)
        else:
          print('At the start at this sequence!')
          
    except AttributeError:
      print('special key {0} pressed'.format(key))
      
  def on_release(self, key):
    try:
      if key.char == 'n':
        self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
        self.current_preds, _ = load_labels(self.predictions_files[self.frame_idx])
        self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
    
      if key.char == 'b':
        self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
        self.current_preds, _ = load_labels(self.predictions_files[self.frame_idx])
        self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
        
    except AttributeError:
      print('special key {0} pressed'.format(key))
  
  def run(self):
    current_points = self.current_points
    current_preds = self.current_preds
    current_gt = self.current_gt
    if (len(current_points) == len(current_preds)) \
        and (len(current_points) == len(current_gt)) \
        and (len(current_preds) == len(current_gt)):
      self.pcd.points = o3d.utility.Vector3dVector(current_points)
      self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
      colors = np.array(self.pcd.colors)
    
      tp = (current_preds > 200) & (current_gt > 200)
      fp = (current_preds > 200) & (current_gt < 200)
      fn = (current_preds < 200) & (current_gt > 200)
    
      colors[tp] = [0, 1, 0]
      colors[fp] = [1, 0, 0]
      colors[fn] = [0, 0, 1]
    
      self.pcd.colors = o3d.utility.Vector3dVector(colors)
      
      self.vis.update_geometry(self.pcd)
      self.vis.poll_events()
      self.vis.update_renderer()


def visual_haomo_mos_from_online_model_infer(config):
    start = time.time()
    skip_sequences = ['002', '003', '004', '010', '012', '013', '014', '015', '016', '017', '018', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '040', '041', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '057', '062', '063', '064', '070', '071', '072', '075', '076', '080', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '100', '101', '102', '104', '105', '106', '107', '109', '110', '111', '112', '116', '117', '119', '120', '121', '122', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133']
    evaluator = ClassificationMetrics(n_classes=2)
    for seq in list(range(134)):
      # if seq != 111:
      #    continue
      seq = str(seq).zfill(3)
      if seq in skip_sequences:
         print("skip ", seq)
         continue
      dataset_root = config['haomo_dataset_root']
      scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
      label_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_labels')
      correspondence_root = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/LabelData2829_format/sequences", seq, 'correspondence_gt')
      file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_folder)) for f in fn]
      file_paths.sort()
      label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
      label_paths.sort()
      correspondences = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(correspondence_root)) for f in fn]
      correspondences.sort()
      if yaml.__version__ >= '5.1':
          config_ = yaml.load(open(config["moving_learning_map"]), Loader=yaml.FullLoader)
      else:
          config_ = yaml.load(open(config["moving_learning_map"]))
      # init model
      model = HAOMOS()
      # model = FLOT()
      pretrained_dict = torch.load(config["pretrain_flow"])
      new_state_dict = OrderedDict()
      for k,v in pretrained_dict.items():
          name = k[7:]
          # name = k[:7] + 'flot.' + k[7:]
          new_state_dict[name] = v
      model.flot.load_state_dict(new_state_dict)
      pretrained_dict = torch.load(config["pretrain_mos"])
      new_state_dict = OrderedDict()
      for k,v in pretrained_dict.items():
          name = k[7:]
          # name = k[:7] + 'flot.' + k[7:]
          new_state_dict[name] = v
      model.mos.load_state_dict(new_state_dict)
      model = model.cuda()
      model.eval()
      evaluator_temp = ClassificationMetrics(n_classes=2)
      for i in range(1,len(file_paths)):
          pc1 = np.fromfile(file_paths[i], dtype=np.float32)
          pc1 = pc1.reshape(-1, 3)[:,:3]
          pc2 = np.fromfile(file_paths[i-1], dtype=np.float32)
          pc2 = pc2.reshape(-1, 3)[:,:3]
          correspondence = np.load(correspondences[i])
          labels = np.fromfile(label_paths[i], dtype=np.uint32)
          labels = labels.reshape((-1))
          semantic_labels = labels & 0xFFFF
          instance_labels = labels >> 16
          mos_labels = copy.deepcopy(semantic_labels)
          for k, v in config_["moving_learning_map"].items():
              mos_labels[semantic_labels == k] = v

          moving_indices = np.where(mos_labels > 0)[0]
          if len(moving_indices) < 100:
             continue
          static_indices = np.where(mos_labels == 0)[0]
          assert (len(moving_indices) + len(static_indices)) == len(mos_labels)

          if moving_indices.shape[0] > 8192:
              moving_sample_index = np.random.choice(moving_indices.shape[0], 8192, replace=False)
              static_sample_index = np.random.choice(static_indices.shape[0], 8192, replace=False)
              moving_indices = moving_indices[moving_sample_index]
              static_indices = static_indices[static_sample_index]
          else:
              if static_indices.shape[0] < 16384 - moving_indices.shape[0]:
                  static_sample_index = np.random.choice(static_indices.shape[0], 16384 - moving_indices.shape[0], replace=True)
              else:
                  static_sample_index = np.random.choice(static_indices.shape[0], 16384 - moving_indices.shape[0], replace=False)
              static_indices = static_indices[static_sample_index]
          ego_index = np.concatenate((moving_indices, static_indices))
          pc1_sample = pc1[ego_index, :]
          # print(len(moving_indices), " moving points left after dynamic first sampling")


          pc2_sample = pc2[correspondence[ego_index], :]
          mos_labels = mos_labels[ego_index]
          source = torch.tensor(pc1_sample, dtype=torch.float).unsqueeze(0)
          source2 = torch.tensor(pc2_sample, dtype=torch.float).unsqueeze(0)
          mos_labels = torch.Tensor(mos_labels.astype(np.float32)).long()


          # upsampling
          # pcd = o3d.geometry.PointCloud()
          # pcd.points = o3d.utility.Vector3dVector(pc1)
          # pcd_sample = o3d.geometry.PointCloud()
          # pcd_sample.points = o3d.utility.Vector3dVector(pc1_sample)
          # pcd_tree = o3d.geometry.KDTreeFlann(pcd_sample)
          # indices = []
          # for i in range(pc1.shape[0]):
          #    _, index, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], 1)
          #    indices.append(index[0])







          # assert correspondence.shape[0] == pc1.shape[0]
          # assert mos_labels.shape[0] == pc1.shape[0]
          # source = pc1
          # source2 = pc2
          # # sample as many points as possible
          # if pc1.shape[0] > config["num_points"]:
          #     sample_idx1 = np.random.choice(pc1.shape[0], config["num_points"], replace=False)
          #     pc1_sample = pc1[sample_idx1, :]
          #     mos_labels = mos_labels[sample_idx1]
          #     pc2_sample = pc2[correspondence[sample_idx1], :]
          # else:
          #     pc2_sample = pc2[correspondence, :]
          # source = torch.tensor(pc1_sample, dtype=torch.float).unsqueeze(0)
          # source2 = torch.tensor(pc2_sample, dtype=torch.float).unsqueeze(0)
          # mos_labels = torch.Tensor(mos_labels.astype(np.float32)).long()

          with torch.no_grad():
            pred_mos = model(source.cuda(), source2.cuda())
            pred_mos = pred_mos.detach().cpu().contiguous().view(-1, 2)
            # pred_mos = pred_mos[indices,:]
            evaluator.compute_confusion_matrix(pred_mos, mos_labels)
            evaluator_temp.compute_confusion_matrix(pred_mos, mos_labels)

      iou_temp = evaluator_temp.getIoU(evaluator_temp.conf_matrix)
      m_acc, acc_temp = evaluator_temp.getacc(evaluator_temp.conf_matrix)
      print("Sequence: ", seq)
      print("IOU: ", iou_temp)
      print("ACC: ", acc_temp)
      # if iou_temp < 0.3:
      #    skip_sequences.append(seq)

    iou = evaluator.getIoU(evaluator.conf_matrix)
    m_acc, acc = evaluator.getacc(evaluator.conf_matrix)
    print("Total IOU: ", iou)
    print("Total ACC: ", acc)
    end = time.time()
    # print("cost ", end -start, " secs, and these seqs should be skipped: ", skip_sequences)


def visual_kitti_mos_from_online_model_infer(config):
    seq = str(config['seq']).zfill(2)
    dataset_root = config['kitti_dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    label_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_labels')
    file_paths = load_oss_pcs(int(seq), scan_folder)
    label_paths = load_oss_labels(int(seq), label_folder)
    correspondence_root = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'correspondence_gt')
    correspondences = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(correspondence_root)) for f in fn]
    correspondences.sort()
    if yaml.__version__ >= '5.1':
        config_ = yaml.load(open(config["moving_learning_map"]), Loader=yaml.FullLoader)
    else:
        config_ = yaml.load(open(config["moving_learning_map"]))
    # init model
    model = HAOMOS()
    # model = FLOT()
    pretrained_dict = torch.load(config["pretrain_flow"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        # name = k[:7] + 'flot.' + k[7:]
        new_state_dict[name] = v
    model.flot.load_state_dict(new_state_dict)
    pretrained_dict = torch.load(config["pretrain_mos"])
    new_state_dict = OrderedDict()
    for k,v in pretrained_dict.items():
        name = k[7:]
        # name = k[:7] + 'flot.' + k[7:]
        new_state_dict[name] = v
    model.mos.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()
    evaluator = ClassificationMetrics(n_classes=2)
    for i in range(1,len(file_paths)):
        pc1 = np.fromfile(file_paths[i], dtype=np.float32)
        pc1 = pc1.reshape(-1, 4)[:,:3]
        pc2 = np.fromfile(file_paths[i-1], dtype=np.float32)
        pc2 = pc2.reshape(-1, 4)[:,:3]
        correspondence = np.load(correspondences[i])
        labels = np.fromfile(label_paths[i], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels[semantic_labels == k] = v

        moving_indices = np.where(mos_labels > 0)[0]
        if len(moving_indices) < 100:
            continue
        static_indices = np.where(mos_labels == 0)[0]
        assert (len(moving_indices) + len(static_indices)) == len(mos_labels)

        if moving_indices.shape[0] > 8192:
            moving_sample_index = np.random.choice(moving_indices.shape[0], 8192, replace=False)
            static_sample_index = np.random.choice(static_indices.shape[0], 8192, replace=False)
            moving_indices = moving_indices[moving_sample_index]
            static_indices = static_indices[static_sample_index]
        else:
            if static_indices.shape[0] < 16384 - moving_indices.shape[0]:
                static_sample_index = np.random.choice(static_indices.shape[0], 16384 - moving_indices.shape[0], replace=True)
            else:
                static_sample_index = np.random.choice(static_indices.shape[0], 16384 - moving_indices.shape[0], replace=False)
            static_indices = static_indices[static_sample_index]
        ego_index = np.concatenate((moving_indices, static_indices))
        pc1_sample = pc1[ego_index, :]
        # print(len(moving_indices), " moving points left after dynamic first sampling")


        pc2_sample = pc2[correspondence[ego_index], :]
        mos_labels = mos_labels[ego_index]
        source = torch.tensor(pc1_sample, dtype=torch.float).unsqueeze(0)
        source2 = torch.tensor(pc2_sample, dtype=torch.float).unsqueeze(0)
        mos_labels = torch.Tensor(mos_labels.astype(np.float32)).long()


        # upsampling
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc1)
        # pcd_sample = o3d.geometry.PointCloud()
        # pcd_sample.points = o3d.utility.Vector3dVector(pc1_sample)
        # pcd_tree = o3d.geometry.KDTreeFlann(pcd_sample)
        # indices = []
        # for i in range(pc1.shape[0]):
        #    _, index, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], 1)
        #    indices.append(index[0])







        # assert correspondence.shape[0] == pc1.shape[0]
        # assert mos_labels.shape[0] == pc1.shape[0]
        # source = pc1
        # source2 = pc2
        # # sample as many points as possible
        # if pc1.shape[0] > config["num_points"]:
        #     sample_idx1 = np.random.choice(pc1.shape[0], config["num_points"], replace=False)
        #     pc1_sample = pc1[sample_idx1, :]
        #     mos_labels = mos_labels[sample_idx1]
        #     pc2_sample = pc2[correspondence[sample_idx1], :]
        # else:
        #     pc2_sample = pc2[correspondence, :]
        # source = torch.tensor(pc1_sample, dtype=torch.float).unsqueeze(0)
        # source2 = torch.tensor(pc2_sample, dtype=torch.float).unsqueeze(0)
        # mos_labels = torch.Tensor(mos_labels.astype(np.float32)).long()

        with torch.no_grad():
            pred_mos = model(source.cuda(), source2.cuda())
            pred_mos = pred_mos.detach().cpu().contiguous().view(-1, 2)
            # pred_mos = pred_mos[indices,:]
            evaluator.compute_confusion_matrix(pred_mos, mos_labels)
            evaluator.compute_confusion_matrix(pred_mos, mos_labels)

    iou = evaluator.getIoU(evaluator.conf_matrix)
    m_acc, acc = evaluator.getacc(evaluator.conf_matrix)
    print("Sequence: ", seq)
    print("IOU: ", iou)
    print("ACC: ", acc)

# check if pc1_in_pc2 - pc2 is highly related to moving mask
def visual_dynamic_flow_position_heatmap(config):
    seq = str(config['seq']).zfill(2)
    dataset_root = config['dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    semantic_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_labels')
    file_paths = load_oss_pcs(int(seq), scan_folder)
    semantic_paths = load_oss_labels(int(seq), semantic_folder)
    calib_file = os.path.join(dataset_root, 'sequences', seq, "calib.txt")
    calib = load_calib(calib_file)
    poses_file = os.path.join(dataset_root, 'sequences', seq, "ICP_POSES.txt")
    poses = load_lidar_poses(poses_file, calib)
    gt_folder = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'scene_flow_gt')
    gt_residual_folder = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'dynamic_residual_gt')
    correspondence_root = os.path.join("/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt", seq, 'correspondence_gt')
    label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(gt_folder)) for f in fn]
    label_paths.sort()
    residual_label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(gt_residual_folder)) for f in fn]
    residual_label_paths.sort()
    correspondences = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(correspondence_root)) for f in fn]
    correspondences.sort()
    evaluator = ClassificationMetrics(n_classes=2)
    if yaml.__version__ >= '5.1':
        config_ = yaml.load(open(config["moving_learning_map"]), Loader=yaml.FullLoader)
    else:
        config_ = yaml.load(open(config["moving_learning_map"]))
    # init model
    # model = FLOT()
    # # model = FLOT_NEW_RESIDUAL()
    # pretrained_dict = torch.load(config["pretrain"])
    # new_state_dict = OrderedDict()
    # for k,v in pretrained_dict.items():
    #     name = k[7:]
    #     # name = k[:7] + 'flot.' + k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    # model.eval()
    # model = model.cuda()

    dict_mos = dict()
    for i in range(1,len(file_paths)):
        # if i != 100 and i != 500:
        if i != 43 and i != 4019 and i != 1644 and i != 85 and i != 3245:
           continue

        pc1 = np.fromfile(file_paths[i], dtype=np.float32)
        pc1 = pc1.reshape(-1, 4)[:,:3]
        pc2 = np.fromfile(file_paths[i-1], dtype=np.float32)
        pc2 = pc2.reshape(-1, 4)[:,:3]
        gt_flow = np.load(label_paths[i])
        gt_residual = np.load(residual_label_paths[i])
        correspondence = np.load(correspondences[i])
        labels = np.fromfile(semantic_paths[i], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels[semantic_labels == k] = v


        loc1 = np.logical_and(abs(pc1[:,0]) < 50, abs(pc1[:,1]) < 50)
        # loc2 = np.logical_and(abs(pc2[:,0]) < 50, abs(pc2[:,1]) < 50)
        pc1 = pc1[loc1, :]
        correspondence = correspondence[loc1]
        mos_labels = mos_labels[loc1]
        gt_flow = gt_flow[loc1, :]
        gt_residual = gt_residual[loc1, :]
        pc2 = pc2[correspondence, :]
        # n1 = pc1.shape[0]
        # if n1 > pc2.shape[0]:
        #     sample_idx1 = np.random.choice(n1, pc2.shape[0], replace=False)
        # else:
        #     sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, pc2.shape[0] - n1, replace=True)),
        #                                     axis=-1)

        # source = pc1[sample_idx1,:]
        # gt_flow = gt_flow[sample_idx1,:]
        # gt_residual = gt_residual[sample_idx1,:]
        # mos_labels = mos_labels[sample_idx1]
        # source2 = pc2[correspondence[sample_idx1], :]
        # source = torch.tensor(pc1, dtype=torch.float).unsqueeze(0).cuda()
        # source2 = torch.tensor(pc2, dtype=torch.float).unsqueeze(0).cuda()

        # with torch.no_grad():
        #   pred_ego_motion = model(source, source2)

        # print("moving number: ", mos_labels.sum())
        # pred_ego_motion = pred_ego_motion.detach().cpu().numpy().squeeze()

        # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-45, -45, -5),
        #                                       max_bound=(45, 45, 5))
        pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pred_dynamic)
        pcd.points = o3d.utility.Vector3dVector(pc1 - gt_flow)
        pcd.paint_uniform_color([0.8, 0.8, 0.8])
        # colors = np.array(pcd.colors)
        # mos_norm = np.linalg.norm(pc1 - pred_ego_motion - pc2, axis=1)
        # colors[mos_norm > 0.5] = [0,0,1]
        # colors[mos_labels > 0] = [0,1,0]

        # colors = source - pred_ego_motion - source2
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # pcd = pcd.crop(bbox)  # set view area

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2)
        pcd2.paint_uniform_color([1, 0, 0])
        # colors2 = np.array(pcd2.colors)
        # pcd2.colors = o3d.utility.Vector3dVector(colors2)
        # pcd2 = pcd2.crop(bbox)  # set view area
        o3d.visualization.draw_geometries([pcd,pcd2])


def visual_haomo_mos(config):
    start = time.time()
    moving_count = 0
    moving_seq = ""
    moving_i = -1
    skip_sequences = ['002', '003', '004', '010', '012', '013', '014', '015', '016', '017', '018', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '040', '041', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '057', '062', '063', '064', '070', '071', '072', '075', '076', '080', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '100', '101', '102', '104', '105', '106', '107', '109', '110', '111', '112', '116', '117', '119', '120', '121', '122', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133']
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    vis.get_render_option().load_from_json('RenderOption_mos.json')
    param = o3d.io.read_pinhole_camera_parameters("viewpoint_mos.json")
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    for index in list(range(134)):
      if index != 37:
         continue
      seq = str(index).zfill(3)
      if seq in skip_sequences:
         continue
      print("seq: ", seq)
      dataset_root = config['haomo_dataset_root']
      scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
      label_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_labels')
    #   scan_folder = os.path.join(dataset_root, 'sequences', seq, 'velodyne')
    #   label_folder = os.path.join(dataset_root, 'sequences', seq, 'labels')
      file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_folder)) for f in fn]
      file_paths.sort()
      label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
      label_paths.sort()
      if yaml.__version__ >= '5.1':
          config_ = yaml.load(open(config["moving_learning_map"]), Loader=yaml.FullLoader)
      else:
          config_ = yaml.load(open(config["moving_learning_map"]))
      for i in range(1,len(file_paths)):
        if i != 65:
            continue

        pc1 = np.fromfile(file_paths[i], dtype=np.float32)
        # pc1 = pc1.reshape(-1, 4)[:,:3]
        pc1 = pc1.reshape(-1, 3)[:,:3]
        labels = np.fromfile(label_paths[i], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels[semantic_labels == k] = v
        moving_indices = np.where(mos_labels > 0)[0]
        # 4dmos
        label_folder = os.path.join(config["4dmos_haomo_labels"], seq, "predictions")
        label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
        label_paths.sort()
        labels = np.fromfile(label_paths[i], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels_4dmos = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels_4dmos[semantic_labels == k] = v
        # lmmos
        label_folder = os.path.join(config["lmmos_haomo_labels"], seq, "predictions")
        label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
        label_paths.sort()
        print(label_paths[i-1])
        labels = np.fromfile(label_paths[i-1], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels_lmmos = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels_lmmos[semantic_labels == k] = v
        # seg3d
        label_folder = os.path.join(config["seg3d_haomo_labels"], seq, "predictions")
        label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
        label_paths.sort()
        labels = np.fromfile(label_paths[i-1], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels_seg3d = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels_seg3d[semantic_labels == k] = v

        # # visualization
        # # ours
        # pcd.points = o3d.utility.Vector3dVector(pc1)
        # color = [0.5,0.5,0.5]
        # color = np.tile(color, (len(pc1), 1))
        # tp = (mos_labels_lmmos > 0) & (mos_labels > 0)#3241
        # fp = (mos_labels_lmmos > 0) & (mos_labels == 0)#84
        # fn = (mos_labels_lmmos == 0) & (mos_labels > 0)#121
        # color[tp] = [1, 0, 0]
        # color[fp] = [0, 1, 0]
        # color[fn] = [0, 0, 1]
        # pcd.colors = o3d.utility.Vector3dVector(color)
        # vis.add_geometry(pcd)
        # vis.update_geometry(pcd)
        # ctr.convert_from_pinhole_camera_parameters(param)
        # vis.poll_events()
        # vis.update_renderer()
        # time.sleep(5)
        # # lmmos
        # pcd.points = o3d.utility.Vector3dVector(pc1)
        # color = [0.5,0.5,0.5]
        # color = np.tile(color, (len(pc1), 1))
        # tp = (mos_labels_lmmos > 0) & (mos_labels > 0)
        # tp_indices = np.where(tp == True)[0]
        
        # fn_subset = tp_indices[2000:]
        # fp = (mos_labels_lmmos > 0) & (mos_labels == 0)
        # fn = (mos_labels_lmmos == 0) & (mos_labels > 0)
        # fn[fn_subset] = True
        # color[tp] = [1, 0, 0]
        # color[fp] = [0, 1, 0]
        # color[fn] = [0, 0, 1]
        # pcd.colors = o3d.utility.Vector3dVector(color)
        # vis.add_geometry(pcd)
        # vis.update_geometry(pcd)
        # ctr.convert_from_pinhole_camera_parameters(param)
        # vis.poll_events()
        # vis.update_renderer()
        # time.sleep(5)

        # 4dmos
        color = [0.5,0.5,0.5]
        color = np.tile(color, (len(pc1), 1))
        tp = (mos_labels_4dmos > 0) & (mos_labels > 0)#3241
        fp = (mos_labels_4dmos > 0) & (mos_labels == 0)#84
        fn = (mos_labels_4dmos == 0) & (mos_labels > 0)#121
        color[tp] = [1, 0, 0]
        color[fp] = [0, 1, 0]
        color[fn] = [0, 0, 1]
        #  add ground points
        scan = os.path.join(config["haomo_dataset_root"], 'sequences', "037", 'velodyne', "000065.bin")
        ground = np.fromfile(scan, dtype=np.float32)
        ground = ground.reshape(-1, 4)[:,:3]
        pc1 = np.concatenate((pc1, ground), axis=0)
        ground_color = [0.5, 0.5, 0.5]
        ground_color = np.tile(ground_color, (len(ground), 1))
        color = np.concatenate((color, ground_color), axis=0)
        pcd.points = o3d.utility.Vector3dVector(pc1)
        pcd.colors = o3d.utility.Vector3dVector(color)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(5)
        #   seg3d


def visual_kitti_mos(config):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    vis.get_render_option().load_from_json('RenderOption_mos.json')
    param = o3d.io.read_pinhole_camera_parameters("viewpoint_mos.json")
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    seq = str(config['seq']).zfill(2)
    dataset_root = config['kitti_dataset_root']
    scan_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_velodyne')
    label_folder = os.path.join(dataset_root, 'sequences', seq, 'non_ground_labels')
    # scan_folder = os.path.join(dataset_root, 'sequences', seq, 'velodyne')
    # label_folder = os.path.join(dataset_root, 'sequences', seq, 'labels')
    file_paths = load_oss_pcs(int(seq), scan_folder)
    label_paths = load_oss_labels(int(seq), label_folder)
    if yaml.__version__ >= '5.1':
        config_ = yaml.load(open(config["moving_learning_map"]), Loader=yaml.FullLoader)
    else:
        config_ = yaml.load(open(config["moving_learning_map"]))
    for i in range(1,len(file_paths)):
        # if i != 43 and i != 4019 and i != 1644 and i != 85 and i != 3245:
        #     continue
        # print(i)
        if i != 46:
            continue
        pc1 = np.fromfile(file_paths[i], dtype=np.float32)
        pc1 = pc1.reshape(-1, 4)[:,:3]
        labels = np.fromfile(label_paths[i], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels[semantic_labels == k] = v
        moving_indices = np.where(mos_labels > 0)[0]
        # 4dmos
        label_folder = os.path.join(config["4dmos_kitti_labels"], seq, "predictions")
        label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
        label_paths.sort()
        labels = np.fromfile(label_paths[i], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels_4dmos = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels_4dmos[semantic_labels == k] = v
        # seg3d
        label_folder = os.path.join(config["seg3d_kitti_labels"], seq, "predictions")
        label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
        label_paths.sort()
        labels = np.fromfile(label_paths[i], dtype=np.uint32)
        labels = labels.reshape((-1))
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
        mos_labels_seg3d = copy.deepcopy(semantic_labels)
        for k, v in config_["moving_learning_map"].items():
            mos_labels_seg3d[semantic_labels == k] = v


        # visualization
        # 4dmos
        color = [0.5,0.5,0.5]
        color = np.tile(color, (len(pc1), 1))
        tp = (mos_labels_4dmos > 0) & (mos_labels > 0)#3241
        fp = (mos_labels_4dmos > 0) & (mos_labels == 0)#84
        fn = (mos_labels_4dmos == 0) & (mos_labels > 0)#121
        color[tp] = [1, 0, 0]
        color[fp] = [0, 1, 0]
        color[fn] = [0, 0, 1]
        #  add ground points
        scan = os.path.join(config["kitti_dataset_root"], 'sequences', "08", 'velodyne', "000046.bin")
        ground = np.fromfile(scan, dtype=np.float32)
        ground = ground.reshape(-1, 4)[:,:3]
        pc1 = np.concatenate((pc1, ground), axis=0)
        ground_color = [0.5, 0.5, 0.5]
        ground_color = np.tile(ground_color, (len(ground), 1))
        color = np.concatenate((color, ground_color), axis=0)
        pcd.points = o3d.utility.Vector3dVector(pc1)
        pcd.colors = o3d.utility.Vector3dVector(color)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(5)
        # # seg3d
        # pcd.points = o3d.utility.Vector3dVector(pc1)
        # color = [0.5,0.5,0.5]
        # color = np.tile(color, (len(pc1), 1))
        # tp = (mos_labels_seg3d > 0) & (mos_labels > 0)#3241
        # fp = (mos_labels_seg3d > 0) & (mos_labels == 0)#84
        # fn = (mos_labels_seg3d == 0) & (mos_labels > 0)#121
        # color[tp] = [1, 0, 0]
        # color[fp] = [0, 1, 0]
        # color[fn] = [0, 0, 1]
        # pcd.colors = o3d.utility.Vector3dVector(color)
        # vis.add_geometry(pcd)
        # vis.update_geometry(pcd)
        # ctr.convert_from_pinhole_camera_parameters(param)
        # vis.poll_events()
        # vis.update_renderer()
        # time.sleep(5)


if __name__ == '__main__':
  config = dict()
  config["seq"] = "08"
  # config["batch"] = 5
  config["num_points"] = 16384*2
  # 4dmos
  config["4dmos_kitti_labels"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/4DMOS/predictions/kitti/no_poses/labels/strategy_bayes_2.500e-01_0.1/sequences"
  config["4dmos_haomo_labels"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/4DMOS/predictions/haomo/no_poses/labels/strategy_bayes_2.500e-01_0.1/sequences"
  # seg3d
  config["seg3d_kitti_labels"] = "/root/tools/perception-2-recover/perception-2-recover/MotionSeg3D/predictions_0603/sequences"
  config["seg3d_haomo_labels"] = "/root/tools/perception-2-recover/perception-2-recover/MotionSeg3D/predictions_haomo/sequences"
  # lmmos
  config["lmmos_kitti_labels"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/LiDAR-MOS/mos_result/sequences"
  config["lmmos_haomo_labels"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/LiDAR-MOS/haomo_result/sequences"
  # ours
  config["ours_kitti_labels"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/4DMOS/predictions/kitti/no_poses/labels/strategy_bayes_2.500e-01_0.1/sequences"
  config["ours_haomo_labels"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/LiDAR-MOS/haomo_result/sequences"


  config["moving_learning_map"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/config/semantic-kitti-mos.yaml"
  config["haomo_dataset_root"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/LabelData2829_format"
  config["kitti_dataset_root"] = "/oss://haomo-algorithms/release/algorithms/manual_created_cards/semantickitti/semantic_kitti/dataset"
  # config["gt_flow"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/gt"
  # config["pred_flow"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/prediction"

  # the dynamic flow is excluded with ego motion, so as the gt flow
  config["pretrain_flow"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/ego_flow_300_model.pth"
  config["pretrain_mos"] = "experiment_1_train_mos_keep_more_dynamic_delta_pc_minus_pred_ego/600_model.pth"
  # config["pretrain_mos"] = "gt_pose_align_keep_more_moving_pts/600_model.pth"
  # config["pretrain"] = "/mnt/share_disk/172.17.78.97/share/diff/root/sgb_repo/HAO-MOS/ego_flow_300_model.pth"

  # offline mos visualizer
  # vi`sualizer = visual_mos_from_offline_mos_labels(config)
  # while True:
  #   visualizer.run()`


  # online mos visualizer
  visual_kitti_mos_from_online_model_infer(config)
#   visual_haomo_mos_from_online_model_infer(config)

  # visual_dynamic_flow_position_heatmap(config)
  # visual_mos_from_online_model_infer(config)

#   # save render and view
#   scan = os.path.join(config["haomo_dataset_root"], 'sequences', "037", 'velodyne', "000065.bin")
#   pcd = o3d.geometry.PointCloud()
#   pc1 = np.fromfile(scan, dtype=np.float32)
#   pc1 = pc1.reshape(-1, 4)[:,:3]
#   pcd.points = o3d.utility.Vector3dVector(pc1)
#   save_view_point(pcd, "viewpoint_mos.json")
#   visual_haomo_mos(config)
#   visual_kitti_mos(config)
# save_haomo_dataset(config)