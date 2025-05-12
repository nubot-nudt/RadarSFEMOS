import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import matplotlib.pyplot as plt

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

def xyz2blh(x, y, z):
    
    """Convert XYZ coordinates to BLH,
    return tuple(latitude, longitude, height).
    """
    
    A = 6378137.0
    B = 6356752.314245
    
    e = np.sqrt(1 - (B**2)/(A**2))
    # calculate longitude, in radians
    longitude = np.arctan2(y, x)

    # calculate latitude, in radians
    xy_hypot = np.hypot(x, y)

    lat0 = 0
    latitude = np.arctan(z / xy_hypot)

    while abs(latitude - lat0) > 1E-9:
        lat0 = latitude
        N = A / np.sqrt(1 - e**2 * np.sin(lat0)**2)
        latitude = np.arctan((z + e**2 * N * np.sin(lat0)) / xy_hypot)

    # calculate height, in meters
    N = A / np.sqrt(1 - e**2 * np.sin(latitude)**2)
    if abs(latitude) < np.pi / 4:
        R, phi = np.hypot(xy_hypot, z), np.arctan(z / xy_hypot)
        height = R * np.cos(phi) / np.cos(latitude) - N
    else:
        height = z / np.sin(latitude) - N * (1 - e**2)

    # convert angle unit to degrees
    longitude = np.degrees(longitude)
    latitude = np.degrees(latitude)

    return latitude, longitude, height

def blh2xyz(latitude, longitude, height):
    
    """Convert BLH coordinates to XYZ.
    return [X, Y, Z].
    """
    A = 6378137.0
    B = 6356752.314245
    
    # convert angle unit to radians
    latitude = np.radians(latitude)
    longitude = np.radians(longitude)

    e = np.sqrt(1 - (B**2)/(A**2))
    N = A / np.sqrt(1 - e**2 * np.sin(latitude)**2)
    # calculate X, Y, Z
    X = (N + height) * np.cos(latitude) * np.cos(longitude)
    Y = (N + height) * np.cos(latitude) * np.sin(longitude)
    Z = (N * (1 - e**2) + height) * np.sin(latitude)
    
    return X, Y, Z

def xyz2neu(x0, y0, z0, x, y, z):
    
    """Convert cartesian coordinate system to site-center system.
    """
    # calculate the lat, lon and height of center site
    lat, lon, _ = xyz2blh(x0, y0, z0)
    # convert angle unit to radians
    lat, lon = np.radians(lat), np.radians(lon)
    # calculate NEU
    north = (-np.sin(lat) * np.cos(lon) * (x - x0) - 
             np.sin(lat) * np.sin(lon) * (y - y0) +
             np.cos(lat) * (z - z0))
    east = -np.sin(lon) * (x - x0) + np.cos(lon) * (y - y0)
    up = (np.cos(lat) * np.cos(lon) * (x- x0) +
          np.cos(lat) * np.sin(lon) * (y - y0) +
          np.sin(lat) * (z - z0))

    return north, east, up

def get_trans_from_gnssimu(path):
    
    """
    Get the transformation matrix from raw gnssimu file.
    The reference coordinate system is based on the initial point, north is x, east is y, up is z.
    
    """
    
    tb_data = pd.read_table(path, sep=",", header=None).values
    timestamps = tb_data[3:,1].astype(np.float)
    tb_data[9827,7] = tb_data[9826,7]
    pose_data = np.concatenate((tb_data[:,5:8],tb_data[:,9:12]),axis=1)[3:].astype(np.float64)
    ## [Longitude Latitude Altitude Orientation Pitch Roll]
    ## convert GNSS data to ECEF coordinates
    ndata = len(pose_data[:,0])
    yaw, pitch, roll = pose_data[:,3], pose_data[:,4], pose_data[:,5]
    x, y, z = blh2xyz(pose_data[:,1], pose_data[:,0], pose_data[:,2])
    ## set the initial position as the reference point
    x_r, y_r, z_r = x[0], y[0], z[0]
    ## convert ECEF coordinates to NEU coordinates
    north, east, up = xyz2neu(x_r,y_r,z_r,x,y,z)
    ## transformation from the reference coordinates 
    exts = np.vstack((north, -east, up, yaw, pitch, roll)).T
    poses = get_matrix_from_ext(exts)
    
    ego_trans = np.zeros((np.size(poses,0),4,4))
    
    for i in range(np.size(ego_trans,0)-10):
        
        ego_trans[i] = np.dot(np.linalg.inv(poses[i]), poses[i+1])
        
    # plt.figure()
    # plt.plot(roll)
    # plt.savefig('roll_2.png')
    # plt.figure()
    # plt.plot(ego_trans[:,0,3])
    # plt.savefig('trans_x.png')
    # plt.figure()
    # plt.plot(ego_trans[:,1,3])
    # plt.savefig('trans_y.png')
    # plt.figure()
    # plt.plot(ego_trans[:,2,3])
    # plt.savefig('trans_z.png')
    
    return exts, poses, timestamps

def get_interpolate_pose(path,scale):
    
    exts, poses, ts = get_trans_from_gnssimu(path)
    num_t = len(ts)
    times = scale
    new_ts = np.linspace(ts[0],ts[-1],num_t*times)
    new_exts = np.zeros((num_t*times,6))
    for i in range(6):
        f = interpolate.interp1d(ts, exts[:,i])
        new_exts[:,i] = f(new_ts)
      
    new_poses = get_matrix_from_ext(new_exts)
    
    return new_poses, new_ts
    
    
    
def front_radar_pose(path):
    
    ## Getting radar sensor transformation according to the reference
    radar_front_ext = np.array([0.06, -0.2, 0.7, 4.5, -0.2, 180])
    ego_to_radar = get_matrix_from_ext(radar_front_ext)
    ego_trans, pose_ts = get_trans_from_gnssimu(path)
    trans =  ego_trans @ ego_to_radar 
    
    return trans, pose_ts