import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import matplotlib.pyplot as plt

path = "/home/toytiny/SAIC_radar/scene_flow_data/seq_1/gnssimu-sample-v6@2.csv"
save_path = "/home/toytiny/SAIC_radar/scene_flow_data/seq_1/gnssimu-sample-v6@3.csv"
tb_data = pd.read_table(path, sep=",", header=None).values
cor_yaw = tb_data[1:,9].astype(np.float32) * -1
tb_data[1:,9] = cor_yaw
data = pd.DataFrame(tb_data)
data.to_csv(save_path,header=False,index=False)