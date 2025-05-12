from scipy.optimize import least_squares
import numpy as np
import os
from matplotlib import pyplot as plt
from ipdb import set_trace

np.random.seed(1234)

class Static_Seg:
    
    '''
        input:  theta - target azimuth
                vr - target range velocity
        output: stationary mask
    '''
    def __init__(self, threshold=0.05, max_iter=20):
        self.threshold = threshold
        self.max_iter = max_iter

    def least_square_solver(self, theta, vr):

        def err(p, theta, vr):
            alpha = p[0]
            vs = p[1]
            error = vs * np.cos(alpha - theta) - vr
            return error

        p0 = [0, -5]                                                            # alpha, vs
        ret = least_squares(err, p0, args=(theta, vr), verbose=0)
        alpha_ = ret['x'][0]
        vs_ = ret['x'][1]
        return alpha_, vs_

    def ransac(self, theta, vr):
        max_nbr_inliers = 0
        best_alpha_pre = -1
        best_vs_pre = -1
        best_mask = []
        for i in range(self.max_iter):
            inds = np.random.choice(len(vr) - 1, 5)
            # set_trace()
            alpha_pre, vs_pre = self.least_square_solver(theta[inds], vr[inds])
            residual = abs(vr - vs_pre * np.cos(alpha_pre - theta))
            mask = np.array(residual) < abs(self.threshold * vs_pre)
            nbr_inlier = np.sum(mask)
            if nbr_inlier >= max_nbr_inliers:
                max_nbr_inliers = nbr_inlier
                best_alpha_pre = alpha_pre
                best_vs_pre = vs_pre
                best_mask = mask
        return best_mask, best_alpha_pre, best_vs_pre

    def ransac_carla(self,pcl,v_r):
        
        theta = np.arctan(pcl[:, 1] / (pcl[:, 0] + 1e-5))          # azimuth
        low_sp_r = np.sum(np.abs(v_r) < 1) / v_r.shape[0]
        if low_sp_r > 0.5:
            best_mask = v_r < 1.0
            best_alpha_pre = None
            best_vs_pre = None
            # vis = True
        else:
            best_mask, best_alpha_pre, best_vs_pre = self.ransac(theta, v_r)
        o_r1 = 1 - np.sum(best_mask) / pcl.shape[0]
    

        return best_mask, best_alpha_pre, best_vs_pre