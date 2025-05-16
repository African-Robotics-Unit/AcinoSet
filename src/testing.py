import matplotlib.pyplot as plt
import pickle
from calib import calib, utils
import numpy as np
import math
import glob
import os
import pandas as pd

scene_fpath = 'C://Users//user-pc//Documents//Scripts//amaan//data//results//scene_sba.json'

k, d, r, t, _ = utils.load_scene(scene_fpath)
d = d.reshape((-1,4))

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

cheetah_fte_path = 'C://Users//user-pc//Documents//Scripts//amaan//data//results//cheetah_final//traj_results.pickle'
human_fte_path = 'C://Users//user-pc//Documents//Scripts//amaan//data//results//human_final//traj_results.pickle'

cheetah_fte = load_data(cheetah_fte_path)["positions"]
human_fte = load_data(human_fte_path)["positions"]

print(len(cheetah_fte))
print(len(human_fte))

parts_cheetah = ["l_eye", "r_eye", "nose",
    "neck_base", "spine",
    "tail_base", "tail1", "tail2",
    "l_shoulder", "l_front_knee", "l_front_ankle",
    "r_shoulder", "r_front_knee", "r_front_ankle",
    "l_hip", "l_back_knee", "l_back_ankle",
    "r_hip", "r_back_knee", "r_back_ankle"]

parts_human = ["chin", "forehead", "neck", "shoulder1", "shoulder2",
                "hip1", "hip2", "elbow1", "elbow2", "wrist1", "wrist2", "knee1", "knee2", "ankle1", "ankle2"]

parts_cheetah_gt = ['r_eye','l_eye',
'r_shoulder','r_front_knee','r_front_ankle','r_front_paw',
'spine',
'r_hip','r_back_knee','r_back_ankle','r_back_paw',
'tail1','tail2',
'l_shoulder','l_front_knee','l_front_ankle','l_front_paw',
'l_hip','l_back_knee','l_back_ankle','l_back_paw',
'tail_base','nose','neck_base']

parts_human_gt = ['ankle1','knee1','hip1',
'hip2','knee2','ankle2',
'wrist1','elbow1','shoulder1',
'shoulder2','elbow2','wrist2',
'chin','forehead']

cheetah_gt1_path = 'C://Users//user-pc//Documents//Scripts//amaan//data//results//cheetah_final//gt_cam1.h5'
cheetah_gt2_path = 'C://Users//user-pc//Documents//Scripts//amaan//data//results//cheetah_final//gt_cam2.h5'


human_gt1_path = 'C://Users//user-pc//Documents//Scripts//amaan//data//results//human_final//gt_cam1.h5'
human_gt2_path = 'C://Users//user-pc//Documents//Scripts//amaan//data//results//human_final//gt_cam2.h5'

cheetah_gt1 = pd.read_hdf(cheetah_gt1_path)
cheetah_gt2 = pd.read_hdf(cheetah_gt2_path)

human_gt1 = pd.read_hdf(human_gt1_path)
human_gt2 = pd.read_hdf(human_gt2_path)

#print(len(cheetah_gt1.values[0]))
print(human_gt1.values[29:])

#pts_2d_sba.append(calib.project_points_fisheye(pts_3d_sba['positions'][70], k[cam],d[cam],r[cam],t[cam]))
human_2d = []
cheetah_2d = []
for fn in range(20):
    human_temp = []
    cheetah_temp = []
    for cam in range(2):
        human_temp.append(calib.project_points_fisheye(np.array(human_fte[fn]), k[cam],d[cam],r[cam],t[cam]))
        cheetah_temp.append(calib.project_points_fisheye(cheetah_fte[fn], k[cam],d[cam],r[cam],t[cam]))

    human_2d.append(human_temp)
    cheetah_2d.append(cheetah_temp)

#print(human_2d)
#print(cheetah_2d)

sba_tot = 0
ekf_tot = 0
traj_tot = 0

sum1 = 0
sum2 = 0
n1=0
n2=0

human_errs = []
cheetah_errs = []

avg_x = []
avg_y = []
bounds = []
bounds2 = []
n_corr = 0
n_pck = 0
n_corr1 = 0
n_pck1 = 0
for cam in range(2):
    for i in range(20):
        xs = []
        ys = []
        for j in range(len(parts_human)):
            if(parts_human[j]=="neck" or "2" in parts_human[j]):
                pass
            else:
                gtj = parts_human_gt.index(parts_human[j])
                x_fte = human_2d[i][cam][j][0]
                y_fte = human_2d[i][cam][j][1]
                #print(y_fte)
                if(cam==0):
                    x_gt = human_gt1.values[29:][i][2*gtj]
                    y_gt = human_gt1.values[29:][i][2*gtj+1]
                    
                    #print(x_gt)
                else:
                    x_gt = human_gt2.values[29:][i][2*gtj]
                    y_gt = human_gt2.values[29:][i][2*gtj+1]
                    #print(x_gt)
                
                if not math.isnan(x_gt):
                    xs.append(x_gt)
                    ys.append(y_gt)
                    sum1 += (x_fte - x_gt) ** 2
                    sum1 += (y_fte - y_gt) **2
                    n1+=2
                    if np.sqrt(abs((x_fte - x_gt)*(y_fte - y_gt)))<15:
                        n_corr += 1
                    n_pck +=1
                    human_errs.append((x_fte - x_gt)+(y_fte - y_gt)/2)
        x_bound = np.max(xs)-np.min(xs)
        avg_x.append(x_bound)
        x_bounding = np.mean(avg_x)
        print(x_bounding)
        y_bound = np.max(ys)-np.min(ys)
        avg_y.append(y_bound)
        y_bounding = np.mean(avg_y)
        bounds.append(np.sqrt(x_bounding*y_bounding))
        print(y_bounding)
        xs = []
        ys = []
        for j in range(len(parts_cheetah)):
            if(parts_cheetah[j]=="head" or "l_" in parts_cheetah[j]):
                pass
            else:
                gtj = parts_cheetah_gt.index(parts_cheetah[j])
                x_fte = cheetah_2d[i][cam][j][0]
                y_fte = cheetah_2d[i][cam][j][1]
                #print(y_fte)
                if(cam==0):
                    x_gt = cheetah_gt1.values[i][2*gtj]
                    y_gt = cheetah_gt1.values[i][2*gtj+1]
                    #print(x_gt)
                else:
                    x_gt = cheetah_gt2.values[i][2*gtj]
                    y_gt = cheetah_gt2.values[i][2*gtj+1]
                    #print(x_gt)
                if not math.isnan(x_gt):
                    xs.append(x_gt)
                    ys.append(y_gt)
                    sum2 += (x_fte - x_gt) ** 2
                    sum2 += (y_fte - y_gt) ** 2
                    n2+=2
                    if np.sqrt(abs((x_fte - x_gt)*(y_fte - y_gt)))<30:
                        n_corr1 += 1
                    n_pck1 +=1
                    cheetah_errs.append((x_fte - x_gt)+(y_fte - y_gt)/2)
        x_bound = np.max(xs)-np.min(xs)
        avg_x.append(x_bound)
        x_bounding = np.mean(avg_x)
        print(x_bounding)
        y_bound = np.max(ys)-np.min(ys)
        avg_y.append(y_bound)
        y_bounding = np.mean(avg_y)
        bounds2.append(np.sqrt(x_bounding*y_bounding))
        print(y_bounding)

print("Human")
bounds_avg = np.mean(bounds)
rmse1 = np.sqrt(sum1/n1)
stdev = np.sqrt(sum1/(n1-1))
nrmse1 = np.sqrt(rmse1/bounds_avg)
print("RMSE: {}".format(rmse1))
print("Std Dev: {}".format(stdev))
print("PCK: {}".format(n_corr/n_pck))
print("NRMSE: {}".format(nrmse1))
print("Thresh: {}".format(bounds_avg*0.1))

plt.rcParams.update({'font.size': 18})
plt.subplot(1,2,1)
plt.hist(human_errs, bins=20)
plt.title("Human")
plt.xlabel("Reprojection Error (px)")
plt.ylabel("Frequency")

print("Cheetah")
bounds2_avg = np.mean(bounds2)
rmse2 = np.sqrt(sum2/n2)
stdev = np.sqrt(sum2/(n2-1))
nrmse2 = np.sqrt(rmse2/bounds2_avg)
print("RMSE: {}".format(rmse2))
print("Std Dev: {}".format(stdev))
print("PCK: {}".format(n_corr1/n_pck1))
print("NRMSE: {}".format(nrmse2))
print("Thresh: {}".format(bounds2_avg*0.2))

plt.subplot(1,2,2)
plt.hist(cheetah_errs, bins=20)
plt.title("Cheetah")
plt.xlabel("Reprojection Error (px)")
plt.ylabel("Frequency")
plt.show()

"""
        xs = pts_2d_gt[(i+49,cam+1)][:,0]
        ys = pts_2d_gt[(i+49,cam+1)][:,1]
        x_bound = np.max(xs)-np.min(xs)
        y_bound = np.max(ys)-np.min(ys)
        avg_x.append(x_bound)
        avg_y.append(y_bound)
        #print("sba:")
        #print(len(xs))
        for j in range(len(xs)):
            if not math.isnan(pts_2d_sba[i][cam][j][0]):
                    sum1 += (pts_2d_sba[i][cam][j][0] - xs[j])**2
                    sum1 += (pts_2d_sba[i][cam][j][1] - ys[j])**2
            else:
                sum1+=1080**2
            n1+=2

        #print("ekf:")
        
        for j in range(len(xs)):
            if not math.isnan(pts_2d_ekf[i][cam][j][0]):
                sum2 += (pts_2d_ekf[i][cam][j][0] - xs[j])**2
                sum2 += (pts_2d_ekf[i][cam][j][1] - ys[j])**2
            else:
                sum2+=1080**2
            n2+=2

x_bounding = np.mean(avg_x)
y_bounding = np.mean(avg_y)
print(x_bounding)
print(y_bounding)

rmse1 = np.sqrt(sum1/n1)/6
stdev = np.sqrt(sum1/(n1-1))

rmse2 = np.sqrt(sum2/n2)/6

rmse3 = np.sqrt(sum3/n3)/6
print(rmse1)
print(stdev)

print("SBA:")
print(rmse1)
print(rmse1/np.sqrt(n1))
print(rmse1/(np.sqrt(x_bounding*y_bounding)))
print("EKF:")
print(rmse2)
print(rmse2/np.sqrt(n2))
print(rmse2/(np.sqrt(x_bounding*y_bounding)))
print("TRAJ:")
print(rmse3)
print(rmse3/np.sqrt(n3))
print(rmse3/(np.sqrt(x_bounding*y_bounding)))
"""