import matplotlib.pyplot as plt
import os
from calib import calib, app, extract, utils, plotting
import glob
import numpy as np
from scipy import stats

def plot_skeleton(project_dir, part):
    """
    Returns arrays of the x, y, and z coordinates for a given body part in a given project
    """
    scene_fpath = os.path.join(project_dir, 'scene_sba.json')
    print(scene_fpath)
    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_fpath)
    D_arr = D_arr.reshape((-1,4))

    print(f"\n\n\nLoading data")
    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    #print(df_paths)

    points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    triangulate_func = calib.triangulate_points_fisheye
    points_2d_filtered_df = points_2d_df[points_2d_df['likelihood']>0.5]
    points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_filtered_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)

    # estimate initial points
    nose_pts = points_3d_df[points_3d_df["marker"]==part][["x", "y", "z", "frame"]].values
    x_slope, x_intercept, *_ = stats.linregress(nose_pts[:,3], nose_pts[:,0])
    y_slope, y_intercept, *_ = stats.linregress(nose_pts[:,3], nose_pts[:,1])
    z_slope, z_intercept, *_ = stats.linregress(nose_pts[:,3], nose_pts[:,2])
    frame_est = np.arange(100)
    x_est = frame_est*x_slope + x_intercept
    y_est = frame_est*y_slope + y_intercept
    z_est = frame_est*z_slope + z_intercept
    psi_est = np.arctan2(y_slope, x_slope)
    
    #print(points_2d_df)
    #print(points_2d_df[points_2d_df['frame']==160])
    #return([nose_pts[:,0], nose_pts[:,1], nose_pts[:,2]])
    return(x_est, y_est, z_est)

def get_bodyparts(project_dir):
    """
    Returns an array of all the bodyparts labelled for a specific project
    """
    print(f"\n\n\nLoading data")
    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    arr = points_2d_df[points_2d_df["frame"]==0][["marker"]][points_2d_df["camera"]==0].values
    final_arr = arr.flatten().tolist()
    return(final_arr)

if __name__=="__main__":
    get_bodyparts("C://Users/user-pc/Documents/Scripts/FYP/data")
    