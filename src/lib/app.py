# clean imports after video writer funcs are cleaned and moved
import os
import pickle
import numpy as np
from glob import glob
from .misc import get_3d_marker_coords

import cv2 as cv
from .misc import get_markers
from .extract import draw_text

from .calib import calibrate_camera, \
    calibrate_fisheye_camera, \
    calibrate_pair_extrinsics, \
    calibrate_pair_extrinsics_fisheye, \
    create_undistort_point_function, \
    create_undistort_fisheye_point_function, \
    triangulate_points, \
    triangulate_points_fisheye, \
    project_points, \
    project_points_fisheye, \
    _calibrate_pairwise_extrinsics

from .sba import _sba_board_points, _sba_points

from .points import find_corners_images

from .utils import create_board_object_pts, \
    save_points, load_points, \
    save_camera, load_camera, \
    load_manual_points, \
    load_dlc_points_as_df, find_scene_file, save_cheetah

from .plotting import plot_calib_board, plot_optimized_states, Cheetah, plot_extrinsics


def extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len, window_size=11, remove_unused_images=False):
    print(f"Finding calibration board corners for images in {img_dir}")
    filepaths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
    points, fpaths, cam_res = find_corners_images(filepaths, board_shape, window_size=window_size)
    saved_fnames = [os.path.basename(f) for f in fpaths]
    saved_points = points.tolist()
    if remove_unused_images:
        for f in filepaths:
            if os.path.basename(f) not in saved_fnames:
                print(f"Removing {f}")
                os.remove(f)
    save_points(out_fpath, saved_points, saved_fnames, board_shape, board_edge_len, cam_res)

    
# ==========  CALIBRATION  ==========

def calibrate_standard_intrinsics(points_fpath, out_fpath):
    points, fnames, board_shape, board_edge_len, cam_res = load_points(points_fpath)
    obj_pts = create_board_object_pts(board_shape, board_edge_len)
    k, d, r, t = calibrate_camera(obj_pts, points, cam_res)
    print("K:\n", k, "\nD:\n", d)
    save_camera(out_fpath, cam_res, k, d)
    return k, d, r, t, points


def calibrate_fisheye_intrinsics(points_fpath, out_fpath):
    points, fnames, board_shape, board_edge_len, cam_res = load_points(points_fpath)
    obj_pts = create_board_object_pts(board_shape, board_edge_len)
    k, d, r, t, used_points, rms = calibrate_fisheye_camera(obj_pts, points, cam_res)
    print("K:\n", k, "\nD:\n", d)
    save_camera(out_fpath, cam_res, k, d)
    return k, d, r, t, used_points, rms


def calibrate_standard_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath, dummy_scene_fpath=None, manual_points_fpath=None):
    _calibrate_pairwise_extrinsics(calibrate_pair_extrinsics, camera_fpaths, points_fpaths, out_fpath,dummy_scene_fpath, manual_points_fpath)


def calibrate_fisheye_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath, dummy_scene_fpath=None, manual_points_fpath=None):
    _calibrate_pairwise_extrinsics(calibrate_pair_extrinsics_fisheye, camera_fpaths, points_fpaths, out_fpath, dummy_scene_fpath, manual_points_fpath)


# ==========  SBA  ==========
    
def sba_board_points_standard(scene_fpath, points_fpaths, out_fpath, manual_points_fpath=None, manual_points_only=False, camera_indices=None):
    triangulate_func = triangulate_points
    project_func = project_points
    return _sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, triangulate_func, project_func, camera_indices, manual_points_only)


def sba_board_points_fisheye(scene_fpath, points_fpaths, out_fpath, manual_points_fpath=None, manual_points_only=False, camera_indices=None):
    triangulate_func = triangulate_points_fisheye
    project_func = project_points_fisheye
    return _sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, triangulate_func, project_func, camera_indices, manual_points_only)


def sba_points_standard(scene_fpath, points_2d_df):
    triangulate_func = triangulate_points
    project_func = project_points
    return _sba_points(scene_fpath, points_2d_df, triangulate_func, project_func)
    

def sba_points_fisheye(scene_fpath, points_2d_df):
    triangulate_func = triangulate_points_fisheye
    project_func = project_points_fisheye
    return _sba_points(scene_fpath, points_2d_df, triangulate_func, project_func)

    
# ==========  PLOTTING  ==========

def plot_corners(points_fpath):
    points, fnames, board_shape, board_edge_len, cam_res = load_points(points_fpath)
    plot_calib_board(points, board_shape, cam_res)


def plot_points_standard_undistort(points_fpath, camera_fpath):
    k, d, cam_res = load_camera(camera_fpath)
    points, _, board_shape, *_ = load_points(points_fpath)
    undistort_pts = create_undistort_point_function(k, d)
    undistorted_points = undistort_pts(points).reshape(points.shape)
    plot_calib_board(undistorted_points, board_shape, cam_res)


def plot_points_fisheye_undistort(points_fpath, camera_fpath):
    k, d, cam_res = load_camera(camera_fpath)
    points, _, board_shape, *_ = load_points(points_fpath)
    undistort_pts = create_undistort_fisheye_point_function(k, d)
    undistorted_points = undistort_pts(points).reshape(points.shape)
    plot_calib_board(undistorted_points, board_shape, cam_res)
    
    
def plot_scene(data_dir, scene_fname=None, manual_points_only=False, **kwargs):
    *_, scene_fpath = find_scene_file(data_dir, scene_fname)
    points_dir = os.path.join(os.path.dirname(scene_fpath), "points")
    pts_2d, frames = [], []
    if manual_points_only:
        points_fpaths = os.path.join(points_dir, "manual_points.json")
        pts_2d, frames, _ = load_manual_points(points_fpaths)
        pts_2d = pts_2d.swapaxes(0, 1)
        frames = [frames]*len(pts_2d)
    else:
        points_fpaths = glob(os.path.join(points_dir, 'points[1-9].json'))
        for fpath in points_fpaths:
            img_pts, img_names, *_ = load_points(fpath)
            pts_2d.append(img_pts)
            frames.append(img_names)

    plot_extrinsics(scene_fpath, pts_2d, frames, triangulate_points_fisheye, manual_points_only, **kwargs)
    
    
def plot_cheetah_states(states, smoothed_states=None, out_fpath=None, mplstyle_fpath=None):
    fig, axs = plot_optimized_states(states, smoothed_states, mplstyle_fpath)
    if out_fpath is not None:
        fig.savefig(out_fpath, transparent=True)
        print(f'Saved to {out_fpath}\n')

        
def _plot_cheetah_reconstruction(positions, data_dir, scene_fname=None, labels=None, **kwargs):
    positions = np.array(positions)
    *_, scene_fpath = find_scene_file(data_dir, scene_fname, verbose=True)
    ca = Cheetah(positions, scene_fpath, labels, project_points_fisheye, **kwargs)
    ca.animation()
    
    
def plot_cheetah_reconstruction(data_fpath, scene_fname=None, **kwargs):
    label = os.path.basename(os.path.splitext(data_fpath)[0]).upper()
    with open(data_fpath, 'rb') as f:
        data = pickle.load(f)
    positions = data["smoothed_positions"] if 'EKF' in label else data["positions"]
    _plot_cheetah_reconstruction([positions], os.path.dirname(data_fpath), scene_fname, labels=[label], **kwargs)
    

def plot_multiple_cheetah_reconstructions(data_fpaths, scene_fname=None, **kwargs):
    positions = []
    labels = []
    for data_fpath in data_fpaths:
        label = os.path.basename(os.path.splitext(data_fpath)[0]).upper()
        with open(data_fpath, 'rb') as f:
            data = pickle.load(f)
        positions.append(data["smoothed_positions"] if 'EKF' in label else data["positions"])
        labels.append(label)
    _plot_cheetah_reconstruction(positions, os.path.dirname(data_fpath), scene_fname, labels, **kwargs)


# ==========  SAVE FUNCS  ==========

def save_sba(positions, out_dir, pad_len):
    for s in  ["", "_padded"]:
        ## concatenate a NaN array at the beginning of positions
        if "padded" in s:
            nan_arr = np.full((pad_len, len(positions[0]), 3), np.nan)
            positions = np.concatenate((nan_arr, positions))
            
        out_fpath = os.path.join(out_dir, f"sba{s}.pickle")
        save_cheetah(positions, out_fpath)


def save_ekf(states, out_dir, pad_len):
    for s in  ["", "_padded"]:
        ## concatenate a NaN array at the beginning of states
        if "padded" in s:
            nan_arr = np.full((pad_len, len(states['x'][0])), np.nan)
            for key in states:
                states[key] = np.concatenate((nan_arr, states[key]))

        positions = [get_3d_marker_coords(state) for state in states['x']]
        smoothed_positions = [get_3d_marker_coords(state) for state in states['smoothed_x']]
        
        out_fpath = os.path.join(out_dir, f"ekf{s}.pickle")
        save_cheetah(positions, out_fpath, data=dict(smoothed_positions=smoothed_positions, **states))


def save_fte(states, out_dir, pad_len):
    for s in  ["", "_padded"]:
        ## concatenate a NaN array at the beginning of x, dx and ddx
        if "padded" in s:
            nan_arr = np.full((pad_len, len(states['x'][0])), np.nan)
            for key in states:
                states[key] = np.concatenate((nan_arr, states[key]))

        positions = [get_3d_marker_coords(state) for state in states['x']]
        
        out_fpath = os.path.join(out_dir, f"fte{s}.pickle")
        save_cheetah(positions, out_fpath, data=states)


# ==========  VIDS  ==========

def reconstruction_reprojection_video(original_vids_dir, out_fpath, positions, output_res=None, output_fps=10.0, overlay_dlc_points=True, dlc_thresh=0.8):
    """Saves a video of the reconstruction reprojected into all the cameras
    :param original_vids_dir: The path to the directory holding the original cheetah videos
    :param out_fpath: The name of the output video
    :param positions: The array of 3D point positions from the optimization. Shape: (n frames, m points). 
    :param output_res: The resolution of the output video
    :param output_fps: The framerate of the output video. The default is 10
    :param overlay_dlc_points: If True, the DLC points for each camera will also be included in the output video. The default is True
    :param dlc_thresh: Same as DLC's p_cutoff. Points with a likelihood below dlc_thresh will not be included in the output video. The default is 0.8
    """

    og_vids_paths = glob(os.path.join(original_vids_dir, "cam[1-9].mp4"))
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = find_scene_file(original_vids_dir)
    camera_params = [[K, D, R, T] for K, D, R, T in zip(k_arr, d_arr, r_arr, t_arr)]
    assert len(og_vids_paths) == n_cams, f"Number of original videos != Number of cams in {os.path.basename(scene_fpath)}"
    
    n_cols, n_rows = n_cams - n_cams//2, 2
    n_frames, pts_per_frame = positions.shape[0], positions.shape[1]*2
    fname = os.path.splitext(os.path.basename(out_fpath))[0].upper()
    markers = get_markers()
    
    if output_res:
        aspect_ratio = cam_res[0]/cam_res[1]
        out_vid_height = int(output_res[0]/n_cols/aspect_ratio*n_rows)
        if output_res[1] != out_vid_height:
            print(f"Warning: An output resolution of {output_res} does not maintain the videos' aspect ratio",
                  f"The output resolution will be changed to {(output_res[0], out_vid_height)} to maintain the aspect ratio")
            output_res = (output_res[0], out_vid_height)
    else:
        output_res = (cam_res[0]*n_cols, cam_res[1]*n_rows)

    vid_caps = [cv.VideoCapture(v) for v in og_vids_paths]
    total_frames = int(vid_caps[0].get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(vid_caps[0].get(cv.CAP_PROP_FPS))

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out_vid = cv.VideoWriter(out_fpath, fourcc, output_fps, output_res)

    for vc in vid_caps:
        assert vc.isOpened()

    if overlay_dlc_points:
        dlc_2d_point_files = glob(os.path.join(original_vids_dir, 'dlc','*.h5'))
        assert(len(dlc_2d_point_files) == n_cams), f"Number of dlc '.h5' files != Number of cams in {n_cams}_cam_scene_sba.json"
    
        # Load DLC Data (pixels, likelihood)
        points_2d_df = load_dlc_points_as_df(dlc_2d_point_files)
        points_2d_df = points_2d_df[points_2d_df['likelihood'] > dlc_thresh]

    for frame in range(total_frames):
        print(f"Writing frame {frame+1}\r", end='')
        
        img_tiled = np.zeros((cam_res[1]*n_rows, cam_res[0]*n_cols, 3), dtype='uint8')
        frame_df = points_2d_df[points_2d_df['frame']==frame]
        
        for i in range(n_cams):

            # Plot image
            vid_caps[i].set(cv.CAP_PROP_POS_FRAMES, frame)
            ret, img = vid_caps[i].read()
            
            cam_df = frame_df[frame_df['camera']==i]

            if ret:
                if frame < n_frames:
                    # Plot pixels estimated
                    for xy_pt in project_points_fisheye(positions[frame], *camera_params[i]):
                        if not np.isnan(xy_pt).any():
                            if (0 <= xy_pt[0] <= cam_res[0]) and (0 <= xy_pt[1] <= cam_res[1]):
                                cv.circle(img,(int(xy_pt[0]),int(xy_pt[1])),cam_res[1]//250,(255,0,255),-1)

                # Plot pixels measured
                if overlay_dlc_points:
                    pix_meas = cam_df[cam_df['marker'].isin(markers)][['x','y']].values
                    for xy_pt in pix_meas:
                        cv.circle(img,(int(xy_pt[0]),int(xy_pt[1])),cam_res[1]//250,(255,255,0),-1)
                    draw_text(img, fname, fontColor=(255,0,255))
                    draw_text(img, "DLC", move_text_lower_pixels=80, fontColor=(255,255,0))

                h_start = cam_res[1] * (i//n_cols)
                h_end = h_start + cam_res[1]
                w_start = cam_res[0] * (i%n_cols)
                w_end = w_start + cam_res[0]
                img_tiled[h_start:h_end, w_start:w_end] = img

        out_vid.write(cv.resize(img_tiled, output_res))
        
    print("\nDone!")

    for v in vid_caps:
        v.release()
        out_vid.release()


def combine_dlc_vids(dlc_vids_dir, output_res=None, output_fps=10.0):
    """Saves a video of the optimized result reprojected into all cameras in the scene
    :param dlc_vids_dir: The path to the cheetah videos with the DeepLabCut overlay
    :param output_res: The resolution of the output video 
    :param output_fps: The framerate of the output video
    """    

    dlc_vids_paths = glob(os.path.join(dlc_vids_dir, "cam[1-9]*dlc*.mp4"))
    if len(dlc_vids_paths)>1:
        vid_caps = [cv.VideoCapture(v) for v in dlc_vids_paths]

        n_cams = len(dlc_vids_paths)
        n_cols, n_rows = n_cams - n_cams//2, 2
        cam_res = (int(vid_caps[0].get(cv.CAP_PROP_FRAME_WIDTH)), int(vid_caps[1].get(cv.CAP_PROP_FRAME_HEIGHT)))
        total_frames = int(vid_caps[0].get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(vid_caps[0].get(cv.CAP_PROP_FPS))

        if output_res:
            aspect_ratio = cam_res[0]/cam_res[1]
            out_vid_height = int(output_res[0]/n_cols/aspect_ratio*n_rows)
            if output_res[1] != out_vid_height:
                print(f"Warning: An output resolution of {output_res} does not maintain the videos' aspect ratio",
                      f"The output resolution will be changed to {(output_res[0], out_vid_height)} to maintain the aspect ratio")
                output_res = (output_res[0], out_vid_height)
        else:
            output_res = (cam_res[0]*n_cols, cam_res[1]*n_rows)

        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out_vid = cv.VideoWriter(os.path.join(dlc_vids_dir, 'dlc.avi'), fourcc, output_fps, output_res)

        for vc in vid_caps:
            assert vc.isOpened()

        for frame in range(total_frames):
            print(f"Writing frame {frame+1}\r", end='')

            img_tiled = np.zeros((cam_res[1]*n_rows, cam_res[0]*n_cols, 3), dtype='uint8')

            for i in range(n_cams):

                # Plot image
                vid_caps[i].set(cv.CAP_PROP_POS_FRAMES, frame)
                ret, img = vid_caps[i].read()

                if ret:
                    h_start = cam_res[1] * (i//n_cols)
                    h_end = h_start + cam_res[1]
                    w_start = cam_res[0] * (i%n_cols)
                    w_end = w_start + cam_res[0]
                    img_tiled[h_start:h_end, w_start:w_end] = img

            out_vid.write(cv.resize(img_tiled, output_res))

        print("\nDone!")

        for v in vid_caps:
            v.release()
            out_vid.release()
