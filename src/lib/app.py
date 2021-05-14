import os
import sys
import pickle
import cv2 as cv
import numpy as np
from glob import glob
from .points import find_corners_images, EOM_curve_fit
from .misc import get_3d_marker_coords, get_markers, get_skeleton, Logger
from .vid import proc_video, VideoProcessorCV
from .utils import create_board_object_pts, save_points, load_points, \
    save_camera, load_camera, load_manual_points, load_dlc_points_as_df, \
    find_scene_file, save_optimised_cheetah, save_3d_cheetah_as_2d, get_pairwise_3d_points_from_df
from .sba import _sba_board_points, _sba_points
from .calib import calibrate_camera, calibrate_fisheye_camera, \
    calibrate_pair_extrinsics, calibrate_pair_extrinsics_fisheye, \
    create_undistort_point_function, create_undistort_fisheye_point_function, \
    triangulate_points, triangulate_points_fisheye, \
    project_points, project_points_fisheye, \
    _calibrate_pairwise_extrinsics
from .plotting import plot_calib_board, plot_optimized_states, \
    plot_extrinsics, Cheetah


def extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len, window_size=11, remove_unused_images=False):
    print(f'Finding calibration board corners for images in {img_dir}')
    filepaths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg') or fname.endswith('.png')])
    points, fpaths, cam_res = find_corners_images(filepaths, board_shape, window_size=window_size)
    saved_fnames = [os.path.basename(f) for f in fpaths]
    saved_points = points.tolist()
    if remove_unused_images:
        for f in filepaths:
            if os.path.basename(f) not in saved_fnames:
                print(f'Removing {f}')
                os.remove(f)
    save_points(out_fpath, saved_points, saved_fnames, board_shape, board_edge_len, cam_res)


def initialize_marker_3d(pts_2d_df, marker, k_arr, d_arr, r_arr, t_arr, dlc_thresh_step=0.01, **kwargs):
    # when curve_fit can handle missing data/nans (see https://github.com/scipy/scipy/issues/11841), change this code to
    # increase dlc_thresh while num_frames >= frac*tot_frames where frac = 0.7 or something similar

    # determine highest usable dlc_thresh
    dlc_thresh = -dlc_thresh_step
    frames     = pts_2d_df['frame'].unique()
    tot_frames = num_frames = len(frames)

    # frac = 0.7
    # while num_frames >= frac*tot_frames
    while num_frames == tot_frames:
        dlc_thresh += dlc_thresh_step
        pts_3d_df = get_pairwise_3d_points_from_df(
            pts_2d_df[pts_2d_df['likelihood'] > dlc_thresh],
            k_arr, d_arr, r_arr, t_arr, triangulate_points_fisheye,
            verbose=False
        )

        num_frames = pts_3d_df[pts_3d_df['marker']==marker]['frame'].size

    dlc_thresh -= dlc_thresh_step

    print(f"Initializing {marker}'s 3D points using an interim dlc_thresh of {dlc_thresh:.2f}")

    # run get_pairwise_3d_points_from_df once more with correct dlc_thresh (verbose)
    pts_3d_df = get_pairwise_3d_points_from_df(
        pts_2d_df[pts_2d_df['likelihood'] > dlc_thresh],
        k_arr, d_arr, r_arr, t_arr, triangulate_points_fisheye
    )

    # the following loop has no real effect currently, but will be essential for when curve_fit can handle NaNs
    pts_3d = np.full((tot_frames, 3), np.nan)
    for frame, *pt_3d in pts_3d_df[pts_3d_df['marker']==marker][['frame', 'x', 'y', 'z']].values:
        pts_3d[int(frame) - frames[0], :] = pt_3d

    return EOM_curve_fit(pts_3d, frames=frames, fig_title=f'Fitted curves to initialize {marker}', **kwargs)


# ==========  CALIBRATION  ==========

def calibrate_standard_intrinsics(points_fpath, out_fpath):
    points, fnames, board_shape, board_edge_len, cam_res = load_points(points_fpath)
    obj_pts = create_board_object_pts(board_shape, board_edge_len)
    k, d, r, t = calibrate_camera(obj_pts, points, cam_res)
    print('K:\n', k, '\nD:\n', d)
    save_camera(out_fpath, cam_res, k, d)
    return k, d, r, t, points


def calibrate_fisheye_intrinsics(points_fpath, out_fpath):
    points, fnames, board_shape, board_edge_len, cam_res = load_points(points_fpath)
    obj_pts = create_board_object_pts(board_shape, board_edge_len)
    k, d, r, t, used_points, rms = calibrate_fisheye_camera(obj_pts, points, cam_res)
    print('K:\n', k, '\nD:\n', d)
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
    *_, scene_fpath = find_scene_file(data_dir, scene_fname, verbose=False)
    points_dir = os.path.join(os.path.dirname(scene_fpath), 'points')
    pts_2d, frames = [], []
    if manual_points_only:
        points_fpaths = os.path.join(points_dir, 'manual_points.json')
        pts_2d, frames, _ = load_manual_points(points_fpaths)
        pts_2d = pts_2d.swapaxes(0, 1)
        frames = [frames]*len(pts_2d)
    else:
        points_fpaths = sorted(glob(os.path.join(points_dir, 'points[1-9].json')))
        for fpath in points_fpaths:
            img_pts, img_names, *_ = load_points(fpath)
            pts_2d.append(img_pts)
            frames.append(img_names)

    plot_extrinsics(scene_fpath, pts_2d, frames, triangulate_points_fisheye, manual_points_only, **kwargs)


def plot_cheetah_states(states, smoothed_states=None, out_fpath=None, mplstyle_fpath=None):
    fig, axs = plot_optimized_states(states, smoothed_states, mplstyle_fpath)
    if out_fpath is not None:
        fig.savefig(out_fpath, transparent=True)
        print(f'Saved {out_fpath}\n')


def _plot_cheetah_reconstruction(positions, data_dir, scene_fname=None, labels=None, **kwargs):
    positions = np.array(positions)
    *_, scene_fpath = find_scene_file(data_dir, scene_fname, verbose=False)
    ca = Cheetah(positions, scene_fpath, labels, project_points_fisheye, **kwargs)
    ca.animation()


def plot_cheetah_reconstruction(data_fpath, scene_fname=None, **kwargs):
    label = os.path.basename(os.path.splitext(data_fpath)[0]).upper()
    with open(data_fpath, 'rb') as f:
        data = pickle.load(f)
    positions = data['smoothed_positions'] if 'EKF' in label else data['positions']
    _plot_cheetah_reconstruction([positions], os.path.dirname(data_fpath), scene_fname, labels=[label], **kwargs)


def plot_multiple_cheetah_reconstructions(data_fpaths, scene_fname=None, **kwargs):
    positions = []
    labels = []
    for data_fpath in data_fpaths:
        label = os.path.basename(os.path.splitext(data_fpath)[0]).upper()
        with open(data_fpath, 'rb') as f:
            data = pickle.load(f)
        positions.append(data['smoothed_positions'] if 'EKF' in label else data['positions'])
        labels.append(label)
    _plot_cheetah_reconstruction(positions, os.path.dirname(data_fpath), scene_fname, labels, **kwargs)


# ==========  SAVE FUNCS  ==========
# All these save functions are very similar... Generalise!!
# Also use this instead: out_fpath = os.path.join(out_dir, f'{os.path.basename(out_dir)}.pickle')

def save_tri(positions, out_dir, scene_fpath, start_frame, dlc_thresh, save_videos=True):
    out_fpath = os.path.join(out_dir, 'tri.pickle')
    save_optimised_cheetah(positions, out_fpath, extra_data=dict(start_frame=start_frame))
    save_3d_cheetah_as_2d(positions, out_dir, scene_fpath, get_markers(), project_points_fisheye, start_frame)

    if save_videos:
        video_fpaths = sorted(glob(os.path.join(os.path.dirname(out_dir), 'cam[1-9].mp4'))) # original vids should be in the parent dir
        create_labeled_videos(video_fpaths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)


def save_sba(positions, out_dir, scene_fpath, start_frame, dlc_thresh, save_videos=True):
    out_fpath = os.path.join(out_dir, 'sba.pickle')
    save_optimised_cheetah(positions, out_fpath, extra_data=dict(start_frame=start_frame))
    save_3d_cheetah_as_2d(positions, out_dir, scene_fpath, get_markers(), project_points_fisheye, start_frame)

    if save_videos:
        video_fpaths = sorted(glob(os.path.join(os.path.dirname(out_dir), 'cam[1-9].mp4'))) # original vids should be in the parent dir
        create_labeled_videos(video_fpaths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)


def save_ekf(states, out_dir, scene_fpath, start_frame, dlc_thresh, save_videos=True):
    positions = [get_3d_marker_coords(state) for state in states['x']]
    smoothed_positions = [get_3d_marker_coords(state) for state in states['smoothed_x']]

    out_fpath = os.path.join(out_dir, 'ekf.pickle')
    save_optimised_cheetah(positions, out_fpath, extra_data=dict(smoothed_positions=smoothed_positions, **states, start_frame=start_frame))
    save_3d_cheetah_as_2d(smoothed_positions, out_dir, scene_fpath, get_markers(), project_points_fisheye, start_frame)

    if save_videos:
        video_fpaths = sorted(glob(os.path.join(os.path.dirname(out_dir), 'cam[1-9].mp4'))) # original vids should be in the parent dir
        create_labeled_videos(video_fpaths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)

def save_fte(states, out_dir, scene_fpath, start_frame, dlc_thresh, save_videos=True):
    positions = [get_3d_marker_coords(state) for state in states['x']]

    out_fpath = os.path.join(out_dir, 'fte.pickle')
    save_optimised_cheetah(positions, out_fpath, extra_data=dict(**states, start_frame=start_frame))
    save_3d_cheetah_as_2d(positions, out_dir, scene_fpath, get_markers(), project_points_fisheye, start_frame)

    if save_videos:
        video_fpaths = sorted(glob(os.path.join(os.path.dirname(out_dir), 'cam[1-9].mp4'))) # original vids should be in the parent dir
        create_labeled_videos(video_fpaths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)


# ==========  STDOUT LOGGING  ==========

def start_logging(out_fpath):
    """Start logger, appending print output to given output file"""
    sys.stdout = Logger(out_fpath)


def stop_logging():
    """Stop logging and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal


# ==========  VIDS  ==========

def get_vid_info(path_dir, vid_extension='mp4'):
    """Finds a video specified in/by the path variable and returns its info

    :param path: Either a directory containing a video or the path to a specific video file
    """
    from errno import ENOENT

    orig_path = path_dir
    if not os.path.isfile(path_dir):
        files = sorted(glob(os.path.join(path_dir, f'*.{vid_extension}'))) # assume path is a dir that holds video file(s)
        if files:
            path_dir = files[0]
        else:
            raise FileNotFoundError(ENOENT, os.strerror(ENOENT), orig_path) # assume videos didn't open due to incorrect path

    vid = VideoProcessorCV(in_name=path_dir)
    vid.close()
    return (vid.width(), vid.height()), vid.fps(), vid.frame_count(), vid.codec()


def create_labeled_videos(video_fpaths, videotype='mp4', codec='mp4v', outputframerate=None, out_dir=None, draw_skeleton=False, pcutoff=0.5, dotsize=6, colormap='jet', skeleton_color='white'):
    from functools import partial
    from multiprocessing import Pool

    print('Saving labeled videos...')

    bodyparts = get_markers()
    bodyparts2connect = get_skeleton() if draw_skeleton else None

    if not video_fpaths:
        print('No videos were found. Please check your paths\n')
        return

    if out_dir is None:
        out_dir = os.path.relpath(os.path.dirname(video_fpaths[0]), os.getcwd())

    func = partial(proc_video, out_dir, bodyparts, codec, bodyparts2connect, outputframerate, draw_skeleton, pcutoff, dotsize, colormap, skeleton_color)

    with Pool(min(os.cpu_count(), len(video_fpaths))) as pool:
        pool.map(func,video_fpaths)

    print('Done!\n')