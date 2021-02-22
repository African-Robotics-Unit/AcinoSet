import os
import numpy as np

import pandas as pd
from glob import glob
from .misc import get_markers
import cv2
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

from .sba import _sba_board_points

from .points import find_corners_images, common_image_points

from .utils import create_board_object_pts, \
    save_points, load_points, \
    save_camera, load_camera, \
    load_manual_points, \
    load_dlc_points_as_df, find_scene_file

from .plotting import plot_calib_board, Scene


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
    
    
def plot_scene(data_dir, manual_points_only=False, **kwargs):
    scene = Scene(data_dir, **kwargs)
    k_arr, d_arr, r_arr, t_arr, n_cams, scene_fpath = scene.k_arr, scene.d_arr, scene.r_arr, scene.t_arr, scene.n_cams, scene.scene_fpath
    points_fpaths = os.path.join(os.path.dirname(scene.scene_fpath), "points")
    if manual_points_only:
        points_fpaths = os.path.join(points_fpaths, "manual_points.json")
        pts_2d, *_ = load_manual_points(points_fpaths)
    else:
        points_fpaths = glob(os.path.join(points_fpaths, 'points[1-9].json'))

    colors = [[1,0,0],                       # red: cam pair 0&1
              [0,1,0],                       # greeen: cam pair 1&2
              [kwargs.get('dark_mode',0)]*3, # white if dark_mode else black: cam pair 2&3
              [0,0,1],                       # blue: cam pair 3&4
              [0,0.8,0.8],                   # light blue: cam pair 4&5
              [1,0,1]]                        # fuchsia/magenta: cam pair 5&0

    for i in range(len(colors)):
        colors[i] += [1] # add transparency channel to colors to avoid error msg

    for cam in range(n_cams):
        a, b = cam%n_cams, (cam+1)%n_cams
        if manual_points_only:
            img_pts_1, img_pts_2 = np.array(pts_2d[:, a]), np.array(pts_2d[:, b])
        else:
            pts_1, names_1, *_ = load_points(points_fpaths[a])
            pts_2, names_2, *_ = load_points(points_fpaths[b])
            img_pts_1, img_pts_2, _ = common_image_points(pts_1, names_1, pts_2, names_2)
            
        try:
            pts_3d = triangulate_points_fisheye(
                img_pts_1, img_pts_2, 
                k_arr[a], d_arr[a], r_arr[a], t_arr[a],
                k_arr[b], d_arr[b], r_arr[b], t_arr[b]
            )
            scene.plot_points(pts_3d, color=colors[cam])
        except:
            msg = "Could not triangulate points" if len(img_pts_1) else "No points exist"
            print(msg, f"for cam pair with indices {[a,b]}")
    
    scene.show()


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


def calibrate_standard_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath,
                                           dummy_scene_fpath=None, manual_points_fpath=None
                                          ):
    _calibrate_pairwise_extrinsics(calibrate_pair_extrinsics,
                                   camera_fpaths, points_fpaths, out_fpath,
                                   dummy_scene_fpath, manual_points_fpath
                                  )


def calibrate_fisheye_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath,
                                          dummy_scene_fpath=None, manual_points_fpath=None
                                         ):
    _calibrate_pairwise_extrinsics(calibrate_pair_extrinsics_fisheye,
                                   camera_fpaths, points_fpaths, out_fpath,
                                   dummy_scene_fpath, manual_points_fpath
                                  )


def sba_board_points(scene_fpath, points_fpaths, out_fpath, manual_points_fpath=None, manual_points_only=False, camera_indices=None):
    triangulate_func = triangulate_points
    project_func = project_points
    return _sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, triangulate_func, project_func, camera_indices, manual_points_only)


def sba_board_points_fisheye(scene_fpath, points_fpaths, out_fpath, manual_points_fpath=None, manual_points_only=False, camera_indices=None):
    triangulate_func = triangulate_points_fisheye
    project_func = project_points_fisheye
    return _sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, triangulate_func, project_func, camera_indices, manual_points_only)


def reconstruction_reprojection_video(original_vids_folder, out_fpath, positions, include_lure=False, output_res=None, output_fps=10.0, overlay_dlc_points=True, dlc_thresh=0.8):
    """Saves a video of the reconstruction reprojected into all the cameras
    :param original_vids_folder: The path to the reconstruction's original cheetah videos
    :param out_fpath: The name of the output video
    :param positions: The array of 3D point positions of the reconstruction. Shape: (n frames, m points)
    :param output_res: The resolution of the output video
    :param output_fps: The framerate of the output video
    :param overlay_dlc_points: If True, the DLC points for each camera will also be included in the output video
    :param dlc_thresh: Same as DLC's p_cutoff. Points with a likelihood below dlc_thresh will not be included in the output video
    """    

    og_vids_paths = glob(os.path.join(original_vids_folder, "cam[1-9].mp4"))
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = find_scene_file(original_vids_folder, suppress_output=True)
    camera_params = [[K, D, R, T] for K, D, R, T in zip(k_arr, d_arr, r_arr, t_arr)]
    assert len(og_vids_paths) == n_cams, f"Number of original videos != Number of cams in {os.path.basename(scene_fpath)}"
    
    n_cols, n_rows = n_cams - n_cams//2, 2
    n_frames, pts_per_frame = positions.shape[0], positions.shape[1]*2
    fname = os.path.splitext(os.path.basename(out_fpath))[0].upper()
    markers = get_markers(include_lure)
    
    if output_res:
        aspect_ratio = cam_res[0]/cam_res[1]
        out_vid_height = int(output_res[0]/n_cols/aspect_ratio*n_rows)
        if output_res[1] != out_vid_height:
            print(f"Warning: An output resolution of {output_res} does not maintain the videos' aspect ratio",
                  f"The output resolution will be changed to {(output_res[0], out_vid_height)} to maintain the aspect ratio")
            output_res = (output_res[0], out_vid_height)
    else:
        output_res = (cam_res[0]*n_cols, cam_res[1]*n_rows)

    vid_caps = [cv2.VideoCapture(v) for v in og_vids_paths]
    total_frames = int(vid_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid_caps[0].get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_vid = cv2.VideoWriter(out_fpath, fourcc, output_fps, output_res)

    for vc in vid_caps:
        assert vc.isOpened()

    if overlay_dlc_points:
        dlc_2d_point_files = glob(os.path.join(original_vids_folder, '*.h5'))
        assert(len(dlc_2d_point_files) == n_cams), f"Number of dlc '.h5' files != Number of cams in {n_cams}_cam_scene_sba.json"
    
        # Load DLC Data (pixels, likelihood)
        points_2d_df = load_dlc_points_as_df(dlc_2d_point_files)

        # Restructure dataframe
        points_df = points_2d_df.set_index(['frame', 'camera','marker'])
        points_df = points_df.stack().unstack(level=1).unstack(level=1).unstack()

        # Pixels array
        pixels_df = points_df.loc[:, (range(n_cams), markers, ['x','y'])]
        pixels_df = pixels_df.reindex(columns=pd.MultiIndex.from_product([range(n_cams), markers, ['x','y']]))
        pixels_arr = pixels_df.to_numpy() #shape - (n_frames, n_cams * n_markers * 2)

        # Likelihood array
        likelihood_df = points_df.loc[:, (range(n_cams), markers, 'likelihood')]
        likelihood_df = likelihood_df.reindex(columns=pd.MultiIndex.from_product([range(n_cams), markers, ['likelihood']]))
        likelihood_arr = likelihood_df.to_numpy() #shape - (n_frames, n_cams * n_markers * 1)

        pixels_2_plot = pixels_arr.copy()
        pixels_2_plot[likelihood_arr.repeat(2, axis=1) < dlc_thresh] = np.nan
        pixels_2_plot = pixels_2_plot[:n_frames]

    for frame in range(total_frames):
        print(f"Writing frame {frame+1}\r", end='')
        
        img_tiled = np.zeros((cam_res[1]*n_rows, cam_res[0]*n_cols, 3), dtype='uint8')
        
        for i in range(n_cams):

            # Plot image
            vid_caps[i].set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = vid_caps[i].read()

            if ret:
                if frame < n_frames:
                    # Plot pixels estimated
                    for xy_pt in project_points_fisheye(positions[frame], *camera_params[i]):
                        if not np.isnan(xy_pt).any() and (0 <= xy_pt[0] <= cam_res[0]) and (0 <= xy_pt[1] <= cam_res[1]):
                            cv2.circle(img,(int(xy_pt[0]),int(xy_pt[1])),cam_res[1]//250,(255,0,255),-1)

                # Plot pixels measured
                if overlay_dlc_points:
                    pix_meas_x = pixels_2_plot[frame, i*pts_per_frame:(i+1)*pts_per_frame:2]
                    pix_meas_y = pixels_2_plot[frame, i*pts_per_frame+1:(i+1)*pts_per_frame:2]
                    for xy_pt in zip(pix_meas_x, pix_meas_y):
                        if not np.isnan(xy_pt).any():
                            cv2.circle(img,(int(xy_pt[0]),int(xy_pt[1])),cam_res[1]//250,(255,255,0),-1)
                    draw_text(img, fname, fontColor=(255,0,255))
                    draw_text(img, "DLC", move_text_lower_pixels=80, fontColor=(255,255,0))

                h_start = cam_res[1] * (i//n_cols)
                h_end = h_start + cam_res[1]
                w_start = cam_res[0] * (i%n_cols)
                w_end = w_start + cam_res[0]
                img_tiled[h_start:h_end, w_start:w_end] = img

        out_vid.write(cv2.resize(img_tiled, output_res))
        
    print("\nDone!")

    for v in vid_caps:
        v.release()
        out_vid.release()

        
def combine_dlc_vids(dlc_vids_folder, out_fpath, output_res=None, output_fps=10.0):
    """Saves a video of the reconstruction reprojected into all the cameras
    :param dlc_vids_folder: The path to the cheetah videos with the DeepLabCut overlay
    :param out_fpath: The name of the output video
    :param output_res: The resolution of the output video
    :param output_fps: The framerate of the output video
    """    

    dlc_vids_paths = glob(os.path.join(dlc_vids_folder, "cam[1-9]*dlc*.mp4"))
    if len(dlc_vids_paths)>1:
        vid_caps = [cv2.VideoCapture(v) for v in dlc_vids_paths]

        n_cams = len(dlc_vids_paths)
        n_cols, n_rows = n_cams - n_cams//2, 2
        cam_res = (int(vid_caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_caps[1].get(cv2.CAP_PROP_FRAME_HEIGHT)))
        total_frames = int(vid_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vid_caps[0].get(cv2.CAP_PROP_FPS))

        if output_res:
            aspect_ratio = cam_res[0]/cam_res[1]
            out_vid_height = int(output_res[0]/n_cols/aspect_ratio*n_rows)
            if output_res[1] != out_vid_height:
                print(f"Warning: An output resolution of {output_res} does not maintain the videos' aspect ratio",
                      f"The output resolution will be changed to {(output_res[0], out_vid_height)} to maintain the aspect ratio")
                output_res = (output_res[0], out_vid_height)
        else:
            output_res = (cam_res[0]*n_cols, cam_res[1]*n_rows)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_vid = cv2.VideoWriter(out_fpath, fourcc, output_fps, output_res)

        for vc in vid_caps:
            assert vc.isOpened()

        for frame in range(total_frames):
            print(f"Writing frame {frame+1}\r", end='')

            img_tiled = np.zeros((cam_res[1]*n_rows, cam_res[0]*n_cols, 3), dtype='uint8')

            for i in range(n_cams):

                # Plot image
                vid_caps[i].set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, img = vid_caps[i].read()

                if ret:
                    h_start = cam_res[1] * (i//n_cols)
                    h_end = h_start + cam_res[1]
                    w_start = cam_res[0] * (i%n_cols)
                    w_end = w_start + cam_res[0]
                    img_tiled[h_start:h_end, w_start:w_end] = img

            out_vid.write(cv2.resize(img_tiled, output_res))

        print("\nDone!")

        for v in vid_caps:
            v.release()
            out_vid.release()
