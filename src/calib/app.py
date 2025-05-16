import os
import numpy as np
import json

from .calib import calibrate_camera, \
    calibrate_fisheye_camera, \
    calibrate_pair_extrinsics, \
    calibrate_pair_extrinsics_fisheye, \
    create_undistort_img_function, \
    create_undistort_point_function, \
    create_undistort_fisheye_img_function, \
    create_undistort_fisheye_point_function, \
    calibrate_pairwise_extrinsics, \
    triangulate_points, \
    triangulate_points_fisheye, \
    project_points, \
    project_points_fisheye, \
    bundle_adjust_board_points_and_extrinsics

from .points import find_corners_images

from .utils import create_board_object_pts, \
    save_points, load_points, \
    save_camera, load_camera, \
    save_scene, load_scene

from .plotting import plot_calib_board, Scene


def extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len, window_size=11, remove_unused_images=False):
    print(f"Finding calibration board corners for images in {img_dir}")
    filepaths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
    points, fpaths, camera_resolution = find_corners_images(filepaths, board_shape, window_size=window_size)
    saved_fnames = [os.path.basename(f) for f in fpaths]
    saved_points = points.tolist()
    if remove_unused_images:
        for f in filepaths:
            if os.path.basename(f) not in saved_fnames:
                print(f"Removing {f}")
                os.remove(f)
    save_points(out_fpath, saved_points, saved_fnames, board_shape, board_edge_len, camera_resolution)


def plot_corners(points_fpath):
    points, fnames, board_shape, board_edge_len, camera_resolution = load_points(points_fpath)
    plot_calib_board(points, board_shape, camera_resolution)


def plot_points_standard_undistort(points_fpath, camera_fpath):
    k, d, camera_resolution = load_camera(camera_fpath)
    points, _, board_shape, *_ = load_points(points_fpath)
    undistort_pts = create_undistort_point_function(k, d)
    undistorted_points = undistort_pts(points).reshape(points.shape)
    plot_calib_board(undistorted_points, board_shape, camera_resolution)


def plot_points_fisheye_undistort(points_fpath, camera_fpath):
    k, d, camera_resolution = load_camera(camera_fpath)
    points, _, board_shape, *_ = load_points(points_fpath)
    undistort_pts = create_undistort_fisheye_point_function(k, d)
    undistorted_points = undistort_pts(points).reshape(points.shape)
    plot_calib_board(undistorted_points, board_shape, camera_resolution)



def calibrate_standard_intrinsics(points_fpath, out_fpath):
    points, fnames, board_shape, board_edge_len, camera_resolution = load_points(points_fpath)
    obj_pts = create_board_object_pts(board_shape, board_edge_len)
    k, d, r, t = calibrate_camera(obj_pts, points, camera_resolution)
    print("K:\n", k, "\nD:\n", d)
    save_camera(out_fpath, camera_resolution, k, d)
    return k, d, r, t, points


def calibrate_fisheye_intrinsics(points_fpath, out_fpath):
    points, fnames, board_shape, board_edge_len, camera_resolution = load_points(points_fpath)
    obj_pts = create_board_object_pts(board_shape, board_edge_len)
    k, d, r, t, used_points, rms = calibrate_fisheye_camera(obj_pts, points, camera_resolution)
    print("K:\n", k, "\nD:\n", d)
    save_camera(out_fpath, camera_resolution, k, d)
    return k, d, r, t, used_points, rms


def _calibrate_pairwise_extrinsics(calib_func, camera_fpaths, points_fpaths, out_fpath):
    k_arr = []
    d_arr = []
    camera_resolution = None
    # Load camera parameters
    for c in camera_fpaths:
        k1, d1, camera_resolution_1 = load_camera(c)
        k_arr.append(k1)
        d_arr.append(d1)
        if camera_resolution is None:
            camera_resolution = camera_resolution_1
        else:
            assert camera_resolution == camera_resolution_1
    # Load the image points
    img_pts_arr = []
    fnames_arr = []
    board_shape = None
    board_edge_len = None
    for p in points_fpaths:
        points_1, fnames_1, board_shape_1, board_edge_len_1, camera_resolution_1 = load_points(p)
        img_pts_arr.append(points_1)
        fnames_arr.append(fnames_1)
        if board_shape is None:
            board_shape = board_shape_1
        else:
            assert board_shape == board_shape_1
        if board_edge_len is None:
            board_edge_len = board_edge_len_1
        else:
            assert board_edge_len == board_edge_len_1
    r_arr, t_arr = calibrate_pairwise_extrinsics(calib_func, img_pts_arr, fnames_arr, k_arr, d_arr,
                                                 camera_resolution, board_shape, board_edge_len)
    save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, camera_resolution)


def calibrate_standard_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath):
    _calibrate_pairwise_extrinsics(calibrate_pair_extrinsics, camera_fpaths, points_fpaths, out_fpath)


def calibrate_fisheye_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath):
    _calibrate_pairwise_extrinsics(calibrate_pair_extrinsics_fisheye, camera_fpaths, points_fpaths, out_fpath)


def _calibrate_pairwise_extrinsics_manual(calib_func, camera_fpaths, points_fpath, out_fpath):
    k_arr = []
    d_arr = []
    camera_resolution = None
    # Load camera parameters
    for c in camera_fpaths:
        k1, d1, camera_resolution_1 = load_camera(c)
        k_arr.append(k1)
        d_arr.append(d1)
        if camera_resolution is None:
            camera_resolution = camera_resolution_1
        else:
            assert camera_resolution == camera_resolution_1
    # Load the image points
    with open(points_fpath, 'r') as f:
        img_pts_arr = np.array(json.load(f)["points"])
        r_arr, t_arr = calibrate_pairwise_extrinsics_manual(calib_func, img_pts_arr, k_arr, d_arr, camera_resolution)
        save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, camera_resolution)


def calibrate_pairwise_extrinsics_manual(calib_func, img_pts_arr, k_arr, d_arr, camera_resolution):
    # calib_func is one of 'calibrate_pair_extrinsics' or 'calibrate_pair_extrinsics_fisheye'
    n_cam = len(img_pts_arr)
    r_arr = []
    t_arr = []
    # Set camera 1's initial position and rotation
    R1 = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]], dtype=np.float32)
    T1 = np.array([[0, 0, 0]], dtype=np.float32).T
    r_arr.append(R1)
    t_arr.append(T1)
    # Get relative pairwise transformations between subsequent cameras
    for i in range(n_cam - 1):
        k1 = k_arr[i]
        d1 = d_arr[i]
        k2 = k_arr[i + 1]
        d2 = d_arr[i + 1]
        points_1 = img_pts_arr[i]
        points_2 = img_pts_arr[i + 1]
        # Extract corresponding points between cameras into img_pts 1 & 2
        img_pts = np.array([v for v in img_pts_arr[:, i:i+2] if not np.isnan(v).any()]).swapaxes(0,1)
        img_pts_1 = np.array(img_pts[0].reshape((-1,1,1,2)), dtype=np.float32)
        img_pts_2 = np.array(img_pts[1].reshape((-1,1,1,2)), dtype=np.float32)
        # Create object points
        obj_pts = np.array([[0,0,0]], dtype=np.float32)
        rms, r, t = calib_func(obj_pts, img_pts_1, img_pts_2, k1, d1, k2, d2, camera_resolution)
        # Calculate camera pose in the world coordinates
        # Note: T is the world origin position in the camera coordinates
        #       the world position of the camera C = -(R^-1)@T.
        #       Similarly, the rotation of the camera in world coordinates
        #       is given by R^-1.
        #       The inverse of a rotation matrix is also its transpose.
        # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
        R2 = r @ R1
        T2 = r @ T1 + t
        # Update and add to list
        R1 = R2
        T1 = T2
        # Add camera extrinsic params
        r_arr.append(R1)
        t_arr.append(T1)
    return r_arr, t_arr


def plot_scene(scene_fpath):
    _, _, r_arr, t_arr, _ = load_scene(scene_fpath)
    scene = Scene()
    for r, t in zip(r_arr, t_arr):
        scene.plot_camera(r, t)
    scene.show()
    return scene


def _sba_board_points(scene_fpath, points_fpaths, out_fpath, triangulate_func, project_func):
    img_pts_arr = []
    fnames_arr = []
    board_shape = None
    for p_fp in points_fpaths:
        points, fnames, board_shape, *_ = load_points(p_fp)
        img_pts_arr.append(points)
        fnames_arr.append(fnames)
    k_arr, d_arr, r_arr, t_arr, camera_resolution = load_scene(scene_fpath)
    assert len(k_arr) == len(points_fpaths)
    obj_pts, r_arr, t_arr, res = bundle_adjust_board_points_and_extrinsics(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func)
    save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, camera_resolution)
    return res

def sba_board_points(scene_fpath, points_fpaths, out_fpath):
    triangulate_func = triangulate_points
    project_func = project_points
    return _sba_board_points(scene_fpath, points_fpaths, out_fpath, triangulate_func, project_func)

def sba_board_points_fisheye(scene_fpath, points_fpaths, out_fpath):
    triangulate_func = triangulate_points_fisheye
    project_func = project_points_fisheye
    return _sba_board_points(scene_fpath, points_fpaths, out_fpath, triangulate_func, project_func)
