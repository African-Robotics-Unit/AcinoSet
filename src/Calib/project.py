import os
from calib.app import extract_corners_from_images, plot_corners, calibrate_fisheye_extrinsics_pairwise, plot_scene, sba_board_points_fisheye, _calibrate_pairwise_extrinsics_manual
from calib.utils import load_scene, load_points, save_scene
from calib.calib import triangulate_points_fisheye, prepare_manual_points_for_bundle_adjustment, bundle_adjust_points_only, calibrate_pair_extrinsics_fisheye, project_points_fisheye, bundle_adjust_points_and_extrinsics
from calib.plotting import plot_calib_board, Scene

import numpy as np
import json

points_dir = "/home/liam/Desktop/27_02_2019/Extrinsic Calibration/"
extracted_frames_dir = "/home/liam/Desktop/27_02_2019/Extrinsic Calibration/extracted_frames/"
scene_dir = points_dir


def extract_corners():
    cameras = [1,2,3,4,5,6]
    for i in cameras:
        img_dir = os.path.join(extracted_frames_dir, f"{i}")
        out_fpath = os.path.join(points_dir, f"points_{i}.json")
        board_shape = (9, 6)
        board_edge_len = 0.088
        extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len, window_size=5, remove_unused_images=True)


def plot_all_corners_for_cameras():
    cameras = [1, 2, 3, 4, 5, 6]
    for i in cameras:
        points_fpath = os.path.join(points_dir, f"points_{i}.json")
        plot_corners(points_fpath)


def plot_corners_on_frames():
    cameras = [1, 2, 3, 4, 5, 6]
    for i in cameras:
        points_fpath = os.path.join(points_dir, f"points_{i}.json")
        points, fnames, board_shape, board_edge_len, camera_resolution = load_points(points_fpath)
        for frame_points, fname in zip(points, fnames):
            fpath = os.path.join(extracted_frames_dir, f"{i}", fname)
            plot_calib_board([frame_points], board_shape, camera_resolution, fpath)


def pairwise_extrinsics():
    camera_fpaths = []
    points_fpaths = []
    cameras = [4, 5, 6]
    for i in cameras:
        camera_fpaths.append("tests/assets/intrinsic_calibration/data/fisheye_camera.json")
        points_fpaths.append(os.path.join(points_dir, f"points_{i}.json"))
    out_fpath = os.path.join(scene_dir, f"fisheye_scene_456.json")
    calibrate_fisheye_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath)


def pairwise_extrinsics_manual():
    camera_fpaths = []
    points_fpath = "/home/liam/Desktop/somerset_cheetah_reconstruction/data/manual_points.json"
    cameras = [1,2,3]#,4,5,6]
    for i in cameras:
        camera_fpaths.append("tests/assets/intrinsic_calibration/data/fisheye_camera.json")
    out_fpath = os.path.join(scene_dir, f"fisheye_scene_manual.json")
    _calibrate_pairwise_extrinsics_manual(calibrate_pair_extrinsics_fisheye, camera_fpaths, points_fpath, out_fpath)


def plot_cameras_in_scene():
    scene_fpath = os.path.join(scene_dir, f"fisheye_scene_456.json")
    scene = plot_scene(scene_fpath)


def plot_cameras_with_points():
    scene_fpath = os.path.join(scene_dir, f"fisheye_scene.json")
    manual_points_path = os.path.join(scene_dir, f"manual_points.json")
    manual_camera_idxs = [0,1]
    with open(manual_points_path, "r") as f:
        img_pts_arr = np.array(json.load(f)["points"])
        img_pts = np.array([v for v in img_pts_arr[:, manual_camera_idxs] if not np.isnan(v).any()]).swapaxes(0,1)
        img_pts_1 = np.array(img_pts[0].reshape((-1,1,1,2)), dtype=np.float32)
        img_pts_2 = np.array(img_pts[1].reshape((-1,1,1,2)), dtype=np.float32)
    k_arr, d_arr, r_arr, t_arr, _ = load_scene(scene_fpath)
    pts_3d = triangulate_points_fisheye(img_pts_1, img_pts_2, k_arr[0], d_arr[0], r_arr[0], t_arr[0], k_arr[1], d_arr[1], r_arr[1], t_arr[1])
    print(pts_3d)
    scene = Scene()
    for r, t in zip(r_arr, t_arr):
        scene.plot_camera(r, t)
    scene.plot_points(pts_3d, color=(1,1,1,0.9))
    scene.plot_xy_grid()
    scene.show()

def plot_cameras_with_points_manual():
    manual_points_path = os.path.join(scene_dir, f"manual_points.json")

    scene_fpath_1 = os.path.join(scene_dir, f"fisheye_scene_123.json")
    scene_fpath_2 = os.path.join(scene_dir, f"fisheye_scene_456.json")
    cam_idxs_1 = [0,1]
    cam_idxs_2 = [3,4]

    with open(manual_points_path, "r") as f:
        img_pts_arr = np.array(json.load(f)["points"])
        img_pts = np.array([v for v in img_pts_arr[:, cam_idxs_1] if not np.isnan(v).any()]).swapaxes(0,1)
        img_pts_1 = np.array(img_pts[0].reshape((-1,1,1,2)), dtype=np.float32)
        img_pts_2 = np.array(img_pts[1].reshape((-1,1,1,2)), dtype=np.float32)
    k_arr_1, d_arr_1, r_arr_1, t_arr_1, camera_resolution = load_scene(scene_fpath_1)
    pts_3d_1 = triangulate_points_fisheye(img_pts_1, img_pts_2, k_arr_1[0], d_arr_1[0], r_arr_1[0], t_arr_1[0], k_arr_1[1], d_arr_1[1], r_arr_1[1], t_arr_1[1])

    with open(manual_points_path, "r") as f:
        img_pts_arr = np.array(json.load(f)["points"])
        img_pts = np.array([v for v in img_pts_arr[:, cam_idxs_2] if not np.isnan(v).any()]).swapaxes(0,1)
        img_pts_1 = np.array(img_pts[0].reshape((-1,1,1,2)), dtype=np.float32)
        img_pts_2 = np.array(img_pts[1].reshape((-1,1,1,2)), dtype=np.float32)
    k_arr_2, d_arr_2, r_arr_2, t_arr_2, _ = load_scene(scene_fpath_2)

    #rotate and translate second set
    psi = np.pi
    c = np.cos(psi)
    s = np.sin(psi)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0,0,1]
    ])

    for i, r in enumerate(r_arr_2):
        r_arr_2[i] = r@R

    T = np.array([[6.5,19,1]]).T
    for i, (r,t) in enumerate(zip(r_arr_2, t_arr_2)):
        t_arr_2[i] = t-r@T

    pts_3d_2 = triangulate_points_fisheye(img_pts_1, img_pts_2, k_arr_2[0], d_arr_2[0], r_arr_2[0], t_arr_2[0], k_arr_2[1], d_arr_2[1], r_arr_2[1], t_arr_2[1])

    scene_fpath = os.path.join(scene_dir, f"fisheye_scene.json")
    k_arr = np.concatenate([k_arr_1, k_arr_2])
    d_arr = np.concatenate([d_arr_1, d_arr_2])
    r_arr = np.concatenate([r_arr_1, r_arr_2])
    t_arr = np.concatenate([t_arr_1, t_arr_2])

    save_scene(scene_fpath, k_arr, d_arr, r_arr, t_arr, camera_resolution)


    scene = Scene()
    for r, t in zip(r_arr_1, t_arr_1):
        scene.plot_camera(r, t)
    for r, t in zip(r_arr_2, t_arr_2):
        scene.plot_camera(r, t)
    scene.plot_points(pts_3d_1, color=(1,1,0,0.9))
    scene.plot_points(pts_3d_2, color=(1,0,1,0.9))
    scene.plot_xy_grid()
    scene.show()


def do_calib_bundle_adjustment():
    points_fpaths = [os.path.join(points_dir, f"points_{i}.json") for i in range(1,4)]
    scene_fpath = os.path.join(scene_dir, f"fisheye_scene.json")
    out_fpath = os.path.join(scene_dir, f"fisheye_scene_sba.json")
    sba_board_points_fisheye(scene_fpath, points_fpaths, out_fpath)




def do_point_bundle_adjustment():
    scene_fpath = os.path.join(scene_dir, f"fisheye_scene.json")
    manual_points_path = os.path.join(scene_dir, f"manual_points.json")
    k_arr, d_arr, r_arr, t_arr, camera_resolution = load_scene(scene_fpath)
    with open(manual_points_path, "r") as f:
        img_pts_arr = np.array(json.load(f)["points"])
    triangulate_func = triangulate_points_fisheye
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_manual_points_for_bundle_adjustment(img_pts_arr, k_arr, d_arr, r_arr, t_arr, triangulate_func)
    project_func = project_points_fisheye
    obj_pts, r_arr, t_arr = bundle_adjust_points_and_extrinsics(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func)

    scene = Scene()
    for r, t in zip(r_arr, t_arr):
        scene.plot_camera(r, t)
    scene.plot_points(obj_pts, color=(1,1,0,0.9))
    scene.plot_xy_grid()
    scene.show()

if __name__ == "__main__":
    # extract_corners()
    # plot_all_corners_for_cameras()
    # plot_corners_on_frames()
    # pairwise_extrinsics()
    # pairwise_extrinsics_manual()
    # plot_cameras_in_scene()
    # plot_cameras_with_points()
    # plot_cameras_with_points_manual()
    # do_calib_bundle_adjustment()
    do_point_bundle_adjustment()
