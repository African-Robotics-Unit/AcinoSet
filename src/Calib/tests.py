import cv2
import os
import random
import json
import time
from datetime import datetime
from calib import calib
from calib.calib import triangulate_points, \
    triangulate_points_fisheye, \
    project_points, \
    project_points_fisheye, \
    cost_func_points_only, cost_func_points_extrinsics, \
    bundle_adjust_board_points_only, bundle_adjust_board_points_and_extrinsics, \
    prepare_calib_board_data_for_bundle_adjustment, get_pairwise_3d_points_from_df
from calib.plotting import Scene
from calib.utils import load_points, load_scene, create_dlc_points_2d_file
from calib.app import extract_corners_from_images, \
    plot_corners, \
    calibrate_standard_intrinsics, \
    calibrate_fisheye_intrinsics, \
    plot_points_standard_undistort, \
    plot_points_fisheye_undistort, \
    calibrate_standard_extrinsics_pairwise, \
    calibrate_fisheye_extrinsics_pairwise, \
    plot_scene


def test_find_corners():
    board_shape = (9, 6)
    filepath = "./assets/gopro/4.png"
    img = cv2.imread(filepath)
    assert img is not None, f"Couldn't read image: {filepath}"
    corners = calib.find_corners(img, board_shape)
    calib.plot_corners(img, corners, board_shape)
    cv2.destroyAllWindows()



def test_find_corners_images():
    board_shape = (9, 6)
    filepaths = ["./assets/gopro/4.png", "./assets/doesnt_exist.png"]
    corners, *_ = calib.find_corners_images(filepaths, board_shape)
    img = cv2.imread(filepaths[0])
    calib.plot_corners(img, corners[0], board_shape)
    cv2.destroyAllWindows()


def test_calibrate_camera():
    filepaths = sorted([os.path.abspath("./assets/gopro/" + fname) for fname in os.listdir("./assets/gopro/") if
                        fname.endswith(".png")])
    board_shape = (9, 6)
    corners, *_ = calib.find_corners_images(filepaths, board_shape)
    obj_pts = calib.create_board_object_pts(board_shape, 0.088)
    print(calib.calibrate_camera(obj_pts, corners, (2704, 1520)))


def test_create_undistort_functions():
    filepaths = sorted([os.path.abspath("../frames/" + fname) for fname in os.listdir("frames/") if
                        fname.endswith(".jpg")])
    board_shape = (8, 6)
    corners, *_ = calib.find_corners_images(filepaths, board_shape)
    obj_pts = calib.create_board_object_pts(board_shape, 0.088)
    k, d, *_ = calib.calibrate_camera(obj_pts, corners, (2704, 1520))
    undistort_img = calib.create_undistort_img_function(k, d, (2704, 1520))
    undistort_pts = calib.create_undistort_point_function(k, d)
    # load image
    img = cv2.imread(filepaths[0])
    assert img is not None, f"Couldn't read image: {filepaths[0]}"
    # do undistortion
    undistorted_img = undistort_img(img)
    undistorted_corners = undistort_pts(corners[0])
    calib.plot_corners(undistorted_img, undistorted_corners, board_shape)
    cv2.destroyAllWindows()


def test_calibrate_fisheye_camera():
    filepaths = sorted([os.path.abspath("./assets/gopro/" + fname) for fname in os.listdir("./assets/gopro/") if
                        fname.endswith(".png")])

    board_shape = (9, 6)
    corners, *_ = calib.find_corners_images(filepaths, board_shape)
    obj_pts = calib.create_board_object_pts(board_shape, 0.088)
    print(calib.calibrate_fisheye_camera(obj_pts, corners, (2704, 1520)))


def test_create_undistort_fisheye_functions():
    filepaths = sorted([os.path.abspath("../frames/" + fname) for fname in os.listdir("frames/") if
                        fname.endswith(".jpg")])
    random.shuffle(filepaths)
    board_shape = (8, 6)
    camera_resolution = (2704, 1520)
    corners, *_ = calib.find_corners_images(filepaths, board_shape)
    obj_pts = calib.create_board_object_pts(board_shape, 0.04)
    k, d, *_ = calib.calibrate_fisheye_camera(obj_pts, corners, (camera_resolution[1], camera_resolution[0]))
    print(k)
    print(d)

    camera = {"resolution": camera_resolution, "k": k.tolist(), "d": d.tolist()}
    with open("tests/cam_params.json", 'w') as fp:
        json.dump(camera, fp, indent=2)

    undistort_img = calib.create_undistort_fisheye_img_function(k, d, camera_resolution) #Note that resolution order is swapped!!
    undistort_pts = calib.create_undistort_fisheye_point_function(k, d)
    # load image
    img = cv2.imread(filepaths[0])
    assert img is not None, f"Couldn't read image: {filepaths[0]}"
    # do undistortion
    undistorted_img = undistort_img(img)

    undistorted_corners = undistort_pts(corners[0])
    calib.plot_corners(undistorted_img, undistorted_corners, board_shape)
    cv2.destroyAllWindows()


def test_find_and_save_extrinsic_points():
    filepaths = sorted([os.path.join("tests/assets/extrinsic_calibration/images/2", fname) for fname in os.listdir(
        "tests/assets/extrinsic_calibration/images/2") if
                        fname.endswith(".jpg")])
    board_shape = (9, 6)
    corners, fnames, _ = calib.find_corners_images(filepaths, board_shape)
    points = {k: v for k, v in zip(fnames, corners.tolist())}
    data = {
        "created_timestamp": str(datetime.now()),
        "board_shape":board_shape,
        "points": points
    }
    with open("tests/assets/extrinsic_calibration/data/extrinsic_points_2.json", "w") as f:
        json.dump(data, f)


def test_triangulate_points():
    points_1, *_ = load_points("tests/assets/extrinsic_calibration/data/points_1.json")
    points_2, *_ = load_points("tests/assets/extrinsic_calibration/data/points_2.json")
    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/standard_scene.json")
    pts_3d = triangulate_points(points_1, points_2, k_arr[0], d_arr[0], r_arr[0], t_arr[0], k_arr[1], d_arr[1], r_arr[1], t_arr[1])
    scene = Scene()
    for r, t in zip(r_arr[:2], t_arr[:2]):
        scene.plot_camera(r, t)
    scene.plot_points(pts_3d)
    scene.show()


def test_triangulate_points_fisheye():
    points_1, *_ = load_points("tests/assets/extrinsic_calibration/data/points_1.json")
    points_2, *_ = load_points("tests/assets/extrinsic_calibration/data/points_2.json")
    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/fisheye_scene.json")
    pts_3d = triangulate_points_fisheye(points_1, points_2, k_arr[0], d_arr[0], r_arr[0], t_arr[0], k_arr[1], d_arr[1], r_arr[1], t_arr[1])
    scene = Scene()
    for r, t in zip(r_arr[:2], t_arr[:2]):
        scene.plot_camera(r, t)
    scene.plot_points(pts_3d)
    scene.show()


def test_project_points():
    points_1, *_ = load_points("tests/assets/extrinsic_calibration/data/points_1.json")
    points_2, *_ = load_points("tests/assets/extrinsic_calibration/data/points_2.json")
    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/standard_scene.json")
    pts_3d = triangulate_points(points_1, points_2, k_arr[0], d_arr[0], r_arr[0], t_arr[0], k_arr[1], d_arr[1], r_arr[1], t_arr[1])
    print(type(r_arr[0]))
    pts_2d = project_points(pts_3d, k_arr[0], d_arr[0], r_arr[0], t_arr[0])
    print(pts_2d)

def test_project_points_fisheye():
    points_1, *_ = load_points("tests/assets/extrinsic_calibration/data/points_1.json")
    points_2, *_ = load_points("tests/assets/extrinsic_calibration/data/points_2.json")
    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/fisheye_scene.json")
    pts_3d = triangulate_points_fisheye(points_1, points_2, k_arr[0], d_arr[0], r_arr[0], t_arr[0], k_arr[1], d_arr[1], r_arr[1], t_arr[1])
    pts_2d = project_points_fisheye(pts_3d, k_arr[0], d_arr[0], r_arr[0], t_arr[0])
    print(pts_2d)


def test_app_extract_corners_from_images():
    img_dir = "tests/assets/intrinsic_calibration/images"
    out_fpath = "tests/assets/intrinsic_calibration/data/points.json"
    board_shape = (8, 6)
    board_edge_len = 0.04
    extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len)

    img_dir = "tests/assets/extrinsic_calibration/images/1"
    out_fpath = "tests/assets/extrinsic_calibration/data/points_1.json"
    board_shape = (9, 6)
    board_edge_len = 0.088
    extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len, window_size=5)

    img_dir = "tests/assets/extrinsic_calibration/images/2"
    out_fpath = "tests/assets/extrinsic_calibration/data/points_2.json"
    board_shape = (9, 6)
    board_edge_len = 0.088
    extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len, window_size=5)


def test_app_plot_board():
    plot_corners("tests/assets/intrinsic_calibration/data/points.json")


def test_app_calibration():
    points_fpath = "tests/assets/intrinsic_calibration/data/points.json"
    out_fpath_std = "tests/assets/intrinsic_calibration/data/standard_camera.json"
    out_fpath_fish = "tests/assets/intrinsic_calibration/data/fisheye_camera.json"
    calibrate_fisheye_intrinsics(points_fpath, out_fpath_fish)
    calibrate_standard_intrinsics(points_fpath, out_fpath_std)


def test_app_plot_undistort():
    points_fpath = "tests/assets/intrinsic_calibration/data/points.json"
    camera_fpath_std = "tests/assets/intrinsic_calibration/data/standard_camera.json"
    camera_fpath_fish = "tests/assets/intrinsic_calibration/data/fisheye_camera.json"
    plot_points_standard_undistort(points_fpath, camera_fpath_std)
    plot_points_fisheye_undistort(points_fpath, camera_fpath_fish)


def test_app_pairwise_extrinsics():
    camera_fpaths = [
        "tests/assets/intrinsic_calibration/data/standard_camera.json",
        "tests/assets/intrinsic_calibration/data/standard_camera.json",
        "tests/assets/intrinsic_calibration/data/standard_camera.json"
    ]
    points_fpaths = [
        "tests/assets/extrinsic_calibration/data/points_1.json",
        "tests/assets/extrinsic_calibration/data/points_2.json",
        "tests/assets/extrinsic_calibration/data/points_1.json"
    ]
    out_fpath = "tests/assets/extrinsic_calibration/data/standard_scene.json"
    calibrate_standard_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath)
    camera_fpaths = [
        "tests/assets/intrinsic_calibration/data/fisheye_camera.json",
        "tests/assets/intrinsic_calibration/data/fisheye_camera.json",
        "tests/assets/intrinsic_calibration/data/fisheye_camera.json"
    ]
    out_fpath = "tests/assets/extrinsic_calibration/data/fisheye_scene.json"
    calibrate_fisheye_extrinsics_pairwise(camera_fpaths, points_fpaths,out_fpath)


def test_app_plot_scene():
    scene_fpath = "tests/assets/extrinsic_calibration/data/fisheye_scene.json"
    plot_scene(scene_fpath)


def test_prepare_data_for_bundle_adjustment():
    points_1, fnames_1, *_= load_points("tests/assets/extrinsic_calibration/data/points_1.json")
    points_2, fnames_2, board_shape, *_ = load_points("tests/assets/extrinsic_calibration/data/points_2.json")
    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/standard_scene.json")
    img_pts_arr = [points_1, points_2, points_1]
    fnames_arr = [fnames_1, fnames_2, fnames_1]
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr)
    print(points_2d.shape, points_3d.shape, point_3d_indices.shape, camera_indices.shape)

def test_cost_func():
    points_1, fnames_1, *_ = load_points("tests/assets/extrinsic_calibration/data/points_1.json")
    points_2, fnames_2, board_shape, *_ = load_points("tests/assets/extrinsic_calibration/data/points_2.json")
    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/standard_scene.json")
    img_pts_arr = [points_1, points_2, points_1]
    fnames_arr = [fnames_1, fnames_2, fnames_1]
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(img_pts_arr, fnames_arr,
                                                                                                            board_shape, k_arr,
                                                                                                            d_arr, r_arr, t_arr)

    n_points = len(points_3d)
    error = cost_func_points_only(points_3d.flatten(), n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d)
    print(error.shape)


def test_bundle_adjust_points_only():
    points_1, fnames_1, *_ = load_points("tests/assets/extrinsic_calibration/data/points_1.json")
    points_2, fnames_2, board_shape, *_ = load_points("tests/assets/extrinsic_calibration/data/points_2.json")
    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/standard_scene.json")
    img_pts_arr = [points_1, points_2, points_1]
    fnames_arr = [fnames_1, fnames_2, fnames_1]
    triangulate_func = triangulate_points
    project_func = project_points
    bundle_adjust_board_points_only(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func)

    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/fisheye_scene.json")
    triangulate_func = triangulate_points_fisheye
    project_func = project_points_fisheye
    bundle_adjust_board_points_only(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func)



def test_bundle_adjust_points_extrinsics():
    points_1, fnames_1, *_ = load_points("tests/assets/extrinsic_calibration/data/points_1.json")
    points_2, fnames_2, board_shape, *_ = load_points("tests/assets/extrinsic_calibration/data/points_2.json")
    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/standard_scene.json")
    img_pts_arr = [points_1, points_2, points_1]
    fnames_arr = [fnames_1, fnames_2, fnames_1]
    triangulate_func = triangulate_points
    project_func = project_points
    bundle_adjust_board_points_and_extrinsics(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func)

    k_arr, d_arr, r_arr, t_arr, _ = load_scene("tests/assets/extrinsic_calibration/data/fisheye_scene.json")
    triangulate_func = triangulate_points_fisheye
    project_func = project_points_fisheye
    bundle_adjust_board_points_and_extrinsics(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func)



def test_points_from_df():
    paths = [
        "/home/liam/SynologyDrive/Varsity/MSc/Software/kf_project_2/dlc/05_03_2019LilyRun1CAM1DLC_resnet50_CheetahOct14shuffle1_1030000.h5",
        "/home/liam/SynologyDrive/Varsity/MSc/Software/kf_project_2/dlc/05_03_2019LilyRun1CAM2DLC_resnet50_CheetahOct14shuffle1_1030000.h5",
        "/home/liam/SynologyDrive/Varsity/MSc/Software/kf_project_2/dlc/05_03_2019LilyRun1CAM3DLC_resnet50_CheetahOct14shuffle1_1030000.h5",
        "/home/liam/SynologyDrive/Varsity/MSc/Software/kf_project_2/dlc/05_03_2019LilyRun1CAM4DLC_resnet50_CheetahOct14shuffle1_1030000.h5",
        "/home/liam/SynologyDrive/Varsity/MSc/Software/kf_project_2/dlc/05_03_2019LilyRun1CAM5DLC_resnet50_CheetahOct14shuffle1_1030000.h5",
        "/home/liam/SynologyDrive/Varsity/MSc/Software/kf_project_2/dlc/05_03_2019LilyRun1CAM6DLC_resnet50_CheetahOct14shuffle1_1030000.h5",
    ]
    points_2d_df = create_dlc_points_2d_file(paths)

    k_arr, d_arr, r_arr, t_arr, _ = load_scene("/home/liam/SynologyDrive/Varsity/MSc/Software/kf_project_2/fisheye_scene.json")
    triangulate_func = triangulate_points_fisheye
    points_2d_filtered_df = points_2d_df[points_2d_df['likelihood']>0.5][points_2d_df['camera']<5]
    points_3d_df = get_pairwise_3d_points_from_df(points_2d_filtered_df, k_arr, d_arr, r_arr, t_arr, triangulate_func)

    points = points_3d_df[['x','y','z']].to_numpy()
    scene = Scene()
    for r, t in zip(r_arr, t_arr):
        scene.plot_camera(r, t)
    scene.plot_points(points)
    scene.plot_xy_grid()
    scene.show()

    # prepare_point_data_for_bundle_adjustment()





if __name__ == "__main__":
    # test_find_corners()
    # test_find_corners_images()
    # test_calibrate_camera()
    # test_calibrate_fisheye_camera()
    # test_create_undistort_functions()
    # test_create_undistort_fisheye_functions()
    # test_find_and_save_extrinsic_points()

    # test_app_extract_corners_from_images()

    # test_app_plot_board()
    # test_app_calibration()
    # test_app_plot_undistort()
    # test_app_pairwise_extrinsics()
    # test_app_plot_scene()
    # test_triangulate_points()
    # test_triangulate_points_fisheye()
    # test_project_points()
    # test_project_points_fisheye()
    # test_prepare_data_for_bundle_adjustment()
    # test_cost_func()
    # test_bundle_adjust()
    # test_bundle_adjust_points_extrinsics()

    test_points_from_df()