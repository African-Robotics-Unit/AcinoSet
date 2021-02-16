import os
import numpy as np
import json
from pprint import pprint

from .calib import calibrate_camera, \
    calibrate_fisheye_camera, \
    calibrate_pair_extrinsics, \
    calibrate_pair_extrinsics_fisheye, \
    create_undistort_img_function, \
    create_undistort_point_function, \
    create_undistort_fisheye_img_function, \
    create_undistort_fisheye_point_function, \
    fix_skew_scene, \
    adjust_extrinsics_manual_points, \
    calibrate_pairwise_extrinsics, \
    triangulate_points, \
    triangulate_points_fisheye, \
    project_points, \
    project_points_fisheye, \
    bundle_adjust_board_points_and_extrinsics, \
    bundle_adjust_board_points_and_extrinsics_with_manual_points, \
    bundle_adjust_board_points_and_extrinsics_with_only_manual_points

from .points import find_corners_images

from .utils import create_board_object_pts, \
    save_points, load_points, \
    save_camera, load_camera, \
    save_scene, load_scene, load_manual_points

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


def _calibrate_pairwise_extrinsics(
    calib_func,
    camera_fpaths, points_fpaths, out_fpath,
    dummy_scene_fpath=None
):
    # determine cam pairs to be used in calibration
    cams = np.array([int(list(filter(str.isdigit, fpath))[-1]) for fpath in points_fpaths])
    cam_pairs = np.array([cams[0:-1], cams[1:]]).T
    if np.where(cams > 4)[0].size > 0:
        # check if common frames exist between cams[0] and cams[-1]
        _, frames_1, *_ = load_points(points_fpaths[0])
        _, frames_2, *_ = load_points(points_fpaths[len(cams)-1])
        num_common_frames = len(set(frames_1).intersection(set(frames_2)))
        if num_common_frames > 2:
            cam_set_2_idxs = np.where(cams > 3)[0] # find any cam in 2nd cam set
            temp_arr = np.concatenate([[cams[0]], cams[cam_set_2_idxs[-1::-1]]])
            cam_pairs[cam_set_2_idxs-1] = np.array([temp_arr[0:-1], temp_arr[1:]]).T
    cams, cam_pairs = cams.tolist(), cam_pairs.tolist()

    # Load camera parameters
    k_arr = []
    d_arr = []
    cam_res = None
    for c in camera_fpaths:
        k1, d1, cam_res_1 = load_camera(c)
        k_arr.append(k1)
        d_arr.append(d1)
        if cam_res is None:
            cam_res = cam_res_1
        else:
            assert cam_res == cam_res_1

    # Load the image points
    img_pts_arr = []
    fnames_arr = []
    board_shape = None
    board_edge_len = None
    for p in points_fpaths:
        points_1, fnames_1, board_shape_1, board_edge_len_1, cam_res_1 = load_points(p)
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
            
    # Load the dummy scene
    with open(dummy_scene_fpath, 'rb') as f:
        dummy_scene_data = json.load(f)
    camera_params = dummy_scene_data['cameras']
    dummy_rs = []
    dummy_ts = []
    for camera in camera_params:
        dummy_rs.append(camera['r'])
        dummy_ts.append(camera['t'])
    dummy_scene_data = {
        'r': dummy_rs,
        't': dummy_ts,
    }

    r_arr, t_arr, incomplete_cams = calibrate_pairwise_extrinsics(
        calib_func, img_pts_arr, fnames_arr, k_arr, d_arr,
        cam_res, board_shape, board_edge_len,
        dummy_scene_data, cams, cam_pairs
    )

    if incomplete_cams:
        i = out_fpath.rfind(".")
        incomplete_fpath = out_fpath[:i]+'_before_corrections'+out_fpath[i:]
        save_scene(incomplete_fpath, k_arr, d_arr, r_arr, t_arr, cam_res)

        # this uses only the first incorrect cam - must be generalised to use all incorrect cams!!
        print("Running Least Squares using manual points to correct extrinsics...")
        i = out_fpath.rfind(os.path.sep)
        manual_pts_fpath = os.path.join(out_fpath[:i],"points","manual_points.json")
        try:
            img_pts_arr, *_ = load_manual_points(manual_pts_fpath)
            cam_idxs_to_correct = list(range(cams.index(incomplete_cams[0]),len(cams)))
            r_arr, t_arr = adjust_extrinsics_manual_points(calib_func, img_pts_arr, cam_idxs_to_correct, k_arr, d_arr, r_arr, t_arr) 
        except OSError as e:
            print("manual_points.json not found. Please rerun this calibration after obtaining manually-labelled points")
            raise OSError(e)

    r_arr, t_arr = fix_skew_scene(cams, r_arr, t_arr)
    save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, cam_res)

def calibrate_standard_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath,
                                           dummy_scene_fpath=None
                                          ):
    _calibrate_pairwise_extrinsics(calibrate_pair_extrinsics,
                                   camera_fpaths, points_fpaths, out_fpath,
                                   dummy_scene_fpath
                                  )


def calibrate_fisheye_extrinsics_pairwise(camera_fpaths, points_fpaths, out_fpath,
                                          dummy_scene_fpath=None
                                         ):
    _calibrate_pairwise_extrinsics(calibrate_pair_extrinsics_fisheye,
                                   camera_fpaths, points_fpaths, out_fpath,
                                   dummy_scene_fpath
                                  )

# this doesnt work!!
def _calibrate_pairwise_extrinsics_manual(calib_func, camera_fpaths, points_fpath, out_fpath):
    k_arr = []
    d_arr = []
    cam_res = None
    # Load camera parameters
    for c in camera_fpaths:
        k1, d1, cam_res_1 = load_camera(c)
        k_arr.append(k1)
        d_arr.append(d1)
        if cam_res is None:
            cam_res = cam_res_1
        else:
            assert cam_res == cam_res_1
    # Load the image points
    with open(points_fpath, 'r') as f:
        img_pts_arr = np.array(json.load(f)["points"])
        r_arr, t_arr = calibrate_pairwise_extrinsics_manual(calib_func, img_pts_arr, k_arr, d_arr, cam_res)
        save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, cam_res)

# Should this not be in calib.py?
def calibrate_pairwise_extrinsics_manual(calib_func, img_pts_arr, k_arr, d_arr, cam_res):
    # This code still needs to be updated so that its similar to calibrate_pairwise_extrinsics_fisheye
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
        print(img_pts_1)
        print(img_pts_1.shape)
        rms, r, t = calib_func(obj_pts, img_pts_1, img_pts_2, k1, d1, k2, d2, cam_res)
        # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
        # T is the world origin position in the camera coordinates.
        # The world position of the camera is C = -(R^-1)@T.
        # Similarly, the rotation of the camera in world coordinates is given by R^-1
        R2 = r @ R1
        T2 = r @ T1 + t
        # Update and add to list
        R1 = R2
        T1 = T2
        # Add camera extrinsic params
        r_arr.append(R1)
        t_arr.append(T1)
    return r_arr, t_arr


# def plot_scene(scene_fpath):
#     _, _, r_arr, t_arr, _ = load_scene(scene_fpath)
#     scene = Scene()
#     for r, t in zip(r_arr, t_arr):
#         scene.plot_camera(r, t)
#     scene.show()
#     return scene


def _sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, triangulate_func, project_func, camera_indices=None, only_manual_points=False):
    # load points
    img_pts_arr = []
    fnames_arr = []
    board_shape = None
    if camera_indices is None:
        camera_indices = range(len(points_fpaths))
    for i in camera_indices:
        p_fp = points_fpaths[i]
        points, fnames, board_shape, *_ = load_points(p_fp)
        img_pts_arr.append(points)
        fnames_arr.append(fnames)
    # load scene
    k_arr, d_arr, r_arr, t_arr, cam_res = load_scene(scene_fpath)
    assert len(k_arr) == len(img_pts_arr)
    if manual_points_fpath is not None:
        # load manually manual points
        manual_points, manual_fnames, *_ = load_manual_points(manual_points_fpath)
        # optimize
        if only_manual_points:
            print('bundle_adjust_board_points_and_extrinsics_with_only_manual_points')
            obj_pts, r_arr, t_arr, res = bundle_adjust_board_points_and_extrinsics_with_only_manual_points(
                manual_points, board_shape,
                k_arr, d_arr, r_arr, t_arr,
                triangulate_func, project_func
            )
        else:
            print('bundle_adjust_board_points_and_extrinsics_with_manual_points')
            obj_pts, r_arr, t_arr, res = bundle_adjust_board_points_and_extrinsics_with_manual_points(
                img_pts_arr, fnames_arr,
                manual_points, board_shape,
                k_arr, d_arr, r_arr, t_arr,
                triangulate_func, project_func
            )
    else:
        print('bundle_adjust_board_points_and_extrinsics')
        obj_pts, r_arr, t_arr, res = bundle_adjust_board_points_and_extrinsics(
            img_pts_arr, fnames_arr,
            board_shape,
            k_arr, d_arr, r_arr, t_arr,
            triangulate_func, project_func
        )
    save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, cam_res)
    return res

def sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath):
    triangulate_func = triangulate_points
    project_func = project_points
    return _sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, triangulate_func, project_func)

def sba_board_points_fisheye(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, camera_indices=None, only_manual_points=False):
    triangulate_func = triangulate_points_fisheye
    project_func = project_points_fisheye
    return _sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, triangulate_func, project_func, camera_indices, only_manual_points)
