import numpy as np
from typing import Tuple, Union
from nptyping import Array
import cv2
from .utils import create_board_object_pts, save_scene, \
    load_camera, load_points, load_manual_points
from .points import common_image_points
from .misc import redescending_loss, global_positions, rotation_matrix_from_vectors, rot_z
from scipy.optimize import least_squares
import json


# ========== STANDARD CAMERA MODEL ==========

def calibrate_camera(obj_pts: Array[np.float32, ..., 3], img_pts: Array[np.float32, ..., ..., 2],
                     cam_res: Tuple[int, int]) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]:
    assert len(img_pts)>=4, "Need at least 4 vaild frames to perform calibration."
    obj_pts = np.repeat(obj_pts[np.newaxis, :, :], img_pts.shape[0], axis=0).reshape((img_pts.shape[0], -1, 1, 3))
    img_pts = img_pts.reshape((img_pts.shape[0], -1, 1, 2))
    flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT
    ret, k, d, r, t = cv2.calibrateCamera(obj_pts, img_pts, cam_res, None, None, flags=flags)
    if ret:
        return k, d, r, t
    return None


def create_undistort_point_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...]):
    def undistort_points(pts: Array[np.float32, ..., 2]):
        pts = pts.reshape((-1, 1, 2))
        undistorted = cv2.undistortPoints(pts, k, d, P=k)
        return undistorted.reshape((-1,2))
    return undistort_points


def create_undistort_img_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...], cam_res):
    map_x, map_y = cv2.initUndistortRectifyMap(k, d, None, k, cam_res, cv2.CV_32FC1)
    def undistort_image(img):
        dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        return dst
    return undistort_image


def calibrate_pair_extrinsics(obj_pts, img_pts_1, img_pts_2, k1, d1, k2, d2, cam_res):
    flags = cv2.CALIB_FIX_INTRINSIC
    term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-5)
    obj_pts = np.repeat(obj_pts[np.newaxis, :, :], img_pts_1.shape[0], axis=0).reshape((img_pts_1.shape[0], -1, 1, 3))
    img_pts_1 = img_pts_1.reshape((img_pts_1.shape[0], img_pts_1.shape[1]*img_pts_1.shape[2], 1, 2))
    img_pts_2 = img_pts_2.reshape((img_pts_2.shape[0], img_pts_2.shape[1]*img_pts_2.shape[2], 1, 2))
    rms, *_, r, t, _, _ = cv2.stereoCalibrate(obj_pts, img_pts_1, img_pts_2, k1, d1,k2, d2,
                                              cam_res, flags=flags, criteria=term_crit)
    return rms, r, t


def triangulate_points(img_pts_1, img_pts_2, k1, d1, r1, t1, k2, d2, r2, t2):
    pts_1 = img_pts_1.reshape((-1,1,2))
    pts_2 = img_pts_2.reshape((-1, 1, 2))
    pts_1 = cv2.undistortPoints(pts_1, k1, d1)
    pts_2 = cv2.undistortPoints(pts_2, k2, d2)
    p1 = np.hstack((r1, t1))
    p2 = np.hstack((r2, t2))
    pts_4d = cv2.triangulatePoints(p1, p2, pts_1, pts_2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


def project_points(obj_pts, k, d, r, t):
    pts =  cv2.projectPoints(obj_pts, r, t, k, d)[0].reshape((-1, 2))
    return pts


# ========== FISHEYE CAMERA MODEL ==========

def calibrate_fisheye_camera(obj_pts: Array[np.float32, ..., 3], img_pts: Array[np.float32, ..., ..., 2],
                             cam_res: Tuple[int, int]) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                    np.ndarray, Array[np.float32, ..., ..., 2]], None]:
    assert len(img_pts) >= 4, "Need at least 4 valid frames to perform calibration."
    obj_pts_new = np.repeat(obj_pts[np.newaxis, :, :], img_pts.shape[0], axis=0).reshape((img_pts.shape[0], -1, 1, 3))
    img_pts_new = img_pts.reshape((img_pts.shape[0], -1, 1, 2))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)
    try:
        ret, k, d, r, t = cv2.fisheye.calibrate(obj_pts_new, img_pts_new, cam_res, None, None, None, None, flags,
                                                criteria)
        if ret:
            return k, d, r, t, img_pts, ret
    except Exception as e:
        if "CALIB_CHECK_COND" in str(e):
            idx = int(str(e)[str(e).find("input array ") + 12:].split(" ")[0])
            print(f"Image points at index {idx} caused an ill-conditioned matrix. Removing from array...")
            img_pts = img_pts[np.arange(len(img_pts)) != idx]
            return calibrate_fisheye_camera(obj_pts, img_pts, cam_res)


def create_undistort_fisheye_point_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...]):
    def undistort_points(pts: Array[np.float32, ..., 2]):
        pts = pts.reshape((-1, 1, 2))
        undistorted = cv2.fisheye.undistortPoints(pts, k, d, P=k)
        return undistorted.reshape((-1,2))
    return undistort_points


def create_undistort_fisheye_img_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...], cam_res):
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, cam_res, cv2.CV_32FC1)
    def undistort_image(img):
        dst = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return dst
    return undistort_image


def calibrate_pair_extrinsics_fisheye(obj_pts, img_pts_1, img_pts_2, k1, d1, k2, d2, cam_res):
    flags = cv2.fisheye.CALIB_FIX_INTRINSIC
    term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    obj_pts = np.repeat(obj_pts[np.newaxis, :, :], img_pts_1.shape[0], axis=0).reshape((img_pts_1.shape[0], 1, -1, 3))
    img_pts_1 = img_pts_1.reshape((img_pts_1.shape[0], 1, img_pts_1.shape[1]*img_pts_1.shape[2], 2))
    img_pts_2 = img_pts_2.reshape((img_pts_2.shape[0], 1, img_pts_2.shape[1]*img_pts_2.shape[2], 2))
    rms, *_, r, t = cv2.fisheye.stereoCalibrate(obj_pts, img_pts_1, img_pts_2, k1, d1, k2, d2, cam_res,
                                                flags=flags, criteria=term_crit)
    return rms, r, t


def triangulate_points_fisheye(img_pts_1, img_pts_2, k1, d1, r1, t1, k2, d2, r2, t2):
    pts_1 = img_pts_1.reshape((-1, 1, 2))
    pts_2 = img_pts_2.reshape((-1, 1, 2))
    pts_1 = cv2.fisheye.undistortPoints(pts_1, k1, d1)
    pts_2 = cv2.fisheye.undistortPoints(pts_2, k2, d2)
    p1 = np.hstack((r1, t1))
    p2 = np.hstack((r2, t2))
    pts_4d = cv2.triangulatePoints(p1, p2, pts_1, pts_2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


def project_points_fisheye(obj_pts, k, d, r, t):
    obj_pts_reshaped = obj_pts.reshape((-1, 1, 3))
    r_vec = cv2.Rodrigues(r)[0]
    pts =  cv2.fisheye.projectPoints(obj_pts_reshaped, r_vec, t, k, d)[0].reshape((-1, 2))
    return pts


# ========== ESTIMATION ALGORITHMS ==========

def calibrate_pairwise_extrinsics(calib_func, img_pts_arr, fnames_arr, k_arr, d_arr,
                                     cam_res, board_shape, board_edge_len,
                                     dummy_scene_data, cams, cam_pairs=None):
    # calib_func is one of 'calibrate_pair_extrinsics' or 'calibrate_pair_extrinsics_fisheye'
    cam_pairs = [[i, j] for i, j in zip(cams[0:-1], cams[1:]) if not cam_pairs]
    r_arr = [[] for _ in cams]
    t_arr = r_arr.copy()
    # Set camera 1's initial position and rotation
    r_arr[0] = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=np.float32)
    t_arr[0] = np.array([[0, 0, 0]], dtype=np.float32).T
    # Get relative pairwise transformations between subsequent cameras
    print(f"Pairwise calibration using cam pairs {cam_pairs}\n")
    print("camera pair", 'common frames', 'RMS reprojection error', sep='\t')
    incomplete_cams = []
    for cam_a, cam_b in cam_pairs:
        print(f"{cam_a} & {cam_b}", end='\t'*2)
        i, j = cams.index(cam_a), cams.index(cam_b)
        # Extract common points between cameras into img_pts 1 & 2
        img_pts_1, img_pts_2, _ = common_image_points(img_pts_arr[i], fnames_arr[i], img_pts_arr[j], fnames_arr[j])
        print(len(img_pts_1), end='\t'*2)
        if not len(img_pts_1):
            r_arr[j] = np.array(dummy_scene_data['r'][cam_b-1])
            t_arr[j] = np.array(dummy_scene_data['t'][cam_b-1])
            print(f"\nInstead, R[{cam_b-1}] and T[{cam_b-1}] from dummy_scene.json were used for cam {cam_b}")
            incomplete_cams.append(cam_b)
        else:
            # Create object points
            obj_pts = create_board_object_pts(board_shape, board_edge_len)
            rms, r, t = calib_func(obj_pts, img_pts_1, img_pts_2, k_arr[i], d_arr[i], k_arr[j], d_arr[j], cam_res)
            print(f"{rms:.5f} pixels")
            # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
            # T is the world origin position in the camera coordinates.
            # The world position of the camera is C = -(R^-1)@T.
            # Similarly, the rotation of the camera in world coordinates is given by R^-1
            r_arr[j] = r @ r_arr[i] # matrix product
            t_arr[j] = r @ t_arr[i] + t

    print("\nDone!")
    return r_arr, t_arr, incomplete_cams


# ========== EXTRINSIC REFINEMENT ALGORITHMS ==========

def fix_skew_scene(cams, r_arr, t_arr, ave_cam_height=0.5):
    # seperate cams list into side 1 & 2
    cam_sets = [list(filter(lambda x: x<4, cams)), list(filter(lambda x: x>3, cams))]
    cam_sets_len = np.array([len(cam_sets[0]),len(cam_sets[1])])
     # check if one of the cam sets have more than 1 cam
    idx = np.where(cam_sets_len > 1)[0]
    if len(idx):
        # get cams on the one side of the scene that forms a line
        idxs = [cams.index(i) for i in cam_sets[idx[0]]]
        positions = global_positions(r_arr, t_arr)[idxs].reshape(-1, 3)
        line_vec, *_ = np.linalg.svd(positions - positions.mean(axis=0))[-1] # best fit line vector
        line_vec *= -1 if line_vec[0] < 0 else 1 # ensure positive x direction
        R = rotation_matrix_from_vectors(np.array([[1, 0, 0]]).T, line_vec)
        r_arr = [r @ R for r in r_arr] # align cams with x direction

    # check if cams lie on a plane
    z_vec = np.array([[0, 0, 1]]).T
    if len(cams)>2 and cam_sets_len.all():
        positions = global_positions(r_arr, t_arr).reshape(-1, 3) # A
        *_, plane_normal = np.linalg.svd(positions - positions.mean(axis=0))[-1] # best fit plane
        plane_normal *= -1 if plane_normal[-1] < 0 else 1 # ensure positive z direction
        R = rotation_matrix_from_vectors(z_vec, plane_normal)
        r_arr = [r @ R for r in r_arr] # align plane of cam positions with xy plane

    # place cams at a height above ground
    t_arr = [t - ave_cam_height * r @ z_vec for r, t in zip(r_arr, t_arr)]
    return r_arr, t_arr


def adjust_extrinsics_manual_points(calib_func, img_pts_arr, cam_idxs_to_correct, k_arr, d_arr, r_arr, t_arr):
    """Performs Least Squares Minimization to correct the pose of misaligned cameras.
    :param scene_fpath: path to scene*.json file that holds the scene's extrinsics
    :param manual_pts_fpath: path to file that hold the manually-obtained image points
    :param cam_idxs_to_correct: the index/indices of the camera(s) whose pose(s) is/are to be corrected.
    This variable must be either single integer or a 1D list of integers. Here, the cam's index corresponds
    to its index in the scene*.json file.
    :return extrinsic params k_arr, d_arr, r_arr, t_arr, cam_res
    """
    def residual_arr(R_t):
        R = cv2.Rodrigues(R_t[0:3])[0] # convert fromm rodrigues vector to rotation matrix
        t = R_t[3:6].reshape((3,1))

        # for each skew cam pair
        all_skew_3d_pts = []
        for a, b in cam_pairs:
            cam_a_params = [k_arr[a], d_arr[a]]
            cam_a_params += [r_arr[a] @ R.T, t_arr[a] - r_arr[a] @ t] if a in cam_idxs_to_correct else [r_arr[a], t_arr[a]]
            cam_b_params = [k_arr[b], d_arr[b]]
            cam_b_params += [r_arr[b] @ R.T, t_arr[b] - r_arr[b] @ t] if b in cam_idxs_to_correct else [r_arr[b], t_arr[b]]

            skew_3d_pts = triangulate_func(
                np.array(img_pts_arr[:, a]), np.array(img_pts_arr[:, b]), 
                *cam_a_params, *cam_b_params
            )

            all_skew_3d_pts.append(skew_3d_pts)

        # reproject skew 3d pts into cam i's view and get array of reprojection errors
        all_cams_reprojection_err = []
        for i in range(n_cams):
            cam_i_params = [k_arr[i], d_arr[i]]
            cam_i_params += [r_arr[i] @ R.T, t_arr[i] - r_arr[i] @ t] if i in cam_idxs_to_correct else [r_arr[i], t_arr[i]]

            for skew_3d_pts in all_skew_3d_pts:
                reprojected_pts = project_func(skew_3d_pts, *cam_i_params)
                reprojection_errs = img_pts_arr[:, i] - reprojected_pts

                all_cams_reprojection_err.append([redescending_loss(e, 3, 10, 20) for e in reprojection_errs])

        return np.array(all_cams_reprojection_err).flatten()

    if calib_func == calibrate_pair_extrinsics_fisheye:
        triangulate_func = triangulate_points_fisheye
        project_func = project_points_fisheye
        calib_type = 'fisheye'
    else:
        triangulate_func = triangulate_points
        project_func = project_points
        calib_type = 'standard'

    if type(cam_idxs_to_correct) is int:
        cam_idxs_to_correct = [cam_idxs_to_correct]
        msg = f"Cam with index {cam_idxs_to_correct} is to have its pose corrected."
    else:
        msg = f"Cams with indices {cam_idxs_to_correct} are to have their poses corrected"

    n_cams = len(k_arr)
    assert n_cams == img_pts_arr.shape[1], "Number of cams in intrinsic file differs from number of cams in manual points file"

    cam_pairs = []
    for i in cam_idxs_to_correct:
        cam_pairs.append(sorted([(i-1) % n_cams, i]))
        cam_pairs.append(sorted([i, (i+1) % n_cams]))
    cam_pairs = np.unique(cam_pairs,axis=0).tolist() # remove duplicates

    print(msg, "\nMinimizing error for", calib_type, "cam pairs with indices", cam_pairs, "...\n")

    R0, t0 = np.identity(3), np.zeros(3)
    R0_t0 = np.concatenate([cv2.Rodrigues(R0)[0].flatten(), t0]) # pack initial R_t

    res = least_squares(residual_arr, R0_t0) # loss = linear (default)
    print(res.message, f"success: {res.success}", f"func evals: {res.nfev}", f"cost: {res.cost}\n", sep="\n")

    R = cv2.Rodrigues(res.x[0:3])[0] # convert fromm rodrigues vector to rotation matrix
    t = res.x[3:6].reshape((3,1))

    for cam_idx in cam_idxs_to_correct:
        # adjust skew cam's extrinsics by a r and t in world frame
        t_arr[cam_idx] -= r_arr[cam_idx] @ t # t_f = t_i - r_i @ t
        r_arr[cam_idx] = r_arr[cam_idx] @ R.T # r_f = r_i @ r.T # r_arr[cam_idx] @= R.T not yet supported

    return r_arr, t_arr


# ==========  FRONT-END FUNCTION MANAGER  ==========

def _calibrate_pairwise_extrinsics(
    calib_func,
    camera_fpaths, points_fpaths, out_fpath,
    dummy_scene_fpath=None, manual_points_fpath=None
):
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
            np.testing.assert_equal(board_edge_len, board_edge_len_1) # handles nan case
            
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
    
    # determine cam pairs to be used in calibration
    cams = np.array([int(list(filter(str.isdigit, fpath))[-1]) for fpath in points_fpaths])
    cam_pairs = None
    if np.where(cams > 4)[0].size > 0:
        # check if common frames exist between cams[0] and cams[-1]
        _, frames_1, *_ = load_points(points_fpaths[0])
        _, frames_2, *_ = load_points(points_fpaths[len(cams)-1])
        num_common_frames = len(set(frames_1).intersection(set(frames_2)))
        if num_common_frames > 2:
            cam_pairs =  np.array([cams[0:-1], cams[1:]]).T
            cam_set_2_idxs = np.where(cams > 3)[0] # find any cam in 2nd cam set
            temp_arr = np.concatenate([[cams[0]], cams[cam_set_2_idxs[-1::-1]]])
            cam_pairs[cam_set_2_idxs-1] = np.array([temp_arr[0:-1], temp_arr[1:]]).T
            cam_pairs = cam_pairs.tolist()
    cams = cams.tolist()

    r_arr, t_arr, incomplete_cams = calibrate_pairwise_extrinsics(
        calib_func, img_pts_arr, fnames_arr, k_arr, d_arr,
        cam_res, board_shape, board_edge_len,
        dummy_scene_data, cams, cam_pairs=cam_pairs
    )

    if incomplete_cams:
        incomplete_fpath = out_fpath.replace(".json", "_before_corrections.json")
        save_scene(incomplete_fpath, k_arr, d_arr, r_arr, t_arr, cam_res)

        # this uses only the first incorrect cam - must be generalised to use all incorrect cams!!
        print("Running Least Squares using manual points to correct extrinsics...")
        try:
            img_pts_arr, *_ = load_manual_points(manual_points_fpath)
        except Exception as e:
            print("\nmanual_points.json not found. Please rerun this calibration after obtaining manually-labelled points")
            raise Exception(e)
            
        cam_idxs_to_correct = list(range(cams.index(incomplete_cams[0]),len(cams)))
        r_arr, t_arr = adjust_extrinsics_manual_points(calib_func, img_pts_arr, cam_idxs_to_correct, k_arr, d_arr, r_arr, t_arr)

    r_arr, t_arr = fix_skew_scene(cams, r_arr, t_arr)
    save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, cam_res)
