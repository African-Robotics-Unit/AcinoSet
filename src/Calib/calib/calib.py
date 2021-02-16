import numpy as np
from typing import Tuple, Union
from nptyping import Array
import cv2
from .utils import create_board_object_pts
from .points import common_image_points
from .misc import redescending_loss, global_positions, rotation_matrix_from_vectors, rot_z
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import time
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

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
    assert len(img_pts) >= 4, "Need at least 4 vaild frames to perform calibration."
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
    """ Performs Least Squares Minimization to correct the pose of misaligned cameras
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
        calib_type = ''

    if type(cam_idxs_to_correct) is int:
        cam_idxs_to_correct = [cam_idxs_to_correct]
        msg = f"Cam with index {cam_idxs_to_correct} is to have its pose corrected."
    else:
        msg = f"Cams with indices {cam_idxs_to_correct} are to have their poses corrected"

    n_cams = len(k_arr)
    assert n_cams == img_pts_arr.shape[1], "Number of cams in intrinsic file differs from number of cams in manual points file"

    cam_pairs = []
    for i in cam_idxs_to_correct:
        cam_pairs.append([(i-1) % n_cams, i])
        cam_pairs.append([i, (i+1) % n_cams])
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


def calibrate_pairwise_extrinsics(calib_func, img_pts_arr, fnames_arr, k_arr, d_arr,
                                     cam_res, board_shape, board_edge_len,
                                     dummy_scene_data, cams, cam_pairs):
    # calib_func is one of 'calibrate_pair_extrinsics' or 'calibrate_pair_extrinsics_fisheye'
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
            print("{:.5f} pixels".format(rms))
            # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
            # T is the world origin position in the camera coordinates.
            # The world position of the camera is C = -(R^-1)@T.
            # Similarly, the rotation of the camera in world coordinates is given by R^-1
            r_arr[j] = r @ r_arr[i] # matrix product
            t_arr[j] = r @ t_arr[i] + t

    print("\nDone!")
    return r_arr, t_arr, incomplete_cams


def create_bundle_adjustment_jacobian_sparsity_matrix(n_cameras, n_params_per_camera, camera_indices, n_points, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * n_params_per_camera + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(n_params_per_camera):
        A[2 * i, camera_indices * n_params_per_camera + s] = 1
        A[2 * i + 1, camera_indices * n_params_per_camera + s] = 1
    for s in range(3):
        A[2 * i, n_cameras * n_params_per_camera + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * n_params_per_camera + point_indices * 3 + s] = 1
    return A


def prepare_calib_board_data_for_bundle_adjustment(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func):
    # Find all images names
    n_cam = len(img_pts_arr)
    fname_set = set()
    for fnames in fnames_arr:
        fname_set.update(fnames)
    fname_dict = dict.fromkeys(fname_set, 0)
    # Remove fnames not seen by more than 1 camera
    # first count how many cameras have image with same name,
    # then remove those seen by fewer than 2 cameras
    for cam_idx in range(n_cam):
        for k, v in fname_dict.items():
            if k in fnames_arr[cam_idx]:
                fname_dict[k] = v+1
    items = dict(fname_dict)
    for k,v in items.items():
        if v < 2:
            fname_dict.pop(k)
    # Initialize array
    points_3d = []
    point_3d_indices = []
    points_2d = []
    camera_indices = []
    # Get corresponding points between images
    point_idx_counter = 0
    points_per_image = board_shape[0] * board_shape[1]
    for fname in fname_dict:
        triangulation_point_indices = []
        triangulation_camera_indices = []
        for cam_idx in range(n_cam):
            new_point_3d_indices = range(point_idx_counter, point_idx_counter+points_per_image)
            if fname in fnames_arr[cam_idx]:
                f_idx = fnames_arr[cam_idx].index(fname)
                triangulation_point_indices.append(f_idx)
                triangulation_camera_indices.append(cam_idx)
                points_2d.extend(np.array(img_pts_arr[cam_idx][f_idx]).reshape(points_per_image, 2))
                point_3d_indices.extend(new_point_3d_indices)
                camera_indices.extend([cam_idx]*points_per_image)
        # Use the first two cameras to get the initial estimate
        a_pt = triangulation_point_indices[0]
        b_pt = triangulation_point_indices[1]
        a = triangulation_camera_indices[0]
        b = triangulation_camera_indices[1]
        point_3d_est = triangulate_func(
            img_pts_arr[a][a_pt], img_pts_arr[b][b_pt],
            k_arr[a], d_arr[a], r_arr[a], t_arr[a],
            k_arr[b], d_arr[b], r_arr[b], t_arr[b]
        )
        points_3d.extend(point_3d_est)
        point_idx_counter += points_per_image
    # Convert to numpy arrays
    points_2d = np.array(points_2d, dtype=np.float32)
    points_3d = np.array(points_3d, dtype=np.float32)
    point_3d_indices = np.array(point_3d_indices, dtype=np.int)
    camera_indices = np.array(camera_indices, dtype=np.int)
    return points_2d, points_3d, point_3d_indices, camera_indices


def prepare_manual_points_for_bundle_adjustment(img_pts_arr, k_arr, d_arr, r_arr, t_arr, triangulate_func):
    # img_pts_arr shape: (n_points, n_cameras, 2)
    pts = img_pts_arr.swapaxes(0,1) # should now have shape (n_cameras, n_points, 2)
    n_cam = pts.shape[0]
    n_pts = pts.shape[1]

    points_2d = []
    point_3d_indices = []
    camera_indices = []
    points_3d = []

    pt_3d_idx = 0

    for i in range(n_pts):
        points_2d_temp = []
        camera_indices_temp = []
        for cam_idx in range(n_cam):
            if not np.isnan(pts[cam_idx, i]).any():
                points_2d_temp.append(pts[cam_idx, i])
                camera_indices_temp.append(cam_idx)
        seen_by = len(points_2d_temp)
        if seen_by > 1:
            points_2d.extend(points_2d_temp)
            camera_indices.extend(camera_indices_temp)
            point_3d_indices.extend([pt_3d_idx]*seen_by)
            #Do triangulation
            a = camera_indices_temp[0]
            b = camera_indices_temp[1]
            point_3d_est = triangulate_func(
                points_2d_temp[0], points_2d_temp[1],
                k_arr[a], d_arr[a], r_arr[a], t_arr[a],
                k_arr[b], d_arr[b], r_arr[b], t_arr[b]
            )
            points_3d.extend(point_3d_est)
            pt_3d_idx += 1

    points_2d = np.array(points_2d, dtype=np.float32)
    point_3d_indices = np.array(point_3d_indices, dtype=np.int)
    camera_indices = np.array(camera_indices, dtype=np.int)
    points_3d = np.array(points_3d, dtype=np.float32)

    return points_2d, points_3d, point_3d_indices, camera_indices


def params_to_points_only(params, n_points):
    obj_pts = params.reshape((n_points, 3))
    return obj_pts


def cost_func_points_only(params, n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d, project_func):
    obj_pts = params_to_points_only(params, n_points)
    reprojected_pts = np.array([project_func(obj_pts[i], k_arr[j], d_arr[j], r_arr[j], t_arr[j])[0] for i,j in zip(point_3d_indices, camera_indices)])
    error = (reprojected_pts - points_2d).ravel()
    return error


def bundle_adjust_board_points_only(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func):
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(
        img_pts_arr, fnames_arr,
        board_shape, k_arr,
        d_arr, r_arr, t_arr, triangulate_func
    )
    obj_pts, residuals = bundle_adjust_points_only(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func)
    return obj_pts, residuals


def bundle_adjust_points_only(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func, f_scale=50):
    n_points = len(points_3d)
    n_cameras = len(k_arr)
    n_params_per_camera = 0
    x0 = points_3d.flatten()
    f0 = cost_func_points_only(x0, n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d, project_func)
    A = create_bundle_adjustment_jacobian_sparsity_matrix(n_cameras, n_params_per_camera, camera_indices, n_points, point_3d_indices)
    t0 = time.time()
    res = least_squares(cost_func_points_only, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-15, method='trf', loss='cauchy', f_scale=f_scale,
                        args=(n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d, project_func), max_nfev=500)
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    residuals = dict(before=f0, after=res.fun)
    obj_pts = params_to_points_only(res.x, n_points)
    return obj_pts, residuals



def params_to_points_extrinsics(params, n_cameras, n_points):
    r_end_idx = n_cameras*3
    t_end_idx = r_end_idx + n_cameras*3
    r_vecs = params[:r_end_idx].reshape((n_cameras,3))
    r_arr = np.array([cv2.Rodrigues(r)[0] for r in r_vecs], dtype=np.float64)
    t_arr = params[r_end_idx:t_end_idx].reshape((n_cameras,3, 1))
    obj_pts = params[t_end_idx:].reshape((n_points, 3))
    return obj_pts, r_arr, t_arr


def cost_func_points_extrinsics(params, n_cameras, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d, project_func):
    obj_pts, r_arr, t_arr = params_to_points_extrinsics(params, n_cameras, n_points)
    reprojected_pts = np.array([project_func(obj_pts[i], k_arr[j], d_arr[j], r_arr[j], t_arr[j])[0] for i,j in zip(point_3d_indices, camera_indices)])
    error = (reprojected_pts - points_2d).ravel()
    return error


def bundle_adjust_board_points_and_extrinsics(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func):
    # (n_points, [x,y]), (n_3d_points, [x,y,z]), (3d_point_indics), (camera_indices)
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(
        img_pts_arr, fnames_arr,
        board_shape,
        k_arr, d_arr, r_arr, t_arr,
        triangulate_func
    )

    return bundle_adjust_points_and_extrinsics(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func)

def bundle_adjust_board_points_and_extrinsics_with_manual_points(img_pts_arr, fnames_arr, manual_points, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func):
    # (n_points, [x,y]), (n_3d_points, [x,y,z]), (3d_point_indics), (camera_indices)
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(
        img_pts_arr, fnames_arr,
        board_shape,
        k_arr, d_arr, r_arr, t_arr,
        triangulate_func
    )
    manual_points_2d, manual_points_3d, manual_point_3d_indices, manual_camera_indices = prepare_manual_points_for_bundle_adjustment(
        manual_points,
        k_arr, d_arr, r_arr, t_arr,
        triangulate_func
    )

    # combine both data
    points_2d = np.append(points_2d, manual_points_2d, axis=0)
    points_3d = np.append(points_3d, manual_points_3d, axis=0)
    point_3d_indices = np.append(point_3d_indices, manual_point_3d_indices + point_3d_indices.max(), axis=0)
    camera_indices = np.append(camera_indices, manual_camera_indices, axis=0)

    return bundle_adjust_points_and_extrinsics(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func)

def bundle_adjust_board_points_and_extrinsics_with_only_manual_points(manual_points, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func):
    # (n_points, [x,y]), (n_3d_points, [x,y,z]), (3d_point_indics), (camera_indices)
    manual_points_2d, manual_points_3d, manual_point_3d_indices, manual_camera_indices = prepare_manual_points_for_bundle_adjustment(
        manual_points,
        k_arr, d_arr, r_arr, t_arr,
        triangulate_func
    )

    return bundle_adjust_points_and_extrinsics(manual_points_2d, manual_points_3d, manual_point_3d_indices, manual_camera_indices, k_arr, d_arr, r_arr, t_arr, project_func)


def bundle_adjust_points_and_extrinsics(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func):
    n_points = len(points_3d)
    n_cameras = len(k_arr)
    n_params_per_camera = 6
    r_vecs = np.array([cv2.Rodrigues(r)[0] for r in r_arr], dtype=np.float64).flatten()
    t_vecs = t_arr.flatten()
    x0 = np.concatenate([r_vecs, t_vecs, points_3d.flatten()])
    f0 = cost_func_points_extrinsics(x0, n_cameras, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d,
                                     project_func)
    A = create_bundle_adjustment_jacobian_sparsity_matrix(n_cameras, n_params_per_camera, camera_indices, n_points,
                                                          point_3d_indices)
    t0 = time.time()
    res = least_squares(cost_func_points_extrinsics, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10,
                        method='trf', loss='cauchy',
                        args=(n_cameras, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d, project_func),
                        max_nfev=1000)
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    obj_pts, r_arr, t_arr = params_to_points_extrinsics(res.x, n_cameras, n_points)
    residuals = dict(before=f0, after=res.fun)
    return obj_pts, r_arr, t_arr, residuals



def get_pairwise_3d_points_from_df(points_2d_df, k_arr, d_arr, r_arr, t_arr, triangulate_func):
    n_cameras = len(k_arr)
    camera_pairs = list([(i, i+1) for i in range(n_cameras-1)])
    df_pairs = pd.DataFrame(columns=['x','y','z'])
    #get pairwise estimates
    for (cam_a, cam_b) in camera_pairs:
        d0 = points_2d_df[points_2d_df['camera']==cam_a]
        d1 = points_2d_df[points_2d_df['camera']==cam_b]
        intersection_df = d0.merge(d1, how='inner', on=['frame','marker'], suffixes=('_a', '_b'))
        if intersection_df.shape[0] > 0:
            print(f"Found {intersection_df.shape[0]} pairwise points between camera {cam_a} and {cam_b}")
            cam_a_points = np.array(intersection_df[['x_a','y_a']], dtype=np.float).reshape((-1,1,2))
            cam_b_points = np.array(intersection_df[['x_b','y_b']], dtype=np.float).reshape((-1,1,2))
            points_3d = triangulate_func(cam_a_points, cam_b_points,
                                            k_arr[cam_a], d_arr[cam_a], r_arr[cam_a], t_arr[cam_a],
                                            k_arr[cam_b], d_arr[cam_b], r_arr[cam_b], t_arr[cam_b])
            intersection_df['x'] = points_3d[:, 0]
            intersection_df['y'] = points_3d[:, 1]
            intersection_df['z'] = points_3d[:, 2]
            df_pairs = pd.concat([df_pairs, intersection_df], ignore_index=True, join='outer', sort=False)
        else:
            print(f"No pairwise points between camera {cam_a} and {cam_b}")

    points_3d_df = df_pairs[['frame', 'marker', 'x','y','z']].groupby(['frame','marker']).mean().reset_index()
    return points_3d_df



# def prepare_point_data_for_bundle_adjustment(points_2d_df, k_arr, d_arr, r_arr, t_arr, triangulate_func):
#
#     return points_2d, points_3d, point_3d_indices, camera_indices
