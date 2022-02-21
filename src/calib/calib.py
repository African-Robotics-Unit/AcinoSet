import numpy as np
from typing import Tuple, Union
from nptyping import Array
import cv2
from .utils import create_board_object_pts
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import time
import pandas as pd
import matplotlib.pyplot as plt

# ========== STANDARD CAMERA MODEL ==========
def calibrate_camera(obj_pts: Array[np.float32, ..., 3], img_pts: Array[np.float32, ..., ..., 2],
                     camera_resolution: Tuple[int, int]) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]:
    assert len(img_pts)>=4, "Need at least 4 vaild frames to perform calibration."
    obj_pts = np.repeat(obj_pts[np.newaxis, :, :], img_pts.shape[0], axis=0).reshape((img_pts.shape[0], -1, 1, 3))
    img_pts = img_pts.reshape((img_pts.shape[0], -1, 1, 2))
    flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT
    ret, k, d, r, t = cv2.calibrateCamera(obj_pts, img_pts, camera_resolution, None, None, flags=flags)
    if ret:
        return k, d, r, t
    return None


def create_undistort_point_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...]):
    def undistort_points(pts: Array[np.float32, ..., 2]):
        pts = pts.reshape((-1, 1, 2))
        undistorted = cv2.undistortPoints(pts, k, d, P=k)
        return undistorted.reshape((-1,2))
    return undistort_points


def create_undistort_img_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...], camera_resolution):
    map_x, map_y = cv2.initUndistortRectifyMap(k, d, None, k, camera_resolution, cv2.CV_32FC1)
    def undistort_image(img):
        dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        return dst
    return undistort_image


def calibrate_pair_extrinsics(obj_pts, img_pts_1, img_pts_2, k_1, d_1, k_2, d_2, camera_resolution):
    flags = cv2.CALIB_FIX_INTRINSIC
    term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-5)
    obj_pts = np.repeat(obj_pts[np.newaxis, :, :], img_pts_1.shape[0], axis=0).reshape((img_pts_1.shape[0], -1, 1, 3))
    img_pts_1 = img_pts_1.reshape((img_pts_1.shape[0], img_pts_1.shape[1]*img_pts_1.shape[2], 1, 2))
    img_pts_2 = img_pts_2.reshape((img_pts_2.shape[0], img_pts_2.shape[1]*img_pts_2.shape[2], 1, 2))
    rms, *_, r, t, _, _ = cv2.stereoCalibrate(obj_pts, img_pts_1, img_pts_2, k_1, d_1,k_2, d_2,
                                              camera_resolution, flags=flags, criteria=term_crit)
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
                             camera_resolution: Tuple[int, int]) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                    np.ndarray, Array[np.float32, ..., ..., 2]], None]:
    assert len(img_pts) >= 4, "Need at least 4 vaild frames to perform calibration."
    obj_pts_new = np.repeat(obj_pts[np.newaxis, :, :], img_pts.shape[0], axis=0).reshape((img_pts.shape[0], -1, 1, 3))
    img_pts_new = img_pts.reshape((img_pts.shape[0], -1, 1, 2))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)
    try:
        ret, k, d, r, t = cv2.fisheye.calibrate(obj_pts_new, img_pts_new, camera_resolution, None, None, None, None, flags,
                                                criteria)
        if ret:
            return k, d, r, t, img_pts, ret
    except Exception as e:
        if "CALIB_CHECK_COND" in str(e):
            idx = int(str(e)[str(e).find("input array ") + 12:].split(" ")[0])
            print(f"Image points at index {idx} caused an ill-conditioned matrix")
            img_pts = img_pts[np.arange(len(img_pts)) != idx]
            return calibrate_fisheye_camera(obj_pts, img_pts, camera_resolution)


def create_undistort_fisheye_point_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...]):
    def undistort_points(pts: Array[np.float32, ..., 2]):
        pts = pts.reshape((-1, 1, 2))
        undistorted = cv2.fisheye.undistortPoints(pts, k, d, P=k)
        return undistorted.reshape((-1,2))
    return undistort_points


def create_undistort_fisheye_img_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...], camera_resolution):
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, camera_resolution, cv2.CV_32FC1)
    def undistort_image(img):
        dst = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return dst
    return undistort_image



def calibrate_pair_extrinsics_fisheye(obj_pts, img_pts_1, img_pts_2, k_1, d_1, k_2, d_2, camera_resolution):
    flags = cv2.fisheye.CALIB_FIX_INTRINSIC
    term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    obj_pts = np.repeat(obj_pts[np.newaxis, :, :], img_pts_1.shape[0], axis=0).reshape((img_pts_1.shape[0], 1, -1, 3))
    img_pts_1 = img_pts_1.reshape((img_pts_1.shape[0], 1, img_pts_1.shape[1]*img_pts_1.shape[2], 2))
    img_pts_2 = img_pts_2.reshape((img_pts_2.shape[0], 1, img_pts_2.shape[1]*img_pts_2.shape[2], 2))
    rms, *_, r, t = cv2.fisheye.stereoCalibrate(obj_pts, img_pts_1, img_pts_2, k_1, d_1, k_2, d_2, camera_resolution,
                                                flags=flags, criteria=term_crit)
    return rms, r, t


def triangulate_points_fisheye(img_pts_1, img_pts_2, k1, d1, r1, t1, k2, d2, r2, t2):
    pts_1 = img_pts_1.reshape((-1,1,2))
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

def calibrate_pairwise_extrinsics(calib_func, img_pts_arr, fnames_arr, k_arr, d_arr, camera_resolution, board_shape, board_edge_len):
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
    for i in range(n_cam-1):
        k1 = k_arr[i]
        d1 = d_arr[i]
        k2 = k_arr[i+1]
        d2 = d_arr[i+1]
        points_1 = img_pts_arr[i]
        fnames_1 = fnames_arr[i]
        points_2 = img_pts_arr[i+1]
        fnames_2 = fnames_arr[i+1]
        # Extract corresponding points between cameras into img_pts 1 & 2
        img_pts_1 = []
        img_pts_2 = []
        corresponding_points = False
        for a, f in enumerate(fnames_1):
            if f in fnames_2:
                b = fnames_2.index(f)
                img_pts_1.append(points_1[a])
                img_pts_2.append(points_2[b])
                corresponding_points = True
        assert corresponding_points, f"No corresponding points between img_pts at index {i} and {i+1}"
        img_pts_1 = np.array(img_pts_1, dtype=np.float32)
        img_pts_2 = np.array(img_pts_2, dtype=np.float32)
        # Create object points
        obj_pts = create_board_object_pts(board_shape, board_edge_len)
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
        for k,v in fname_dict.items():
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
        point_3d_est = triangulate_func(img_pts_arr[a][a_pt], img_pts_arr[b][b_pt],
                                      k_arr[a], d_arr[a], r_arr[a], t_arr[a],
                                      k_arr[b], d_arr[b], r_arr[b], t_arr[b])
        points_3d.extend(point_3d_est)
        point_idx_counter += points_per_image
    # Convert to numpy arrays
    points_2d = np.array(points_2d, dtype=np.float32)
    points_3d = np.array(points_3d, dtype=np.float32)
    point_3d_indices = np.array(point_3d_indices, dtype=np.int)
    camera_indices = np.array(camera_indices, dtype=np.int)
    return points_2d, points_3d, point_3d_indices, camera_indices


def prepare_manual_points_for_bundle_adjustment(img_pts_arr, k_arr, d_arr, r_arr, t_arr, triangulate_func):
    pts = img_pts_arr.swapaxes(0,1) #should now have shape (n_cameras, n_points, 2)
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
            point_3d_est = triangulate_func(points_2d_temp[0], points_2d_temp[1],
                                            k_arr[a], d_arr[a], r_arr[a], t_arr[a],
                                            k_arr[b], d_arr[b], r_arr[b], t_arr[b])
            points_3d.append(point_3d_est)
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
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(img_pts_arr, fnames_arr,
                                                                                                            board_shape, k_arr,
                                                                                                            d_arr, r_arr, t_arr, triangulate_func)
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
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(img_pts_arr, fnames_arr,
                                                                                                            board_shape, k_arr,
                                                                                                            d_arr, r_arr, t_arr, triangulate_func)
    return bundle_adjust_points_and_extrinsics(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func)


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
                        args=(
                        n_cameras, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d, project_func),
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
    if "lab" in str(points_2d_df['frame'][0]):
        points_2d_df['frame'] = points_2d_df['frame'].str.replace(r".*img", '')
        points_2d_df['frame'] = points_2d_df['frame'].str.replace(".png", '')
    #print(points_2d_df)
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

    #print(df_pairs)
    points_3d_df = df_pairs[['frame', 'marker', 'x','y','z']].groupby(['frame','marker']).mean().reset_index()
    return points_3d_df



# def prepare_point_data_for_bundle_adjustment(points_2d_df, k_arr, d_arr, r_arr, t_arr, triangulate_func):
#
#     return points_2d, points_3d, point_3d_indices, camera_indices