import numpy as np
from time import time
from cv2 import Rodrigues
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from .utils import *


# ========== SBA UTILS ==========

def create_bundle_adjustment_jacobian_sparsity_matrix(n_cams, n_params_per_camera, camera_indices, n_points, point_indices):
    m = camera_indices.size * 2
    n = n_cams * n_params_per_camera + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(n_params_per_camera):
        A[2 * i, camera_indices * n_params_per_camera + s] = 1
        A[2 * i + 1, camera_indices * n_params_per_camera + s] = 1
    for s in range(3):
        A[2 * i, n_cams * n_params_per_camera + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cams * n_params_per_camera + point_indices * 3 + s] = 1
    return A


def params_to_points_extrinsics(params, n_cams, n_points):
    r_end_idx = n_cams*3
    t_end_idx = r_end_idx + n_cams*3
    r_vecs = params[:r_end_idx].reshape((n_cams,3))
    r_arr = np.array([Rodrigues(r)[0] for r in r_vecs], dtype=np.float64)
    t_arr = params[r_end_idx:t_end_idx].reshape((n_cams,3, 1))
    obj_pts = params[t_end_idx:].reshape((n_points, 3))
    return obj_pts, r_arr, t_arr


# ========== SBA PREPARATION ==========

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
    # img_pts_arr shape: (n_points, n_cams, 2)
    pts = img_pts_arr.swapaxes(0,1) # should now have shape (n_cams, n_points, 2)
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
    points_3d = np.array(points_3d, dtype=np.float32)
    point_3d_indices = np.array(point_3d_indices, dtype=np.int)
    camera_indices = np.array(camera_indices, dtype=np.int)
    
    return points_2d, points_3d, point_3d_indices, camera_indices


# ========== COST FUNCTIONS ==========

def cost_func_points_extrinsics(params, n_cams, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d, project_func):
    obj_pts, r_arr, t_arr = params_to_points_extrinsics(params, n_cams, n_points)
    reprojected_pts = np.array([project_func(obj_pts[i], k_arr[j], d_arr[j], r_arr[j], t_arr[j])[0] for i,j in zip(point_3d_indices, camera_indices)])
    error = (reprojected_pts - points_2d).ravel()
    return error


def cost_func_points_only(params, n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d, project_func):
    obj_pts = params.reshape((n_points, 3))
    reprojected_pts = np.array([project_func(obj_pts[i], k_arr[j], d_arr[j], r_arr[j], t_arr[j])[0] for i,j in zip(point_3d_indices, camera_indices)])
    error = (reprojected_pts - points_2d).ravel()
    return error


# ========== SBA ==========

def bundle_adjust_points_and_extrinsics(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func):
    n_points = len(points_3d)
    n_cams = len(k_arr)
    n_params_per_camera = 6
    r_vecs = np.array([Rodrigues(r)[0] for r in r_arr], dtype=np.float64).flatten()
    t_vecs = t_arr.flatten()
    x0 = np.concatenate([r_vecs, t_vecs, points_3d.flatten()])
    f0 = cost_func_points_extrinsics(x0, n_cams, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d,
                                     project_func)
    A = create_bundle_adjustment_jacobian_sparsity_matrix(n_cams, n_params_per_camera, camera_indices, n_points,
                                                          point_3d_indices)
    t0 = time()
    res = least_squares(cost_func_points_extrinsics, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10,
                        method='trf', loss='cauchy',
                        args=(n_cams, n_points, point_3d_indices, camera_indices, k_arr, d_arr, points_2d, project_func),
                        max_nfev=1000)
    t1 = time()
    print(f'\nOptimization took {t1-t0:.2f} seconds')
    obj_pts, r_arr, t_arr = params_to_points_extrinsics(res.x, n_cams, n_points)
    residuals = dict(before=f0, after=res.fun)
    return obj_pts, r_arr, t_arr, residuals


def bundle_adjust_points_only(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func, f_scale=50):
    n_points = len(points_3d)
    n_cams = len(k_arr)
    n_params_per_camera = 0
    x0 = points_3d.flatten()
    f0 = cost_func_points_only(x0, n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d, project_func)
    A = create_bundle_adjustment_jacobian_sparsity_matrix(n_cams, n_params_per_camera, camera_indices, n_points, point_3d_indices)
    t0 = time()
    res = least_squares(cost_func_points_only, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-15, method='trf', loss='cauchy', f_scale=f_scale,
                        args=(n_points, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, points_2d, project_func), max_nfev=500)
    t1 = time()
    print(f'\nOptimization took {t1-t0:.2f} seconds')
    residuals = dict(before=f0, after=res.fun)
    obj_pts = res.x.reshape((n_points, 3))
    return obj_pts, residuals


def bundle_adjust_board_points_only(img_pts_arr, fnames_arr, board_shape, k_arr, d_arr, r_arr, t_arr, triangulate_func, project_func):
    points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(
        img_pts_arr, fnames_arr,
        board_shape,
        k_arr, d_arr, r_arr, t_arr,
        triangulate_func
    )
    obj_pts, residuals = bundle_adjust_points_only(points_2d, points_3d, point_3d_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func)
    return obj_pts, residuals


# ==========  BACK-END FUNCTION MANAGER  ==========

def _sba_board_points(scene_fpath, points_fpaths, manual_points_fpath, out_fpath, triangulate_func, project_func, camera_indices=None, manual_points_only=False):
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
        # load manual points
        manual_points, manual_fnames, *_ = load_manual_points(manual_points_fpath)
        # optimize
        
        # (n_points, [x,y]), (n_3d_points, [x,y,z]), (3d_point_indices), (camera_indices)
        manual_points_2d, manual_points_3d, manual_point_3d_indices, manual_camera_indices = prepare_manual_points_for_bundle_adjustment(
            manual_points,
            k_arr, d_arr, r_arr, t_arr,
            triangulate_func
        )
        if manual_points_only:
            print('bundle_adjust_board_points_and_extrinsics (with manual points only)')
            
            obj_pts, r_arr, t_arr, res = bundle_adjust_points_and_extrinsics(
                manual_points_2d, manual_points_3d, manual_point_3d_indices, manual_camera_indices,
                k_arr, d_arr, r_arr, t_arr,
                project_func
            )
        else:
            print('bundle_adjust_board_points_and_extrinsics (with manual points)')
            # (n_points, [x,y]), (n_3d_points, [x,y,z]), (3d_point_indices), (camera_indices)
            points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(
                img_pts_arr, fnames_arr, board_shape,
                k_arr, d_arr, r_arr, t_arr,
                triangulate_func
            )

            # combine both data
            points_2d = np.append(points_2d, manual_points_2d, axis=0)
            points_3d = np.append(points_3d, manual_points_3d, axis=0)
            point_3d_indices = np.append(point_3d_indices, manual_point_3d_indices + point_3d_indices.max(), axis=0)
            camera_indices = np.append(camera_indices, manual_camera_indices, axis=0)
            
            obj_pts, r_arr, t_arr, res = bundle_adjust_points_and_extrinsics(
                points_2d, points_3d, point_3d_indices, camera_indices,
                k_arr, d_arr, r_arr, t_arr,
                project_func
            )
    else:
        print('bundle_adjust_board_points_and_extrinsics')
        # (n_points, [x,y]), (n_3d_points, [x,y,z]), (3d_point_indices), (camera_indices)
        points_2d, points_3d, point_3d_indices, camera_indices = prepare_calib_board_data_for_bundle_adjustment(
            img_pts_arr, fnames_arr, board_shape,
            k_arr, d_arr, r_arr, t_arr,
            triangulate_func
        )
        obj_pts, r_arr, t_arr, res = bundle_adjust_points_and_extrinsics(
            points_2d, points_3d, point_3d_indices, camera_indices,
            k_arr, d_arr, r_arr, t_arr,
            project_func
        )
    print(f"\nBefore: mean: {np.mean(res['before'])}, std: {np.std(res['before'])}")
    print(f"After: mean: {np.mean(res['after'])}, std: {np.std(res['after'])}\n")
        
    save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, cam_res)
    return res


def _sba_points(scene_fpath, points_2d_df, triangulate_func, project_func):
    # load scene
    k_arr, d_arr, r_arr, t_arr, cam_res = load_scene(scene_fpath)
    assert len(k_arr) == points_2d_df['camera'].nunique()
    
    points_3d_df = get_pairwise_3d_points_from_df(
        points_2d_df,
        k_arr, d_arr.reshape((-1,4)), r_arr, t_arr,
        triangulate_func
    )

    points_3d_df['point_index'] = points_3d_df.index
    points_3d = np.array(points_3d_df[['x', 'y', 'z']])
    
    points_df = points_2d_df.merge(points_3d_df, how='inner', on=['frame','marker'], suffixes=('_cam',''))
    points_2d = np.array(points_df[['x_cam', 'y_cam']])
    point_indices = np.array(points_df['point_index'])
    camera_indices = np.array(points_df['camera'])
    
    print('bundle_adjust_points_only')
    pts_3d, res = bundle_adjust_points_only(
        points_2d, points_3d, point_indices, camera_indices, k_arr, d_arr, r_arr, t_arr, project_func, f_scale=50
    )
    print(f"\nBefore: mean: {np.mean(res['before'])}, std: {np.std(res['before'])}")
    print(f"After: mean: {np.mean(res['after'])}, std: {np.std(res['after'])}\n")

    new_points_3d_df = points_3d_df.copy()
    new_points_3d_df[['x','y','z']] = pts_3d
    return new_points_3d_df, res