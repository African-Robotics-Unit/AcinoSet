import os
import json
import pickle
import numpy as np
import pandas as pd
from glob import glob
from errno import ENOENT
from typing import Tuple
from nptyping import Array
from scipy.io import savemat
from datetime import datetime


# ========== LOAD FUNCTIONS ==========

def load_points(fpath, verbose=False):
    with open(fpath, 'r') as f:
        data = json.load(f)
        fnames = list(data['points'].keys())
        points = np.array(list(data['points'].values()), dtype=np.float32)
        board_shape = tuple(data['board_shape'])
        board_square_len = data['board_square_len']
        cam_res = tuple(data['camera_resolution'])
    if verbose:
        print(f'Loaded checkerboard points from {fpath}\n')
    return points, fnames, board_shape, board_square_len, cam_res


def load_manual_points(fpath, verbose=True):
    with open(fpath, 'r') as f:
        data = json.load(f)
        points = np.array(data['points'])
        fnames = []
        for i in data['frame_idx']:
            fnames.append('img{}.jpg'.format(str(i).zfill(5)))
        cam_res = tuple(data['camera_resolution'])
    if verbose:
        print(f'Loaded manual points from {fpath}\n')
    return points, fnames, cam_res


def load_camera(fpath, verbose=False):
    with open(fpath, 'r') as f:
        data = json.load(f)
        cam_res = tuple(data['camera_resolution'])
        k = np.array(data['k'], dtype=np.float64)
        d = np.array(data['d'], dtype=np.float64)
    if verbose:
        print(f'Loaded intrinsics from {fpath}\n')
    return k, d, cam_res


def load_scene(fpath, verbose=True):
    with open(fpath, 'r') as f:
        data = json.load(f)
        cam_res = tuple(data['camera_resolution'])
        k_arr = []
        d_arr = []
        r_arr = []
        t_arr = []
        for c in data['cameras']:
            k_arr.append(c['k'])
            d_arr.append(c['d'])
            r_arr.append(c['r'])
            t_arr.append(c['t'])
        k_arr = np.array(k_arr, dtype=np.float64)
        d_arr = np.array(d_arr, dtype=np.float64)
        r_arr = np.array(r_arr, dtype=np.float64)
        t_arr = np.array(t_arr, dtype=np.float64)
    if verbose:
        print(f'Loaded extrinsics from {fpath}\n')
    return k_arr, d_arr, r_arr, t_arr, cam_res


def load_dlc_points_as_df(dlc_df_fpaths, verbose=True):
    dfs = []
    for path in dlc_df_fpaths:
        dlc_df = pd.read_hdf(path)
        dlc_df = dlc_df.droplevel([0], axis=1).swaplevel(0,1,axis=1).T.unstack().T.reset_index().rename({'level_0':'frame'}, axis=1)
        dlc_df.columns.name = ''
        dfs.append(dlc_df)
    #create new dataframe
    dlc_df = pd.DataFrame(columns=['frame', 'camera', 'marker', 'x', 'y', 'likelihood'])
    for i, df in enumerate(dfs):
        df['camera'] = i
        df.rename(columns={'bodyparts':'marker'}, inplace=True)
        dlc_df = pd.concat([dlc_df, df], sort=True, ignore_index=True)

    dlc_df = dlc_df[['frame', 'camera', 'marker', 'x', 'y', 'likelihood']]
    if verbose:
        print(f'DLC points dataframe:\n{dlc_df}\n')
    return dlc_df


# ========== SAVE FUNCTIONS ==========

def save_points(out_fpath, img_points, img_fnames, board_shape, board_square_len, cam_res):
    created_timestamp = str(datetime.now())
    if isinstance(img_points, np.ndarray):
        img_points = img_points.tolist()
    points = dict(zip(img_fnames, img_points))
    data = {
        'timestamp': created_timestamp,
        'board_shape': board_shape,
        'board_square_len': board_square_len,
        'camera_resolution': cam_res,
        'points': points
    }
    with open(out_fpath, 'w') as f:
        json.dump(data, f)
    print(f'Saved points to {out_fpath}\n')


def save_camera(out_fpath, cam_res, k, d):
    created_timestamp = str(datetime.now())
    data = {
        'timestamp': created_timestamp,
        'camera_resolution': cam_res,
        'k': k.tolist(),
        'd': d.tolist(),
    }
    with open(out_fpath, 'w') as f:
        json.dump(data, f)
    print(f'Saved intrinsics to {out_fpath}\n')


def save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, cam_res):
    created_timestamp = str(datetime.now())
    cameras = []
    for k,d,r,t in zip(k_arr, d_arr, r_arr, t_arr):
        cameras.append({
            'k': k.tolist(),
            'd': d.tolist(),
            'r': r.tolist(),
            't': t.tolist()
        })
    data = {
        'timestamp': created_timestamp,
        'camera_resolution': cam_res,
        'cameras': cameras
    }
    with open(out_fpath, 'w') as f:
        json.dump(data, f)
    print(f'Saved extrinsics to {out_fpath}\n')


def save_optimised_cheetah(positions, out_fpath, extra_data=None, for_matlab=False, save_as_csv=False):
    file_data = dict(positions=positions)

    if extra_data is not None:
        assert type(extra_data) is dict
        file_data.update(extra_data)

    with open(out_fpath, 'wb') as f:
            pickle.dump(file_data, f)
    print('Saved', out_fpath)

    if for_matlab:
        out_fpath = os.path.splitext(out_fpath)[0] + '.mat'
        savemat(out_fpath, file_data)
        print('Saved', out_fpath)

    if save_as_csv:
        # to-do??
        # should use a similar method as save_3d_cheetah_as 3d, along the lines of:
        # xyz_labels = ['x', 'y', 'z']
        # pdindex = pd.MultiIndex.from_product([bodyparts, xyz_labels], names=["bodyparts", "coords"])

        # for i in range(len(video_fpaths)):
        #     cam_name = os.path.splitext(os.path.basename(video_fpaths[i]))[0]
        #     fpath = os.path.join(out_dir, cam_name + '_' + out_fname + '.h5')

        #     df = pd.DataFrame(data.reshape((n_frames, -1)), columns=pdindex, index=range(start_frame, start_frame+n_frames))
        #     df.to_csv(os.path.splitext(fpath)[0] + ".csv")
        pass


def save_3d_cheetah_as_2d(positions_3d, out_dir, scene_fpath, bodyparts, project_func, start_frame, save_as_csv=True, out_fname=None):
    assert os.path.dirname(os.path.dirname(scene_fpath)) in out_dir, 'scene_fpath does not belong to the same parent folder as out_dir'

    video_fpaths = sorted(glob(os.path.join(out_dir, 'cam[1-9].mp4'))) # check current dir for videos
    if not video_fpaths:
        video_fpaths = sorted(glob(os.path.join(os.path.dirname(out_dir), 'cam[1-9].mp4'))) # check parent dir for videos

    if video_fpaths:
        k_arr, d_arr, r_arr, t_arr, cam_res = load_scene(scene_fpath, verbose=False)
        assert len(k_arr)==len(video_fpaths)

        xyz_labels = ['x', 'y', 'likelihood'] # same format as DLC
        pdindex = pd.MultiIndex.from_product([bodyparts, xyz_labels], names=['bodyparts', 'coords'])

        positions_3d = np.array(positions_3d)
        n_frames = len(positions_3d)

        out_fname = os.path.basename(out_dir) if out_fname is None else out_fname

        for i in range(len(video_fpaths)):
            projections = project_func(positions_3d, k_arr[i], d_arr[i], r_arr[i], t_arr[i])
            out_of_range_indices = np.where((projections > cam_res) | (projections < [0]*2))[0]
            projections[out_of_range_indices] = np.nan

            data = np.full(positions_3d.shape, np.nan)
            data[:, :, 0:2] = projections.reshape((n_frames,-1, 2))

            cam_name = os.path.splitext(os.path.basename(video_fpaths[i]))[0]
            fpath = os.path.join(out_dir, cam_name + '_' + out_fname + '.h5')

            df = pd.DataFrame(data.reshape((n_frames, -1)), columns=pdindex, index=range(start_frame, start_frame+n_frames))
            if save_as_csv:
                df.to_csv(os.path.splitext(fpath)[0] + '.csv')
            df.to_hdf(fpath, f'{out_fname}_df', format='table', mode='w')

        fpath = fpath.replace(cam_name, 'cam*')
        print('Saved', fpath)
        if save_as_csv:
            print('Saved', os.path.splitext(fpath)[0] + '.csv')
        print()
    else:
        print('Could not save 3D cheetah to 2D - No videos were found in', out_dir, 'or', os.path.dirname(out_dir))

# ========== OTHER ==========

def find_scene_file(dir_path, scene_fname=None, verbose=True):
    if scene_fname is None:
        n_cams = len(glob(os.path.join(dir_path, 'cam[1-9].mp4'))) # reads up to cam9.mp4 only
        scene_fname = f'{n_cams}_cam_scene_sba.json' if n_cams else '[1-9]_cam_scene*.json'

    if dir_path and dir_path != os.path.join('..', 'data'):
        scene_fpath = os.path.join(dir_path, 'extrinsic_calib', scene_fname)
        # ignore [1-9]_cam_scene_before_corrections.json unless specified
        scene_files = sorted([scene_file for scene_file in glob(scene_fpath) if ('before_corrections' not in scene_file) or (scene_file == scene_fpath)])

        if scene_files:
            k_arr, d_arr, r_arr, t_arr, cam_res = load_scene(scene_files[-1], verbose)
            scene_fname = os.path.basename(scene_files[-1])
            n_cams = int(scene_fname[0]) # assuming scene_fname is of the form '[1-9]_cam_scene*'
            return k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_files[-1]
        else:
            return find_scene_file(os.path.dirname(dir_path), scene_fname, verbose)

    raise FileNotFoundError(ENOENT, os.strerror(ENOENT), os.path.join('extrinsic_calib', scene_fname))


def create_board_object_pts(board_shape: Tuple[int, int], square_edge_length: np.float32) -> Array[np.float32, ..., 3]:
    object_pts = np.zeros((board_shape[0]*board_shape[1], 3), np.float32)
    object_pts[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2) * square_edge_length
    return object_pts


def get_pairwise_3d_points_from_df(points_2d_df, k_arr, d_arr, r_arr, t_arr, triangulate_func):
    n_cams = len(k_arr)
    camera_pairs = [[i % n_cams, (i+1) % n_cams] for i in range(n_cams)]
    df_pairs = pd.DataFrame(columns=['x','y','z'])
    #get pairwise estimates
    for cam_a, cam_b in camera_pairs:
        d0 = points_2d_df[points_2d_df['camera']==cam_a]
        d1 = points_2d_df[points_2d_df['camera']==cam_b]
        intersection_df = d0.merge(d1, how='inner', on=['frame','marker'], suffixes=('_a', '_b'))
        if intersection_df.shape[0] > 0:
            print(f'Found {intersection_df.shape[0]} pairwise points between camera {cam_a} and {cam_b}')
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
            print(f'No pairwise points between camera {cam_a} and {cam_b}')

    print()
    points_3d_df = df_pairs[['frame', 'marker', 'x','y','z']].groupby(['frame','marker']).mean().reset_index()
    return points_3d_df