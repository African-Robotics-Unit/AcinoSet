import numpy as np
import pandas as pd
from typing import Tuple
from nptyping import Array
from datetime import datetime
import json
import os
from glob import glob
import errno


def create_board_object_pts(board_shape: Tuple[int, int], square_edge_length: np.float32) -> Array[np.float32, ..., 3]:
    object_pts = np.zeros((board_shape[0]*board_shape[1], 3), np.float32)
    object_pts[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2) * square_edge_length
    return object_pts


def save_points(out_fpath, img_points, img_fnames, board_shape, board_square_len, cam_res):
    created_timestamp = str(datetime.now())
    if isinstance(img_points, np.ndarray):
        img_points = img_points.tolist()
    points = dict(zip(img_fnames, img_points))
    data = {
        "timestamp": created_timestamp,
        "board_shape": board_shape,
        "board_square_len": board_square_len,
        "camera_resolution": cam_res,
        "points": points
    }
    with open(out_fpath, "w") as f:
        json.dump(data, f)
    print(f"Saved points to {out_fpath}\n")


def load_points(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
        fnames = list(data["points"].keys())
        points = np.array(list(data["points"].values()), dtype=np.float32)
        board_shape = tuple(data["board_shape"])
        board_square_len = data["board_square_len"]
        cam_res = tuple(data["camera_resolution"])
    return points, fnames, board_shape, board_square_len, cam_res


def load_manual_points(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
        points = np.array(data['points'])
        fnames = []
        for i in data['frame_idx']:
            fnames.append('img{}.jpg'.format(str(i).zfill(5)))
        cam_res = tuple(data["camera_resolution"])

    return points, fnames, cam_res


def save_camera(out_fpath, cam_res, k, d):
    created_timestamp = str(datetime.now())
    data = {
        "timestamp": created_timestamp,
        "camera_resolution": cam_res,
        "k": k.tolist(),
        "d": d.tolist(),
    }
    with open(out_fpath, "w") as f:
        json.dump(data, f)
    print(f"Saved intrinsics to {out_fpath}\n")


def load_camera(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
        cam_res = tuple(data["camera_resolution"])
        k = np.array(data["k"], dtype=np.float64)
        d = np.array(data["d"], dtype=np.float64)
    return k, d, cam_res


def save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, cam_res):
    created_timestamp = str(datetime.now())
    cameras = []
    for k,d,r,t in zip(k_arr, d_arr, r_arr, t_arr):
        cameras.append({
            "k": k.tolist(),
            "d": d.tolist(),
            "r": r.tolist(),
            "t": t.tolist()
        })
    data = {
        "timestamp": created_timestamp,
        "camera_resolution": cam_res,
        "cameras": cameras
    }
    with open(out_fpath, "w") as f:
        json.dump(data, f)
    print(f"Saved extrinsics to {out_fpath}\n")


def load_scene(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
        cam_res = tuple(data["camera_resolution"])
        k_arr = []
        d_arr = []
        r_arr = []
        t_arr = []
        for c in data["cameras"]:
            k_arr.append(c["k"])
            d_arr.append(c["d"])
            r_arr.append(c["r"])
            t_arr.append(c["t"])
        k_arr = np.array(k_arr, dtype=np.float64)
        d_arr = np.array(d_arr, dtype=np.float64)
        r_arr = np.array(r_arr, dtype=np.float64)
        t_arr = np.array(t_arr, dtype=np.float64)
    return k_arr, d_arr, r_arr, t_arr, cam_res


def load_dlc_points_as_df(dlc_df_fpaths):
    dfs = []
    for path in dlc_df_fpaths:
        dlc_df = pd.read_hdf(path)
        dlc_df = dlc_df.droplevel([0], axis=1).swaplevel(0,1,axis=1).T.unstack().T.reset_index().rename({'level_0':'frame'}, axis=1)
        dlc_df.columns.name = ''
        dfs.append(dlc_df)
    #create new dataframe
    dlc_df = pd.DataFrame(columns=['frame', 'camera', 'marker', 'x', 'y', 'likelihood'])
    for i, df in enumerate(dfs):
        df["camera"] = i
        df.rename(columns={"bodyparts":"marker"}, inplace=True)
        dlc_df = pd.concat([dlc_df, df], sort=True, ignore_index=True)

    dlc_df = dlc_df[['frame', 'camera', 'marker', 'x', 'y', 'likelihood']]
    return dlc_df


def find_scene_file(dir_path, scene_fname=None, suppress_output=False):
    if not scene_fname:
        n_cams = len(glob(os.path.join(dir_path, "cam[1-9].mp4"))) # reads up to cam9.mp4 only
        scene_fname = f"{n_cams}_cam_scene_sba.json" if n_cams else "[1-9]_cam_scene*.json"

    if dir_path and dir_path != os.path.join("..", "data"):
        scene_fpath = os.path.join(dir_path, "extrinsic_calib", scene_fname)
        scene_files = [scene_file for scene_file in glob(scene_fpath) if "before_corrections" not in scene_file ]

        if scene_files:
            k_arr, d_arr, r_arr, t_arr, cam_res = load_scene(scene_files[-1])
            if not suppress_output:
                print("Loaded extrinsics from", scene_files[-1])
            scene_fname = os.path.basename(scene_files[-1])
            n_cams = int([j for j in scene_fname if j.isdigit()][0])
            return k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath
        else:
            return find_scene_file(os.path.dirname(dir_path), scene_fname, suppress_output)

    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), os.path.join("extrinsic_calib", scene_fname))
