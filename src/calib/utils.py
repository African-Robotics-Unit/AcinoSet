import numpy as np
import pandas as pd
from typing import Tuple
from nptyping import Array
from datetime import datetime
import json
import os


def create_board_object_pts(board_shape: Tuple[int, int], square_edge_length: np.float32) -> Array[np.float32, ..., 3]:
    object_pts = np.zeros((board_shape[0]*board_shape[1], 3), np.float32)
    object_pts[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2) * square_edge_length
    return object_pts


def save_points(out_fpath, img_points, img_fnames, board_shape, board_edge_len, camera_resolution):
    print(f"Saving points to {out_fpath}")
    created_timestamp = str(datetime.now())
    if isinstance(img_points, np.ndarray):
        img_points = img_points.tolist()
    points = dict(zip(img_fnames, img_points))
    data = {
        "created_timestamp": created_timestamp,
        "board_shape": board_shape,
        "board_edge_len": board_edge_len,
        "camera_resolution": camera_resolution,
        "points": points
    }
    with open(out_fpath, "w") as f:
        json.dump(data, f)


def load_points(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
        fnames = list(data["points"].keys())
        points = np.array(list(data["points"].values()), dtype=np.float32)
        board_shape = tuple(data["board_shape"])
        board_edge_len = data["board_edge_len"]
        camera_resolution = tuple(data["camera_resolution"])
    return points, fnames, board_shape, board_edge_len, camera_resolution


def save_camera(out_fpath, camera_resolution, k, d):
    created_timestamp = str(datetime.now())
    data = {
        "created_timestamp": created_timestamp,
        "camera_resolution": camera_resolution,
        "k": k.tolist(),
        "d": d.tolist(),
    }
    with open(out_fpath, "w") as f:
        json.dump(data, f)


def load_camera(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
        camera_resolution = tuple(data["camera_resolution"])
        k = np.array(data["k"], dtype=np.float64)
        d = np.array(data["d"], dtype=np.float64)
    return k, d, camera_resolution


def save_scene(out_fpath, k_arr, d_arr, r_arr, t_arr, camera_resolution):
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
        "created_timestamp": created_timestamp,
        "camera_resolution": camera_resolution,
        "cameras": cameras
    }
    with open(out_fpath, "w") as f:
        json.dump(data, f)


def load_scene(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
        camera_resolution = tuple(data["camera_resolution"])
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
    return k_arr, d_arr, r_arr, t_arr, camera_resolution



def create_dlc_points_2d_file(dlc_df_fpaths):
    dfs = []
    for path in dlc_df_fpaths:
        dlc_df = pd.read_hdf(path)
        dlc_df=dlc_df.droplevel([0], axis=1).swaplevel(0,1,axis=1).T.unstack().T.reset_index().rename({'level_0':'frame'}, axis=1)
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
