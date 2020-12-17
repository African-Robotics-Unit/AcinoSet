# This program converts the output data of Argus Clicker to the data for SBA.
import os, sys
import glob
import pandas as pd
import yaml
from pprint import pprint
import numpy as np
from datetime import datetime
import json
import argparse


# DATA_DIR = '../data/27_02_2019/extrinsic_calib/defined_points'

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--data_dir',
    type=str,
    help='data directory path'
)
parser.add_argument(
    '--cam_res_w',
    type=int,
    default=2704,
    help='the pixel witdth of the camera resolution'
)
parser.add_argument(
    '--cam_res_h',
    type=int,
    default=1520,
    help='the pixel height of the camera resolution'
)
args = parser.parse_args()


if __name__ == "__main__":
    cam_res = [args.cam_res_w, args.cam_res_h]
    # load input data
    argus_point_fpath = glob.glob(os.path.join(args.data_dir, '*-xypts.csv'))[0]
    argus_points = pd.read_csv(argus_point_fpath)
    argus_config_fpath = glob.glob(os.path.join(args.data_dir, '*-config.yaml'))[0]
    with open(argus_config_fpath, 'r') as f:
        argus_config = yaml.safe_load(f)
    argus_res_fpath = glob.glob(os.path.join(args.data_dir, '*-xyzres.csv'))[0]
    argus_res = pd.read_csv(argus_res_fpath)

    # params
    n_cameras = len(argus_config[0]['videos'])
    track_names = list(argus_res.columns)

    # convert
    points = []     # (n_points, n_cameras, 2)
    frame_idx = []
    for track_name in track_names:
        columns = argus_points.columns.str.contains(track_name)
        point_rows = argus_points.loc[:, columns]

        for row_idx, row in point_rows.iterrows():
            if row.isnull().all():
                continue

            cameras = []
            for camera_idx in range(n_cameras):
                camera_idx += 1
                point = []
                for axis in ['x', 'y']:
                    column_name = f'{track_name}_cam_{camera_idx}_{axis}'
                    point.append(row[column_name] if axis == 'x' else cam_res[1] - row[column_name])
                cameras.append(point)
            points.append(cameras)
            frame_idx.append(row_idx)

    result = {
        'created_timestamp': str(datetime.now()),
        'camera_resolution': cam_res,
        'points': points,
        'frame_idx': frame_idx
    }

    # output converted data
    output_fpath = os.path.join(args.data_dir, 'defined_points.json')
    with open(output_fpath, "w") as write_file:
        json.dump(result, write_file)
    print(f'Success to output into {output_fpath}')
