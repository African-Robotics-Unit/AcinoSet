import numpy as np
from typing import Tuple, List, Union
from nptyping import Array
import cv2
import os
import pandas as pd

def plot_corners(img: Array[np.uint8, ..., ..., 3], corners: Array[np.float32, ..., ..., 2], board_shape: Tuple[int, int], show_window=False):
    """
    :param img: image as a numpy array of ints of shape (height, width, 3)
    :param corners: pixel positions of found corners as a numpy array of ints of shape (points_per_row, points_per_column, 2)
    :return:
    """
    corners = corners.reshape((-1, 1, 2))
    # note that for drawChessboardCorners to work, points must have type float32
    cv2.drawChessboardCorners(img, board_shape, corners, True)
    if show_window:
        name = f"Calibration board - columns: {board_shape[0]}, rows: {board_shape[1]}"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(0)
    return img


def find_corners(fname, img: Array[np.uint8, ..., ..., 3], board_shape: Tuple[int, int], window_size=11) -> Union[Array[np.float32, ..., ..., 2], None]:
    """
    :param img: image as a numpy array of ints of shape (height, width, 3)
    :param board_shape: tuple of ints representing the number of corners (points_per_row, points_per_column)
    :return: corners: pixel positions of found corners as a numpy array of ints of shape (points_per_row, points_per_column, 2)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(
        gray,
        board_shape,
        flags
    )
    if ret:
        subpixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        corners = cv2.cornerSubPix(gray, corners, (window_size, window_size), (-1, -1), subpixel_criteria)
        # draw checkerboard points on image and save in folder 'drawn'
        cv2.drawChessboardCorners(img, board_shape, corners, ret)
        fpath = fname[:-12]+'drawn'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        cv2.imwrite(f'{fpath}/{fname[-12:]}', img)
        return corners.reshape((*board_shape, 2))
    return None


def find_corners_images(filepaths: List[str], board_shape: Tuple[int, int], window_size=11) -> Tuple[Array[np.float32, ..., ..., ..., 2], List[str], Tuple[int, int]]:
    """
    :param filepaths: paths to image files
    :param board_shape: tuple of ints representing the number of corners (points_per_row, points_per_column)
    :return: corners: pixel positions of found corners as a numpy array of ints of shape (num_imgs_with_points, points_per_row, points_per_column, 2)
    """
    corners = []
    found_filepaths = []
    cam_res = None
    for i, fp in enumerate(filepaths):
        img = cv2.imread(fp)
        if img is None:
            print(f"Couldn't read image: {fp}")
        else:
            if cam_res is None:
                cam_res = (img.shape[1], img.shape[0])
            else:
                assert cam_res == (img.shape[1], img.shape[0])
            img_corners = find_corners(fp, img, board_shape, window_size)
            if img_corners is not None:
                corners.append(img_corners)
                found_filepaths.append(fp)
                print(f"Found corners for file {i}: {fp}")
            else:
                print(f"No corners found for file {i}: {fp}")
    return np.array(corners, dtype=np.float32).reshape((-1, *board_shape, 2)), found_filepaths, cam_res


def common_image_points(pts_1, fnames_1, pts_2, fnames_2):
    fnames = []
    img_pts_1 = []
    img_pts_2 = []
    for a, f in enumerate(fnames_1):
        if f in fnames_2:
            b = fnames_2.index(f)
            fnames.append(f)
            img_pts_1.append(pts_1[a])
            img_pts_2.append(pts_2[b])

    img_pts_1 = np.array(img_pts_1, dtype=np.float32)
    img_pts_2 = np.array(img_pts_2, dtype=np.float32)
    return img_pts_1, img_pts_2, fnames


def get_pairwise_3d_points_from_df(points_2d_df, k_arr, d_arr, r_arr, t_arr, triangulate_func):
    n_cams = len(k_arr)
    camera_pairs = [[i % n_cams, (i+1) % n_cams] for i in range(n_cams)]
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
