import numpy as np
from typing import Tuple, List, Union
from nptyping import Array
import cv2
import os

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
