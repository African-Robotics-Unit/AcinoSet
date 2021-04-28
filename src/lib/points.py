import cv2 as cv
import numpy as np
from nptyping import Array
from typing import Tuple, List, Union


def plot_corners(img: Array[np.uint8, ..., ..., 3], corners: Array[np.float32, ..., ..., 2], board_shape: Tuple[int, int], show_window=False):
    """
    :param img: image as a numpy array of ints of shape (height, width, 3)
    :param corners: pixel positions of found corners as a numpy array of ints of shape (points_per_row, points_per_column, 2)
    :return:
    """
    corners = corners.reshape((-1, 1, 2))
    # note that for drawChessboardCorners to work, points must have type float32
    cv.drawChessboardCorners(img, board_shape, corners, True)
    if show_window:
        name = f'Calibration board - columns: {board_shape[0]}, rows: {board_shape[1]}'
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(0)
    return img


def find_corners(fname, img: Array[np.uint8, ..., ..., 3], board_shape: Tuple[int, int], window_size=11) -> Union[Array[np.float32, ..., ..., 2], None]:
    """
    :param img: image as a numpy array of ints of shape (height, width, 3)
    :param board_shape: tuple of ints representing the number of corners (points_per_row, points_per_column)
    :return: corners: pixel positions of found corners as a numpy array of ints of shape (points_per_row, points_per_column, 2)
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv.findChessboardCorners(
        gray,
        board_shape,
        flags
    )
    if ret:
        subpixel_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        corners = cv.cornerSubPix(gray, corners, (window_size, window_size), (-1, -1), subpixel_criteria)
        # draw checkerboard points on image and save in folder 'drawn'
        cv.drawChessboardCorners(img, board_shape, corners, ret)
        fpath = fname[:-12]+'drawn'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        cv.imwrite(f'{fpath}/{fname[-12:]}', img)
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
        img = cv.imread(fp)
        if img is None:
            print(f'Could not read image: {fp}')
        else:
            if cam_res is None:
                cam_res = (img.shape[1], img.shape[0])
            else:
                assert cam_res == (img.shape[1], img.shape[0])
            img_corners = find_corners(fp, img, board_shape, window_size)
            if img_corners is not None:
                corners.append(img_corners)
                found_filepaths.append(fp)
                print(f'Found corners for file {i}: {fp}')
            else:
                print(f'No corners found for file {i}: {fp}')
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
