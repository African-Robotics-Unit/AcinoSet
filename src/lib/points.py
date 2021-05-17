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


def EOM_curve_fit(pts_3d, frames=None, fit_order=3):
    from sympy import lambdify, diff
    from scipy.optimize import curve_fit

    # equation of motion series is s_new = s + v*t + 1/2*a*t**2 + ... + 1/n!*x*t**n
    # here we use fit = a + b*t + c*t**2 + ... + var*t**n

    assert 0 < fit_order < 19 and isinstance(fit_order, int), 'fit_order must be an integer from 1 to 18'

    if frames is None:
        frames_new = np.arange(len(pts_3d))

    num_axes = pts_3d.shape[1] # x, y, z
    n = fit_order + 1

    syms     = ['t', 'a'] # independent var & zeroth order term in series
    nth_term = f'var*{syms[0]}**order' # general expression for nth term in series

    fit_func_str = syms[1]
    func_params  = np.zeros((num_axes, 1)) # store the fit params for every axis
    # loop from lowest to highest order fit so that lower can be used to initialise higher
    for order in range(1, n): # skip zeroth order fit func because it's already in fit_func_str
        syms.append(chr(ord(syms[-1]) + 1)) # append next letter
        fit_func_str += ' + ' + nth_term.replace('var', syms[-1]).replace('order', str(order)) # add new nth term
        fit_func = lambdify(syms, fit_func_str)

        func_params = np.append(func_params, np.zeros((num_axes, 1)), axis=1) # concat zeros to initialize new higher order fit_func
        for ax in range(num_axes):
            func_params[ax, :], _ = curve_fit(fit_func, frames, pts_3d[:, ax], p0=func_params[ax, :], method='trf', loss='cauchy') # initialize with previous fit params

    syms.remove(syms[1]) # remove zeroth/constant term
    fit_func_deriv = lambdify(syms, diff(fit_func_str, syms[0]))

    fit, fit_deriv = np.full(pts_3d.shape, np.nan), np.full(pts_3d.shape, np.nan)
    for ax in range(num_axes):
        fit[:, ax]       = fit_func(frames, *func_params[ax, :])
        fit_deriv[:, ax] = fit_func_deriv(frames, *func_params[ax, 1:])

    return fit, fit_deriv
