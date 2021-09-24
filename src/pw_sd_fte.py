import os
import json
from typing import Union, Tuple, Dict, List
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from pyomo.core.expr.current import log as pyomo_log
import sympy as sp
import pandas as pd
from glob import glob
from time import time
from scipy.interpolate import UnivariateSpline
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from lib import misc, utils, app, metric
from lib.calib import triangulate_points_fisheye, project_points_fisheye
import matplotlib.pyplot as plt

plt.style.use(os.path.join('../configs', 'mechatronics_style.yaml'))

def validate_dataset(root_dir: str) -> List:
    markers = misc.get_markers()
    coords = ['x', 'y', 'z']

    # get velocity of virtual body
    def get_velocity(position, h):
        # body COM approximated as mean of tail base, spine and neck points
        body_x = np.mean([position[m, 'x'] for m in ['tail_base', 'spine', 'neck_base']],0)
        body_y = np.mean([position[m, 'y'] for m in ['tail_base', 'spine', 'neck_base']],0)
        body_z = np.mean([position[m, 'z'] for m in ['tail_base', 'spine', 'neck_base']],0)

        body_dx = (body_x[1:]-body_x[:-1])/h
        body_dy = (body_y[1:]-body_y[:-1])/h
        body_dz = (body_z[1:]-body_z[:-1])/h

        body_v = np.sqrt(body_dx**2 + body_dy**2 + body_dz**2)
        return body_v

    # extract position data from individual pickle file
    def extract_data(filename):
        data = utils.load_pickle(filename)

        position = {}
        for m_i,m in enumerate(markers):
            for c_i,c in enumerate(coords):
                    position.update({(m,c):np.array([data['positions'][f][m_i][c_i] for f in range(len(data['positions']))])})
        # flip direction if running towards -x
        if position['neck_base','x'][-1] < position['neck_base','x'][0]:
            for m in markers:
                for c in ['x','y']:
                    position[m,c] = -position[m,c]
        return position

    bad_trajectories = []
    fte_fpaths = sorted(glob(os.path.join(root_dir, "**/fte.pickle"), recursive=True))
    for fpath in fte_fpaths:
        position = extract_data(fpath)
        temp = fpath.split(root_dir)[1]
        info = temp.split(os.sep)
        date = info[1]
        # sanity checks
        fail = 0
        h = 1/90
        if date[:4] == '2017': h = 1/90
        if date[:4] == '2019': h = 1/120
        body_v = get_velocity(position, h)
        if np.max(np.abs(body_v)) > 50: # if velocity implausibly high
            fail += 1
        for m in markers:
            if np.min(position[m, 'z']) < -0.3: # goes too deep
                fail += 1
            if m not in ['tail_base', 'tail1', 'tail2'] and np.max(position[m, 'z']) > 1: # goes too high
                fail += 1

        if fail != 0:
            bad_trajectories.append(os.sep.join(info[1:-2]))

    return bad_trajectories

def run_acinoset(root_dir: str, video_data: Dict, out_dir_prefix: str):
    """
    Runs through the video list in AcinoSet and performs the 3D reconstruction for each video.

    Args:
        root_dir: The root directory where the videos are stored, along with the `pose_3d_functions.pickle` file, and `gt_labels` directory.
        video_data: The list of videos stored in a dictionary with a key `test_dirs` for each directory in AcinoSet.
        out_dir_prefix: Used to change the output directory from the root. This is often used if you have the cheetah data stored in a location and you want the output to be saved elsewhere.
    """
    import gc
    tests = video_data['test_dirs']
    manually_selected_frames = {
        '2019_03_03/phantom/run': (73, 272),
        '2017_12_12/top/cetane/run1_1': (100, 241),
        '2019_03_05/jules/run': (58, 176),
        '2019_03_09/lily/run': (80, 180),
        '2017_09_03/top/zorro/run1_2': (20, 120),
        '2017_08_29/top/phantom/run1_1': (20, 170),
        '2017_12_21/top/lily/run1': (7, 106),
        '2019_03_03/menya/run': (20, 130),
        '2017_12_10/top/phantom/run1': (30, 130),
        '2017_09_03/bottom/zorro/run2_1': (126, 325),
        '2019_02_27/ebony/run': (20, 200),
        '2017_12_09/bottom/phantom/run2': (18, 117),
        '2017_09_03/bottom/zorro/run2_3': (1, 200),
        '2017_08_29/top/jules/run1_1': (10, 110),
        '2017_09_02/top/jules/run1': (10, 110),
        '2019_03_07/menya/run': (60, 160),
        '2017_09_02/top/phantom/run1_2': (20, 160),
        '2019_03_05/lily/run': (150, 250),
        '2017_12_12/top/cetane/run1_2': (3, 202),
        '2019_03_07/phantom/run': (100, 200),
        '2019_02_27/romeo/run': (12, 190),
        '2017_08_29/top/jules/run1_2': (30, 130),
        '2017_12_16/top/cetane/run1': (110, 210),
        '2017_09_02/top/phantom/run1_1': (33, 150),
        '2017_09_02/top/phantom/run1_3': (35, 135),
        '2017_09_03/top/zorro/run1_1': (4, 203),
        '2019_02_27/kiara/run': (10, 110),
        '2017_09_02/bottom/jules/run2': (35, 171),
        '2017_09_03/bottom/zorro/run2_2': (32, 141),
        '2019_03_05/lily/flick': (100, 200),
        '2017_08_29/top/zorro/flick1_2': (20, 140),
        '2017_09_02/bottom/phantom/flick2_1': (5, 100),
        '2017_12_12/bottom/big_girl/flick2': (30, 100),
        '2019_03_03/phantom/flick': (270, 460),
        '2019_03_09/lily/flick': (10, 100),
        '2019_03_09/jules/flick2': (80, 185),
        '2017_09_03/top/zorro/flick1_1': (62, 150),
        '2017_12_21/top/lily/flick1': (40, 180),
        '2017_12_21/bottom/jules/flick2_1': (50, 200),
        '2017_09_03/top/phantom/flick1': (100, 240),
        '2017_09_02/top/jules/flick1_1': (60, 200),
        '2017_08_29/top/phantom/flick1_1': (50, 200)
    }

    bad_videos = ('2017_09_03/bottom/phantom/flick2', '2017_09_02/top/phantom/flick1_1', '2017_12_17/top/zorro/flick1')
    time0 = time()
    print('Run reconstruction on all videos...')
    # Initialise the Ipopt solver.
    optimiser = SolverFactory('ipopt', executable='/home/zico/lib/ipopt/build/bin/ipopt')
    # solver options
    optimiser.options['print_level'] = 5
    optimiser.options['max_iter'] = 400
    optimiser.options['max_cpu_time'] = 10000
    optimiser.options['Tol'] = 1e-1
    optimiser.options['OF_print_timing_statistics'] = 'yes'
    optimiser.options['OF_print_frequency_time'] = 10
    optimiser.options['OF_hessian_approximation'] = 'limited-memory'
    optimiser.options['OF_accept_every_trial_step'] = 'yes'
    optimiser.options['linear_solver'] = 'ma86'
    optimiser.options['OF_ma86_scaling'] = 'none'
    for test_vid in tqdm(tests):
        # Force garbage collection so that the repeated model creation does not overflow the memory!
        gc.collect()
        current_dir = test_vid.split('/cheetah_videos/')[1]
        # Filter parameters based on past experience.
        if current_dir in bad_videos:
            # Skip these videos because of erroneous input data.
            continue
        start = 1
        end = -1
        if current_dir in set(manually_selected_frames.keys()):
            start = manually_selected_frames[current_dir][0]
            end = manually_selected_frames[current_dir][1]
        try:
            run(root_dir,
                current_dir,
                start_frame=start,
                end_frame=end,
                dlc_thresh=0.5,
                enable_shutter_delay=True,
                enable_ppms=True if 'flick' in current_dir else False,
                opt=optimiser,
                generate_reprojection_videos=True,
                out_dir_prefix=out_dir_prefix)
        except Exception:
            run(root_dir,
                current_dir,
                start_frame=-1,
                end_frame=1,
                dlc_thresh=0.5,
                enable_shutter_delay=True,
                enable_ppms=True if 'flick' in current_dir else False,
                opt=optimiser,
                generate_reprojection_videos=True,
                out_dir_prefix=out_dir_prefix)

    time1 = time()
    print(f'Run through all videos took {time1 - time0:.2f}s')

def acinoset_comparison(root_dir: str, use_3D_gt: bool = False) -> Dict:
    """
    Generates results for a subset of videos from AcinoSet for each method: FTE, SD-FTE, PW-FTE, PW-SD-FTE.

    Args:
        root_dir: The root directory where the videos are stored, along with the `pose_3d_functions.pickle` file, and `gt_labels` directory.
        use_3D_gt: Flag to select 3D ground truth for evaluation. Defaults to False.

    Returns:
        results in a dictionary.
    """
    if not use_3D_gt:
        data_paths = [os.path.join('2019_03_09', 'jules', 'flick2'), os.path.join('2019_03_09', 'lily', 'flick'),
        os.path.join('2017_12_16', 'bottom', 'phantom', 'flick2_1'), os.path.join('2017_09_03', 'top', 'zorro', 'flick1_1')]
        frames = [(80, 180), (10, 110), (140, 240), (60, 200)]
    else:
        data_paths = [os.path.join('2019_03_09', 'jules', 'flick2'), os.path.join('2017_09_03', 'top', 'zorro', 'flick1_1')]
        frames = [(80, 180), (60, 200)]
    dlc_thresh = 0.5
    # Initialise the Ipopt solver.
    optimiser = SolverFactory('ipopt', executable='/home/zico/lib/ipopt/build/bin/ipopt')
    # solver options
    optimiser.options['print_level'] = 5
    optimiser.options['max_iter'] = 400
    optimiser.options['max_cpu_time'] = 10000
    optimiser.options['Tol'] = 1e-1
    optimiser.options['OF_print_timing_statistics'] = 'yes'
    optimiser.options['OF_print_frequency_time'] = 10
    optimiser.options['OF_hessian_approximation'] = 'limited-memory'
    optimiser.options['OF_accept_every_trial_step'] = 'yes'
    optimiser.options['linear_solver'] = 'ma86'
    optimiser.options['OF_ma86_scaling'] = 'none'

    results = {'fte': {}, 'sd_fte': {}, 'pw_fte': {}, 'pw_sd_fte': {}}
    for i, data_path in enumerate(data_paths):
        for test in results.keys():
            # Run the optimisation
            run(root_dir,
                data_path,
                start_frame=frames[i][0],
                end_frame=frames[i][1],
                dlc_thresh=dlc_thresh,
                opt=optimiser,
                enable_shutter_delay=True if 'sd' in test else False,
                enable_ppms=True if 'pw' in test else False)
            # Produce results
            results[test][data_path] = metrics(root_dir,
                                            data_path,
                                            start_frame=frames[i][0],
                                            end_frame=frames[i][1],
                                            dlc_thresh=dlc_thresh,
                                            use_3D_gt=use_3D_gt,
                                            type_3D_gt=test)
    return results

def metrics(
    root_dir: str,
    data_path: str,
    start_frame: int,
    end_frame: int,
    dlc_thresh: float = 0.5,
    use_3D_gt: bool = False,
    type_3D_gt: str = "fte",
    out_dir_prefix: str = None,
) -> Tuple[float, float, float]:
    """
    Generate metrics for a particular reconstruction. Note, the `fte.pickle` needs to be generated prior to calling this function.

    Args:
        root_dir: The root directory where the videos are stored.
        data_path: Path to video set of interest.
        start_frame: The start frame number. Note, this value is deducted by `-1` to compensate for `0` based indexing.
        end_frame: The end frame number.
        dlc_thresh: The DLC confidence score to filter 2D keypoints. Defaults to 0.5.
        use_3D_gt: Flag to select 3D ground truth for evaluation. Defaults to False.
        type_3D_gt: Sets the type of 3D ground truth to expect. Valid values are fte, pw_fte, sd_fte, pw_sd_fte.
        out_dir_prefix: Used to change the output directory from the root. This is often used if you have the cheetah data stored in a location and you want the output to be saved elsewhere.

    Returns:
        A tuple consisting of the mean error [px], median error [px], and PCK [%].
    """
    if out_dir_prefix:
        out_dir = os.path.join(out_dir_prefix, data_path, type_3D_gt)
    else:
        out_dir = os.path.join(root_dir, data_path, type_3D_gt)
    # load DLC data
    data_dir = os.path.join(root_dir, data_path)
    dlc_dir = os.path.join(data_dir, 'dlc')
    assert os.path.exists(dlc_dir)

    try:
        k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(data_dir)
    except Exception:
        print('Early exit because extrinsic calibration files could not be located')
        return []
    d_arr = d_arr.reshape((-1, 4))

    dlc_points_fpaths = sorted(glob(os.path.join(dlc_dir, '*.h5')))
    assert n_cams == len(dlc_points_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'

    # calculate residual error
    states = utils.load_pickle(os.path.join(out_dir, 'fte.pickle'))
    markers = misc.get_markers()
    data_location = data_path.split('/')
    test_data = [loc.capitalize() for loc in data_location]
    gt_name = str.join('', test_data)
    gt_name = gt_name.replace('Top', '').replace('Bottom', '')

    gt_points_fpaths = sorted(glob(os.path.join(os.path.join(root_dir, 'gt_labels', gt_name), '*.h5')))
    if not use_3D_gt and len(gt_points_fpaths) > 0:
        points_2d_df = utils.load_dlc_points_as_df(gt_points_fpaths, verbose=False)
    elif use_3D_gt:
        points_2d_df = pd.read_csv(os.path.join(root_dir, 'gt_labels', gt_name, f'{gt_name}_{type_3D_gt}.csv'))
    else:
        print('No ground truth labels for this test.')
        points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
        points_2d_df = points_2d_df[points_2d_df['likelihood'] > dlc_thresh]  # ignore points with low likelihood

    positions_3ds = misc.get_all_marker_coords_from_states(states, n_cams)
    points_3d_dfs = []
    for positions_3d in positions_3ds:
        frames = np.arange(start_frame - 1, end_frame).reshape((-1, 1))
        n_frames = len(frames)
        points_3d = []
        for i, m in enumerate(markers):
            _pt3d = np.squeeze(positions_3d[:, i, :])
            marker_arr = np.array([m] * n_frames).reshape((-1, 1))
            _pt3d = np.hstack((frames, marker_arr, _pt3d))
            points_3d.append(_pt3d)
        points_3d_df = pd.DataFrame(
            np.vstack(points_3d),
            columns=['frame', 'marker', 'x', 'y', 'z'],
        ).astype({
            'frame': 'int64',
            'marker': 'str',
            'x': 'float64',
            'y': 'float64',
            'z': 'float64'
        })
        points_3d_dfs.append(points_3d_df)
    px_errors = metric.residual_error(points_2d_df, points_3d_dfs, markers,
                                      (k_arr, d_arr, r_arr, t_arr, cam_res, n_cams))
    mean_error, med_error, pck = _save_error_dists(px_errors, out_dir)

    # Calculate the per marker error and save results.
    marker_errors_2d = dict.fromkeys(markers, [])
    # Calculate the number of measurements included for each marker.
    meas_weights = states.get('meas_weight')
    num_included_meas = dict.fromkeys(markers, 0)
    for i, m in enumerate(markers):
        marker_meas_weights = meas_weights[:, :, i, :].flatten()
        num_included_meas[m] = 100.0 * (((marker_meas_weights != 0.0).sum()) / len(marker_meas_weights))
        temp_dist = []
        for k, df in px_errors.items():
            temp_dist += df.query(f'marker == "{m}"')['pixel_residual'].tolist()
        marker_errors_2d[m] = np.asarray(list(map(float, temp_dist)))

    meas_stats_df = pd.DataFrame(num_included_meas, index=['%'])
    error_df = pd.DataFrame(
        pd.DataFrame(dict([(k, pd.Series(v))
                           for k, v in marker_errors_2d.items()])).describe(include='all')).transpose()
    error_df.to_csv(os.path.join(out_dir, 'reprojection_results.csv'))
    meas_stats_df.to_csv(os.path.join(out_dir, 'measurement_results.csv'))

    return mean_error, med_error, pck


def run(root_dir: str,
        data_path: str,
        start_frame: int,
        end_frame: int,
        dlc_thresh: float,
        loss='redescending',
        enable_ppms: bool = False,
        enable_shutter_delay: bool = False,
        opt=None,
        generate_reprojection_videos: bool = False,
        out_dir_prefix: str = None,) -> None:
    """
    Runs the FTE 3D reconstruction.

    Args:
        root_dir: The root directory where the videos are stored.
        data_path: Path to video set of interest.
        start_frame: The start frame number. Note, this value is deducted by `-1` to compensate for `0` based indexing.
        end_frame: The end frame number.
        dlc_thresh: The DLC confidence score to filter 2D keypoints. Defaults to 0.5.
        loss: Select the loss function to use for the measurement cost, either squared loss or redecending. Defaults to 'redescending'.
        enable_ppms: Flag to indicate if the pairwise pseudo-measurements should be incorporated into the optimisation. Defaults to False.
        enable_shutter_delay: Flag to indicate if shutter delay correction should be performed. Defaults to False.
        opt: A instance of the Pyomo optimiser. This is useful if you are calling this function in a loop; preventing re-initialisation of the optimiser. Defaults to None.
        generate_reprojection_videos: Flag to enable the generation of 2D reprojection videos. Defaults to False.
        out_dir_prefix: Used to change the output directory from the root. This is often used if you have the cheetah data stored in a location and you want the output to be saved elsewhere.
    """
    print('Prepare data - Start')

    t0 = time()
    out_dir_name = 'fte'
    if enable_shutter_delay:
        out_dir_name = 'sd_' + out_dir_name
    if enable_ppms:
        out_dir_name = 'pw_' + out_dir_name
    if out_dir_prefix:
        out_dir = os.path.join(out_dir_prefix, data_path, out_dir_name)
    else:
        out_dir = os.path.join(root_dir, data_path, out_dir_name)

    data_dir = os.path.join(root_dir, data_path)
    assert os.path.exists(data_dir)
    dlc_dir = os.path.join(data_dir, 'dlc')
    assert os.path.exists(dlc_dir)
    os.makedirs(out_dir, exist_ok=True)

    app.start_logging(os.path.join(out_dir, 'fte.log'))

    # ========= IMPORT CAMERA & SCENE PARAMS ========
    try:
        k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(data_dir)
    except Exception:
        print('Early exit because extrinsic calibration files could not be located')
        return
    d_arr = d_arr.reshape((-1, 4))

    # load video info
    res, fps, num_frames, _ = app.get_vid_info(data_dir)  # path to the directory having original videos
    assert res == cam_res

    # load DLC data
    dlc_points_fpaths = sorted(glob(os.path.join(dlc_dir, '*.h5')))
    assert n_cams == len(dlc_points_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'

    # load measurement dataframe (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
    filtered_points_2d_df = points_2d_df[points_2d_df['likelihood'] > dlc_thresh]  # ignore points with low likelihood

    assert 0 != start_frame < num_frames, f'start_frame must be strictly between 0 and {num_frames}'
    assert 0 != end_frame <= num_frames, f'end_frame must be less than or equal to {num_frames}'
    assert 0 <= dlc_thresh <= 1, 'dlc_thresh must be from 0 to 1'

    if end_frame == -1:
        # Automatically set start and end frame
        # defining the first and end frame as detecting all the markers on any of cameras simultaneously
        target_markers = misc.get_markers()
        markers_condition = ' or '.join([f'marker=="{ref}"' for ref in target_markers])
        num_marker = lambda i: len(
            filtered_points_2d_df.query(f'frame == {i} and ({markers_condition})')['marker'].unique())

        start_frame, end_frame = -1, -1
        max_idx = points_2d_df['frame'].max() + 1
        for i in range(max_idx):
            if num_marker(i) == len(target_markers):
                start_frame = i
                break
        for i in range(max_idx, 0, -1):
            if num_marker(i) == len(target_markers):
                end_frame = i
                break
        if start_frame == -1 or end_frame == -1:
            raise Exception('Setting frames failed. Please define start and end frames manually.')
    elif start_frame == -1:
        # Use the entire video.
        start_frame = 1
        end_frame = num_frames
    else:
        # User-defined frames
        start_frame = start_frame - 1  # 0 based indexing
        end_frame = end_frame % num_frames + 1 if end_frame == -1 else end_frame
    assert len(k_arr) == points_2d_df['camera'].nunique()

    N = end_frame - start_frame
    Ts = 1.0 / fps  # timestep

    # Check that we have a valid range of frames - if not perform the entire sequence.
    if N == 0:
        end_frame = num_frames
        N = end_frame - start_frame

    # For memory reasons - do not perform optimisation on trajectories larger than 200 points.
    if N > 200:
        end_frame = start_frame + 200
        N = end_frame - start_frame

    ## ========= POSE FUNCTIONS ========
    try:
        pose_to_3d, pos_funcs = utils.load_dill(os.path.join(root_dir, 'pose_3d_functions.pickle'))
    except FileNotFoundError:
        print('Lambdify pose functions and save to file for re-use in the future...')
        _create_pose_functions(root_dir)

    pose_to_3d, pos_funcs = utils.load_dill(os.path.join(root_dir, 'pose_3d_functions.pickle'))
    idx = misc.get_pose_params()
    sym_list = list(idx.keys())

    # ========= PROJECTION FUNCTIONS ========
    def pt3d_to_2d(x, y, z, K, D, R, t):
        x_2d = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t.flatten()[0]
        y_2d = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t.flatten()[1]
        z_2d = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t.flatten()[2]
        #project onto camera plane
        a = x_2d / z_2d
        b = y_2d / z_2d
        #fisheye params
        r = (a**2 + b**2)**0.5
        th = pyo.atan(r)
        #distortion
        th_d = th * (1 + D[0] * th**2 + D[1] * th**4 + D[2] * th**6 + D[3] * th**8)
        x_p = a * th_d / (r + 1e-12)
        y_p = b * th_d / (r + 1e-12)
        u = K[0, 0] * x_p + K[0, 2]
        v = K[1, 1] * y_p + K[1, 2]
        return u, v

    def pt3d_to_x2d(x, y, z, K, D, R, t):
        return pt3d_to_2d(x, y, z, K, D, R, t)[0]

    def pt3d_to_y2d(x, y, z, K, D, R, t):
        return pt3d_to_2d(x, y, z, K, D, R, t)[1]

    # ========= IMPORT DATA ========
    markers = misc.get_markers()
    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    # measurement standard deviation
    R = np.array(
        [
            1.2,  # nose
            1.24,  # l_eye
            1.18,  # r_eye
            2.08,  # neck_base
            2.04,  # spine
            2.52,  # tail_base
            2.73,  # tail1
            1.83,  # tail2
            3.47,  # r_shoulder
            2.75,  # r_front_knee
            2.69,  # r_front_ankle
            2.24,  # r_front_paw
            3.4,  # l_shoulder
            2.91,  # l_front_knee
            2.85,  # l_front_ankle
            2.27,  # l_front_paw
            3.26,  # r_hip
            2.76,  # r_back_knee
            2.33,  # r_back_ankle
            2.4,  # r_back_paw
            3.53,  # l_hip
            2.69,  # l_back_knee
            2.49,  # l_back_ankle
            2.34,  # l_back_paw
        ],
        dtype=float)
    R_pw = np.array([
        R,
        [
            2.71, 3.06, 2.99, 4.07, 5.53, 4.67, 6.05, 5.6, 5.01, 5.11, 5.24, 4.85, 5.18, 5.28, 5.5, 4.9, 4.7, 4.7, 5.21,
            5.11, 5.1, 5.27, 5.75, 5.44
        ],
        [
            2.8, 3.24, 3.42, 3.8, 4.4, 5.43, 5.22, 7.29, 8.19, 6.5, 5.9, 6.18, 8.83, 6.52, 6.22, 6.34, 6.8, 6.12, 5.37,
            5.98, 7.83, 6.44, 6.1, 6.38
        ]
    ],
                    dtype=float)
    # Provides some extra uncertainty to the measurements to accomodate for the rigid body body assumption.
    R_pw *= 1.5

    Q = [  # model parameters variance
        4,
        7,
        5,  # head position in inertial
        13,
        9,
        26,  # head rotation in inertial
        32,
        18,
        12,  # neck
        43,  # front torso
        10,
        53,
        34,  # back torso
        90,
        43,  # tail_base
        118,  # tail_mid
        51,
        247,  # l_shoulder
        186,  # l_front_knee
        194,  # r_shoulder
        164,  # r_front_knee
        295,  # l_hip
        243,  # l_back_knee
        334,  # r_hip
        149,  # r_back_knee
        91,  # l_front_ankle
        91,  # r_front_ankle
        132,  # l_back_ankle
        132  # r_back_ankle
    ]
    Q = np.array(Q, dtype=float)**2

    #===================================================
    #                   Load in data
    #===================================================
    print('Load H5 2D DLC prediction data')
    df_paths = sorted(glob(os.path.join(dlc_dir, '*.h5')))

    points_3d_df = utils.get_pairwise_3d_points_from_df(filtered_points_2d_df, k_arr, d_arr, r_arr, t_arr,
                                                        triangulate_points_fisheye)

    # estimate initial points
    print('Estimate the initial trajectory')
    # Use the cheetahs spine to estimate the initial trajectory with a 3rd degree spline.
    frame_est = np.arange(end_frame)

    nose_pts = points_3d_df[points_3d_df['marker'] == 'nose'][['frame', 'x', 'y', 'z']].values
    nose_pts[:, 1] = nose_pts[:, 1] - 0.055
    nose_pts[:, 3] = nose_pts[:, 3] + 0.055
    traj_est_x = UnivariateSpline(nose_pts[:, 0], nose_pts[:, 1])
    traj_est_y = UnivariateSpline(nose_pts[:, 0], nose_pts[:, 2])
    traj_est_z = UnivariateSpline(nose_pts[:, 0], nose_pts[:, 3])
    x_est = np.array(traj_est_x(frame_est))
    y_est = np.array(traj_est_y(frame_est))
    z_est = np.array(traj_est_z(frame_est))

    # Calculate the initial yaw.
    dx_est = np.diff(x_est) / Ts
    dy_est = np.diff(y_est) / Ts
    psi_est = np.arctan2(dy_est, dx_est)
    # Duplicate the last heading estimate as the difference calculation returns N-1.
    psi_est = np.append(psi_est, [psi_est[-1]])

    # Remove datafames from memory to conserve memory usage.
    del points_2d_df
    del filtered_points_2d_df
    del points_3d_df

    # Obtain base and pairwise measurments.
    pw_data = {}
    base_data = {}
    cam_idx = 0
    for path in df_paths:
        # Pairwise correspondence data.
        h5_filename = os.path.basename(path)
        pw_data[cam_idx] = utils.load_pickle(
            os.path.join(dlc_dir + '_pw', f'{h5_filename[:4]}DLC_resnet152_CheetahOct14shuffle4_650000.pickle'))
        df_temp = pd.read_hdf(os.path.join(dlc_dir, h5_filename))
        base_data[cam_idx] = list(df_temp.to_numpy())
        cam_idx += 1

    # There has been a case where some camera view points have less frames than others.
    # This can cause an issue when using automatic frame selection.
    # Therefore, ensure that the end frame is within range.
    min_num_frames = min([len(val) for val in pw_data.values()])
    if end_frame > min_num_frames:
        end_frame = min_num_frames
        N = end_frame - start_frame

    print('Prepare data - End')

    # save parameters
    with open(os.path.join(out_dir, 'reconstruction_params.json'), 'w') as f:
        json.dump(dict(start_frame=start_frame+1, end_frame=end_frame, dlc_thresh=dlc_thresh), f)

    print(f'Start frame: {start_frame}, End frame: {end_frame}, Frame rate: {fps}')

    #===================================================
    #                   Optimisation
    #===================================================
    print('Setup optimisation - Start')
    m = pyo.ConcreteModel(name='Cheetah from measurements')
    m.Ts = Ts
    # ===== SETS =====
    P = len(list(sym_list))  # number of pose parameters
    L = len(markers)  # number of dlc labels per frame
    C = len(k_arr)  # number of cameras
    m.N = pyo.RangeSet(N)
    m.P = pyo.RangeSet(P)
    m.L = pyo.RangeSet(L)
    m.C = pyo.RangeSet(C)
    # Dimensionality of measurements
    m.D2 = pyo.RangeSet(2)
    m.D3 = pyo.RangeSet(3)
    # Number of pairwise terms to include + the base measurement.
    m.W = pyo.RangeSet(3 if enable_ppms else 1)

    index_dict = misc.get_dlc_marker_indices()
    pair_dict = misc.get_pairwise_graph()

    # ======= WEIGHTS =======
    def init_meas_weights(m, n, c, l, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        cam_idx = c - 1
        marker = markers[l - 1]
        values = pw_data[cam_idx][(n - 1) + start_frame]
        likelihoods = values['pose'][2::3]
        if w < 2:
            base = index_dict[marker]
            likelihoods = base_data[cam_idx][(n - 1) + start_frame][2::3]
        else:
            try:
                base = pair_dict[marker][w - 2]
            except IndexError:
                return 0.0

        # Filter measurements based on DLC threshold.
        # This does ensures that badly predicted points are not considered in the objective function.
        return 1 / R_pw[w - 1][l - 1] if likelihoods[base] > dlc_thresh else 0.0

    m.meas_err_weight = pyo.Param(m.N, m.C, m.L, m.W, initialize=init_meas_weights, mutable=True)
    m.model_err_weight = pyo.Param(m.P, initialize=lambda m, p: 1 / Q[p - 1] if Q[p - 1] != 0.0 else 0.0)

    # ===== PARAMETERS =====
    def init_measurements(m, n, c, l, d2, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        cam_idx = c - 1
        marker = markers[l - 1]
        if w < 2:
            base = index_dict[marker]
            val = base_data[cam_idx][(n - 1) + start_frame][d2 - 1::3]

            return val[base]
        else:
            try:
                values = pw_data[cam_idx][(n - 1) + start_frame]
                val = values['pose'][d2 - 1::3]
                base = pair_dict[marker][w - 2]
                val_pw = values['pws'][:, :, :, d2 - 1]
                return val[base] + val_pw[0, base, index_dict[marker]]
            except IndexError:
                return 0.0

    m.meas = pyo.Param(m.N, m.C, m.L, m.D2, m.W, initialize=init_measurements)

    print('Measurement initialisation...Done')
    # ===== VARIABLES =====
    m.x = pyo.Var(m.N, m.P)  #position
    m.dx = pyo.Var(m.N, m.P)  #velocity
    m.ddx = pyo.Var(m.N, m.P)  #acceleration
    m.poses = pyo.Var(m.N, m.L, m.D3)
    m.slack_model = pyo.Var(m.N, m.P, initialize=0.0)
    m.slack_meas = pyo.Var(m.N, m.C, m.L, m.D2, m.W, initialize=0.0)
    if enable_shutter_delay:
        m.shutter_delay = pyo.Var(m.C, initialize=0.0)

    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N, P))
    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))
    init_x[:, idx['x_0']] = x_est[start_frame:start_frame + N]  #x # change this to [start_frame: end_frame]?
    init_x[:, idx['y_0']] = y_est[start_frame:start_frame + N]  #y
    init_x[:, idx['z_0']] = z_est[start_frame:start_frame + N]  #z
    init_x[:, idx['psi_0']] = psi_est[start_frame:start_frame + N]  # yaw = psi
    for n in m.N:
        for p in m.P:
            if n < len(init_x):  #init using known values
                m.x[n, p].value = init_x[n - 1, p - 1]
                m.dx[n, p].value = init_dx[n - 1, p - 1]
                m.ddx[n, p].value = init_ddx[n - 1, p - 1]
            else:  #init using last known value
                m.x[n, p].value = init_x[-1, p - 1]
                m.dx[n, p].value = init_dx[-1, p - 1]
                m.ddx[n, p].value = init_ddx[-1, p - 1]
        #init pose
        var_list = [m.x[n, p].value for p in range(1, P + 1)]
        for l in m.L:
            [pos] = pos_funcs[l - 1](*var_list)
            for d3 in m.D3:
                m.poses[n, l, d3].value = pos[d3 - 1]

    print('Variable initialisation...Done')

    # ===== CONSTRAINTS =====
    # 3D POSE
    def pose_constraint(m, n, l, d3):
        # Get 3d points
        var_list = [m.x[n, p] for p in range(1, P + 1)]
        [pos] = pos_funcs[l - 1](*var_list)
        return pos[d3 - 1] == m.poses[n, l, d3]

    m.pose_constraint = pyo.Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    # INTEGRATION
    def backwards_euler_pos(m, n, p):  # position
        return m.x[n, p] == m.x[n - 1, p] + m.Ts * m.dx[n, p] if n > 1 else pyo.Constraint.Skip

    m.integrate_p = pyo.Constraint(m.N, m.P, rule=backwards_euler_pos)

    def backwards_euler_vel(m, n, p):  # velocity
        return m.dx[n, p] == m.dx[n - 1, p] + m.Ts * m.ddx[n, p] if n > 1 else pyo.Constraint.Skip

    m.integrate_v = pyo.Constraint(m.N, m.P, rule=backwards_euler_vel)

    # MODEL
    def constant_acc(m, n, p):
        return m.ddx[n, p] == m.ddx[n - 1, p] + m.slack_model[n, p] if n > 1 else pyo.Constraint.Skip

    m.constant_acc = pyo.Constraint(m.N, m.P, rule=constant_acc)

    if enable_shutter_delay:
        m.shutter_base_constraint = pyo.Constraint(rule=lambda m: m.shutter_delay[1] == 0.0)
        m.shutter_delay_constraint = pyo.Constraint(m.C, rule=lambda m, c: (-m.Ts, m.shutter_delay[c], m.Ts))

    # MEASUREMENT
    def measurement_constraints(m, n, c, l, d2, w):
        #project
        cam_idx = c - 1
        tau = m.shutter_delay[c] if enable_shutter_delay else 0.0
        K, D, R, t = k_arr[cam_idx], d_arr[cam_idx], r_arr[cam_idx], t_arr[cam_idx]
        x = m.poses[n, l, _pyo_i(idx['x_0'])] + m.dx[n, _pyo_i(idx['x_0'])] * tau + m.ddx[n, _pyo_i(idx['x_0'])] * (tau**2)
        y = m.poses[n, l, _pyo_i(idx['y_0'])] + m.dx[n, _pyo_i(idx['y_0'])] * tau + m.ddx[n, _pyo_i(idx['y_0'])] * (tau**2)
        z = m.poses[n, l, _pyo_i(idx['z_0'])] + m.dx[n, _pyo_i(idx['z_0'])] * tau + m.ddx[n, _pyo_i(idx['z_0'])] * (tau**2)

        return proj_funcs[d2 - 1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2, w] - m.slack_meas[n, c, l, d2, w] == 0.0

    m.measurement = pyo.Constraint(m.N, m.C, m.L, m.D2, m.W, rule=measurement_constraints)

    #===== POSE CONSTRAINTS (Note 1 based indexing for pyomo!!!!...@#^!@#&) =====
    # Head
    m.head_phi_0 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx['phi_0'])], np.pi / 6))
    m.head_theta_0 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx['theta_0'])], np.pi / 6))
    # Neck
    m.neck_phi_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 2, m.x[n, _pyo_i(idx['phi_1'])], np.pi / 2))
    m.neck_theta_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx['theta_1'])], np.pi / 6))
    m.neck_psi_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx['psi_1'])], np.pi / 6))
    # Front torso
    m.front_torso_theta_2 = pyo.Constraint(m.N,
                                           rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx['theta_2'])], np.pi / 6))
    # Back torso
    m.back_torso_theta_3 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx['theta_3'])], np.pi / 6))
    m.back_torso_phi_3 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx['phi_3'])], np.pi / 6))
    m.back_torso_psi_3 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx['psi_3'])], np.pi / 6))
    # Tail base
    m.tail_base_theta_4 = pyo.Constraint(m.N,
                                         rule=lambda m, n: (-(2 / 3) * np.pi, m.x[n, _pyo_i(idx['theta_4'])],
                                                            (2 / 3) * np.pi))
    m.tail_base_psi_4 = pyo.Constraint(m.N,
                                       rule=lambda m, n: (-(2 / 3) * np.pi, m.x[n, _pyo_i(idx['psi_4'])],
                                                          (2 / 3) * np.pi))
    # Tail mid
    m.tail_mid_theta_5 = pyo.Constraint(m.N,
                                        rule=lambda m, n: (-(2 / 3) * np.pi, m.x[n, _pyo_i(idx['theta_5'])],
                                                           (2 / 3) * np.pi))
    m.tail_mid_psi_5 = pyo.Constraint(m.N,
                                      rule=lambda m, n: (-(2 / 3) * np.pi, m.x[n, _pyo_i(idx['psi_5'])],
                                                         (2 / 3) * np.pi))
    # Front left leg
    m.l_shoulder_theta_6 = pyo.Constraint(m.N,
                                          rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, _pyo_i(idx['theta_6'])],
                                                             (3 / 4) * np.pi))
    m.l_front_knee_theta_7 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi, m.x[n, _pyo_i(idx['theta_7'])], 0))
    # Front right leg
    m.r_shoulder_theta_8 = pyo.Constraint(m.N,
                                          rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, _pyo_i(idx['theta_8'])],
                                                             (3 / 4) * np.pi))
    m.r_front_knee_theta_9 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi, m.x[n, _pyo_i(idx['theta_9'])], 0))
    # Back left leg
    m.l_hip_theta_10 = pyo.Constraint(m.N,
                                      rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, _pyo_i(idx['theta_10'])],
                                                         (3 / 4) * np.pi))
    m.l_back_knee_theta_11 = pyo.Constraint(m.N, rule=lambda m, n: (0, m.x[n, _pyo_i(idx['theta_11'])], np.pi))
    # Back right leg
    m.r_hip_theta_12 = pyo.Constraint(m.N,
                                      rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, _pyo_i(idx['theta_12'])],
                                                         (3 / 4) * np.pi))

    m.r_back_knee_theta_13 = pyo.Constraint(m.N, rule=lambda m, n: (0, m.x[n, _pyo_i(idx['theta_13'])], np.pi))
    m.l_front_ankle_theta_14 = pyo.Constraint(m.N,
                                              rule=lambda m, n: (-np.pi / 4, m.x[n, _pyo_i(idx['theta_14'])],
                                                                 (3 / 4) * np.pi))
    m.r_front_ankle_theta_15 = pyo.Constraint(m.N,
                                              rule=lambda m, n: (-np.pi / 4, m.x[n, _pyo_i(idx['theta_15'])],
                                                                 (3 / 4) * np.pi))
    m.l_back_ankle_theta_16 = pyo.Constraint(m.N,
                                             rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, _pyo_i(idx['theta_16'])], 0))
    m.r_back_ankle_theta_17 = pyo.Constraint(m.N,
                                             rule=lambda m, n: (-(3 / 4) * np.pi, m.x[n, _pyo_i(idx['theta_17'])], 0))

    print('Constaint initialisation...Done')

    # ======= OBJECTIVE FUNCTION =======

    def obj(m):
        slack_model_err = 0.0
        slack_meas_err = 0.0
        for n in m.N:
            #Model Error
            for p in m.P:
                slack_model_err += m.model_err_weight[p] * m.slack_model[n, p]**2
            #Measurement Error
            for l in m.L:
                for c in m.C:
                    for d2 in m.D2:
                        for w in m.W:
                            slack_meas_err += _loss_function(
                                m.meas_err_weight[n, c, l, w] * m.slack_meas[n, c, l, d2, w], loss)
        return 1e-3 * (slack_meas_err + slack_model_err)

    m.obj = pyo.Objective(rule=obj)

    print('Objective initialisation...Done')
    # RUN THE SOLVER
    if opt is None:
        opt = SolverFactory(
            'ipopt',  #executable='/home/zico/lib/ipopt/build/bin/ipopt'
        )
        # solver options
        opt.options['print_level'] = 5
        opt.options['max_iter'] = 400
        opt.options['max_cpu_time'] = 10000
        opt.options['Tol'] = 1e-1
        opt.options['OF_print_timing_statistics'] = 'yes'
        opt.options['OF_print_frequency_time'] = 10
        opt.options['OF_hessian_approximation'] = 'limited-memory'
        opt.options['OF_accept_every_trial_step'] = 'yes'
        opt.options['linear_solver'] = 'ma86'
        opt.options['OF_ma86_scaling'] = 'none'

    print('Setup optimisation - End')
    t1 = time()
    print(f'Initialisation took {t1 - t0:.2f}s')

    t0 = time()
    opt.solve(m, tee=True)
    t1 = time()
    print(f'Optimisation solver took {t1 - t0:.2f}s')

    app.stop_logging()

    print('Generate outputs...')
    if enable_shutter_delay:
        print('shutter delay:', [m.shutter_delay[c].value for c in m.C])

    # ===== SAVE FTE RESULTS =====
    x_optimised = _get_vals_v(m.x, [m.N, m.P])
    dx_optimised = _get_vals_v(m.dx, [m.N, m.P])
    ddx_optimised = _get_vals_v(m.ddx, [m.N, m.P])
    positions = [pose_to_3d(*states) for states in x_optimised]
    model_weight = _get_vals_v(m.model_err_weight, [m.P])
    model_err = _get_vals_v(m.slack_model, [m.N, m.P])
    meas_err = _get_vals_v(m.slack_meas, [m.N, m.C, m.L, m.D2, m.W])
    meas_weight = _get_vals_v(m.meas_err_weight, [m.N, m.C, m.L, m.W])
    shutter_delay = _get_vals_v(m.shutter_delay, [m.C]) if enable_shutter_delay else None

    states = dict(x=x_optimised,
                  dx=dx_optimised,
                  ddx=ddx_optimised,
                  model_err=model_err,
                  model_weight=model_weight,
                  meas_err=meas_err,
                  meas_weight=meas_weight,
                  shutter_delay=shutter_delay)

    positions_3ds = misc.get_all_marker_coords_from_states(states, n_cams)

    del m

    out_fpath = os.path.join(out_dir, 'fte.pickle')
    utils.save_optimised_cheetah(positions, out_fpath, extra_data=dict(**states, start_frame=start_frame))
    utils.save_3d_cheetah_as_2d(positions_3ds,
                                out_dir,
                                scene_fpath,
                                markers,
                                project_points_fisheye,
                                start_frame,
                                out_fname='fte')

    # Create 2D reprojection videos.
    if generate_reprojection_videos:
        video_paths = sorted(glob(os.path.join(root_dir, data_path,
                                               'cam[1-9].mp4')))  # original vids should be in the parent dir
        app.create_labeled_videos(video_paths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)

    print('Done')

def _create_pose_functions(data_dir: str):
    # symbolic vars
    idx = misc.get_pose_params()
    sym_list = sp.symbols(list(idx.keys()))
    positions = misc.get_3d_marker_coords(sym_list)

    func_map = {'sin': pyo.sin, 'cos': pyo.cos, 'ImmutableDenseMatrix': np.array}
    pose_to_3d = sp.lambdify(sym_list, positions, modules=[func_map])
    pos_funcs = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i, :], modules=[func_map])
        pos_funcs.append(lamb)

    # Save the functions to file.
    utils.save_dill(os.path.join(data_dir, 'pose_3d_functions.pickle'), (pose_to_3d, pos_funcs))

def _save_error_dists(px_errors, output_dir: str) -> Tuple[float, float, float]:
    ratio = 0.5
    distances = []
    pck_threshold = []
    for k, df in px_errors.items():
        distances += df['pixel_residual'].tolist()
        pck_threshold += df['pck_threshold'].tolist()
    distances = np.asarray(list(map(float, distances)))
    pck_threshold = np.asarray(list(map(float, pck_threshold)))

    pck = 100.0 * (np.sum(distances <= (ratio * pck_threshold)) / len(distances))
    mean_error = float(np.mean(distances))
    med_error = float(np.median(distances))
    utils.save_pickle(os.path.join(output_dir, 'reprojection.pickle'), {'error': distances, 'mean_error': mean_error, 'med_error': med_error, 'pck': pck})

    # plot the error histogram
    xlabel = 'Error [px]'
    ylabel = 'Frequency'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(distances, bins=50)
    ax.set_title(
        f'Error Overview (N={len(distances)}, \u03BC={mean_error:.3f}, med={med_error:.3f}, PCK={pck:.3f})'
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(os.path.join(output_dir, 'overall_error_hist.pdf'))

    hist_data = []
    labels = []
    for k, df in px_errors.items():
        i = int(k)
        e = df['pixel_residual'].tolist()
        e = list(map(float, e))
        hist_data.append(e)
        labels.append('cam{} (N={})'.format(i + 1, len(e)))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(hist_data, bins=10, density=True, histtype='bar')
    ax.legend(labels)
    ax.set_title('Reprojection Pixel Error')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(os.path.join(output_dir, 'cams_error_hist.pdf'))

    return mean_error, med_error, pck

def _get_vals_v(var: Union[pyo.Var, pyo.Param], idxs: list) -> np.ndarray:
    '''
    Verbose version that doesn't try to guess stuff for ya. Usage:
    >>> get_vals(m.q, (m.N, m.DOF))
    '''
    arr = np.array([pyo.value(var[idx]) for idx in var]).astype(float)
    return arr.reshape(*(len(i) for i in idxs))


def _pyo_i(i: int) -> int:
    return i + 1


def _loss_function(residual: float, loss='redescending') -> float:
    if loss == 'redescending':
        return misc.redescending_loss(residual, 3, 10, 20)
    elif loss == 'lsq':
        return residual**2

    return 0.0

# ========= MAIN ========
if __name__ == '__main__':
    parser = ArgumentParser(description='All Optimizations')
    parser.add_argument('--run', action='store_true', help='Run reconstruction over all videos in AcinoSet.')
    parser.add_argument('--eval', action='store_true', help='Evaluate reconstruction over a subset of videos in AcinoSet.')
    args = parser.parse_args()

    root_dir = os.path.join('/', 'data', 'dlc', 'to_analyse', 'cheetah_videos')

    if args.eval:
        results = acinoset_comparison(root_dir)
        utils.save_pickle(os.path.join(root_dir, 'acinoset_comparison_results.pickle'), results)
        results_table = pd.DataFrame.from_dict({(i,j): results[i][j] for i in results.keys() for j in results[i].keys()}, orient='index', columns=['Mean Error', 'Median Error', 'PCK'])
        print(results_table)
        results_table.to_csv(os.path.join(root_dir, 'acinoset_comparison_results.csv'))

    if args.run:
        video_list = utils.load_pickle('/data/zico/CheetahResults/test_videos_list.pickle')
        dir_prefix = '/data/zico/CheetahResults/pw-sd-fte'
        run_acinoset(root_dir, video_data=video_list, out_dir_prefix=dir_prefix)
