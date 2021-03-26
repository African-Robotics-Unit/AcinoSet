import os
import pickle
import argparse
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from glob import glob
from time import time
from scipy.stats import linregress
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from lib import misc, utils, app
from lib.calib import triangulate_points_fisheye


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="the path of the dir containing videos")
parser.add_argument("--config", default=os.path.join('..', 'configs', 'mplstyle.yaml'), type=str, help="the path of config file")
parser.add_argument("--start_frame", type=int, default=50)
parser.add_argument("--end_frame", type=int, default=115)
parser.add_argument("--dlc_thresh", type=int, default=0.8, help="change this only if FTE result is unsatisfactory")
parser.add_argument("--verbosity", action="count", default=0)
args = parser.parse_args()


plt.style.use(args.config)

# ROOT_DATA_DIR = os.path.join("..", "data")
# DATA_DIR = os.path.join(ROOT_DATA_DIR, "2019_03_07", "phantom", "flick")


if __name__ == "__main__":
    # reconstruction parameters
    start_frame = 50
    end_frame = 115

    # DLC p_cutoff - any points with likelihood < dlc_thresh are not trusted in FTE
    dlc_thresh = 0.8  # change this only if FTE result is unsatisfactory

    # robust cost function
    # PLOT OF REDESCENDING, ABSOLUTE AND QUADRATIC COST FUNCTIONS
    # we use a redescending cost to stop outliers affecting the optimisation negatively
    redesc_a = 3
    redesc_b = 10
    redesc_c = 20

    # optimization
    data_dir = args.data
    assert os.path.exists(data_dir)
    out_dir = os.path.join(data_dir, 'fte')
    dlc_dir = os.path.join(data_dir, 'dlc')
    assert os.path.exists(dlc_dir)
    os.makedirs(out_dir, exist_ok=True)

    app.start_logging(os.path.join(out_dir, 'fte.log'))

    t0 = time()

    start_frame -= 1    # 0 based indexing
    N = end_frame - start_frame
    Ts = 1/120 if '2019' in data_dir else 1/90 # timestep
    print(f"Framerate: {1/Ts} fps")

    # SYMBOLIC CHEETAH POSE POSITIONS
    idx = misc.get_pose_params()
    x = sp.symbols(list(idx.keys()))
    positions = misc.get_3d_marker_coords(x)

    # ========= LAMBDIFY SYMBOLIC FUNCTIONS ========
    func_map = {"sin": sin, "cos": cos, "ImmutableDenseMatrix": np.array}
    pose_to_3d = sp.lambdify(x, positions, modules=[func_map])
    pos_funcs = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(x, positions[i,:], modules=[func_map])
        pos_funcs.append(lamb)

    # ========= IMPORT CAMERA & SCENE PARAMS ========
    K_arr, D_arr, R_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(data_dir)
    D_arr = D_arr.reshape((-1,4))

    # ========= IMPORT DATA ========
    R = 5 # measurement standard deviation
    Q_list = [ # model parameters variance
        4, 7, 5,    # head position in inertial
        13, 9, 26,  # head rotation in inertial
        32, 18, 12, # neck
        43,         # front torso
        10, 53, 34, # back torso
        90, 43,     # tail_base
        118, 51,    # tail_mid
        247, 186,   # l_shoulder, l_front_knee
        194, 164,   # r_shoulder, r_front_knee
        295, 243,   # l_hip, l_back_knee
        334, 149    # r_hip, r_back_knee
    ]
    Q_list += Q_list[0:3] # lure's x, y, z variance - same as head
    Q = np.array(Q_list, dtype=np.float64)**2

    markers = misc.get_markers()

    def get_meas_from_df(n, c, l, d):
        n_mask = points_2d_df["frame"] == n-1
        l_mask = points_2d_df["marker"] == markers[l-1]
        c_mask = points_2d_df["camera"] == c-1
        d_idx = {1:"x", 2:"y"}
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val[d_idx[d]].values[0]

    def get_likelihood_from_df(n, c, l):
        n_mask = points_2d_df["frame"] == n-1
        l_mask = points_2d_df["marker"] == markers[l-1]
        c_mask = points_2d_df["camera"] == c-1
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val["likelihood"].values[0]

    # ========= POSE FUNCTIONS ========
    def pt3d_to_2d(x, y, z, K, D, R, t):
        x_2d = x*R[0,0] + y*R[0,1] + z*R[0,2] + t.flatten()[0]
        y_2d = x*R[1,0] + y*R[1,1] + z*R[1,2] + t.flatten()[1]
        z_2d = x*R[2,0] + y*R[2,1] + z*R[2,2] + t.flatten()[2]
        #project onto camera plane
        a = x_2d/z_2d
        b = y_2d/z_2d
        #fisheye params
        r = (a**2 + b**2 +1e-12)**0.5
        th = atan(r)
        #distortion
        th_D = th * (1 + D[0]*th**2 + D[1]*th**4 + D[2]*th**6 + D[3]*th**8)
        x_P = a*th_D/r
        y_P = b*th_D/r
        u = K[0,0]*x_P + K[0,2]
        v = K[1,1]*y_P + K[1,2]
        return u, v

    # ========= PROJECTION FUNCTIONS ========
    def pt3d_to_x2d(x, y, z, K, D, R, t):
        u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
        return u

    def pt3d_to_y2d(x, y, z, K, D, R, t):
        v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
        return v

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    #===================================================
    #                   Load in data
    #===================================================
    print("Loading data")

    df_paths = glob(os.path.join(dlc_dir, '*.h5'))

    points_2d_df = utils.load_dlc_points_as_df(df_paths)
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df[points_2d_df['likelihood']>dlc_thresh],
        K_arr, D_arr, R_arr, t_arr,
        triangulate_points_fisheye
    )

    #===================================================
    #                   Optimisation
    #===================================================
    print("\nStarted Optimisation")
    m = ConcreteModel(name = "Cheetah from measurements")
    m.Ts = Ts

    # ===== SETS =====
    N = end_frame - start_frame     # number of timesteps in trajectory
    P = len(x)                # number of pose parameters
    L = len(pos_funcs)        # number of dlc labels per frame
    C = n_cams                # number of cameras
    D2 = 2                    # dimensionality of measurements (image points)
    D3 = 3                    # dimensionality of measurements (3d points)

    m.N = RangeSet(N)
    m.P = RangeSet(P)
    m.L = RangeSet(L)
    m.C = RangeSet(C)
    m.D2 = RangeSet(D2)
    m.D3 = RangeSet(D3)

    # ======= WEIGHTS =======
    def init_meas_weights(model, n, c, l):
        likelihood = get_likelihood_from_df(n+start_frame, c, l)
        if likelihood > dlc_thresh:
            return 1/R
        else:
            return 0
    m.meas_err_weight = Param(m.N, m.C, m.L, initialize=init_meas_weights, mutable=True)  # IndexError: index 0 is out of bounds for axis 0 with size 0 means that N is too large

    def init_model_weights(m, p):
        if Q[p-1] != 0.0:
            return 1/Q[p-1]
        else:
            return 0
    m.model_err_weight = Param(m.P, initialize=init_model_weights)

    # ===== PARAMETERS =====
    print("Initialising params & variables")

    def init_measurements_df(m, n, c, l, d2):
        return get_meas_from_df(n+start_frame, c, l, d2)
    m.meas = Param(m.N, m.C, m.L, m.D2, initialize=init_measurements_df)

    # ===== VARIABLES =====
    m.x = Var(m.N, m.P)   # position
    m.dx = Var(m.N, m.P)  # velocity
    m.ddx = Var(m.N, m.P) # acceleration
    m.poses = Var(m.N, m.L, m.D3)
    m.slack_model = Var(m.N, m.P)
    m.slack_meas = Var(m.N, m.C, m.L, m.D2, initialize=0.0)

    # ===== VARIABLES INITIALIZATION =====

    # estimate initial points
    frame_est = np.arange(end_frame)
    init_x = np.zeros((N, P))
    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))

    lure_pts = points_3d_df[points_3d_df["marker"]=="lure"][["frame", "x", "y", "z"]].values
    lure_x_slope, lure_x_intercept, *_ = linregress(lure_pts[:,0], lure_pts[:,1])
    lure_y_slope, lure_y_intercept, *_ = linregress(lure_pts[:,0], lure_pts[:,2])
    lure_z_slope, lure_z_intercept, *_ = linregress(lure_pts[:,0], lure_pts[:,3])
    lure_x_est = frame_est*lure_x_slope + lure_x_intercept
    lure_y_est = frame_est*lure_y_slope + lure_y_intercept
    lure_z_est = frame_est*lure_z_slope + lure_z_intercept

    init_x[:,idx['x_l']] = lure_x_est[start_frame: end_frame] # x
    init_x[:,idx['y_l']] = lure_y_est[start_frame: end_frame] # y
    init_x[:,idx['z_l']] = lure_z_est[start_frame: end_frame] # z

    points_3d_df = points_3d_df[points_3d_df['frame'].between(start_frame, end_frame-1)]

    nose_pts = points_3d_df[points_3d_df["marker"]=="nose"][["frame", "x", "y", "z"]].values
    nose_x_slope, nose_x_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,1])
    nose_y_slope, nose_y_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,2])
    nose_z_slope, nose_z_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,3])
    nose_x_est = frame_est*nose_x_slope + nose_x_intercept
    nose_y_est = frame_est*nose_y_slope + nose_y_intercept
    nose_z_est = frame_est*nose_z_slope + nose_z_intercept
    psi_est = np.arctan2(nose_y_slope, nose_x_slope)

    init_x[:,idx['x_0']] = nose_x_est[start_frame: end_frame] # x
    init_x[:,idx['y_0']] = nose_y_est[start_frame: end_frame] # y
    init_x[:,idx['z_0']] = nose_z_est[start_frame: end_frame] # z
    init_x[:,idx['psi_0']] = psi_est # yaw = psi

    for n in m.N:
        for p in m.P:
            if n<len(init_x): #init using known values
                m.x[n,p].value = init_x[n-1,p-1]
                m.dx[n,p].value = init_dx[n-1,p-1]
                m.ddx[n,p].value = init_ddx[n-1,p-1]
            else: #init using last known value
                m.x[n,p].value = init_x[-1,p-1]
                m.dx[n,p].value = init_dx[-1,p-1]
                m.ddx[n,p].value = init_ddx[-1,p-1]
        #init pose
        var_list = [m.x[n,p].value for p in range(1, P+1)]
        for l in m.L:
            [pos] = pos_funcs[l-1](*var_list)
            for d3 in m.D3:
                m.poses[n,l,d3].value = pos[d3-1]

    # ===== CONSTRAINTS =====
    # 3D POSE
    def pose_constraint(m,n,l,d3):
        #get 3d points
        var_list = [m.x[n,p] for p in range(1, P+1)]
        [pos] = pos_funcs[l-1](*var_list)
        return pos[d3-1] == m.poses[n,l,d3]
    m.pose_constraint = Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    # INTEGRATION
    print("Initialising numerical integration\n")
    def backwards_euler_pos(m,n,p): # position
        if n > 1:
            return m.x[n,p] == m.x[n-1,p] + m.Ts*m.dx[n,p]

        else:
            return Constraint.Skip
    m.integrate_p = Constraint(m.N, m.P, rule = backwards_euler_pos)

    def backwards_euler_vel(m,n,p): # velocity
        if n > 1:
            return m.dx[n,p] == m.dx[n-1,p] + m.Ts*m.ddx[n,p]
        else:
            return Constraint.Skip
    m.integrate_v = Constraint(m.N, m.P, rule = backwards_euler_vel)

    # MODEL
    def constant_acc(m, n, p):
        if n > 1:
            return m.ddx[n,p] == m.ddx[n-1,p] + m.slack_model[n,p]
        else:
            return Constraint.Skip
    m.constant_acc = Constraint(m.N, m.P, rule = constant_acc)

    # MEASUREMENT
    def measurement_constraints(m, n, c, l, d2):
        #project
        K, D, R, t = K_arr[c-1], D_arr[c-1], R_arr[c-1], t_arr[c-1]
        x, y, z = m.poses[n,l,1], m.poses[n,l,2], m.poses[n,l,3]
        return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] == 0
    m.measurement = Constraint(m.N, m.C, m.L, m.D2, rule = measurement_constraints)

    #===== POSE CONSTRAINTS =====
    # Note 1 based indexing for pyomo!!!!...@#^!@#&
    for state in idx:
        idx[state] += 1

    #Head
    def head_psi_0(m, n):
        return abs(m.x[n,idx['psi_0']]) <= np.pi/6
    m.head_psi_0 = Constraint(m.N, rule=head_psi_0)
    def head_theta_0(m, n):
        return abs(m.x[n,idx['theta_0']]) <= np.pi/6
    m.head_theta_0 = Constraint(m.N, rule=head_theta_0)

    #Neck
    def neck_phi_1(m, n):
        return abs(m.x[n,idx['phi_1']]) <= np.pi/6
    m.neck_phi_1 = Constraint(m.N, rule=neck_phi_1)
    def neck_theta_1(m, n):
        return abs(m.x[n,idx['theta_1']]) <= np.pi/6
    m.neck_theta_1 = Constraint(m.N, rule=neck_theta_1)
    def neck_psi_1(m, n):
        return abs(m.x[n,idx['psi_1']]) <= np.pi/6
    m.neck_psi_1 = Constraint(m.N, rule=neck_psi_1)

    #Front torso
    def front_torso_theta_2(m, n):
        return abs(m.x[n,idx['theta_2']]) <= np.pi/6
    m.front_torso_theta_2 = Constraint(m.N, rule=front_torso_theta_2)

    #Back torso
    def back_torso_theta_3(m, n):
        return abs(m.x[n,idx['theta_3']]) <= np.pi/6
    m.back_torso_theta_3 = Constraint(m.N, rule=back_torso_theta_3)
    def back_torso_phi_3(m, n):
        return abs(m.x[n,idx['phi_3']]) <= np.pi/6
    m.back_torso_phi_3 = Constraint(m.N, rule=back_torso_phi_3)
    def back_torso_psi_3(m, n):
        return abs(m.x[n,idx['psi_3']]) <= np.pi/6
    m.back_torso_psi_3 = Constraint(m.N, rule=back_torso_psi_3)

    #Tail base
    def tail_base_theta_4(m, n):
        return abs(m.x[n,idx['theta_4']]) <= np.pi/1.5
    m.tail_base_theta_4 = Constraint(m.N, rule=tail_base_theta_4)
    def tail_base_psi_4(m, n):
        return abs(m.x[n,idx['psi_4']]) <= np.pi/1.5
    m.tail_base_psi_4 = Constraint(m.N, rule=tail_base_psi_4)

    #Tail mid
    def tail_mid_theta_5(m, n):
        return abs(m.x[n,idx['theta_5']]) <= np.pi/1.5
    m.tail_mid_theta_5 = Constraint(m.N, rule=tail_mid_theta_5)
    def tail_mid_psi_5(m, n):
        return abs(m.x[n,idx['psi_5']]) <= np.pi/1.5
    m.tail_mid_psi_5 = Constraint(m.N, rule=tail_mid_psi_5)

    #Front left leg
    def l_shoulder_theta_6(m, n):
        return abs(m.x[n,idx['theta_6']]) <= np.pi/2
    m.l_shoulder_theta_6 = Constraint(m.N, rule=l_shoulder_theta_6)
    def l_front_knee_theta_7(m, n):
        return abs(m.x[n,idx['theta_7']] + np.pi/2) <= np.pi/2
    m.l_front_knee_theta_7 = Constraint(m.N, rule=l_front_knee_theta_7)

    #Front right leg
    def r_shoulder_theta_8(m, n):
        return abs(m.x[n,idx['theta_8']]) <= np.pi/2
    m.r_shoulder_theta_8 = Constraint(m.N, rule=r_shoulder_theta_8)
    def r_front_knee_theta_9(m, n):
        return abs(m.x[n,idx['theta_9']] + np.pi/2) <= np.pi/2
    m.r_front_knee_theta_9 = Constraint(m.N, rule=r_front_knee_theta_9)

    #Back left leg
    def l_hip_theta_10(m, n):
        return abs(m.x[n,idx['theta_10']]) <= np.pi/2
    m.l_hip_theta_10 = Constraint(m.N, rule=l_hip_theta_10)
    def l_back_knee_theta_11(m, n):
        return abs(m.x[n,idx['theta_11']] - np.pi/2) <= np.pi/2
    m.l_back_knee_theta_11 = Constraint(m.N, rule=l_back_knee_theta_11)

    #Back right leg
    def r_hip_theta_12(m, n):
        return abs(m.x[n,idx['theta_12']]) <= np.pi/2
    m.r_hip_theta_12 = Constraint(m.N, rule=r_hip_theta_12)
    def r_back_knee_theta_13(m, n):
        return abs(m.x[n,idx['theta_13']] - np.pi/2) <= np.pi/2
    m.r_back_knee_theta_13 = Constraint(m.N, rule=r_back_knee_theta_13)

    # ======= OBJECTIVE FUNCTION =======
    def obj(m):
        slack_model_err = 0.0
        slack_meas_err = 0.0
        for n in m.N:
            #Model Error
            for p in m.P:
                slack_model_err += m.model_err_weight[p] * m.slack_model[n, p] ** 2
            #Measurement Error
            for l in m.L:
                for c in m.C:
                    for d2 in m.D2:
                        slack_meas_err += misc.redescending_loss(m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2], redesc_a, redesc_b, redesc_c)
        return slack_meas_err + slack_model_err

    m.obj = Objective(rule = obj)

    # RUN THE SOLVER
    opt = SolverFactory(
        'ipopt',    # use this if MA86 solver is not installed
        # executable='./CoinIpopt/build/bin/ipopt'
    )

    # solver options
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 10000
    opt.options["max_cpu_time"] = 3600
    opt.options["tol"] = 1e-1
    opt.options["OF_print_timing_statistics"] = "yes"
    opt.options["OF_print_frequency_iter"] = 10
    opt.options["OF_hessian_approximation"] = "limited-memory"
    # opt.options["linear_solver"] = "ma86"

    t1 = time()
    print("Initialization took {0:.2f} seconds\n".format(t1 - t0))

    t0 = time()
    results = opt.solve(m, tee=True)
    t1 = time()
    print("Optimization took {0:.2f} seconds\n".format(t1 - t0))

    app.stop_logging()

    # save FTE results
    x, dx, ddx =  [], [], []
    for n in m.N:
        x.append([value(m.x[n, p]) for p in m.P])
        dx.append([value(m.dx[n, p]) for p in m.P])
        ddx.append([value(m.ddx[n, p]) for p in m.P])

    app.save_fte(dict(x=x, dx=dx, ddx=ddx), out_dir, start_frame)

    fig_fpath= os.path.join(out_dir, 'fte.svg')
    app.plot_cheetah_states(x, out_fpath=fig_fpath)

    with open(os.path.join(out_dir, 'fte_padded.pickle'), 'rb') as f:
        fte_data = pickle.load(f)
    vid_fpath = os.path.join(out_dir, 'fte.avi')
    app.reconstruction_reprojection_video(data_dir, vid_fpath, np.array(fte_data['positions']))
