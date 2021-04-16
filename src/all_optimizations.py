from argparse import ArgumentParser

# imports from notebooks
import os
import pickle
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from glob import glob
from time import time
from scipy.stats import linregress
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from lib import misc, utils, app
from lib.calib import triangulate_points_fisheye, project_points_fisheye

plt.style.use(os.path.join('..', 'configs', 'mplstyle.yaml'))


def fte(DATA_DIR, DLC_DIR, start_frame, end_frame, dlc_thresh, plot: bool = False):
    # PLOT OF REDESCENDING, ABSOLUTE AND QUADRATIC COST FUNCTIONS
    # we use a redescending cost to stop outliers affecting the optimisation negatively
    redesc_a = 3
    redesc_b = 10
    redesc_c = 20

    # plot
    r_x = np.arange(-20, 20, 1e-1)
    r_y1 = [misc.redescending_loss(i, redesc_a, redesc_b, redesc_c) for i in r_x]
    r_y2 = abs(r_x)
    r_y3 = r_x ** 2
    plt.figure()
    plt.plot(r_x,r_y1, label="Redescending")
    plt.plot(r_x,r_y2, label="Absolute (linear)")
    plt.plot(r_x,r_y3, label="Quadratic")
    ax = plt.gca()
    ax.set_ylim((-5, 50))
    ax.legend()
    if plot:
        plt.show(block=True)

    t0 = time()

    assert os.path.exists(DATA_DIR)
    OUT_DIR = os.path.join(DATA_DIR, 'fte')
    os.makedirs(OUT_DIR, exist_ok=True)

    app.start_logging(os.path.join(OUT_DIR, 'fte.log'))

    # load video info
    res, fps, tot_frames, _ = app.get_vid_info(DATA_DIR) # path to original videos
    assert end_frame <= tot_frames, f'end_frame must be less than or equal to {tot_frames}'
    end_frame = tot_frames if end_frame == -1 else end_frame

    start_frame -= 1    # 0 based indexing
    assert start_frame >= 0
    N = end_frame - start_frame
    Ts = 1.0 / fps  # timestep

    ## ========= POSE FUNCTIONS ========
    #SYMBOLIC ROTATION MATRIX FUNCTIONS
    def rot_x(x):
        c = sp.cos(x)
        s = sp.sin(x)
        return sp.Matrix([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]
        ])

    def rot_y(y):
        c = sp.cos(y)
        s = sp.sin(y)
        return sp.Matrix([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])

    def rot_z(z):
        c = sp.cos(z)
        s = sp.sin(z)
        return sp.Matrix([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])

    L = 14  # number of joints in the cheetah model

    # defines arrays of angles, velocities and accelerations
    phi     = [sp.symbols(f"\\phi_{{{l}}}")   for l in range(L)]
    theta   = [sp.symbols(f"\\theta_{{{l}}}") for l in range(L)]
    psi     = [sp.symbols(f"\\psi_{{{l}}}")   for l in range(L)]

    #ROTATIONS
    RI_0 = rot_z(psi[0]) @ rot_x(phi[0]) @ rot_y(theta[0]) # head
    R0_I = RI_0.T
    RI_1 = rot_z(psi[1]) @ rot_x(phi[1]) @ rot_y(theta[1]) @ RI_0 # neck
    R1_I = RI_1.T
    RI_2 = rot_y(theta[2]) @ RI_1 # front torso
    R2_I = RI_2.T
    RI_3 = rot_z(psi[3])@ rot_x(phi[3]) @ rot_y(theta[3]) @ RI_2 # back torso
    R3_I = RI_3.T
    RI_4 = rot_z(psi[4]) @ rot_y(theta[4]) @ RI_3 # tail base
    R4_I = RI_4.T
    RI_5 = rot_z(psi[5]) @ rot_y(theta[5]) @ RI_4 # tail mid
    R5_I = RI_5.T
    RI_6 = rot_y(theta[6]) @ RI_2 # l_shoulder
    R6_I = RI_6.T
    RI_7 = rot_y(theta[7]) @ RI_6 # l_front_knee
    R7_I = RI_7.T
    RI_8 = rot_y(theta[8]) @ RI_2 # r_shoulder
    R8_I = RI_8.T
    RI_9 = rot_y(theta[9]) @ RI_8 # r_front_knee
    R9_I = RI_9.T
    RI_10 = rot_y(theta[10]) @ RI_3 # l_hip
    R10_I = RI_10.T
    RI_11 = rot_y(theta[11]) @ RI_10 # l_back_knee
    R11_I = RI_11.T
    RI_12 = rot_y(theta[12]) @ RI_3 # r_hip
    R12_I = RI_12.T
    RI_13 = rot_y(theta[13]) @ RI_12 # r_back_knee
    R13_I = RI_13.T

    # defines the position, velocities and accelerations in the inertial frame
    x,   y,   z   = sp.symbols("x y z")
    dx,  dy,  dz  = sp.symbols("\\dot{x} \\dot{y} \\dot{z}")
    ddx, ddy, ddz = sp.symbols("\\ddot{x} \\ddot{y} \\ddot{z}")
    # x_l, y_l, z_l = sp.symbols("x_l y_l z_l") # exclude lure for now


    # SYMBOLIC CHEETAH POSE POSITIONS
    p_head          = sp.Matrix([x, y, z])

    p_l_eye         = p_head         + R0_I  @ sp.Matrix([0, 0.03, 0])
    p_r_eye         = p_head         + R0_I  @ sp.Matrix([0, -0.03, 0])
    p_nose          = p_head         + R0_I  @ sp.Matrix([0.055, 0, -0.055])

    p_neck_base     = p_head         + R1_I  @ sp.Matrix([-0.28, 0, 0])
    p_spine         = p_neck_base    + R2_I  @ sp.Matrix([-0.37, 0, 0])

    p_tail_base     = p_spine        + R3_I  @ sp.Matrix([-0.37, 0, 0])
    p_tail_mid      = p_tail_base    + R4_I  @ sp.Matrix([-0.28, 0, 0])
    p_tail_tip      = p_tail_mid     + R5_I  @ sp.Matrix([-0.36, 0, 0])

    p_l_shoulder    = p_neck_base    + R2_I  @ sp.Matrix([-0.04, 0.08, -0.10])
    p_l_front_knee  = p_l_shoulder   + R6_I  @ sp.Matrix([0, 0, -0.24])
    p_l_front_ankle = p_l_front_knee + R7_I  @ sp.Matrix([0, 0, -0.28])

    p_r_shoulder    = p_neck_base    + R2_I  @ sp.Matrix([-0.04, -0.08, -0.10])
    p_r_front_knee  = p_r_shoulder   + R8_I  @ sp.Matrix([0, 0, -0.24])
    p_r_front_ankle = p_r_front_knee + R9_I  @ sp.Matrix([0, 0, -0.28])

    p_l_hip         = p_tail_base    + R3_I  @ sp.Matrix([0.12, 0.08, -0.06])
    p_l_back_knee   = p_l_hip        + R10_I @ sp.Matrix([0, 0, -0.32])
    p_l_back_ankle  = p_l_back_knee  + R11_I @ sp.Matrix([0, 0, -0.25])

    p_r_hip         = p_tail_base    + R3_I  @ sp.Matrix([0.12, -0.08, -0.06])
    p_r_back_knee   = p_r_hip        + R12_I @ sp.Matrix([0, 0, -0.32])
    p_r_back_ankle  = p_r_back_knee  + R13_I @ sp.Matrix([0, 0, -0.25])

    # p_lure          = sp.Matrix([x_l, y_l, z_l])

    # ========= LAMBDIFY SYMBOLIC FUNCTIONS ========
    positions = sp.Matrix([
        p_l_eye.T, p_r_eye.T, p_nose.T,
        p_neck_base.T, p_spine.T,
        p_tail_base.T, p_tail_mid.T, p_tail_tip.T,
        p_l_shoulder.T, p_l_front_knee.T, p_l_front_ankle.T,
        p_r_shoulder.T, p_r_front_knee.T, p_r_front_ankle.T,
        p_l_hip.T, p_l_back_knee.T, p_l_back_ankle.T,
        p_r_hip.T, p_r_back_knee.T, p_r_back_ankle.T,
    #     p_lure.T
    ])

    func_map = {"sin":sin, "cos":cos, "ImmutableDenseMatrix":np.array}
    sym_list = [x, y, z,
                *phi, *theta, *psi,
    #             x_l, y_l, z_l
               ]
    pose_to_3d = sp.lambdify(sym_list, positions, modules=[func_map])
    pos_funcs = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i,:], modules=[func_map])
        pos_funcs.append(lamb)

    # ========= PROJECTION FUNCTIONS ========
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

    def pt3d_to_x2d(x, y, z, K, D, R, t):
        u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
        return u

    def pt3d_to_y2d(x, y, z, K, D, R, t):
        v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
        return v

    # ========= IMPORT CAMERA & SCENE PARAMS ========
    K_arr, D_arr, R_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR)
    D_arr = D_arr.reshape((-1,4))

    # ========= IMPORT DATA ========
    markers = misc.get_markers()

    def get_meas_from_df(n, c, l, d):
        n_mask = points_2d_df["frame"]== n-1
        l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"]== c-1
        d_idx = {1:"x", 2:"y"}
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val[d_idx[d]].values[0]

    def get_likelihood_from_df(n, c, l):
        n_mask = points_2d_df["frame"]== n-1
        l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"]== c-1
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val["likelihood"].values[0]

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    R = 5 # measurement standard deviation

    Q = [ # model parameters variance
        4, 7, 5, # x, y, z
        13, 32, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, #  phi_1, ... , phi_14
        9, 18, 43, 53, 90, 118, 247, 186, 194, 164, 295, 243, 334, 149, # theta_1, ... , theta_n
        26, 12, 0, 34, 43, 51, 0, 0, 0, 0, 0, 0, 0, 0, # psi_1, ... , psi_n
    #     ?, ?, ? # lure's x, y, z variance
    ]
    Q = np.array(Q, dtype=np.float64)**2

    #===================================================
    #                   Load in data
    #===================================================
    print("Loading data")

    df_paths = sorted(glob(os.path.join(DLC_DIR, '*.h5')))

    points_2d_df = utils.load_dlc_points_as_df(df_paths, verbose=False)
    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df[points_2d_df['likelihood'] > dlc_thresh],
        K_arr, D_arr, R_arr, t_arr,
        triangulate_points_fisheye
    )

    # estimate initial points
    nose_pts = points_3d_df[points_3d_df["marker"]=="nose"][["frame", "x", "y", "z"]].values
    x_slope, x_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,1])
    y_slope, y_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,2])
    z_slope, z_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,3])
    frame_est = np.arange(end_frame)
    x_est = frame_est*x_slope + x_intercept
    y_est = frame_est*y_slope + y_intercept
    z_est = frame_est*z_slope + z_intercept
    psi_est = np.arctan2(y_slope, x_slope)

    #===================================================
    #                   Optimisation
    #===================================================
    print("\nStarted Optimisation")
    m = ConcreteModel(name = "Cheetah from measurements")
    m.Ts = Ts

    # ===== SETS =====
    N = end_frame-start_frame # number of timesteps in trajectory
    P = 3 + len(phi)+len(theta)+len(psi)# + 3  # number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n, x_l, y_l, z_l)
    L = len(pos_funcs) # number of dlc labels per frame
    C = len(K_arr) # number of cameras
    D2 = 2 # dimensionality of measurements
    D3 = 3 # dimensionality of measurements

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
    m.x = Var(m.N, m.P) #position
    m.dx = Var(m.N, m.P) #velocity
    m.ddx = Var(m.N, m.P) #acceleration
    m.poses = Var(m.N, m.L, m.D3)
    m.slack_model = Var(m.N, m.P)
    m.slack_meas = Var(m.N, m.C, m.L, m.D2, initialize=0.0)

    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N, P))
    init_x[:,0] = x_est[start_frame: start_frame+N] #x # change this to [start_frame: end_frame]?
    init_x[:,1] = y_est[start_frame: start_frame+N] #y
    init_x[:,2] = z_est[start_frame: start_frame+N] #z
    init_x[:,31] = psi_est # yaw = psi
    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))
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
    print("Numerical integration")
    def backwards_euler_pos(m,n,p): # position
        if n > 1:
    #             return m.x[n,p] == m.x[n-1,p] + m.h*m.dx[n-1,p] + m.h**2 * m.ddx[n-1,p]/2
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
        return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] ==0
    m.measurement = Constraint(m.N, m.C, m.L, m.D2, rule = measurement_constraints)

    #===== POSE CONSTRAINTS (Note 1 based indexing for pyomo!!!!...@#^!@#&) =====
    #Head
    def head_psi_0(m,n):
        return abs(m.x[n,4]) <= np.pi/6
    m.head_psi_0 = Constraint(m.N, rule=head_psi_0)
    def head_theta_0(m,n):
        return abs(m.x[n,18]) <= np.pi/6
    m.head_theta_0 = Constraint(m.N, rule=head_theta_0)

    #Neck
    def neck_phi_1(m,n):
        return abs(m.x[n,5]) <= np.pi/6
    m.neck_phi_1 = Constraint(m.N, rule=neck_phi_1)
    def neck_theta_1(m,n):
        return abs(m.x[n,19]) <= np.pi/6
    m.neck_theta_1 = Constraint(m.N, rule=neck_theta_1)
    def neck_psi_1(m,n):
        return abs(m.x[n,33]) <= np.pi/6
    m.neck_psi_1 = Constraint(m.N, rule=neck_psi_1)

    #Front torso
    def front_torso_theta_2(m,n):
        return abs(m.x[n,20]) <= np.pi/6
    m.front_torso_theta_2 = Constraint(m.N, rule=front_torso_theta_2)

    #Back torso
    def back_torso_theta_3(m,n):
        return abs(m.x[n,21]) <= np.pi/6
    m.back_torso_theta_3 = Constraint(m.N, rule=back_torso_theta_3)
    def back_torso_phi_3(m,n):
        return abs(m.x[n,7]) <= np.pi/6
    m.back_torso_phi_3 = Constraint(m.N, rule=back_torso_phi_3)
    def back_torso_psi_3(m,n):
        return abs(m.x[n,35]) <= np.pi/6
    m.back_torso_psi_3 = Constraint(m.N, rule=back_torso_psi_3)

    #Tail base
    def tail_base_theta_4(m,n):
        return abs(m.x[n,22]) <= np.pi/1.5
    m.tail_base_theta_4 = Constraint(m.N, rule=tail_base_theta_4)
    def tail_base_psi_4(m,n):
        return abs(m.x[n,36]) <= np.pi/1.5
    m.tail_base_psi_4 = Constraint(m.N, rule=tail_base_psi_4)

    #Tail mid
    def tail_mid_theta_5(m,n):
        return abs(m.x[n,23]) <= np.pi/1.5
    m.tail_mid_theta_5 = Constraint(m.N, rule=tail_mid_theta_5)
    def tail_mid_psi_5(m,n):
        return abs(m.x[n,37]) <= np.pi/1.5
    m.tail_mid_psi_5 = Constraint(m.N, rule=tail_mid_psi_5)

    #Front left leg
    def l_shoulder_theta_6(m,n):
        return abs(m.x[n,24]) <= np.pi/2
    m.l_shoulder_theta_6 = Constraint(m.N, rule=l_shoulder_theta_6)
    def l_front_knee_theta_7(m,n):
        return abs(m.x[n,25] + np.pi/2) <= np.pi/2
    m.l_front_knee_theta_7 = Constraint(m.N, rule=l_front_knee_theta_7)

    #Front right leg
    def r_shoulder_theta_8(m,n):
        return abs(m.x[n,26]) <= np.pi/2
    m.r_shoulder_theta_8 = Constraint(m.N, rule=r_shoulder_theta_8)
    def r_front_knee_theta_9(m,n):
        return abs(m.x[n,27] + np.pi/2) <= np.pi/2
    m.r_front_knee_theta_9 = Constraint(m.N, rule=r_front_knee_theta_9)

    #Back left leg
    def l_hip_theta_10(m,n):
        return abs(m.x[n,28]) <= np.pi/2
    m.l_hip_theta_10 = Constraint(m.N, rule=l_hip_theta_10)
    def l_back_knee_theta_11(m,n):
        return abs(m.x[n,29] - np.pi/2) <= np.pi/2
    m.l_back_knee_theta_11 = Constraint(m.N, rule=l_back_knee_theta_11)

    #Back right leg
    def r_hip_theta_12(m,n):
        return abs(m.x[n,30]) <= np.pi/2
    m.r_hip_theta_12 = Constraint(m.N, rule=r_hip_theta_12)
    def r_back_knee_theta_13(m,n):
        return abs(m.x[n,31] - np.pi/2) <= np.pi/2
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
        'ipopt',
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
    print("\nInitialization took {0:.2f} seconds\n".format(t1 - t0))

    t0 = time()
    results = opt.solve(m, tee=True)
    t1 = time()
    print("\nOptimization took {0:.2f} seconds\n".format(t1 - t0))

    app.stop_logging()

    # ===== SAVE FTE RESULTS =====

    def convert_m(m, pose_indices):
        x_optimised, dx_optimised, ddx_optimised = [], [], []
        for n in m.N:
            x_optimised.append([value(m.x[n, p]) for p in m.P])
            dx_optimised.append([value(m.dx[n, p]) for p in m.P])
            ddx_optimised.append([value(m.ddx[n, p]) for p in m.P])

        positions = [pose_to_3d(*states) for states in x_optimised]

        # remove zero-valued vars
        for n in m.N:
            n -= 1 # remember pyomo's 1-based indexing
            for p in pose_indices[::-1]:
                    assert x_optimised[n][p] == 0
                    del x_optimised[n][p]
                    del dx_optimised[n][p]
                    del ddx_optimised[n][p]

        states = dict(
            x=x_optimised,
            dx=dx_optimised,
            ddx=ddx_optimised,
        )
        return positions, states

    [unused_pose_indices] = np.where(Q == 0)
    positions, states = convert_m(m, unused_pose_indices)

    out_fpath = os.path.join(OUT_DIR, f"fte.pickle")
    app.save_optimised_cheetah(positions, out_fpath, extra_data=dict(**states, start_frame=start_frame))
    app.save_3d_cheetah_as_2d(positions, OUT_DIR, scene_fpath, markers, project_points_fisheye, start_frame)

    video_fpaths = sorted(glob(os.path.join(os.path.dirname(OUT_DIR), 'cam[1-9].mp4'))) # original vids should be in the parent dir
    app.create_labeled_videos(video_fpaths, out_dir=OUT_DIR, draw_skeleton=True, pcutoff=dlc_thresh)

    fig_fpath= os.path.join(OUT_DIR, 'fte.svg')
    app.plot_cheetah_states(states['x'], out_fpath=fig_fpath)


def ekf(DATA_DIR, DLC_DIR, start_frame, end_frame, dlc_thresh):
    # ========= INIT VARS ========

    t0 = time()

    OUT_DIR = os.path.join(DATA_DIR, 'ekf')
    os.makedirs(OUT_DIR, exist_ok=True)

    app.start_logging(os.path.join(OUT_DIR, 'ekf.log'))

    idx = misc.get_pose_params() # define the indices for the states
    markers = misc.get_markers() # define DLC labels

    n_markers = len(markers)
    n_pose_params = len(idx)
    n_states = 3*n_pose_params
    vel_idx = n_states//3
    acc_idx = n_states*2//3

    derivs = {'d'+state: vel_idx+idx[state] for state in idx}
    derivs.update({'d'+state: vel_idx+derivs[state] for state in derivs})
    idx.update(derivs)

    # load video info
    res, fps, tot_frames, _ = app.get_vid_info(DATA_DIR) # path to original videos
    assert end_frame <= tot_frames, f'end_frame must be less than or equal to {tot_frames}'
    end_frame = tot_frames if end_frame == -1 else end_frame

    # Load extrinsic params
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR)
    assert res == cam_res
    camera_params = [[K, D, R, T] for K, D, R, T in zip(k_arr, d_arr, r_arr, t_arr)]

    # other vars
    start_frame -= 1 # 0 based indexing
    assert start_frame >= 0
    n_frames = end_frame-start_frame
    sigma_bound = 3
    max_pixel_err = cam_res[0] # used in measurement covariance R
    sT = 1.0/fps # timestep

    # ========= FUNCTION DEFINITINOS ========

    def h_function(x: np.ndarray, k: np.ndarray, d: np.ndarray, r: np.ndarray, t: np.ndarray):
        """Returns a numpy array of the 2D marker pixel coordinates (shape Nx2) for a given state vector x and camera parameters k, d, r, t.
        """
        coords_3d = misc.get_3d_marker_coords(x)
        coords_2d = project_points_fisheye(coords_3d, k, d, r, t) # Project the 3D positions to 2D

        return coords_2d


    def predict_next_state(x: np.ndarray, dt: np.float32):
        """Returns a numpy array of the predicted states for a given state vector x and time delta dt.
        """
        acc_prediction = x[acc_idx:]
        vel_prediction = x[vel_idx:acc_idx] + dt*acc_prediction
        pos_prediction = x[:vel_idx] + dt*vel_prediction + (0.5*dt**2)*acc_prediction

        return np.concatenate([pos_prediction, vel_prediction, acc_prediction]).astype(np.float32)


    def numerical_jacobian(func, x: np.ndarray, *args):
        """Returns a numerically approximated jacobian of func with respect to x.
        Additional parameters will be passed to func using *args in the format: func(*x, *args)
        """
        n = len(x)
        eps = 1e-3

        fx = func(x, *args).flatten()
        xpeturb=x.copy()
        jac = np.empty((len(fx), n))
        for i in range(n):
            xpeturb[i] = xpeturb[i]+eps
            jac[:,i] = (func(xpeturb, *args).flatten() - fx)/eps
            xpeturb[i]=x[i]

        return jac


    # ========= LOAD DLC DATA ========

    # Load DLC 2D point files (.h5 outputs)
    dlc_2d_point_files = sorted(glob(os.path.join(DLC_DIR, '*.h5')))
    assert(len(dlc_2d_point_files) == n_cams), f"# of dlc '.h5' files != # of cams in {n_cams}_cam_scene_sba.json"

    # Load Measurement Data (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_2d_point_files, verbose=False)

    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df[points_2d_df['likelihood']>dlc_thresh], # ignore points with low likelihood
        k_arr, d_arr.reshape((-1,4)), r_arr, t_arr,
        triangulate_points_fisheye
    )

    # Restructure dataframe
    points_df = points_2d_df.set_index(['frame', 'camera','marker'])
    points_df = points_df.stack().unstack(level=1).unstack(level=1).unstack()

    # Pixels array
    pixels_df = points_df.loc[:, (range(n_cams), markers, ['x','y'])]
    pixels_df = pixels_df.reindex(columns=pd.MultiIndex.from_product([range(n_cams), markers, ['x','y']]))
    pixels_arr = pixels_df.to_numpy() #shape - (n_frames, n_cams * n_markers * 2)

    # Likelihood array
    likelihood_df = points_df.loc[:, (range(n_cams), markers, 'likelihood')]
    likelihood_df = likelihood_df.reindex(columns=pd.MultiIndex.from_product([range(n_cams), markers, ['likelihood']]))
    likelihood_arr = likelihood_df.to_numpy() #shape - (n_frames, n_cams * n_markers * 1)

    # ========= INITIALIZE EKF MATRICES ========

    # estimate initial points
    states = np.zeros(n_states)

    # try:
    #     lure_pts = points_3d_df[points_3d_df["marker"]=="lure"][["frame", "x", "y", "z"]].values
    #     lure_x_slope, lure_x_intercept, *_ = linregress(lure_pts[:,0], lure_pts[:,1])
    #     lure_y_slope, lure_y_intercept, *_ = linregress(lure_pts[:,0], lure_pts[:,2])

    #     lure_x_est = start_frame*lure_x_slope + lure_x_intercept # initial lure x
    #     lure_y_est = start_frame*lure_y_slope + lure_y_intercept # initial lure y

    #     states[[idx['x_l'], idx['y_l']]] = [lure_x_est, lure_y_est]             # lure x & y in inertial
    #     states[[idx['dx_l'], idx['dy_l']]] = [lure_x_slope/sT, lure_y_slope/sT] # lure x & y velocity in inertial
    # except ValueError as e: # for when there is no lure data
    #     print(f"Lure initialisation error: '{e}' -> Lure states initialised to zero")

    points_3d_df = points_3d_df[points_3d_df['frame'].between(start_frame, end_frame-1)]

    nose_pts = points_3d_df[points_3d_df["marker"]=="nose"][["frame", "x", "y", "z"]].values
    nose_x_slope, nose_x_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,1])
    nose_y_slope, nose_y_intercept, *_ = linregress(nose_pts[:,0], nose_pts[:,2])

    nose_x_est = start_frame*nose_x_slope + nose_x_intercept # initial nose x
    nose_y_est = start_frame*nose_y_slope + nose_y_intercept # initial nose y
    nose_psi_est = np.arctan2(nose_y_slope, nose_x_slope)    # initial yaw angle relative to inertial

    # INITIAL STATES
    states[[idx['x_0'], idx['y_0'],idx['psi_0']]] = [nose_x_est, nose_y_est, nose_psi_est] # head x, y & psi (yaw) in inertial
    states[[idx['dx_0'], idx['dy_0']]] = [nose_x_slope/sT, nose_y_slope/sT]                # head x & y velocity in inertial

    # INITIAL STATE COVARIANCE P - how much do we trust the initial states
    # position
    p_lin_pos = np.ones(3)*3**2                       # Know initial position within 4m
    p_ang_pos = np.ones(n_pose_params-3)*(np.pi/4)**2 # Know initial angles within 60 degrees, heading may need to change
    # p_lure_pos = p_lin_pos
    # velocity
    p_lin_vel = np.ones(3)*5**2                       # Know this within 2.5m/s and it's a uniform random variable
    p_ang_vel = np.ones(n_pose_params-3)*3**2
    # p_lure_vel = p_lin_vel
    # acceleration
    p_lin_acc = np.ones(3)*3**2
    p_ang_acc = np.ones(n_pose_params-3)*3**2
    p_ang_acc[10:] = 5**2
    # p_lure_acc = p_lin_acc

    P = np.diag(np.concatenate([p_lin_pos, p_ang_pos, #p_lure_pos,
                                p_lin_vel, p_ang_vel, #p_lure_vel,
                                p_lin_acc, p_ang_acc, #p_lure_acc
                               ]))

    # PROCESS COVARIANCE Q - how "noisy" the constant acceleration model is
    qb_list = [
        5.0, 5.0, 5.0,    # head x, y, z in inertial
        10.0, 10.0, 10.0, # head phi, theta, psi in inertial
        5.0, 25.0, 5.0,   # neck phi, theta, psi
        50.0,             # front-torso theta
        5.0, 50.0, 25.0,  # back torso phi, theta, psi
        100.0, 30.0,      # tail base theta, psi
        140.0, 40.0,      # tail mid theta, psi
        350.0, 200.0,     # l_shoulder theta, l_front_knee theta
        350.0, 200.0,     # r_shoulder theta, r_front_knee theta
        450.0, 400.0,     # l_hip theta, l_back_knee theta
        450.0, 400.0,     # r_hip theta, r_back_knee theta
    ]
    # qb_list += qb_list[0:3] # lure x, y, z in inertial - same as head

    qb = (np.diag(qb_list)/2)**2
    Q = np.block([
        [sT**4/4 * qb, sT**3/2 * qb, sT**2/2 * qb],
        [sT**3/2 * qb, sT**2 * qb, sT * qb],
        [sT**2/2 * qb, sT * qb, qb],
    ])

    # MEASUREMENT COVARIANCE R
    dlc_cov = 5**2

    # State prediction function jacobian F - shape: (n_states, n_states)
    rng = np.arange(n_states - vel_idx)
    rng_acc = np.arange(n_states - acc_idx)
    F = np.eye(n_states)
    F[rng, rng+vel_idx] = sT
    F[rng_acc, rng_acc+acc_idx] = sT**2/2

    # Allocate space for storing EKF data
    states_est_hist = np.zeros((n_frames, n_states))
    states_pred_hist = states_est_hist.copy()
    P_est_hist = np.zeros((n_frames, n_states, n_states))
    P_pred_hist = P_est_hist.copy()

    t1 = time()
    print("\nInitialization took {0:.2f} seconds\n".format(t1 - t0))

    # ========= RUN EKF & SMOOTHER ========

    t0 = time()

    outliers_ignored = 0

    for i in range(n_frames):
        print(f"Running frame {i+start_frame+1}\r", end='')

        # ========== PREDICTION ==========

        # Predict State
        states = predict_next_state(states, sT).flatten()
        states_pred_hist[i] = states

        # Projection of the state covariance
        P = F @ P @ F.T + Q
        P_pred_hist[i] = P

        # ============ UPDATE ============

        z_k = pixels_arr[i+start_frame]
        likelihood = likelihood_arr[i+start_frame]

        # Measurement
        H = np.zeros((n_cams*n_markers*2, n_states))
        h = np.zeros((n_cams*n_markers*2)) # same as H[:, 0].copy()
        for j in range(n_cams):
            # State measurement
            h[j*n_markers*2:(j+1)*n_markers*2] = h_function(states[:vel_idx], *camera_params[j]).flatten()
            # Jacobian - shape: (2*n_markers, n_states)
            H[j*n_markers*2:(j+1)*n_markers*2, 0:vel_idx] = numerical_jacobian(h_function, states[:vel_idx], *camera_params[j])

        # Measurement Covariance R
        bad_point_mask = np.repeat(likelihood<dlc_thresh, 2)
        dlc_cov_arr = dlc_cov*np.ones((n_cams*n_markers*2))
        dlc_cov_arr[bad_point_mask] = max_pixel_err # change this to be independent of cam res?
        R = np.diag(dlc_cov_arr**2)

        # Residual
        residual = z_k - h

        # Residual Covariance S
        S = (H @ P @ H.T) + R
        temp = sigma_bound*np.sqrt(np.diag(S)) # if measurement residual is worse than 3 sigma, set residual to 0 and rely on predicted state only
        for j in range(0, len(residual), 2):
            if np.abs(residual[j])>temp[j] or np.abs(residual[j+1])>temp[j+1]:
                residual[j:j+2] = 0
                outliers_ignored += 1

        # Kalman Gain
        K = P @ H.T @ np.linalg.inv(S)

        # Correction
        states = states + K @ residual
        states_est_hist[i] = states

        # Update State Covariance
        P = (np.eye(K.shape[0]) - K @ H) @ P
        P_est_hist[i] = P

    print("EKF complete!")
    print("Outliers ignored:", outliers_ignored)

    # Run Kalman Smoother
    smooth_states_est_hist = states_est_hist.copy()
    smooth_P_est_hist = P_est_hist.copy()
    for i in range(n_frames-2, 0, -1):
        A = P_est_hist[i] @ F.T @ np.linalg.inv(P_pred_hist[i+1])
        smooth_states_est_hist[i] = states_est_hist[i] + A @ (smooth_states_est_hist[i+1] - states_pred_hist[i+1])
        smooth_P_est_hist[i] = P_est_hist[i] + A @ (smooth_P_est_hist[i+1] - P_pred_hist[i+1]) @ A.T

    print("\nKalman Smoother complete!\n")
    t1 = time()
    print("Optimization took {0:.2f} seconds\n".format(t1 - t0))

    app.stop_logging()

    # ========= SAVE EKF RESULTS ========

    states = dict(x=states_est_hist[:, :vel_idx],
                  dx=states_est_hist[:, vel_idx:acc_idx],
                  ddx=states_est_hist[:, acc_idx:],
                  smoothed_x=smooth_states_est_hist[:, :vel_idx],
                  smoothed_dx=smooth_states_est_hist[:, vel_idx:acc_idx],
                  smoothed_ddx=smooth_states_est_hist[:, acc_idx:]
                 )
    app.save_ekf(states, OUT_DIR, scene_fpath, start_frame, dlc_thresh)

    fig_fpath = os.path.join(OUT_DIR, 'ekf.pdf')
    app.plot_cheetah_states(states['x'], states['smoothed_x'], fig_fpath)


def sba(DATA_DIR, DLC_DIR, start_frame, end_frame, dlc_thresh):
    t0 = time()

    OUT_DIR = os.path.join(DATA_DIR, 'sba')
    os.makedirs(OUT_DIR, exist_ok=True)

    app.start_logging(os.path.join(OUT_DIR, 'sba.log'))

    # load video info
    res, fps, tot_frames, _ = app.get_vid_info(DATA_DIR) # path to original videos
    assert end_frame <= tot_frames, f'end_frame must be less than or equal to {tot_frames}'

    start_frame -= 1    # 0 based indexing
    assert start_frame >= 0
    N = end_frame-start_frame

    *_, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR, verbose=False)

    dlc_points_fpaths = glob(os.path.join(DLC_DIR, '*.h5'))
    assert n_cams == len(dlc_points_fpaths)

    # Load Measurement Data (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
    points_2d_df = points_2d_df[points_2d_df["frame"].between(start_frame, end_frame-1)]
    points_2d_df = points_2d_df[points_2d_df['likelihood']>dlc_thresh] # ignore points with low likelihood

    t1 = time()
    print("Initialization took {0:.2f} seconds\n".format(t1 - t0))

    points_3d_df, residuals = app.sba_points_fisheye(scene_fpath, points_2d_df)

    app.stop_logging()

    plt.plot(residuals['before'], label="Cost before")
    plt.plot(residuals['after'], label="Cost after")
    plt.legend()
    fig_fpath = os.path.join(OUT_DIR, 'sba.svg')
    plt.savefig(fig_fpath, transparent=True)
    print(f'Saved {fig_fpath}\n')
    plt.show(block=False)

    # ========= SAVE SBA RESULTS ========

    markers = misc.get_markers()

    positions = np.full((N, len(markers), 3), np.nan)
    for i, marker in enumerate(markers):
        marker_pts = points_3d_df[points_3d_df["marker"]==marker][["frame", "x", "y", "z"]].values
        for frame, *pt_3d in marker_pts:
            positions[int(frame)-start_frame, i] = pt_3d

    app.save_sba(positions, OUT_DIR, scene_fpath, start_frame, dlc_thresh)


def tri(DATA_DIR, DLC_DIR, start_frame: int, end_frame: int, dlc_thresh):
    OUT_DIR = os.path.join(DATA_DIR, 'tri')
    os.makedirs(OUT_DIR, exist_ok=True)

    start_frame -= 1 # 0 based indexing
    assert start_frame >= 0
    N = end_frame - start_frame

    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = utils.find_scene_file(DATA_DIR, verbose=False)

    dlc_points_fpaths = sorted(glob(os.path.join(DLC_DIR, '*.h5')))
    assert n_cams == len(dlc_points_fpaths)

    # Load Measurement Data (pixels, likelihood)
    points_2d_df = utils.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
    points_2d_df = points_2d_df[points_2d_df["frame"].between(start_frame, end_frame-1)]
    points_2d_df = points_2d_df[points_2d_df['likelihood']>dlc_thresh] # ignore points with low likelihood

    assert len(k_arr) == points_2d_df['camera'].nunique(), '{}'.format(points_2d_df['camera'])

    points_3d_df = utils.get_pairwise_3d_points_from_df(
        points_2d_df,
        k_arr, d_arr.reshape((-1, 4)), r_arr, t_arr,
        triangulate_points_fisheye
    )

    points_3d_df['point_index'] = points_3d_df.index

    # ========= SAVE TRIANGULATION RESULTS ========

    markers = misc.get_markers()

    positions = np.full((N, len(markers), 3), np.nan)
    for i, marker in enumerate(markers):
        marker_pts = points_3d_df[points_3d_df["marker"]==marker][["frame", "x", "y", "z"]].values
        for frame, *pt_3d in marker_pts:
            positions[int(frame)-start_frame, i] = pt_3d

    app.save_tri(positions, OUT_DIR, scene_fpath, start_frame, dlc_thresh)


def dlc(DATA_DIR, dlc_thresh):
    video_fpaths = sorted(glob(os.path.join(DATA_DIR, 'cam[1-9].mp4'))) # original vids should be in the parent dir
    out_dir = os.path.join(DATA_DIR, 'dlc')
    app.create_labeled_videos(video_fpaths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)


# ========= MAIN ========

if __name__ == "__main__":
    # python all_optimizations.py --data_dir /data/2019_03_07/phantom/flick --plot
    parser = ArgumentParser(description='All Optimizations')
    parser.add_argument('--data_dir', type=str, help='The data directory path to the flick/run to be optimized')
    parser.add_argument('--start_frame', type=int, default=1, help='The frame at which the optimized reconstruction will start at')
    parser.add_argument('--end_frame', type=int, default=-1, help='The frame at which the optimized reconstruction will end at')
    parser.add_argument('--dlc_thresh', type=float, default=0.8, help='The likelihood of the dlc points below which will be excluded from the optimization')
    parser.add_argument('--plot', action='store_true', help='Showing plots')
    args = parser.parse_args()

    # ROOT_DATA_DIR = os.path.join("..", "data")
    DATA_DIR = os.path.normpath(args.data_dir)
    assert os.path.exists(DATA_DIR)
    DLC_DIR = os.path.join(DATA_DIR, 'dlc')
    assert os.path.exists(DLC_DIR)

    # load DLC info
    res, fps, tot_frames, _ = app.get_vid_info(DATA_DIR) # path to original videos
    assert args.end_frame <= tot_frames, f'end_frame must be less than or equal to {tot_frames}'
    end_frame = tot_frames if args.end_frame == -1 else args.end_frame

    print('========== DLC ==========\n')
    dlc(DATA_DIR, args.dlc_thresh)
    print('========== Triangulation ==========\n')
    tri(DATA_DIR, DLC_DIR, args.start_frame, end_frame, args.dlc_thresh)
    plt.close('all')
    # print('========== SBA ==========\n')
    # sba(DATA_DIR, DLC_DIR, args.start_frame, end_frame, args.dlc_thresh)
    # plt.close('all')
    # print('========== EKF ==========\n')
    # ekf(DATA_DIR, DLC_DIR, args.start_frame, end_frame, args.dlc_thresh)
    # plt.close('all')
    # print('========== FTE ==========\n')
    # fte(DATA_DIR, DLC_DIR, args.start_frame, end_frame, args.dlc_thresh)
    # plt.close('all')

    if args.plot:
        print('Plotting results...')
        data_fpaths = [
            os.path.join(DATA_DIR, 'tri', 'tri.pickle'), # plot is too busy when tri is included
            os.path.join(DATA_DIR, 'sba', 'sba.pickle'),
            os.path.join(DATA_DIR, 'ekf', 'ekf.pickle'),
            os.path.join(DATA_DIR, 'fte', 'fte.pickle')
        ]
        app.plot_multiple_cheetah_reconstructions(data_fpaths, reprojections=False, dark_mode=True)