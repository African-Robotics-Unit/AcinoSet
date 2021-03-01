import numpy as np
import sympy as sp


def get_markers():
    return ["l_eye", "r_eye", "nose",
            "neck_base", "spine",
            "tail_base", "tail1", "tail2",
            "l_shoulder", "l_front_knee", "l_front_ankle",
            "r_shoulder", "r_front_knee", "r_front_ankle",
            "l_hip", "l_back_knee", "l_back_ankle",
            "r_hip", "r_back_knee", "r_back_ankle",
            "lure"
           ]


def get_pose_params():
    states = ['x_0', 'y_0', 'z_0',         # head position in inertial
              'phi_0', 'theta_0', 'psi_0', # head rotation in inertial
              'phi_1', 'theta_1', 'psi_1', # neck
              'theta_2',                   # front torso
              'phi_3', 'theta_3', 'psi_3', # back torso
              'theta_4', 'psi_4',          # tail_base
              'theta_5', 'psi_5',          # tail_mid
              'theta_6', 'theta_7',        # l_shoulder, l_front_knee
              'theta_8', 'theta_9',        # r_shoulder, r_front_knee
              'theta_10', 'theta_11',      # l_hip, l_back_knee
              'theta_12', 'theta_13',      # r_hip, r_back_knee
              'x_l', 'y_l', 'z_l'          # lure position in inertial
             ]
    return dict(zip(states ,range(len(states))))


def get_3d_marker_coords(x):
    """Returns either a numpy array or a sympy Matrix of the 3D marker coordinates (shape Nx3) for a given state vector x.
    """
    func = sp.Matrix if isinstance(x[0], sp.Expr) else np.array
    idx = get_pose_params()
    
    # rotations
    RI_0  = rot_z(x[idx['psi_0']]) @ rot_x(x[idx['phi_0']]) @ rot_y(x[idx['theta_0']])         # head
    R0_I  = RI_0.T
    RI_1  = rot_z(x[idx['psi_1']]) @ rot_x(x[idx['phi_1']]) @ rot_y(x[idx['theta_1']]) @ RI_0  # neck
    R1_I  = RI_1.T
    RI_2  = rot_y(x[idx['theta_2']]) @ RI_1                                                    # front torso
    R2_I  = RI_2.T
    RI_3  = rot_z(x[idx['psi_3']]) @ rot_x(x[idx['phi_3']]) @ rot_y(x[idx['theta_3']]) @ RI_2  # back torso
    R3_I  = RI_3.T
    RI_4  = rot_z(x[idx['psi_4']]) @ rot_y(x[idx['theta_4']]) @ RI_3                           # tail base
    R4_I  = RI_4.T
    RI_5  = rot_z(x[idx['psi_5']]) @ rot_y(x[idx['theta_5']]) @ RI_4                           # tail mid
    R5_I  = RI_5.T
    RI_6  = rot_y(x[idx['theta_6']]) @ RI_2                                                    # l_shoulder
    R6_I  = RI_6.T
    RI_7  = rot_y(x[idx['theta_7']]) @ RI_6                                                    # l_front_knee
    R7_I  = RI_7.T
    RI_8  = rot_y(x[idx['theta_8']]) @ RI_2                                                    # r_shoulder
    R8_I  = RI_8.T
    RI_9  = rot_y(x[idx['theta_9']]) @ RI_8                                                    # r_front_knee
    R9_I  = RI_9.T
    RI_10 = rot_y(x[idx['theta_10']]) @ RI_3                                                   # l_hip
    R10_I = RI_10.T
    RI_11 = rot_y(x[idx['theta_11']]) @ RI_10                                                  # l_back_knee
    R11_I = RI_11.T
    RI_12 = rot_y(x[idx['theta_12']]) @ RI_3                                                   # r_hip
    R12_I = RI_12.T
    RI_13 = rot_y(x[idx['theta_13']]) @ RI_12                                                  # r_back_knee
    R13_I = RI_13.T

    # positions
    p_head          = func([x[idx['x_0']], x[idx['y_0']],x[idx['z_0']]])

    p_l_eye         = p_head         + R0_I  @ func([0, 0.03, 0])
    p_r_eye         = p_head         + R0_I  @ func([0, -0.03, 0])
    p_nose          = p_head         + R0_I  @ func([0.055, 0, -0.055])

    p_neck_base     = p_head         + R1_I  @ func([-0.28, 0, 0])
    p_spine         = p_neck_base    + R2_I  @ func([-0.37, 0, 0])

    p_tail_base     = p_spine        + R3_I  @ func([-0.37, 0, 0])
    p_tail_mid      = p_tail_base    + R4_I  @ func([-0.28, 0, 0])
    p_tail_tip      = p_tail_mid     + R5_I  @ func([-0.36, 0, 0])

    p_l_shoulder    = p_neck_base    + R2_I  @ func([-0.04, 0.08, -0.10])
    p_l_front_knee  = p_l_shoulder   + R6_I  @ func([0, 0, -0.24])
    p_l_front_ankle = p_l_front_knee + R7_I  @ func([0, 0, -0.28])

    p_r_shoulder    = p_neck_base    + R2_I  @ func([-0.04, -0.08, -0.10])
    p_r_front_knee  = p_r_shoulder   + R8_I  @ func([0, 0, -0.24])
    p_r_front_ankle = p_r_front_knee + R9_I  @ func([0, 0, -0.28])

    p_l_hip         = p_tail_base    + R3_I  @ func([0.12, 0.08, -0.06])
    p_l_back_knee   = p_l_hip        + R10_I @ func([0, 0, -0.32])
    p_l_back_ankle  = p_l_back_knee  + R11_I @ func([0, 0, -0.25])

    p_r_hip         = p_tail_base    + R3_I  @ func([0.12, -0.08, -0.06])
    p_r_back_knee   = p_r_hip        + R12_I @ func([0, 0, -0.32])
    p_r_back_ankle  = p_r_back_knee  + R13_I @ func([0, 0, -0.25])

    p_lure = func([x[idx['x_l']], x[idx['y_l']], x[idx['z_l']]])

    return func([p_l_eye.T, p_r_eye.T, p_nose.T,
                 p_neck_base.T, p_spine.T,
                 p_tail_base.T, p_tail_mid.T, p_tail_tip.T,
                 p_l_shoulder.T, p_l_front_knee.T, p_l_front_ankle.T,
                 p_r_shoulder.T, p_r_front_knee.T, p_r_front_ankle.T,
                 p_l_hip.T, p_l_back_knee.T, p_l_back_ankle.T,
                 p_r_hip.T, p_r_back_knee.T, p_r_back_ankle.T,
                 p_lure.T
                ])


def redescending_loss(err, a, b, c):
    # outlier rejecting cost function
    def func_step(start, x):
        return 1/(1+np.e**(-1*(x - start)))

    def func_piece(start, end, x):
        return func_step(start, x) - func_step(end, x)
    
    e = abs(err)
    cost = 0.0
    cost += (1 - func_step(a, e))/2*e**2
    cost += func_piece(a, b, e)*(a*e - (a**2)/2)
    cost += func_piece(b, c, e)*(a*b - (a**2)/2 + (a*(c-b)/2)*(1-((c-e)/(c-b))**2))
    cost += func_step(c, e)*(a*b - (a**2)/2 + (a*(c-b)/2))
    return cost


def global_positions(R_arr, t_arr):
    "Returns a vector of camera position vectors in the world frame"
    R_arr = np.array(R_arr).reshape((-1, 3, 3))
    t_arr = np.array(t_arr).reshape((-1, 3, 1))
    
    positions = []
    assert R_arr.shape[0]==t_arr.shape[0], "Number of cams in R_arr do not match t_arr"
    for r, t in zip(R_arr, t_arr):
        pos = -r.T @ t
        positions.append(pos)
        
    return np.array(positions, dtype=np.float32)


def rotation_matrix_from_vectors(u,v):
    """ Find the rotation matrix that aligns u to v
    :param u: A 3D "source" vector
    :param v: A 3D "destination" vector
    :return mat: A transform matrix (3x3) which when applied to u, aligns it with v.
    """
    # https://stackoverflow.com/questions/36409140/create-a-rotation-matrix-from-2-normals
    # Suppose you want to write the rotation that maps a vector u to a vector v.
    # if U and V are their unit vectors then W = U^V (cross product) is the axis of rotation and is an invariant
    # Let M be the associated matrix.
    # We have finally: (V,W,V^W) = M.(U,W,U^W)

    U = (u/np.linalg.norm(u)).reshape(3)
    V = (v/np.linalg.norm(v)).reshape(3)
    
    W = np.cross(U, V)
    A = np.array([U, W, np.cross(U, W)]).T
    B = np.array([V, W, np.cross(V, W)]).T
    return np.dot(B, np.linalg.inv(A))


def rot_x(x):
    if isinstance(x, sp.Expr):
        c = sp.cos(x)
        s = sp.sin(x)
        func = sp.Matrix
    else:
        c = np.cos(x)
        s = np.sin(x)
        func = np.array
    return func([[1, 0, 0],
                 [0, c, s],
                 [0, -s, c]])


def rot_y(y):
    if isinstance(y, sp.Expr):
        c = sp.cos(y)
        s = sp.sin(y)
        func = sp.Matrix
    else:
        c = np.cos(y)
        s = np.sin(y)
        func = np.array
    return func([[c, 0, -s],
                 [0, 1, 0],
                 [s, 0, c]])


def rot_z(z):
    if isinstance(z, sp.Expr):
        c = sp.cos(z)
        s = sp.sin(z)
        func = sp.Matrix
    else:
        c = np.cos(z)
        s = np.sin(z)
        func = np.array
    return func([[c, s, 0],
                 [-s, c, 0],
                 [0, 0, 1]])
