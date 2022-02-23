from typing import Dict
import pickle
from pyomo.core.base.constraint import Constraint, ConstraintList
import sympy as sp
import numpy as np
import os
import glob
from calib import utils, calib, plotting, app, extract
from scipy import stats
from pprint import pprint
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.base.PyomoModel import ConcreteModel

pose_to_3d = []

def load_skeleton(skel_file) -> Dict:
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(skel_file, 'rb') as handle:
        skel_dict = pickle.load(handle)

    return skel_dict

def build_model(skel_dict, project_dir) -> ConcreteModel:
    """
    Builds a pyomo model from a given saved skeleton dictionary
    """
    links = skel_dict["links"]
    positions = skel_dict["positions"]
    dofs = skel_dict["dofs"]
    print(dofs)
    markers = skel_dict["markers"]
    for joint in markers:
        dofs[joint] = [1, 1, 1]
    
    print("New dofs:")
    print(dofs)
    rot_dict = {}
    pose_dict = {}
    L = len(positions)

    phi     = [sp.symbols(f"\\phi_{{{l}}}")   for l in range(L)]
    theta   = [sp.symbols(f"\\theta_{{{l}}}") for l in range(L)]
    psi     = [sp.symbols(f"\\psi_{{{l}}}")   for l in range(L)]

    i=0
    for part in dofs:
        rot_dict[part] = sp.eye(3)
        if dofs[part][1]:
            rot_dict[part] = rot_y(theta[i]) @ rot_dict[part]
        if dofs[part][0]:
            rot_dict[part] = rot_x(phi[i]) @ rot_dict[part]
        if dofs[part][2]:
            rot_dict[part] = rot_z(psi[i]) @ rot_dict[part]
        
        rot_dict[part + "_i"] = rot_dict[part].T
        i+=1
    
    x,   y,   z   = sp.symbols("x y z")
    dx,  dy,  dz  = sp.symbols("\\dot{x} \\dot{y} \\dot{z}")
    ddx, ddy, ddz = sp.symbols("\\ddot{x} \\ddot{y} \\ddot{z}")

    for link in links:
        if len(link) == 1:
            pose_dict[link[0]] = sp.Matrix([x, y, z])
        else:
            if link[0] not in pose_dict:
                pose_dict[link[0]] = sp.Matrix([x, y, z])

            translation_vec = sp.Matrix([positions[link[1]][0] - positions[link[0]][0],
                 positions[link[1]][1] - positions[link[0]][1],
                 positions[link[1]][2] - positions[link[0]][2]])
            rot_dict[link[1]] = rot_dict[link[1]] @ rot_dict[link[0]]
            rot_dict[link[1]+"_i"] = rot_dict[link[1]+"_i"].T
            pose_dict[link[1]] = pose_dict[link[0]] + rot_dict[link[0] + "_i"] @ translation_vec
    
    t_poses = []
    for pose in pose_dict:
        t_poses.append(pose_dict[pose].T)
    
    t_poses_mat = sp.Matrix(t_poses)

    func_map = {"sin":sin, "cos":cos, "ImmutableDenseMatrix":np.array} 
    sym_list = [x, y, z, *phi, *theta, *psi]
    pose_to_3d = sp.lambdify(sym_list, t_poses_mat, modules=[func_map])
    pos_funcs = []

    for i in range(t_poses_mat.shape[0]):
        lamb = sp.lambdify(sym_list, t_poses_mat[i,:], modules=[func_map])
        pos_funcs.append(lamb)
    
    scene_path = os.path.join(project_dir, "4_cam_scene_static_sba.json")

    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
    D_arr = D_arr.reshape((-1,4))

    markers_dict = dict(enumerate(markers))

    print(f"\n\n\nLoading data")

    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    print("2d df points:")
    print(points_2d_df)

    # break here

    def get_meas_from_df(n, c, l, d):
        n_mask = points_2d_df["frame"]== n-1
        l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"]== c-1
        d_idx = {1:"x", 2:"y"}
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val[d_idx[d]].values[0]

    def get_likelihood_from_df(n, c, l):
        n_mask = points_2d_df["frame"]== n-1
        if(markers[l-1] == "neck"):
            return 0
        else:
            l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"]== c-1
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val["likelihood"].values[0]
    
    h = 1/120 #timestep
    start_frame = 60 # 50
    N = 100
    P = 3 + len(phi)+len(theta)+len(psi)
    L = len(pos_funcs)
    C = len(K_arr)
    D2 = 2
    D3 = 3

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    R = 3 # measurement standard deviation
    Q = np.array([ # model parameters variance
        4.0,
        7.0,
        5.0,
        13.0,
        32.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        9.0,
        18.0,
        43.0,
        53.0,
        90.0,
        118.0,
        247.0,
        186.0,
        194.0,
        164.0,
        295.0,
        243.0,
        334.0,
        149.0,
        26.0,
        12.0,
        0.0,
        34.0,
        43.0,
        51.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])**2

    triangulate_func = calib.triangulate_points_fisheye
    points_2d_filtered_df = points_2d_df[points_2d_df['likelihood']>0.5]
    points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_filtered_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)
    print("3d points")
    print(points_3d_df)

    # estimate initial points
    nose_pts = points_3d_df[points_3d_df["marker"]=="forehead"][["x", "y", "z", "frame"]].values
    print(nose_pts[:,3])
    print(nose_pts[:,0])
    xs = stats.linregress(nose_pts[:,3], nose_pts[:,0])
    ys = stats.linregress(nose_pts[:,3], nose_pts[:,1])
    zs = stats.linregress(nose_pts[:,3], nose_pts[:,2])
    frame_est = np.arange(N)
    
    x_est = np.array([frame_est[i]*(xs.slope) + (xs.intercept) for i in range(len(frame_est))])
    y_est = np.array([frame_est[i]*(ys.slope) + (ys.intercept) for i in range(len(frame_est))])
    z_est = np.array([frame_est[i]*(zs.slope) + (zs.intercept) for i in range(len(frame_est))])
    print(x_est)
    #print("x est shape:")
    #print(x_est.shape)
    psi_est = np.arctan2(ys.slope, xs.slope)
    
    print("Started Optimisation")
    m = ConcreteModel(name = "Skeleton")

    # ===== SETS =====
    m.N = RangeSet(N) #number of timesteps in trajectory
    m.P = RangeSet(P) #number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n)
    m.L = RangeSet(L) #number of labels
    m.C = RangeSet(C) #number of cameras
    m.D2 = RangeSet(D2) #dimensionality of measurements
    m.D3 = RangeSet(D3) #dimensionality of measurements

    def init_meas_weights(model, n, c, l):
        likelihood = get_likelihood_from_df(n+start_frame, c, l)
        if likelihood > 0.5:
            return 1/R
        else:
            return 0
    m.meas_err_weight = Param(m.N, m.C, m.L, initialize=init_meas_weights, mutable=True, within=Any)  # IndexError: index 0 is out of bounds for axis 0 with size 0

    def init_model_weights(m, p):
        #if Q[p-1] != 0.0:
            #return 1/Q[p-1]
        #else:
        return 0.01
    m.model_err_weight = Param(m.P, initialize=init_model_weights, within=Any)

    m.h = h

    def init_measurements_df(m, n, c, l, d2):
        if(markers[l-1]=="neck"):
            return Constraint.Skip
        else:
            return get_meas_from_df(n+start_frame, c, l, d2)
    m.meas = Param(m.N, m.C, m.L, m.D2, initialize=init_measurements_df, within=Any)

    # ===== VARIABLES =====
    m.x = Var(m.N, m.P) #position
    m.dx = Var(m.N, m.P) #velocity
    m.ddx = Var(m.N, m.P) #acceleration
    m.poses = Var(m.N, m.L, m.D3)
    m.slack_model = Var(m.N, m.P)
    m.slack_meas = Var(m.N, m.C, m.L, m.D2, initialize=0.0)


    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N, P))
    init_x[:,0] = x_est #x
    init_x[:,1] = y_est #y
    init_x[:,2] = z_est #z
    #init_x[:,(3+len(pos_funcs)*2)] = psi_est #yaw - psi
    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))
    for n in range(1,N+1):
        for p in range(1,P+1):
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
        for l in range(1,L+1):
            [pos] = pos_funcs[l-1](*var_list)
            for d3 in range(1,D3+1):
                m.poses[n,l,d3].value = pos[d3-1]

    # ===== CONSTRAINTS =====
    # 3D POSE
    def pose_constraint(m,n,l,d3):
        #get 3d points
        var_list = [m.x[n,p] for p in range(1, P+1)]
        [pos] = pos_funcs[l-1](*var_list)
        return pos[d3-1] == m.poses[n,l,d3]
    
    m.pose_constraint = Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    def backwards_euler_pos(m,n,p): # position
        if n > 1:
    #             return m.x[n,p] == m.x[n-1,p] + m.h*m.dx[n-1,p] + m.h**2 * m.ddx[n-1,p]/2
            return m.x[n,p] == m.x[n-1,p] + m.h*m.dx[n,p]

        else:
            return Constraint.Skip
    m.integrate_p = Constraint(m.N, m.P, rule = backwards_euler_pos)


    def backwards_euler_vel(m,n,p): # velocity
        if n > 1:
            return m.dx[n,p] == m.dx[n-1,p] + m.h*m.ddx[n,p]
        else:
            return Constraint.Skip 
    m.integrate_v = Constraint(m.N, m.P, rule = backwards_euler_vel)

    m.angs = ConstraintList()
    for n in range(1,N):
        for i in range(3, 3*len(positions)):
            m.angs.add(expr=(abs(m.x[n,i]) <= np.pi/2))

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
        if(markers[l-1]=="neck"):
            return Constraint.Skip
        else:
            return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] ==0
    m.measurement = Constraint(m.N, m.C, m.L, m.D2, rule = measurement_constraints)

    def obj(m):
        slack_model_err = 0.0
        slack_meas_err = 0.0
        for n in range(1, N+1):
            #Model Error
            for p in range(1, P+1):
                slack_model_err += m.model_err_weight[p] * m.slack_model[n, p] ** 2
            #Measurement Error
            for l in range(1, L+1):
                for c in range (1, C+1):
                    for d2 in range(1, D2+1):
                        #slack_meas_err += redescending_loss(m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2], 3, 5, 15)
                        slack_meas_err += abs(m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2])
        return slack_meas_err + slack_model_err

    m.obj = Objective(rule = obj)

    return(m, pose_to_3d)

def solve_optimisation(model, exe_path, project_dir, poses) -> None:
    """
    Solves a given trajectory optimisation problem given a model and solver
    """
    opt = SolverFactory(
        'ipopt',
        executable=exe_path
    )

    # solver options
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 1000
    opt.options["max_cpu_time"] = 3600
    opt.options["tol"] = 1e-1
    opt.options["OF_print_timing_statistics"] = "yes"
    opt.options["OF_print_frequency_iter"] = 10
    opt.options["OF_hessian_approximation"] = "limited-memory"
    #opt.options["linear_solver"] = "ma86"

    LOG_DIR = os.path.join("C://Users//user-pc//Documents//Scripts//amaan", "logs")

    # --- This step may take a while! ---
    results = opt.solve(
        model, tee=True, 
        keepfiles=True, 
        logfile=os.path.join(LOG_DIR, "solver.log")
    )

    result_dir = os.path.join(project_dir, "results")
    save_data(model, file_path=os.path.join(result_dir, 'traj_results.pickle'), poses=poses)

#def save_data(file_data, filepath, dict=False):
    #if dict:
        #file_data = convert_to_dict(file_data)

    #with open(filepath, 'wb') as f:
        #pickle.dump(file_data, f)

def convert_to_dict(m, poses) -> Dict:
    x_optimised = []
    dx_optimised = []
    ddx_optimised = []
    for n in m.N:
        x_optimised.append([value(m.x[n, p]) for p in m.P])
        dx_optimised.append([value(m.dx[n, p]) for p in m.P])
        ddx_optimised.append([value(m.ddx[n, p]) for p in m.P])
    x_optimised = np.array(x_optimised)
    dx_optimised = np.array(dx_optimised)
    ddx_optimised = np.array(ddx_optimised)

    print(poses)
    print(x_optimised)
    
    positions = np.array([poses(*states) for states in x_optimised])
    file_data = dict(
        positions=positions,
        x=x_optimised,
        dx=dx_optimised,
        ddx=ddx_optimised,
    )
    return file_data

def save_data(file_data, file_path, poses, dict=True) -> None:
    
    if dict:
        file_data = convert_to_dict(file_data, poses)
        
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(file_data, f)
    
    print(f'save {file_path}')

# --- OUTLIER REJECTING COST FUNCTION (REDESCENDING LOSS) ---

def func_step(start, x):
        return 1/(1+np.e**(-1*(x - start)))

def func_piece(start, end, x):
        return func_step(start, x) - func_step(end, x)
    
def redescending_loss(err, a, b, c):
    e = abs(err)
    cost = 0.0
    cost += (1 - func_step(a, e))/2*e**2
    cost += func_piece(a, b, e)*(a*e - (a**2)/2)
    cost += func_piece(b, c, e)*(a*b - (a**2)/2 + (a*(c-b)/2)*(1-((c-e)/(c-b))**2))
    cost += func_step(c, e)*(a*b - (a**2)/2 + (a*(c-b)/2))
    return cost

# --- Rotation matrices for x, y, and z axes ---

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

# --- Numpy equivalent rotation matrices ---

def np_rot_x(x):
    c = np.cos(x)
    s = np.sin(x)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def np_rot_y(y):
    c = np.cos(y)
    s = np.sin(y)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def np_rot_z(z):
    c = np.cos(z)
    s = np.sin(z)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

# --- Reprojection functions ---

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

if __name__ == "__main__":
    skeleton_path = os.path.join("C://Users//user-pc//Documents//Scripts//amaan", "skeletons", "new_human.pickle")
    skelly = load_skeleton(skeleton_path)
    print(skelly)
    data_path = os.path.join("C://Users//user-pc//Documents//Scripts//amaan", "data")
    model1, pose3d = build_model(skelly, data_path)
    solve_optimisation(model1, exe_path="C://Users//user-pc//anaconda3//pkgs//ipopt-3.11.1-2//Library//bin//ipopt.exe", project_dir=data_path, poses=pose3d)
