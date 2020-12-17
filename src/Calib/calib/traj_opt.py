import numpy as np

#PYOMO
from pyomo.environ import atan

def redescending_loss(err, a, b, c):
    #OUTLIER REJECTING COST FUNCTION (REDESCENDING LOSS)
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

def kalman_smoother_3d_traj(pts_3d, sample_t):
    from pykalman import KalmanFilter
    # x = Fx + wk, zk = Hx + vk

    # State transtition matrix F
    rng = np.arange(6)
    rng_acc = np.arange(3)
    F = np.eye(9, dtype=np.float32)
    F[rng, rng+3] = sample_t
    F[rng_acc, rng_acc+6] = sample_t**2/2

    # Observation/measurement matrix H - extracts states used in the measurement
    H = np.diag(np.ones(3))
    H = np.hstack((H, np.zeros((3, 6))))

    # Process/state covariance Q  - how "noisy" the constant acceleration model is
    q = (np.diag([5.0, 5.0, 5.0])/2)**2 # state noise variance - head_root x, y, z in inertial
    Q = np.block([
        [sample_t**4/4 * q, sample_t**3/2 * q, sample_t**2/2 * q],
        [sample_t**3/2 * q, sample_t**2 * q, sample_t * q],
        [sample_t**2/2 * q, sample_t * q, q],
    ])

    # Observation/measurement covariance R
    dlc_cov = 5**2 # Observation/measurement noise variance
    R = np.diag(dlc_cov*np.ones(3))**2

    # Inital states X0
    init_pos = pts_3d[0]
    init_vel = (pts_3d[2]-init_pos)/(2*sample_t) # average of first 2 velocites
    X0 = np.array([
        init_pos[0], init_pos[1], init_pos[2],
        init_vel[0], init_vel[1], init_vel[2],
        0, 0, 0
    ])

    # Initial state covariance P0 - how much do we trust the initial states
    p_pos = np.ones(3)*3**2 # Know initial position within 4m
    p_vel = np.ones(3)*5**2 # Know this within 5m/s and it's a uniform random variable 
    p_acc = np.ones(3)*3**2
    P0 = np.diag(np.concatenate([p_pos, p_vel, p_acc]))

    kf = KalmanFilter(
        transition_matrices = F, observation_matrices = H,
        transition_covariance = Q, observation_covariance = R,
        initial_state_mean = X0, initial_state_covariance = P0
    )

    kf = kf.em(pts_3d, n_iter=5)
    smoothed_state_means, smoothed_state_covariances = kf.smooth(pts_3d)
    return smoothed_state_means, smoothed_state_covariances

def unscented_kalman_smoother_3d_traj(pts_3d, sample_t):
    from pykalman import UnscentedKalmanFilter

    def transition_func(state, noise): # non-linear transition function x = f(x, u) +wk
        rng = np.arange(6)
        rng_acc = np.arange(3)
        # state transtition matrix F
        F = np.eye(9, dtype=np.float32)
        F[rng, rng+3] = sample_t
        F[rng_acc, rng_acc+6] = sample_t**2/2
        return F@state + noise

    def observation_func(state, noise):  # non-linear observation/measurement function zk = h(x) +vk
        # observation/measurement matrix H - extracts states used in the measurement
        H = np.diag(np.ones(3))
        H = np.hstack((H, np.zeros((3, 6))))
        return H@state + noise

    # Process/state covariance Q - how "noisy" the constant acceleration model is
    q = (np.diag([5.0, 5.0, 5.0])/2)**2 # state noise varaince - head_root x, y, z in inertial
    Q = np.block([
        [sample_t**4/4 * q, sample_t**3/2 * q, sample_t**2/2 * q],
        [sample_t**3/2 * q, sample_t**2 * q, sample_t * q],
        [sample_t**2/2 * q, sample_t * q, q],
    ])

    # Observation/measurement covariance R
    dlc_cov = 5**2 # Observation/measurement noise variance
    R = np.diag(dlc_cov*np.ones(3))**2

    # Inital states X0
    init_pos = pts_3d[0]
    init_vel = (pts_3d[2]-init_pos)/(2*sample_t) # average of first 2 velocites
    X0 = np.array([
        init_pos[0], init_pos[1], init_pos[2],
        init_vel[0], init_vel[1], init_vel[2],
        0, 0, 0
    ])

    # Initial state covariance P0 - how much do we trust the initial states
    p_pos = np.ones(3)*3**2 # Know initial position within 4m
    p_vel = np.ones(3)*5**2 # Know this within 5m/s and it's a uniform random variable 
    p_acc = np.ones(3)*3**2
    P0 = np.diag(np.concatenate([p_pos, p_vel, p_acc]))

    ukf = UnscentedKalmanFilter(
        transition_func, observation_func,
        # Q is too small so scale it, https://en.wikipedia.org/wiki/Kalman_filter#Square_root_form
        transition_covariance = 10**Q, observation_covariance = R, # Try fixing scalar and Q, right now its a hack - the higher the scalr the more sensitive the kalman
        initial_state_mean = X0, initial_state_covariance = P0
    )

    smoothed_state_means, smoothed_state_covariances = ukf.smooth(pts_3d)
    return smoothed_state_means, smoothed_state_covariances