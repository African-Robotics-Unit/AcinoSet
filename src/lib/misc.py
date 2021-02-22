import numpy as np

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


def rot_x(x: np.float32):
    c = np.cos(x)
    s = np.sin(x)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ], dtype=np.float32)


def rot_y(y: np.float32):
    c = np.cos(y)
    s = np.sin(y)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ], dtype=np.float32)


def rot_z(z: np.float32):
    c = np.cos(z)
    s = np.sin(z)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)

def get_markers(include_lure=False):
    markers = [
        "l_eye", "r_eye", "nose",
        "neck_base", "spine",
        "tail_base", "tail1", "tail2",
        "l_shoulder", "l_front_knee", "l_front_ankle",
        "r_shoulder", "r_front_knee", "r_front_ankle",
        "l_hip", "l_back_knee", "l_back_ankle",
        "r_hip", "r_back_knee", "r_back_ankle",
    ]
    
    if include_lure:
        markers += ["lure"]
        
    return markers