import numpy as np


def global_positions(R_arr, t_arr):
    "Returns a vector of camera position column vectors"
    R_arr = R_arr.reshape((-1, 3, 3))
    t_arr = t_arr.reshape((-1, 3, 1))
    
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
    W = np.cross(U,V)
    A = np.array([U,W,np.cross(U,W)]).T
    B = np.array([V,W,np.cross(V,W)]).T
    return np.dot(B,np.linalg.inv(A))


def rot_z(z: np.float32):
    c = np.cos(z)
    s = np.sin(z)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)