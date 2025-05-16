import numpy as np
import math

def rotate_point(point, angle):
    """
    Uses the Euler-Rodrigues formula to return new 3d co-ords x, y, and z after a vector is rotated about the vertical axis by <angle> radians

    Parameters:
    ---
    angle: float
        The angle by which the vector is rotated in radians

    """
    axis = np.asarray([0,0,1])
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return(np.dot(rot_mat, point))

def count_to_rad(enc_count):
    """
    Returns the given encoder count as its equivalent angle in radians
    """
    ang = enc_count * 2 * np.pi / 102000
    return ang

#sv = [3, 5, 0]
#theta = -1.2 

#print(rotate_point(v, theta))
#print(count_to_rad(9802))