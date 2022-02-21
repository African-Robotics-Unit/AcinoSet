from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui
import pyqtgraph.opengl as gl
import pyqtgraph
# pyqtgraph.setConfigOption('background', (255, 255, 200))
# pyqtgraph.setConfigOption('foreground', 'k')

pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')
pyqtgraph.setConfigOptions(antialias=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import cv2
from .utils import create_board_object_pts

def create_camera(color=(0.1, 0.1, 0.1, 1)):
    ## plot camera
    bx = 0.15
    by = 0.1
    bz = 0.1
    m = max([bx, by, bz])
    verts = np.array([
        [bx, by, bz],
        [bx, by, -bz],
        [bx, -by, bz],
        [bx, -by, -bz],
        [-bx, by, bz],
        [-bx, by, -bz],
        [-bx, -by, bz],
        [-bx, -by, -bz],
        [2 * bx, 2 * by, 2 * bz],
        [2 * bx, -2 * by, 2 * bz],
        [-2 * bx, 2 * by, 2 * bz],
        [-2 * bx, -2 * by, 2 * bz],
        [0, 0, 0],
        [5*m, 0, 0],
        [0, 5*m, 0],
        [0, 0, 5*m]
    ])
    edges = np.array([
        [0,1],
        [0,2],
        [0,4],
        [1,3],
        [1,5],
        [2,3],
        [2,6],
        [3,7],
        [4,5],
        [4,6],
        [5,7],
        [6,7],
        [0,8],
        [2,9],
        [4,10],
        [6,11],
        [8,9],
        [8,10],
        [9,11],
        [10,11],
        [12,13],
        [12,14],
        [12,15]
    ]).flatten()
    colors = np.array([color for _ in range(len(edges))])
    colors[[40,41]] = (1., 0., 0., 1)
    colors[[42,43]] = (0., 1., 0., 1)
    colors[[44,45]] = (0., 0., 1., 1)

    mesh = gl.GLLinePlotItem(pos=verts[edges], color=colors, width=1.0, antialias=True, mode='lines')
    return mesh


def create_grid(obj_points, board_shape, color=(0.5, 0.5, 0.5, 1)):
    cols = board_shape[0]
    rows = board_shape[1]
    xyz_quiver = np.array([
        [0, 0, 0],
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1]
    ])
    verts = np.vstack([obj_points, xyz_quiver])

    edges = []
    for r in range(rows):
        for c in range(cols-1):
            edges.append(c+r*cols)
            edges.append(c+r*cols+1)
    for c in range(cols):
        for r in range(rows-1):
            edges.append(c+r*cols)
            edges.append(c+(r+1)*cols)
    edges.extend(np.array([0,1,0,2,0,3])+(rows*cols))

    colors = np.array([color for _ in range(len(edges))])
    colors[[-6,-5]] = (1, 0, 0, 1)
    colors[[-4,-3]] = (0, 1, 0, 1)
    colors[[-2,-1]] = (0, 0, 1, 1)


    mesh = gl.GLLinePlotItem(pos=verts[edges], color=colors, width=1.0, antialias=True, mode='lines')
    return mesh


def plot_calib_board(img_points, board_shape, camera_resolution, frame_fpath=None):
    corners = np.array(img_points, dtype=np.float32)
    plt.figure(figsize=(8, 4.5))
    if frame_fpath:
        plt.imshow(plt.imread(frame_fpath))
    for pts in corners:
        pts = pts.reshape(-1, 2)
        cols = board_shape[0]
        rows = board_shape[1]
        edges = []
        for r in range(rows):
            for c in range(cols - 1):
                edges.append(c + r * cols)
                edges.append(c + r * cols + 1)
        for c in range(cols):
            for r in range(rows - 1):
                edges.append(c + r * cols)
                edges.append(c + (r + 1) * cols)
        lc = mc.LineCollection(pts[edges].reshape(-1, 2, 2), color='r', linewidths=1)

        plt.gca().add_collection(lc)
        plt.gca().set_xlim((0, camera_resolution[0]))
        plt.gca().set_ylim((camera_resolution[1], 0))
    plt.show()


class Scene:

    def __init__(self):
        self.app = QApplication.instance()
        if self.app == None:
            self.app = QApplication([])
        self.view = gl.GLViewWidget(devicePixelRatio=1)

        self.view.setBackgroundColor("#FFFFFF")

        self.view.pan(2.5,5,0)
        self.view.orbit(-120,10)
        self.view.opts['distance']=13
        self.view.update()
        self.view.show()

    def rodrigues_to_vec(self, r):
        ang = 180 / np.pi * np.linalg.norm(r)
        return (ang, *r)

    def plot_camera(self, r, t):
        # Uses camera pose!
        # Note: T is the world origin position in the camera coordinates
        #       the world position of the camera C = -(R^-1)@T.
        #       Similarly, the rotation of the camera in world coordinates
        #       is given by R^-1.
        #       The inverse of a rotation matrix is also its transpose.
        # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
        R = r.T
        T = -r.T@t
        # Convert for plotting
        R = cv2.Rodrigues(R)[0]
        R = self.rodrigues_to_vec(R)
        cam = create_camera()
        cam.rotate(*R)
        cam.translate(*T)
        self.view.addItem(cam)

    def plot_calib_board(self, r, t, board_shape, board_edge_len):
        obj_pts = create_board_object_pts(board_shape, board_edge_len)
        calib_board = create_grid(obj_pts, board_shape)
        calib_board.translate(*t)
        r = self.rodrigues_to_vec(r)
        calib_board.rotate(*r)
        self.view.addItem(calib_board)

    def plot_points(self, points, color=(0,0.5,0.5,0.5)):
        scatter = gl.GLScatterPlotItem(pos=points, color=color, size=3, pxMode=True)
        scatter.setGLOptions('translucent')
        self.view.addItem(scatter)

    def plot_xy_grid(self):
        grid = gl.GLGridItem(size=QtGui.QVector3D(50,50,1), color=(200,200,200,255))
        grid.setGLOptions('translucent')
        self.view.addItem(grid)



    def show(self):
        self.app.exec_()

    def save(self, filename, size=(800, 450)):
        pyqtgraph.makeQImage(self.view.renderToArray(size)).save(filename)
        self.app.quit()

