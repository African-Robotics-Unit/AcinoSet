import os
import sys 
import numpy as np
import cv2

#GUI
from PyQt5.QtWidgets import QApplication, QGridLayout, QSizePolicy
from PyQt5 import QtGui, QtCore
import pyqtgraph.opengl as gl
import pyqtgraph
pyqtgraph.setConfigOptions(antialias=True) # antialias options have been included below so perhaps remove this?

#MPL
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 250 # to change the size of mpl plots
from matplotlib import collections  as mc

from .calib import project_points_fisheye
from .utils import create_board_object_pts, \
    load_scene, \
    find_scene_file

def create_camera(color=[0.1]*3):
    ## plot camera
    bx, by, bz = 0.15, 0.1, 0.1
    m = max([bx, by, bz])
    verts = np.array([
        [bx, by, bz], [bx, by, -bz], [bx, -by, bz], [bx, -by, -bz],
        [-bx, by, bz], [-bx, by, -bz], [-bx, -by, bz], [-bx, -by, -bz],
        [2 * bx, 2 * by, 2 * bz], [2 * bx, -2 * by, 2 * bz],
        [-2 * bx, 2 * by, 2 * bz], [-2 * bx, -2 * by, 2 * bz],
        [0, 0, 0], [5*m, 0, 0], [0, 5*m, 0], [0, 0, 5*m]
    ])
    edges = np.array([
        [0,1], [0,2], [0,4], [1,3], [1,5], [2,3],
        [2,6], [3,7], [4,5], [4,6], [5,7], [6,7],
        [0,8], [2,9], [4,10],[6,11],[8,9], [8,10],
        [9,11], [10,11], [12,13], [12,14], [12,15]
    ]).flatten()
    colors = np.array([color for _ in edges])
    colors[[40,41]] = (1, 0, 0)
    colors[[42,43]] = (0, 1, 0)
    colors[[44,45]] = (0, 0, 1)

    mesh = gl.GLLinePlotItem(pos=verts[edges], color=colors, width=1, antialias=True, mode='lines')
    return mesh


def create_grid(obj_points, board_shape, color=[0.5]*3):
    cols = board_shape[0]
    rows = board_shape[1]
    xyz_quiver = np.array([
        [0]*3,
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
    colors[[-6,-5]] = (1, 0, 0)
    colors[[-4,-3]] = (0, 1, 0)
    colors[[-2,-1]] = (0, 0, 1)

    mesh = gl.GLLinePlotItem(pos=verts[edges], color=colors, width=1.0, antialias=True, mode='lines')
    return mesh


def plot_calib_board(img_points, board_shape, camera_resolution, frame_fpath=None):
#     from IPython.display import clear_output
    
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
        lc = mc.LineCollection(pts[edges].reshape(-1, 2, 2), color='r', linewidths=0.25)

        plt.gca().add_collection(lc)
        plt.gca().set_xlim((0, camera_resolution[0]))
        plt.gca().set_ylim((camera_resolution[1], 0))
    
    plt.show()


class Animation:
    def __init__(self, title, data_dir, centered=False, dark_mode=False):
        self.app = QApplication.instance()
        if self.app == None:
            self.app = QApplication([])
        
        self.centered = False if "Scene" in title else centered
        self.dark_mode = dark_mode
        self.screen_res = self.app.desktop().screenGeometry()
        self.screen_res = np.array([self.screen_res.width(), self.screen_res.height()])
        
        theme_colors = ['w','k']
        pyqtgraph.setConfigOption('background', theme_colors[dark_mode])
        pyqtgraph.setConfigOption('foreground', theme_colors[not dark_mode])
        
        self.win = pyqtgraph.GraphicsWindow(title=title, size=self.screen_res/2)
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)
        
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor([255*(not dark_mode)]*3)
#         self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.opts['distance']=15
        self.view.orbit(-135,10)
        
        # camera pose
        self.k_arr, self.d_arr, self.r_arr, self.t_arr, self.cam_res, self.n_cams, self.scene_fpath = find_scene_file(data_dir)
        self.cam_pos = []
        
        grid = gl.GLGridItem(size=QtGui.QVector3D(50,50,0), color=[abs(55-255*(not dark_mode))]*3)
        self.view.addItem(grid)
        
        if not self.centered:
            for r, t in zip(self.r_arr, self.t_arr):
                self.plot_camera(r, t)

            scene_center = np.mean(self.cam_pos, axis=0)
            scene_center[2] = 0
            grid.translate(*scene_center)
            self.view.pan(*scene_center)
        else:
            grid.translate(0,0,-0.5)

        
    def rodrigues_to_vec(self, r):
        ang = 180 / np.pi * np.linalg.norm(r)
        return (ang, *r)

    def plot_camera(self, r, t):
        # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
        # T is the world origin position in the camera coordinates.
        # The world position of the camera is C = -(R^-1)@T.
        # Similarly, the rotation of the camera in world coordinates is given by R^-1
        R = r.T
        T = -R @ t
        self.cam_pos.append(T)
        # Convert for plotting
        R = cv2.Rodrigues(R)[0]
        R = self.rodrigues_to_vec(R)
        cam = create_camera([self.dark_mode]*3)
        cam.rotate(*R)
        cam.translate(*T)
        self.view.addItem(cam)
        
    def save_snapshot(self, filename, size=None):
        pyqtgraph.makeQImage(self.view.renderToArray(size if size else self.screen_res)).save(filename)
        self.app.quit()
        
    def show(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.exec_()


class Scene(Animation):
    def __init__(self, data_dir, **kwargs):
        Animation.__init__(self, "Scene Reconstruction", data_dir, **kwargs)
        self.layout.addWidget(self.view, 0, 0, 1, 1)
        
    def plot_calib_board(self, r, t, board_shape, board_edge_len):
        obj_pts = create_board_object_pts(board_shape, board_edge_len)
        calib_board = create_grid(obj_pts, board_shape)
        calib_board.translate(*t)
        r = self.rodrigues_to_vec(r)
        calib_board.rotate(*r)
        self.view.addItem(calib_board)

    def plot_points(self, points, color=[0]+[0.5]*3, size=None):
        scatter = gl.GLScatterPlotItem(pos=points, color=color, size=size if size else self.screen_res[0]/500, pxMode=True)
        scatter.setGLOptions('translucent')
        self.view.addItem(scatter)
    
    
class Cheetah(Animation):
    """ Plots a reconstruction of a cheetah
    :param scatter_frames: array the 3d positions of the dlc markers (joint positions and lure) with shape (. Use Cheetah(fte_positions, data_dir) to plot a single reconstruction (fte in thgis example). 
    """
    def __init__(self, scatter_frames, data_dir, reprojections=True, **kwargs):
        Animation.__init__(self, "Cheetah Reconstruction", data_dir, **kwargs)
        
        # indices correspond to joints in 'markers' variable
        lines_idxs = [0,1,0,2,1,2,1,3,0,3,2,3,3,4,4,5,5,6,6,7,
                      3,8,4,8,8,9,9,10,      # left front leg
                      3,11,4,11,11,12,12,13, # right front leg
                      4,14,5,14,14,15,15,16,
                      4,17,5,17,17,18,18,19]
#         lines_frames = [frame[lines_idxs, :] for frame in scatter_frames]
        self.scatter_frames = []
        self.lines_frames = []
        for frame in scatter_frames:
            if self.centered:
                dots_ave = np.nanmean(frame, axis=0)
                frame -= dots_ave
            self.scatter_frames.append(frame)
            self.lines_frames.append(frame[lines_idxs, :])
        
        self.scatter_frames = np.array(self.scatter_frames)
        self.lines_frames = np.array(self.lines_frames)
        self.n_frames = len(scatter_frames)
        self.frame = 0
        
        self.layout.addWidget(self.view, 0, 0, self.n_cams, 1)
        
        # create dots
        self.scatter = gl.GLScatterPlotItem(pos=self.scatter_frames[0], color=[1,0,0,1], size=self.screen_res[0]/250, pxMode=True)
        self.scatter.setGLOptions('translucent')
        self.view.addItem(self.scatter)
        
        # create links
        self.lines = gl.GLLinePlotItem(pos=self.lines_frames[0], color=[self.dark_mode]*3+[1], width=self.screen_res[0]/1250, antialias=True, mode='lines')
        self.lines.setGLOptions('translucent')
        self.view.addItem(self.lines)
        
        # ====== 2D ======
        if self.centered and reprojections:
            print("Reprojections are not permitted in centered mode")
        self.reprojections = False if self.centered else reprojections
        if self.reprojections:
            self.cam_data = []
            self.cam_lines = []
            cam_w = []
            for i in range(self.n_cams):
                cam_w.append(pyqtgraph.PlotWidget())
    #             cam_w.append(pyqtgraph.PlotWidget(color=[0]*3))
                self.cam_data.append(pyqtgraph.PlotDataItem(connect="pairs", name=f"Cam {i+1} Reprojection"))
    #             self.cam_data.append(pyqtgraph.PlotDataItem(connect="pairs", pen=pyqtgraph.mkPen("000000")))
                cam_w[i].getPlotItem().addItem(self.cam_data[i])
                cam_w[i].getPlotItem().setXRange(0, self.cam_res[0])
                cam_w[i].getPlotItem().setYRange(self.cam_res[1], 0)
                cam_w[i].getPlotItem().invertY()
                cam_w[i].getPlotItem().addLegend() # the legend doesn't show for some reason

                cam_w[i].sizeHint = lambda: pyqtgraph.QtCore.QSize(self.screen_res[0]/7, self.screen_res[1]/7)
                cam_w[i].setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                cam_w[i].setBackground([255*(not self.dark_mode)]*3)

                self.layout.addWidget(cam_w[i], i, 1, 1, 1)
                
                cam_lines_temp = []
                for j in range(self.n_frames):
                    cam_params = [self.k_arr[i], self.d_arr[i], self.r_arr[i], self.t_arr[i]]
                    lines_frames_2d = [project_points_fisheye(pt, *cam_params)[0] for pt in self.lines_frames[j]]
                    cam_lines_temp.append(lines_frames_2d)
                self.cam_lines.append(np.array(cam_lines_temp))
                self.cam_data[i].setData(self.cam_lines[i][0])

    def update(self):
        dots = self.scatter_frames[self.frame]
        links = self.lines_frames[self.frame]
        
#         if self.centered:
#             dots_ave = np.nanmean(dots,axis=0)
#             dots -= dots_ave
#             links -= dots_ave
# #             dots -= self.scatter_frames[self.frame][0]
# #             links -= self.lines_frames[self.frame][0]
        if self.reprojections:
            for i in range(self.n_cams):
                self.cam_data[i].setData(self.cam_lines[i][self.frame])
        
        self.scatter.setData(pos=dots)
        self.lines.setData(pos=links)
        
        self.frame = (self.frame+1) % self.n_frames

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(100) # speed of reconstruction
        self.show()
        

# def plot_cheetah_graphs(plot_style_fpath, lure_pts, x_opt):
def plot_cheetah_graphs(plot_style_fpath, x_opt):
    plt.style.use(plot_style_fpath)
    fig, axs = plt.subplots(8, 2, figsize=(13,30))
    
#     axs[0,0].plot(lure_pts)
#     axs[0,0].set_title("Lure positions")
#     axs[0,0].legend(['x', 'y', 'z'])
    
    axs[0,1].plot(x_opt[:, [0,1,2]])
    axs[0,1].set_title("Head positions")
    axs[0,1].legend(['x0', 'y0', 'z0'])

    axs[1,0].plot(x_opt[:, [3,17,31]])
    axs[1,0].set_title("Head angles")
    axs[1,0].legend(['phi0', 'theta0', 'psi0'])

    axs[1,1].plot(x_opt[:, [4,18,32]])
    axs[1,1].set_title("Neck angles")
    axs[1,1].legend(['phi1', 'theta1', 'psi1'])

    axs[2,0].plot(x_opt[:, [19]])
    axs[2,0].set_title("Front torso angles")
    axs[2,0].legend(['theta2'])

    axs[2,1].plot(x_opt[:, [6,20,34]])
    axs[2,1].set_title("Back torso angles")
    axs[2,1].legend(['phi3','theta3', 'psi3'])

    axs[3,0].plot(x_opt[:, [21,35]])
    axs[3,0].set_title("Tail base")
    axs[3,0].legend(['theta4', 'psi4'])

    axs[3,1].plot(x_opt[:, [22,36]])
    axs[3,1].set_title("Tail Mid")
    axs[3,1].legend(['theta5', 'psi5'])

    axs[4,0].plot(x_opt[:, [23]])
    axs[4,0].set_title("Left shoulder angles")
    axs[4,0].legend(['theta6'])

    axs[4,1].plot(x_opt[:, [24]])
    axs[4,1].set_title("Left front knee angle")
    axs[4,1].legend(['theta7'])

    axs[5,0].plot(x_opt[:, [25]])
    axs[5,0].set_title("Right shoulder angles")
    axs[5,0].legend(['theta8'])

    axs[5,1].plot(x_opt[:, [26]])
    axs[5,1].set_title("Right front knee angle")
    axs[5,1].legend(['theta9'])

    axs[6,0].plot(x_opt[:, [27]])
    axs[6,0].set_title("Left hip angle")
    axs[6,0].legend(['theta10'])

    axs[6,1].plot(x_opt[:, [28]])
    axs[6,1].set_title("Left back knee angle")
    axs[6,1].legend(['theta11'])

    axs[7,0].plot(x_opt[:, [29]])
    axs[7,0].set_title("Right hip angle")
    axs[7,0].legend(['theta12'])

    axs[7,1].plot(x_opt[:, [30]])
    axs[7,1].set_title("Right back knee angle")
    axs[7,1].legend(['theta13'])
    
#     fig.savefig(os.path.join(DATA_DIR, 'figure.pdf'))
    return fig, axs
