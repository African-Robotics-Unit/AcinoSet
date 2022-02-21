import pickle
from typing import Dict
import os

def load_pickle(pickle_file) -> Dict:
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    return(data)

def rotate_right() -> None:
    """
    Rotates the axes right
    """
    azimuth = a.azim
    a.view_init(elev=20., azim=azimuth+10)
    update_canvas()

def rotate_left() -> None:
    """
    Rotates the axes left
    """
    azimuth = a.azim
    a.view_init(elev=20., azim=azimuth-10)
    update_canvas()

def update_canvas() -> None:
    """
    Replots canvas on the GUI with updated points
    """
    a.set_xlim3d(3, 7)
    a.set_ylim3d(6, 10)
    a.set_zlim3d(0,4)
    canvas = FigureCanvasTkAgg(f, self)
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    canvas._tkcanvas.place(relx=0.5, rely=0.45, anchor="center")

def init():
    a.set_xlim3d(-1, 8)
    a.set_ylim3d(6, 10)
    a.set_zlim3d(0,1)
    a.view_init(elev=20., azim=30)

def update(i):
    plot_results(i)

def plot_results(frame=0) -> None:
    """
    Plots results for the given skeleton (frame 0)
    """
    pose_dict = {}
    currdir = os.getcwd()
    skel_name = (field_name1.get())
    skelly_dir = os.path.join(currdir, "skeletons", (skel_name + ".pickle"))
    results_dir = os.path.join(currdir, "data", "results", (skel_name + ".pickle"))

    skel_dict = bd.load_skeleton(skelly_dir)
    results = an.load_pickle(results_dir)
    links = skel_dict["links"]
    markers = skel_dict["markers"]

    for i in range(len(markers)):
        pose_dict[markers[i]] = [results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2]]
        a.scatter(results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2])
    
    for link in links:
        if len(link)>1:
            a.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
                [pose_dict[link[0]][1], pose_dict[link[1]][1]],
            [pose_dict[link[0]][2], pose_dict[link[1]][2]])

    update_canvas()

def next_frame() -> None:
    """
    Plots the next frame of the results
    """
    self.current_frame+=1
    a.clear()
    plot_results(self.current_frame)

def prev_frame() -> None:
    """
    Plots the previous frame of the results
    """
    self.current_frame-=1
    a.clear()
    plot_results(self.current_frame)

def play_animation() -> None:
    """
    Creates an animation or "slide show" of plotted results in the GUI
    """
    ani = FuncAnimation(f, update, 19, 
                        interval=40, blit=True)
    writer = PillowWriter(fps=25)  
    ani.save("test.gif", writer=writer)

if __name__=="__main__":
    results_dir = os.path.join("C://Users//user-pc//Documents//Scripts//amaan", "data", "results", ("traj_results.pickle"))
    res = load_pickle(results_dir)
    print(res["positions"].shape)