# AcinoSet: A 3D Pose Estimation Dataset and Baseline Models for Cheetahs in the Wild<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1608473251355-R6MD2DPAGXD541O6KSPO/ke17ZwdGBToddI8pDm48kDJiRRinvyl0ibURJcD42oMUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcQRhUxETRWa-oq147TtIoC7IIYHcXSEvrmlBoYmbrKNZ_GGuik8tacc4P7_d_fn_0/cheetahTurn.png?format=2500w" width="375" title="AcinoSet" alt="Cheetah" align="right" vspace = "50">

Daniel Joska, Liam Clark, Naoya Muramatsu, Ricardo Jericevich, Fred Nicolls, Alexander Mathis, Mackenzie W. Mathis, Amir Patel 


 AcinoSet is a dataset of free-running cheetahs in the wild that contains 119,490 frames of multi-view synchronized high-speed video footage, camera calibration files and 7,588 human-annotated frames. We utilize markerless animal pose estimation with DeepLabCut to provide 2D keypoints (in the 119K frames). Then, we use three methods that serve as strong baselines for 3D pose estimation tool development: traditional sparse bundle adjustment, an Extended Kalman Filter, and a trajectory optimization-based method we call Full Trajectory Estimation. The resulting 3D trajectories, human-checked 3D ground truth, and an interactive tool to inspect the data is also provided. We believe this dataset will be useful for a diverse range of fields such as ecology, robotics, biomechanics, as well as computer vision.

### AcinoSet code by:
- [Liam Clark](https://github.com/LiamClarkZA) | [Ricky Jericevich](https://github.com/@rickyjericevich) | [Daniel Joska](https://github.com/DJoska) | [Naoya Muramatsu](https://github.com/DenDen047)

## Prerequisites

- Python3, anaconda, code dependencies are within conda env files.

## What we provide: 

- 7,588 [ground truth 2D frames](https://www.dropbox.com/sh/z3uv6pnk7paygph/AAAiJOavquW89uPlz_Jzjtfua?dl=0)
- 119,490 processed frames with 2D keypoint estimation outputs [(H5 files as in the DLC format, and raw video)](https://www.dropbox.com/sh/kp5kmatbv5cdjx2/AABfJGb7ktVK_L0lybOLQIbJa?dl=0) 
    - this is currently organized by date > animal ID > "run/attempt"
- [3D files that are processed using our FTE baseline model](https://www.dropbox.com/sh/kp5kmatbv5cdjx2/AABfJGb7ktVK_L0lybOLQIbJa?dl=0). These can be used for 3D GT.
   - these files are called `fte.pickle`, have a related `(n)_cam_scene_sba.json` file, and can be loaded in the GUI.
- A GUI to inspect the 3D dataset, which can be found [here](https://github.com/African-Robotics-Unit/acinoset_viewer)


The following sections document how this was created by the code within this repo:

### Pre-trained DeepLabCut Model:

- You can use the `full_cheetah` model provided in the [DLC Model Zoo](http://modelzoo.deeplabcut.org) to re-create the existing H5 files (or on new videos). 
- Here, we also already provide the videos and H5 outputs of all frames, [here](https://www.dropbox.com/sh/kp5kmatbv5cdjx2/AABfJGb7ktVK_L0lybOLQIbJa?dl=0).

### Labelling Cheetah Body Positions:

If you want to label more cheetah data, you can also do so within the [DeepLabCut framework](https://github.com/DeepLabCut/DeepLabCut). We provide a conda file for an easy-install, but please see the [repo](https://github.com/DeepLabCut/DeepLabCut) for installation and instructions for use.
```sh
$ conda env create -f conda_envs/DLC.yml -n DLC
```

## AcinoSet Setup:

Navigate to the AcinoSet folder and build the environment:
```sh
$ conda env create -f conda_envs/acinoset.yml
```

Launch Jupyter Lab:
```sh
$ jupyter lab
```

## Camera Calibration and 3D Reconstruction:

### Intrinsic & Extrinsic Calibration:

Run `calib_with_gui.ipynb` and follow the instructions.

Alternatively, if the checkerboard points detected in `calib_with_gui.ipynb` are unsatisfactory, open `saveMatlabPointsForAcinoSet.m` in MATLAB and follow the instructions. Note that this requires MATLAB 2020b or later.

#### Optionally: Manually Defining the Shared Points for extrinsic calibration:

You can manually define points on each video in a scene with [Argus](http://argus.web.unc.edu/) Clicker. A quick tutorial is found [here](http://argus.web.unc.edu/tutorial/#Clicker).

Build the environment:
```sh
$ conda env create -f conda_envs/argus.yml
```

Launch Argus Clicker:
```sh
$ python
>>> import argus_gui as ag; ag.ClickerGUI()
```

Keyboard Shortcuts (See documentation [here](http://argus.web.unc.edu/wp-content/uploads/sites/9976/2019/01/Argus-Documentation_1.1.pdf) for more):
- `G` ... to a specific frame
- `X` ... to switch the sync mode setting the windows to the same frame
- `O` ... to bring up the options dialog
- `S` ... to bring up a save dialog

Then you must convert the output data from Argus to work with the rest of the pipeline (here is an example):
```sh
$ python argus_converter.py \
    --data_dir ../data/2019_03_07/extrinsic_calib/argus_folder
```

### 3D Reconstruction:

To reconstruct a cheetah into 3D, we offer three different trajectory refinement options on top of standard triangulation (TRI):

-  Sparse Bundle Adjustment (SBA)
-  Extended Kalman Filter (EKF)
-  Full Trajectory Estimation (FTE)

You can run each refinement option seperately. For example, simply open `FTE.ipynb` and follow the instructions!
Otherwise, you can run all types of refinements in one go:
```sh
python all_optimizations.py --data_dir 2019_03_09/lily/run --start_frame 70 --end_frame 170 --dlc_thresh 0.5
```

**NB**: When running the FTE, we recommend that you use the MA86 solver. For details on how to set this up, see [these instructions](https://github.com/African-Robotics-Unit/docs/blob/main/linear-solvers.md).
