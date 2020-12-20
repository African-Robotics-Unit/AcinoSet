# AcinoSet: A 3D pose dataset of Cheetahs in the wild<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1608473251355-R6MD2DPAGXD541O6KSPO/ke17ZwdGBToddI8pDm48kDJiRRinvyl0ibURJcD42oMUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcQRhUxETRWa-oq147TtIoC7IIYHcXSEvrmlBoYmbrKNZ_GGuik8tacc4P7_d_fn_0/cheetahTurn.png?format=2500w" width="375" title="AcinoSet" alt="Cheetah" align="right" vspace = "50">

Daniel Joska, Liam Clark, Naoya Muramatsu, Ricardo Jericevich, Fred Nicolls, Alexander Mathis, Mackenzie W. Mathis, Amir Patel 


 AcinoSet is a dataset of 13 free-running cheetahs in the wild that contains 119,490 frames of multi-view synchronized high-speed video footage, camera calibration files and 8,522 human-annotated frames. We utilize markerless animal pose estimation with DeepLabCut to provide 2P keypoints. Then we use three methods that can serve as strong baselines for 3D pose estimation tool development: traditional sparse bundle adjustment, an Extended Kalman Filter, and a trajectory optimization-based method we call Full Trajectory Estimation. We believe this dataset will be useful benchmark for a diverse range of fields such as ecology, robotics, biomechanics as well as computer vision.

### AcinoSet code by:
- [Naoya Muramatsu](https://github.com/DenDen047) | [Daniel Joska](https://github.com/DJoska) | [Liam Clark](https://github.com/LiamClarkZA) | [Ricky Jericevich](https://github.com/@rickyjericevich)

## Prerequisites

- Python3, anaconda, code dependencies are within conda env files.

## 2D --> 3D Data Pipeline:

### What we provide: 
- 8,522 [ground truth 2D frames](https://www.dropbox.com/sh/9y3rb9m5n3sbhwh/AACvUBuloEvAUFJFYZ9IqtbLa/data/hand_labeled_data?dl=0&subfolder_nav_tracking=1)
- 119,490 processed frames with 2D keypoint estimation outputs [(H5 files as in the DLC format, and raw video)](https://www.dropbox.com/sh/9y3rb9m5n3sbhwh/AABnfdKGHb0GrfHT7ynqf1APa/data?dl=0&subfolder_nav_tracking=1) 
    - this is currently organized by date > animal ID > "run/attempt"
- [3D files that are processed using our FTE baseline model](https://www.dropbox.com/sh/9y3rb9m5n3sbhwh/AABnfdKGHb0GrfHT7ynqf1APa/data?dl=0&subfolder_nav_tracking=1). These can be used for 3D GT.
   - these files are called `traj_opt.pickle`, have a related `scene_sba.json` file, and can be loaded in the GUI.
- A GUI to inspect the 3D dataset, which can be found [here](https://github.com/African-Robotics-Unit/acinonet_viewer)


The following sections document how this was created by the code within this repo:

#### Pre-trained DeepLabCut Model:

- [ ] You can use the `full_cheetah` model provided in the [DLC Model Zoo](http://modelzoo.deeplabcut.org)  To re-create the H5 files (or on new videos). 
- [X] Here, we also already provide the videos and H5 outputs of all frames, [here]().

##### Labelling Cheetah Body Positions:

If you want to label more cheetah data, you can also do so within the [DeepLabCut framework](https://github.com/DeepLabCut/DeepLabCut). We provide a conda file for an easy-install, but please see the [repo](https://github.com/DeepLabCut/DeepLabCut) for installation and instructions for use.
```sh
$ conda env create -f conda_envs/dDLC.yml -n DLC
```

#### Optionally: Manually Defining the Shared Points for 3D calibration:

You can manually define points on each video with [Argus](http://argus.web.unc.edu/). Documentation is [here](http://argus.web.unc.edu/tutorial/#Clicker).

Build the environment:
```sh
$ conda env create -f conda_envs/argus.yml -n argus
```

Launch Argus/Clicker:
```sh
$ python
>>> import argus_gui as ag; ag.ClickerGUI()
```

Keyboard Shortcuts:
- `G` ... to a specific frame
- `X` ... to switch the sync mode setting the windows to the same frame
- `A` ... to use the auto-tracker
- `7`, `Y`, `U`, `I` ... growing the view finder at the bottom right
- `O` ... to bring up the options dialog
- `S` ... to bring up a save dialog

Then you must convert the output data from Argus to work with the rest of the pipeline (here is an example):
```sh
$ python converter_argus.py \
    --data_dir ../data/2019_03_07/extrinsic_calib/videos
```

### Intrinsic & Extrinsic Calibration:

Build the environment.
```sh
$ conda env create --file conda_envs/cv.yml
```

Launch Jupyter Lab:
```sh
$ jupyter lab
```

Run `calib_with_gui.ipynb`, and follow the instructions.


### Full Trajectory Optimization:

Prepare the environment.
```sh
$ pyenv local anaconda3-5.2.0/envs/cv
```

Run `full_traj_opt.py`, or use the supplied Jupyter Notebook:
```sh
$ python full_traj_opt.py \
    --n_camera 6 \
    --logs_dir ../logs \
    --configs_dir ../configs \
    --data_dir ../data/2019_03_09/jules/flick1 \
    --scene_file ../data/2019_03_09/extrinsic_calib/scene_sba.json
```

If you want to view the 3D animation, run `full_traj_optimisation.ipynb` and follow the instructions!
