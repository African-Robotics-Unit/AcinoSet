# AcinoNet

Authors
- Liam Clark (@LiamClarkZA)
- Naoya Muramatsu (@DenDen047)
- Ricky Jericevich (@rickyjericevich)

Original Repository obtained from @DenDen047

## Prerequisites

- Pyenv

## The Whole Step

### Preparation

#### Pre-training DeepLabCut Model

TBA

#### Labelling Cheetah Body Positions

Build the environment on pyenv in Linux.
```sh
$ pyenv install miniconda3-4.7.12   # take a while
$ pyenv local miniconda3-4.7.12
$ conda env create -f conda_envs/dlc.yml -n dlc
$ pyenv local --unset   # unset the version of the current directory
$ pyenv local miniconda3-4.7.12/envs/dlc
```

Run below command.
```sh
$ python dlc_labelling.py \
    --data_dir /data/dlc/cheetah_videos/2019_02_27/kiara/flick \
    --dlc_dir /data/dlc/Cheetah-UCT-2019-10-14 \
    --iteration 18
```

#### Option: Manually Defining the Shared Points

You can manually define points on each video with [Argus](http://argus.web.unc.edu/).
You can easily learn the usage on http://argus.web.unc.edu/tutorial/#Clicker.

Build the environment on pyenv.
```sh
$ pyenv install miniconda3-4.7.12   # take a while
$ pyenv local miniconda3-4.7.12
$ conda env create -f conda_envs/argus.yml -n argus
$ pyenv local --unset   # unset the version of the current directory
$ pyenv local miniconda3-4.7.12/envs/argus
```

Launch Argus/Clicker as follows.
```sh
$ python
>>> import argus_gui as ag; ag.ClickerGUI()
```

Keyboard:
- `G` ... to a specific frame
- `X` ... to switch the sync mode setting the windows to the same frame
- `A` ... to use the auto-tracker
- `7`, `Y`, `U`, `I` ... growing the view finder at the bottom right
- `O` ... to bring up the options dialog
- `S` ... to bring up a save dialog

Then you must convert the output data from Argus.
```sh
$ python converter_argus.py \
    --data_dir ../data/2019_03_07/extrinsic_calib/videos
```

### Intrinsic & Extrinsic Calibration

Build the environment.
```sh
$ pyenv install anaconda3-5.2.0     # take a while
$ pyenv local anaconda3-5.2.0
$ conda env create --file conda_envs/cv.yml
$ pyenv local --unset   # カレントディレクトリのバージョン指定を除去
$ pyenv local anaconda3-5.2.0/envs/cv
```

Launch Jupyter Lab.
```sh
$ jupyter lab
```

Run `calib_with_gui.ipynb`, and follow the instruction in it.


### Full Trajectory Optimization

Prepare the environment.
```sh
$ pyenv local anaconda3-5.2.0/envs/cv
```

Run `full_traj_opt.py`. Like,
```sh
$ python full_traj_opt.py \
    --n_camera 6 \
    --logs_dir ../logs \
    --configs_dir ../configs \
    --data_dir ../data/2019_03_09/jules/flick1 \
    --scene_file ../data/2019_03_09/extrinsic_calib/scene_sba.json
```

If you want to watch 3D animation, run `full_traj_optimisation.ipynb` and follow the instruction in it.


## References

- [pyenv内のAnacondaで環境構築したい](https://qiita.com/kabayan55/items/40cb9a8ddbb5763ed5a5)

