import os
import yaml
os.environ['DLClight'] = 'True'
import deeplabcut
from pathlib import Path
import sys
import os
import glob
from pprint import pprint

# HACKY FIX FOR SOME DLC BUG
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse


# examples of data_dir
# 05_03_2019/Jules/Flick
# 05_03_2019/Jules/Run
# 05_03_2019/Lily/Flick
# 05_03_2019/Lily/Run
# 07_03_2019/Menya/Flick
# 07_03_2019/Menya/Run
# 07_03_2019/Phantom/Flick
# 07_03_2019/Phantom/Run
# 09_03_2019/Jules/Flick1
# 09_03_2019/Jules/Flick2
# 09_03_2019/Lily/Flick
# 09_03_2019/Lily/Run
# 27_02_2019/Kiara/Flick
# 27_02_2019/Kiara/Run
# 27_02_2019/Romeo/Flick
# 27_02_2019/Romeo/Run
# 27_02_2019/Ebony


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', type=str, default='/data/dlc/cheetah_videos/27_02_2019/Kiara/Flick', help='the directory path where the videos are put')
parser.add_argument('--dlc_dir', type=str, default='/data/dlc/Cheetah-UCT-2019-10-14', help='the directory path where the pre-trained data are put')
parser.add_argument('--iteration', type=int, default=18, help='the number of iterations you want to use')
parser.add_argument('--video_type', type=str, default='mp4', help='the extension of videos')
args = parser.parse_args()
pprint(args)


args.config = os.path.join(args.dlc_dir, 'config.yaml')

# getting paths!
train_pose_config_path, test_pose_config_path, snapshot_dir = deeplabcut.return_train_network_path(args.config)
train_pose_config_path = os.path.join(
    args.dlc_dir,
    f"dlc-models/iteration-{args.iteration}/CheetahOct14-trainset95shuffle1/train/pose_cfg.yaml"
)
test_pose_config_path = os.path.join(
    args.dlc_dir,
    f"dlc-models/iteration-{args.iteration}/CheetahOct14-trainset95shuffle1/test/pose_cfg.yaml"
)


if __name__ == "__main__":

    # ##########################################
    # ######### STEP 0 -- create training dataset:
    # ##########################################

    # print("\n\n========== STEP 0:Creating the dataset - this takes a long time... ==========\n")
    # deeplabcut.create_training_dataset(args.config)
    # # deeplabcut.load_demo_data(args.config)

    # #########################################
    # ######## STEP 1 -- collect pairwise stats:
    # #########################################
    # print("\n\n========== STEP 1: collecting pairwise stats ==========\n")

    # cfg=deeplabcut.utils.read_plainconfig(train_pose_config_path)
    # #setting path:
    # cfg['pairwise_stats_fn']=str(os.path.join(Path(train_pose_config_path).parent,"pwstats.mat"))
    # cfg['pairwise_stats_collect']=True
    # cfg['pairwise_predict']=False #otherwise they are loaded, but don't exist yet...
    # cfg['dataset_type']='pairwise'
    # cfg['max_input_size']=3000

    # #for pairwise stats collection we need scale 1!
    # cfg['global_scale']=1.
    # cfg['scale_jitter_lo']=1.
    # cfg['scale_jitter_up']=1.
    # cfg['set_pairwise_stats_collect']=True
    # deeplabcut.utils.write_plainconfig(train_pose_config_path,cfg)
    # deeplabcut.pairwise_stats(train_pose_config_path) #will not train without this!!

    # #####################################
    # ######### STEP 2: SWITCH STATES >> to prepare for training
    # ####################################
    # print("\n\n========== STEP 2: Preparing for training ==========\n")

    # cfg = deeplabcut.utils.read_plainconfig(train_pose_config_path)
    # cfg['location_refinement'] = True
    # cfg['pairwise_predict'] = True
    # cfg['pairwise_loss_weight'] = .1 #relative weight of loss
    # # now change params to usual training params
    # cfg['global_scale'] = 1.
    # cfg['scale_jitter_lo'] = .25
    # cfg['scale_jitter_up'] = 1.1
    # cfg['dataset_type'] = 'pairwise'
    # cfg['pairwise_stats_collect'] = False
    # cfg['cropratio'] = 1.0
    # cfg['max_input_size'] = 3000
    # cfg['net_type'] = 'resnet_50'
    # deeplabcut.utils.write_plainconfig(train_pose_config_path,cfg)
    # # train the network
    # print("\n\n========== TRAINING!!! :D ==========\n")
    # deeplabcut.train_network(config, saveiters=50000, displayiters=100)

    # ######################################
    # ########## STEP 3: Evaluate
    # #####################################
    # # Run this block before you analyze, even if you don't evaluate!
    # cfg = deeplabcut.utils.read_plainconfig(test_pose_config_path)
    # cfg['pairwise_predict'] = True
    # deeplabcut.utils.write_plainconfig(test_pose_config_path, cfg)
    # deeplabcut.evaluate_network(args.config)

    ######################################
    ########## STEP 4: Analyze
    #####################################
    video_file_paths = glob.glob(os.path.join(args.data_dir, f'*.{args.video_type}'))

    deeplabcut.analyze_videos(
        args.config, video_file_paths,
        videotype=args.video_type,
        save_as_csv=True
    )

    deeplabcut.create_labeled_video(
        args.config, video_file_paths,
        draw_skeleton=True
    )



    ##########################
    ####@change that path:
    #############

    # import matplotlib.pyplot as plt
    # import numpy as np
    # import pickle

    # resultsfilename='/media/liam/f5e329a7-24ee-411f-85a2-0435a1ee9c09/liam/Cheetah-UCT-2019-10-14/evaluation-results/iteration-10/CheetahOct14-trainset95shuffle1/DLC_resnet50_CheetahOct14shuffle1_1030000-snapshot-1030000.h5'
    # with open(resultsfilename.split('.h5')[0] + 'pairwise.pickle', 'rb') as f:
    #             data=pickle.load(f)

    # fn=22

    # im = deeplabcut.auxfun_videos.imread(os.path.join(base,data[fn]['name']))

    # plt.imshow(im)

    # # x & y bodypart detections:
    # x=data[fn]['pose'][0::3]
    # y=data[fn]['pose'][1::3]

    # xpw=data[fn]['pws'][:,:,:,0]
    # ypw=data[fn]['pws'][:,:,:,1]

    # #plot bodyparts
    # plt.plot(x,y,'r.')

    # #and vector predictions!
    # numbodyparts=len(x)
    # for base in range(numbodyparts):
    #     for i in range(numbodyparts):
    #         plt.plot([x[base],x[base]+xpw[0,base,i]],
    #                     [y[base],y[base]+ypw[0,base,i]])

    # plt.show()

