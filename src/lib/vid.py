"""
This code was derived from deeplabcut.utils make_labeled_video & video_processor
https://github.com/DeepLabCut/DeepLabCut/blob/master/deeplabcut/utils/make_labeled_video.py
https://github.com/DeepLabCut/DeepLabCut/blob/master/deeplabcut/utils/video_processor.py
"""

import os
import cv2
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm import trange
from skimage.draw import circle, line_aa


class VideoProcessor(object):
    """
    Base class for a video processing unit, implementation is required for video loading and saving.
    out_h and out_w are the output height and width respectively.
    """

    def __init__(self, in_name="", out_name="", nframes=-1, fps=30, codec="X264", out_h="", out_w=""):
        self.in_name = in_name
        self.out_name = out_name
        self.nframes = nframes
        self.CODEC = codec
        self.in_h = 0
        self.in_w = 0
        self.FPS = fps
        self.nc = 3
        self.i = 0

        try:
            if self.in_name != "":
                self.in_vid = self.get_video()
                self.get_info()
                self.out_h = 0
                self.out_w = 0
            if self.out_name != "":
                if out_h == "" and out_w == "":
                    self.out_h = self.in_h
                    self.out_w = self.in_w
                else:
                    self.out_w = out_w
                    self.out_h = out_h
                self.out_vid = self.create_video()

        except Exception as ex:
            print("Error: %s", ex)

    def load_frame(self):
        try:
            frame = self._read_frame()
            self.i += 1
            return frame
        except Exception as ex:
            print("Error: %s", ex)

    def height(self):
        return self.in_h

    def width(self):
        return self.in_w

    def fps(self):
        return self.FPS

    def counter(self):
        return self.i

    def frame_count(self):
        return self.nframes
    
    def codec(self):
        return self.CODEC

    def get_video(self):
        """
        implement your own
        """
        pass

    def get_info(self):
        """
        implement your own
        """
        pass

    def create_video(self):
        """
        implement your own
        """
        pass

    def _read_frame(self):
        """
        implement your own
        """
        pass

    def save_frame(self, frame):
        """
        implement your own
        """
        pass

    def close(self):
        """
        implement your own
        """
        pass


class VideoProcessorCV(VideoProcessor):
    """
    OpenCV implementation of VideoProcessor
    requires opencv-python==3.4.0.12
    """

    def __init__(self, *args, **kwargs):
        super(VideoProcessorCV, self).__init__(*args, **kwargs)

    def get_video(self):
        return cv2.VideoCapture(self.in_name)

    def get_info(self):
        self.in_w = int(self.in_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.in_h = int(self.in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frames = int(self.in_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = self.in_vid.get(cv2.CAP_PROP_FPS)
        self.nc = 3
        if self.nframes == -1 or self.nframes > all_frames:
            self.nframes = all_frames

    def create_video(self):
        fourcc = cv2.VideoWriter_fourcc(*self.codec())
        return cv2.VideoWriter(self.out_name, fourcc, self.FPS, (self.out_w, self.out_h), True)

    def _read_frame(self):  # return RGB (rather than BGR)!
        # return cv2.cvtColor(np.flip(self.vid.read()[1],2), cv2.COLOR_BGR2RGB)
        return np.flip(self.in_vid.read()[1], 2)

    def save_frame(self, frame):
        self.out_vid.write(np.flip(frame, 2))

    def close(self):
        self.out_vid.release()
        self.in_vid.release()


def get_segment_indices(bodyparts2connect, all_bpts):
    bpts2connect = []
    for bpt1, bpt2 in bodyparts2connect:
        if bpt1 in all_bpts and bpt2 in all_bpts:
            bpts2connect.extend(
                zip(
                    *(
                        np.flatnonzero(all_bpts == bpt1),
                        np.flatnonzero(all_bpts == bpt2),
                    )
                )
            )
    return bpts2connect


def CreateVideo(clip, df, pcutoff, dotsize, colormap, bodyparts2plot, bodyparts2connect, draw_skeleton):
    """Creating individual frames with labeled body parts and making a video"""
    bpts = df.columns.get_level_values("bodyparts")
    all_bpts = bpts.values[::3]
    if draw_skeleton:
        color_for_skeleton = (np.array(mcolors.to_rgba('white'))[:3] * 255).astype(np.uint8)
        # recode the bodyparts2connect into indices for df_x and df_y for speed
        bpts2connect = get_segment_indices(bodyparts2connect, all_bpts)

    ny, nx = clip.height(), clip.width()

    print("Duration of video [s]: {}, recorded with {} fps!".format(round(clip.frame_count() / clip.fps(), 2), round(clip.fps(), 2)))
    print("Overall # of frames: {} with cropped frame dimensions: {} {}".format(clip.frame_count(), nx, ny))
    print("Generating frames and creating video.")

    df_x, df_y, df_likelihood = df.values.reshape((df.index.size, -1, 3)).T
    colorclass = plt.cm.ScalarMappable(cmap=colormap)

    bplist = bpts.unique().to_list()
    nbodyparts = len(bplist)
    map2bp = list(range(len(all_bpts)))
    map2id = [0 for _ in map2bp]

    keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
    bpts2color = [(ind, map2bp[ind], map2id[ind]) for ind in keep]

    C = colorclass.to_rgba(np.linspace(0, 1, nbodyparts))
    colors = (C[:, :3] * 255).astype(np.uint8)
    with np.errstate(invalid="ignore"):
        for frame_idx in trange(clip.frame_count()):
            image = clip.load_frame()
            try:
                idx = df.index.get_loc(frame_idx)
                # Draw the skeleton for specific bodyparts to be connected
                if draw_skeleton:
                    for bpt1, bpt2 in bpts2connect:
                        if (
                            np.all(df_likelihood[[bpt1, bpt2], idx] > pcutoff)
                            or np.isnan(df_likelihood[[bpt1, bpt2], idx]).all()
                        ):
                            if not (
                                np.isnan(df_x[[bpt1, bpt2], idx]).any()
                                or np.isnan(df_y[[bpt1, bpt2], idx]).any()
                            ):
                                # change to cv2.line
                                rr, cc, val = line_aa(
                                    int(np.clip(df_y[bpt1, idx], 0, ny - 1)),
                                    int(np.clip(df_x[bpt1, idx], 0, nx - 1)),
                                    int(np.clip(df_y[bpt2, idx], 1, ny - 1)),
                                    int(np.clip(df_x[bpt2, idx], 1, nx - 1)),
                                )
                                image[rr, cc] = color_for_skeleton

                for ind, num_bp, num_ind in bpts2color:
                    if (df_likelihood[ind, idx] > pcutoff) or np.isnan(df_likelihood[ind, idx]):
                        color = colors[num_bp]
                        # change to cv2.circle
                        rr, cc = circle(df_y[ind, idx], df_x[ind, idx], dotsize, shape=(ny, nx))
                        image[rr, cc] = color
            except KeyError:
                pass
            clip.save_frame(image)
    clip.close()


def proc_video(out_dir, bodyparts, codec, bodyparts2connect, outputframerate, draw_skeleton, pcutoff, dotsize, colormap, video):
    """Helper function for create_labeled_videos"""
    video = os.path.abspath(video)
    vname = os.path.splitext(os.path.basename(video))[0]
    
    start_path = os.getcwd()
    os.chdir(out_dir) 

    print("Loading {} and data.".format(vname))
    try:
        filepath = glob(vname + '*.h5')[0]
        videooutname = filepath.replace(".h5", ".mp4")

        df = pd.read_hdf(filepath)
        labeled_bpts = [bp for bp in df.columns.get_level_values("bodyparts").unique() if bp in bodyparts]
        clip = VideoProcessorCV(in_name=video, out_name=videooutname, codec=codec)
        
        CreateVideo(clip, df, pcutoff, dotsize, colormap, labeled_bpts, bodyparts2connect, draw_skeleton)

        os.chdir(start_path)
        
    except FileNotFoundError as e:
        print(e)