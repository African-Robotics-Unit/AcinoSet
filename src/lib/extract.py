import os
import cv2
import json
import colorsys
import numpy as np


def draw_text(img, text, move_text_lower_pixels=0, fontColor = [255]*3):
    fontScale = img.shape[0]/750
    scale = fontScale/2
    bottomLeftCornerOfText = (int(10*scale), int((60+move_text_lower_pixels)*scale))
    font = cv2.FONT_HERSHEY_DUPLEX # cv2.FONT_HERSHEY_SIMPLEX
    fontScale = img.shape[0]/750
    lineType = 4

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    
def get_frames(vid_fpath, frame_output_dir):
    cap = cv2.VideoCapture(vid_fpath)
    if cap.isOpened():
        winname = vid_fpath
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        go_next = True
        while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) > 0:
            if go_next:
                ret, frame = cap.read()

            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if ret:
                draw_text(frame, f'frame {int(curr_frame)}')
                cv2.imshow(winname, frame)
            else:
                break

            key = cv2.waitKey(10) & 0xFF
            if key == ord('.'):
                #cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame+48) # skip 50 frames
                go_next = True
            elif key == ord(','):
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame - 2)
            elif key == ord('s'):
                cv2.imwrite(os.path.join(frame_output_dir, f'img{int(curr_frame):05}.jpg'), frame)
            elif key == ord('q'):
                break
            else:
                go_next = False
        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Could not open', vid_fpath)
        
        
def manual_label(vid_fpaths, out_fpath):
    VideoLabelSession(vid_fpaths, out_fpath)


# This is a WIP! Improve this to the point where Argus Clicker is no longer needed
class VideoLabelSession(object):

    def __init__(self, video_filepaths, out_fpath):
        self.out_fpath = out_fpath
        self.n_vids = len(video_filepaths)
        self.vid_caps = []
        self.vid_names = []
        self.show_prev=True
        for i, p in enumerate(video_filepaths):
            self.vid_caps.append(cv2.VideoCapture(p))
            if self.vid_caps[i].isOpened():
                self.vid_names.append(os.path.basename(p))
                print('Successfully opened', self.vid_names[i])
                if i == 0:
                    self.frame_count = int(self.vid_caps[i].get(cv2.CAP_PROP_FRAME_COUNT))
                    self.frame_width = int(self.vid_caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.frame_height = int(self.vid_caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    assert self.frame_count == int(self.vid_caps[i].get(cv2.CAP_PROP_FRAME_COUNT))
                    assert self.frame_width == int(self.vid_caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
                    assert self.frame_height == int(self.vid_caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                print('Could not open', p)
        self.frame_position = 0
        self.frame_circles = np.empty((self.n_vids, 2))
        self.frame_circles[:, :] = np.nan
        if os.path.isfile(out_fpath):
            with open(self.out_fpath, 'r') as f:
                self.frame_circles_saved = np.array(json.load(f)['points'])
                print('Points loaded.')
        else:
            self.frame_circles_saved = None
        self.frames = np.zeros((self.n_vids, self.frame_width, self.frame_height, 3), dtype=np.uint8)
        self.frame_imgs = np.zeros((self.n_vids, self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.mode = 'draw_circle'

        # Create window
        cv2.startWindowThread()

        for i in range(self.n_vids):
            cv2.namedWindow(self.vid_names[i], cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.resizeWindow(self.vid_names[i], 820, 600)
            cv2.setMouseCallback(self.vid_names[i], self.handle_mouse_events, i)
        self.load_frame_images()
        self.loop()

    def load_frame_images(self):
        for i in range(self.n_vids):
            self.vid_caps[i].set(cv2.CAP_PROP_POS_FRAMES, self.frame_position)
            ret, img = self.vid_caps[i].read()
            if ret:
                self.frame_imgs[i] = img
            else:
                print('Could not read frame', self.frame_position, 'from', self.vid_names[i])

    def render_frames(self):
        self.frames = self.frame_imgs.copy()
        for i in range(self.n_vids):
            if not np.isnan(self.frame_circles[i]).any():
                pt = tuple(self.frame_circles[i].astype(np.int))
                cv2.circle(self.frames[i], pt, 5, (255, 0, 255), -1)
            if (self.frame_circles_saved is not None) and (self.show_prev):
                for j, pts in enumerate(self.frame_circles_saved):
                    if not np.isnan(pts[i]).any():
                        pt = tuple(pts[i].astype(np.int))
                        col = tuple(
                            [int(c * 255) for c in colorsys.hsv_to_rgb(j / len(self.frame_circles_saved), 1., 1.)])
                        cv2.circle(self.frames[i], pt, 5, col, -1)

    def handle_mouse_events(self, event, x, y, flags, param):
        i = param
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.mode == 'draw_circle':
                self.frame_circles[i] = np.array([x, y])

    def loop(self):
        while (1):
            self.render_frames()
            for i in range(self.n_vids):
                cv2.imshow(self.vid_names[i], self.frames[i])
            k = cv2.waitKeyEx(300)
            if k == 27:
                self.cleanup()
                break

            elif k == ord('s'):
                if self.frame_circles_saved is None:
                    self.frame_circles_saved = np.array([self.frame_circles.copy()])
                else:
                    self.frame_circles_saved = np.stack([*self.frame_circles_saved, self.frame_circles])
                self.frame_circles[:, :] = np.nan
                with open(self.out_fpath, 'w') as f:
                    json.dump(dict(points=self.frame_circles_saved.tolist()), f)
                    print('Points saved.')

            elif k == ord(','):
                if self.frame_position > 0:
                    self.frame_position -= 1
                    self.load_frame_images()

            elif k == ord('.'):
                if (self.frame_position + 1) < self.frame_count:
                    self.frame_position += 1
                    self.load_frame_images()

            elif k == ord('c'):
                self.frame_circles[:, :] = np.nan

            elif k == ord('g'):
                self.frame_position = int(input(f'Enter a frame to go to between 0 and {self.frame_count}: '))
                if self.frame_position < 0:
                    self.frame_position = 0
                if self.frame_position > self.frame_count:
                    self.frame_position =self.frame_count -1
                self.load_frame_images()

            elif k == ord('h'):
                self.show_prev = not self.show_prev

    def cleanup(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
