import imageio
import os
import analyse as an
import re
import cv2
import numpy as np

def extract_frames(start, n, filepath, folder="C://Users//user-pc//Desktop//frames"):
    """
    Extracts a range of frames from a given video filepath (absolute)
    """
    myFrameNumber = start
    cap = cv2.VideoCapture(filepath)

    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(totalFrames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    # check for valid frame number
    if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
    
    i = 0

    while i<n:
        ret, frame = cap.read()
        image_file = os.path.join(folder, "img"+str(i).zfill(5)+".png")
        print(image_file)
        cv2.imwrite(image_file, frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        i+=1
    
    print("Saved!")

    cv2.destroyAllWindows()

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)

def make_anim(path):
    """
    Creates an animation of a set of images saved in a given folder path.

    Parameters
    ---
    path: string
        The absolute filepath to the folder containing the frames of your animation

    Returns
    ---
    None
    """

    image_folder = path
    video_name = 'C://Users//user-pc//Desktop//video_lidar.avi'

    images = natural_sort([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 5, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print("saved!")

def combine_ims(folder1, folder2, output_folder = "C://Users//user-pc//Desktop//combined"):
    """
    Combines a pair of opencv-read images vertically
    """
    
    images1 = natural_sort([img for img in os.listdir(folder1) if img.endswith(".jpg")])
    images2 = natural_sort([img for img in os.listdir(folder2) if img.endswith(".jpg")])

    for i in range(len(images1)):
        img1 = cv2.imread(os.path.join(folder1,images1[i]))
        img2 = cv2.imread(os.path.join(folder2,images2[i]))
        vis = np.concatenate((img1, img2), axis=0)
        savepath = os.path.join(output_folder, "img"+str(i).zfill(2)+".jpg")
        cv2.imwrite(savepath, vis)
        print("Combined "+savepath)

if __name__ == "__main__":
    path = "C://Users//user-pc//Desktop//gooddark"
    make_anim(path)
    #vidfile = "C://Users//user-pc//Desktop//vidraw.avi"
    #extract_frames(12600, 800, vidfile)
    #folder1 = "C://Users//user-pc//Desktop//im_axes"
    #folder2 = "C://Users//user-pc//Desktop//plotted"
    #combine_ims(folder1,folder2)