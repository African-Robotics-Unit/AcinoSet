import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help="the path of the video")
args = parser.parse_args()


if __name__ == "__main__":
    fpath = args.video
    cap = cv.VideoCapture(fpath)

    if not cap.isOpened():
        print("could not open :", fpath)
    else:
        num_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(f'{fpath}\t{num_frame}')
