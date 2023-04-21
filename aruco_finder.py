import csv
import numpy as np
import cv2 as cv
import matplotlib

# Camera Params
FOCAL_LENGTH = [1920, 1920]
PRINCIPAL_POINT = [1920, 1080]
FOV = 92
RADIAL_DIST = [0,0,0]
TANG_DIST = [0,0]

class ArucoFinder:

    def __init__(self):
        self.aruco_data = csv.reader(open('data/aruco_positions.csv'))
        self.video_frame = cv.VideoCapture("video/vid.MP4")
        self.total_frame_count= int(self.video_frame.get(cv.CAP_PROP_FRAME_COUNT))



if __name__ == '__main__':
    finder = ArucoFinder()
i = 0
while True:
    
    if finder.video_frame.isOpened():
        ret, frame = finder.video_frame.read()

    if i == finder.total_frame_count:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if frame is not None:
        frame = cv.resize(frame,[640,480])
        cv.imshow('frame', frame)
        i += 1

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

finder.video_frame.release()
cv.destroyAllWindows()