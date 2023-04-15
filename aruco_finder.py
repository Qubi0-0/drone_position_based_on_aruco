import csv
import numpy as np
import cv2 as cv
import matplotlib

class ArucoFinder:

    def __init__(self):
        self.aruco_data = csv.reader(open('data/aruco_positions.csv'))
        self.video_frame = cv.VideoCapture("video/vid.MP4")
        self.totalframecount= int(self.video_frame.get(cv.CAP_PROP_FRAME_COUNT))

i = 0

if __name__ == '__main__':
    finder = ArucoFinder()

while True:
    
    if finder.video_frame.isOpened():
        ret, frame = finder.video_frame.read()

    if i == finder.totalframecount:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if frame is not None:
        frame = cv.resize(frame,[640,480])
        cv.imshow('frame', frame)
        i = i + 1

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

finder.video_frame.release()
cv.destroyAllWindows()