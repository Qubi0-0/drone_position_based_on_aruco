import csv
import numpy as np
import cv2 as cv
import matplotlib


class ArucoFinder:

    def __init__(self):
        self.aruco_data = csv.reader(open('data/aruco_positions.csv'))
        self.video_frame = cv.VideoCapture("video/vid.MP4")




if __name__ == '__main__':
    finder = ArucoFinder()
while True:
    
    if finder.video_frame.isOpened():
        ret, frame = finder.video_frame.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.resize(frame,[640,480])
    cv.imshow('frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

finder.video_frame.release()
cv.destroyAllWindows()

    


