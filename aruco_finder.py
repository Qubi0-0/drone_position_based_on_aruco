import csv
import numpy as np
import cv2 as cv
import matplotlib

class ArucoFinder:

    def __init__(self):
        self.aruco_data = csv.reader(open('data/aruco_positions.csv'))
        self.video_frame = cv.VideoCapture("/home/dawid/Pobrane/GX010280.MP4")
        self.totalframecount= int(self.video_frame.get(cv.CAP_PROP_FRAME_COUNT))
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        self.aruco_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

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
        (corners, ids, rejected) = finder.detector.detectMarkers(frame)
        if len(corners) > 0:
            ids = ids.flatten()
            for (one_corner, one_id) in zip(corners, ids):
                one_corner = one_corner.reshape(4, 2)
                one_corner=one_corner.astype(int)
                (top_left, top_right, bottom_left, bottom_right) = one_corner
                cv.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv.line(frame, bottom_left, top_left, (0, 255, 0), 2)

        cv.imshow('frame', frame)
        i = i + 1

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

finder.video_frame.release()
cv.destroyAllWindows()