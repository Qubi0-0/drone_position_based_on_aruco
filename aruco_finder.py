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


# idk if camera matrix is ok, it was created using https://learnopencv.com/camera-calibration-using-opencv/
camera_matrix = np.array([(FOCAL_LENGTH[0], 0, PRINCIPAL_POINT[0]/2),
                          (0, FOCAL_LENGTH[1], PRINCIPAL_POINT[1]/2),
                          (0, 0, 1)])

def getCorners(one_corner):
    one_corner = one_corner.reshape(4, 2)
    one_corner=one_corner.astype(int)
    return one_corner

def drawLines(frame, top_left, top_right, bottom_right, bottom_left):
    cv.line(frame, top_left, top_right, (0, 255, 0), 2)
    cv.line(frame, top_right, bottom_right, (0, 255, 0), 2)
    cv.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
    cv.line(frame, bottom_left, top_left, (0, 255, 0), 2)

def drawIds(frame, one_id, top_left, top_right, bottom_right, bottom_left):
    cv.putText(
        frame,
        str(one_id),
        (top_left[0], top_left[1] - 15),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )
def get3dPoints(one_id):
    pass



class ArucoFinder:

    def __init__(self):
        self.aruco_data = csv.reader(open('data/aruco_positions.csv'))
        self.video_frame = cv.VideoCapture("/home/dawid/Pobrane/GX010280.MP4")
        self.total_frame_count= int(self.video_frame.get(cv.CAP_PROP_FRAME_COUNT))
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        self.aruco_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
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
        (corners, ids, rejected) = finder.detector.detectMarkers(frame)
        if len(corners) > 0:
            ids = ids.flatten()
            for (one_corner, one_id) in zip(corners, ids):
                if (one_id) in range(1, 83):
                    (top_left, top_right, bottom_right, bottom_left) = getCorners(one_corner)
                    drawLines(frame, top_left, top_right, bottom_right, bottom_left)
                    drawIds(frame, one_id, top_left, top_right, bottom_right, bottom_left)


        cv.imshow('frame', frame)
        i += 1

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

finder.video_frame.release()
cv.destroyAllWindows()  

