import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Camera Params
FOCAL_LENGTH = [1920, 1920]
PRINCIPAL_POINT = [1920, 1080]
FOV = 92
RADIAL_DIST = [0,0,0]
TANG_DIST = [0,0]


# idk if camera matrix is ok, it was created using https://learnopencv.com/camera-calibration-using-opencv/
camera_matrix = np.array([(FOCAL_LENGTH[0], 0, PRINCIPAL_POINT[0]),
                          (0, FOCAL_LENGTH[1], PRINCIPAL_POINT[1]),
                          (0, 0, 1)])

def plot_trajectory(trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [pos[0] for pos in trajectory]
    ys = [pos[1] for pos in trajectory]
    zs = [pos[2] for pos in trajectory]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-5000, 5000])
    ax.set_ylim([-5000, 5000])
    ax.set_zlim([-3000, 3000])
    plt.show()

def getCorners(one_corner):
    one_corner = one_corner.reshape(4, 2)
    one_corner = one_corner.astype(int)
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

def get3DPoints(one_id, aruco_data):
    x = aruco_data.loc[one_id]['X']
    y = aruco_data.loc[one_id]['Y']
    z = aruco_data.loc[one_id]['Z']
    return np.array([x, y, z])
    
class ArucoFinder:

    def __init__(self):
        self.aruco_data = pd.read_csv('data/aruco_positions.csv').set_index('Marker_ID')
        self.mocap_data = pd.read_csv('data/mocap_ref_data.csv')
        self.video_frame = cv.VideoCapture("video/vid.MP4")
        self.total_frame_count= int(self.video_frame.get(cv.CAP_PROP_FRAME_COUNT))
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        self.aruco_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.camera_fixed_positions = []

    def check_camera_pose(self):
        pass


if __name__ == '__main__':
    finder = ArucoFinder()
    i = 0
    cam_positions = []

    while True:
        
        if finder.video_frame.isOpened():
            ret, frame = finder.video_frame.read()

        if i == finder.total_frame_count:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frame is not None:
            (corners, ids, rejected) = finder.detector.detectMarkers(frame)
            if len(corners) > 0:
                ids = ids.flatten()
                points_2D = []
                points_3D = []
                for (one_corner, one_id) in zip(corners, ids):
                    if (one_id) in range(1, 83):
                        (top_left, top_right, bottom_right, bottom_left) = getCorners(one_corner)
                        drawLines(frame, top_left, top_right, bottom_right, bottom_left)
                        drawIds(frame, one_id, top_left, top_right, bottom_right, bottom_left)
                        points_2D.append((top_right)/1.0)
                        points_3D.append(get3DPoints(one_id, finder.aruco_data))
                        
                if len(points_2D) >= 4 and len(points_3D) == len(points_2D):
                    _, rvec, tvec = cv.solvePnP(np.array(points_3D), np.array(points_2D), camera_matrix, None)
                    R, _ = cv.Rodrigues(rvec)
                    camera_pos = np.dot(np.array(
                            [(1, 0, 0),
                            (0, -1, 0),
                            (0, 0, -1)]), np.dot(np.array(-R), tvec))
                    cam_positions.append(camera_pos)

            frame = cv.resize(frame,[640,480])
            cv.imshow('frame', frame)
            print(finder.mocap_data)
            i += 1

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    finder.video_frame.release()
    cv.destroyAllWindows()  
    plot_trajectory(cam_positions)