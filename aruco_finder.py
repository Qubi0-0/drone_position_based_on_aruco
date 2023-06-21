import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Camera Params
FOCAL_LENGTH = [1920, 1920]
PRINCIPAL_POINT = [1920, 1080]
FOV = 92
RADIAL_DIST = [0, 0, 0]
TANG_DIST = [0, 0]

# idk if camera matrix is ok, it was created using https://learnopencv.com/camera-calibration-using-opencv/
camera_matrix = np.array([(FOCAL_LENGTH[0], 0, PRINCIPAL_POINT[0]),
                          (0, FOCAL_LENGTH[1], PRINCIPAL_POINT[1]),
                          (0, 0, 1)])

def plot_trajectory(trajectory, actual_trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory
    xs_traj = [pos[0] for pos in trajectory]
    ys_traj = [pos[1] for pos in trajectory]
    zs_traj = [pos[2] for pos in trajectory]
    ax.scatter(xs_traj, ys_traj, zs_traj, label='Estimated Trajectory')

    # Actual Trajectory
    xs_actual = [pos[0] for pos in actual_trajectory]
    ys_actual = [pos[1] for pos in actual_trajectory]
    zs_actual = [pos[2] for pos in actual_trajectory]
    ax.scatter(xs_actual, ys_actual, zs_actual, label='Actual Trajectory')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-5000, 5000])
    ax.set_ylim([-5000, 5000])
    ax.set_zlim([-3000, 3000])
    ax.legend()
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

def calculateError(camera_pos, cam_pos_csv_arr):
    cam_pos_csv_arr = [np.array(np.array([[cam_pos_csv_arr[0]], [cam_pos_csv_arr[1]], [cam_pos_csv_arr[2]]]))]
    error = np.linalg.norm(camera_pos - cam_pos_csv_arr)
    return error

class ArucoFinder:

    def __init__(self):
        self.aruco_data = pd.read_csv('data/aruco_positions.csv').set_index('Marker_ID')
        self.mocap_data = pd.read_csv('data/mocap_ref_data.csv')
        self.mark3_data = self.mocap_data.iloc[455:,8:11].values.tolist()
        self.camera_poses = [[x + 2, y + 5, z - 25] for x, y, z in self.mark3_data]
        self.video_frame = cv.VideoCapture("video/vid.MP4")
        self.total_frame_count = int(self.video_frame.get(cv.CAP_PROP_FRAME_COUNT))
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
        self.aruco_params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.hz25_camera_poses = self.camera_poses[::4]



if __name__ == '__main__':
    finder = ArucoFinder()
    i = 0
    cam_positions = []

    max_cam_distance = 800  # max camera distance between 2 consecutive positions
    last_cam_pos = None

    total_error = 0.0
    error_count = 0

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
                        points_2D.append((top_right) / 1.0)
                        points_3D.append(get3DPoints(one_id, finder.aruco_data))

                if len(points_2D) >= 4 and len(points_3D) == len(points_2D):
                    _, rvec, tvec = cv.solvePnP(np.array(points_3D), np.array(points_2D), camera_matrix, None)
                    R, _ = cv.Rodrigues(rvec)
                    camera_pos = np.dot(np.array([(1, 0, 0),
                                                (0, -1, 0),
                                                (0, 0, -1)]), np.dot(np.array(-R), tvec))

                    if last_cam_pos is not None:
                        cam_distance = np.linalg.norm(camera_pos - last_cam_pos)
                        if cam_distance < max_cam_distance:
                            cam_positions.append(camera_pos)
                            last_cam_pos = camera_pos
                            error = calculateError(camera_pos, finder.hz25_camera_poses[i])
                            total_error += error
                            error_count += 1

                    else:
                        cam_positions.append(camera_pos)
                        last_cam_pos = camera_pos
                        error = calculateError(camera_pos, finder.hz25_camera_poses[i])
                        total_error += error
                        error_count += 1

            frame = cv.resize(frame, [640, 480])
            cv.imshow('frame', frame)
            i += 1

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    finder.video_frame.release()
    cv.destroyAllWindows()
    plot_trajectory(cam_positions,finder.hz25_camera_poses)

    if error_count > 0:
        average_error = total_error / error_count
        print("Total Error: {:.2f}".format(total_error))
        print("Average Error: {:.2f}".format(average_error))