import cv2
import numpy as np
from libs.fall_detector.preprocessor import BasePreprocessor
from libs.pose.poseyolov8 import YoloV8PosePredictor
from libs.pose.poseyolov8 import YoloV8PoseEstimator


DETECTION_SIZE = (640, 640)
EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (3, 5): "m",
    (2, 2): "c",
    (4, 6): "c",
    (1, 2): "y",
    (1, 7): "m",
    (2, 8): "c",
    (7, 8): "y",
    (7, 9): "m",
    (9, 11): "m",
    (8, 10): "c",
    (10, 12): "c",
}


def draw_poses(frame, pose, detection_size=DETECTION_SIZE):
    _draw_connections(frame, pose, EDGES, detection_size)
    _draw_keypoints(frame, pose, detection_size)


def _draw_keypoints(frame, keypoints, detection_size=DETECTION_SIZE):
    """
    detectionSize: x, y
    """
    x, y = detection_size
    imgY, imgX, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [imgY / y, imgX / x, 1]))

    for kp in shaped:
        kx, ky, _kp_conf = kp
        cv2.circle(frame, (int(kx), int(ky)), 1, (0, 255, 0), -1)


def _draw_connections(frame, keypoints, edges, detection_size=DETECTION_SIZE):
    x, y = detection_size
    imgY, imgX, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [imgY / y, imgX / x, 1]))

    for edge, _ in edges.items():
        p1, p2 = edge
        x1, y1, _ = shaped[p1]
        x2, y2, _ = shaped[p2]

        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


def new_yolov8_pose_estimator(device="cpu"):
    pose_estimator = YoloV8PoseEstimator.new(
        preprocessor=BasePreprocessor(),
        predictor=YoloV8PosePredictor(),
    )
    pose_estimator.set_predictor_device(device=device)

    return pose_estimator
