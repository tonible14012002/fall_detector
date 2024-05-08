from libs.fall_detector.detection import utils
from libs.fall_detector.config import BaseConfig
from libs.fall_detector.preprocessor import BasePreprocessor
from libs.fall_detector.pose_predictor import (
    BasePosePredictor,
    BasedPoseEstimator,
)
import numpy as np
import cv2


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


def resize(image, size):
    """
    image: cv2 image
    size: tuple (width, height)
    """
    resizer = utils.ResizePadding(*size)
    image = resizer(image)
    return image


class YoloConfig(BaseConfig):
    version = "8"
    size = (640, 640)
    pretrain_weights = "coco_pose"


class ResizeProcessor(BasePreprocessor):
    def preprocess(self, image):
        return resize(image, YoloConfig.size)


def draw_poses(frame, pose, detection_size=(640, 640)):
    _draw_connections(frame, pose, EDGES, detection_size)
    _draw_keypoints(frame, pose, detection_size)


def _draw_keypoints(frame, keypoints, detection_size=(640, 640)):
    """
    detectionSize: x, y
    """
    x, y = detection_size
    imgY, imgX, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [imgY / y, imgX / x, 1]))

    for kp in shaped:
        kx, ky, _kp_conf = kp
        cv2.circle(frame, (int(kx), int(ky)), 1, (0, 255, 0), -1)


def _draw_connections(frame, keypoints, edges, detection_size=(640, 640)):
    x, y = detection_size
    imgY, imgX, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [imgY / y, imgX / x, 1]))

    for edge, _ in edges.items():
        p1, p2 = edge
        x1, y1, _ = shaped[p1]
        x2, y2, _ = shaped[p2]

        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


class YoloUltralisticPosePredictor(BasePosePredictor):
    model = None
    device = "cpu"

    def setup(self):
        from ultralytics import YOLO

        self.model = getattr(YOLO("yolov8n-pose"), self.device)()

    def preprocess(self, image):
        return super().preprocess(image)

    def predict(self, image):
        prediction = self.model.predict(image)
        return prediction[0]

    def set_device(self, device):
        self.device = device

    def postprocess(self, result):  # noqa
        confs_list = (
            result.keypoints.conf if result.keypoints.conf is not None else []
        )
        keypoints_list = result.keypoints.xy
        result_keypoints = []  # List of (13 keypoints)
        results_conf = (
            []
        )  # list of confidences (each confidence correspond to 13 keypoint)
        for kpts, confs in zip(keypoints_list, confs_list):
            rekpts = [[0, 0]] * 17
            # is Good keypoint (contain important parts of body)
            is_good_enough_pose = (
                (
                    kpts[0][0]
                    or kpts[1][0]
                    or kpts[2][0]
                    or kpts[3][0]
                    or kpts[4][0]
                )
                and (kpts[5][0] or kpts[6][0])  # Head
                and (kpts[11][0] or kpts[12][0])  # Hip
                and (kpts[13][0] or kpts[14][0])  # Knee
            )

            if not is_good_enough_pose:
                continue

            for kpt, c, i in zip(kpts, confs, range(len(kpts))):
                if kpt[0]:
                    rekpts[i] = np.array([*kpt, c])

            # Determine Shoulder First
            if not (
                rekpts[5][0] and rekpts[6][0]
            ):  # if only one of two is predicted
                detect_kpt = rekpts[5] if rekpts[5][0] else rekpts[6]
                rekpts[5] = rekpts[6] = detect_kpt
            # 5, 6 determined
            if (
                # if head parts of body is predicted
                rekpts[0][0]  # nose
                or rekpts[1][0]  # right Eye
                or rekpts[2][0]  # left Eye
                or rekpts[3][0]  # right Ear
                or rekpts[4][0]  # left Ear
            ):
                detected_headkpts = [kpt for kpt in rekpts[0:5] if kpt[0] > 0]
                stacked_headkpts = np.stack(detected_headkpts)
                head_kpt = np.mean(stacked_headkpts, axis=0)
                rekpts[0] = head_kpt
            else:
                rekpts[0] = rekpts[5]
            # Determined 0,1,2,3,4,5,6
            # Determine Left Shoulder
            if rekpts[8][0] and not rekpts[10][0]:
                rekpts[10] = rekpts[8]
            elif not rekpts[8][0] and rekpts[10][0]:
                stacked = np.stack([rekpts[6], rekpts[10]])
                rekpts[8] = np.mean(stacked, axis=0)
            elif not rekpts[8][0] and not rekpts[10][0]:
                rekpts[8] = rekpts[10] = rekpts[6]

            # Determine Right Shoulder
            if rekpts[7][0] and not rekpts[9][0]:
                rekpts[9] = rekpts[7]
            elif not rekpts[7][0] and rekpts[9][0]:
                stacked = np.stack([rekpts[5], rekpts[9]])
                rekpts[7] = np.mean(stacked, axis=0)
            elif not rekpts[7][0] and not rekpts[9][0]:
                rekpts[7] = rekpts[9] = rekpts[5]

            if not (rekpts[11][0] and rekpts[12][0]):
                rekpts[11] = rekpts[12] = (
                    rekpts[11] if rekpts[11][0] else rekpts[12]
                )

            if not (rekpts[13][0] and rekpts[14][0]):
                rekpts[13] = rekpts[14] = (
                    rekpts[13] if rekpts[13][0] else rekpts[14]
                )

            if not rekpts[15][0]:
                rekpts[15] = rekpts[13]
            if not rekpts[16][0]:
                rekpts[16] = rekpts[14]

            pose = np.array([rekpts[0], *rekpts[5:17]])
            result_keypoints.append(pose)  # remove 1,2,3,4
            c = np.mean(pose[:2])
            results_conf.append(c)

        result_keypoints = np.array(result_keypoints)
        results_conf = np.array(results_conf)

        print(result_keypoints.shape)

        class Prediction:
            poses = result_keypoints
            bboxes_xyxy = result.boxes.xyxy.numpy()
            scores = results_conf

        return Prediction()


class YoloUltralisticPoseEstimator(BasedPoseEstimator):
    preprocessor = ResizeProcessor()
    predictor = YoloUltralisticPosePredictor()
    config = YoloConfig()

    @classmethod
    def new(cls, preprocessor, predictor, config):
        n = cls()
        n.config = config
        n.preprocessor = preprocessor
        n.predictor = predictor

    def set_predictor_device(self, device):
        self.predictor.set_device(device)
