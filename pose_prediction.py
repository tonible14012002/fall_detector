import super_gradients.training
import super_gradients.training.models
from libs.fall_detector.detection import utils
from libs.fall_detector.config import BaseConfig
from libs.fall_detector.preprocessor import BasePreprocessor
from libs.fall_detector.pose_predictor import (
    BasePosePredictor,
    BasedPoseEstimator,
)
import super_gradients
from super_gradients.common.object_names import Models
import torch
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
    tensorrt_version = "8.5.2"
    pretrain_weights = "coco_pose"
    model_name = Models.YOLO_NAS_POSE_S


class YoloPosePredictor(BasePosePredictor):
    config = YoloConfig
    model = None
    device = "cpu"  # cuda

    def set_device(self, device):
        self.device = device

    def postprocess(self, result):
        prediction = result.prediction

        keypoints = prediction.poses
        prediction.poses = np.delete(keypoints, [1, 2, 3, 4], axis=1)

        return prediction

    def predict(self, image):
        return self.model.predict(image)

    def setup(self):
        self.model = super_gradients.training.models.get(
            self.config.model_name,
            pretrained_weights=self.config.pretrain_weights,
        )
        getattr(self.model, self.device)()


class ResizeProcessor(BasePreprocessor):
    def preprocess(self, image):
        return resize(image, YoloConfig.size)


class YoloBasedPoseEstimator(BasedPoseEstimator):
    preprocessor = ResizeProcessor()
    predictor = YoloPosePredictor()
    config = YoloConfig()

    def set_predictor_device(self, device):
        self.predictor.set_device(device)


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

    def postprocess(self, result):
        conf = result.keypoints.conf
        keypoints = result.keypoints.xy

        if keypoints.shape[1] == 17:
            keypoints = np.delete(keypoints, [1, 2, 3, 4], axis=1)

        if conf is not None:
            conf = np.delete(conf, [1, 2, 3, 4], axis=1)
            mean_tensor = torch.mean(conf, axis=1)
            confidence_expanded = conf.unsqueeze(-1).expand_as(
                keypoints[:, :, :1]
            )

            keypoints = torch.cat(
                (keypoints, confidence_expanded), dim=-1
            ).numpy()
        else:
            mean_tensor = np.array([])

        class Prediction:
            poses = keypoints
            bboxes_xyxy = result.boxes.xyxy.numpy()
            scores = mean_tensor

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
