import super_gradients.training
import super_gradients.training.models
from libs.fall_detector.preprocessor import BasePreprocessor
from libs.fall_detector.pose_predictor import (
    BasePosePredictor,
    BasedPoseEstimator,
)
import super_gradients
from super_gradients.common.object_names import Models
import numpy as np
import cv2


PRETRAIN_WEIGHTS = "coco_pose"


class YoloNasPosePredictor(BasePosePredictor):
    model = None

    def postprocess(self, result):
        prediction = result.prediction

        keypoints = prediction.poses
        prediction.poses = np.delete(keypoints, [1, 2, 3, 4], axis=1)

        return prediction

    def predict(self, image):
        return self.model.predict(image)

    def setup(self):
        self.model = super_gradients.training.models.get(
            Models.YOLO_NAS_POSE_S,
            pretrained_weights=PRETRAIN_WEIGHTS,
        )
        getattr(self.model, self.device)()


class YoloNasPoseEstimator(BasedPoseEstimator):
    preprocessor: BasePreprocessor = None
    predictor: YoloNasPosePredictor = None

    def set_predictor_device(self, device):
        self.predictor.set_device(device)

    @classmethod
    def new(
        cls,
        preprocessor: BasePreprocessor,
        predictor: YoloNasPosePredictor,
    ):
        n = cls()
        n.preprocessor = preprocessor
        n.predictor = predictor
        return n
