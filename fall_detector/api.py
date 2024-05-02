import cv2
import super_gradients.training
import super_gradients.training.models
from .detection import utils
from .config import BaseConfig
from .preprocessor import BasePreprocessor
from .pose_predictor import BasePosePredictor, BasedPoseEstimator
import super_gradients
from super_gradients.common.object_names import Models


class YoloConfig(BaseConfig):
    version = "8"
    size = (640, 640)
    tensorrt_version = "8.5.2"
    pretrain_weights = "coco_pose"
    model_name = Models.YOLO_NAS_POSE_S


class YoloPosePredictor(BasePosePredictor):
    config = YoloConfig
    model = None

    def predict(self, image):
        return self.model.predict(image)

    def setup(self):
        self.model = super_gradients.training.models.get(
            self.config.model_name,
            pretrained_weights=self.config.pretrain_weights,
        ).cpu()


class YoloBasedPoseEstimator(BasedPoseEstimator):
    preprocessor = BasePreprocessor()
    predictor = YoloPosePredictor()
    config = YoloConfig()
