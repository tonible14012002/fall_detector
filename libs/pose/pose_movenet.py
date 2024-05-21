from libs.fall_detector.pose_predictor import (
    BasePosePredictor,
    BasedPoseEstimator,
)
from libs.fall_detector.preprocessor import BasePreprocessor
import tensorflow_hub as tf_hub
import tensorflow as tf
import numpy as np


def cast_to_tf_tensor(image, size):
    return tf.cast(
        tf.image.resize_with_pad(
            tf.expand_dims(image, axis=0),
            target_height=size[1],
            target_width=size[0],
        ),
        dtype=tf.int32,
    )


def kpt2bbox_tf(kpt, ex=20, frame_size=(640, 480)):
    """Get bbox that holds on all of the keypoints (x, y)
    kpt: tensor of shape `(N, 3)` (N keypoint with (x, y, confidence) each),
    ex: (int) expand bounding box,
    frame_size: (tuple) (width, height) of the frame
    """
    kpt = np.multiply(kpt, [frame_size[1], frame_size[0], 1])
    bbox_min = tf.reduce_min(kpt[:, :2], axis=0) - ex
    bbox_max = tf.reduce_max(kpt[:, :2], axis=0) + ex
    bbox = tf.stack([bbox_min[1], bbox_min[0], bbox_max[1], bbox_max[0]])
    return bbox


class MovenetPosePredictor(BasePosePredictor):
    model = None
    image_size = (384, 384)

    def set_size(self, size):
        self.size = size

    def setup(self):
        self.model = tf_hub.load(
            "https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1"
        ).signatures["serving_default"]

    def preprocess(self, image):
        return cast_to_tf_tensor(image, self.image_size)

    def postprocess(self, result) -> BasePosePredictor.PoseResults:
        return super().postprocess(result)

    def predict(self, pose_input):
        results = self.model(pose_input)
        keypoints = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))
        keypoints = tf.concat(
            [keypoints[:, :1, :], keypoints[:, 5:, :]], axis=1
        )  # y, x, confidence
        poses = self.filter_poses(keypoints, 0.3)
        bboxes_xyxy = [
            kpt2bbox_tf(pose, 20, self.image_size).numpy() for pose in poses
        ]
        scores = [tf.reduce_mean(pose[:, 2]) for pose in poses]

        final = self.PoseResults(
            poses=poses[:, :, [1, 0, 2]],  # x, y, confidence
            bboxes_xyxy=bboxes_xyxy,
            scores=scores,
        )

        return final

    def filter_poses(self, tf_keypoints, threshold=0.2):
        scores = tf_keypoints[:, :, 2]
        mean_score_each = tf.reduce_mean(scores, axis=1)
        mask_above_threshold = mean_score_each > threshold

        tf_keypoints = tf.boolean_mask(
            tf_keypoints, mask_above_threshold, axis=0
        ).numpy()
        return tf_keypoints


class MovenetPoseEstimator(BasedPoseEstimator):
    @classmethod
    def new(
        cls, preprocessor: BasePreprocessor, predictor: MovenetPosePredictor
    ):
        n = cls()
        n.preprocessor = preprocessor
        n.predictor = predictor
        return n
