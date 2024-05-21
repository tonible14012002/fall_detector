from libs.pose.pose_movenet import MovenetPoseEstimator, MovenetPosePredictor
from libs.fall_detector.preprocessor import BasePreprocessor


def new_movenet_pose_estimator(device="cpu", size=(192, 192)):
    predictor = MovenetPosePredictor()
    predictor.set_size(size)
    predictor.set_device(device)
    pose_estimator = MovenetPoseEstimator.new(
        preprocessor=BasePreprocessor(),
        predictor=predictor,
    )
    # pose_estimator.set_predictor_device(device=device)

    return pose_estimator
