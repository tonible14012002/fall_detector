from libs.pose.posetrt import (
    TrtPoseEstimator,
    TrtPosePredictor,
    TrtPreprocessor,
)


def new_trt_pose_estimator():
    return TrtPoseEstimator.new(
        preprocessor=TrtPreprocessor(),
        predictor=TrtPosePredictor(),
    )
