import cv2
from libs.fall_detector.detection import detector
from libs.fall_detector.pose_predictor import BasePosePredictor
from libs.fall_detector.preprocessor import BasePreprocessor
from libs.fall_detector.tracker import Tracker
from libs.pose.pose_movenet import MovenetPoseEstimator, MovenetPosePredictor
from utils.cam import CamLoader_Q
from libs.utils import pose_movenet as pose_mn_utils
from utils.resize_preproc import resize384_bgr2rgb_preproc
from libs.utils.pose_movenet import new_movenet_pose_estimator
from libs import fall_detection
import config as app_config
import time


if __name__ == "__main__":
    device = "cpu"  # "cpu" or "cuda"
    cam = CamLoader_Q(
        "./scripts/samples/fall-test.mp4",
        queue_size=8000,
        preprocess=resize384_bgr2rgb_preproc,
    ).start()

    config = app_config.init_default_config(device=device)
    pose_estimator = pose_mn_utils.new_movenet_pose_estimator(
        "cpu", (384, 384)
    )
    MEAN_POSE_DELAY = 0
    TOTAL_FRAME = 0

    class MeasureMovenetPredictor(MovenetPosePredictor):
        def process(self, image):
            start = time.time()
            result = super().process(image)
            end = time.time()
            delay = end - start
            global MEAN_POSE_DELAY, TOTAL_FRAME
            MEAN_POSE_DELAY += delay
            TOTAL_FRAME += 1
            print("POSE DELAY - ", delay, "s")
            return result

    predictor = MeasureMovenetPredictor()
    predictor.set_size(config.detection_size)
    predictor.set_device(device=config.device)

    pose_estimator = MovenetPoseEstimator.new(
        preprocessor=BasePreprocessor(),
        predictor=predictor,
    )

    MEAN_ACTION_DELAY = 0
    TOTAL_FRAME_ACTION = 0

    class MeasureActionDetector(fall_detection.ActionDetector):
        def process(self, *args, **kwargs):
            begin = time.time()
            result = super().process(*args, *kwargs)
            end = time.time()
            global MEAN_ACTION_DELAY, TOTAL_FRAME_ACTION
            delay = end - begin
            MEAN_ACTION_DELAY += delay
            TOTAL_FRAME_ACTION += 1
            print("ACTION DELAY - ", delay, "s")
            return result

    fall_detector = fall_detection.FallDetection.new(
        action_detector=MeasureActionDetector.new(
            config=config,
            tracker=Tracker(max_age=30, max_iou_distance=0.7, n_init=3),
            action_model=detector.TSSTG(device=config.device),
        ),
        pose_estimator=pose_estimator,
    )

    fps_time = 0
    while cam.grabbed():
        frames = cam.getitem()
        results = fall_detector.process(frames)
        for result in results:
            fall_detection.draw_bbox(
                frames,
                result.bbox,
                result.track_id,
                result.action,
                result.center,
                result.confidence,
            )
            if (
                result.action == fall_detection.ACTIONS["Fall Down"]
                and result.confidence > 0.35
            ):
                print("Fall detected")
        frames = cv2.putText(
            frames,
            "FPS: %f" % (1.0 / (time.time() - fps_time)),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.imshow("frame", frames)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fps_time = time.time()

    cam.stop()
    cv2.destroyAllWindows()

    MEAN_ACTION_DELAY /= TOTAL_FRAME_ACTION
    print("MEAN ACTION DELAY - ", MEAN_ACTION_DELAY, "s")
    MEAN_POSE_DELAY /= TOTAL_FRAME
    print("MEAN POSE DELAY - ", MEAN_POSE_DELAY, "s")


# cam = CamLoader(0).start()
# while cam.grabbed():
#     frames = cam.getitem()

#     frames = cv2.putText(
#         frames,
#         "FPS: %f" % (1.0 / (time.time() - fps_time)),
#         (10, 20),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 255, 0),
#         2,
#     )
#     fps_time = time.time()
#     cv2.imshow("frame", frames)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cam.stop()
# cv2.destroyAllWindows()
