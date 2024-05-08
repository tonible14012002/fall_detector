from libs.fall_detector.tracker import Tracker, Detection
from libs.fall_detector.detection import detector
from libs.fall_detector.detection import utils
from pose_prediction import (
    YoloBasedPoseEstimator,
    YoloUltralisticPoseEstimator,
    draw_poses,
)
import cv2
import config as app_config
import numpy as np
from utils.cam import CamLoader_Q

ACTIONS = {
    "Pending": "Pending",
    "Standing": "Standing",
    "Walking": "Walking",
    "Sitting": "Sitting",
    "Lying Down": "Lying Down",
    "Stand up": "Stand up",
    "Sit down": "Sit down",
    "Fall Down": "Fall Down",
}


class ActionDetector:
    tracker = None
    action_model: detector.TSSTG = None
    config: app_config.AppConfig = None

    class Result:
        track_id = None
        action = str
        confidence = float
        bbox = None
        center = None

        def __init__(
            self,
            track_id,
            action,
            confidence,
            bbox,
            center,
        ):
            self.track_id = track_id
            self.action = action
            self.confidence = confidence
            self.bbox = bbox
            self.center = center

    @classmethod
    def new(
        cls,
        config: app_config.AppConfig,
        tracker: Tracker,
        action_model: detector.TSSTG,
    ):
        a = cls()
        a.config = (
            config
            if config is not None
            else Tracker(max_age=30, max_iou_distance=0.7, n_init=3)
        )
        a.action_model = action_model
        a.tracker = tracker
        return a

    def _update_tracker(self, poses, bboxs, scores):
        self.tracker.predict()

        detections = [
            Detection(
                bbox,
                ps,
                score,
            )
            for ps, bbox, score in zip(poses, bboxs, scores)
        ]

        self.tracker.update(detections)

    def process(self, poses, bboxs, scores) -> "list[Result]":
        """
        poses: nparray - (n, 17, 3)
        bboxs: nparray - (n, 4)
        scores: nparray - (n,)
        Call before detection
        """
        self._update_tracker(poses, bboxs, scores)

        results = []
        for _, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action_name = ACTIONS["Pending"]
            confidence = 0
            if len(track.keypoints_list) == self.tracker.max_age:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = self.action_model.predict(
                    pts, self.config.detection_size[::-1]
                )
                confidence = out[0].max()
                action_name = self.action_model.class_names[out[0].argmax()]

            if track.time_since_update == 0:
                results.append(
                    self.Result(
                        track_id=track_id,
                        action=ACTIONS[action_name],
                        confidence=confidence,
                        bbox=bbox,
                        center=center,
                    )
                )
                continue
        return results


class FallDownEventEmitor:
    _callbacks = []
    _to_remove_event: "list[ToRemoveEvent]" = []
    max_age = 10  # frames

    class ToRemoveEvent:
        id: str = None
        age: int = None

        def __init__(self, id, age):
            self.id = id
            self.age = age

    def get_event_if_exist(self, track_id):
        for event in self._to_remove_event:
            if event.id == track_id:
                return event
        return None

    def on_falldown(self, func):
        """
        Decorator for regiser falling event
        Event fire only once on the same human pose (same track ID)
        """
        self._callbacks.append(func)
        return func

    def emit_falldown(self, result: ActionDetector.Result):
        """
        Embed this function in to event loop
        """
        event = self.get_event_if_exist(result.track_id)
        if event is not None:
            event.age += 1
            if event.age > self.max_age:
                self._to_remove_event.remove(event)
            return

        self._to_remove_event.append(
            self.ToRemoveEvent(id=result.track_id, age=0)
        )

        for callback in self._callbacks:
            callback(result)


class FallDetection:
    pose_estimator: YoloBasedPoseEstimator = None
    action_detector: ActionDetector = None

    @classmethod
    def new(
        cls,
        action_detector: ActionDetector,
        pose_estimator: YoloBasedPoseEstimator,
    ):
        f = cls()
        f.action_detector = action_detector
        f.pose_estimator = pose_estimator
        return f

    def process(self, image):
        prediction = self.pose_estimator.get_prediction(image)
        results = self.action_detector.process(
            prediction.poses, prediction.bboxes_xyxy, prediction.scores
        )

        return results


def init_detection(device, config: app_config.AppConfig) -> FallDetection:
    pose_estimator = YoloBasedPoseEstimator()
    pose_estimator.set_predictor_device(device=device)

    fall_detection = FallDetection.new(
        action_detector=ActionDetector.new(
            config=config,
            tracker=Tracker(max_age=30, max_iou_distance=0.7, n_init=3),
            action_model=detector.TSSTG(
                device=device,
            ),
        ),
        pose_estimator=pose_estimator,
    )

    return fall_detection


def draw_bbox(frame, bbox, track_id, action, center, confidence):
    action_str = "{}: {:.2f}%".format(action, confidence * 100)
    clr = (0, 255, 0)
    if action == "Fall Down":
        clr = (255, 0, 0)
    if action == "Lying Down":
        clr = (255, 200, 0)

    frame = cv2.rectangle(
        frame,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (0, 255, 0),
        1,
    )

    frame = cv2.putText(
        frame,
        str(track_id),
        (int(center[0]), int(center[1])),
        cv2.FONT_HERSHEY_COMPLEX,
        0.4,
        (255, 0, 0),
        2,
    )
    frame = cv2.putText(
        frame,
        action_str,
        (int(bbox[0]) + 5, int(bbox[1]) + 15),
        cv2.FONT_HERSHEY_COMPLEX,
        0.4,
        clr,
        1,
    )


class FallDetectionUltralistic(FallDetection):
    pose_estimator: YoloUltralisticPoseEstimator = None
    action_detector: ActionDetector = None

    @classmethod
    def new(
        cls,
        action_detector: ActionDetector,
        pose_estimator: YoloBasedPoseEstimator,
    ):
        f = cls()
        f.action_detector = action_detector
        f.pose_estimator = pose_estimator
        return f

    def process(self, image):
        prediction = self.pose_estimator.get_prediction(image)
        for pose in prediction.poses:
            try:
                draw_poses(image, pose)
            except:  # noqa
                pass
        results = self.action_detector.process(
            prediction.poses, prediction.bboxes_xyxy, prediction.scores
        )
        return results


def init_ultralistic_detection(
    device, config: app_config.AppConfig
) -> FallDetection:
    pose_estimator = YoloUltralisticPoseEstimator()
    pose_estimator.set_predictor_device(device=device)

    fall_detection = FallDetectionUltralistic.new(
        action_detector=ActionDetector.new(
            config=config,
            tracker=Tracker(max_age=30, max_iou_distance=0.7, n_init=3),
            action_model=detector.TSSTG(
                device=device,
            ),
        ),
        pose_estimator=pose_estimator,
    )
    return fall_detection


if __name__ == "__main__":

    def preproc(image):
        """preprocess function for CameraLoader."""
        resizer = utils.ResizePadding(640, 640)
        image = resizer(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    test_device = "cpu"
    config = app_config.AppConfig()
    # fall_detection = init_detection(test_device, config)
    fall_detection = init_ultralistic_detection(test_device, config)
    cam = CamLoader_Q(
        "./test-samples/sample3.mp4", queue_size=10000, preprocess=preproc
    ).start()

    import time

    fps_time = time.time()

    fall_emitor = FallDownEventEmitor()

    @fall_emitor.on_falldown
    def on_falldown(result: ActionDetector.Result):
        print("FALL DOWN", result.track_id)

    while cam.grabbed():
        frame = cam.getitem()
        frame = cv2.resize(frame, config.detection_size)
        results = fall_detection.process(frame)

        for result in results:
            if result.action == "Fall Down":
                fall_emitor.emit_falldown(result)
            draw_bbox(
                frame,
                result.bbox,
                result.track_id,
                result.action,
                result.center,
                result.confidence,
            )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.putText(
            frame,
            "FPS: %.2f" % (1.0 / (time.time() - fps_time)),
            (10, 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        fps_time = time.time()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.stop()
    cv2.destroyAllWindows()
