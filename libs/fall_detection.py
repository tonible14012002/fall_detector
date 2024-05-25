from libs.fall_detector.tracker import Tracker, Detection
from libs.fall_detector.detection import detector
from libs.fall_detector.detection import utils
from libs.fall_detector.pose_predictor import (
    BasedPoseEstimator,
)
import cv2
import config as app_config
import numpy as np

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
    loss_track_count = 0

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
    max_age = 40  # frames

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

    def emit_falldown(self, result: ActionDetector.Result, image=None):
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
            callback(result, image)


class FallDetection:
    pose_estimator: BasedPoseEstimator = None
    action_detector: ActionDetector = None

    @classmethod
    def new(
        cls,
        action_detector: ActionDetector,
        pose_estimator: BasedPoseEstimator,
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
