import entities as app_entities
from libs.fall_detector.detection import detector
from libs.fall_detector.tracker import Tracker
import libs.streamer as app_streamer
import config as app_config
from utils import cam as utils_cam
from utils.resize_preproc import (
    resize_bgr2rgb_preproc,
    resize384_bgr2rgb_preproc,
    resize224_bgr2rgb_preproc,
)
from typing import Union
from video_track import VideoTransformTrack
import libs.fall_detection as fall_detection
import cv2
import time

from pynput import keyboard
import threading


# def on_press(key):
#     try:
#         if key.char == "a":
#             print('Key "a" was pressed')
#     except AttributeError:
#         pass


# # Setting up the listener


# def start_listener():
#     with keyboard.Listener(on_press=on_press) as listener:
#         listener.join()


# threading.Thread(target=start_listener).start()


class FpsCounter:
    fpstime = time.time()

    def get_fps(self):
        return 1.0 / (time.time() - self.fpstime)

    def update(self):
        self.fpstime = time.time()


class App:
    entities: app_entities.Entities = None
    streamer: app_streamer.Streamer = None
    detection: fall_detection.FallDetection = None
    cam: Union[utils_cam.CamLoader_Q, utils_cam.CamLoader] = None
    fall_emitor = fall_detection.FallDownEventEmitor()
    fps_counter = FpsCounter()

    allow_detection = True

    def set_allow_detection(self, allow_detection: bool):
        self.allow_detection = allow_detection

    def _set_streamer_info(self):
        # Do api call to get device info
        self.streamer.set_stream_id(self.entities.config.device_id)
        self.streamer.set_username(self.entities.config.device_id)

    def _init_stream_track(self):
        class DetectVideoTrack(VideoTransformTrack):
            def process_cv_frame(track_self, frame):

                if self.allow_detection:
                    results = self.detection.process(frame)
                    for result in results:

                        fall_detection.draw_bbox(
                            frame,
                            result.bbox,
                            result.track_id,
                            result.action,
                            result.center,
                            result.confidence,
                        )
                        if (
                            result.action
                            == fall_detection.ACTIONS["Fall Down"]
                        ):
                            self.fall_emitor.emit_falldown(result, frame)
                frame = cv2.putText(
                    frame,
                    "FPS: %.2f" % self.fps_counter.get_fps(),
                    (10, 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.fps_counter.update()
                return super(DetectVideoTrack, track_self).process_cv_frame(
                    frame
                )

        return DetectVideoTrack(cam=self.cam)

    @classmethod
    def new(
        cls,
        entities: app_entities.Entities,
        streamer: app_streamer.Streamer,
        detection: fall_detection.FallDetection,
        cam: Union[utils_cam.CamLoader_Q, utils_cam.CamLoader],
    ):
        app = cls()
        app.entities = entities
        app.detection = detection
        app.streamer = streamer
        app.cam = cam
        return app

    def init(self):
        self.cam.start()
        stream_track = self._init_stream_track()
        self._register_events()
        self._set_streamer_info()
        self.streamer.set_stream_track(stream_track=stream_track)

    def _register_events(self):
        # Register on Falling Down callback
        @self.fall_emitor.on_falldown
        def fire_noty(
            result: fall_detection.ActionDetector.Result, image=None
        ):
            print("FALL DOWN", result.track_id)
            import threading

            threading.Thread(
                target=self._handle_falldown, args=(image,)
            ).start()
            # Implement Firebase Notification

    def _handle_falldown(self, image):
        import requests
        import base64

        img_bytes = cv2.imencode(".jpg", image)[1].tobytes()
        jpg_as_text = base64.b64encode(img_bytes)

        print(jpg_as_text)
        url = "http://192.168.1.197:4000/fall-notify"

        resp = requests.post(
            url=url,
            json={
                "image": jpg_as_text.decode("utf-8"),
                "deviceSerial": "AAA-BBB-CCC-EEE",
            },
        )
        print("received", resp.text)

    def start(self):
        self.streamer.start()


def init_trtpose_falldetector(
    config: app_config.AppConfig,
) -> fall_detection.FallDetection:
    from libs.utils.posetrt import new_trt_pose_estimator

    pose_estimator = new_trt_pose_estimator()
    return fall_detection.FallDetection.new(
        action_detector=fall_detection.ActionDetector.new(
            config=config,
            tracker=Tracker(max_age=30, max_iou_distance=0.7, n_init=3),
            action_model=detector.TSSTG(device=config.device),
        ),
        pose_estimator=pose_estimator,
    )


def init_movenet_falldetector(
    config: app_config.AppConfig,
):
    pass
    from libs.utils.pose_movenet import new_movenet_pose_estimator

    pose_estimator = new_movenet_pose_estimator(
        device=config.device, size=config.detection_size
    )
    return fall_detection.FallDetection.new(
        action_detector=fall_detection.ActionDetector.new(
            config=config,
            tracker=Tracker(max_age=30, max_iou_distance=0.7, n_init=3),
            action_model=detector.TSSTG(device=config.device),
        ),
        pose_estimator=pose_estimator,
    )


def init_yolonas_falldetector(
    config: app_config.AppConfig,
) -> fall_detection.FallDetection:
    from libs.utils.yoloposenas import new_yolovnas_pose_estimator

    pose_estimator = new_yolovnas_pose_estimator(device=config.device)
    return fall_detection.FallDetection.new(
        action_detector=fall_detection.ActionDetector.new(
            config=config,
            tracker=Tracker(max_age=30, max_iou_distance=0.7, n_init=3),
            action_model=detector.TSSTG(device=config.device),
        ),
        pose_estimator=pose_estimator,
    )


def init_yolov8_falldetector(
    config: app_config.AppConfig,
) -> fall_detection.FallDetection:
    from libs.utils.yolov8 import new_yolov8_pose_estimator

    pose_estimator = new_yolov8_pose_estimator(device=config.device)
    return fall_detection.FallDetection.new(
        action_detector=fall_detection.ActionDetector.new(
            config=config,
            tracker=Tracker(max_age=30, max_iou_distance=0.7, n_init=3),
            action_model=detector.TSSTG(
                device=config.device,
            ),
        ),
        pose_estimator=pose_estimator,
    )


if __name__ == "__main__":
    device = "cpu"
    backer = "movenet"  # "yolov8" | "yolonas"| "trt_pose" | "movenet"

    if backer == "yolov8":
        config = app_config.init_default_config(device=device)
        config.detection_size = (640, 640)
        detection = init_yolov8_falldetector(config=config)
        cam = utils_cam.CamLoader_Q(
            "./scripts/samples/fall-vid.mp4",
            queue_size=10000,
            preprocess=resize_bgr2rgb_preproc,
        )
    elif backer == "yolonas":
        config = app_config.init_default_config(device=device)
        config.detection_size = (640, 640)
        detection = init_yolonas_falldetector(config=config)
        cam = utils_cam.CamLoader_Q(
            "./scripts/samples/fall-vid.mp4",
            queue_size=10000,
            preprocess=resize_bgr2rgb_preproc,
        )
    elif backer == "movenet":
        # Default to yolov8
        config = app_config.init_default_config(device=device)
        config.detection_size = (384, 384)
        detection = init_movenet_falldetector(config=config)
        cam = utils_cam.CamLoader_Q(
            "./scripts/samples/fall-test.mp4",
            queue_size=10000,
            preprocess=resize384_bgr2rgb_preproc,
        )
    else:
        config = app_config.init_default_config(device=device)
        config.detection_size = (224, 224)
        detection = init_trtpose_falldetector(config=config)
        cam = utils_cam.CamLoader_Q(
            "./scripts/samples/fall-test.mp4",
            queue_size=10000,
            preprocess=resize224_bgr2rgb_preproc,  # update 224 resizer
        )

    # cam = utils_cam.CamLoader(
    #     0,
    #     # queue_size=10000,
    #     preprocess=resize384_bgr2rgb_preproc,  # update 224 resizer
    # )

    entities = app_entities.Entities.new(
        fall_detector=detection, config=config
    )

    streamer = app_streamer.init_streamer(entities=entities)
    app = App.new(
        entities=entities, streamer=streamer, detection=detection, cam=cam
    )

    app.set_allow_detection(True)
    app.streamer.set_stream_id("1")
    app.streamer.set_username("1")
    app.init()
    app.start()
