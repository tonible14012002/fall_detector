import entities as app_entities
import streamer as app_streamer
import config as app_config
from utils import cam as utils_cam
from utils.resize_preproc import detection_preproc
from typing_extensions import Union
from video_track import VideoTransformTrack
import fall_detection
import cv2
import time


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
                        if (
                            result.action
                            == fall_detection.ACTIONS["Fall Down"]
                        ):
                            self.fall_emitor.emit_falldown(result)
                        fall_detection.draw_bbox(
                            frame,
                            result.bbox,
                            result.track_id,
                            result.action,
                            result.center,
                            result.confidence,
                        )
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
        def fire_noty(result: fall_detection.ActionDetector.Result):
            # Implement Firebase Notification
            print("FALL DOWN", result.track_id)

    def start(self):
        self.streamer.start()


if __name__ == "__main__":

    device = "cpu"
    config = app_config.init_default_config(device=device)

    detection = fall_detection.init_ultralistic_detection(
        device=device, config=config
    )
    entities = app_entities.Entities.new(
        fall_detector=detection, config=config
    )

    streamer = app_streamer.init_streamer(entities=entities)
    app = App.new(
        entities=entities,
        streamer=streamer,
        detection=detection,
        cam=utils_cam.CamLoader_Q(
            config.video_src,
            queue_size=10000,
            preprocess=detection_preproc,
        ),
        # cam=utils_cam.CamLoader(0, preprocess=detection_preproc),
    )

    app.set_allow_detection(True)

    app.streamer.set_stream_id("1")
    app.streamer.set_username("1")
    app.init()
    app.start()
