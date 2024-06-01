import cv2
from utils.resize_preproc import resize384_bgr2rgb_preproc
from app import init_movenet_falldetector
from utils import cam as utils_cam
import config as app_config

if __name__ == "__main__":
    config = app_config.init_default_config(device="cpu")
    fall_detection = init_movenet_falldetector(config=config)
    config.detection_size = (384, 384)
    cam = utils_cam.CamLoader_Q(
        "./scripts/samples/fall-test.mp4",
        queue_size=10000,
        preprocess=resize384_bgr2rgb_preproc,
    )

    while not cam.grabbed():
        frame = cam.getitem()
        results = fall_detection.process(frame)
        for result in results:
            fall_detection.draw_bbox(
                frame,
                result.bbox,
                result.track_id,
                result.action,
                result.center,
                result.confidence,
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
