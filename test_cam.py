from utils.cam import CamLoader, CamLoader_Q
from libs.fall_detector.detection import utils
import argparse
import cv2
import os
import time
import numpy as np

DEFAULT_CAMERA_SOURCE = "./scripts/samples/fall-vid.mp4"


def preproc(image, size=(640, 640)):
    """preprocess function for CameraLoader."""
    resizer = utils.ResizePadding(*size[::-1])
    image = resizer(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_parsed_args():
    par = argparse.ArgumentParser(description="Human Fall Detection Demo")
    par.add_argument(
        "-C",
        "--camera",
        default=DEFAULT_CAMERA_SOURCE,
        help="use 0 for using webcam, or path to video file (default={})".format(
            DEFAULT_CAMERA_SOURCE
        ),
    )
    par.add_argument(
        "--detection_input_size",
        type=int,
        default=256,
        help="Size of input in detection model in square must be divisible by 32 (int).",
    )
    par.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run model on cpu or cuda.",
    )

    args = par.parse_args()
    inp_dets = args.detection_input_size

    cam_source = args.camera
    return {
        "cam_source": cam_source,  # 0 or path to video file
        "detection_input_size": inp_dets,  # 256
        "device": args.device,  # cuda or cpu
    }


if __name__ == "__main__":

    args = get_parsed_args()

    cam_source = args["cam_source"]
    detection_size = args["detection_input_size"]
    device = args["device"]

    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(
            cam_source, queue_size=10000, preprocess=preproc
        ).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(
            int(cam_source) if cam_source.isdigit() else cam_source,
            preprocess=preproc,
        ).start()

    fps_time = 0
    frame_count = 0

    # Initialize pose estimator
    while cam.grabbed():
        frame = cam.getitem()
        image = frame.copy()

        # RENDER STATE
        frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        frame = cv2.putText(
            frame,
            "%d, FPS: %f" % (frame_count, 1.0 / (time.time() - fps_time)),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        frame = frame[:, :, ::-1]

        fps_time = time.time()

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()
