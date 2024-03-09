from lib.cam import CamLoader, CamLoader_Q
from fall_detector.detection import utils
from fall_detector import pose
import argparse
import cv2
import os
import time
import tensorflow as tf

DEFAULT_CAMERA_SOURCE = ".scripts/samples/fall-vid.mp4"


def preproc(image):
    """preprocess function for CameraLoader."""
    resizer = utils.ResizePadding(256, 256)
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
        default=384,
        help="Size of input in detection model in square must be divisible by 32 (int).",
    )

    args = par.parse_args()
    inp_dets = args.detection_input_size

    cam_source = args.camera
    return {
        "cam_source": cam_source,
        "detection_input_size": inp_dets,
    }


if __name__ == "__main__":

    args = get_parsed_args()
    cam_source = args["cam_source"]
    detection_size = args["detection_input_size"]

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

    pose_estimator = pose.PoseEstimator(
        sizeX=detection_size, sizeY=detection_size
    )
    pose_estimator.load_model()

    while cam.grabbed():
        frame = cam.getitem()
        image = frame.copy()

        pose_input = pose_estimator.cast_to_tf_tensor(image)
        keypoints_with_scores = pose_estimator.detect(
            pose_input, body_only=True
        )

        pose.loop_through_people(frame, keypoints_with_scores, 0.3)

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
