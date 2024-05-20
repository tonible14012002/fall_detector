from lib_cam.cam import CamLoader, CamLoader_Q
from fall_detector.detection import utils
from fall_detector import pose
from fall_detector.tracker import Tracker, Detection, utils as tracker_utils
from fall_detector.detection import detector
import argparse
import cv2
import os
import time
import tensorflow as tf
import numpy as np

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

    # Initialize pose estimator
    pose_estimator = pose.PoseEstimator(
        sizeX=detection_size, sizeY=detection_size
    )
    pose_estimator.load_model()

    # Initialize tracker
    max_age = 30
    tracker = Tracker(max_age=30, max_iou_distance=0.7, n_init=3)

    # Initialize action detector
    action_detector = detector.TSSTG(device="cpu")

    while cam.grabbed():
        frame = cam.getitem()
        image = frame.copy()

        # DETECTION
        pose_input = pose_estimator.cast_to_tf_tensor(image)
        pose_estimator.detect(
            pose_input, body_only=True
        )  # cut eyes, ears -> 13 keypoinst
        pose_estimator.filter_poses(0.3)  # filter low confidence poses
        poses = pose_estimator.get_poses()  # y, x, confidence
        total_poses = len(poses)

        # TRACK POSES
        tracker.predict()

        detections = [
            Detection(
                tracker_utils.kpt2bbox_tf(
                    ps, 20, frame_size=(frame.shape[1], frame.shape[0])
                ).numpy(),
                ps,
                tf.reduce_mean(ps[:, 2]),
            )
            for ps in poses
        ]

        tracker.update(detections)

        # ACTION DETECTOR
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)
            action = "pending..."
            clr = (0, 255, 0)

            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_detector.predict(pts, image.shape[:2])
                action_name = action_detector.class_names[out[0].argmax()]
                action = "{}: {:.2f}%".format(action_name, out[0].max() * 100)
                if action_name == "Fall Down":
                    clr = (255, 0, 0)
                elif action_name == "Lying Down":
                    clr = (255, 200, 0)

            if track.time_since_update == 0:
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
                    action,
                    (int(bbox[0]) + 5, int(bbox[1]) + 15),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.4,
                    clr,
                    1,
                )

        # RENDER STATE
        pose.loop_through_people(frame, poses, 0)

        frame = cv2.putText(
            frame,
            "%d, FPS: %f" % (frame_count, 1.0 / (time.time() - fps_time)),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        frame = cv2.resize(frame, (640, 480))
        frame = frame[:, :, ::-1]

        fps_time = time.time()

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()
