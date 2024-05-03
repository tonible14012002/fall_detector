from utils.cam import CamLoader, CamLoader_Q
from libs.fall_detector.detection import utils
from libs.fall_detector.tracker import (
    Tracker,
    Detection,
    utils as tracker_utils,
)
from libs.fall_detector.detection import detector
import argparse
import cv2
import os
import time
import numpy as np
from setup import YoloBasedPoseEstimator, draw_poses

DEFAULT_CAMERA_SOURCE = "./scripts/samples/fall-vid.mp4"


def preproc(image):
    """preprocess function for CameraLoader."""
    resizer = utils.ResizePadding(640, 640)
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


def initialize_detection():
    # Initialize pose estimator
    pose_estimator = YoloBasedPoseEstimator()
    pose_estimator.set_predictor_device(device=device)

    # Initialize tracker
    tracker = Tracker(max_age=30, max_iou_distance=0.7, n_init=3)

    # Initialize action detector
    action_detector = detector.TSSTG(device=device)
    return pose_estimator, tracker, action_detector


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
    pose_estimator, tracker, action_detector = initialize_detection()

    while cam.grabbed():
        frame = cam.getitem()
        image = frame.copy()

        prediction = pose_estimator.get_prediction(image)

        poses = prediction.poses
        bboxs = prediction.bboxes_xyxy
        scores = prediction.scores
        print(prediction.edge_links)

        # TRACK POSES
        tracker.predict()

        detections = [
            Detection(
                bbox,
                ps,
                score,
            )
            for ps, bbox, score in zip(poses, bboxs, scores)
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
            if (
                track.keypoints_list is not None
                and len(track.keypoints_list) > 0
            ):
                draw_poses(
                    frame=frame,
                    pose=track.keypoints_list[len(track.keypoints_list) - 1],
                    detection_size=(640, 640),
                )

            # RENDER BBOX
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
