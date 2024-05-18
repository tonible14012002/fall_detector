from aiortc.mediastreams import MediaStreamTrack
import cv2
from av import VideoFrame
from libs.fall_detector.detection import utils
from utils.cam import CamLoader_Q, CamLoader
from typing import Union


def preproc(image):
    """preprocess function for CameraLoader."""
    resizer = utils.ResizePadding(640, 640)
    image = resizer(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"
    cam: Union[CamLoader_Q, CamLoader] = None

    def __init__(self, cam: Union[CamLoader_Q, CamLoader]):
        super().__init__()  # don't forget this!
        self.cam = cam
        self.frame_count = 0
        self.prev_frame = None

    def process_cv_frame(self, frame):
        return frame

    async def recv(self):
        import time

        if not self.cam.grabbed():
            if self.prev_frame is not None:
                self.frame_count += 1
                self.prev_frame.pts = self.frame_count + 1
                return self.prev_frame
            return None

        img = self.cam.getitem()
        img = self.process_cv_frame(img)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = self.frame_count  # frame.pts
        new_frame.time_base = "1"  # frame.time_base

        self.prev_frame = new_frame
        self.frame_count += 1
        return new_frame
