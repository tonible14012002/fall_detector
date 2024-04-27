import time
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.mediastreams import MediaStreamTrack
import cv2
from av import VideoFrame


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.fps_time = 0
        self.frame_count = 0
        # self.transform = "cartoon"


    async def recv(self):
        frame = await self.track.recv()
        start = time.time()
        img = frame.to_ndarray(format="bgr24")
        end = time.time()
        print("convert time ", end - start)

        # Resize the image only if necessary
        if img.shape[0] != 480 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 480))

        end1 = time.time()
        print("convert time 2", end1 - end)

        # Use in-place operation
        cv2.putText(
            img=img,
            text="%d, FPS: %f"
            % (self.frame_count, 1.0 / (time.time() - self.fps_time)),
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 255, 0),
            thickness=4,
        )

        self.fps_time = time.time()
        print("convert time 3", self.fps_time - end1)

        # Avoid unnecessary conversion
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        end4 = time.time()
        print("convert time 4", end4 - self.fps_time)

        return new_frame
