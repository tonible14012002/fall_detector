from libs.fall_detector.detection import utils
import cv2


def detection_preproc(image, size=(640, 640)):
    """preprocess function for CameraLoader."""
    resizer = utils.ResizePadding(*size[::-1])
    image = resizer(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
