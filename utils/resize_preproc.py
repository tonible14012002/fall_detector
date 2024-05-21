from libs.fall_detector.detection import utils
import cv2


def resize_bgr2rgb_preproc(image, size=(640, 640)):
    """preprocess function for CameraLoader."""
    resizer = utils.ResizePadding(*size[::-1])
    image = resizer(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


resizer384 = utils.ResizePadding(384, 384)


def resize384_bgr2rgb_preproc(image, size=(384, 384)):
    """preprocess function for CameraLoader."""
    image = resizer384(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
