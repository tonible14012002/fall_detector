from .config import ConfigLoaderMixin


class BasePreprocessor:
    """
    Empty preprocessor
    Override `preprocess` method, it should return `cv2 image`
    """

    def preprocess(self, image):
        return image
