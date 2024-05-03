from .preprocessor import BasePreprocessor


class BasePosePredictor:
    def __init__(self) -> None:
        self.setup()
        super().__init__()

    def setup(self):
        """
        overide and return the underline model
        """
        raise Exception("Not Implemented")

    def process(self, image):
        """
        return: poses (6, 17, 3) - (poses), (keypoints), (y, x, confidence)
        """
        self.preprocess(image)
        result = self.predict(image)
        return self.postprocess(result=result)

    def predict(self, image):
        """
        return: poses (6, 17, 3) - (poses), (keypoints), (y, x, confidence)
        """
        raise Exception("Implement predict method")

    def preprocess(self, image):
        return image

    def postprocess(self, result):
        return result


class BasePosePredictorLoaderMixin:
    """
    specify `preprocessor` and `predictor`
    or overload the `get_preprocessor` `get_predictor` method
    """

    preprocessor = None
    predictor = None

    def get_preprocessor(self) -> BasePreprocessor:
        return self.preprocessor

    def get_predictor(self) -> BasePosePredictor:
        return self.predictor


class BasedPoseEstimator(BasePosePredictorLoaderMixin):
    """
    specify `preprocessor`,  `predictor`, `config`
    """

    def get_prediction(self, image):
        preprocessor = self.get_preprocessor()
        predictor = self.get_predictor()

        predict_img = preprocessor.preprocess(image=image)
        result = predictor.process(predict_img)
        return result
