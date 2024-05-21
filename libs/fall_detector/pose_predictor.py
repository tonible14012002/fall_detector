from .preprocessor import BasePreprocessor


class BasePosePredictor:
    device = "cpu"
    size = (640, 640)

    class PoseResults:
        poses = []  # (x,y) 13 keypoints (ignore 1,2,3,4 eyes, ears keypoints)
        bboxes_xyxy = []  # (x1, y1, x2, y2) bounding box
        scores = []  # confidence score of each poses

        def __init__(self, poses, bboxes_xyxy, scores) -> None:
            self.poses = poses
            self.bboxes_xyxy = bboxes_xyxy
            self.scores = scores

    def __init__(self) -> None:
        self.setup()
        super().__init__()

    def set_device(self, device="cpu"):
        if device not in ["cpu", "cuda"]:
            raise Exception("Invalid Device 'cuda' or 'cpu' only")
        self.device = device

    def setup(self):
        """
        overide and return the underline model
        """
        raise Exception("Not Implemented")

    def process(self, image):
        """
        return: poses (6, 17, 3) - (poses), (keypoints), (y, x, confidence)
        """
        resulst = self.preprocess(image)
        result = self.predict(resulst)
        return self.postprocess(result=result)

    def predict(self, image):
        """
        return: poses (6, 17, 3) - (poses), (keypoints), (y, x, confidence)
        """
        raise Exception("Implement predict method")

    def preprocess(self, image):
        return image

    def postprocess(self, result) -> PoseResults:
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


class ApplyPrePosProcessorMixin:
    def get_prediction(self, image):
        preprocessor = self.get_preprocessor()
        predictor = self.get_predictor()

        predict_img = preprocessor.preprocess(image=image)
        result = predictor.process(predict_img)
        return result


class BasedPoseEstimator(
    BasePosePredictorLoaderMixin, ApplyPrePosProcessorMixin
):
    """
    specify `preprocessor`,  `predictor`, `config`
    """
