import config as app_config
import fall_detection


class Entities:
    config: app_config.AppConfig = None
    fall_detector: fall_detection.FallDetection = None

    @classmethod
    def new(
        cls,
        fall_detector: fall_detection.FallDetection = None,
        config: app_config.AppConfig = None,
    ):

        if fall_detector is None:
            raise Exception("fall_detector is required")
        if config is None:
            raise Exception("config is required")

        entities = cls()
        entities.config = config
        entities.fall_detector = fall_detector
        return entities
