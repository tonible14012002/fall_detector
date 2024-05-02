import numpy as np
import cv2
import os

EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (3, 5): "m",
    (2, 2): "c",
    (4, 6): "c",
    (1, 2): "y",
    (1, 7): "m",
    (2, 8): "c",
    (7, 8): "y",
    (7, 9): "m",
    (9, 11): "m",
    (8, 10): "c",
    (10, 12): "c",
}


def is_valid_size(w, h):
    # Detection size should be divisible by 32.
    return w % 32 == 0 and h % 32 == 0


def cast_cv2_img_to_tf_tensor(image, size):
    """
    image: cv2 image
    size: (x, y)
    """
    return image
    # return tf.cast(
    #     tf.image.resize_with_pad(
    #         tf.expand_dims(image, axis=0),
    #         target_height=size[1],
    #         target_width=size[0],
    #     ),
    #     dtype=tf.int32,
    # )


def load_model():
    # os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
    # model = tf_hub.load(
    #     "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1"
    # )
    # return model.signatures["serving_default"]
    pass


# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, EDGES, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        kx, ky, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 1, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        x1, y1, c1 = shaped[p1]
        x2, y2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4
            )


class PoseEstimator:
    def __init__(self, sizeX, sizeY) -> None:
        # assert is_valid_size(sizeX, sizeY), "Invalid size for detection model."
        self.size = (sizeX, sizeY)
        self.model = None
        self.poses = None

    def load_model(self):
        pass
        # self.model = tf_hub.load s(
        #     "https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1"
        # ).signatures["serving_default"]

    def cast_to_tf_tensor(self, image):
        """
        cast cv2 image to tf tensor with resized to detection size
        """
        tf_tensor = cast_cv2_img_to_tf_tensor(image, self.size)
        print("##### cast_to_tf_tensor #####")
        try:
            print("input type: ", type(image))
            print("shape: ", image.shape)
            print("dtype", image.dtype)
        except Exception as e:
            print("error: ", e)
        print("#############################")
        return tf_tensor

    def detect(self, pose_input, body_only=True):
        return []
        # """
        # pose_input = tensor image
        # body_only: bool -> cut eyes, ears
        # return: tensor (6, 17, 3) - (poses), (keypoints), (y, x, confidence)
        # """
        # # assert self.model is not None, "Model not loaded."

        # print("##### detect #####")

        # results = self.model(pose_input)

        # print(
        #     "result model",
        #     type(results["output_0"]),
        #     results["output_0"].shape,
        # )

        # keypoints = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))
        # if body_only:
        #     keypoints = tf.concat(
        #         [keypoints[:, :1, :], keypoints[:, 5:, :]], axis=1
        #     )
        # self.poses = keypoints

        # print("keypoints type: ", type(keypoints), keypoints.shape)
        # print("now self.poses = keypoints")
        # print("#############################")
        # return keypoints

    def get_poses(self):
        return []
        # """
        # get current state poses
        # """
        # # return self.poses
        # print("##### get_poses #####")
        # print("same as filter_poses")
        # print("########################")
        # return self.poses[:, :, [1, 0, 2]]  # x, y, confidence

    def filter_poses(self, threshold=0.2):
        return []
        # assert self.poses is not None
        # print("##### filter_poses #####")
        # self.poses
        # scores = self.poses[:, :, 2]
        # mean_score_each = tf.reduce_mean(scores, axis=1)
        # mask_above_threshold = mean_score_each > threshold

        # self.poses = tf.boolean_mask(
        #     self.poses, mask_above_threshold, axis=0
        # ).numpy()
        # print(
        #     "result self.poses - ", type(self.poses), "shape", self.poses.shape
        # )
        # print("########################")
