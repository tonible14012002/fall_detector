import tensorflow as tf
import numpy as np


def kpt2bbox_tf(kpt, ex=20, frame_size=(640, 480)):
    """Get bbox that holds on all of the keypoints (x, y)
    kpt: tensor of shape `(N, 3)` (N keypoint with (x, y, confidence) each),
    ex: (int) expand bounding box,
    frame_size: (tuple) (width, height) of the frame
    """
    kpt = np.multiply(kpt, [frame_size[1], frame_size[0], 1])
    bbox_min = tf.reduce_min(kpt[:, :2], axis=0) - ex
    bbox_max = tf.reduce_max(kpt[:, :2], axis=0) + ex
    bbox = tf.stack([bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]])
    return bbox
