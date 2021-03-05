from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from io import BytesIO
from typing import Dict

import numpy as np
import tensorflow_hub as hub

from src.utils.image import load_image_to_numpy


class IDetector(metaclass=ABCMeta):

    @abstractmethod
    def detect(self,
               img: BytesIO) -> Dict:
        # FIXME: Missing docstring.
        pass

    @abstractmethod
    def load_detector(self):
        # FIXME: Missing docstring.
        pass

    @abstractmethod
    def preprocess_image(self,
                         image: object) -> object:  # FIXME: Object type hint is too broad
        # FIXME: Missing docstring.
        pass


class TensorflowHubModel(Enum):
    CenterNetHourGlass512x512 = "https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1"
    SSDMobilNetV2 = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"


class TensorflowHubDetector(IDetector):

    def __init__(self, model: TensorflowHubModel):
        # FIXME: Missing docstring.
        self.model = model
        self.detector = None

    @staticmethod
    def _load_model(model: TensorflowHubModel) -> object:
        # FIXME: Missing docstring.
        return hub.load(model.value)

    def preprocess_image(self, image: BytesIO) -> np.ndarray:
        # FIXME: Missing docstring.
        return load_image_to_numpy(image)

    def detect(self, img: BytesIO) -> Dict:
        """
        Runs inference on using the detection model.
        Args:
            img (BytesIO): BytesIO interfaced image data.

        Returns:
            Dict: Dictionary of inference data containing the following key:values
                num_detections: a tf.int tensor with only one value, the number of detections [N].
                detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
                detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
                detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
                raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without Non-Max suppression. M is the number of raw detections.
                raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw detection boxes. M is the number of raw detections.
                detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the detections after NMS.
                detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 91] and contains class score distribution (including background) for detection boxes in the image including background class.

        """
        return self.detector(self.preprocess_image(img))

    def load_detector(self):
        # FIXME: Missing docstring.
        self.detector = self._load_model(self.model)
