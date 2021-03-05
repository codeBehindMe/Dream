from abc import ABCMeta
from abc import abstractmethod
from io import BytesIO

import tensorflow as tf
import tensorflow_hub as hub
from enum import Enum

class IDetector(metaclass=ABCMeta):

    @abstractmethod
    def detect(self, img: BytesIO) -> tf.Tensor: # FIXME: Object type hint is too broad
        # FIXME: Missing docstring.
        pass

    @abstractmethod
    def load_detector(self):
        # FIXME: Missing docstring.
        pass

    @abstractmethod
    def preprocess_image(self, image : object) -> object: # FIXME: Object type hint is too broad
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

    def _load_model(self, model: TensorflowHubModel) -> object:
        # FIXME: Missing docstring.
        return hub.load(model.value)

    def preprocess_image(self, image: object) -> object:
        # FIXME: Missing docstring.
        return image

    def detect(self, img: object) -> tf.Tensor:
        # FIXME: Missing docstring.
        raise NotImplementedError() 

    def load_detector(self):
        # FIXME: Missing docstring.
        self.detector = self._load_model(self.model)


