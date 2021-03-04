import pytest
from src.detector.detector import TensorflowHubDetector
from src.detector.detector import TensorflowHubModel

class TestTensorflowHubDetector:

    def test_load_centernet_hourglass_512x512(self):
        """
        Checks that the hourglass model loads
        """
        # FIXME: High memory usage on test
        centernet = TensorflowHubDetector(TensorflowHubModel.CenterNetHourGlass512x512)
        centernet.load_detector_to_memory()