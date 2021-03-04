import pytest
from src.detector.detector import TensorflowHubDetector
from src.detector.detector import TensorflowHubModel

class TestTensorflowHubDetector:

    def test_load_hub_model(self):
        """
        Check's that the hub model loads correctly.
        """
        # FIXME: High memory usage on test
        ssd_net = TensorflowHubDetector(TensorflowHubModel.SSDMobilNetV2)
        ssd_net.load_detector_to_memory()

        assert ssd_net.detector