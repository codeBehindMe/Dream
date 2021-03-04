from io import BytesIO

import numpy as np
import pytest
import requests

from src.utils.image import load_image_to_numpy


@pytest.fixture(scope='class')
def image():
    url = "https://images.unsplash.com/photo-1532362996300-fbce5a30bd6d?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80"

    response = requests.get(url)
    return BytesIO(response.content)


@pytest.mark.usefixtures('image')
class TestLoadImageToNumpy:

    def test_load_image_to_numpy_returns_np_array(self, image: BytesIO):
        assert isinstance(load_image_to_numpy(image), np.ndarray)
