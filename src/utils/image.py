from six import BytesIO
from PIL import Image
import numpy as np

def load_image_to_numpy(b : BytesIO) -> np.array:
    """Loads a an image from BytesIO object to numpy array

    Args:
        b (BytesIO): BytesIO object

    Returns:
        np.array: numpy array with image data
    """

    img = Image.open(b)
    
    w, h = img.size
    return np.array(img.getdata()).reshape(1, w, h, 3).astype(np.uint8)
