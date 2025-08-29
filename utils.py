import numpy as np
from PIL import Image
import io
import base64

def preprocess_image(base64_str, target_size):
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # normalize
    # Expand dims for batch: (1, height, width, channels)
    return np.expand_dims(image_array, axis=0)
