import numpy as np
from PIL import Image

"""
for highest quality, set the quality to 95 in the Image.save method as recommended
by the documentation (https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#)

the jpg uses the jpeg compression algorithm.
"""


def compress_to_jpg(compressed_image_path: str, image: np.ndarray):
    try:
        img = Image.fromarray(image.astype("uint8"), "RGB")
        # quality to 95 as that is recommend from the docs for the highest image res
        img.save(compressed_image_path, quality=95)
    except Exception as e:
        print("Error: {}".format(e))
