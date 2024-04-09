import numpy as np

from utils.compressions.compress_to_jpg import compress_to_jpg
from utils.compressions.compress_to_npz import compress_to_npz


def compress_files_wrapper(
    file_type: str,
    compressed_image_path: str,
    image: np.ndarray,
):
    if file_type == "npz":
        compress_to_npz(compressed_image_path, image)
    elif file_type == "jpg":
        compress_to_jpg(compressed_image_path, image)
    else:
        # put other compression algorithms here
        pass
