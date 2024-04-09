import os
from typing import Dict, List, Tuple

import numpy as np

from utils.calculations.calculate_compression_ratio import calculate_compression_ratio
from utils.calculations.calculate_entropy import calculate_entropy, count_occurrences
from utils.compressions.compress_file_wrapper import compress_file_wrapper
from utils.deletions.delete_file import delete_file

"""
Serves as a wrapper for all of the compressions and for calculating the compression
ratio.

return a dictionary of the compression_ratio and file_size for each compressed_file_type
"""


def calculate_img_data(
    file_name: str,
    image: np.ndarray,
    dimensions: Tuple,
    compressed_file_types: List[str],
    compressed_save_paths: Dict[str, str],
) -> Dict[str, float]:

    result: Dict = {"file_name": file_name}

    # calculate entropy
    occurances = count_occurrences(image)
    entropy = calculate_entropy(occurances, dimensions)

    # save entropy to result object
    result["entropy"] = entropy

    for file_type in compressed_file_types:

        key_size: str = "{}_compressed_image_size".format(file_type)
        key_cr: str = "{}_compression_ratio".format(file_type)

        compressed_image_path: str = os.path.join(
            compressed_save_paths[file_type],
            f"{file_name}.{file_type}",
        )

        compress_file_wrapper(file_type, compressed_image_path, image)

        # calculate the compression ratio
        compression_ratio: float = calculate_compression_ratio(
            compressed_image_path, dimensions
        )

        # calculate compressed file size
        compressed_file_size: int = os.path.getsize(compressed_image_path)

        # save to results object
        result[key_cr] = compression_ratio
        result[key_size] = compressed_file_size

        # delete compressed image here
        delete_file(compressed_image_path)

    return result
