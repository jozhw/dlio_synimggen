import numpy as np

"""
Use the np.savez_compressed method that uses a lossless compresson algorithm to
create a npz file.

The specific algorithm comes from the zipfile (https://docs.python.org/3/library/zipfile.html)
particularly the zlib module (https://docs.python.org/3/library/zlib.html#module-zlib).

There is a useful article to get into the crux of the algorithm involved in zlib here (https://www.euccas.me/zlib/)
"""


def compress_to_npz(compressed_image_path: str, image: np.ndarray):
    try:
        np.savez_compressed(compressed_image_path, image)
    except Exception as e:
        print("Error: {}".format(e))
