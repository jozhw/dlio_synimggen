import os


def calculate_compression_ratio(compressed_filename: str, dimensions):
    # Get the size of the image file
    compressed_file_size: int = os.path.getsize(compressed_filename)

    # Calculate the number of pixels in the image
    num_pixels: int = dimensions[0] * dimensions[1]

    # Calculate the compression ratio
    # assuming 3 bytes per pixel for RGB
    # must be greater than or equal to 1
    # equation is uncompressed/compressed
    compression_ratio: float = (num_pixels * 3) / compressed_file_size

    return compression_ratio
