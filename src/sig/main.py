import os
import sys

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from image_gen import ImageGen

COMPRESSED_FILE_TYPES = ["npz", "jpg"]


def main(imgs_data, compressed_file_types):

    img = ImageGen(imgs_data, compressed_file_types)
    img.gsid()


if __name__ == "__main__":
    imgs_data = "./results/polaris/data/2024-04-09/results_imagenet_rand_100.csv"
    main(imgs_data, COMPRESSED_FILE_TYPES)
