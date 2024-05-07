import os
import sys
from typing import Dict, List

import pandas as pd

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)


from utils.calculations.calculate_errors import jpg_sums, npz_compressed_sums

O_PATH = "results/local/2024-04-02/results_all_local_imgs_paths_on_2024-04-02.csv"
S_PATH = "results/synthetic/2024-05-02/synthetic_imgs_results.csv"
# O_PATH = "results/polaris/2024-04-26/results_imagenet_rand_300000.csv"
# S_PATH = "results/synthetic/2024-04-26/synthetic_imgs_results.csv"


if __name__ == "__main__":
    o_size, s_size, diff_size, num_files = npz_compressed_sums(O_PATH, S_PATH)
    jpeg_o_size, jpeg_s_size, jpeg_diff_size, jpeg_num_files = jpg_sums(O_PATH, S_PATH)

    print(
        "The original total npz compressed size in bytes for {} images is: {}".format(
            num_files, o_size
        )
    )
    print(
        "The synthetic total npz compressed size in bytes for {} images is: {}".format(
            num_files, s_size
        )
    )
    print(
        "The difference in bytes between the total original and synthetic (original size - synthetic size) for {} images is: {}".format(
            num_files, diff_size
        )
    )
    print("The difference ratio for npz is {}".format(s_size / o_size))

    print(
        "The original total jpg compressed size in bytes for {} images is: {}".format(
            num_files, jpeg_o_size
        )
    )
    print(
        "The synthetic total jpg compressed size in bytes for {} images is: {}".format(
            num_files, jpeg_s_size
        )
    )
    print(
        "The difference in bytes between the total original and synthetic (original size - synthetic size) for {} images is: {}".format(
            num_files, jpeg_diff_size
        )
    )
    print("The difference ratio for jpeg is {}".format(jpeg_s_size / jpeg_o_size))
