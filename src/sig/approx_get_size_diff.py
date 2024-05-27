import os
import sys
from typing import Dict, List

import pandas as pd

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)


from utils.calculations.calculate_errors import jpg_sums, npz_compressed_sums

# O_PATH = "results/local/2024-04-02/results_all_local_imgs_paths_on_2024-04-02.csv"
# S_PATH = "results/synthetic/2024-05-20/approx_synthetic_imgs_results.csv"
O_PATH = "results/polaris/2024-04-26/results_imagenet_rand_300000.csv"
S_PATH = "results/synthetic/2024-05-21/approx_synthetic_imgs_results.csv"


if __name__ == "__main__":
    n = 30000
    num_files = n
    odf = pd.read_csv(O_PATH)

    osize_column = odf["npz_compressed_image_size"]
    sample_osize = osize_column.sample(n)
    o_size = sum(sample_osize)

    uosize_column = odf["uncompressed_size"]
    sample_uosize = uosize_column.sample(n)
    uo_size = sum(sample_uosize)

    sdf = pd.read_csv(S_PATH)
    s_size = sum(sdf["npz_compressed_image_size"])
    us_size = sum(sdf["uncompressed_size"])

    udiff_size = abs(uo_size - us_size)

    diff_size = abs(o_size - s_size)

    print(
        "The original total uncompressed size in bytes for {} images is: {}".format(
            num_files, uo_size
        )
    )

    print(
        "The synthetic total uncompressed size in bytes for {} images is: {}".format(
            num_files, us_size
        )
    )
    print(
        "The difference in bytes between the total original and synthetic for {} images is: {}".format(
            num_files, udiff_size
        )
    )
    print("The difference ratio for uncompressed is {}".format(us_size / uo_size))

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
        "The difference in bytes between the total original and synthetic for {} images is: {}".format(
            num_files, diff_size
        )
    )
    print("The difference ratio for npz is {}".format(s_size / o_size))
