import os
import sys
import time
from math import sqrt

import pandas as pd
from mpi4py import MPI

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from distr_image_gen import DistrImageGen

COMPRESSED_FILE_TYPES = ["npz", "jpg"]

"""
In order to get the data to generate the synthetic image, I would need to get
the average entropy, average intensity value (mean), and the x_size

Since the image data I gathered contains the width, and height, in order to find
the x_size I would need to do a sqrt(width * height) to the nearest integer.

"""


def main(imgs_data, compressed_file_types):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.time()

    df = pd.read_csv(imgs_data)

    mean_entropy = df["entropy"].mean()
    mean_width = df["uncompressed_width"].mean()
    mean_height = df["uncompressed_height"].mean()

    x = int(sqrt(mean_width * mean_height))

    # Create an instance of ImageGen on each process
    img = DistrImageGen(imgs_data, compressed_file_types)

    # Call gsid method on each process
    img.gsid(1000, mean_entropy, x)

    # only root process (rank 0) prints the elapsed time
    if rank == 0:
        # End the timer
        end_time = time.time()

        # Calculate and print elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    # Finalize MPI environment
    MPI.Finalize()


if __name__ == "__main__":
    imgs_data = (
        "./results/local/2024-04-02/results_all_local_imgs_paths_on_2024-04-02.csv"
    )
    main(imgs_data, COMPRESSED_FILE_TYPES)
