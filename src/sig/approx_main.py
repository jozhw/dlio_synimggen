import os
import sys
import time

from mpi4py import MPI

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from approx_image_gen import ApproxImageGen

COMPRESSED_FILE_TYPES = ["npz", "jpg"]


def main(imgs_data, compressed_file_types):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.time()

    # Create an instance of ImageGen on each process
    img = ApproxImageGen(imgs_data, compressed_file_types)

    # get occurance data
    # img.get_occurance_data("local_data.json")

    # Call gsid method on each process
    img.gsid("data/local_data.json", 1000)

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
