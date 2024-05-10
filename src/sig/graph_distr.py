import math
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from utils.generations.generate_save_paths import \
    generate_save_result_plot_path
from utils.setting.set_date import set_date


def graph_mean_distr(data_path: str, source):
    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)

    compression_ratio = df["mean_intensity_value"]
    num_rows = df.shape[0]

    fname = "mean_intensity_value_{}_histogram_{}.png".format(source, num_rows)
    plt.figure(figsize=(12, 8))
    plt.hist(compression_ratio, bins=50, range=(0, 255), color="blue", alpha=0.5)
    plt.title("Mean Intensity Values for {} Images".format(num_rows))
    plt.xlabel("Mean Intensity Value")
    plt.ylabel("Occurance")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


def graph_entropy_distr(data_path: str, source):
    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)

    compression_ratio = df["entropy"]
    num_rows = df.shape[0]

    fname = "entropy_{}_histogram_{}.png".format(source, num_rows)
    plt.figure(figsize=(12, 8))
    plt.hist(compression_ratio, bins=50, range=(0, 8), color="blue", alpha=0.5)
    plt.title("Entropies for {} Images".format(num_rows))
    plt.xlabel("Entropy")
    plt.ylabel("Occurance")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


def graph_size_distr(data_path: str, source):
    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)

    compression_ratio = df["uncompressed_size"]
    num_rows = df.shape[0]

    # Create a new column to store the results
    df["closest_x"] = None

    # Iterate over the rows
    for index, row in df.iterrows():
        uncompressed_size = row["uncompressed_size"]

        # Check if uncompressed_size is a multiple of 3
        if uncompressed_size % 3 == 0:
            quotient = uncompressed_size // 3

            # Find the closest integer x such that x^2 is closest to the quotient
            closest_x = int(round(math.sqrt(quotient)))

            # Update the 'closest_x2 column
            df.at[index, "closest_x"] = closest_x

    fname = "x_dimension_{}_histogram_{}.png".format(source, num_rows)
    plt.figure(figsize=(12, 8))
    plt.hist(df["closest_x"], bins=100, range=(0, 2000), color="blue", alpha=0.5)
    plt.title("X dimension for {} Images".format(num_rows))
    plt.xlabel("X dimension")
    plt.ylabel("Occurance")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


if __name__ == "__main__":

    data_path = "./results/polaris/2024-04-26/results_imagenet_rand_300000.csv"
    graph_entropy_distr(data_path, "polaris")
    graph_mean_distr(data_path, "polaris")

    data_path2 = (
        "./results/local/2024-04-02/results_all_local_imgs_paths_on_2024-04-02.csv"
    )
    graph_entropy_distr(data_path2, "local")
    graph_mean_distr(data_path2, "local")

    graph_size_distr(data_path, "polaris")
    graph_size_distr(data_path2, "local")
