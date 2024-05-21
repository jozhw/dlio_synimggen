import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, poisson, skew

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


def graph_x_distr(data_path, source):
    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)
    sizes = df["uncompressed_size"].values

    size_array = np.array(sizes)

    x_array = np.rint(np.sqrt(size_array / 3))
    x = np.mean(x_array)
    var_x = np.var(x_array)
    skew_x = skew(x_array)
    kurtosis_x = kurtosis(x_array)
    std = np.std(x_array)

    print(x)
    print(var_x)
    print(skew_x)
    print(kurtosis_x)
    print(std)

    num_rows = df.shape[0]
    fname = "x_dimension_{}_histogram_{}.png".format(source, num_rows)
    plt.figure(figsize=(12, 8))
    plt.hist(x_array, bins=100, range=(0, 2000), color="blue", alpha=0.5)
    plt.title("x dimension for {} Images".format(num_rows))
    plt.xlabel("x dimension")
    plt.ylabel("Occurance")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


def graph_x_quantiles_distr(data_path, source):
    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)
    lower_percentile = df["uncompressed_size"].quantile(0.25)
    upper_percentile = df["uncompressed_size"].quantile(0.75)
    iqr = upper_percentile - lower_percentile

    lower_bound = lower_percentile - (1.5 * iqr)

    if lower_bound < 0:
        lower_bound = 0

    upper_bound = upper_percentile + (1.5 * iqr)

    column_array = df["uncompressed_size"].values
    filtered_array = column_array[
        (column_array >= lower_bound) & (column_array <= upper_bound)
    ]
    size_array = np.array(filtered_array)

    x_array = np.rint(np.sqrt(size_array / 3))
    x = np.mean(x_array)
    var_x = np.var(x_array)
    skew_x = skew(x_array)
    kurtosis_x = kurtosis(x_array)
    lowerbound = min(x_array)
    upperbound = max(x_array)
    std = np.std(x_array)

    print(x)
    print(var_x)
    print(skew_x)
    print(kurtosis_x)
    print(lowerbound)
    print(upperbound)
    print(std)

    num_rows = df.shape[0]
    fname = "x_dimension_{}_histogram_{}.png".format(source, num_rows)
    plt.figure(figsize=(12, 8))
    plt.hist(x_array, bins=100, range=(0, 2000), color="blue", alpha=0.5)
    plt.title("x dimension for {} Images".format(num_rows))
    plt.xlabel("x dimension")
    plt.ylabel("Occurance")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


def graph_width_distr(data_path, source):

    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)

    width = df["uncompressed_width"]

    print(width.mean())
    print(width.var())
    print(skew(width))

    num_rows = df.shape[0]
    fname = "width_dimension_{}_histogram_{}.png".format(source, num_rows)
    plt.figure(figsize=(12, 8))
    plt.hist(width, bins=100, range=(0, 2000), color="blue", alpha=0.5)
    plt.title("width dimension for {} Images".format(num_rows))
    plt.xlabel("width dimension")
    plt.ylabel("Occurance")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


def graph_height_distr(data_path, source):

    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)

    width = df["uncompressed_height"]
    print(width.mean())
    print(width.var())
    print(skew(width))
    num_rows = df.shape[0]
    fname = "height_dimension_{}_histogram_{}.png".format(source, num_rows)
    plt.figure(figsize=(12, 8))
    plt.hist(width, bins=100, range=(0, 2000), color="blue", alpha=0.5)

    # x = np.linspace(0, 2000, 1000)
    mu = width.mean()
    x = np.arange(0, 2000)

    plt.plot(x, poisson.pmf(x, mu), label="approximate function")
    plt.title("height dimension for {} Images".format(num_rows))
    plt.xlabel("height dimension")
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
    graph_x_quantiles_distr(data_path, "polaris")

    data_path2 = (
        "./results/local/2024-04-02/results_all_local_imgs_paths_on_2024-04-02.csv"
    )
    graph_entropy_distr(data_path2, "local")
    graph_mean_distr(data_path2, "local")
    graph_x_quantiles_distr(data_path2, "local")
