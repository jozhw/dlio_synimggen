import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from utils.generations.generate_save_paths import generate_save_result_plot_path
from utils.setting.set_date import set_date


def graph_npz_compressed_synthetic_original_ratio(data_path: str):
    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)

    compression_ratio = df["npz_compressed_synthetic_original_ratio"]
    num_rows = df.shape[0]

    fname = "histogram_npz_compressed_synthetic_original_ratio_plot_{}.png".format(
        num_rows
    )
    plt.figure(figsize=(12, 8))
    plt.hist(compression_ratio, bins=30, color="blue", alpha=0.5)
    plt.title(
        " Synthetic Image Size / Original Image Size for {} NPZ Compressed Images".format(
            num_rows
        )
    )
    plt.xlabel("Synthetic/Original NPZ Compressed Image Size Ratio")
    plt.ylabel("Occurance")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


if __name__ == "__main__":

    data_path = "./results/synthetic/2024-04-24/synthetic_imgs_results.csv"
    graph_npz_compressed_synthetic_original_ratio(data_path)