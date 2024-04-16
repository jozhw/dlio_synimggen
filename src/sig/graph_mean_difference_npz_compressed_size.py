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


def graph_mean_difference_npz_compressed_size(
    data_path_constant_mean: str, data_path_adjusted_mean: str
):
    date = set_date(data_path_constant_mean)
    save_path = generate_save_result_plot_path(date)
    df_const_mean = pd.read_csv(data_path_constant_mean)
    df_adjusted_mean = pd.read_csv(data_path_adjusted_mean)

    comp_ratio_diff = (
        df_const_mean["npz_compressed_image_size"]
        - df_adjusted_mean["npz_compressed_image_size"]
    )
    mean_diff = df_const_mean["mean_used"] - df_adjusted_mean["mean_used"]
    num_rows = df_const_mean.shape[0]

    # negative_values = df["est_target_entropy_ratio"] < 0.0
    # rows_with_negative_values = df[negative_values]
    # print(rows_with_negative_values)

    fname = "npz_comp_size_diff_by_mean_diff_plot_{}.png".format(num_rows)
    plt.figure(figsize=(12, 8))
    plt.scatter(
        mean_diff, comp_ratio_diff, color="blue", alpha=0.5, marker="o", linestyle="-"
    )
    plt.title(
        "NPZ Compressed Image Size Difference from Standard Mean vs. Adjusted Mean"
    )
    plt.xlabel("mean difference (172 - adjusted mean)")
    plt.ylabel("npz compressed image size difference (172 - adjusted mean)")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


if __name__ == "__main__":
    data_path_constant_mean = "results/synthetic/2024-04-16/synthetic_imgs_results.csv"
    data_path_adjusted_mean = (
        "results/synthetic/2024-04-16/synthetic_imgs_results_adjusted_mean.csv"
    )
    graph_mean_difference_npz_compressed_size(
        data_path_constant_mean, data_path_adjusted_mean
    )
