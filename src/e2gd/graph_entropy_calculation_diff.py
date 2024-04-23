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


def graph_entropy_calculation_diff(data_path: str):
    date = set_date(data_path)
    save_path = generate_save_result_plot_path(date)
    df = pd.read_csv(data_path)

    ratio = df["target_est_entropy_diff"]
    target_entropy = df["target_entropy"]
    num_rows = df.shape[0]

    # negative_values = df["est_target_entropy_ratio"] < 0.0
    # rows_with_negative_values = df[negative_values]
    # print(rows_with_negative_values)

    fname = "target_est_entropy_diff_plot_{}.png".format(num_rows)
    plt.figure(figsize=(12, 8))
    plt.scatter(
        target_entropy, ratio, color="blue", alpha=0.5, marker="o", linestyle="-"
    )
    plt.title("Target Entropy - Estimated Entropy Difference by Target Entropy")
    plt.xlabel("target_entropy")
    plt.ylabel("Target Entropy and Estimated Entropy Difference")
    plt.grid(True)

    plt.savefig(os.path.join(save_path, fname))

    plt.show()


if __name__ == "__main__":
    data_path = "./results/synthetic/2024-04-23/results_e2gd.csv"
    graph_entropy_calculation_diff(data_path)
