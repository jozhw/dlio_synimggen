import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_entropy_vs_std(path_to_csv: str):
    # read CSV file into a pandas DataFrame
    df = pd.read_csv(path_to_csv)

    full_fname = os.path.basename(path_to_csv)
    fname = os.path.splitext(full_fname)[0]

    # identify unique mean values and assign a color to each
    unique_means = df["mean"].unique()
    num_means = len(unique_means)
    # define a list of colors to use
    colors = ["b", "g", "r", "c", "m", "y", "k"][:num_means]

    # generate scatter plot
    plt.figure(figsize=(12, 8))
    legend_handles = []
    for i, row in df.iterrows():
        std = row["std"]
        entropy = row["entropy"]
        mean = row["mean"]
        color_index = list(unique_means).index(mean) % num_means
        color = colors[color_index]
        # Only add legend for the first occurrence of each mean
        if mean not in legend_handles:
            legend_handles.append(mean)
            plt.scatter(std, entropy, c=color, label=f"Mean: {mean:.2f}")
        else:
            plt.scatter(std, entropy, c=color)

    plt.xlabel("Standard Deviation")
    plt.ylabel("Entropy")
    plt.title("Scatter Plot of Mean, Standard Deviation, and Entropy")

    plt.legend()

    # save plot
    function_name = plot_entropy_vs_std.__name__  # Get function name dynamically
    plot_filename = (
        f"{function_name}_{os.path.splitext(os.path.basename(fname))[0]}.png"
    )
    plt.savefig(plot_filename)

    plt.grid(True)
    plt.show()
