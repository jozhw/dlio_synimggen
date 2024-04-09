import os
import sys

import numpy as np

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from graph_plots import plot_entropy_vs_std
from pdf_entropy_calculator import PDFEntropyCalculator


def main(means, stds):
    pdfec = PDFEntropyCalculator(means, stds)

    pdfec.calculate_entropies()
    pdfec.save_to_csv()


if __name__ == "__main__":
    means = [127]
    array1 = np.arange(0.01, 2.01, 0.02)
    array2 = np.arange(2, 30.1, 0.1)
    array3 = np.arange(30, 3000.5, 0.5)
    stds = np.concatenate((array1, array2, array3))
    main(means, stds)
    # plot_entropy_vs_std("results/results.csv")
