import numpy as np

from graph_plots import plot_entropy_vs_std
from pdf_entropy_calculator import PDFEntropyCalculator


def main(means, stds):
    pdfec = PDFEntropyCalculator(means, stds)

    pdfec.calculate_entropies()
    pdfec.save_to_csv()


if __name__ == "__main__":
    means = [1, 64, 127, 191, 255]
    array1 = np.arange(0.01, 2.01, 0.02)
    array2 = np.arange(2, 30, 0.2)
    array3 = np.arange(30, 200, 2)
    stds = np.concatenate((array1, array2, array3))
    main(means, stds)
    plot_entropy_vs_std("results/results.csv")
