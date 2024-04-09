from typing import List

import numpy as np

from utils.calculations.pdf_calculate_entropy import pdf_calculate_entropy
from utils.generations.generate_csv import generate_results_csv


class PDFEntropyCalculator:

    def __init__(self, means: List[int], stds: List[int]):
        self.means: List[int] = means
        self.stds: List[int] = stds

        # gen gaussian distribution
        self.pdx = np.arange(256)

        self.results: List = []

    def calculate_entropies(self):

        for mean in self.means:
            for std in self.stds:
                entropy = pdf_calculate_entropy(self.pdx, mean, std)
                self.results.append({"mean": mean, "std": std, "entropy": entropy})

    def save_to_csv(self):
        fname = "results_gd2e"
        generate_results_csv(self.results, fname)
