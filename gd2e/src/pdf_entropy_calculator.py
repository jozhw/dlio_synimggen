import csv
from typing import List

import numpy as np

from calculate_entropy import calculate_entropy


class PDFEntropyCalculator:

    def __init__(self, means: List[int], stds: List[int]):
        self.means: List[int] = means
        self.stds: List[int] = stds

        # gen gaussian distribution
        self.pd = np.arange(256)

        self.results: List = []

    def calculate_entropies(self):

        for mean in self.means:
            for std in self.stds:
                entropy = calculate_entropy(self.pd, mean, std)
                self.results.append({"mean": mean, "std": std, "entropy": entropy})

    def save_to_csv(self):
        fname = "results/results.csv"
        fieldnames = list(self.results[0].keys())
        with open(fname, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                writer.writerow(result)
