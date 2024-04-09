from typing import Dict, List

from utils.calculations.entropy_calculate_std import entropy_calculate_std
from utils.calculations.pdf_calculate_entropy import pdf_calculate_entropy
from utils.generations.generate_csv import generate_results_csv
from utils.generations.generate_pdf import generate_pdf


def entropy_pdf_analyzer(target_entropies: List):
    # results array of dictionaries to save to csv
    results: List[Dict] = []

    for target_entropy in target_entropies:
        # local scope result dictionary
        result: Dict = {}

        est_std = entropy_calculate_std(target_entropy)
        pdf = generate_pdf(std=est_std)
        est_entropy = pdf_calculate_entropy(pdf)
        target_est_entropy_diff = target_entropy - est_entropy

        result["target_entropy"] = target_entropy
        result["estimated_std"] = est_std
        result["estimated_entropy"] = est_entropy
        result["target_est_entropy_diff"] = target_est_entropy_diff

        results.append(result)

    generate_results_csv(results, "synthetic", "results_e2gd")
