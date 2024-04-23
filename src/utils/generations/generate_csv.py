import csv
import os
from datetime import datetime
from typing import Dict, List


def generate_results_csv(
    results: List[Dict[str, any]], img_source: str, fname: str, chunk_size: int = 10000
) -> None:
    current_date: str = datetime.now().strftime("%Y-%m-%d")
    path: str = "results/{}/{}".format(img_source, current_date)

    if not os.path.exists(path):
        os.makedirs(path)

    save_file_path: str = os.path.join(path, fname + ".csv")
    fieldnames: List = list(results[0].keys())

    with open(save_file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(results), chunk_size):
            chunk = results[i : i + chunk_size]
            writer.writerows(chunk)

    print("Results have been saved to: {}".format(save_file_path))
