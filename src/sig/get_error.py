import os
import sys
from typing import Dict, List

import pandas as pd

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)


from utils.calculations.calculate_errors import (
    mean_absolute_error,
    mean_error_rate,
    mean_squared_error,
)

DATA = [
    "results/synthetic/2024-04-18/synthetic_imgs_results.csv",
    "results/synthetic/2024-04-18/synthetic_imgs_adjusted_entropy_and_mean_results.csv",
    "results/synthetic/2024-04-19/synthetic_imgs_results.csv",
    "results/synthetic/2024-04-19/synthetic_imgs_adjusted_entropy_and_mean_results.csv",
]


def get_error(data_paths: List[str]) -> Dict[str, Dict]:
    errors: Dict = {}
    for data_path in data_paths:
        df: pd.DataFrame = pd.read_csv(data_path)
        values = list(df["npz_compressed_synthetic_original_ratio"])
        s_img_compressed_size = list(df["npz_compressed_image_size"])
        o_img_compressed_size = list(df["original_npz_compressed_image_size"])
        mse = mean_squared_error(values)
        mae = mean_absolute_error(values)
        mer = mean_error_rate(s_img_compressed_size, o_img_compressed_size)

        errors[data_path] = {
            "mean_squared_error": mse,
            "mean_absolute_error": mae,
            "mean_error_rate": mer,
        }
    return errors


if __name__ == "__main__":
    print(get_error(DATA))
