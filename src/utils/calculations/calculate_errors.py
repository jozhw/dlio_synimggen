from typing import List

import numpy as np
import pandas as pd


def mean_absolute_error(actual_values: List[float], ideal_value: float = 1.0):
    size = len(actual_values)
    ideal_values: List[float] = [ideal_value] * size
    absolute_errors = np.abs(np.array(actual_values) - np.array(ideal_values))
    return np.mean(absolute_errors)


def mean_squared_error(actual_values: List[float], ideal_value: float = 1.0):
    size = len(actual_values)
    ideal_values: List[float] = [ideal_value] * size
    squared_errors = (np.array(actual_values) - np.array(ideal_value)) ** 2
    return np.mean(squared_errors)


def mean_error_rate(actual_values: List[int], ground_truth_values: List[int]):
    error_rate = (
        (np.array(actual_values) - np.array(ground_truth_values))
        / np.array(ground_truth_values)
        * 100
    )
    return np.mean(error_rate)


def npz_compressed_sums(path_to_original_data, path_to_synthetic_data):
    odf = pd.read_csv(path_to_original_data)
    sdf = pd.read_csv(path_to_synthetic_data)

    num_files = len(odf["npz_compressed_image_size"])

    if num_files != len(sdf["npz_compressed_image_size"]):
        raise ValueError("Wrong dataset")

    sum_ocompressed_size = sum(odf["npz_compressed_image_size"])
    sum_scompressed_size = sum(sdf["npz_compressed_image_size"])

    diff = sum_ocompressed_size - sum_scompressed_size

    return sum_ocompressed_size, sum_scompressed_size, diff, num_files


def jpg_sums(path_to_original_data, path_to_synthetic_data):
    odf = pd.read_csv(path_to_original_data)
    sdf = pd.read_csv(path_to_synthetic_data)

    num_files = len(odf["jpg_compressed_image_size"])

    if num_files != len(sdf["jpg_compressed_image_size"]):
        raise ValueError("Wrong dataset")

    sum_ocompressed_size = sum(odf["jpg_compressed_image_size"])
    sum_scompressed_size = sum(sdf["jpg_compressed_image_size"])

    diff = sum_ocompressed_size - sum_scompressed_size

    return sum_ocompressed_size, sum_scompressed_size, diff, num_files
