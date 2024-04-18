from typing import List

import numpy as np


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
