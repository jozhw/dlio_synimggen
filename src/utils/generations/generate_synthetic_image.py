from typing import Tuple

import numpy as np
from scipy.stats import norm


def generate_synthetic_image(
    img_dimensions: Tuple[int, int, int], std, mean: float = 127.0
) -> np.ndarray:
    width: int = img_dimensions[0]
    height: int = img_dimensions[1]
    img_size: int = width * height * 3  # rgb channels

    raw_synthetic_img = norm.rvs(loc=mean, scale=std, size=img_size)

    np.random.shuffle(raw_synthetic_img)

    processed_synthetic_img: np.ndarray = np.array(
        raw_synthetic_img, dtype=np.uint8
    ).reshape((width, height, 3))

    return processed_synthetic_img


def generate_adjusted_synthetic_image(
    img_dimensions: Tuple[int, int, int],
    std: Tuple[float, float, float],
    mean: Tuple[float, float, float],
) -> np.ndarray:
    width: int = img_dimensions[0]
    height: int = img_dimensions[1]
    channel_size: int = width * height

    red_channel = np.array(norm.rvs(loc=mean[0], scale=std[0], size=channel_size))
    green_channel = np.array(norm.rvs(loc=mean[1], scale=std[1], size=channel_size))
    blue_channel = np.array(norm.rvs(loc=mean[2], scale=std[2], size=channel_size))

    processed_synthetic_img: np.ndarray = np.stack(
        (red_channel, green_channel, blue_channel), axis=-1, dtype=np.uint8
    )
    return processed_synthetic_img
