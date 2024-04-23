from typing import Tuple

import numpy as np
from scipy.stats import norm
from skimage.filters import gaussian


def generate_intensity_values(mean, std, size: int) -> np.ndarray:
    """
    Generate a NumPy array of the given size with random integer values from a normal distribution
    with the specified mean and standard deviation, within the range [0, 255].
    """
    # calculate the probabilities for each integer value in the range [0, 255]
    x = np.arange(0, 256)
    pdf = norm.pdf(x, loc=mean, scale=std)
    pdf /= pdf.sum()  # normalize the probabilities to sum to 1

    integer_values = np.random.choice(x, size=size, p=pdf)

    return integer_values.astype(np.uint8)


def generate_synthetic_image(
    img_dimensions: Tuple[int, int, int], std, mean: float = 127.0
) -> np.ndarray:
    width: int = img_dimensions[0]
    height: int = img_dimensions[1]
    img_size: int = width * height * 3  # rgb channels

    raw_synthetic_img = generate_intensity_values(mean, std, img_size)
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

    red_channel: np.ndarray = generate_intensity_values(mean[0], std[0], channel_size)
    green_channel: np.ndarray = generate_intensity_values(mean[1], std[1], channel_size)
    blue_channel: np.ndarray = generate_intensity_values(mean[2], std[2], channel_size)

    processed_synthetic_img: np.ndarray = np.stack(
        (red_channel, green_channel, blue_channel), axis=-1, dtype=np.uint8
    )

    processed_synthetic_img = processed_synthetic_img.reshape((width, height, 3))

    return processed_synthetic_img


def generate_gaussian_blur(image):
    gaussian_blurred_image = gaussian(image, sigma=1, mode="constant", cval=0.0)

    return gaussian_blurred_image
