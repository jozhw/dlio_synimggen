import math
from typing import Tuple

import numpy as np
from scipy.stats import norm


def sigmoid(x, x0, k):

    return 1 / (1 + np.exp(-k * (x - x0)))


def cosine_curve(x):

    return (np.cos(np.pi * x / 8) + 1) / 2


def sine_curve(x):
    return np.sin(np.pi * x / 8)


def determine_weights(entropy):

    weight_uniform = sigmoid(entropy, x0=8, k=2)
    weight_singular = sine_curve(entropy) / 2
    # weight_gaussian = cosine_curve(entropy)
    weight_gaussian = 0
    weight_poisson = (
        weight_uniform + weight_singular + weight_gaussian + cosine_curve(entropy)
    )

    total_weight = weight_uniform + weight_singular + weight_poisson + weight_gaussian

    weight_uniform = weight_uniform / total_weight
    # weight_uniform = 0

    weight_singular = weight_singular / total_weight
    # weight_singular = 0

    weight_poisson = weight_poisson / total_weight

    weight_gaussian = weight_gaussian / total_weight

    # if weight_poisson_gaussian + weight_singular + weight_uniform != 1:
    #    raise ValueError("Weight should equal 1.0")

    return weight_uniform, weight_singular, weight_poisson, weight_gaussian


def generate_uniform_intensity_values(size) -> np.ndarray:
    uniform_distribution = np.random.randint(0, 255, size=size, dtype=np.uint8)

    return uniform_distribution


def generate_singular_intensity_values(size, mean):

    return np.full(size, mean, dtype=np.uint8)


def generate_gaussian_intensity_values(mean, std, size: int) -> np.ndarray:
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


def generate_poisson_intensity_values(mean, std, size) -> np.ndarray:
    rng = np.random.default_rng()

    intensity_values = np.clip(rng.poisson(mean, size).astype(np.uint8), 0, 255)

    if len(intensity_values) < size:
        size_diff = size - len(intensity_values)
        array = generate_gaussian_intensity_values(mean, std, size_diff)
        intensity_values = np.concatenate((intensity_values, array))

    return intensity_values


def generate_intensity_values(mean, std, size, entropy):
    uniform_weight, singular_weight, poisson_gaussian_weight, gaussian_weight = (
        determine_weights(entropy)
    )

    poission_size = math.floor(poisson_gaussian_weight * size)

    uniform_size = math.floor(uniform_weight * size)

    gaussian_size = math.floor(gaussian_weight * size)

    singular_size = size - poission_size - uniform_size - gaussian_size

    if poission_size + uniform_size + singular_size + gaussian_size != size:
        raise ValueError("Total size discrepancy")

    poission_dist = generate_poisson_intensity_values(mean, std, poission_size)
    uniform_dist = generate_uniform_intensity_values(uniform_size)
    singular_dist = generate_singular_intensity_values(singular_size, mean)
    gaussian_dist = generate_gaussian_intensity_values(mean, std, gaussian_size)

    dis = np.concatenate((poission_dist, uniform_dist, singular_dist, gaussian_dist))

    return dis


def generate_synthetic_image(
    img_dimensions: Tuple[int, int, int], std, mean: float = 127.0
) -> np.ndarray:
    width: int = img_dimensions[0]
    height: int = img_dimensions[1]
    img_size: int = width * height * 3  # rgb channels

    raw_synthetic_img = generate_gaussian_intensity_values(mean, std, img_size)
    np.random.shuffle(raw_synthetic_img)

    processed_synthetic_img: np.ndarray = np.array(
        raw_synthetic_img, dtype=np.uint8
    ).reshape((width, height, 3))

    return processed_synthetic_img


def generate_adjusted_synthetic_image(
    img_dimensions: Tuple[int, int, int],
    stds: Tuple[float, float, float],
    means: Tuple[float, float, float],
    entropies: Tuple[float, float, float],
) -> np.ndarray:
    width: int = img_dimensions[0]
    height: int = img_dimensions[1]
    channel_size: int = width * height

    red_channel: np.ndarray = generate_intensity_values(
        means[0], stds[0], channel_size, entropies[0]
    )
    green_channel: np.ndarray = generate_intensity_values(
        means[1], stds[0], channel_size, entropies[1]
    )
    blue_channel: np.ndarray = generate_intensity_values(
        means[2], stds[2], channel_size, entropies[2]
    )

    processed_synthetic_img: np.ndarray = np.stack(
        (red_channel, green_channel, blue_channel), axis=-1, dtype=np.uint8
    )

    processed_synthetic_img = processed_synthetic_img.reshape((width, height, 3))

    return processed_synthetic_img
