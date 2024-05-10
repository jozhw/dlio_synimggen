import numpy as np
from scipy.stats import norm

"""
Given the histogram, looks like a poisson with a cut

Given that poisson generates only integers, to increase accuracy
I will have it scaled by 1000 and then when actually putting in the entropy
will have to divide by 1000
"""


def generate_entropy_dist(n: int, mean_entropy: float):

    scale = 1000

    mean_entropy = int(mean_entropy * scale)

    rng = np.random.default_rng()

    intensity_values = np.clip(rng.poisson(mean_entropy, n), 0, 8 * scale)

    intensity_values = intensity_values / scale

    return intensity_values


"""
Looks like a uniform with a given range with an added poisson for the mean value

What if i just used a poisson?
"""


def generate_x_dist(n, mean_x):
    rng = np.random.default_rng()

    intensity_values = rng.poisson(mean_x, n).astype(np.uint16)

    return intensity_values


"""
Given the histogram, looks like a gaussian with std of 30 and a mean of 127
"""


def generate_intensity_dist(
    n,
    mean_intensity_value: int = 127,
    std=30,
):

    # calculate the probabilities for each integer value in the range [0, 255]
    x = np.arange(0, 256)
    pdf = norm.pdf(x, loc=mean_intensity_value, scale=std)
    pdf /= pdf.sum()  # normalize the probabilities to sum to 1

    integer_values = np.random.choice(x, size=n, p=pdf)

    return integer_values.astype(np.uint8)


"""
Acts as a wrapper for the entropy, intensity, and x_size parameter distribution generation

All of these distributions are for random pulling to get each of the entropy, intensity, and x_size
parameters for the synthetic image generation

Returns in the following array in the following format of tuples (entropy, x_size, intensity)
"""


def generate_parameter_dist(n, entropy, x_size, intensity=127):

    values = []

    entropies = np.array(generate_entropy_dist(n, entropy))
    np.random.shuffle(entropies)

    x_sizes = np.array(generate_x_dist(n, x_size))
    np.random.shuffle(x_sizes)

    intensities = np.array(generate_intensity_dist(n, intensity))
    np.random.shuffle(intensities)

    for i in range(n):
        values.append((entropies[i], x_sizes[i], intensity))

    return values


if __name__ == "__main__":
    n = 10
    entropy = 7.5
    intensity = 127
    x_size = 400

    print(generate_parameter_dist(n, entropy, x_size, intensity))
