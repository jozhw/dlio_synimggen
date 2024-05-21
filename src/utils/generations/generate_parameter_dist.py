import random

import numpy as np

# from generate_x_distr import generate_x_values
from scipy.stats import beta, johnsonsb, norm, skewnorm

"""
Given the histogram, looks like a poisson with a cut

Given that poisson generates only integers, to increase accuracy
I will have it scaled by 1000 and then when actually putting in the entropy
will have to divide by 1000
"""


# first must get the total occurrances for each intensity value and entropy
# issue with entropy is that the numbers are particular. One option is
# to round to the nearest tenth.


def calculate_probabilities(occurrences, total):
    # create a list of tuples with value and probability
    probabilities = []
    for value, count in occurrences.items():
        probability = count / total
        probabilities.append((value, probability))

    return probabilities


def get_rand_values(probabilities, n):
    # generate the list of values given n
    values = []
    for _ in range(n):
        # generate a random number between 0 and 1 to be used to determine values
        random_number = random.random()

        # find the value based on the cumulative probabilities compared to random number
        cumulative_probability = 0
        for value, probability in probabilities:
            cumulative_probability += probability
            if random_number <= cumulative_probability:
                values.append(value)
                break

    return values


def generate_x_values(x_occurrences, n):
    # calculate the total number of occurrences
    total_x = sum(x_occurrences.values())

    x_probabilities = calculate_probabilities(x_occurrences, total_x)

    # sort the list based on probabilities in ascending order
    x_probabilities.sort(key=lambda probability: probability[1])

    # generate the list of values given n
    x_values = get_rand_values(x_probabilities, n)

    return x_values


def estimate_beta_params(mu, var, skew):
    """
    Estimate the alpha and beta parameters of the beta distribution
    from the sample mean, variance, and skewness.
    """
    a = (mu * (mu * (1 - mu) / var - 1) - skew) / (skew - 2 * mu * (1 - mu) / var)
    b = (1 - mu) * (mu * (1 - mu) / var - 1 - skew) / (skew - 2 * mu * (1 - mu) / var)
    a = max(a, 0.001)
    b = max(b, 0.001)
    return a, b


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


def generate_x_skew_dist(n, x, std, skew, lowerbound, upperbound):

    intensity_values = np.clip(
        skewnorm.rvs(a=skew, loc=x, scale=std, size=n), lowerbound, upperbound
    )

    intensity_values = np.array(intensity_values).astype(np.uint16)

    return intensity_values


def generate_x_beta_dist(n, a, b):

    values = beta.rvs(a, b, size=n)

    return values


def generate_x_triangular_dist(n, mean, lowerbound, upperbound):
    x_array = np.random.triangular(lowerbound, mean, upperbound, n).astype(np.uint16)

    return x_array


def generate_x_johnson_dist(n, mean, variance, skew):
    """
    Generate a distribution based on the given mean, variance, and skew.
    Returns an array of n random variates from the distribution.
    """
    # Calculate parameters for Johnson's SU distribution
    norm_mean, norm_std = norm.stats(0, 1)
    norm_samples = norm.rvs(size=10000, loc=norm_mean, scale=norm_std)
    alpha, beta, loc, scale = johnsonsb.fit(norm_samples)

    # Adjust parameters to match the desired moments
    johnsonsb_mean, johnsonsb_variance, johnsonsb_skew = johnsonsb.stats(
        alpha, beta, loc=loc, scale=scale, moments="mvs"
    )
    scale_factor = np.sqrt(variance / johnsonsb_variance)
    loc_adjust = mean - johnsonsb_mean * scale_factor
    scale *= scale_factor

    # Adjust alpha and beta to match the desired skew
    skew_diff = skew - johnsonsb_skew
    beta_new = beta + skew_diff * np.sqrt(johnsonsb_variance) / scale
    alpha_new = alpha + skew_diff * mean / (scale * np.sqrt(johnsonsb_variance))

    # Generate random variates from the Johnson's SU distribution
    return johnsonsb.rvs(alpha_new, beta_new, loc=loc + loc_adjust, scale=scale, size=n)


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


def generate_parameter_dist(n, entropy, x_occurances, intensity=127):

    values = []

    entropies = np.array(generate_entropy_dist(n, entropy))
    np.random.shuffle(entropies)

    # x_sizes = generate_x_dist(n, x_size, x_std, x_skew, x_lowerbound, x_upperbound)
    x_sizes = generate_x_values(x_occurances, n)

    # x_sizes = np.array(generate_x_dist(n, x_size))
    # np.random.shuffle(x_sizes)

    intensities = np.array(generate_intensity_dist(n, intensity))
    np.random.shuffle(intensities)

    for i in range(n):
        values.append((entropies[i], x_sizes[i], intensity))

    return values


if __name__ == "__main__":
    n = 10
    entropy = 7.5
    intensity = 127
    x_occurances = {10: 5, 200: 1, 100: 1}

    print(
        generate_parameter_dist(
            n,
            entropy,
            x_occurances,
            intensity=intensity,
        )
    )
