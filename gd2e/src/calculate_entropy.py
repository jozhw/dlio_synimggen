from scipy.stats import entropy, norm


def calculate_entropy(pd, mean: int, std: int):
    # gen probability density function
    pdf = norm.pdf(pd, mean, std)

    # calculate shannon entropy using the probability density function
    shannon_ent = entropy(pdf, base=2)

    return shannon_ent
