import numpy as np
from scipy.stats import entropy


def pdf_calculate_entropy(pdf):

    # calculate shannon entropy using the probability density function
    shannon_ent = entropy(pdf, base=2)

    return shannon_ent
