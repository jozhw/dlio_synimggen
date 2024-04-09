import numpy as np
from scipy.stats import norm


def generate_pdf(pdx=np.arange(256), mean: float = 127.0, std: float = 30.0):
    # gen probability density function
    pdf = norm.pdf(pdx, mean, std)

    return pdf
