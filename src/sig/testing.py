import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import johnsonsb, johnsonsu, kurtosis, skew, yeojohnson


def fit_johnson_distribution(data, dist_type="SU"):
    if dist_type not in ["SU", "SB"]:
        raise ValueError("dist_type must be 'SU' or 'SB'")

    if dist_type == "SU":
        distribution = johnsonsu
    elif dist_type == "SB":
        distribution = johnsonsb
    else:
        distribution = yeojohnson

    # Fit the distribution to the data
    params = distribution.fit(data)

    # Extract parameters
    gamma, eta, loc, scale = params

    return params, distribution


def plot_distribution(data, params, distribution):
    # Plot the histogram of the data
    plt.hist(data, bins=30, density=True, alpha=0.6, color="g")

    # Generate x values
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    # Generate the PDF of the fitted distribution
    p = distribution.pdf(x, *params)

    # Plot the PDF
    plt.plot(x, p, "k", linewidth=2)
    title = "Fit results: gamma = %.2f, eta = %.2f, loc = %.2f, scale = %.2f" % (
        params[0],
        params[1],
        params[2],
        params[3],
    )
    plt.title(title)

    plt.show()


# Example usage
data = "results/polaris/2024-04-26/results_imagenet_rand_300000.csv"
df = pd.read_csv(data)
size_array = np.array(df["uncompressed_size"].values)
x_array = np.sqrt(size_array / 3).astype(np.uint16)
print(skew(x_array))
print(kurtosis(x_array))
# Fit and plot Johnson SU distribution
params_su, dist_su = fit_johnson_distribution(x_array, "SU")
plot_distribution(x_array, params_su, dist_su)

# Fit and plot Johnson SB distribution
params_sb, dist_sb = fit_johnson_distribution(x_array, "SB")
plot_distribution(x_array, params_sb, dist_sb)
