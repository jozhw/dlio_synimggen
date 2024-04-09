from utils.calculations.pdf_calculate_entropy import pdf_calculate_entropy
from utils.generations.generate_pdf import generate_pdf


def entropy_calculate_std(
    target_entropy,
    lower_bound: float = 0.01,
    upper_bound: float = 5000.0,
    tolerance: float = 1e-6,
):
    if target_entropy > 8:
        raise ValueError("RGB image entropy cannot be greater than 8.")

    while abs(upper_bound - lower_bound) > tolerance:
        mid_std = (lower_bound + upper_bound) / 2
        pdf = generate_pdf(std=mid_std)
        mid_entropy = pdf_calculate_entropy(pdf)

        if mid_entropy < target_entropy:
            lower_bound = mid_std
        else:
            upper_bound = mid_std

    return (lower_bound + upper_bound) / 2
