import os
import sys

import numpy as np

# Add the project's root directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from entropy_pdf_analyzer import entropy_pdf_analyzer


def main(target_entropies):

    entropy_pdf_analyzer(target_entropies)


if __name__ == "__main__":

    target_entropies = np.arange(0.1, 8.00, 0.001)

    main(target_entropies)
