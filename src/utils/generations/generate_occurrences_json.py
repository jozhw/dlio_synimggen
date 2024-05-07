import json
import os


def generate_occurrences_json(entropy_data, mean_data, dim_data, fname="data.json"):

    file_path = os.path.join("data", fname)

    data = {
        "entropies": entropy_data,
        "means": mean_data,
        "dimensions": dim_data,
    }

    with open(file_path, "w") as file:

        json.dump(data, file, indent=4)
