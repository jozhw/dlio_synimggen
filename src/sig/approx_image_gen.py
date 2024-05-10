import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI

from utils.calculations.calculate_img_data import calculate_img_data
from utils.calculations.entropy_calculate_std import entropy_calculate_std
from utils.generations.generate_csv import generate_results_csv
from utils.generations.generate_occurrences_json import generate_occurrences_json
from utils.generations.generate_save_paths import (
    generate_compressed_img_save_paths,
    generate_save_result_data_path,
)
from utils.generations.generate_synthetic_image import (
    generate_adjusted_synthetic_image,
    generate_approx_synthetic_image,
    generate_synthetic_image,
)
from utils.generations.generate_synthetic_image_from_prob import generate_values
from utils.setting.set_date import set_date


class ApproxImageGen:
    ACCEPTED_FILE_TYPES: List[str] = ["npz", "jpg"]

    def __init__(
        self,
        imgs_data: str,
        compressed_file_types: List[str],
        img_source: str = "synthetic",
    ) -> None:
        self.df_imgs: pd.DataFrame = pd.read_csv(imgs_data)
        self.compressed_file_types: List[str] = compressed_file_types
        self.img_source: str = img_source
        self.pdx: np.ndarray = np.arange(256)

        self.regex_date_pattern: str = r"\b\d{4}-\d{2}-\d{2}\b"
        self.date: str = set_date(imgs_data)

        self.compressed_img_save_paths: Dict[str, str] = (
            generate_compressed_img_save_paths(
                ApproxImageGen.ACCEPTED_FILE_TYPES,
                self.compressed_file_types,
                self.img_source,
                self.date,
            )
        )
        self.save_result_data_path: str = generate_save_result_data_path(
            self.img_source, self.date
        )

    """
    gsi = generate synthetic img data
    """

    def gsid(self, occurrences_data, n) -> None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Broadcast the DataFrame to all processes
        if rank == 0:
            with open(occurrences_data, "r") as file:
                loaded_data = json.load(file)
                entropies = loaded_data["entropies"][0]
                means = loaded_data["means"][0]
                dimensions = {
                    tuple(int(x) for x in key[1:-1].split(", ")): value
                    for key, value in loaded_data["dimensions"][0].items()
                }
                values = generate_values(entropies, means, dimensions, n)

        else:
            values = None
        self.values = comm.bcast(values, root=0)

        num_rows = len(self.df_imgs)
        rows_per_process = num_rows // size
        start = rank * rows_per_process
        end = start + rows_per_process if rank != size - 1 else num_rows

        results: List[Dict[str, Any]] = []

        for index, item in enumerate(self.values[start:end], start=start):

            entropy, mean, dim = item

            # synthetics
            syn_file_name: str = "synthetic_" + str(index)

            # generate the synthetic image
            synthetic_img: np.ndarray = generate_approx_synthetic_image(
                float(entropy), int(mean), dim
            )

            dimensions = synthetic_img.shape

            syn_calculations: Dict[str, Any] = calculate_img_data(
                syn_file_name,
                synthetic_img,
                dimensions,
                self.compressed_file_types,
                self.compressed_img_save_paths,
            )

            syn_calculations["entropy_used"] = entropy
            syn_calculations["mean_used"] = mean
            syn_calculations["dim_used"] = dim

            results.append(syn_calculations)

        gathered_results = comm.gather(results, root=0)

        if rank == 0:
            if gathered_results is not None:

                flat_results = [
                    result for sublist in gathered_results for result in sublist
                ]
                generate_results_csv(
                    flat_results, self.img_source, "approx_synthetic_imgs_results"
                )

    def get_occurance_data(self, fname):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Broadcast the DataFrame to all processes
        if rank == 0:
            df_imgs_to_broadcast = self.df_imgs
        else:
            df_imgs_to_broadcast = None
        self.df_imgs = comm.bcast(df_imgs_to_broadcast, root=0)

        num_rows = len(self.df_imgs)
        rows_per_process = num_rows // size
        start = rank * rows_per_process
        end = start + rows_per_process if rank != size - 1 else num_rows

        entropies: Dict[float, int] = {}
        means: Dict[int, int] = {}
        dimensions: Dict[str, int] = {}

        for _, row in self.df_imgs.iloc[start:end].iterrows():
            original_entropy: float = row.at["entropy"]
            mean: int = int(row.at["mean_intensity_value"])
            original_size = int(row.at["uncompressed_size"])
            original_width: int = int(row.at["uncompressed_width"])
            original_height: int = int(row.at["uncompressed_height"])
            dimension: Tuple[int, int, int] = (original_width, original_height, 3)
            dim = str(dimension)

            if original_entropy in entropies:
                entropies[original_entropy] += 1
            else:
                entropies[original_entropy] = 1

            if mean in means:
                means[mean] += 1
            else:
                means[mean] = 1

            if dim in dimensions:
                dimensions[dim] += 1
            else:
                dimensions[dim] = 1

        gathered_entropies = comm.gather(entropies, root=0)
        gathered_means = comm.gather(means, root=0)
        gathered_dims = comm.gather(dimensions, root=0)

        if rank == 0:
            if (
                gathered_entropies is not None
                and gathered_means is not None
                and gathered_dims is not None
            ):

                generate_occurrences_json(
                    gathered_entropies, gathered_means, gathered_dims, fname=fname
                )
