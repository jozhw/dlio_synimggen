import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from mpi4py import MPI

from utils.calculations.calculate_img_data import calculate_img_data
from utils.generations.generate_csv import generate_results_csv
from utils.generations.generate_parameter_dist import generate_parameter_dist
from utils.generations.generate_save_paths import (
    generate_compressed_img_save_paths,
    generate_save_result_data_path,
)
from utils.generations.generate_synthetic_image import generate_distr_synthetic_image
from utils.setting.set_date import set_date


class DistrImageGen:
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
                DistrImageGen.ACCEPTED_FILE_TYPES,
                self.compressed_file_types,
                self.img_source,
                self.date,
            )
        )
        self.save_result_data_path: str = generate_save_result_data_path(
            self.img_source, self.date
        )

    def get_x_occurances(self):

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

        x_occurances: Dict[int, int] = {}

        for _, row in self.df_imgs.iloc[start:end].iterrows():
            original_size = int(row.at["uncompressed_size"])
            x = int(np.sqrt(original_size / 3))

            if x in x_occurances:
                x_occurances[x] += 1
            else:
                x_occurances[x] = 1

        gathered_x = comm.gather(x_occurances, root=0)

        return gathered_x

    """
    gsi = generate synthetic img data
    """

    # instead of the skew, lowerbound, and upperbound, will use occurances instead

    def gsid(self, n, entropy, intensity=127) -> None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Broadcast the DataFrame to all processes
        if rank == 0:
            x_occurances = {}
            for _, row in self.df_imgs.iterrows():
                original_size = int(row.at["uncompressed_size"])
                x = int(np.sqrt(original_size / 3))

                if x in x_occurances:
                    x_occurances[x] += 1
                else:
                    x_occurances[x] = 1

            values = generate_parameter_dist(n, entropy, x_occurances, intensity)

        else:
            values = None
        self.values = comm.bcast(values, root=0)

        num_rows = len(self.df_imgs)
        rows_per_process = num_rows // size
        start = rank * rows_per_process
        end = start + rows_per_process if rank != size - 1 else num_rows

        results: List[Dict[str, Any]] = []

        for index, item in enumerate(self.values[start:end], start=start):

            entropy, x, mean = item

            # synthetics
            syn_file_name: str = "synthetic_" + str(index)

            # generate the synthetic image
            synthetic_img: np.ndarray = generate_distr_synthetic_image(
                float(entropy), int(mean), x
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
            syn_calculations["dim_used"] = x

            print(syn_calculations)

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
