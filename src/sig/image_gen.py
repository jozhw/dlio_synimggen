import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI

from utils.calculations.calculate_img_data import calculate_img_data
from utils.calculations.entropy_calculate_std import entropy_calculate_std
from utils.generations.generate_csv import generate_results_csv
from utils.generations.generate_save_paths import (
    generate_compressed_img_save_paths,
    generate_save_result_data_path,
)
from utils.generations.generate_synthetic_image import (
    generate_adjusted_synthetic_image,
    generate_gaussian_blur,
    generate_synthetic_image,
)
from utils.setting.set_date import set_date


class ImageGen:
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
                ImageGen.ACCEPTED_FILE_TYPES,
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

    def gsid(self) -> None:
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

        results: List[Dict[str, Any]] = []
        for _, row in self.df_imgs.iloc[start:end].iterrows():
            file_name: str = row.at["file_name"]
            original_entropy: float = row.at["entropy"]
            original_npz_compressed_image_size: int = row.at[
                "npz_compressed_image_size"
            ]
            original_width: int = int(row.at["uncompressed_width"])
            original_height: int = int(row.at["uncompressed_height"])
            dimensions: Tuple[int, int, int] = (original_width, original_height, 3)
            mean: int = int(row.at["mean_intensity_value"])

            red_channel_mean: int = int(row.at["red_mean_intensity_value"])
            red_channel_entropy: float = row.at["red_entropy_intensity_value"]

            green_channel_mean: int = int(row.at["green_mean_intensity_value"])
            green_channel_entropy: float = row.at["green_entropy_intensity_value"]

            blue_channel_mean: int = int(row.at["blue_mean_intensity_value"])
            blue_channel_entropy: float = row.at["blue_entropy_intensity_value"]

            channel_entropies: Tuple[float, float, float] = (
                red_channel_entropy,
                green_channel_entropy,
                blue_channel_entropy,
            )

            channel_means: Tuple[int, int, int] = (
                red_channel_mean,
                green_channel_mean,
                blue_channel_mean,
            )

            # synthetics
            syn_file_name: str = "synthetic_" + str(file_name)

            # this will only calculate using the overall entropy
            # calculated_std: float = entropy_calculate_std(original_entropy)
            # synthetic_img: np.ndarray = generate_synthetic_image(
            #    dimensions, calculated_std, mean=mean
            # )

            red_std: float = entropy_calculate_std(red_channel_entropy)
            green_std: float = entropy_calculate_std(green_channel_entropy)
            blue_std: float = entropy_calculate_std(blue_channel_entropy)

            calculated_stds: Tuple[float, float, float] = (red_std, green_std, blue_std)

            # generate the synthetic image
            synthetic_img: np.ndarray = generate_adjusted_synthetic_image(
                dimensions, calculated_stds, channel_means, channel_entropies
            )

            syn_calculations: Dict[str, Any] = calculate_img_data(
                syn_file_name,
                synthetic_img,
                dimensions,
                self.compressed_file_types,
                self.compressed_img_save_paths,
            )

            # filter df_images for the current image

            filtered_df: pd.DataFrame = self.df_imgs.loc[
                self.df_imgs["file_name"] == file_name
            ]
            if filtered_df.empty:
                raise ValueError(f"The file {file_name} does not exist.")

            # ratio between syn/orig compressed size

            npz_compressed_synthetic_original_ratio: float = (
                syn_calculations["npz_compressed_image_size"]
                / original_npz_compressed_image_size
            )
            syn_calculations["npz_compressed_synthetic_original_ratio"] = (
                npz_compressed_synthetic_original_ratio
            )

            syn_calculations["mean_used"] = mean

            syn_calculations["original_npz_compressed_image_size"] = (
                original_npz_compressed_image_size
            )

            results.append(syn_calculations)

        gathered_results = comm.gather(results, root=0)

        if rank == 0:
            if gathered_results is not None:
                flat_results = [
                    result for sublist in gathered_results for result in sublist
                ]
                generate_results_csv(
                    flat_results, self.img_source, "synthetic_imgs_results"
                )
