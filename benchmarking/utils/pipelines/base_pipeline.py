import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import polars as pl
from skfp.utils import run_in_parallel

from utils.downloading import download_single_file, unpack_archive
from utils.filtering import feasibility_filter_batch
from utils.standardization import inchi_to_inchi_standardize

INPUTS_DIR = "inputs"
OUTPUTS_DIR = Path("benchmark_times") / "benchmark_times_saved"


class BasePipeline(ABC):
    def __init__(
        self,
        source_name: str,
        filename: str,
        verbose: int | bool = False,
        input_dir: str = INPUTS_DIR,
        output_dir: str = OUTPUTS_DIR,
        archive_name: str | None = None,
    ):
        self.source_name = source_name
        self.filename = filename
        self.preprocessed_filename = f"{self.source_name.lower()}_preprocessed.parquet"
        self.standardized_filename = f"{self.source_name.lower()}_standardized.parquet"
        self.filtered_filename = f"{self.source_name.lower()}_filtered.parquet"
        self.verbose = verbose
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.archive_name = archive_name

    def process(self) -> None:
        print("Downloading dataset")
        self.download()
        print("Finished downloading")

    @abstractmethod
    def download(self, force_download: bool = False) -> None:
        """
        Download the dataset file(s).

        Skips download if resulting file already exists, unless force_download
        is set to True.
        """

    def _download_single_file_archive(
        self,
        url: str,
        force_download: bool,
    ) -> bool:
        """
        Download single archive file, unpack it and remove the archive.

        :return: True if the file was downloaded or it was forced to download it, False if it was already present
        """
        if self._check_skip_download(force_download):
            print(f"Found existing file {self.filename}, skipping download")
            return False

        print(f"Downloading from {url} to {self.output_dir}")
        output_path = download_single_file(
            url,
            output_dir=self.output_dir,
            output_file=self.archive_name,
        )
        print("Download finished, unpacking dataset")
        unpack_archive(output_path)
        print("Unpacked dataset")

        os.remove(os.path.join(self.output_dir, self.archive_name))

        return True

    def _check_skip_download(self, force_download: bool) -> bool:
        return not force_download and os.path.exists(
            os.path.join(self.output_dir, self.filename)
        )

    @abstractmethod
    def preprocess(self) -> None:
        """
        Initial preprocessing of the dataset. Transforms it into parquet files
        with columns: id, InChI. Column `id` is the original identifier of the
        molecule per dataset.
        """

    def standardize(self) -> None:
        """
        Standardize molecules in the dataset. Performs the cleanup, standardization
        and canonicalization of InChI, and removes the duplicates.
        The output file consists of columns: id, InChI.
        """
        input_file_path = os.path.join(self.output_dir, self.preprocessed_filename)
        output_file_path = os.path.join(self.output_dir, self.standardized_filename)

        # check if standardized file already exists
        if os.path.exists(output_file_path):
            print("Found standardized dataset, skipping")
            return

        df = pl.read_parquet(input_file_path)

        inchis = run_in_parallel(
            inchi_to_inchi_standardize,
            data=df["InChI"],
            n_jobs=-1,
            batch_size=1000,
            flatten_results=True,
            verbose=True,
        )
        df = df.with_columns(pl.Series("InChI", values=inchis))

        df = df.filter(pl.col("InChI").is_not_null())
        df = df.unique("InChI")

        df.write_parquet(output_file_path)

    def filter(self) -> None:
        """
        Filter molecules in the dataset. Performs feasibility filtering, removing
        compounds that are simply nonsensical physically or that cannot be
        interpreted as small molecules.
        Input and output files have columns: id, InChI.
        """
        input_file_path = os.path.join(self.output_dir, self.standardized_filename)
        output_file_path = os.path.join(self.output_dir, self.filtered_filename)

        # check if standardized file already exists
        if os.path.exists(output_file_path):
            print("Found filtered dataset, skipping")
            return

        df = pl.read_parquet(input_file_path)

        pass_filter = run_in_parallel(
            feasibility_filter_batch,
            data=df["InChI"],
            n_jobs=-1,
            batch_size=1000,
            verbose=True,
        )
        pass_filter = np.concatenate(pass_filter)

        initial_length = len(df)
        df = df.filter(pass_filter)
        filtered_length = len(df)

        print(f"Filtering reduced molecules from {initial_length} to {filtered_length}")

        df.write_parquet(output_file_path)
