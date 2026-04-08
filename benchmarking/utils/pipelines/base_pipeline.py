import os
from abc import ABC, abstractmethod
from pathlib import Path

from utils.downloading import download_single_file, unpack_archive

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
