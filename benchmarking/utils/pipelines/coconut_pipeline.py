import os
from shutil import rmtree

from utils.pipelines.base_pipeline import BasePipeline


class COCONUTPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            source_name="Coconut",
            filename="coconut.tsv",
            archive_name="coconut-10-2024.csv.zip",
        )
        self.url = "https://coconut.s3.uni-jena.de/prod/downloads/2024-10/coconut-10-2024.csv.zip"

    def download(self, force_download: bool = False) -> None:
        downloaded_file = self._download_single_file_archive(
            url=self.url,
            force_download=force_download,
        )
        if downloaded_file:
            os.rename(
                os.path.join(self.output_dir, "coconut-10-2024.csv"),
                os.path.join(self.output_dir, self.filename),
            )
            rmtree(os.path.join(self.output_dir, "__MACOSX"))
