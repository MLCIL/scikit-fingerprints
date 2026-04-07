import os

import polars as pl
from skfp.utils import run_in_parallel

from utils.pipelines.base_pipeline import BasePipeline
from utils.standardization import smiles_to_inchi_convert


class MculePipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            source_name="Mcule",
            filename="mcule.tsv",
            archive_name="mcule.tsv.gz",
        )
        self.url = "https://dl.mcule.com/database/mcule_purchasable_full_260129.smi.gz"

    def download(self, force_download: bool = False) -> None:
        """
        Mcule output: single TSV file, without headers, but the columns are
        SMILES string and Mcule ID.
        """
        super()._download_single_file_archive(
            url=self.url,
            force_download=force_download,
        )

    def preprocess(self) -> None:
        input_file_path = os.path.join(self.output_dir, self.filename)
        output_file_path = os.path.join(self.output_dir, self.preprocessed_filename)

        # check if preprocessed file already exists
        if os.path.exists(output_file_path):
            print("Found preprocessed dataset, skipping")
            return

        df = pl.read_csv(
            input_file_path,
            has_header=False,
            separator="\t",
            new_columns=["SMILES", "id"],
        )
        df = df.select(["id", "SMILES"])

        inchis = run_in_parallel(
            smiles_to_inchi_convert,
            data=df["SMILES"],
            n_jobs=-1,
            batch_size=1000,
            flatten_results=True,
            verbose=True,
        )
        df = df.with_columns(pl.Series("SMILES", values=inchis))
        df = df.rename({"SMILES": "InChI"})

        df = df.filter(pl.col("InChI").is_not_null())
        df = df.unique("InChI")

        df.write_parquet(output_file_path)
