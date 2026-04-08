from utils.pipelines.base_pipeline import BasePipeline


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
