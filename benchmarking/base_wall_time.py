"""
Benchmark wall time for computing ECFP fingerprints on a dataset using scikit-fingerprints.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import polars as pl
from skfp.fingerprints import ECFPFingerprint

FP_SIZE = 2048
FP_RADIUS = 2

OUTPUTS_DIR = Path("benchmark_times") / "benchmark_times_saved"

MCULE_URL = "https://dl.mcule.com/database/mcule_purchasable_full_260129.smi.gz"
COCONUT_URL = (
    "https://coconut.s3.uni-jena.de/prod/downloads/2024-10/coconut-10-2024.csv.zip"
)

MCULE_GZ = os.path.join(OUTPUTS_DIR, "mcule.smi.gz")
COCONUT_ZIP = os.path.join(OUTPUTS_DIR, "coconut-10-2024.csv.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["mcule", "coconut"],
        required=True,
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Number of rows to load from the dataset. Loads all rows if not specified.",
    )
    args = parser.parse_args()
    run_benchmark(args.dataset, args.n_rows)


def run_benchmark(
    dataset_name: str,
    n_rows: int | None = None,
) -> None:
    """
    Run a benchmark for ECFP fingerprint computation on a selected dataset.

    Steps:
    1. Select the dataset (MCULE or COCONUT) and download it if necessary.
    2. Load SMILES strings from the dataset (optionally limited by n_rows).
    3. Compute ECFP fingerprints using scikit-fingerprints.
    4. Measure total wall time for fingerprint generation.
    5. Report the size of the resulting fingerprint matrix in MB.
    """
    if dataset_name == "mcule":
        dataset_path = MCULE_GZ
        smiles_column = "SMILES"
        csv_args = {
            "has_header": False,
            "separator": "\t",
            "new_columns": ["SMILES", "id"],
        }
        if not os.path.exists(MCULE_GZ):
            _download_file(MCULE_URL, MCULE_GZ)
    elif dataset_name == "coconut":
        dataset_path = COCONUT_ZIP
        smiles_column = "canonical_smiles"
        csv_args = {"columns": [smiles_column]}
        if not os.path.exists(COCONUT_ZIP):
            _download_file(COCONUT_URL, COCONUT_ZIP)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if n_rows is not None:
        csv_args["n_rows"] = n_rows

    df = pl.read_csv(dataset_path, **csv_args)
    smiles_list = df[smiles_column].to_list()
    print(f"Loaded {len(smiles_list)} molecules from {dataset_path}")

    ecfp = ECFPFingerprint(fp_size=FP_SIZE, radius=FP_RADIUS, n_jobs=-1)

    print("Computing ECFP fingerprints with scikit-fingerprints...")
    start = time.perf_counter()
    fps = ecfp.transform(smiles_list)
    elapsed = time.perf_counter() - start

    size_in_mb = fps.nbytes / (1024**2)
    print(f"Finished. Fingerprint matrix shape: {fps.shape}")
    print(f"Fingerprint matrix size: {size_in_mb:.4f} MB")
    print(f"Wall time: {elapsed:.2f} s")


def _download_file(url: str, output_path: str) -> None:
    subprocess.run(["wget", "-c", "-O", os.path.abspath(output_path), url], check=True)


if __name__ == "__main__":
    main()
