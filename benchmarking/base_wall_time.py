"""
Benchmark wall time for computing ECFP fingerprints on a dataset using scikit-fingerprints.
"""

import argparse
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from shutil import rmtree

import polars as pl
from rapidgzip import RapidgzipFile
from skfp.fingerprints import ECFPFingerprint

FP_SIZE = 2048
FP_RADIUS = 2

OUTPUTS_DIR = Path("benchmark_times") / "benchmark_times_saved"

MCULE_URL = "https://dl.mcule.com/database/mcule_purchasable_full_260129.smi.gz"
COCONUT_URL = (
    "https://coconut.s3.uni-jena.de/prod/downloads/2024-10/coconut-10-2024.csv.zip"
)

MCULE_TSV = os.path.join(OUTPUTS_DIR, "mcule.tsv")
COCONUT_TSV = os.path.join(OUTPUTS_DIR, "coconut.tsv")

UNITS = {0: "B", 1: "KB", 2: "MB", 3: "GB"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["mcule", "coconut"],
        required=True,
    )
    parser.add_argument(
        "--size-exp",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="Exponent for size unit: 0=B, 1=KB, 2=MB, 3=GB",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Number of rows to load from the dataset. Loads all rows if not specified.",
    )
    args = parser.parse_args()
    run_benchmark(args.dataset, args.size_exp, args.n_rows)


def run_benchmark(
    dataset_name: str,
    size_exp: int = 2,
    n_rows: int | None = None,
) -> None:
    """
    Run a benchmark for ECFP fingerprint computation on a selected dataset.

    Steps:
    1. Select the dataset (MCULE or COCONUT) and download it if necessary.
    2. Load SMILES strings from the dataset (optionally limited by n_rows).
    3. Compute ECFP fingerprints using scikit-fingerprints.
    4. Measure total wall time for fingerprint generation.
    5. Report the size of the resulting fingerprint matrix in the selected unit.
    """
    if size_exp not in UNITS:
        raise ValueError(f"Invalid size exponent: {size_exp}. Must be 0 (B) to 3 (GB).")

    if dataset_name == "mcule":
        dataset_path = MCULE_TSV
        smiles_column = "SMILES"
        csv_args = {
            "has_header": False,
            "separator": "\t",
            "new_columns": ["SMILES", "id"],
        }
        _download_mcule()
    elif dataset_name == "coconut":
        dataset_path = COCONUT_TSV
        smiles_column = "canonical_smiles"
        csv_args = {"columns": [smiles_column]}
        _download_coconut()
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

    size_in_unit = fps.nbytes / (1024**size_exp)
    print(f"Finished. Fingerprint matrix shape: {fps.shape}")
    print(f"Fingerprint matrix size: {size_in_unit:.4f} {UNITS[size_exp]}")
    print(f"Wall time: {elapsed:.2f} s")


def _download_mcule() -> None:
    if os.path.exists(MCULE_TSV):
        return
    archive_path = str(MCULE_TSV) + ".gz"
    _download_file(MCULE_URL, archive_path)
    with RapidgzipFile(archive_path) as f_in, open(MCULE_TSV, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(archive_path)


def _download_coconut() -> None:
    if os.path.exists(COCONUT_TSV):
        return
    archive_path = os.path.join(OUTPUTS_DIR, "coconut-10-2024.csv.zip")
    _download_file(COCONUT_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(OUTPUTS_DIR)
    os.rename(os.path.join(OUTPUTS_DIR, "coconut-10-2024.csv"), COCONUT_TSV)
    rmtree(os.path.join(OUTPUTS_DIR, "__MACOSX"), ignore_errors=True)
    os.remove(archive_path)


def _download_file(url: str, output_path: str) -> None:
    subprocess.run(["wget", "-c", "-O", os.path.abspath(output_path), url], check=True)


if __name__ == "__main__":
    main()
