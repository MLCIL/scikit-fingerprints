"""
Benchmark wall time for computing ECFP fingerprints on a dataset using scikit-fingerprints.
"""

import argparse
import os
import time
from pathlib import Path

import polars as pl
from skfp.fingerprints import ECFPFingerprint

from utils.pipelines.coconut_pipeline import COCONUTPipeline
from utils.pipelines.mcule_pipeline import MculePipeline

FP_SIZE = 2048
FP_RADIUS = 2

INPUTS_DIR = "inputs"
OUTPUTS_DIR = Path("benchmark_times") / "benchmark_times_saved"

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
        pipeline, dataset_path = MculePipeline(), MCULE_TSV
        smiles_column = "SMILES"
        csv_args = {
            "has_header": False,
            "separator": "\t",
            "new_columns": ["SMILES", "id"],
        }
    elif dataset_name == "coconut":
        pipeline, dataset_path = COCONUTPipeline(), COCONUT_TSV
        smiles_column = "canonical_smiles"
        csv_args = {"columns": [smiles_column]}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    pipeline.download()

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


if __name__ == "__main__":
    main()
