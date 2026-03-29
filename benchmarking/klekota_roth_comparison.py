"""
Klekota-Roth scikit-fingerprints vs CDK benchmark.
"""

import csv
import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
from benchmarking.naive_fp_klekota_roth import NaiveSkfpKlekotaRothFingerprint
from benchmarking.utils.timing import measure_time
from CDK_pywrapper import CDK, FPType
from matplotlib import pyplot as plt
from rdkit import Chem
from skfp.datasets.moleculenet import load_hiv
from skfp.fingerprints import KlekotaRothFingerprint

N_REPEATS = 5
STEP = 100  # Step for increasing dataset size
DATASET_CUTOFF = 1000  # Maximum number of molecules to benchmark

NUM_THREADS = joblib.effective_n_jobs(n_jobs=-1)  # Use all available CPU cores

OUTPUTS_DIR = Path("benchmark_times") / "benchmark_times_saved"
PLOTS_DIR = Path("benchmark_times") / "benchmark_times_plotted"

USE_PDF = True  # If True, save plot as PDF, otherwise save as PNG
USE_ERROR_BARS = False  # If True, use error bars instead of shaded fill_between

CSV_FILENAME = "skfp_cdk_kr_timings"
PLOT_FILENAME = "skfp_cdk_kr_timings"

file_ext = ".pdf" if USE_PDF else ".png"

RESULT_CSV_PATH = OUTPUTS_DIR / f"{CSV_FILENAME}.csv"
RESULT_PLOT_PATH = PLOTS_DIR / f"{PLOT_FILENAME}{file_ext}"


def main():
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    run_benchmark()
    plot_results()


def skfp_kr(smiles_list: list[str]) -> None:
    """
    Klekota-Roth fingerprint (optimized scikit-fingerprints implementation)
    """
    KlekotaRothFingerprint(n_jobs=NUM_THREADS).transform(smiles_list)


def naive_skfp_kr(smiles_list: list[str]) -> None:
    """
    Klekota-Roth fingerprint (naive scikit-fingerprints implementation)
    """
    NaiveSkfpKlekotaRothFingerprint(n_jobs=NUM_THREADS).transform(smiles_list)


def cdk_kr(smiles_list: list[str]) -> None:
    """
    Klekota-Roth fingerprint (CDK implementation)
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    CDK(fingerprint=FPType.KRFP).calculate(mols, show_banner=False, njobs=NUM_THREADS)


def run_benchmark():
    """
    Run Klekota-Roth fingerprint computation benchmark for multiple implementations.

    Steps:
    1. Load the HIV dataset from MoleculeNet (subset if DATASET_CUTOFF is set).
    2. Incrementally increase the number of molecules and measure the time
       required to compute Klekota-Roth fingerprints using:
       - scikit-fingerprints (optimized implementation),
       - scikit-fingerprints (naive implementation),
       - CDK.
    3. Compute both total execution time and time per molecule.
    4. Save the timing results into a CSV file.
    """
    smiles, _ = load_hiv()
    num_mols = len(smiles)

    # subset of dataset for testing
    if DATASET_CUTOFF:
        num_mols = min(num_mols, DATASET_CUTOFF)

    # include the last value by using length + 1
    steps = list(range(STEP, num_mols + 1, STEP))

    with open(RESULT_CSV_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "n_molecules",
                "skfp_mean_s",
                "skfp_std_s",
                "naive_skfp_mean_s",
                "naive_skfp_std_s",
                "cdk_mean_s",
                "cdk_std_s",
                "skfp_per_mol_ms",
                "naive_skfp_per_mol_ms",
                "cdk_per_mol_ms",
            ]
        )

    for n in steps:
        print(f"Processing {n} molecules...")
        subset = smiles[:n]

        skfp_mean, skfp_std = measure_time(
            skfp_kr, subset, label="scikit-fingerprints", iterations=N_REPEATS
        )
        naive_skfp_mean, naive_skfp_std = measure_time(
            naive_skfp_kr,
            subset,
            label="naive scikit-fingerprints",
            iterations=N_REPEATS,
        )
        cdk_mean, cdk_std = measure_time(
            cdk_kr, subset, label="CDK", iterations=N_REPEATS
        )

        skfp_per_mol = (skfp_mean / n) * 1000
        naive_skfp_per_mol = (naive_skfp_mean / n) * 1000
        cdk_per_mol = (cdk_mean / n) * 1000

        with open(RESULT_CSV_PATH, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    n,
                    skfp_mean,
                    skfp_std,
                    naive_skfp_mean,
                    naive_skfp_std,
                    cdk_mean,
                    cdk_std,
                    skfp_per_mol,
                    naive_skfp_per_mol,
                    cdk_per_mol,
                ]
            )

    print(f"Benchmark finished. Results saved to {RESULT_CSV_PATH}")


def plot_results():
    """
    Plot timing results for scikit-fingerprints vs CDK and save as PNG or PDF.
    """
    df = pd.read_csv(RESULT_CSV_PATH)

    plt.figure(figsize=(10, 6))

    if USE_ERROR_BARS:
        plt.errorbar(
            df["n_molecules"],
            df["skfp_mean_s"],
            yerr=df["skfp_std_s"],
            label="scikit-fingerprints",
            capsize=3,
        )
        plt.errorbar(
            df["n_molecules"],
            df["naive_skfp_mean_s"],
            yerr=df["naive_skfp_std_s"],
            label="naive scikit-fingerprints",
            capsize=3,
        )
        plt.errorbar(
            df["n_molecules"],
            df["cdk_mean_s"],
            yerr=df["cdk_std_s"],
            label="CDK",
            capsize=3,
        )
    else:
        plt.plot(df["n_molecules"], df["skfp_mean_s"], label="scikit-fingerprints")
        plt.fill_between(
            df["n_molecules"],
            df["skfp_mean_s"] - df["skfp_std_s"],
            df["skfp_mean_s"] + df["skfp_std_s"],
            alpha=0.3,
        )
        plt.plot(
            df["n_molecules"],
            df["naive_skfp_mean_s"],
            label="naive scikit-fingerprints",
        )
        plt.fill_between(
            df["n_molecules"],
            df["naive_skfp_mean_s"] - df["naive_skfp_std_s"],
            df["naive_skfp_mean_s"] + df["naive_skfp_std_s"],
            alpha=0.3,
        )
        plt.plot(df["n_molecules"], df["cdk_mean_s"], label="CDK")
        plt.fill_between(
            df["n_molecules"],
            df["cdk_mean_s"] - df["cdk_std_s"],
            df["cdk_mean_s"] + df["cdk_std_s"],
            alpha=0.3,
        )

    plt.title("Klekota-Roth fingerprint computation times")
    plt.xlabel("Number of molecules")
    plt.ylabel("Time [s]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_PLOT_PATH)

    print(f"Plot saved to {RESULT_PLOT_PATH}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
