"""
Klekota-Roth SKFP vs CDK benchmark.
"""

import csv
import os
import time
import warnings
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd
from benchmarking.naive_fp_klekota_roth import NaiveSkfpKlekotaRothFingerprint
from CDK_pywrapper import CDK, FPType
from matplotlib import pyplot as plt
from rdkit import Chem
from skfp.datasets.moleculenet import load_hiv
from skfp.fingerprints import KlekotaRothFingerprint

N_REPEATS = 5
STEP = 100  # Step for increasing dataset size
DATASET_CUTOFF = 1000  # Maximum number of molecules to benchmark

NUM_THREADS = joblib.effective_n_jobs(n_jobs=-1)  # Use all available CPU cores

OUTPUTS_DIR = os.path.join("benchmark_times", "benchmark_times_saved")
PLOTS_DIR = os.path.join("benchmark_times", "benchmark_times_plotted")

USE_PDF = True  # If True, save plot as PDF, otherwise save as PNG
USE_ERROR_BARS = False  # If True, use error bars instead of shaded fill_between

CSV_FILENAME = "skfp_cdk_kr_timings"
PLOT_FILENAME = "skfp_cdk_kr_timings"

RESULT_CSV_PATH = os.path.join(OUTPUTS_DIR, f"{CSV_FILENAME}.csv")
RESULT_PLOT_PATH = os.path.join(
    PLOTS_DIR, f"{PLOT_FILENAME}.pdf" if USE_PDF else f"{PLOT_FILENAME}.png"
)


def main():
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    run_benchmark()
    plot_results()


def measure_fp_time(
    func: Callable[[list[str]], None], smiles_list: list[str], label: str
) -> tuple[float, float]:
    """
    Measure the average execution time of a function over N_REPEATS.
    """
    times: list[float] = []
    print(f"Benchmarking {label} Klekota-Roth FP computation...")

    for _ in range(N_REPEATS):
        start = time.time()
        func(smiles_list)
        end = time.time()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time


def skfp_kr(smiles_list: list[str]) -> None:
    """
    Klekota-Roth fingerprint (optimized SKFP implementation)
    """
    kr_fp = KlekotaRothFingerprint(n_jobs=NUM_THREADS)
    _ = kr_fp.transform(smiles_list)


def naive_skfp_kr(smiles_list: list[str]) -> None:
    """
    Klekota-Roth fingerprint (naive SKFP implementation)
    """
    kr_fp = NaiveSkfpKlekotaRothFingerprint(n_jobs=NUM_THREADS)
    _ = kr_fp.transform(smiles_list)


def cdk_kr(smiles_list: list[str]) -> None:
    """
    Klekota-Roth fingerprint (CDK implementation)
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    cdk = CDK(fingerprint=FPType.KRFP)
    _ = cdk.calculate(mols, show_banner=False, njobs=NUM_THREADS)


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

    steps = list(range(STEP, num_mols + 1, STEP))
    # top off the dataset in case the number of molecules isn't a multiple of STEP
    if steps[-1] != num_mols:
        steps.append(num_mols)

    with open(RESULT_CSV_PATH, "w", newline="") as file_out:
        writer = csv.writer(file_out)
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

        skfp_mean, skfp_std = measure_fp_time(skfp_kr, subset, "scikit-fingerprints")
        naive_skfp_mean, naive_skfp_std = measure_fp_time(
            naive_skfp_kr, subset, "naive scikit-fingerprints"
        )
        cdk_mean, cdk_std = measure_fp_time(cdk_kr, subset, "CDK")

        skfp_per_mol = (skfp_mean / n) * 1000
        naive_skfp_per_mol = (naive_skfp_mean / n) * 1000
        cdk_per_mol = (cdk_mean / n) * 1000

        with open(RESULT_CSV_PATH, "a", newline="") as file_out:
            writer = csv.writer(file_out)
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
    try:
        df = pd.read_csv(RESULT_CSV_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {RESULT_CSV_PATH}")

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
