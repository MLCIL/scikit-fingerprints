"""
Benchmark bulk Tanimoto similarity computation using skfp and RDKit.
"""

import csv
import os.path
import time
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from skfp.datasets.moleculenet import load_hiv
from skfp.distances.tanimoto import bulk_tanimoto_binary_similarity
from skfp.fingerprints import ECFPFingerprint

N_REPEATS = 5
STEP = 500  # Step for increasing dataset size
DATASET_CUTOFF = 5000  # Maximum number of molecules to benchmark
USE_ERROR_BARS = True  # If True, use error bars instead of shaded fill_between

NUM_THREADS = joblib.effective_n_jobs(n_jobs=-1)  # Use all available CPU cores

OUTPUTS_DIR = os.path.join("benchmark_times", "benchmark_times_saved")
PLOTS_DIR = os.path.join("benchmark_times", "benchmark_times_plotted")
CSV_FILENAME = "bulk_tanimoto_timings.csv"
USE_PDF = True  # If True, save plot as PDF, otherwise save as PNG
PLOT_FILENAME = "bulk_tanimoto"

RESULT_CSV_PATH = os.path.join(OUTPUTS_DIR, CSV_FILENAME)
RESULT_PLOT_PATH = os.path.join(
    PLOTS_DIR, f"{PLOT_FILENAME}.pdf" if USE_PDF else f"{PLOT_FILENAME}.png"
)


def run_benchmarks():
    """
    Run bulk Tanimoto similarity benchmarks for scikit-fingerprints and RDKit.

    Steps:
    1. Load the HIV dataset from MoleculeNet (subset if DATASET_CUTOFF is set).
    2. Incrementally increase the number of molecules and measure the time
       required to compute pairwise Tanimoto similarity matrices.
    3. Save the timing results into a CSV file.
    """
    X, _ = load_hiv()
    num_mols = len(X)

    # subset of dataset for testing
    if DATASET_CUTOFF:
        num_mols = DATASET_CUTOFF

    header = [
        "n",
        "skfp_mean_tanimoto",
        "skfp_std_tanimoto",
        "rdkit_mean_tanimoto",
        "rdkit_std_tanimoto",
    ]

    with open(RESULT_CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

    steps = list(range(STEP, num_mols, STEP))
    # top off the dataset in case the number of molecules isn't a multiple of STEP
    if steps[-1] != num_mols:
        steps.append(num_mols)

    for n in steps:
        print(f"\nExperiment with {n} molecules:")
        smiles_list = X[:n]
        skfp_mean, skfp_std = benchmark_skfp(smiles_list)
        rdkit_mean, rdkit_std = benchmark_rdkit(smiles_list)
        row = [n, skfp_mean, skfp_std, rdkit_mean, rdkit_std]

        with open(RESULT_CSV_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    print(f"Benchmark results saved to {RESULT_CSV_PATH}")


def benchmark_skfp(smiles: list[str]) -> tuple[float, float]:
    """
    Compute pairwise Tanimoto similarity using scikit-fingerprints ECFP.
    """
    print("\nBenchmarking scikit-fingerprints...")
    ecfp = ECFPFingerprint(fp_size=2048, radius=2, n_jobs=NUM_THREADS)

    fps = ecfp.transform(smiles)

    mean_tanimoto, std_tanimoto = measure_time(
        func=bulk_tanimoto_binary_similarity,
        data=fps,
        desc="scikit-fingerprints Tanimoto",
        iterations=N_REPEATS,
    )

    return mean_tanimoto, std_tanimoto


def benchmark_rdkit(smiles: list[str]) -> tuple[float, float]:
    """
    Compute pairwise Tanimoto similarity using RDKit.
    """
    print("Benchmarking RDKIT...")
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = morgan_gen.GetFingerprints(mols, numThreads=NUM_THREADS)

    def bulk_tanimoto_similarity_rdkit(fps: list) -> np.ndarray:
        fps = list(fps)
        n = len(fps)
        tanimoto_matrix = np.zeros((n, n))
        for i in range(n):
            sims = BulkTanimotoSimilarity(fps[i], fps[i:])
            tanimoto_matrix[i, i:] = sims
        return tanimoto_matrix

    mean_tanimoto, std_tanimoto = measure_time(
        func=bulk_tanimoto_similarity_rdkit,
        data=fps,
        desc="RDKit Tanimoto",
        iterations=N_REPEATS,
    )

    return mean_tanimoto, std_tanimoto


def measure_time(
    func: Callable,
    data: list[str] | np.ndarray | list[object],
    desc: str,
    iterations: int,
) -> tuple[float, float]:
    """
    Measure the average execution time of a function over N_REPEATS.
    """
    times: list[float] = []

    print(f"\tBenchmarking {desc} calculation time...")
    for _ in range(iterations):
        start = time.time()
        _ = func(data)
        end = time.time()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time


def plot_results() -> None:
    """
    Plot timing results for scikit-fingerprints vs RDKit and save as PNG.
    """
    try:
        df = pd.read_csv(RESULT_CSV_PATH)
    except FileNotFoundError:
        print("CSV file not found")
        return

    plt.figure(figsize=(10, 6))
    if USE_ERROR_BARS:
        plt.errorbar(
            df["n"],
            df["skfp_mean_tanimoto"],
            yerr=df["skfp_std_tanimoto"],
            label="scikit-fingerprints",
            capsize=3,
        )
        plt.errorbar(
            df["n"],
            df["rdkit_mean_tanimoto"],
            yerr=df["rdkit_std_tanimoto"],
            label="RDKit",
            capsize=3,
        )
    else:
        plt.plot(df["n"], df["skfp_mean_tanimoto"], label="scikit-fingerprints")
        plt.fill_between(
            df["n"],
            df["skfp_mean_tanimoto"] - df["skfp_std_tanimoto"],
            df["skfp_mean_tanimoto"] + df["skfp_std_tanimoto"],
            alpha=0.3,
        )
        plt.plot(df["n"], df["rdkit_mean_tanimoto"], label="RDKit")
        plt.fill_between(
            df["n"],
            df["rdkit_mean_tanimoto"] - df["rdkit_std_tanimoto"],
            df["rdkit_mean_tanimoto"] + df["rdkit_std_tanimoto"],
            alpha=0.3,
        )

    plt.title("Pairwise Tanimoto similarity computation time")
    plt.xlabel("Number of molecules")
    plt.ylabel("Time [s]")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(RESULT_PLOT_PATH)


if __name__ == "__main__":
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    run_benchmarks()
    plot_results()
