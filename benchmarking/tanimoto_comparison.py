"""
Benchmark bulk Tanimoto similarity computation using skfp and RDKit.
"""

import csv
import os.path
import time
from collections.abc import Callable
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import rdkit
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from skfp.datasets.moleculenet import load_hiv
from skfp.distances.tanimoto import bulk_tanimoto_binary_similarity
from skfp.fingerprints import ECFPFingerprint

N_REPEATS = 10
FP_SIZE = 2048  # Size of the fingerprint in bits
FP_RADIUS = 2  # Radius of atom environment considered for Morgan fingerprint
STEP = 500  # Step for increasing dataset size
DATASET_CUTOFF = 5000  # Maximum number of molecules to benchmark

NUM_THREADS = joblib.effective_n_jobs(n_jobs=-1)  # Use all available CPU cores

OUTPUTS_DIR = r"benchmark_times\benchmark_times_saved"
PLOTS_DIR = r"benchmark_times\benchmark_times_plotted"
CSV_FILENAME = "bulk_tanimoto_timings.csv"
PLOT_FILENAME = "bulk_tanimoto.png"

RESULT_CSV_PATH = os.path.join(OUTPUTS_DIR, CSV_FILENAME)
RESULT_PLOT_PATH = os.path.join(PLOTS_DIR, PLOT_FILENAME)


def run_benchmarks():
    """
    Run bulk Tanimoto similarity benchmarks for scikit-fingerprints and RDKit.

    Steps:
    1. Load the HIV dataset from MoleculeNet (subset if DATASET_CUTOFF is set).
    2. Incrementally increase the number of molecules and measure the time
       required to compute pairwise Tanimoto similarity matrices.
    3. Save the timing results into a CSV file.
    """
    print(f"scikit-fingerprints version: {version('scikit-fingerprints')}")
    print(f"RDKit version: {rdkit.__version__}")

    X, _ = load_hiv()
    length = len(X)

    # subset of dataset for testing
    if DATASET_CUTOFF:
        length = DATASET_CUTOFF

    header = ["n", "skfp_mean_tanimoto", "rdkit_mean_tanimoto"]

    with open(RESULT_CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

    steps = list(range(STEP, length, STEP))
    # Top off the dataset in case the number of molecules isn't a multiple of STEP
    if steps[-1] != length:
        steps.append(length)

    for n in steps:
        print(f"\nExperiment with {n} molecules:")
        smiles_list = X[:n]
        skfp_results = benchmark_skfp(smiles_list)
        rdkit_results = benchmark_rdkit(smiles_list)
        row = [n, skfp_results, rdkit_results]

        with open(RESULT_CSV_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    print(f"Benchmark results saved to {RESULT_CSV_PATH}")


def benchmark_skfp(smiles: list[str]) -> float:
    """
    Compute pairwise Tanimoto similarity using scikit-fingerprints ECFP.
    """
    print("\nBenchmarking scikit-fingerprints...")
    ecfp = ECFPFingerprint(fp_size=FP_SIZE, radius=FP_RADIUS, n_jobs=NUM_THREADS)

    fps = ecfp.transform(smiles)

    _, mean_tanimoto = measure_time(
        func=bulk_tanimoto_binary_similarity,
        data=fps,
        desc="scikit-fingerprints Tanimoto",
    )

    return mean_tanimoto


def benchmark_rdkit(smiles: list[str]) -> float:
    """
    Compute pairwise Tanimoto similarity using RDKit Morgan fingerprints.
    """
    print("Benchmarking RDKIT...")
    morgan_gen = GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)

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

    _, mean_tanimoto = measure_time(
        func=bulk_tanimoto_similarity_rdkit,
        data=fps,
        desc="RDKit Tanimoto",
    )

    return mean_tanimoto


def measure_time(
    func: Callable, data: list[str] | np.ndarray | list[object], desc: str
) -> tuple[np.ndarray, float]:
    """
    Measure the average execution time of a function over N_REPEATS.
    """
    times: list[float] = []
    res = None

    print(f"\tBenchmarking {desc} calculation time...")
    for _ in range(N_REPEATS):
        start = time.time()
        res = func(data)
        end = time.time()
        times.append(end - start)

    mean_time = np.mean(times)

    print(f"\t{desc}: mean={mean_time:.4f}s for {len(data)} items")

    return res, mean_time


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
    plt.plot(df["n"], df["skfp_mean_tanimoto"], label="scikit-fingerprints")
    plt.plot(df["n"], df["rdkit_mean_tanimoto"], label="RDKit")

    plt.title("Bulk Tanimoto similarity computation time")
    plt.xlabel("Number of molecules")
    plt.ylabel("Mean time with standard deviation [s]")
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
