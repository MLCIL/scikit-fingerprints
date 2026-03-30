"""
PubChem Fingerprint scikit-fingerprints (local) vs PubChem API benchmark.
"""

import csv
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import requests
from benchmarking.utils import measure_time
from matplotlib import pyplot as plt
from skfp.datasets.moleculenet import load_hiv
from skfp.fingerprints.pubchem import PubChemFingerprint

N_REPEATS = 5
STEP = 10
DATASET_CUTOFF = 100
RETRY_DELAY = 10
MAX_RETRIES = 5
NUM_THREADS = joblib.effective_n_jobs(n_jobs=-1)


OUTPUTS_DIR = Path("benchmark_times") / "benchmark_times_saved"
PLOTS_DIR = Path("benchmark_times") / "benchmark_times_plotted"
CSV_FILENAME = "pubchem_fp_timings.csv"

USE_PDF = True  # If True, save plot as PDF, otherwise save as PNG

file_ext = ".pdf" if USE_PDF else ".png"

PLOT_FILENAME = "pubchem_fp_timings"

RESULT_CSV_PATH = OUTPUTS_DIR / CSV_FILENAME
RESULT_PLOT_PATH = PLOTS_DIR / f"{PLOT_FILENAME}{file_ext}"

USE_ERROR_BARS = False  # If True, use error bars instead of shaded fill_between


def _retry_request(url: str) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt + 1 == MAX_RETRIES:
                print(f"Request failed: {url}, {e}")
            else:
                print(f"Retry {attempt + 1}/{MAX_RETRIES} for URL: {url}")
            time.sleep(RETRY_DELAY)

    return None


def fetch_cid(smiles: str) -> int | None:
    """
    Fetch the PubChem CID for a single SMILES string using the PubChem API.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/TXT"
    text = _retry_request(url)
    if text:
        first = text.strip().split()[0]
        if first.isdigit():
            return int(first)
    return None


def fetch_fp(cid: int) -> str | None:
    """Fetch PubChem fingerprint for a CID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Fingerprint2D/TXT"
    return _retry_request(url)


def smiles_to_cid(smiles_list: list[str], needed: int) -> list[int]:
    """
    Convert a list of SMILES strings to PubChem CIDs using the PubChem API (joblib, requests).
    """
    cids: list[int] = []
    i = 0
    while len(cids) < needed and i < len(smiles_list):
        batch = smiles_list[i : i + NUM_THREADS]
        results = joblib.Parallel(n_jobs=NUM_THREADS)(
            joblib.delayed(fetch_cid)(smi) for smi in batch
        )
        for cid in results:
            if cid is not None:
                cids.append(cid)
                if len(cids) == needed:
                    break
        i += NUM_THREADS
    print(f"Collected {len(cids)} valid CIDs (needed {needed})")
    return cids


def pubchem_api_pipeline(smiles_list: list[str]) -> list[str]:
    """Fetch CIDs on-the-fly for the subset and retrieve fingerprints."""
    cids = smiles_to_cid(smiles_list, len(smiles_list))
    return joblib.Parallel(n_jobs=NUM_THREADS)(
        joblib.delayed(fetch_fp)(cid) for cid in cids
    )


def skfp_pubchem_fp(smiles_list: list[str], n_jobs: int = 1) -> None:
    """
    PubChem fingerprint (scikit-fingerprints implementation).
    """
    PubChemFingerprint(n_jobs=n_jobs).transform(smiles_list)


def run_benchmark():
    """
    Run a performance benchmark comparing PubChem REST API vs scikit-fingerprints local fingerprints.

    Steps:
    1. Load the HIV dataset from MoleculeNet and optionally subset it using DATASET_CUTOFF.
    2. Fetch PubChem CIDs for the molecules using asynchronous API calls.
    3. Incrementally increase the number of molecules and measure
       execution time for:
       - Retrieving fingerprints via PubChem REST API.
       - Computing fingerprints locally via scikit-fingerprints.
    4. Compute per-molecule timing and write mean/std results to a CSV file.
    """
    X, _ = load_hiv()
    X = X[:DATASET_CUTOFF]

    # include the last value by using length + 1
    steps = list(range(STEP, DATASET_CUTOFF + 1, STEP))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(RESULT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_molecules",
                "api_mean_s",
                "api_std_s",
                "skfp_mean_s",
                "skfp_std_s",
                "api_per_mol_ms",
                "skfp_per_mol_ms",
            ]
        )

        for n in steps:
            print(f"Benchmarking {n} molecules")
            smiles_subset = X[:n]

            api_mean, api_std = measure_time(
                pubchem_api_pipeline,
                smiles_subset,
                "PubChem API",
                N_REPEATS,
            )

            skfp_mean, skfp_std = measure_time(
                skfp_pubchem_fp,
                smiles_subset,
                "scikit-fingerprints",
                N_REPEATS,
            )

            writer.writerow(
                [
                    n,
                    api_mean,
                    api_std,
                    skfp_mean,
                    skfp_std,
                    (api_mean / n) * 1000,
                    (skfp_mean / n) * 1000,
                ]
            )

    print("Benchmark complete.")


def plot_results():
    """
    Plot timing results for scikit-fingerprints vs PubChem API and save as PNG or PDF.
    """
    df = pd.read_csv(RESULT_CSV_PATH)

    plt.figure(figsize=(10, 6))
    if USE_ERROR_BARS:
        plt.errorbar(
            df["n_molecules"],
            df["api_mean_s"],
            yerr=df["api_std_s"],
            marker="o",
            capsize=4,
            label="PubChem API",
        )
        plt.errorbar(
            df["n_molecules"],
            df["skfp_mean_s"],
            yerr=df["skfp_std_s"],
            marker="s",
            capsize=4,
            label="scikit-fingerprints",
        )
    else:
        plt.plot(df["n_molecules"], df["api_mean_s"], marker="o", label="PubChem API")
        plt.fill_between(
            df["n_molecules"],
            df["api_mean_s"] - df["api_std_s"],
            df["api_mean_s"] + df["api_std_s"],
            alpha=0.3,
        )
        plt.plot(
            df["n_molecules"],
            df["skfp_mean_s"],
            marker="s",
            label="scikit-fingerprints",
        )
        plt.fill_between(
            df["n_molecules"],
            df["skfp_mean_s"] - df["skfp_std_s"],
            df["skfp_mean_s"] + df["skfp_std_s"],
            alpha=0.3,
        )

    plt.xlabel("Number of molecules")
    plt.ylabel("Time [s]")
    plt.title("PubChem Fingerprint: PubChem API vs scikit-fingerprints")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_PLOT_PATH)

    print(f"Plot saved to {RESULT_PLOT_PATH}")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    run_benchmark()
    plot_results()


if __name__ == "__main__":
    main()
