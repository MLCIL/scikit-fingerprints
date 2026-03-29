"""
PubChem Fingerprint scikit-fingerprints (local) vs PubChem API benchmark.
"""

import asyncio
import csv
import os
import time
from pathlib import Path

import httpx
import joblib
import pandas as pd
import requests
from benchmarking.utils import measure_time
from matplotlib import pyplot as plt
from skfp.datasets.moleculenet import load_hiv
from skfp.fingerprints.pubchem import PubChemFingerprint

N_REPEATS = 5
STEP = 100
DATASET_CUTOFF = 1000
MAX_CONCURRENT_REQUESTS = 3  # for fetching CIDs
RETRY_DELAY = 10
MAX_RETRIES = 5
NUM_THREADS = joblib.effective_n_jobs(
    n_jobs=-1
)  # for computing fingerpritns and retrieving them via the API


OUTPUTS_DIR = Path("benchmark_times") / "benchmark_times_saved"
PLOTS_DIR = Path("benchmark_times") / "benchmark_times_plotted"
CSV_FILENAME = "pubchem_fp_timings.csv"

USE_PDF = True  # If True, save plot as PDF, otherwise save as PNG

file_ext = ".pdf" if USE_PDF else ".png"

PLOT_FILENAME = "pubchem_fp_timings"

RESULT_CSV_PATH = OUTPUTS_DIR / CSV_FILENAME
RESULT_PLOT_PATH = PLOTS_DIR / f"{PLOT_FILENAME}{file_ext}"

USE_ERROR_BARS = False  # If True, use error bars instead of shaded fill_between


async def fetch_cid(
    client: httpx.AsyncClient, smiles: str, sem: asyncio.Semaphore
) -> int | None:
    """
    Fetch the PubChem CID for a single SMILES string using the PubChem API.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/TXT"

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                r = await client.get(url, timeout=10.0)
                r.raise_for_status()
                text = r.text.strip()

                if text and text.split()[0].isdigit():
                    cid = int(text.split()[0])
                    if cid > 0:
                        return cid

            except Exception as e:
                print(f"CID retry {attempt + 1}/{MAX_RETRIES} for {smiles}: {e}")

            await asyncio.sleep(RETRY_DELAY)

    return None


async def smiles_to_cid(smiles_list: list[str], needed: int) -> list[int]:
    """
    Convert a list of SMILES strings to PubChem CIDs using the PubChem API.

    This uses a different number of concurrent requests compared to the pubchem_api_fp method,
    since it is intended for preparing the data and not for the actual benchmark.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    cids: list[int] = []

    async with httpx.AsyncClient() as client:
        i = 0
        while len(cids) < needed and i < len(smiles_list):
            batch = smiles_list[i : i + MAX_CONCURRENT_REQUESTS]
            tasks = [fetch_cid(client, smi, sem) for smi in batch]
            results = await asyncio.gather(*tasks)

            for cid in results:
                if cid is not None:
                    cids.append(cid)
                    if len(cids) == needed:
                        break

            i += MAX_CONCURRENT_REQUESTS

    print(f"Collected {len(cids)} valid CIDs (needed {needed})")
    return cids


def _fetch_fp_for_cid(cid: int) -> None:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Fingerprint2D/TXT"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt + 1 == MAX_RETRIES:
                print(f"FP failed for CID {cid}: {e}")
            else:
                print(f"FP retry {attempt + 1}/{MAX_RETRIES} for CID {cid}: {e}")
            time.sleep(RETRY_DELAY)
    return None


def pubchem_api_fp(cid_list: list[int], n_jobs: int = 1) -> None:
    """
    Retrieve PubChem fingerprints for a list of CIDs via the PubChem API using joblib.Parallel.
    """
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fetch_fp_for_cid)(cid) for cid in cid_list
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

    # subset of dataset for testing
    X = X[:DATASET_CUTOFF]

    cids = []
    while len(cids) < DATASET_CUTOFF:
        missing = DATASET_CUTOFF - len(cids)
        print(f"Fetching {missing} more CIDs...")
        new_cids = asyncio.run(smiles_to_cid(X, missing))
        cids.extend(new_cids)

        if not new_cids:
            print("Failed to fetch additional CIDs. Stopping benchmark.")
            return

    # include the last value by using length + 1
    steps = list(range(STEP, DATASET_CUTOFF + 1, STEP))

    with open(RESULT_CSV_PATH, "w", newline="") as file_out:
        writer = csv.writer(file_out)
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

        subset_smiles = X[:n]
        subset_cids = cids[:n]

        api_mean, api_std = measure_time(
            lambda cids: pubchem_api_fp(cids, n_jobs=NUM_THREADS),
            subset_cids,
            "PubChem API",
            N_REPEATS,
        )
        skfp_mean, skfp_std = measure_time(
            lambda smiles: skfp_pubchem_fp(smiles, n_jobs=NUM_THREADS),
            subset_smiles,
            "scikit-fingerprints",
            N_REPEATS,
        )

        with open(RESULT_CSV_PATH, "a", newline="") as file_out:
            writer = csv.writer(file_out)
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
