"""
Benchmark the scikit-fingerprints autocorrelation `calc()` against the
autocorrelation descriptors from mordred-community, in terms of runtime,
over the full BACE dataset from MoleculeNet.

Run from the scikit-fingerprints repository root so the real skfp module is
importable:

    pip install mordredcommunity
    python benchmark_autocorrelation.py

Both implementations compute the same 606 autocorrelation descriptors. skfp's
`calc()` takes a precomputed DistanceMatrix; mordred runs through its own
Calculator. The script times computing the descriptors for every molecule in
the dataset and reports total time, per-molecule average, and throughput.
"""

import time

import numpy as np
from mordred import Autocorrelation, Calculator
from rdkit.Chem import AddHs

from skfp.datasets.moleculenet import load_bace
from skfp.fingerprints._new_mordred.descriptors.autocorrelation import calc
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.preprocessing import MolFromSmilesTransformer


def load_molecules() -> list:
    """Load BACE and return explicit-hydrogen RDKit mols (as mordred expects)."""
    smiles_list, _ = load_bace()
    mols = MolFromSmilesTransformer().transform(smiles_list)
    return [AddHs(mol) for mol in mols]


def time_skfp(mols: list) -> float:
    """Total seconds for skfp `calc()` over all molecules (matrix prep included)."""
    start = time.perf_counter()
    for mol in mols:
        calc(mol, DistanceMatrix(mol))
    return time.perf_counter() - start


def time_mordred(mols: list) -> float:
    """Total seconds for mordred autocorrelation over all molecules."""
    mordred_calc = Calculator(Autocorrelation)
    start = time.perf_counter()
    for mol in mols:
        mordred_calc(mol)
    return time.perf_counter() - start


def print_report(n_mols: int, skfp_s: float, mordred_s: float) -> None:
    print(f"BACE dataset: {n_mols} molecules\n")
    header = f"{'implementation':<16}{'total (s)':>11}{'per mol (ms)':>14}{'mol/s':>10}"
    print(header)
    print("-" * len(header))
    for label, total in [("skfp", skfp_s), ("mordred", mordred_s)]:
        print(
            f"{label:<16}{total:>11.3f}{total / n_mols * 1e3:>14.3f}{n_mols / total:>10.1f}"
        )
    print("-" * len(header))
    print(f"\nspeedup: {mordred_s / skfp_s:.1f}x")


if __name__ == "__main__":
    mols = load_molecules()

    # warm up both implementations on a small slice before timing
    warmup = mols[:10]
    time_skfp(warmup)
    time_mordred(warmup)

    # silence divide-by-zero warnings from undefined (zero-pair) lags
    with np.errstate(invalid="ignore", divide="ignore"):
        skfp_s = time_skfp(mols)
        mordred_s = time_mordred(mols)

    print_report(len(mols), skfp_s, mordred_s)
