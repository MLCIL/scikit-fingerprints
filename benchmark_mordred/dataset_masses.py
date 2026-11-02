"""
Srednia masa czasteczkowa kazdego zbioru z benchmarku -> results/dataset_masses.csv.

Wykresy zaleznosci od rozmiaru czasteczki byly rysowane wobec liczby atomow, ale
w chemii ludzie operuja raczej masa. Liczby atomow nie da sie przeliczyc na mase,
bo zalezy od skladu pierwiastkowego, wiec trzeba ja zmierzyc na tych samych
czasteczkach, ktore weszly do pomiaru - stad ten skrypt uzywa loaderow z
`bench_run` i tego samego ziarna.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors

sys.path.insert(0, str(Path(__file__).parent))
import bench_run as B

PEPTIDE_DIR = Path(
    os.environ.get("PEPTIDE_REPO", "peptides_molecular_fingerprints_classification") + "/"
    "data/PeptideReactor"
)
SEED = 0


def datasets() -> list[str]:
    out = list(B.DATASET_SIZES)                        # MoleculeNet
    out += [f"synthpep_{n}" for n in B.PEPTIDE_LADDER]
    out += [f"preactor_{n}" for n in B.PEPTIDE_REACTOR_SETS]
    out += list(B.EXPANSIONRX)
    # zbior pierscieniowy wczytuje sie z pliku, nie przez loader
    out += ["csv:" + os.environ.get("BENCH_RINGS_CSV", "dataset_rings.csv")]
    return out


def main() -> None:
    rows = []
    for ds in datasets():
        rng = np.random.default_rng(SEED)
        try:
            mols = B.load_mols(ds, rng, PEPTIDE_DIR)
        except Exception as exc:
            print(f"  {ds:28s} POMINIETY: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not mols:
            continue
        # przy duzych zbiorach kilkaset czasteczek wystarczy na srednia masy
        if len(mols) > 500:
            idx = rng.choice(len(mols), size=500, replace=False)
            mols = [mols[i] for i in idx]
        mw = np.array([Descriptors.MolWt(m) for m in mols])
        atoms = np.array([m.GetNumAtoms() for m in mols])
        name = Path(ds[4:]).stem if ds.startswith("csv:") else ds
        rows.append({"dataset": name, "mean_mw": mw.mean(), "mean_atoms": atoms.mean(),
                     "n_sampled": len(mols)})
        print(f"  {name:28s} {mw.mean():9.1f} Da   {atoms.mean():7.1f} atomow", flush=True)

    out = Path(os.environ.get("BENCH_RES", "results")) / "dataset_masses.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nzapisane {out} ({len(rows)} zbiorow)")


if __name__ == "__main__":
    main()
