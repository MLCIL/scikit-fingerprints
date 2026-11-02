"""
Where does the time go? cProfile both implementations on the same molecules and
attribute self-time (tottime) to the source file that spent it, then roll the
files up into descriptor families.

Single-threaded, BLAS pinned - same protocol as bench_run.py.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import cProfile, pstats, sys, json, csv
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED = 0

from skfp.datasets.moleculenet import load_bbbp
import numpy as np

smiles, _ = load_bbbp()
rng = np.random.default_rng(SEED)
idx = rng.choice(len(smiles), size=min(N, len(smiles)), replace=False)
mols = []
for i in idx:
    m = Chem.MolFromSmiles(smiles[i])
    if m is not None:
        mols.append(m)
print(f"{len(mols)} molecules, mean atoms {np.mean([m.GetNumAtoms() for m in mols]):.1f}")


def profile(fn, label):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    by_file = defaultdict(float)
    total = 0.0
    for (fname, lineno, func), (cc, nc, tt, ct, callers) in st.stats.items():
        by_file[fname] += tt
        total += tt
    print(f"--- {label}: total self-time {total:.2f} s")
    return dict(by_file), total


def run_ours():
    from skfp.fingerprints import NewMordredFingerprint
    NewMordredFingerprint(n_jobs=1, verbose=False).transform(mols)


def run_mordred():
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    for m in mols:
        list(calc(m))


ours, ours_total = profile(run_ours, "ours")
mord, mord_total = profile(run_mordred, "mordred")

json.dump({"ours": ours, "ours_total": ours_total,
           "mordred": mord, "mordred_total": mord_total,
           "n_mols": len(mols)},
          open("profile_raw.json", "w"), indent=1)

def top(d, total, n=30):
    rows = sorted(d.items(), key=lambda kv: -kv[1])[:n]
    for f, t in rows:
        print(f"  {t:8.3f} s {100*t/total:5.1f}%  {f}")

print("\n=== OURS, top files ===");    top(ours, ours_total)
print("\n=== MORDRED, top files ==="); top(mord, mord_total)
