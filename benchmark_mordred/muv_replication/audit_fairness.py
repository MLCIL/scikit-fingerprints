"""
Rozbiorka wyniku 47.5x na MUV: skad sie bierze i czy warunki sa uczciwe.

Bierze paczke 10 000 czasteczek z tego samego przygotowanego zbioru i tej samej
pozycji, ktorej uzywa measure_embedding.py, potem mierzy warianty:

  mordred-96        mordred, nproc=96, BLAS niepinowany  -> to bylo w replice
  mordred-96-pin    mordred, nproc=96, BLAS pinowany do 1 watku na workera
  ours-96           nasza implementacja, n_jobs=96, z rozbiciem parse/deskryptory

Roznica miedzy pierwszymi dwoma to koszt nadsubskrypcji, ktorej nasz wrapper nie
ma, bo loky pinuje workery sam. Jesli jest duza, czesc ilorazu 47.5x mierzy
konfiguracje mordreda, a nie nasza implementacje - i trzeba to powiedziec wprost.

    python audit_fairness.py mordred-96
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python audit_fairness.py mordred-96-pin
    python audit_fairness.py ours-96
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, ".")
sys.path.insert(0, "model_wrappers/mordred")

MODE = sys.argv[1]
# paczka 2, zeby ominac koszt rozgrzania widoczny w paczce 1
LO, HI = 10_000, 20_000


def pinning() -> dict:
    return {v: os.environ.get(v) for v in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


dataset = joblib.load("data/prepared/ogbg-molmuv.joblib")
illegal = [line.strip() for line in open("config/illegal_smiles.txt")]
dataset.filter_out_problematic_molecules(illegal)
smiles = dataset.data["smiles"].str.replace("*", "C", regex=False).tolist()[LO:HI]
print(f"tryb={MODE}  czasteczek={len(smiles)}  pinowanie={pinning()}")

from wrapper import _embed_batch  # helper przeniesiony 1:1 do obu wrapperow

n_jobs = int(os.environ.get("AUDIT_NJOBS", "96"))

if MODE.startswith("mordred"):
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)

    t0 = time.perf_counter()
    mols = _embed_batch(smiles, with_3d=False, n_jobs=n_jobs)
    mols = [m for m in mols if m is not None]
    t_parse = time.perf_counter() - t0

    t0 = time.perf_counter()
    n = width = 0
    for row in calc.map(mols, nproc=n_jobs, quiet=True):
        n += 1
        width = width or len(row)
    t_desc = time.perf_counter() - t0
    print(f"  parse       {t_parse:8.2f} s")
    print(f"  deskryptory {t_desc:8.2f} s   ({n} wierszy x {width} cech)")
    print(f"  RAZEM       {t_parse + t_desc:8.2f} s")
else:
    from skfp.fingerprints import NewMordredFingerprint

    fp = NewMordredFingerprint(use_3D=False, n_jobs=n_jobs, verbose=False)
    fp.transform([_embed_batch(["CCO"], with_3d=False, n_jobs=1)[0]])  # rozgrzanie

    t0 = time.perf_counter()
    mols = _embed_batch(smiles, with_3d=False, n_jobs=n_jobs)
    mols = [m for m in mols if m is not None]
    t_parse = time.perf_counter() - t0

    t0 = time.perf_counter()
    X = np.asarray(fp.transform(mols), dtype=np.float64)
    t_desc = time.perf_counter() - t0
    print(f"  parse       {t_parse:8.2f} s")
    print(f"  deskryptory {t_desc:8.2f} s   ({X.shape[0]} wierszy x {X.shape[1]} cech)")
    print(f"  RAZEM       {t_parse + t_desc:8.2f} s")
