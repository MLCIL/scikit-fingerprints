"""
Czy pinowanie BLAS-a zmienia policzone deskryptory?

Jesli 7.5x przyspieszenia okupione jest innymi liczbami, to nie jest poprawka,
a kompromis. Porownuje macierze zapisane przez ten sam harness w dwoch
przebiegach mordreda - bez pinowania i z pinowaniem.

Rozne kolejnosci redukcji w BLAS-ie moga dawac roznice na ostatnich bitach, wiec
raportuje i identycznosc bitowa, i zgodnosc w tolerancji.

    python compare_pinned_outputs.py <bez_pinowania.joblib> <z_pinowaniem.joblib>
"""

from __future__ import annotations

import sys

import joblib
import numpy as np

sys.path.insert(0, ".")

a_path, b_path = sys.argv[1], sys.argv[2]
a = joblib.load(a_path).X
b = joblib.load(b_path).X
print(f"bez pinowania : {a_path}\n  ksztalt {a.shape}, dtype {a.dtype}")
print(f"z pinowaniem  : {b_path}\n  ksztalt {b.shape}, dtype {b.dtype}")

if a.shape != b.shape:
    raise SystemExit("ROZNE KSZTALTY - porownanie bez sensu")

identical = np.array_equal(a, b, equal_nan=True)
print(f"\nidentyczne bitowo:            {identical}")
if identical:
    print("  pinowanie nie zmienia ani jednej wartosci")
    raise SystemExit(0)

close = np.isclose(a, b, rtol=1e-9, atol=0.0, equal_nan=True)
n_diff = int((~close).sum())
print(f"  komorek roznych (rtol=1e-9): {n_diff} z {a.size} ({100 * n_diff / a.size:.2e}%)")
if n_diff:
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
    print(f"  najwiekszy blad relatywny:   {np.nanmax(rel):.3e}")
    cols = np.unique(np.nonzero(~close)[1])
    print(f"  dotyka {len(cols)} kolumn z {a.shape[1]}")
