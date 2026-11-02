"""
Metryki z przebiegu MolPILE, w formie gotowej na poster.

Czyta runtime.csv wyprodukowany przez measure_embedding.py i zestawia to z
liczbami z wykresu 28 modeli (results/muv_replication/reference_all_models.csv),
zeby porownania z Chemeleonem, MolE i mordredem byly oparte na jego pomiarach,
a nie na naszych szacunkach.

    python metrics_molpile.py <runtime.csv>
"""

import os
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

MOLPILE_TOTAL = 222_000_000  # arXiv 2509.18353
N_FEATURES = 1613
REFERENCE = Path(
    os.environ.get("BENCH_REPO", ".") + "/results/muv_replication/"
    "reference_all_models.csv"
)
# nasz pomiar mordreda z pinowanym BLAS-em, replika MUV, proba 2
MORDRED_PINNED_S_PER_10K = 25.069368680318195


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / 86400:.2f} dnia"


RUNTIME = Path(sys.argv[1])
df = pd.read_csv(RUNTIME)
row = df[df.embedder == "skfp_mordred_2d"].iloc[-1]
n_batch = int(row.n_samples)
mean, std, total = row.mean_runtime, row.std_runtime, row.total_runtime
batches = int(row.n_full_batches)
n_all = batches * n_batch  # tylko pelne paczki, tak liczy harness srednia
mol_s = n_batch / mean

print("=" * 68)
print("MOLPILE - METRYKI (nasza implementacja, protokol repo benchmarkowego)")
print("=" * 68)
print(f"  paczka                {n_batch:>12,} czasteczek")
print(f"  pelnych paczek        {batches:>12}")
print(f"  czas na paczke        {mean:>12.2f} s  +/- {std:.2f}  ({100 * std / mean:.1f}%)")
print(f"  total_runtime         {total:>12.2f} s  = {human(total)}")
print()
# ile czasteczek w ogole nie weszlo do pomiaru - bez tego liczba przepustowosci
# sugerowalaby, ze implementacja radzi sobie z calym korpusem
rejected = Path(__file__).parent / "data" / "prepared" / f"{row.dataset}-odrzucone.csv"
if rejected.exists():
    n_bad = max(0, sum(1 for _ in rejected.open()) - 1)
    # wzgledem WEJSCIA, nie wzgledem pelnych paczek: harness usredniia po 99
    # paczkach po 10 tys., ale zbior po filtrze ma 996 819 czasteczek, wiec
    # ogon 6 819 sztuk nie wchodzi do n_all i mianownik wyszedlby zanizony
    n_prepared = len(joblib.load(
        Path(__file__).parent / "data" / "prepared" / f"{row.dataset}.joblib"
    ).data)
    n_in = n_prepared + n_bad
    print(f"  odfiltrowane          {n_bad:>12,} z {n_in:,} ({100 * n_bad / n_in:.3f}%)")
    print("    czasteczki, na ktorych nasza implementacja rzuca wyjatkiem albo nie")
    print("    konczy liczenia (fullereny: DetourMatrix, NP-trudna najdluzsza sciezka).")
    print("    Oryginalny mordred zwraca tam placeholder 'timed out' po ~3 min.")
    print()
print("  PRZEPUSTOWOSC")
print(f"    {mol_s:>12,.0f} czasteczek / s")
print(f"    {mol_s * N_FEATURES:>12,.0f} deskryptorow / s  ({N_FEATURES} cech na czasteczke)")
print(f"    {mol_s * 3600 / 1e6:>12.2f} mln czasteczek / h")
print()
print("  CZAS DLA TYPOWYCH SKAL")
for label, n in [
    ("10 000 (paczka z wykresu)", 10_000),
    ("100 000", 100_000),
    ("1 mln (korpus pretreningowy)", 1_000_000),
    ("10 mln", 10_000_000),
    (f"{MOLPILE_TOTAL // 1_000_000} mln (caly MolPILE)", MOLPILE_TOTAL),
]:
    print(f"    {label:32s} {human(n / mol_s):>12s}")
print()
print("  ROZMIAR WYNIKU")
for label, n in [("1 mln", 1_000_000), ("caly MolPILE", MOLPILE_TOTAL)]:
    for prec, nbytes in (("float32, tak zapisuje harness", 4), ("float64, tak oddaje transform", 8)):
        gb = n * N_FEATURES * nbytes / 1e9
        size = f"{gb:>9.1f} GB" if gb < 1000 else f"{gb / 1000:>9.1f} TB"
        print(f"    {label:12s} {prec:32s} {size}")
print()

print("  WOBEC ORYGINALNEGO MORDREDA (nasz pomiar, BLAS pinowany, MUV)")
m_mol_s = 10_000 / MORDRED_PINNED_S_PER_10K
print(f"    mordred                      {m_mol_s:>9,.0f} mol/s   -> 1 mln w {human(1e6 / m_mol_s)}")
print(f"    nasze                        {mol_s:>9,.0f} mol/s   -> 1 mln w {human(1e6 / mol_s)}")
print(f"    iloraz                       {mol_s / m_mol_s:>9.2f}x")
print()

if REFERENCE.exists():
    ref = pd.read_csv(REFERENCE)
    ref = ref[ref.dataset == "ogbg-molmuv"]
    print("  WOBEC MODELI Z WYKRESU 28 MODELI (jego pomiary, 10k czasteczek)")
    print(f"    {'model':26s} {'s/10k':>9s} {'1 mln':>12s} {'my szybciej':>12s}")
    interesting = {
        "chemeleon": "Chemeleon (cuda)",
        "mole": "MolE (cuda)",
        "CLAMP": "CLAMP (cuda)",
        "MoLFormer-XL-both-10pct": "MoLFormer (cuda)",
        "GNN-Motif-sum": "GIN (cpu)",
        "ECFP": "ECFP (cpu)",
        "mordred_2d": "Mordred, jego pomiar",
    }
    ours_10k = 1e4 / mol_s
    for emb, label in interesting.items():
        r = ref[ref.embedder == emb]
        if r.empty:
            continue
        s10k = float(r.iloc[0].mean_runtime)
        print(f"    {label:26s} {s10k:>9.2f} {human(1e6 * s10k / 1e4):>12s} {s10k / ours_10k:>11.1f}x")
    print(f"    {'NASZE':26s} {ours_10k:>9.2f} {human(1e6 / mol_s):>12s} {'-':>12s}")
