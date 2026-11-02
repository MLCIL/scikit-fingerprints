"""
Maksymalna przepustowosc naszej implementacji na MolPILE - liczba na poster.

To NIE jest pomiar porownawczy. Porownanie z oryginalnym mordredem i z modelami
z wykresu 28 modeli mamy z repliki MUV, w jego protokole. Tutaj chodzi o co
innego: pokazac, w jakim czasie da sie policzyc deskryptory dla korpusu wielkosci
pretreningowej, wiec praca jest podzielona tak, zeby maszyna byla wykorzystana,
a nie tak, zeby nasladowac cudzy harness.

Roznica jest konkretna. skfp dzieli dane na `len(data) // n_jobs`, czyli daje
kazdemu workerowi JEDNO zadanie. Przy paczce 10 000 na 96 workerow wychodzi po
~104 czasteczki i czas paczki wyznacza najwolniejszy worker - a koszt czasteczki
ma ciezki ogon, wiec maszyna stoi (zmierzone: ~20 zajetych rdzeni ze 128, czyli
2871 mol/s przy sufcie ~7100). Gdy zadan jest wielokrotnie wiecej niz workerow,
niezbalansowanie sie usrednia.

    python throughput_molpile.py --sweep          # dobor batch_size + pelny pomiar
    python throughput_molpile.py --limit 100000   # szybki test
"""

import os

# musi byc PRZED numpy: OpenBLAS czyta te zmienne przy ladowaniu biblioteki, a bez
# tego kazdy z 96 workerow rozpina wlasna pule watkow i bija sie o rdzenie
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from skfp.fingerprints import NewMordredFingerprint

MOLPILE_TOTAL = 222_000_000  # arXiv 2509.18353
BYTES_PER_VALUE = 8  # float64, tak jak oddaje transform
PREPARED = Path(__file__).parent / "data" / "prepared" / "molpile-1m.joblib"


def human(s: float) -> str:
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    if s < 172800:
        return f"{s / 3600:.2f} h"
    return f"{s / 86400:.2f} dnia"


def parse_smiles(smiles: list[str], n_jobs: int) -> list:
    """SMILES -> RDKit Mol. Mierzone osobno, bo to inny koszt niz deskryptory."""
    def work(part: list[str]) -> list:
        return [Chem.MolFromSmiles(s) for s in part]

    step = max(len(smiles) // (n_jobs * 8), 1)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(work)(smiles[i : i + step]) for i in range(0, len(smiles), step)
    )
    return [m for part in parts for m in part]


def time_transform(mols: list, n_jobs: int, batch_size: int | None):
    fp = NewMordredFingerprint(n_jobs=n_jobs, batch_size=batch_size, verbose=False)
    t = time.perf_counter()
    out = fp.transform(mols)
    return time.perf_counter() - t, out.shape[1]


def time_persist(arr, path: Path) -> float:
    """Ile kosztuje utrwalenie wyniku na dysku.

    Harness tego NIE mierzy - stoper obejmuje tylko `model.embed(batch)`, a zapis
    idzie po petli (src/embedding/embedding.py:155). Tak zmierzono kazdy model na
    wykresie 28 modeli, wiec zeby liczba byla z nimi porownywalna, glowny wynik
    tez musi byc bez zapisu. Ale przemilczenie tego kosztu byloby nieuczciwe:
    milion czasteczek to 12.9 GB w float64 i te dane musza gdzies trafic.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    np.save(path, arr)
    os.sync()
    d = time.perf_counter() - t
    path.unlink(missing_ok=True)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="dobierz batch_size na podprobce przed pelnym pomiarem")
    ap.add_argument("--scratch", default=os.environ.get("MOLPILE_SCRATCH", "molpile_scratch"),
                    help="gdzie zmierzyc koszt zapisu (plik jest kasowany)")
    ap.add_argument("--out", default=os.environ.get("BENCH_REPO", ".") + "/"
                                     "results/molpile/1m/throughput.json")
    args = ap.parse_args()

    ds = joblib.load(PREPARED)
    smiles = ds.data["smiles"].tolist()
    if args.limit:
        smiles = smiles[: args.limit]
    print(f"zbior: {len(smiles):,} czasteczek (po filtrze) | workery: {args.n_jobs} "
          f"z {os.cpu_count()} watkow maszyny", flush=True)

    # rozgrzewka: pierwszy transform placi za rozruch puli loky i importy w workerach
    NewMordredFingerprint(n_jobs=args.n_jobs, verbose=False).transform(
        [Chem.MolFromSmiles("CCO")] * args.n_jobs
    )

    t0 = time.perf_counter()
    mols = [m for m in parse_smiles(smiles, args.n_jobs) if m is not None]
    t_parse = time.perf_counter() - t0
    print(f"parsowanie SMILES: {human(t_parse)} "
          f"({len(mols) / t_parse:,.0f} czast./s)", flush=True)

    batch_size = args.batch_size
    if args.sweep:
        probe = mols[: min(150_000, len(mols))]
        print(f"\ndobor batch_size na {len(probe):,} czasteczkach "
              f"(None = domyslne skfp, jedno zadanie na workera):", flush=True)
        best_rate = -1.0
        for bs in (None, 4000, 1000, 250):
            d, _ = time_transform(probe, args.n_jobs, bs)
            rate = len(probe) / d
            per_task = len(probe) // args.n_jobs if bs is None else bs
            print(f"  batch_size={str(bs):>5} (~{per_task:>6,}/zadanie): "
                  f"{d:7.1f} s  {rate:9,.0f} czast./s", flush=True)
            if rate > best_rate:
                best_rate, batch_size = rate, bs
        print(f"  wybrane: batch_size={batch_size}", flush=True)

    print(f"\npelny pomiar, batch_size={batch_size}...", flush=True)
    fp = NewMordredFingerprint(n_jobs=args.n_jobs, batch_size=batch_size,
                               verbose=False)
    t = time.perf_counter()
    arr = fp.transform(mols)
    t_calc = time.perf_counter() - t
    n_feat = arr.shape[1]

    t_save = time_persist(arr, Path(args.scratch) / "molpile_1m_embeddings.npy")
    n = len(mols)
    rate = n / t_calc
    rate_e2e = n / (t_calc + t_parse)

    print("\n" + "=" * 68)
    print("MOLPILE 1 MLN - MAKSYMALNA PRZEPUSTOWOSC")
    print("=" * 68)
    print(f"  czasteczek            {n:>14,}")
    print(f"  cech na czasteczke    {n_feat:>14,}")
    print(f"  workery               {args.n_jobs:>14}   batch_size={batch_size}")
    print()
    print(f"  deskryptory           {human(t_calc):>14}   {rate:>12,.0f} czast./s")
    print(f"  + parsowanie SMILES   {human(t_calc + t_parse):>14}   {rate_e2e:>12,.0f} czast./s")
    print(f"  deskryptorow/s        {rate * n_feat:>14,.0f}")
    gb = n * n_feat * BYTES_PER_VALUE / 1e9
    print()
    print(f"  zapis {gb:.1f} GB na dysk  {human(t_save):>14}   "
          f"(poza glownym wynikiem - harness tez tego nie mierzy)")
    print(f"  razem z zapisem       {human(t_calc + t_parse + t_save):>14}   "
          f"{n / (t_calc + t_parse + t_save):>12,.0f} czast./s")
    print()
    print("  CO TO ZNACZY W SKALI")
    for label, cnt in [("1 mln (ten pomiar)", n),
                       ("10 mln", 10_000_000),
                       ("100 mln", 100_000_000),
                       (f"{MOLPILE_TOTAL // 1_000_000} mln (caly MolPILE)", MOLPILE_TOTAL)]:
        print(f"    {label:34s} {human(cnt / rate):>12s}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # dopisz wiersz do zbiorczego CSV - zeby kolejne uruchomienia same sie
    # dokladaly i nie trzeba bylo zbierac JSON-ow recznie
    csv_path = out.parent / "pomiary.csv"
    header = ["pomiar", "n_jobs", "batch_size", "proba", "n_molecules", "sekundy",
              "mol_s", "sekundy_parsowanie", "sekundy_zapis", "uwaga"]
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerow(["maks_przepustowosc", args.n_jobs, batch_size, out.stem,
                    n, round(t_calc, 2), round(rate, 1), round(t_parse, 2),
                    round(t_save, 2), ""])
    print(f"  dopisane do {csv_path}")

    out.write_text(json.dumps({
        "n_molecules": n, "n_features": n_feat, "n_jobs": args.n_jobs,
        "batch_size": batch_size, "seconds_descriptors": t_calc,
        "seconds_parse": t_parse, "mol_per_s": rate,
        "mol_per_s_end_to_end": rate_e2e,
        "seconds_persist": t_save,
        "mol_per_s_with_persist": n / (t_calc + t_parse + t_save),
    }, indent=2))
    print(f"\n  zapisane {out}")


if __name__ == "__main__":
    main()
