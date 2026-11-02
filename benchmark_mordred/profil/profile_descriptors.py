"""
Ktore rodziny deskryptorow przyspieszyly najmocniej wzgledem oryginalnego mordreda.

Nie liczymy rodzin osobno, bo nasza implementacja dzieli miedzy nie wspolne
przygotowanie (macierz sasiedztwa, macierz odleglosci, wlasciwosci atomow,
podgrafy) - rozcinanie jej na kawalki zniekształciloby obraz i ukryloby czesc
przewagi. Zamiast tego profilujemy OBA przebiegi w calosci i przypisujemy czas
wlasny (`tottime`) do pliku, w ktorym kod sie wykonuje.

`tottime`, a nie `cumtime`, bo cumtime liczylby wspolne pomocnicze wielokrotnie.
Dzieki temu kod wspoldzielony (macierze, utils) trafia do wlasnego kubelka i
widac go osobno, zamiast doliczac go po cichu do rodzin.

    python profile_descriptors.py [--n 300] [--dataset hiv]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import cProfile
import pstats
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# nasze pliki sa snake_case, mordreda CamelCase - sprowadzamy do wspolnej postaci
def _stem(path: str) -> str:
    q = Path(path)
    # mordred trzyma czesc deskryptorow w pakietach, gdzie kod jest w __init__.py
    stem = q.parent.name if q.stem == "__init__" else q.stem
    return re.sub(r"[^a-z0-9]", "", stem.lower())


ALIAS = {"extendedtopochemicalatom": "eta", "extendedtopochemicalatoms": "eta"}

# moduly mordreda lezace wprost w mordred/, ale bedace infrastruktura, nie
# rodzinami deskryptorow - inaczej zawyzalyby kolumne "rodziny"
MORDRED_INFRA = {"atomicproperty", "matrixattributes", "util", "graphmatrix",
                 "error", "descriptor", "context"}


def klasyfikuj(path: str) -> tuple[str, str]:
    """(kubelek, nazwa) - kubelek mowi, czy to rodzina deskryptorow, czy narzut."""
    if path in ("~", "<string>"):
        return "wbudowane", "wbudowane (C)"
    nazwa = ALIAS.get(_stem(path), _stem(path))
    for lib in ("networkx", "numpy", "scipy", "sklearn", "pandas"):
        if f"/{lib}/" in path:
            return "biblioteki", lib
    if "/rdkit/" in path:
        return "biblioteki", "rdkit"
    # mordred: deskryptory leza wprost w mordred/, framework w mordred/_base/
    if "/site-packages/mordred/" in path:
        if "/_base/" in path or nazwa in MORDRED_INFRA:
            return "framework", "mordred (infrastruktura)"
        return "rodzina", nazwa
    # nasze: deskryptory w _new_mordred/descriptors/, reszta to nasz framework
    if "/_new_mordred/descriptors/" in path:
        return "rodzina", nazwa
    if "/_new_mordred/" in path or "/skfp/" in path:
        return "framework", "skfp"
    return "biblioteki", "python (stdlib)"


def profile(fn) -> dict[tuple[str, str], float]:
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    out: dict[tuple[str, str], float] = defaultdict(float)
    for (fpath, _l, _n), (_cc, _nc, tottime, _ct, _cal) in pstats.Stats(pr).stats.items():
        out[klasyfikuj(fpath)] += tottime
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dataset", default="muv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from skfp.datasets import moleculenet

    loader = getattr(moleculenet, f"load_{args.dataset}")
    smiles = loader()[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(smiles), size=min(args.n, len(smiles)), replace=False)
    mols = [Chem.MolFromSmiles(smiles[i]) for i in idx]
    mols = [m for m in mols if m is not None]
    atoms = np.mean([m.GetNumAtoms() for m in mols])
    print(f"zbior={args.dataset} | {len(mols)} czasteczek | sr. {atoms:.1f} atomu\n")

    from skfp.fingerprints import NewMordredFingerprint

    fp = NewMordredFingerprint(n_jobs=1, verbose=False)
    fp.transform(mols[:2])  # rozgrzewka
    print("profiluje nasza implementacje...", flush=True)
    nasze = profile(lambda: fp.transform(mols))

    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    calc(mols[0])  # rozgrzewka
    print("profiluje oryginalnego mordreda...", flush=True)
    mordred_p = profile(lambda: [calc(m) for m in mols])

    def sumuj(d, kubelek):
        return sum(v for (k, _n), v in d.items() if k == kubelek)

    tot_n, tot_m = sum(nasze.values()), sum(mordred_p.values())
    print(f"\ncalkowity czas w profilu: nasze {tot_n:.2f} s | mordred {tot_m:.2f} s "
          f"-> {tot_m / tot_n:.1f}x")
    print("(cProfile dodaje narzut, wiec iloraz jest przyblizeniem - liczby bezwzgledne")
    print(" sa w results/, tu chodzi o ROZKLAD czasu miedzy rodziny)\n")

    print("=== GDZIE SCHODZI CZAS (kubelki) ===")
    print(f"{'kubelek':24s} {'mordred':>10s} {'nasze':>10s}")
    for kub, opis in [("rodzina", "rodziny deskryptorow"), ("framework", "framework wlasny"),
                      ("biblioteki", "biblioteki zewnetrzne"), ("wbudowane", "wbudowane (C)")]:
        m, n = sumuj(mordred_p, kub), sumuj(nasze, kub)
        print(f"  {opis:22s} {m:9.3f}s {n:9.3f}s   ({100*m/tot_m:4.1f}% / {100*n/tot_n:4.1f}%)")

    print("\n=== BIBLIOTEKI ZEWNETRZNE, z rozbiciem ===")
    for d, label in ((mordred_p, "mordred"), (nasze, "nasze")):
        czesci = sorted(((n, v) for (k, n), v in d.items() if k == "biblioteki"),
                        key=lambda x: -x[1])
        print(f"  {label:8s} " + ", ".join(f"{n} {v:.2f}s" for n, v in czesci[:5]))

    r_m = {n: v for (k, n), v in mordred_p.items() if k == "rodzina"}
    r_n = {n: v for (k, n), v in nasze.items() if k == "rodzina"}
    print("\n=== RODZINY DESKRYPTOROW, wg udzialu w czasie mordreda ===")
    print(f"{'rodzina':26s} {'mordred':>9s} {'nasze':>9s} {'przysp.':>10s} {'udzial':>8s}")
    print("-" * 68)
    for k in sorted(set(r_m) | set(r_n), key=lambda k: -r_m.get(k, 0.0)):
        tm, tn = r_m.get(k, 0.0), r_n.get(k, 0.0)
        if max(tm, tn) < 0.005:
            continue
        sp = f"{tm/tn:9.1f}x" if tn > 1e-6 else "  (brak u nas)"
        gwiazd = " <-- tylko u nas" if tm < 1e-6 else ""
        print(f"{k:26s} {tm:8.3f}s {tn:8.3f}s {sp} {100*tm/tot_m:7.1f}%{gwiazd}")


if __name__ == "__main__":
    main()
