"""
Speedup per rodzina deskryptorow, wariant OBRONNY: kazda strona liczy rodzine
WRAZ Z JEJ PREREKWIZYTAMI, od zera.

Dlaczego nie profilowaniem: `tottime` przypisuje funkcji tylko jej wlasny czas.
Nasze rodziny sa wtedy "cienkie", bo wspolne przygotowanie (macierze, enumeracja
podgrafow) lezy w innych plikach, a rodziny mordreda "grube", bo on te prace
wykonuje wewnatrz kazdej z nich. Iloraz wychodzi wiec z tego, GDZIE kod lezy, a
nie z tego, ile liczy - i nie da sie tego obronic.

Tutaj obie strony placa za prerekwizyty:
  * mordred: `Calculator(modul)` liczy tylko poddrzewo zaleznosci tej rodziny,
  * nasze: budujemy dokladnie te obiekty, ktore `calculator.compute` podaje jej
    na wejsciu, i mierzymy razem z nimi.

Skutek jest dla nas NIEKORZYSTNY - prerekwizyty liczymy tu raz na rodzine, a w
prawdziwym przebiegu raz na czasteczke dla wszystkich rodzin. Czyli te ilorazy
NIE ZAWIERAJA tego, co daje wspoldzielenie, i sa dolnym oszacowaniem. Za to sa
symetryczne i odporne na pytanie "gdzie sie podzialo przygotowanie".

    python isolated_families.py [--n 200] [--dataset muv] [--repeats 3]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np
from rdkit.Chem import AddHs, GetMolFrags, Mol
from rdkit.Chem.rdchem import Bond
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from skfp.fingerprints._new_mordred.descriptors import (
    autocorrelation, barysz_matrix, chi, detour_matrix, estate,
    extended_topochemical_atom, information_content, molecular_distance_edge,
    molecular_id, path_count, ring_count,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    DistanceMatrix,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import (
    bonds_apply_func, preprocess_mol,
)
from skfp.fingerprints._new_mordred.utils.subgraphs import Subgraphs


class Prereq:
    """Leniwie budowane prerekwizyty dla JEDNEJ czasteczki.

    Kazda rodzina dostaje swoj swiezy egzemplarz, wiec nic nie jest wspoldzielone
    miedzy rodzinami - dokladnie tak, jak liczy mordred przy `Calculator(modul)`.
    """

    def __init__(self, mol):
        self._mol = mol
        self._c: dict = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    mol_regular = property(lambda s: s._get("mr", lambda: preprocess_mol(s._mol)))
    n_frags = property(lambda s: s._get("nf", lambda: len(GetMolFrags(s._mol))))
    props_regular = property(lambda s: s._get(
        "pr", lambda: AtomicProperties.from_mol(s.mol_regular)))
    dm_regular = property(lambda s: s._get(
        "dm", lambda: DistanceMatrix.from_mol(s.mol_regular)))
    rings_regular = property(lambda s: s._get(
        "ri", lambda: ring_count.RingSets(s.mol_regular, s.props_regular)))
    subgraphs = property(lambda s: s._get(
        "sg", lambda: Subgraphs(s.props_regular)))
    mol_hydrogens = property(lambda s: s._get("mh", lambda: AddHs(s.mol_regular)))
    props_hydrogens = property(lambda s: s._get(
        "ph", lambda: AtomicProperties.with_hydrogens_added(
            s.mol_hydrogens, s.props_regular)))
    dm_hydrogens = property(lambda s: s._get(
        "dh", lambda: DistanceMatrix.with_hydrogens_added(
            s.dm_regular, s.props_hydrogens)))
    kekulized_bonds = property(lambda s: s._get("kb", lambda: bonds_apply_func(
        Bond.GetBondType,
        preprocess_mol(Mol(s.mol_regular), kekulize=True, sanitize=False),
        np.intp)))
    # wersja z jawnymi wodorami: calculator.py buduje dzis obie i rozne rodziny
    # biora rozne, wiec skrypt musi umiec podac kazda z nich
    kekulized_bonds_hydrogens = property(lambda s: s._get(
        "kbh", lambda: bonds_apply_func(
            Bond.GetBondType,
            preprocess_mol(Mol(s.mol_hydrogens), kekulize=True, sanitize=False),
            np.intp)))


# rodzina -> (modul mordreda, funkcja liczaca u nas z prerekwizytami)
FAMILIES = {
    "Chi": ("Chi", lambda p: chi.calc(p.props_regular, p.subgraphs)),
    "PathCount": ("PathCount", lambda p: path_count.calc(p.props_regular, p.subgraphs)),
    "InformationContent": ("InformationContent", lambda p: information_content.calc(
        p.mol_hydrogens, p.props_hydrogens, p.kekulized_bonds_hydrogens)),
    "MolecularId": ("MolecularId", lambda p: molecular_id.calc(
        p.props_regular, p.n_frags)),
    "Autocorrelation": ("Autocorrelation", lambda p: autocorrelation.calc(
        p.props_hydrogens, p.dm_hydrogens)),
    "ExtendedTopochemicalAtom": ("ExtendedTopochemicalAtom",
        lambda p: extended_topochemical_atom.calc(
            p.kekulized_bonds, p.kekulized_bonds_hydrogens,
            p.props_regular, p.props_hydrogens,
            p.dm_regular, p.rings_regular, p.n_frags)),
    "MolecularDistanceEdge": ("MolecularDistanceEdge",
        lambda p: molecular_distance_edge.calc(
            p.props_regular, p.dm_regular)),
    "DetourMatrix": ("DetourMatrix", lambda p: detour_matrix.calc(
        p.props_regular, p.dm_regular, p.rings_regular, p.n_frags)),
    "EState": ("EState", lambda p: estate.calc(p.mol_regular)),
    "BaryszMatrix": ("BaryszMatrix", lambda p: barysz_matrix.calc(
        p.props_regular, p.n_frags)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dataset", default="muv")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mordred import Calculator
    import mordred
    from skfp.datasets import moleculenet

    smiles = getattr(moleculenet, f"load_{args.dataset}")()[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(smiles), size=min(args.n, len(smiles)), replace=False)
    mols = [Chem.MolFromSmiles(smiles[i]) for i in idx]
    mols = [m for m in mols if m is not None]
    atoms = float(np.mean([m.GetNumAtoms() for m in mols]))
    print(f"zbior={args.dataset} | {len(mols)} czasteczek | sr. {atoms:.1f} atomu | "
          f"najlepszy z {args.repeats}\n")

    print(f"{'rodzina':26s} {'cech':>5s} {'mordred':>9s} {'nasze':>9s} {'speedup':>9s}")
    print("-" * 62)
    out = []
    for name, (mod_name, ours_fn) in FAMILIES.items():
        calc = Calculator(getattr(mordred, mod_name), ignore_3D=True)
        n_feat = len(calc.descriptors)
        calc(mols[0]); ours_fn(Prereq(mols[0]))              # rozgrzewka

        best_m = min(_time(lambda: [calc(m) for m in mols]) for _ in range(args.repeats))
        best_o = min(_time(lambda: [ours_fn(Prereq(m)) for m in mols])
                     for _ in range(args.repeats))
        out.append((name, n_feat, best_m, best_o))
        print(f"{name:26s} {n_feat:5d} {best_m:8.2f}s {best_o:8.2f}s "
              f"{best_m / best_o:8.1f}x", flush=True)

    print("\n--- LaTeX ---")
    print(r"\begin{tabular}{|l|c|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{descriptor family} & \textbf{features} & \textbf{mordred} & "
          r"\textbf{ours} & \textbf{speedup} \\")
    print(r"\hline")
    for name, f, tm, to in sorted(out, key=lambda r: -r[2] / r[3]):
        print(rf"{name} & {f} & {tm:.2f} s & {to:.2f} s & {tm / to:.1f}$\times$ \\")
        print(r"\hline")
    print(r"\end{tabular}")


def _time(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


if __name__ == "__main__":
    main()
