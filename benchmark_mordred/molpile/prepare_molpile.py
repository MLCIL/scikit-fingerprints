"""
Buduje przygotowany zbior MolPILE dla harnessu measure_embedding.py.

Zrodlo: scikit-fingerprints/MolPILE na HuggingFace, plik molpile_subset_1M.parquet,
skonwertowany raz do CSV. Caly MolPILE ma 222 mln czasteczek (arXiv 2509.18353),
wiec ten milion to 0.45% korpusu.

Format musi byc taki, jakiego oczekuje harness: kolumna "smiles", kolumna "graph"
(Dataset.labels bezwarunkowo ja odrzuca, wiec musi istniec), oraz splity. MolPILE
to korpus do pretreningu, nie zbior z etykietami, wiec splity sa sztuczne - harness
ich nie uzywa do pomiaru czasu.

    python prepare_molpile.py [liczba_czasteczek]
"""

import os
import signal
import sys
from contextlib import contextmanager

import joblib
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger

sys.path.insert(0, ".")
from src.common.types import Dataset

RDLogger.DisableLog("rdApp.*")


def _check_chunk(smiles: list[str], lo: int, hi: int, per_mol_s: float) -> list[int]:
    """Indeksy z [lo, hi), ktorych nasza implementacja nie policzy w limicie.

    Dwa powody odpadniecia sa tu traktowane tak samo, bo dla pomiaru znacza to
    samo - czasteczka nie wchodzi do zbioru:
      * wyjatek (organometaliki: nieskonczona walencja, brak zbieznosci LAPACK),
      * przekroczenie limitu czasu.

    Limit jest konieczny, bo MolPILE zawiera fullereny i fragmenty grafenowe
    (126 atomow, 37 skondensowanych pierscieni), a DetourMatrix szuka w nich
    najdluzszej sciezki - problem NP-trudny, ktory na takim ukladzie nie konczy
    sie nigdy. Bez limitu 95 workerow stalo 8.5 h w _walk_every_path.

    Najpierw cala paczka za jednym zamachem; dopiero gdy padnie albo przekroczy
    limit, idziemy sztuka po sztuce, wiec zdrowe paczki nie placa nic.
    """
    from skfp.fingerprints import NewMordredFingerprint

    fp = NewMordredFingerprint(n_jobs=1, verbose=False)
    mols = [Chem.MolFromSmiles(s) for s in smiles[lo:hi]]
    bad = [lo + i for i, m in enumerate(mols) if m is None]
    pairs = [(lo + i, m) for i, m in enumerate(mols) if m is not None]

    # zdrowa paczka 500 czasteczek idzie ~7 s (13.5 ms/szt.), wiec 0.2 s na
    # czasteczke to ~14-krotny zapas; dopiero po tym schodzimy sztuka po sztuce
    with _deadline(max(30.0, 0.2 * len(pairs))):
        try:
            fp.transform([m for _, m in pairs])
            return bad
        except Exception:
            pass

    for idx, mol in pairs:
        with _deadline(per_mol_s):
            try:
                fp.transform([mol])
            except Exception:
                bad.append(idx)
    return bad


@contextmanager
def _deadline(seconds: float):
    """Twardy limit czasu na blok, przez SIGALRM.

    Deskryptory licza sie w czystym Pythonie, wiec sygnal dochodzi miedzy
    instrukcjami bajtkodu. Dziala w workerze loky, bo tam kod idzie w watku
    glownym procesu.
    """
    def fire(_sig, _frm):
        raise TimeoutError("przekroczony limit czasu na czasteczke")

    old = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    except TimeoutError:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _find_uncomputable(smiles: list[str], chunk: int = 500,
                       per_mol_s: float = 5.0) -> list[int]:
    """Jedno przejscie po zbiorze na wszystkich rdzeniach; to nie jest pomiar."""
    n_jobs = max(1, int((os.cpu_count() or 1) * 0.75))
    ranges = [(i, min(i + chunk, len(smiles))) for i in range(0, len(smiles), chunk)]
    print(f"filtr: {len(ranges)} paczek na {n_jobs} rdzeniach, "
          f"limit {per_mol_s:g} s na czasteczke...", flush=True)
    parts = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_check_chunk)(smiles, lo, hi, per_mol_s) for lo, hi in ranges
    )
    return sorted(i for part in parts for i in part)


CSV = os.environ.get("MOLPILE_CSV", "molpile_1M.csv")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
NAME = f"molpile-{N // 1000}k" if N < 1_000_000 else "molpile-1m"

df = pd.read_csv(CSV, nrows=N)
print(f"{NAME}: wczytane {len(df)} SMILES-ow z {CSV}")

# harness kanonizuje SMILES-y w build_dataset, wiec robimy to samo - inaczej
# pomiar parsowania roznilby sie od tego, ktory zmierzylismy na MUV
df["smiles"] = df["SMILES"].map(Chem.CanonSmiles)
df = df.drop(columns=["SMILES"])
# MolPILE nie ma etykiet; kolumna pozorna, zeby Dataset.labels mial co zwrocic
df["dummy_label"] = 0.0
df["graph"] = None

# MolPILE zawiera czasteczki, ktorych nasza implementacja nie odda: rzuca na nich
# wyjatkiem (OverflowError w _implicit_hydrogen_count przy nieskonczonej walencji)
# albo nie konczy liczenia (fullereny - DetourMatrix szuka najdluzszej sciezki w
# ukladzie 37 skondensowanych pierscieni). Jedno i drugie zabija CALA paczke joblib,
# wiec bez tego filtra pomiar sie nie konczy. Filtrujemy raz, tutaj, poza stoperem;
# liczba po filtrze jest widoczna w wynikach.
df = df.reset_index(drop=True)
before = len(df)
bad = _find_uncomputable(df["smiles"].tolist())
if bad:
    df = df.drop(index=bad).reset_index(drop=True)
print(f"odfiltrowane {len(bad)} z {before} ({100 * len(bad) / before:.3f}%) - "
      f"czasteczki nieobliczalne: wyjatek albo przekroczony limit czasu")
pd.Series(sorted(bad), name="index").to_csv(f"data/prepared/{NAME}-odrzucone.csv", index=False)

n = len(df)
splits = {"train": list(range(n)), "valid": [], "test": []}
joblib.dump(
    Dataset(name=NAME, task="regression", data=df, splits=splits),
    f"data/prepared/{NAME}.joblib",
)
print(f"zapisane data/prepared/{NAME}.joblib ({n} czasteczek, {n // 10_000} pelnych paczek)")
