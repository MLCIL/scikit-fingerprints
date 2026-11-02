"""
Benchmark: przepisany Mordred (skfp) vs mordred-community.

Wszystkie pomiary sa JEDNOWATKOWE (n_jobs=1), poza zestawem "threads".
Kazdy wynik ladzie w CSV natychmiast (flush+fsync) - ubicie procesu nic nie gubi.
Restart z tym samym --out pomija juz zmierzone kombinacje (wznawialnosc).

ZESTAWY (jeden na wykres):
  calib      kalibracja: 500 czasteczek, ~10 min. ZAWSZE ODPAL TO NAJPIERW.
  speedup    wykres 1: speedup na 11 datasetach MoleculeNet (cap 10k losowych)
  size       wykresy 2+3: throughput vs rozmiar czasteczki (drabinka peptydowa)
  threads    wykres 4: throughput vs liczba watkow (HIV 10k + peptydy)  [WYLACZNIE]
  reference  wykres 5: slupek do wykresu 10k czasteczek                 [WYLACZNIE]

PROTOKOL NA SERWERZE
  # 1. kalibracja - sprawdz czy budzet sie zgadza
  python bench_run.py --suite calib --out calib.csv

  # 2. FAZA A: rownolegle, jeden proces na dataset (kontencja nie psuje ILORAZU)
  for d in freesolv esol sider clintox bace bbbp lipophilicity tox21 toxcast hiv muv; do
      python bench_run.py --suite speedup --only $d --out res_speedup_$d.csv &
  done; wait

  for L in 5 10 20 40 80 100; do
      python bench_run.py --suite size --only synthpep_$L --out res_size_$L.csv &
  done
  for P in ace_vaxinpad hiv_lpv hiv_nvp; do
      python bench_run.py --suite size --only preactor_$P \
          --peptide-dir /sciezka/do/peptides_.../data/PeptideReactor \
          --out res_size_$P.csv &
  done; wait

  # 3. FAZA B: maszyna MUSI byc pusta - to pomiary bezwzgledne
  python bench_run.py --suite threads   --out res_threads.csv
  python bench_run.py --suite reference --out res_reference.csv

  # 4. analiza
  python bench_analyze.py --inputs "res_*.csv" --outdir figures

UWAGA: deskryptory 3D / GEOM sa swiadomie POZA zakresem - oryginalny Mordred
ma zepsute liczenie konformerow, wiec uczciwy pomiar 3D wymaga osobnego podejscia.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

# n_jobs=1 ogranicza tylko joblib. Dekompozycje spektralne ida przez OpenBLAS,
# ktory bez tego bierze wszystkie rdzenie - nasza implementacja dostawala 20+
# rdzeni, a mordred (czysty Python) jeden, wiec "speedup na 1 watku" bylby
# porownaniem 20 rdzeni z jednym. Musi byc PRZED importem numpy: OpenBLAS
# czyta te zmienne przy ladowaniu biblioteki.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CSV_FIELDS = [
    "timestamp",
    "suite",
    "impl",
    "dataset",
    "device",
    "n_molecules",
    "n_jobs",
    "repeat",
    "wall_seconds",
    "seconds_per_mol",
    "throughput_mol_s",
    "n_features",
    "mean_atoms",
    "mean_bonds",
    "commit",
    "host",
    "status",
    "error",
]

MOLECULENET = [
    "freesolv",
    "esol",
    "sider",
    "clintox",
    "bace",
    "bbbp",
    "lipophilicity",
    "tox21",
    "toxcast",
    "hiv",
    "muv",
]
SUBSAMPLE_CAP = 10_000  # HIV i MUV tna sie do losowych 10k

# ExpansionRx / OpenADMET: dziesiec realnych zbiorow ADME. Wbrew nazwie "male
# czasteczki" srednia to 28.6 atomu, czyli wiecej niz HIV (25.5) i MUV (24.3).
# Ich cecha jest CIASNY rozklad - mediana 28, maksimum 78, gdy HIV ma 214. To
# czyni je testem na dane bez ciezkiego ogona, gdzie efekt marudera nie wystepuje.
EXPANSIONRX = {
    "erx_logd": "LogD",
    "erx_ksol": "KSOL",
    "erx_mlm_clint": "MLM CLint",
    "erx_hlm_clint": "HLM CLint",
    "erx_caco2_efflux": "Caco-2 Permeability Efflux",
    "erx_caco2_papp": "Caco-2 Permeability Papp A>B",
    "erx_mppb": "MPPB",
    "erx_mbpb": "MBPB",
    "erx_rlm_clint": "RLM CLint",
    "erx_mgmb": "MGMB",
}
# liczebnosci zmierzone raz; sluza tylko do wazenia postepu przed zaladowaniem
EXPANSIONRX_SIZES = {
    "erx_logd": 7309, "erx_ksol": 7298, "erx_mlm_clint": 5692,
    "erx_hlm_clint": 4541, "erx_caco2_efflux": 3777, "erx_caco2_papp": 3773,
    "erx_mppb": 1756, "erx_mbpb": 1426, "erx_rlm_clint": 633, "erx_mgmb": 431,
}
# w zestawie threads najwiekszy zbior dostaje wlasne liczebnosci, tak jak hiv
THREADS_ERX = "erx_logd"
# Krotsza drabinka watkow dla ExpansionRx, bo LogD ma tylko 7309 czasteczek.
# Przy 64 workerach nasz worker dostalby 114 sztuk po ~0.015 s, czyli 1.7 s pracy
# na kilkusekundowy start puli - mierzylibysmy narzut, nie skalowalnosc, dokladnie
# ten blad, ktory poprawialem w panelu peptydowym. Przy 16 workerach na workera
# przypada ~7 s, co juz jest sensowne. Wiecej danych po prostu nie ma.
THREAD_COUNTS_ERX = [1, 2, 4, 8, 16]

# drabinka peptydowa: dlugosc w aminokwasach -> ile czasteczek.
# Przy duzych rozmiarach kilka sztuk wystarcza: koszt jest zdominowany przez
# deterministyczne O(n^3), wiec wariancja jest mala.
# 160 AA (~1270 atomow) domyka jedyna realna luke w drabince: zmierzone rozmiary
# ida 42-84-101-167-334-667-764-832-1982, czyli wszystkie odstepy sa rzedu
# 1.1-2.0x poza ostatnim (2.38x). A to wlasnie tam twierdzimy plateau
# przyspieszenia, wiec bez tego punktu teza opiera sie na krancach przedzialu.
PEPTIDE_LADDER = {5: 200, 10: 100, 20: 50, 40: 20, 80: 10, 100: 5, 160: 5}

# Realne peptydy z repo papera promotora (data/PeptideReactor). Trzy klasy
# rozmiaru; liczebnosc maleje, bo koszt rosnie ~n^2.8. Przy duzych czasteczkach
# kilka sztuk wystarcza - wariancja jest mala, koszt deterministyczny.
# UWAGA: hiv_nvp to dlugi biegun calego przebiegu (~2800 s na czasteczke razem
# dla obu implementacji). Jesli budzet nie domyka, tnij najpierw tutaj.
PEPTIDE_REACTOR_SETS = {
    "ace_vaxinpad": 200,  # krotkie peptydy
    "hiv_lpv": 10,  # ~100 AA, ~770 atomow
    "hiv_nvp": 2,  # ~241 AA, ~1980 atomow
}
# srednia liczba atomow w kazdym zbiorze PeptideReactor, zmierzona raz
PEPTIDE_REACTOR_ATOMS = {"ace_vaxinpad": 120, "hiv_lpv": 765, "hiv_nvp": 1990}
# przyblizona srednia liczba atomow w datasetach MoleculeNet; sluzy WYLACZNIE do
# wazenia postepu, zanim dataset sie zaladuje, i jest zastepowana pomiarem
DATASET_ATOMS = {
    "freesolv": 9,
    "esol": 13,
    "sider": 34,
    "clintox": 25,
    "bace": 34,
    "bbbp": 24,
    "lipophilicity": 27,
    "tox21": 19,
    "toxcast": 19,
    "hiv": 26,
    "muv": 23,
}
# koszt rosnie mniej wiecej tak z liczba atomow; wykorzystywane tylko do wazenia
# zadan miedzy soba, wiec dokladnosc wykladnika nie jest krytyczna
COST_EXPONENT = 2.4
ATOMS_PER_RESIDUE = 7.9
# Kolejne potegi dwojki daja rowne odstepy na osi log2, a 64 to liczba rdzeni
# fizycznych maszyny - powyzej wchodzi SMT i dwa watki dziela jeden rdzen, wiec
# ostatni punkt przestalby byc porownywalny z reszta.
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64]
# Zestaw "threads": kazda implementacja dostaje tyle czasteczek, zeby pomiar na
# najszerszym punkcie (64 workery) trwal ~10 s. Rowna liczebnosc dla obu byla
# bledem metodologicznym: przy 320 peptydach mordred wyciagal na 64 workerach 67%
# efektywnosci, a nasza implementacja 19% - nie dlatego, ze skaluje sie gorzej,
# ale dlatego, ze jej worker konczyl swoje 5 czasteczek szybciej, niz trwal start
# procesu. Po zwiekszeniu liczebnosci ta sama implementacja daje 49%, a wykres
# przestaje sugerowac, ze mordred skaluje sie lepiej.
# Throughput jest wielkoscia na czasteczke, wiec krzywe pozostaja porownywalne;
# 50k dla hiv oznacza "caly dataset", bo ma go 41127.
# Zestaw threads jest najdrozszy (punkt jednowatkowy to polowa jego kosztu przy
# idealnym skalowaniu), a mierzy sie na pustej maszynie, wiec rozrzut miedzy
# powtorzeniami jest maly - dwa wystarczaja na kontrole powtarzalnosci.
THREADS_REPEATS = 2
THREADS_PEPTIDE_LENGTH = 20
THREADS_COUNTS = {
    "ours": {"hiv": 50_000, f"synthpep_{THREADS_PEPTIDE_LENGTH}": 3_200,
             # LogD ma tylko 7309 czasteczek, wiec bierzemy caly zbior; przy 64
             # workerach nasz pomiar jest wtedy krotki (~2 s) i to trzeba przy
             # wykresie powiedziec, bo zaniza nasza efektywnosc
             "erx_logd": 50_000},
    "mordred": {"hiv": 6_000, f"synthpep_{THREADS_PEPTIDE_LENGTH}": 320,
                "erx_logd": 7_309},
}
# kalibracja ma zajac kilka minut, wiec swoje wlasne, male liczebnosci
CALIB_COUNTS = {"bbbp": 300, "synthpep_20": 5, "synthpep_80": 1}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Warmup placi jednorazowy koszt importow i leniwych cache, wiec liczy sie tylko
# to, ze cokolwiek przeszlo cala sciezke - nie rozmiar czasteczki. Branie czegos
# ze zmierzonego zbioru bylo drogie: na PeptideReactor kazda czasteczka ma ~2000
# atomow, czyli sam warmup dokladal pelne dodatkowe liczenie (mordred ~20 min).
WARMUP_SMILES = "CCO"


# --------------------------------------------------------------------------
# dane
# --------------------------------------------------------------------------
def load_mols(
    dataset: str,
    rng: np.random.Generator,
    peptide_dir: Path | None,
    count: int | None = None,
) -> list[Chem.Mol]:
    """``count`` nadpisuje domyslna liczebnosc zbioru (patrz THREADS_COUNTS)."""
    if dataset.startswith("synthpep_"):
        length = int(dataset.split("_")[1])
        return synthetic_peptides(length, count or PEPTIDE_LADDER.get(length, 10), rng)

    if dataset.startswith("preactor_"):
        return peptide_reactor(dataset.split("_", 1)[1], peptide_dir, rng)

    if dataset.startswith("csv:"):
        # dowolny plik SMILES-ow, zeby nowy zbior nie wymagal edycji tego pliku:
        #   --only csv:/sciezka/do/pliku.csv
        import pandas as pd

        df = pd.read_csv(dataset[4:])
        col = next(
            (c for c in df.columns if c.strip().lower() == "smiles"), df.columns[0]
        )
        smiles = list(df[col])
        if count and len(smiles) > count:
            idx = rng.choice(len(smiles), size=count, replace=False)
            smiles = [smiles[i] for i in sorted(idx)]
        return parse(smiles)

    if dataset in EXPANSIONRX:
        from skfp.datasets.expansionrx.benchmark import (
            EXPANSIONRX_DATASET_NAME_TO_LOADER_FUNC,
        )

        loader = EXPANSIONRX_DATASET_NAME_TO_LOADER_FUNC[EXPANSIONRX[dataset]]
        smiles = list(loader(as_frame=True)["SMILES"])
        if count and len(smiles) > count:
            idx = rng.choice(len(smiles), size=count, replace=False)
            smiles = [smiles[i] for i in sorted(idx)]
        return parse(smiles)

    from skfp.datasets import moleculenet

    df = getattr(moleculenet, f"load_{dataset}")(as_frame=True)
    col = "SMILES" if "SMILES" in df.columns else df.columns[0]
    smiles = list(df[col])

    # losowa podprobka ze stalym ziarnem: obie implementacje dostaja te same czasteczki
    cap = count or SUBSAMPLE_CAP
    if len(smiles) > cap:
        idx = rng.choice(len(smiles), size=cap, replace=False)
        smiles = [smiles[i] for i in sorted(idx)]

    return parse(smiles)


def peptide_reactor(
    name: str, peptide_dir: Path | None, rng: np.random.Generator
) -> list[Chem.Mol]:
    """Peptydy z repo peptides_molecular_fingerprints_classification (FASTA)."""
    if peptide_dir is None:
        raise ValueError(
            "PeptideReactor wymaga --peptide-dir sciezka/do/data/PeptideReactor"
        )
    seqs, cur = [], ""
    for line in (peptide_dir / name / "seqs.fasta").read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append(cur)
            cur = ""
        else:
            cur += line.strip()
    if cur:
        seqs.append(cur)
    mols = [Chem.MolFromSequence(s) for s in seqs]
    return [m for m in mols if m is not None]


def synthetic_peptides(length: int, n: int, rng: np.random.Generator) -> list[Chem.Mol]:
    out = []
    while len(out) < n:
        seq = "".join(rng.choice(list(AMINO_ACIDS), size=length))
        mol = Chem.MolFromSequence(seq)
        if mol is not None:
            out.append(mol)
    return out


def parse(smiles: list[str]) -> list[Chem.Mol]:
    """Parsowanie POZA pomiarem: mierzymy deskryptory, nie parsowanie SMILES."""
    return [m for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None]


def drop_uncomputable(mols: list[Chem.Mol]) -> tuple[list[Chem.Mol], int]:
    """
    Wyrzuca czasteczki, na ktorych nasza implementacja rzuca wyjatkiem.

    W HIV sa trzy takie na 41127 - organometaliki, na ktorych albo walencja metalu
    wychodzi nieskonczona (OverflowError w extended_topochemical_atom), albo LAPACK
    nie zbiega (LinAlgError). Jedna z nich trafia w podprobke 10k, wiec bez tego
    filtra caly pomiar na HIV konczy sie bledem i wykresy 1 i 5 nie maja slupka.

    Mordred na tych czasteczkach nie przerywa - wstawia placeholder bledu i liczy
    dalej - wiec zeby obie implementacje mierzyly TO SAMO, wyrzucamy je z obu
    zbiorow. Liczba czasteczek po filtrze idzie do CSV, wiec wykluczenie jest widoczne
    w danych. Filtr jest jednowatkowy, jak pomiary, i kosztuje jedno przejscie.
    """
    from skfp.fingerprints import NewMordredFingerprint

    fp = NewMordredFingerprint(n_jobs=1, verbose=False)
    good: list[Chem.Mol] = []
    dropped = 0
    for start in range(0, len(mols), 200):
        chunk = mols[start : start + 200]
        try:
            fp.transform(chunk)
            good.extend(chunk)
        except Exception:
            # blad dotyczy pojedynczych czasteczek, wiec tylko taka paczke
            # przechodzimy sztuka po sztuce
            for mol in chunk:
                try:
                    fp.transform([mol])
                    good.append(mol)
                except Exception:
                    dropped += 1
    return good, dropped


# --------------------------------------------------------------------------
# implementacje
# --------------------------------------------------------------------------
def run_ours(mols, n_jobs: int) -> tuple[int, int]:
    from skfp.fingerprints import NewMordredFingerprint

    X = NewMordredFingerprint(n_jobs=n_jobs, verbose=False).transform(mols)
    return len(X), (X.shape[1] if X.ndim == 2 else 0)


def run_mordred(mols, n_jobs: int) -> tuple[int, int]:
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    # nie materializujemy wynikow: mordred zwraca obiekty Pythona i przy duzych
    # zbiorach pelna lista wysadzilaby RAM
    n = width = 0
    for row in calc.map(mols, nproc=n_jobs, quiet=True):
        n += 1
        width = width or len(row)
    return n, width


IMPLS = {"ours": run_ours, "mordred": run_mordred}


# --------------------------------------------------------------------------
# zestawy -> lista zadan (dataset, n_jobs)
# --------------------------------------------------------------------------
def build_jobs(suite: str, only: str | None) -> list[tuple[str, int]]:
    # plik podany wprost przez --only csv:<sciezka> nie jest czescia zadnego
    # zestawu, wiec sam stanowi cale zadanie - jednowatkowo, jak reszta ilorazow
    if only and only.startswith("csv:"):
        return [(only, 1)]
    if suite == "calib":
        return [("bbbp", 1), ("synthpep_20", 1), ("synthpep_80", 1)]
    if suite == "speedup":
        return [(d, 1) for d in MOLECULENET]
    if suite == "size":
        # generowane peptydy daja gladka, kontrolowana krzywa rozmiaru,
        # a PeptideReactor realne dane z papera - potrzebujemy obu
        return [(f"synthpep_{L}", 1) for L in PEPTIDE_LADDER] + [
            (f"preactor_{name}", 1) for name in PEPTIDE_REACTOR_SETS
        ]
    if suite == "threads":
        return [("hiv", n) for n in THREAD_COUNTS] + [
            (f"synthpep_{THREADS_PEPTIDE_LENGTH}", n) for n in THREAD_COUNTS
        ]
    if suite == "expansionrx":
        # dziesiec realnych zbiorow ADME, jednowatkowo - to ILORAZY, wiec moga
        # liczyc sie rownolegle, jeden proces na zbior
        return [(name, 1) for name in EXPANSIONRX]
    if suite == "erx_threads":
        return [(THREADS_ERX, n) for n in THREAD_COUNTS_ERX]
    if suite == "reference":
        return [("hiv", 1)]
    raise ValueError(f"nieznany zestaw: {suite}")


# --------------------------------------------------------------------------
# postep i szacowany czas zakonczenia
# --------------------------------------------------------------------------
def expected_shape(suite: str, dataset: str, args) -> tuple[int, float]:
    """
    Ile czasteczek i jakiej wielkosci ma dane zadanie, zanim je zaladujemy.

    Sluzy do wazenia postepu z gory. Po zaladowaniu datasetu wartosci sa
    zastepowane zmierzonymi, wiec przyblizenie ma znaczenie tylko dla zadan
    jeszcze nietknietych.
    """
    if dataset.startswith("synthpep_"):
        length = int(dataset.split("_")[1])
        count = PEPTIDE_LADDER.get(length, 10)
        atoms = length * ATOMS_PER_RESIDUE
    elif dataset.startswith("preactor_"):
        name = dataset.split("_", 1)[1]
        count = PEPTIDE_REACTOR_SETS.get(name, 5)
        atoms = float(PEPTIDE_REACTOR_ATOMS.get(name, 500))
    elif dataset in EXPANSIONRX:
        count = EXPANSIONRX_SIZES.get(dataset, 4000)
        atoms = 28.6  # zmierzona srednia dla calej rodziny, rozklad jest ciasny
    else:
        count = DATASET_SIZES.get(dataset, 2000)
        atoms = float(DATASET_ATOMS.get(dataset, 25))

    if suite in ("threads", "erx_threads"):
        count = threads_count(dataset, args.impls) or count
    if args.max_size:
        count = min(count, args.max_size)
    elif suite == "calib":
        count = min(count, CALIB_COUNTS.get(dataset, 5))
    return count, atoms


def threads_count(dataset: str, impls: list[str]) -> int | None:
    """
    Liczebnosc zbioru dla zestawu "threads", osobna dla kazdej implementacji.

    Wywolanie musi dotyczyc jednej implementacji - inaczej nie da sie dobrac
    liczebnosci tak, zeby zaden z workerow nie glodowal.
    """
    if len(impls) != 1:
        raise SystemExit(
            "zestaw threads wymaga jednej implementacji na wywolanie: "
            "--impls ours ALBO --impls mordred"
        )
    return THREADS_COUNTS.get(impls[0], {}).get(dataset)


class Progress:
    """
    Szacuje czas zakonczenia, wazac zadania iloscia pracy zamiast ich liczba.

    Pomiar dwoch peptydow trwa tyle co dwoch tysiecy malych czasteczek, wiec
    postep liczony w "zrobionych pomiarach" bylby bezuzyteczny. Jednostka pracy
    to ``liczba_czasteczek * atomow ** COST_EXPONENT / n_jobs``, a przelicznik
    na sekundy bierze sie z tego, co juz zmierzono - dzieki czemu ETA sam
    kalibruje sie do maszyny po pierwszym pomiarze.

    Przelicznik jest osobny dla kazdej implementacji: mordred jest kilkukrotnie
    wolniejszy od naszej implementacji, wiec jeden wspolny przelicznik dawal ETA
    zanizone kilkukrotnie do samego konca zadania.
    """

    def __init__(self, jobs: list[tuple[str, int]], suite: str, args) -> None:
        self._units: dict[tuple[str, int], float] = {}
        for dataset, n_jobs in jobs:
            count, atoms = expected_shape(suite, dataset, args)
            self._units[(dataset, n_jobs)] = self._work(count, atoms, n_jobs)
        self._impls = list(args.impls)
        self._repeats = max(1, args.repeats)
        total = sum(self._units.values()) * self._repeats
        self._remaining = {impl: total for impl in self._impls}
        self._done_units = {impl: 0.0 for impl in self._impls}
        self._done_seconds = {impl: 0.0 for impl in self._impls}

    @staticmethod
    def _work(count: int, atoms: float, n_jobs: int) -> float:
        return count * (atoms**COST_EXPONENT) / max(1, n_jobs)

    def refine(self, dataset: str, n_jobs: int, count: int, atoms: float) -> None:
        """Podmien oszacowanie na rzeczywisty rozmiar, gdy dataset sie zaladuje."""
        key = (dataset, n_jobs)
        if key not in self._units:
            return
        actual = self._work(count, atoms, n_jobs)
        delta = (actual - self._units[key]) * self._repeats
        for impl in self._impls:
            self._remaining[impl] += delta
        self._units[key] = actual

    def skipped(self, dataset: str, n_jobs: int, impl: str) -> None:
        """Pomiar wznowiony z CSV - praca odpada z puli, ale bez czasu."""
        units = self._units.get((dataset, n_jobs), 0.0)
        self._remaining[impl] = max(0.0, self._remaining[impl] - units)

    def measured(self, dataset: str, n_jobs: int, seconds: float, impl: str) -> str:
        units = self._units.get((dataset, n_jobs), 0.0)
        self._done_units[impl] += units
        self._done_seconds[impl] += seconds
        self._remaining[impl] = max(0.0, self._remaining[impl] - units)

        rates = {
            i: self._done_seconds[i] / self._done_units[i]
            for i in self._impls
            if self._done_units[i] > 0
        }
        if not rates:
            return ""
        # implementacja jeszcze niezmierzona dostaje sredni przelicznik z tych,
        # ktore juz sa - wtedy ETA jest tylko dolnym ograniczeniem
        fallback = sum(rates.values()) / len(rates)
        remaining_s = sum(
            self._remaining[i] * rates.get(i, fallback) for i in self._impls
        )
        done_units = sum(self._done_units.values())
        remaining_units = sum(self._remaining.values())
        share = done_units / (done_units + remaining_units)
        finish = datetime.now().astimezone() + timedelta(seconds=remaining_s)
        bound = "" if len(rates) == len(self._impls) else "  (dolne ograniczenie)"
        return (
            f"      postep {share * 100:5.1f}% | pozostalo {_hms(remaining_s)}"
            f" | koniec ~{finish.strftime('%H:%M')}{bound}"
        )


def _hms(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {seconds % 60:02d}s"


# --------------------------------------------------------------------------
# projekcja budzetu z pomiarow kalibracyjnych
# --------------------------------------------------------------------------
# przyblizone rozmiary datasetow MoleculeNet po przycieciu do SUBSAMPLE_CAP
DATASET_SIZES = {
    "freesolv": 642,
    "esol": 1128,
    "sider": 1427,
    "clintox": 1478,
    "bace": 1513,
    "bbbp": 2039,
    "lipophilicity": 4200,
    "tox21": 7831,
    "toxcast": 8575,
    "hiv": SUBSAMPLE_CAP,
    "muv": SUBSAMPLE_CAP,
}
BUDGET_LIMIT_H = 12.0


def project_budget(
    calib: dict[tuple[str, str], tuple[float, float]],
    repeats: int,
    budget_h: float = BUDGET_LIMIT_H,
) -> None:
    """
    calib: (impl, dataset) -> (sekundy_na_czasteczke, srednia_liczba_atomow)

    Koszt duzych czasteczek skalujemy prawem potegowym t ~ atoms^k, gdzie k
    wyznaczamy z dwoch punktow peptydowych. Male czasteczki bierzemy wprost
    z pomiaru na BBBP.
    """
    print("\n" + "=" * 66)
    print("PROJEKCJA BUDZETU (z pomiarow kalibracyjnych)")
    print("=" * 66)

    per_impl = {}
    for impl in IMPLS:
        try:
            small_s, small_atoms = calib[(impl, "bbbp")]
            p20_s, p20_atoms = calib[(impl, "synthpep_20")]
            p40_s, p40_atoms = calib[(impl, "synthpep_80")]
        except KeyError:
            print(f"  {impl}: brak kompletu pomiarow, pomijam projekcje")
            continue
        k = np.log(p40_s / p20_s) / np.log(p40_atoms / p20_atoms)
        c = p40_s / (p40_atoms**k)
        per_impl[impl] = (small_s, c, k)
        print(
            f"  {impl:8s}: male {small_s * 1000:7.1f} ms/mol | "
            f"peptydy t ~ atomow^{k:.2f}"
        )

    if len(per_impl) < len(IMPLS):
        return

    def peptide_seconds(impl: str, atoms: float) -> float:
        _, c, k = per_impl[impl]
        return c * (atoms**k)

    atoms_per_residue = calib[("ours", "synthpep_80")][1] / 80.0

    # --- wykres 1: speedup na MoleculeNet (rownolegle, jeden proces na dataset)
    per_dataset = {}
    for d, n in DATASET_SIZES.items():
        per_dataset[d] = sum(per_impl[i][0] * n for i in IMPLS) * repeats
    chart1_wall = max(per_dataset.values()) / 3600
    chart1_core = sum(per_dataset.values()) / 3600

    # --- wykresy 2+3: drabinka peptydowa (rownolegle, jeden proces na punkt)
    per_point = {}
    # realne peptydy z PeptideReactor sa czescia zestawu "size" i bywaja jego
    # najdrozszym elementem, wiec musza wejsc do projekcji
    for name, count in PEPTIDE_REACTOR_SETS.items():
        atoms = PEPTIDE_REACTOR_ATOMS[name]
        per_point[name] = (
            sum(peptide_seconds(i, atoms) for i in IMPLS) * count * repeats
        )
    for length, count in PEPTIDE_LADDER.items():
        atoms = length * atoms_per_residue
        per_point[length] = (
            sum(peptide_seconds(i, atoms) for i in IMPLS) * count * repeats
        )
    chart23_wall = max(per_point.values()) / 3600
    chart23_core = sum(per_point.values()) / 3600

    # --- wykres 4: watki (wylacznie). Idealne skalowanie => suma 1/n.
    # Kazda implementacja ma wlasna liczebnosc, wiec liczymy je osobno; hiv jest
    # ograniczony rozmiarem datasetu (41127 czasteczek).
    thread_factor = sum(1.0 / n for n in THREAD_COUNTS)
    pep_atoms = THREADS_PEPTIDE_LENGTH * atoms_per_residue
    threads_one = 0.0
    for impl, counts in THREADS_COUNTS.items():
        threads_one += per_impl[impl][0] * min(counts["hiv"], 41_127)
        threads_one += peptide_seconds(impl, pep_atoms) * counts[
            f"synthpep_{THREADS_PEPTIDE_LENGTH}"
        ]
    chart4 = threads_one * thread_factor * THREADS_REPEATS / 3600

    hiv_one = sum(per_impl[i][0] * SUBSAMPLE_CAP for i in IMPLS)

    # --- wykres 5: slupek referencyjny (wylacznie)
    chart5 = hiv_one * repeats / 3600

    rows = [
        ("1  speedup MoleculeNet", "A rownolegle", chart1_wall, chart1_core),
        ("2+3 drabinka peptydowa", "A rownolegle", chart23_wall, chart23_core),
        ("4  watki (WYLACZNIE)", "B wylacznie", chart4, chart4),
        ("5  slupek 10k (WYLACZNIE)", "B wylacznie", chart5, chart5),
    ]
    print(f"\n  {'wykres':28s} {'faza':14s} {'wall-clock':>11s} {'rdzeniogodz':>12s}")
    print("  " + "-" * 68)
    for name, phase, wall, core in rows:
        print(f"  {name:28s} {phase:14s} {wall:9.2f} h {core:10.2f} h")
    total_wall = sum(r[2] for r in rows)
    total_core = sum(r[3] for r in rows)
    print("  " + "-" * 68)
    print(f"  {'RAZEM':28s} {'':14s} {total_wall:9.2f} h {total_core:10.2f} h")

    print()
    if total_wall <= budget_h:
        print(
            f"  MIESCI SIE w limicie {budget_h:.0f} h "
            f"(zapas {budget_h - total_wall:.1f} h)"
        )
    else:
        over = total_wall - budget_h
        print(f"  NIE MIESCI SIE: przekroczenie o {over:.1f} h.")
        print("  Co przyciac (od najtanszego w skutkach):")
        print("    * zmniejsz PEPTIDE_LADDER dla 80 i 100 AA (np. 10->4, 5->2)")
        print("    * usun 100 AA z drabinki")
        print("    * zmniejsz SUBSAMPLE_CAP z 10000 na 5000")
        print("    * ogranicz THREAD_COUNTS do [1, 8, 45]")
        print("    * zejdz z --repeats 3 na 2")
    print("=" * 66 + "\n")


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def open_csv(path: Path):
    fresh = not path.exists() or path.stat().st_size == 0
    if not fresh:
        # dopisujemy do istniejacego pliku, wiec jego naglowek musi sie zgadzac -
        # inaczej wznowiony przebieg wpisywalby wartosci pod zle kolumny
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != CSV_FIELDS:
            raise SystemExit(
                f"{path} ma inny naglowek niz obecny format CSV.\n"
                f"  w pliku: {','.join(header)}\n"
                f"  obecnie: {','.join(CSV_FIELDS)}\n"
                "Wskaz nowy --out albo odloz stary plik na bok."
            )
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    if fresh:
        writer.writeheader()
        handle.flush()
    return handle, writer


def already_done(path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    done.add(
                        (
                            r["impl"],
                            r["dataset"],
                            int(r["n_molecules"]),
                            int(r["n_jobs"]),
                            int(r["repeat"]),
                        )
                    )
    return done


def git_commit() -> str:
    """Branch i commit REPO SKFP - to jego wersje benchmarkujemy, nie skryptu."""
    try:
        import skfp

        repo = Path(skfp.__file__).resolve().parent.parent
        run = lambda a: subprocess.run(
            a, capture_output=True, text=True, cwd=repo
        ).stdout.strip()
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        sha = run(["git", "rev-parse", "--short", "HEAD"])
        return f"{branch}@{sha}" if sha else "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--suite",
        required=True,
        choices=["calib", "speedup", "size", "threads", "erx_threads",
                 "expansionrx", "reference"],
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", default=None, help="ogranicz do jednego datasetu (faza A)")
    p.add_argument("--impls", nargs="+", default=list(IMPLS), choices=list(IMPLS))
    p.add_argument(
        "--repeats", type=int, default=None, help="domyslnie 3, dla threads 2"
    )
    p.add_argument("--max-size", type=int, default=0, help="twardy limit N (0=brak)")
    p.add_argument(
        "--peptide-dir",
        type=Path,
        default=None,
        help="katalog data/PeptideReactor z repo peptydowego",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--budget-h",
        type=float,
        default=BUDGET_LIMIT_H,
        help="limit wall-clock w godzinach, wg ktorego calib wydaje werdykt",
    )
    p.add_argument(
        "--thread-counts",
        nargs="+",
        type=int,
        default=None,
        help="nadpisz liczby watkow dla zestawu threads",
    )
    args = p.parse_args()

    if args.thread_counts:
        THREAD_COUNTS[:] = args.thread_counts

    if args.suite == "calib":
        args.repeats = 1
    elif args.repeats is None:
        args.repeats = (
            THREADS_REPEATS if args.suite in ("threads", "erx_threads") else 3
        )

    jobs = build_jobs(args.suite, args.only)
    if args.only:
        jobs = [(d, n) for d, n in jobs if d == args.only]
        if not jobs:
            raise SystemExit(
                f"--only {args.only} nie wystepuje w zestawie {args.suite}"
            )

    commit, host = git_commit(), platform.node()
    done = already_done(args.out)
    handle, writer = open_csv(args.out)
    print(
        f"suite={args.suite} | CSV={args.out} | zrobione={len(done)} | {commit}",
        flush=True,
    )

    def record(**kw):
        writer.writerow(
            dict(
                timestamp=datetime.now(timezone.utc).isoformat(),
                suite=args.suite,
                commit=commit,
                host=host,
                # obie implementacje sa wylacznie CPU-owe; kolumna jest tu, bo ma ja
                # CSV z repo benchmarkowego kolegi i chcemy te same metryki
                device="cpu",
                **kw,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())

    WARMUP_MOLS = [Chem.MolFromSmiles(WARMUP_SMILES)]
    progress = Progress(jobs, args.suite, args)
    cache: dict[str, list] = {}
    calib_measurements: dict[tuple[str, str], tuple[float, float]] = {}
    try:
        for dataset, n_jobs in jobs:
            # kazdy dataset ma wlasny generator => ta sama podprobka niezaleznie
            # od kolejnosci zadan i od tego, czy wznawiamy przebieg
            rng = np.random.default_rng(args.seed)
            if dataset not in cache:
                try:
                    count = (
                        threads_count(dataset, args.impls)
                        if args.suite in ("threads", "erx_threads")
                        else None
                    )
                    loaded = load_mols(dataset, rng, args.peptide_dir, count)
                    # przyciecie MUSI byc przed filtrem: peptide_reactor wczytuje
                    # caly FASTA, wiec filtr na hiv_nvp policzylby 733 czasteczki
                    # po ~2000 atomow zamiast dwoch, ktore faktycznie mierzymy.
                    # jawny --max-size zawsze wygrywa - pozwala okroic dowolny
                    # zestaw do testu poprawnosci przed prawdziwym przebiegiem
                    if args.max_size:
                        loaded = loaded[: args.max_size]
                    elif args.suite == "calib":
                        loaded = loaded[: CALIB_COUNTS.get(dataset, 5)]
                    elif dataset.startswith("preactor_"):
                        name = dataset.split("_", 1)[1]
                        loaded = loaded[: PEPTIDE_REACTOR_SETS.get(name, 5)]
                    loaded, dropped = drop_uncomputable(loaded)
                    if dropped:
                        print(
                            f"  pominieto {dropped} czasteczek - nasza implementacja"
                            f" rzuca na nich wyjatkiem (patrz drop_uncomputable)",
                            flush=True,
                        )
                    cache[dataset] = loaded
                except Exception as e:
                    print(f"  BLAD ladowania {dataset}: {e}", flush=True)
                    record(
                        impl="",
                        dataset=dataset,
                        n_molecules=0,
                        n_jobs=n_jobs,
                        repeat=0,
                        wall_seconds="",
                        seconds_per_mol="",
                        throughput_mol_s="",
                        n_features="",
                        mean_atoms="",
                        mean_bonds="",
                        status="load_error",
                        error=str(e)[:300],
                    )
                    continue
            mols = cache[dataset]
            n = len(mols)
            if n == 0:
                continue
            # w kolumnie dataset ma byc czytelna nazwa, nie cala sciezka do pliku
            label = Path(dataset[4:]).stem if dataset.startswith("csv:") else dataset
            atoms = float(np.mean([m.GetNumAtoms() for m in mols]))
            bonds = float(np.mean([m.GetNumBonds() for m in mols]))
            progress.refine(dataset, n_jobs, n, atoms)
            print(
                f"\n=== {dataset} | N={n} | sr.atomow={atoms:.0f} | n_jobs={n_jobs} ===",
                flush=True,
            )

            warmed: set[str] = set()

            # powtorzenia na zewnatrz, implementacje w srodku: obie dostaja te
            # same warunki w kazdej rundzie, a ETA zna przelicznik dla obu juz
            # po pierwszej rundzie zamiast dopiero w polowie zadania
            for rep in range(1, args.repeats + 1):
                for impl in args.impls:
                    # calib always measures: skipping would leave the projection
                    # without the numbers it fits, so a rerun would print nothing
                    if (
                        args.suite != "calib"
                        and (
                            impl,
                            dataset,
                            n,
                            n_jobs,
                            rep,
                        )
                        in done
                    ):
                        progress.skipped(dataset, n_jobs, impl)
                        continue
                    try:
                        if impl not in warmed:
                            IMPLS[impl](WARMUP_MOLS, n_jobs)
                            warmed.add(impl)
                        t0 = time.perf_counter()
                        count, width = IMPLS[impl](mols, n_jobs)
                        dt = time.perf_counter() - t0
                        record(
                            impl=impl,
                            dataset=label,
                            n_molecules=n,
                            n_jobs=n_jobs,
                            repeat=rep,
                            wall_seconds=f"{dt:.4f}",
                            seconds_per_mol=f"{dt / count:.6f}",
                            throughput_mol_s=f"{count / dt:.4f}",
                            n_features=width,
                            mean_atoms=f"{atoms:.2f}",
                            mean_bonds=f"{bonds:.2f}",
                            status="ok",
                            error="",
                        )
                        print(
                            f"  {impl:8s} rep={rep} {dt:9.2f}s "
                            f"{count / dt:9.2f} mol/s  {dt / count:8.3f} s/mol",
                            flush=True,
                        )
                        eta = progress.measured(dataset, n_jobs, dt, impl)
                        if eta:
                            print(eta, flush=True)
                        if args.suite == "calib":
                            calib_measurements[(impl, dataset)] = (dt / count, atoms)
                    except Exception as e:
                        traceback.print_exc()
                        record(
                            impl=impl,
                            dataset=label,
                            n_molecules=n,
                            n_jobs=n_jobs,
                            repeat=rep,
                            wall_seconds="",
                            seconds_per_mol="",
                            throughput_mol_s="",
                            n_features="",
                            mean_atoms=f"{atoms:.2f}",
                            mean_bonds=f"{bonds:.2f}",
                            status="error",
                            error=str(e)[:300],
                        )
    finally:
        handle.close()
        print(f"\nGotowe -> {args.out}", flush=True)

    if args.suite == "calib" and calib_measurements:
        project_budget(calib_measurements, repeats=3, budget_h=args.budget_h)


if __name__ == "__main__":
    main()
