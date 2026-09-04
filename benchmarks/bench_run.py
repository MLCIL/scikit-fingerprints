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

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CSV_FIELDS = [
    "timestamp",
    "suite",
    "impl",
    "dataset",
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

# drabinka peptydowa: dlugosc w aminokwasach -> ile czasteczek.
# Przy duzych rozmiarach kilka sztuk wystarcza: koszt jest zdominowany przez
# deterministyczne O(n^3), wiec wariancja jest mala.
PEPTIDE_LADDER = {5: 200, 10: 100, 20: 50, 40: 20, 80: 10, 100: 5}

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
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 45]
# Krzywa watkow nie potrzebuje 10k czasteczek: przy 2000 kazdy z 45 watkow
# dostaje ~44 sztuki, co w zupelnosci wystarcza do pokazania skalowania,
# a skraca ten (wylaczny!) pomiar czterokrotnie.
THREADS_N = 2000
# kalibracja ma zajac kilka minut, wiec swoje wlasne, male liczebnosci
CALIB_COUNTS = {"bbbp": 300, "synthpep_20": 5, "synthpep_80": 1}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# --------------------------------------------------------------------------
# dane
# --------------------------------------------------------------------------
def load_mols(
    dataset: str, rng: np.random.Generator, peptide_dir: Path | None
) -> list[Chem.Mol]:
    if dataset.startswith("synthpep_"):
        length = int(dataset.split("_")[1])
        return synthetic_peptides(length, PEPTIDE_LADDER.get(length, 10), rng)

    if dataset.startswith("preactor_"):
        return peptide_reactor(dataset.split("_", 1)[1], peptide_dir, rng)

    from skfp.datasets import moleculenet

    df = getattr(moleculenet, f"load_{dataset}")(as_frame=True)
    col = "SMILES" if "SMILES" in df.columns else df.columns[0]
    smiles = list(df[col])

    # losowa podprobka ze stalym ziarnem: obie implementacje dostaja te same czasteczki
    if len(smiles) > SUBSAMPLE_CAP:
        idx = rng.choice(len(smiles), size=SUBSAMPLE_CAP, replace=False)
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
            ("synthpep_40", n) for n in THREAD_COUNTS
        ]
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
    else:
        count = DATASET_SIZES.get(dataset, 2000)
        atoms = float(DATASET_ATOMS.get(dataset, 25))

    if args.max_size:
        count = min(count, args.max_size)
    elif suite == "calib":
        count = min(count, CALIB_COUNTS.get(dataset, 5))
    elif suite == "threads":
        count = min(count, THREADS_N)
    return count, atoms


class Progress:
    """
    Szacuje czas zakonczenia, wazac zadania iloscia pracy zamiast ich liczba.

    Pomiar dwoch peptydow trwa tyle co dwoch tysiecy malych czasteczek, wiec
    postep liczony w "zrobionych pomiarach" bylby bezuzyteczny. Jednostka pracy
    to ``liczba_czasteczek * atomow ** COST_EXPONENT / n_jobs``, a przelicznik
    na sekundy bierze sie z tego, co juz zmierzono - dzieki czemu ETA sam
    kalibruje sie do maszyny po pierwszym pomiarze.
    """

    def __init__(self, jobs: list[tuple[str, int]], suite: str, args) -> None:
        self._units: dict[tuple[str, int], float] = {}
        for dataset, n_jobs in jobs:
            count, atoms = expected_shape(suite, dataset, args)
            self._units[(dataset, n_jobs)] = self._work(count, atoms, n_jobs)
        # kazde zadanie liczy sie dla kazdej implementacji i powtorzenia
        self._per_job = max(1, len(args.impls) * args.repeats)
        self._remaining = sum(self._units.values()) * self._per_job
        self._done_units = 0.0
        self._done_seconds = 0.0
        self._start = time.perf_counter()

    @staticmethod
    def _work(count: int, atoms: float, n_jobs: int) -> float:
        return count * (atoms**COST_EXPONENT) / max(1, n_jobs)

    def refine(self, dataset: str, n_jobs: int, count: int, atoms: float) -> None:
        """Podmien oszacowanie na rzeczywisty rozmiar, gdy dataset sie zaladuje."""
        key = (dataset, n_jobs)
        if key not in self._units:
            return
        actual = self._work(count, atoms, n_jobs)
        self._remaining += (actual - self._units[key]) * self._per_job
        self._units[key] = actual

    def skipped(self, dataset: str, n_jobs: int) -> None:
        """Pomiar wznowiony z CSV - praca odpada z puli, ale bez czasu."""
        self._remaining -= self._units.get((dataset, n_jobs), 0.0)

    def measured(self, dataset: str, n_jobs: int, seconds: float) -> str:
        units = self._units.get((dataset, n_jobs), 0.0)
        self._done_units += units
        self._done_seconds += seconds
        self._remaining = max(0.0, self._remaining - units)

        if self._done_units <= 0:
            return ""
        remaining_s = self._remaining * (self._done_seconds / self._done_units)
        finish = datetime.now().astimezone() + timedelta(seconds=remaining_s)
        share = self._done_units / (self._done_units + self._remaining)
        return (
            f"      postep {share * 100:5.1f}% | pozostalo {_hms(remaining_s)}"
            f" | koniec ~{finish.strftime('%H:%M')}"
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
    calib: dict[tuple[str, str], tuple[float, float]], repeats: int
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

    # --- wykres 4: watki (wylacznie). Idealne skalowanie => suma 1/n
    thread_factor = sum(1.0 / n for n in THREAD_COUNTS)
    hiv_threads_one = sum(per_impl[i][0] * THREADS_N for i in IMPLS)
    hiv_one = sum(per_impl[i][0] * SUBSAMPLE_CAP for i in IMPLS)
    pep_atoms = 40 * atoms_per_residue
    pep_one = sum(peptide_seconds(i, pep_atoms) for i in IMPLS) * PEPTIDE_LADDER[40]
    chart4 = (hiv_threads_one + pep_one) * thread_factor * repeats / 3600

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
    if total_wall <= BUDGET_LIMIT_H:
        print(
            f"  MIESCI SIE w limicie {BUDGET_LIMIT_H:.0f} h "
            f"(zapas {BUDGET_LIMIT_H - total_wall:.1f} h)"
        )
    else:
        over = total_wall - BUDGET_LIMIT_H
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
        choices=["calib", "speedup", "size", "threads", "reference"],
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", default=None, help="ogranicz do jednego datasetu (faza A)")
    p.add_argument("--impls", nargs="+", default=list(IMPLS), choices=list(IMPLS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-size", type=int, default=0, help="twardy limit N (0=brak)")
    p.add_argument(
        "--peptide-dir",
        type=Path,
        default=None,
        help="katalog data/PeptideReactor z repo peptydowego",
    )
    p.add_argument("--seed", type=int, default=0)
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
                **kw,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())

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
                    cache[dataset] = load_mols(dataset, rng, args.peptide_dir)
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
            # jawny --max-size zawsze wygrywa: pozwala okroic dowolny zestaw
            # (przydatne do testu poprawnosci przed prawdziwym przebiegiem)
            if args.max_size:
                mols = mols[: args.max_size]
            elif args.suite == "calib":
                mols = mols[: CALIB_COUNTS.get(dataset, 5)]
            elif dataset.startswith("preactor_"):
                mols = mols[: PEPTIDE_REACTOR_SETS.get(dataset.split("_", 1)[1], 5)]
            elif args.suite == "threads":
                mols = mols[:THREADS_N]
            n = len(mols)
            if n == 0:
                continue
            atoms = float(np.mean([m.GetNumAtoms() for m in mols]))
            bonds = float(np.mean([m.GetNumBonds() for m in mols]))
            progress.refine(dataset, n_jobs, n, atoms)
            print(
                f"\n=== {dataset} | N={n} | sr.atomow={atoms:.0f} | n_jobs={n_jobs} ===",
                flush=True,
            )

            for impl in args.impls:
                for rep in range(1, args.repeats + 1):
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
                        progress.skipped(dataset, n_jobs)
                        continue
                    try:
                        IMPLS[impl](mols[: min(3, n)], n_jobs)  # warmup
                        t0 = time.perf_counter()
                        count, width = IMPLS[impl](mols, n_jobs)
                        dt = time.perf_counter() - t0
                        record(
                            impl=impl,
                            dataset=dataset,
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
                        eta = progress.measured(dataset, n_jobs, dt)
                        if eta:
                            print(eta, flush=True)
                        if args.suite == "calib":
                            calib_measurements[(impl, dataset)] = (dt / count, atoms)
                    except Exception as e:
                        traceback.print_exc()
                        record(
                            impl=impl,
                            dataset=dataset,
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
        project_budget(calib_measurements, repeats=3)


if __name__ == "__main__":
    main()
