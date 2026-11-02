"""
Analiza wynikow benchmarku: agregacja + wykresy na poster.

Czyta jeden lub wiele CSV z bench_run.py (mozna podac wildcard), usrednia
powtorzenia i generuje wykresy. Rysuje tylko to, na co sa dane, wiec mozna
odpalac w trakcie przebiegu, zeby zobaczyc czesciowe wyniki.

UZYCIE
    python bench_analyze.py --inputs "res_*.csv" --outdir figures
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OURS, MORDRED = "ours", "mordred"
COLORS = {OURS: "#2b8cbe", MORDRED: "#e6550d"}


def load(patterns: list[str]) -> pd.DataFrame:
    files = [f for p in patterns for f in glob.glob(p)]
    if not files:
        raise SystemExit(f"brak plikow pasujacych do {patterns}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    # Zestawy ExpansionRx maja te sama role co MoleculeNet-owe, tylko inna rodzine
    # danych: expansionrx to jednowatkowe ilorazy, erx_threads to krzywa skalowania.
    # Bez tego mapowania wykresy nie znalazlyby ich wierszy i nie powstalby zaden.
    df["suite"] = df["suite"].replace(
        {"expansionrx": "speedup", "erx_threads": "threads"}
    )
    for c in ("wall_seconds", "seconds_per_mol", "throughput_mol_s", "mean_atoms"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"wczytano {len(files)} plikow, {len(df)} udanych pomiarow")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Srednia i odchylenie po powtorzeniach."""
    keys = ["suite", "impl", "dataset", "n_molecules", "n_jobs", "mean_atoms"]
    g = df.groupby(keys, as_index=False).agg(
        throughput=("throughput_mol_s", "mean"),
        throughput_std=("throughput_mol_s", "std"),
        sec_per_mol=("seconds_per_mol", "mean"),
        wall=("wall_seconds", "mean"),
        # np.std liczy z ddof=0, a pandas domyslnie z ddof=1 - repo kolegi uzywa
        # np.std, wiec zeby liczby byly te same, wymuszamy ddof=0
        wall_std=("wall_seconds", lambda s: float(np.std(s))),
        wall_total=("wall_seconds", "sum"),
        n_repeats=("repeat", "count"),
    )
    return g.fillna({"throughput_std": 0.0})


# nazwy kolumn i ich kolejnosc z RUNTIME_CSV_FIELDS w repo
# benchmarking_molecular_models_private (src/embedding/embedding.py)
RUNTIME_CSV_FIELDS = [
    "dataset",
    "embedder",
    "device",
    "mean_runtime",
    "std_runtime",
    "total_runtime",
    "n_samples",
    "n_full_batches",
]
# ich nazwy modeli, zeby wiersze dalo sie wprost dopisac do ich runtime.csv
EMBEDDER_NAMES = {OURS: "skfp_mordred_2d", MORDRED: "mordred_2d"}


def runtime_csv(agg: pd.DataFrame, outdir: Path) -> None:
    """
    Zapisuje pomiary w formacie runtime.csv z repo benchmarkowego kolegi.

    Odpowiednioscia jego "pelnej paczki n_samples czasteczek" jest u nas jedno
    powtorzenie na zmierzonym podzbiorze, wiec:
      n_samples      = liczba czasteczek w pomiarze
      n_full_batches = liczba powtorzen
      mean/std       = srednia i odchylenie czasu powtorzenia (std z ddof=0)
      total_runtime  = suma czasow wszystkich powtorzen

    Rozne liczby watkow trafiaja do nazwy embeddera (jego format nie ma na to
    kolumny), zeby wiersze zostaly jednoznaczne.
    """
    rows = []
    for _, r in agg.iterrows():
        name = EMBEDDER_NAMES.get(r["impl"], r["impl"])
        if int(r["n_jobs"]) != 1:
            name = f"{name}_njobs{int(r['n_jobs'])}"
        rows.append(
            {
                "dataset": r["dataset"],
                "embedder": name,
                "device": "cpu",
                "mean_runtime": r["wall"],
                "std_runtime": r["wall_std"],
                "total_runtime": r["wall_total"],
                "n_samples": int(r["n_molecules"]),
                "n_full_batches": int(r["n_repeats"]),
            }
        )
    path = outdir / "runtime.csv"
    pd.DataFrame(rows, columns=RUNTIME_CSV_FIELDS).to_csv(path, index=False)
    print(f"metryki w formacie repo kolegi -> {path}")


def speedups(agg: pd.DataFrame, suite: str) -> pd.DataFrame:
    """Iloraz throughputow ours/mordred dla wspolnych (dataset, n_jobs)."""
    sub = agg[agg["suite"] == suite]
    ours = sub[sub["impl"] == OURS].set_index(["dataset", "n_jobs"])
    mord = sub[sub["impl"] == MORDRED].set_index(["dataset", "n_jobs"])
    common = ours.index.intersection(mord.index)
    if not len(common):
        return pd.DataFrame()
    out = pd.DataFrame({
        "throughput_ours": ours.loc[common, "throughput"],
        "throughput_mordred": mord.loc[common, "throughput"],
        "mean_atoms": ours.loc[common, "mean_atoms"],
        "n_molecules": ours.loc[common, "n_molecules"],
    })
    out["speedup"] = out["throughput_ours"] / out["throughput_mordred"]
    return out.reset_index()


# --------------------------------------------------------------------------
# wykresy
# --------------------------------------------------------------------------
def chart1_speedup_datasets(agg: pd.DataFrame, outdir: Path, label: str = "MoleculeNet") -> None:
    sp = speedups(agg, "speedup")
    if sp.empty:
        print("  [1] brak danych (suite=speedup)")
        return
    sp = sp.sort_values("speedup")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sp["dataset"], sp["speedup"], color=COLORS[OURS])
    ax.axvline(1.0, color="grey", lw=1, ls="--")
    for y, v in enumerate(sp["speedup"]):
        ax.text(v, y, f" {v:.2f}x", va="center", fontsize=9)
    ax.set_xlabel("Speedup (ours / mordred), single thread")
    ax.set_title(f"Speedup on {label} datasets")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(outdir / "1_speedup_datasets.png", dpi=200)
    plt.close(fig)
    print(f"  [1] zapisano, mediana przyspieszenia = {sp['speedup'].median():.2f}x")

    # promotor prosil, zeby obok speedupu podac tez bezwzgledny throughput
    print(f"\n      {'dataset':16s} {'nasze mol/s':>12s} {'mordred mol/s':>14s} {'speedup':>9s}")
    for _, r in sp.sort_values("throughput_ours", ascending=False).iterrows():
        print(f"      {r['dataset']:16s} {r['throughput_ours']:12.1f} "
              f"{r['throughput_mordred']:14.1f} {r['speedup']:8.2f}x")


def source_of(dataset: str) -> str:
    """Skad pochodzi punkt: drabinka syntetyczna, realne peptydy czy MoleculeNet."""
    if dataset.startswith("synthpep_"):
        return "synthetic"
    if dataset.startswith("preactor_"):
        return "real_peptides"
    return "moleculenet"


# Rozne pochodzenie = rozny znacznik. Laczymy linia TYLKO drabinke syntetyczna, bo
# to jedyna kontrolowana seria (ten sam generator, rosnaca dlugosc). Realne peptydy
# i 11 niezaleznych datasetow MoleculeNet to punkty - polaczone jedna lamana dawaly
# zygzaki wynikajace z roznicy zbiorow, nie z liczonych czasow.
SOURCE_STYLE = {
    "synthetic": {"marker": "o", "ls": "-", "lw": 2, "label": "synthetic peptides"},
    "real_peptides": {"marker": "s", "ls": "none", "ms": 9, "label": "real peptides (PeptideReactor)"},
    "moleculenet": {"marker": "^", "ls": "none", "ms": 7, "label": "MoleculeNet"},
}
SOURCE_ORDER = ["synthetic", "real_peptides", "moleculenet"]


def chart2_throughput_vs_size(agg: pd.DataFrame, outdir: Path) -> None:
    sub = agg[agg["suite"].isin(["size", "speedup"])]
    sub = sub[sub["n_jobs"] == 1].copy()
    if sub.empty:
        print("  [2] brak danych")
        return
    sub["source"] = sub["dataset"].map(source_of)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for impl in (OURS, MORDRED):
        for src in SOURCE_ORDER:
            d = sub[(sub["impl"] == impl) & (sub["source"] == src)]
            if d.empty:
                continue
            d = d.sort_values("mean_atoms")
            style = {k: v for k, v in SOURCE_STYLE[src].items() if k != "label"}
            ax.plot(d["mean_atoms"], d["sec_per_mol"], color=COLORS[impl], **style)

    # Jedna legenda, dwie kolumny: kolor mowi o implementacji, znacznik o pochodzeniu
    # punktu. Dwie osobne ramki nachodzily na siebie, bo loc="best" unika danych, a
    # nie innej legendy - obie trafialy w ten sam wolny rog.
    impl_handles = [
        plt.Line2D([], [], color=COLORS[i], lw=2,
                   label="ours" if i == OURS else "mordred")
        for i in (OURS, MORDRED)
    ]
    src_handles = [
        plt.Line2D([], [], color="grey", **SOURCE_STYLE[s]) for s in SOURCE_ORDER
    ]
    ax.legend(handles=impl_handles + src_handles, loc="best", frameon=True,
              fontsize=8, ncol=2)

    ax.set_xlabel("Mean number of atoms per molecule")
    ax.set_ylabel("Seconds per molecule (single thread)")
    ax.set_title("Computation cost vs molecule size")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "2_cost_vs_size.png", dpi=200)
    plt.close(fig)
    print("  [2] zapisano")


def chart3_speedup_large(agg: pd.DataFrame, outdir: Path) -> None:
    sp = speedups(agg, "size")
    if sp.empty:
        print("  [3] brak danych (suite=size)")
        return
    sp = sp.sort_values("mean_atoms")
    sp["source"] = sp["dataset"].map(source_of)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for src in SOURCE_ORDER:
        d = sp[sp["source"] == src]
        if d.empty:
            continue
        style = {k: v for k, v in SOURCE_STYLE[src].items() if k != "label"}
        ax.plot(d["mean_atoms"], d["speedup"], color=COLORS[OURS],
                label=SOURCE_STYLE[src]["label"], **style)
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    for _, r in sp.iterrows():
        ax.annotate(f"{r['speedup']:.1f}x", (r["mean_atoms"], r["speedup"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean number of atoms per molecule")
    ax.set_ylabel("Speedup (ours / mordred)")
    ax.set_title("Speedup grows with molecule size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "3_speedup_vs_size.png", dpi=200)
    plt.close(fig)
    print("  [3] zapisano")

    print(f"\n      {'zbior':18s} {'atomow':>7s} {'nasze mol/s':>12s} "
          f"{'mordred mol/s':>14s} {'speedup':>9s}")
    for _, r in sp.iterrows():
        print(f"      {r['dataset']:18s} {r['mean_atoms']:7.0f} "
              f"{r['throughput_ours']:12.3f} {r['throughput_mordred']:14.3f} "
              f"{r['speedup']:8.2f}x")


def chart4_threads(agg: pd.DataFrame, outdir: Path, log: bool = True) -> None:
    """
    Krzywa skalowania, w dwoch wariantach.

    Logarytmiczny pokazuje oba przebiegi naraz, mimo ze roznia sie o dwa rzedy.
    Liniowy gubi krzywa mordreda przy zerze, ale za to "ideal linear" jest na nim
    prosta, wiec od razu widac, gdzie nasza krzywa od niej odchodzi.
    """
    sub = agg[agg["suite"] == "threads"]
    if sub.empty:
        print("  [4] brak danych (suite=threads)")
        return
    datasets = sorted(sub["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d0 = sub[sub["dataset"] == ds]
        for impl in (OURS, MORDRED):
            d = d0[d0["impl"] == impl].sort_values("n_jobs")
            if d.empty:
                continue
            ax.plot(d["n_jobs"], d["throughput"], "o-", color=COLORS[impl],
                    label="ours" if impl == OURS else "mordred")
        # linia idealnego skalowania od punktu 1-watkowego
        base = d0[(d0["impl"] == OURS) & (d0["n_jobs"] == 1)]["throughput"]
        if len(base):
            n = np.array(sorted(d0["n_jobs"].unique()))
            ax.plot(n, base.iloc[0] * n, ":", color="grey", label="ideal linear")
        if log:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
        else:
            ax.set_xticks(sorted(d0["n_jobs"].unique()))
        ax.set_xlabel("Number of threads")
        ax.set_ylabel("Throughput (mol/s)")
        ax.set_title(ds)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    suffix = "" if log else "_linear"
    scale = "log-log" if log else "linear"
    fig.suptitle(f"Scaling with thread count ({scale} axes)")
    fig.tight_layout()
    fig.savefig(outdir / f"4_threads{suffix}.png", dpi=200)
    plt.close(fig)
    print(f"  [4] zapisano ({scale})")


def chart5_reference(agg: pd.DataFrame, outdir: Path) -> None:
    """Liczba do wykresu kolegi: sekundy na 10 000 czasteczek, 1 watek."""
    sub = agg[(agg["suite"] == "reference") & (agg["n_jobs"] == 1)]
    if sub.empty:
        print("  [5] brak danych (suite=reference)")
        return
    print("\n  [5] SLUPEK DO WYKRESU KOLEGI (sekundy / 10 000 czasteczek, 1 watek):")
    for _, r in sub.iterrows():
        per_10k = r["sec_per_mol"] * 10_000
        label = "nasze" if r["impl"] == OURS else "mordred"
        print(f"        {label:8s} {per_10k:9.1f} s   (zmierzone na {int(r['n_molecules'])} czasteczkach)")
    print("        -> odniesienie z wykresu kolegi: Mordred = 190 s")


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help='np. "res_*.csv"')
    p.add_argument("--outdir", type=Path, default=Path("figures"))
    p.add_argument("--label", default="MoleculeNet",
                   help="rodzina danych, uzywana w tytulach wykresow")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load(args.inputs)
    agg = aggregate(df)

    summary = args.outdir / "summary.csv"
    agg.to_csv(summary, index=False)
    print(f"tabela zbiorcza -> {summary}")
    runtime_csv(agg, args.outdir)
    print()

    incomplete = agg[agg["n_repeats"] < 3]
    if len(incomplete):
        print(f"UWAGA: {len(incomplete)} kombinacji ma mniej niz 3 powtorzenia\n")

    print("wykresy:")
    chart1_speedup_datasets(agg, args.outdir, args.label)
    chart2_throughput_vs_size(agg, args.outdir)
    chart3_speedup_large(agg, args.outdir)
    chart4_threads(agg, args.outdir, log=True)
    chart4_threads(agg, args.outdir, log=False)
    chart5_reference(agg, args.outdir)
    print(f"\nGotowe -> {args.outdir}")


if __name__ == "__main__":
    main()
