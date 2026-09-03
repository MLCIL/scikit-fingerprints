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
        n_repeats=("repeat", "count"),
    )
    return g.fillna({"throughput_std": 0.0})


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
def chart1_speedup_datasets(agg: pd.DataFrame, outdir: Path) -> None:
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
    ax.set_xlabel("Przyspieszenie (nasze / mordred), 1 watek")
    ax.set_title("Przyspieszenie na datasetach MoleculeNet")
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


def chart2_throughput_vs_size(agg: pd.DataFrame, outdir: Path) -> None:
    sub = agg[agg["suite"].isin(["size", "speedup"])]
    sub = sub[sub["n_jobs"] == 1]
    if sub.empty:
        print("  [2] brak danych")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for impl in (OURS, MORDRED):
        d = sub[sub["impl"] == impl].sort_values("mean_atoms")
        if d.empty:
            continue
        ax.plot(d["mean_atoms"], d["sec_per_mol"], "o-", color=COLORS[impl],
                label="nasze" if impl == OURS else "mordred")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Srednia liczba atomow w czasteczce")
    ax.set_ylabel("Sekundy na czasteczke (1 watek)")
    ax.set_title("Koszt obliczen a rozmiar czasteczki")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
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
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sp["mean_atoms"], sp["speedup"], "o-", color=COLORS[OURS], lw=2)
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    for _, r in sp.iterrows():
        ax.annotate(f"{r['speedup']:.2f}x", (r["mean_atoms"], r["speedup"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Srednia liczba atomow w czasteczce")
    ax.set_ylabel("Przyspieszenie (nasze / mordred)")
    ax.set_title("Przyspieszenie rosnie z rozmiarem czasteczki")
    ax.grid(True, which="both", alpha=0.3)
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


def chart4_threads(agg: pd.DataFrame, outdir: Path) -> None:
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
                    label="nasze" if impl == OURS else "mordred")
        # linia idealnego skalowania od punktu 1-watkowego
        base = d0[(d0["impl"] == OURS) & (d0["n_jobs"] == 1)]["throughput"]
        if len(base):
            n = np.array(sorted(d0["n_jobs"].unique()))
            ax.plot(n, base.iloc[0] * n, ":", color="grey", label="idealne liniowe")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Liczba watkow")
        ax.set_ylabel("Throughput (mol/s)")
        ax.set_title(ds)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Skalowanie z liczba watkow")
    fig.tight_layout()
    fig.savefig(outdir / "4_threads.png", dpi=200)
    plt.close(fig)
    print("  [4] zapisano")


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
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load(args.inputs)
    agg = aggregate(df)

    summary = args.outdir / "summary.csv"
    agg.to_csv(summary, index=False)
    print(f"tabela zbiorcza -> {summary}\n")

    incomplete = agg[agg["n_repeats"] < 3]
    if len(incomplete):
        print(f"UWAGA: {len(incomplete)} kombinacji ma mniej niz 3 powtorzenia\n")

    print("wykresy:")
    chart1_speedup_datasets(agg, args.outdir)
    chart2_throughput_vs_size(agg, args.outdir)
    chart3_speedup_large(agg, args.outdir)
    chart4_threads(agg, args.outdir)
    chart5_reference(agg, args.outdir)
    print(f"\nGotowe -> {args.outdir}")


if __name__ == "__main__":
    main()
