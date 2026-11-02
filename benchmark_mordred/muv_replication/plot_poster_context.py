"""
Wykres kontekstowy na poster: nasz Mordred wsrod 28 modeli embeddingowych.

Baza to wykres "Mean time to compute embeddings for 10,000 molecules" z repo
benchmarking_molecular_models_private (jego pomiary, `benchmarking_muv_runtime.csv`).
Dokladamy TRZY slupki mordredowe zmierzone w TYM SAMYM protokole:

  Mordred (as published)   187.37 s  <- jego pomiar, BLAS niepinowany
  Mordred (BLAS pinned)     25.07 s  <- nasza proba kontrolna, ta sama binarka
  Mordred (scikit-fingerprints) 4.02 s  <- nasza implementacja

Kolory: nasz slupek to jedyny mocny akcent (pomaranczowy), oba slupki oryginalnego
mordreda sa niebieskie - wyrazne, ale spokojniejsze - a cala reszta modeli szara,
z odcieniem odrozniajacym CPU od GPU. Chodzi o to, zeby na posterze wzrok szedl
najpierw do nas, potem do porownania z oryginalem, a dopiero na koncu do tla.

    python plot_poster_context.py --runtime-csv ~/benchmarking_muv_runtime.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- paleta ---------------------------------------------------------------
OURS = "#EE7733"          # pomaranczowy, jedyny mocny akcent
MORDRED_PINNED = "#2E6F9E"  # niebieski nasycony - poprawiony pomiar oryginalu
MORDRED_PUBLISHED = "#2E6F9E"  # ten sam niebieski co na pozostalych wykresach
OTHER_CPU = "#A6A6A6"
OTHER_GPU = "#D2D2D2"

F_OURS = "Our Mordred"
F_PINNED = "Original Mordred, optim."
F_PUBLISHED = "Original"
HIGHLIGHT = {F_OURS: OURS, F_PUBLISHED: MORDRED_PUBLISHED}
# ilorazy pominiete - same slupki i wartosci wystarczaja
NOTE: dict[str, str] = {}

# przeniesione z visualization_times.ipynb (branch dev, komorka 2)
MODEL_MAPPING = {
    "mol2vec": "Mol2Vec", "coati": "COATI", "CDDD": "CDDD",
    "AtomPair": "AtomPair", "AtomPair_count": "AtomPair",
    "ECFP_count": "ECFP", "ECFP": "ECFP", "TT": "TT", "TT_count": "TT",
    "GraphMVP_CP-max": "GraphMVP", "GraphMVP_CP-sum": "GraphMVP",
    "GraphMVP_CP-mean": "GraphMVP", "GraphMVP_AM-max": "GraphMVP",
    "GraphMVP_AM-sum": "GraphMVP", "GraphMVP_AM-mean": "GraphMVP",
    "ChemBERTa-10M-MTR": "ChemBERTa", "ChemBERTa-5M-MTR": "ChemBERTa",
    "ChemBERTa-10M-MLM": "ChemBERTa", "ChemBERTa-5M-MLM": "ChemBERTa",
    "ChemBERTa-77M-MTR": "ChemBERTa", "ChemBERTa-77M-MLM": "ChemBERTa",
    "ChemFM-1B": "ChemFM", "ChemFM-3B": "ChemFM",
    "ChemGPT-4.7M": "ChemGPT", "ChemGPT-19M": "ChemGPT",
    "MoLFormer-XL-both-10pct": "MoLFormer",
    "mol_r_tag_1024": "MolR", "mol_r_sage_1024": "MolR",
    "mol_r_gcn_1024": "MolR", "mol_r_gat_1024": "MolR",
    "chemformer_augment": "ChemFormer", "chemformer_mask": "ChemFormer",
    "chemformer_combined": "ChemFormer",
    "SimSon": "SimSon", "MolTransformer": "MolTransformer", "CLAMP": "CLAMP",
    "mat_masking_20M": "MAT", "mat_masking_2M": "MAT", "mat_masking_200k": "MAT",
    "grover_base": "Grover", "grover_large": "Grover", "rmat_4M": "RMAT",
    "GraphFP-C": "GraphFP", "GraphFP-CP": "GraphFP", "molbert": "MolBERT",
    "GEM": "GEM",
    "GNN-EP-sum": "GIN", "GNN-Contextual-sum": "GIN", "GNN-GPT_TNN-sum": "GIN",
    "GNN-CP-sum": "GIN", "GNN-GraphCL-sum": "GIN", "GNN-IG-sum": "GIN",
    "GNN-Motif-sum": "GIN", "GNN-AM-sum": "GIN",
    "unimolv1": "UniMolv1", "unimolv2": "UniMolv2",
    "SELFormer": "SELFormer", "SELFormer-Lite": "SELFormer",
    "mole": "MolE", "chemeleon": "Chemeleon", "CLAMP_noleak": "CLAMP",
    "mordred_2d": F_PUBLISHED,
}


def build(runtime_csv: Path, ours_csv: Path, dataset: str) -> pd.DataFrame:
    theirs = pd.read_csv(runtime_csv)
    theirs["family"] = theirs["embedder"].map(MODEL_MAPPING)
    if theirs["family"].isna().any():
        bad = sorted(theirs[theirs.family.isna()].embedder.unique())
        raise SystemExit(f"embeddery bez wpisu w MODEL_MAPPING: {bad}")

    # nasze dwa slupki: ta sama binarka mordreda z pinowanym BLAS-em oraz my
    ours = pd.read_csv(ours_csv)
    # Slupek mordreda z pinowanym BLAS-em zostal celowo usuniety z wykresu.
    # UWAGA: bez niego iloraz naszego slupka do "as published" wynosi ~47x, co
    # zawyza, bo tamten pomiar mial nieustawione watki BLAS. Poprawiona stawka
    # (399 mol/s) jest w replice MUV, results/muv_replication/.
    ours = ours[ours.embedder == "skfp_mordred_2d"].copy()
    ours["family"] = F_OURS

    df = pd.concat([theirs, ours], ignore_index=True)
    df = df[df.dataset == dataset].copy()
    # na rodzine zostaje najszybszy wariant - tak jak w jego notebooku
    best = df[df.mean_runtime == df.groupby("family").mean_runtime.transform("min")].copy()
    # os pokazuje przepustowosc, nie czas: 10 000 czasteczek / sredni czas paczki
    best["throughput"] = best.n_samples / best.mean_runtime
    return best.sort_values("throughput", ascending=False).reset_index(drop=True)


def color_of(row) -> str:
    if row.family in HIGHLIGHT:
        return HIGHLIGHT[row.family]
    return OTHER_GPU if row.device == "cuda" else OTHER_CPU


def draw(best: pd.DataFrame, out: Path, horizontal: bool) -> None:
    plt.rcParams.update({
        "font.size": 20, "axes.labelsize": 24, "axes.titlesize": 24,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 22,
        "axes.edgecolor": "#CCCCCC", "axes.linewidth": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    colors = [color_of(r) for _, r in best.iterrows()]
    n = len(best)

    if horizontal:
        order = best.iloc[::-1].reset_index(drop=True)  # najwolniejszy u gory
        cols = [color_of(r) for _, r in order.iterrows()]
        fig, ax = plt.subplots(figsize=(13, 0.52 * n + 2.4), layout="constrained")
        ax.barh(range(n), order.throughput, color=cols, height=0.74,
                edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(order.family)
        ax.set_xscale("log")
        ax.set_xlabel("mols / second")
        ax.xaxis.grid(True, which="major", color="#E8E8E8", zorder=0)
        ax.set_axisbelow(True)
        for i, (_, r) in enumerate(order.iterrows()):
            if r.family in HIGHLIGHT:
                txt = f"{r.throughput:,.0f}".replace(",", "\u2009") + " mol/s"
                ax.text(r.throughput * 1.15, i, txt, va="center", ha="left",
                        fontsize=24, fontweight="bold", color=HIGHLIGHT[r.family])
        for lbl in ax.get_yticklabels():
            if lbl.get_text() in HIGHLIGHT:
                lbl.set_fontweight("bold")
                lbl.set_color(HIGHLIGHT[lbl.get_text()])
    else:
        fig, ax = plt.subplots(figsize=(19, 9.0), layout="constrained")
        ax.bar(range(n), best.throughput, color=colors, width=0.76,
               edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(n))
        ax.set_xticklabels(best.family, rotation=45, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("mols / second")
        ax.set_title("Throughput across embedding models", fontsize=24, pad=14)
        ax.yaxis.grid(True, which="major", color="#E8E8E8", zorder=0)
        ax.set_axisbelow(True)
        for i, (_, r) in enumerate(best.iterrows()):
            if r.family in HIGHLIGHT:
                txt = f"{r.throughput:,.0f}".replace(",", "\u2009") + " mol/s"
                # w prawo od slupka: sortowanie jest malejace, wiec po tej stronie
                # sasiedzi sa nizsi i podpis nie wchodzi na nic
                ax.text(i - 0.20, r.throughput * 1.35, txt, ha="left", fontsize=23,
                        fontweight="bold", color=HIGHLIGHT[r.family])
        # miejsce na legende nad slupkami, zeby nie zachodzila na adnotacje
        ax.set_ylim(top=ax.get_ylim()[1] * 9)
        for lbl in ax.get_xticklabels():
            if lbl.get_text() in HIGHLIGHT:
                lbl.set_fontweight("bold")
                lbl.set_color(HIGHLIGHT[lbl.get_text()])

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OURS),
        plt.Rectangle((0, 0), 1, 1, color=MORDRED_PUBLISHED),
        plt.Rectangle((0, 0), 1, 1, color=OTHER_CPU),
        plt.Rectangle((0, 0), 1, 1, color=OTHER_GPU),
    ]
    labels = ["Our Mordred",
              "Original",
              "Other models (CPU)", "Other models (GPU)"]
    ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.96,
              borderpad=0.9, labelspacing=0.6, handlelength=2.0,
              edgecolor="#CCCCCC", facecolor="white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out, dpi=300)
    print(f"  {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-csv", default=Path.home() / "benchmarking_muv_runtime.csv",
                   type=Path)
    p.add_argument("--ours", type=Path,
                   default=Path("results/muv_replication/proba2_blas_pinned/runtime.csv"))
    p.add_argument("--dataset", default="ogbg-molmuv")
    p.add_argument("--outdir", type=Path, default=Path("results/figures"))
    args = p.parse_args()

    best = build(args.runtime_csv, args.ours, args.dataset)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"rodzin na wykresie: {len(best)}")
    draw(best, args.outdir / "5_context_models.png", horizontal=False)
    draw(best, args.outdir / "5_context_models_horizontal.png", horizontal=True)

    rank = best.sort_values("mean_runtime").reset_index(drop=True)
    i = int(rank.index[rank.family == F_OURS][0])
    print(f"\nnasz slupek: {i + 1} z {len(rank)} od najszybszego")
    print(f"  szybsze od nas: {', '.join(rank.family[:i])}")
    pub = float(rank[rank.family == F_PUBLISHED].mean_runtime.iloc[0])
    our = float(rank.mean_runtime.iloc[i])
    print(f"\n  Original Mordred (as published) {pub:8.2f} s   {10000 / pub:8.0f} mol/s")
    print(f"  Our Mordred                    {our:8.2f} s   {10000 / our:8.0f} mol/s")
    print(f"  iloraz na wykresie              {pub / our:8.1f}x   UWAGA: zawyzony,")
    print("    bo pomiar 'as published' mial nieustawione watki BLAS. Poprawiona")
    print("    stawka to 25.07 s / 399 mol/s, czyli uczciwy iloraz 6.2x")
    gpu = rank[(rank.device == "cuda") & (~rank.family.isin(HIGHLIGHT))]
    print(f"  najszybszy model GPU: {gpu.family.iloc[0]} {gpu.mean_runtime.iloc[0]:.2f} s")


if __name__ == "__main__":
    main()
