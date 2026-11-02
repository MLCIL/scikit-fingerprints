"""
Odtwarza wykres "Mean time to compute embeddings for 10,000 molecules" z repo
benchmarking_molecular_models_private, z dodanym slupkiem naszej implementacji.

model_mapping i logika sa przeniesione z visualization_times.ipynb (branch dev,
komorki 2, 6 i 7). Rysowane matplotlibem, a nie seabornem, bo seaborna nie ma w
srodowisku, a matplotlib ma wbudowany styl "seaborn-v0_8-darkgrid" i te same
kolory palety "deep" - wiec wychodzi ten sam wykres bez dokladania zaleznosci.
Slupki sa dzielone po urzadzeniu tak jak przy hue z dodge, czyli szerokosc 0.4
i przesuniecie +-0.2 wewnatrz slotu kategorii.

    python plot_with_ours.py --runtime-csv ~/benchmarking_muv_runtime.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# paleta "deep" seaborna dla dwoch poziomow hue
DEVICE_COLORS = {"cpu": "#4c72b0", "cuda": "#dd8452"}
DEVICE_ORDER = ["cpu", "cuda"]

# przeniesione z visualization_times.ipynb, komorka 2
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
    "mole": "MolE", "mordred_2d": "Mordred", "chemeleon": "Chemeleon",
    "CLAMP_noleak": "CLAMP",
    # nasz slupek. MUSI byc osobna rodzina - przy "Mordred" regula minimum na
    # rodzine wybralaby nasza liczbe i slupek oryginalu zniknalby z wykresu
    "skfp_mordred_2d": "Mordred (skfp)",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-csv", required=True, help="jego runtime.csv")
    p.add_argument("--ours", default="data/clock/runtime.csv", help="pomiar repliki")
    p.add_argument("--dataset", default="ogbg-molmuv")
    p.add_argument("--out", type=Path, default=Path("mean_runtime_with_ours.png"))
    p.add_argument(
        "--our-mordred",
        action="store_true",
        help="uzyj naszego pomiaru oryginalnego mordreda zamiast jego liczby "
             "(domyslnie slupek Mordreda zostaje jego, a my dokladamy tylko swoj)",
    )
    args = p.parse_args()

    theirs = pd.read_csv(args.runtime_csv)
    ours = pd.read_csv(args.ours)
    control = ours[ours.embedder == "mordred_2d"]
    if not args.our_mordred:
        ours = ours[ours.embedder != "mordred_2d"]
    else:
        theirs = theirs[theirs.embedder != "mordred_2d"]

    df = pd.concat([theirs, ours], ignore_index=True)
    df = df[df.dataset == args.dataset].copy()
    df["family"] = df["embedder"].map(MODEL_MAPPING)
    unmapped = sorted(df[df.family.isna()].embedder.unique())
    if unmapped:
        raise SystemExit(
            f"embeddery bez wpisu w MODEL_MAPPING: {unmapped}\n"
            "bez niego groupby wyrzuca wiersz i slupek nie pojawi sie na wykresie"
        )

    # komorka 6: na rodzine zostaje najszybszy wariant/device
    best = df[
        df["mean_runtime"]
        == df.groupby(["dataset", "family"])["mean_runtime"].transform("min")
    ].sort_values("mean_runtime", ascending=False)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(15, 6))
    families = list(best.family)
    for i, (_, r) in enumerate(best.iterrows()):
        # jak hue z dodge: slot kategorii dzielony na dwa urzadzenia
        offset = -0.2 if r.device == "cpu" else 0.2
        ax.bar(i + offset, r.mean_runtime, width=0.4,
               color=DEVICE_COLORS.get(r.device, "grey"))
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=60, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_xlabel("Model Type")
    ax.set_title(
        "Mean time to compute embeddings for 10,000 molecules "
        "across different models and devices"
    )
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=DEVICE_COLORS[d]) for d in DEVICE_ORDER],
        labels=DEVICE_ORDER, title="Acceleration device", loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"{args.out}  ({len(best)} rodzin)")

    rank = best.sort_values("mean_runtime").reset_index(drop=True)
    pos = rank.index[rank.family == "Mordred (skfp)"]
    if len(pos):
        i = int(pos[0])
        print(f"\nnasz slupek: {i + 1} z {len(rank)} od najszybszego")
        if i:
            lo = rank.iloc[i - 1]
            print(f"  przed nami:  {lo.family:16s} {lo.mean_runtime:9.2f} s")
        print(f"  MY:          {'Mordred (skfp)':16s} {rank.iloc[i].mean_runtime:9.2f} s")
        if i + 1 < len(rank):
            hi = rank.iloc[i + 1]
            print(f"  za nami:     {hi.family:16s} {hi.mean_runtime:9.2f} s")
        mordred = rank[rank.family == "Mordred"]
        if len(mordred):
            print(f"\n  Mordred na wykresie: {mordred.iloc[0].mean_runtime:.2f} s"
                  f"  -> iloraz {mordred.iloc[0].mean_runtime / rank.iloc[i].mean_runtime:.1f}x")
    if len(control):
        print(f"\n  nasza kontrola oryginalnego mordreda: {control.iloc[0].mean_runtime:.2f} s"
              f" (jego: 187.37 s)")


if __name__ == "__main__":
    main()
