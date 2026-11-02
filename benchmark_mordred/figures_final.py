"""
Wykresy koncowe na poster - jeden styl, jedna paleta, jeden katalog.

Paleta jest ta sama co na wykresie kontekstowym (`muv_replication/plot_poster_context.py`),
zeby caly poster trzymal sie kolorystycznie:

  #EE7733  nasze                      - jedyny mocny akcent
  #2E6F9E  oryginalny mordred         - wyrazny niebieski
  #A6A6A6  tlo, linie pomocnicze

Odcienie pomaranczowego roznicuja RODZINY ZBIOROW na wykresie 1 - swiadomie
zostajemy w jednej rodzinie barw, bo kazdy slupek tam to nasza przewaga, a
niebieski jest zarezerwowany dla mordreda i mieszanie tego myliloby.

    python benchmark_mordred/figures_final.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OURS = "#EE7733"
OURS_LIGHT = "#F5A96F"
OURS_DARK = "#C25A14"
MORDRED = "#2E6F9E"
GREY = "#A6A6A6"
GREY_LIGHT = "#D2D2D2"
GRID = "#E8E8E8"

RES = Path("results")
OUT = RES / os.environ.get("FIGS_OUT", "figures_finals")

MOLECULENET = {"bace", "bbbp", "clintox", "esol", "freesolv", "hiv",
               "lipophilicity", "muv", "sider", "tox21", "toxcast"}
NICE = {"bace": "BACE", "bbbp": "BBBP", "clintox": "ClinTox", "esol": "ESOL",
        "freesolv": "FreeSolv", "hiv": "HIV", "lipophilicity": "Lipophilicity",
        "muv": "MUV", "sider": "SIDER", "tox21": "Tox21", "toxcast": "ToxCast",
        "dataset_rings": "ring-rich set",
        "erx_logd": "LogD", "erx_ksol": "Kinetic solubility",
        "erx_hlm_clint": "HLM CLint", "erx_mlm_clint": "MLM CLint",
        "erx_rlm_clint": "RLM CLint", "erx_caco2_papp": "Caco-2 Papp",
        "erx_caco2_efflux": "Caco-2 efflux", "erx_mppb": "MPPB",
        "erx_mbpb": "MBPB", "erx_mgmb": "MGMB"}


def style() -> None:
    """Jeden zestaw ustawien dla wszystkich wykresow."""
    plt.rcParams.update({
        # jedna typografia dla wszystkich wykresow, dobrana pod druk A0
        "font.size": 20, "axes.labelsize": 24, "axes.titlesize": 24,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 22,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.axisbelow": True, "legend.frameon": True, "legend.framealpha": 0.96,
        "legend.markerscale": 1.4, "legend.borderpad": 0.7,
        "legend.labelspacing": 0.55, "legend.handlelength": 2.2,
        "legend.edgecolor": "#CCCCCC", "legend.facecolor": "white",
        "lines.linewidth": 2.6, "lines.markersize": 9,
    })


def finish(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def label_of(ds: str) -> str:
    if ds in NICE:
        return NICE[ds]
    if ds.startswith("erx_"):
        return ds[4:].replace("_", " ")
    if ds.startswith("synthpep_"):
        return f"peptide {ds.split('_')[1]} AA"
    if ds.startswith("preactor_"):
        return ds.split("_", 1)[1].replace("_", " ")
    return ds


def family_of(ds: str) -> str:
    if ds in MOLECULENET:
        return "MoleculeNet (drug-like)"
    if ds.startswith("erx_"):
        return "ExpansionRx (ADME)"
    if ds == "dataset_rings":
        return "ring-rich set"
    if ds.startswith("synthpep_"):
        return "synthetic peptides"
    if ds.startswith("preactor_"):
        return "peptide-drug conjugates"
    return "other"


def load() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in glob.glob(str(RES / "res_*.csv"))]
    frames += [pd.read_csv(f) for f in glob.glob(str(RES / "expansionrx/res_*.csv"))]
    d = pd.concat(frames, ignore_index=True)
    return d[d.status == "ok"].copy()


def per_dataset(d: pd.DataFrame, suites: set[str]) -> pd.DataFrame:
    """Sredni koszt czasteczki i speedup dla kazdego zbioru, pomiary 1-watkowe."""
    x = d[d.suite.isin(suites) & (d.n_jobs == 1)]
    g = x.groupby(["dataset", "impl"]).agg(
        s_per_mol=("seconds_per_mol", "mean"), atoms=("mean_atoms", "max"),
        n=("n_molecules", "max")).reset_index()
    p = g.pivot(index="dataset", columns="impl", values="s_per_mol")
    p["atoms"] = g.groupby("dataset").atoms.max()
    p["n"] = g.groupby("dataset").n.max()
    p["speedup"] = p["mordred"] / p["ours"]
    p["family"] = [family_of(i) for i in p.index]
    p = p.reset_index()
    # masa czasteczkowa zamiast liczby atomow: chemicy operuja raczej na masie,
    # a jednego nie da sie przeliczyc na drugie bez skladu pierwiastkowego
    masses = pd.read_csv(RES / "dataset_masses.csv").set_index("dataset").mean_mw
    p["mw"] = p.dataset.map(masses)
    return p


# --- 1: speedup na najwazniejszych zbiorach -------------------------------
# Peptydy ucinamy na 100 aminokwasach: powyzej tego to juz bialka, a wiekszosc
# definicji tnie peptyd nawet na 50. Dotyczy to wszystkich wykresow, takze tych
# zaleznych od rozmiaru - inaczej najwiekszy punkt bylby 241-resztowym bialkiem.
PEPTIDE_MAX_ATOMS = 840  # 100 AA


def chart1(d: pd.DataFrame) -> None:
    """Cztery slupki, po jednym na rodzine zbiorow: srednia, z wasami min-max.

    Kazdy zbior osobno dawal 15 slupkow i 15 liczb, czyli duzo szumu na jedna
    obserwacje ("wszedzie okolo 7-11x"). Wasy niosa rozrzut bez dokladania cyfr.
    """
    p = per_dataset(d, {"speedup", "expansionrx", "size"})
    pep = p[p.family.isin(["synthetic peptides", "peptide-drug conjugates"])
            & (p.atoms <= PEPTIDE_MAX_ATOMS)]
    groups = [
        ("MoleculeNet", p[p.family == "MoleculeNet (drug-like)"], OURS),
        ("ExpansionRx-OpenADMET", p[p.family == "ExpansionRx (ADME)"], OURS_LIGHT),
        ("Peptides", pep, "#8C3A0A"),
    ]
    rows = [(name, float(q.speedup.mean()), float(q.speedup.min()),
             float(q.speedup.max()), len(q), col)
            for name, q, col in groups if len(q)]
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    y = range(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[5] for r in rows],
            height=0.58, edgecolor="white", linewidth=0.6, zorder=2)
    # wartosc na koncu slupka, w jego kolorze - czytelniejsze niz wasy min-max
    for i, (_, mean, _lo, _hi, _n, col) in enumerate(rows):
        ax.text(mean + 0.18, i, f"{mean:.1f}×", va="center", ha="left",
                fontsize=23, fontweight="bold", color=col)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, max(r[1] for r in rows) * 1.16)
    ax.set_title("Speedup over original Mordred", fontsize=24, pad=14)
    ax.yaxis.grid(False)
    finish(ax)
    fig.savefig(OUT / "1_speedup_datasets.png", dpi=300)
    print(f"  1_speedup_datasets.png           {len(rows)} slupkow (srednie rodzin)")
    for name, mean, lo, hi, n, _c in reversed(rows):
        print(f"      {name:16s} srednia {mean:5.1f}x   min-max {lo:4.1f}-{hi:4.1f}x   "
              f"({n} {'zbior' if n == 1 else 'zbiorow'})")


# --- 2: koszt czasteczki wobec rozmiaru -----------------------------------
def chart2(d: pd.DataFrame) -> None:
    p = per_dataset(d, {"speedup", "expansionrx", "size"}).sort_values("atoms")
    fig, ax = plt.subplots(figsize=(10, 7.0), layout="constrained")
    ax.plot(p.atoms, p["mordred"], "o-", color=MORDRED, label="Original")
    ax.plot(p.atoms, p["ours"], "o-", color=OURS, label="Our Mordred")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("atoms per molecule")
    ax.set_ylabel("seconds per molecule, single thread")
    ax.legend(loc="upper left")
    finish(ax)
    fig.savefig(OUT / "2_cost_per_molecule.png", dpi=300)
    # wykladnik skalowania - do podpisu
    for impl in ("mordred", "ours"):
        k = np.polyfit(np.log(p.atoms), np.log(p[impl]), 1)[0]
        print(f"  2_cost_per_molecule.png  {impl}: koszt ~ n^{k:.2f}")


# --- 3: speedup wobec rozmiaru czasteczki ---------------------------------
def chart3(d: pd.DataFrame) -> None:
    p = per_dataset(d, {"speedup", "expansionrx", "size"}).sort_values("atoms")
    fig, ax = plt.subplots(figsize=(10, 7.0), layout="constrained")
    # NIE laczymy punktow linia: to osobne zbiory, a lamana sugerowalaby trajektorie.
    k, b = np.polyfit(np.log(p.atoms), p.speedup, 1)
    xs = np.linspace(np.log(p.atoms.min()), np.log(p.atoms.max()), 100)
    ax.plot(np.exp(xs), k * xs + b, "--", color=GREY, linewidth=2.0, zorder=1)
    ax.plot(p.atoms, p.speedup, "o", color=OURS, markersize=10,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.text(0.04, 0.94, f"+{k:.1f}× per e-fold in size", transform=ax.transAxes,
            fontsize=21, color="#666666", va="top")
    top = p.loc[p.speedup.idxmax()]
    ax.annotate(f"{top.speedup:.1f}×", xy=(top.atoms, top.speedup),
                xytext=(-50, 8), textcoords="offset points", fontsize=28,
                fontweight="bold", color=OURS_DARK)
    ax.set_xscale("log")
    ax.set_xlabel("atoms per molecule")
    ax.set_ylabel("speedup over Original Mordred")
    ax.set_ylim(0, p.speedup.max() * 1.16)
    finish(ax)
    fig.savefig(OUT / "3_speedup_vs_size.png", dpi=300)
    print(f"  3_speedup_vs_size.png    {p.atoms.min():.0f}-{p.atoms.max():.0f} atomow, "
          f"speedup {p.speedup.min():.1f}-{p.speedup.max():.1f}x")


# --- 4: skalowalnosc wielordzeniowa ---------------------------------------
def chart4(d: pd.DataFrame, dataset: str = "synthpep_20") -> None:
    t = d[(d.suite == "threads") & (d.dataset == dataset)]
    g = t.groupby(["impl", "n_jobs"]).throughput_mol_s.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 7.0), layout="constrained")
    base = g["ours"].iloc[0]
    ideal = base * g.index.to_numpy()
    ax.plot(g.index, ideal, "--", color=GREY, linewidth=2.0, label="Ideal scaling")
    ax.plot(g.index, g["mordred"], "o-", color=MORDRED, label="Original")
    ax.plot(g.index, g["ours"], "o-", color=OURS, label="Our Mordred")
    for n, dx, dy in ((1, 12, -30), (g.index.max(), -6, -34)):
        r = g.loc[n, "ours"] / g.loc[n, "mordred"]
        ax.annotate(f"{r:.1f}×", xy=(n, g.loc[n, "ours"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=22, fontweight="bold",
                    color=OURS_DARK, ha="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(list(g.index)); ax.set_xticklabels([str(i) for i in g.index])
    ax.set_xlabel("worker processes")
    ax.set_ylabel("mols / second")
    ax.legend(loc="upper left")
    finish(ax)
    fig.savefig(OUT / "4_parallel_scaling.png", dpi=300)
    lo, hi = g.index.min(), g.index.max()
    print(f"  4_parallel_scaling.png   {dataset}: nasze {g.loc[lo,'ours']:.1f} -> "
          f"{g.loc[hi,'ours']:.1f} mol/s ({g.loc[hi,'ours']/g.loc[lo,'ours']:.1f}x), "
          f"mordred {g.loc[hi,'mordred']/g.loc[lo,'mordred']:.1f}x")


def chart_pair(d: pd.DataFrame) -> None:
    """Koszt i speedup obok siebie, wspolna os X - jako jeden obiekt do wklejenia.

    Same w sobie te dwa panele mowia to samo dwa razy, ale postawione obok siebie
    mowia co innego: lewy, ze KOSZT rosnie tak samo w obu implementacjach (krzywe
    rownolegle, wiec zysk jest staly, nie asymptotyczny), prawy - o ile dokladnie.
    """
    p = per_dataset(d, {"speedup", "expansionrx", "size"})
    # ADME wypada: 10 zbiorow o niemal identycznej masie (378-425 Da) zageszczalo
    # lewa strone, nie wnoszac nic do zaleznosci od rozmiaru. Sa na wykresie 1.
    p = p[(p.family != "ExpansionRx (ADME)") & (p.atoms <= PEPTIDE_MAX_ATOMS)]
    p = p.sort_values("mw")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 7.0), layout="constrained",
                                 sharex=True)

    a1.plot(p.mw, p["mordred"], "o-", color=MORDRED, label="Original", alpha=0.9)
    a1.plot(p.mw, p["ours"], "o-", color=OURS, label="Our Mordred", alpha=0.9)
    a1.set_yscale("log")
    a1.set_ylabel("seconds / mol")
    a1.set_title("Time per molecule", fontsize=24, pad=12)
    a1.legend(loc="upper left")

    # Jeden marker i zadnej legendy: panel ma powiedziec "speedup rosnie z
    # rozmiarem", a piec symboli z opisami tylko zaciemnialo obraz. Rodziny
    # zbiorow sa rozroznione na wykresie slupkowym, tu nie sa potrzebne.
    k, b = np.polyfit(np.log(p.mw), p.speedup, 1)
    xs = np.linspace(np.log(p.mw.min()), np.log(p.mw.max()), 100)
    a2.plot(np.exp(xs), k * xs + b, "--", color=GREY, linewidth=2.0, zorder=1)
    a2.plot(p.mw, p.speedup, "o", color=OURS, markersize=10,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    top = p.loc[p.speedup.idxmax()]
    # pod linia trendu, na wysokosci y=8: tam po prawej stronie nie ma zadnych
    # punktow, wiec etykieta stoi swobodnie
    a2.text(top.mw * 0.62, 10.6, f"{top.speedup:.1f}×", fontsize=28, fontweight="bold",
            color=OURS_DARK, ha="center", va="center")
    a2.set_ylim(0, p.speedup.max() * 1.18)
    a2.set_ylabel("speedup")
    a2.set_title("Speedup over original Mordred", fontsize=24, pad=12)

    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("molecular weight  [Da]")
        finish(ax)
    fig.savefig(OUT / "2_3_large_molecules.png", dpi=300)
    print(f"  2_3_large_molecules.png          para paneli, {p.atoms.min():.0f}-"
          f"{p.atoms.max():.0f} atomow, trend +{k:.1f}x na e-krotnosc")


def chart4_pair(d: pd.DataFrame, explain: bool = False,
                show_scaling: bool = True) -> None:
    """Skalowalnosc na dwoch zbiorach obok siebie - tanie i drogie czasteczki.

    Dwa panele mowia razem to, czego zaden nie mowi sam: narzut zrownoleglenia
    mordreda (apply_async na KAZDA czasteczke, kalkulator przez proxy Manager())
    dominuje tylko na tanich czasteczkach. Na HIV mordred skaluje sie 5.3x, na
    peptydach 35x - i tam to MY skalujemy sie slabiej, bo skfp dzieli prace
    statycznie. Pokazanie samych peptydow bylo wyborem panelu mniej dla nas
    korzystnego, ale takze mniej informacyjnego.
    """
    titles = {"hiv": "Drug-like", "synthpep_20": "Peptides"}
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.0), layout="constrained")
    for ax, ds in zip(axes, ("hiv", "synthpep_20")):
        g = (d[(d.suite == "threads") & (d.dataset == ds)]
             .groupby(["impl", "n_jobs"]).throughput_mol_s.mean().unstack(0))
        n = g.index.to_numpy()
        ax.plot(n, g["ours"].iloc[0] * n, "--", color=GREY, linewidth=2.0,
                label="Ideal scaling")
        ax.plot(n, g["mordred"], "o-", color=MORDRED, label="Original")
        ax.plot(n, g["ours"], "o-", color=OURS, label="Our Mordred")
        # Dwie rozne wielkosci na jednym panelu, wiec kazda dostaje inna forme:
        #  - przewaga nad oryginalem: pionowa klamra miedzy krzywymi. Na osi
        #    logarytmicznej jej dlugosc to logarytm ilorazu, wiec wzrost przewagi
        #    widac geometrycznie, bez czytania liczb.
        #  - wlasne skalowanie z 1 na 64 workery: podpis przy naszej krzywej.
        for k, ha, dx in ((n.min(), "left", 1.13), (n.max(), "right", 0.88)):
            lo, hi = g.loc[k, "mordred"], g.loc[k, "ours"]
            # pionowa kreska przerywana, zakonczenia ciagle - na krotkim
            # odcinku kreskowanie i tak by sie nie odczytalo
            ax.plot([k, k], [lo, hi], "--", color="#555555", linewidth=2.0,
                    dashes=(4, 3), zorder=4)
            for y in (lo, hi):  # zakonczenia klamry
                ax.plot([k * 0.93, k * 1.075], [y, y], color="#555555",
                        linewidth=2.0, zorder=4)
            ax.text(k * dx, (lo * hi) ** 0.5, f"{hi / lo:.1f}×", fontsize=22,
                    fontweight="bold", color="#333333", ha=ha, va="center")
        # podpis skalowania nad linia idealna, w pasie zrobionym przez podniesienie
        # gornej granicy osi - inaczej wchodzil na krzywe
        scaling = g.loc[n.max(), "ours"] / g.loc[n.min(), "ours"]
        ideal_max = g.loc[n.min(), "ours"] * n.max()
        if not show_scaling:
            pass  # wariant bez wlasnego skalowania, same klamry przewagi
        elif explain:
            # Wariant objasniajacy: pozioma linia z poziomu jednego workera i
            # pionowa klamra do poziomu 64 workerow, poza ostatnim punktem, zeby
            # nie kolidowac z klamra przewagi. Widac wtedy wprost, MIEDZY CZYM
            # jest ten mnoznik.
            y1, y64 = g.loc[n.min(), "ours"], g.loc[n.max(), "ours"]
            xr = n.max() * 1.6
            ax.plot([n.min(), xr], [y1, y1], ":", color=OURS, linewidth=2.0, zorder=3)
            ax.plot([xr, xr], [y1, y64], "--", color=OURS_DARK, linewidth=2.2,
                    dashes=(4, 3), zorder=4)
            for y in (y1, y64):
                ax.plot([xr * 0.9, xr * 1.11], [y, y], color=OURS_DARK,
                        linewidth=2.2, zorder=4)
            # z prawej strony klamry: z lewej stoi juz podpis przewagi
            ax.text(xr * 1.18, (y1 * y64) ** 0.5, f"{scaling:.0f}×", fontsize=22,
                    fontweight="bold", color=OURS_DARK, ha="left", va="center")
        else:
            # w prawa czesc pasma nad linia idealna - po lewej stoi legenda
            ax.text(n.max() * 0.42, ideal_max * 1.7,
                    f"{scaling:.0f}× scaling", fontsize=22, fontweight="bold",
                    color=OURS_DARK, ha="center", va="center")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        # granice PO zmianie skali i z danych: przed nia os jest liniowa i ma dol
        # w zerze, co po przejsciu na log rozciaga zakres na kilkanascie dekad
        top_mult = 1.4 if (explain or not show_scaling) else 3.4
        ax.set_ylim(min(g["mordred"].min(), g["ours"].min()) * 0.65,
                    ideal_max * top_mult)
        ax.set_xticks(list(n)); ax.set_xticklabels([str(i) for i in n])
        if explain and show_scaling:
            ax.set_xlim(right=n.max() * 2.6)
        ax.set_xlabel("worker processes")
        ax.set_title(titles[ds], fontsize=24, pad=12)
        finish(ax)
    for ax in axes:
        ax.set_ylabel("mols / second")
    axes[0].legend(loc="upper left")
    if not show_scaling:
        name = "4_parallel_scaling_pair_gap_only.png"
    elif explain:
        name = "4_parallel_scaling_pair_explained.png"
    else:
        name = "4_parallel_scaling_pair.png"
    fig.savefig(OUT / name, dpi=300)
    print(f"  {name:32s} dwa zbiory: hiv + synthpep_20")


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"pomiarow: {len(d)} | katalog: {OUT}")
    chart1(d)
    chart_pair(d)
    chart2(d); chart3(d); chart4(d)
    chart4_pair(d)
    chart4_pair(d, explain=True)
    chart4_pair(d, show_scaling=False)


if __name__ == "__main__":
    main()
