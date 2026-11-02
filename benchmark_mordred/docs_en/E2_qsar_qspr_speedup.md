# E2 — Speedup across QSAR/QSPR datasets

Poster element: the **QSAR/QSPR datasets** bar chart.

## Question

Is the speedup consistent across the datasets practitioners actually use, or does
it depend on a favourable choice of benchmark?

## Data and preprocessing

Three families, 29 datasets, 44 507 molecules in total.

**MoleculeNet, 11 datasets.** FreeSolv, ESOL, SIDER, ClinTox, BACE, BBBP,
Lipophilicity, Tox21, ToxCast, HIV, MUV — loaded through
`skfp.datasets.moleculenet`.

**ExpansionRx-OpenADMET, 10 ADME datasets.** Caco-2 efflux and Papp, HLM/MLM/RLM
CLint, KSOL, LogD, MPPB, MBPB, MGMB.

**Peptides, 8 datasets.** Synthetic peptides of 5-100 residues, built from the 20
standard amino acids with a seeded RNG, plus real peptides from the PeptideReactor
collection (`ace_vaxinpad`, `hiv_lpv`).

### Exclusions applied here

**HIV and MUV are capped at a random 10 000 molecules**, seed 0, both
implementations receiving the same sample. Without the cap these two datasets would
dominate the Phase A budget. Full sizes are 41 127 and 93 087.

**HIV loses one more molecule to `drop_uncomputable`** and appears as 9 999 — the
LinAlgError organometallic that happens to fall inside the seed-0 subsample. See
`00_common_setup.md`.

**Peptides are cut at 840 atoms (100 residues)**, so 8 of the 10 measured peptide
datasets enter this chart.

## Protocol

**Phase A**, single thread, BLAS pinned, 3 repetitions, one process per dataset run
concurrently (contention does not distort ratios).

```bash
for d in freesolv esol sider clintox bace bbbp lipophilicity tox21 toxcast hiv muv; do
    python benchmark_mordred/bench_run.py --suite speedup --only $d --out results/res_speedup_$d.csv &
done
bash benchmark_mordred/run_expansionrx.sh
```

Speedup is computed **per dataset first** (`mordred.s_per_mol / ours.s_per_mol`)
and only then averaged within a family. Averaging molecule costs first would let
the largest datasets dominate.

## Results

| family | mean | min-max | datasets |
|---|---|---|---|
| Peptides | **11.2×** | 7.6 - 13.9× | 8 |
| ExpansionRx-OpenADMET | **7.6×** | 7.4 - 8.2× | 10 |
| MoleculeNet | **7.5×** | 5.6 - 9.0× | 11 |

MoleculeNet in full: FreeSolv 5.63, ESOL 6.83, Lipophilicity 7.05, ToxCast 7.17,
MUV 7.19, Tox21 7.22, BBBP 7.74, BACE 7.87, ClinTox 7.99, SIDER 8.97, HIV 9.04 —
median 7.22.

**"At least 7.5×" refers to the lowest of the three family means.** The weakest
single dataset is FreeSolv at 5.6×, which also has the smallest molecules in the
study (8.7 atoms). If asked for a worst case, that is the honest answer.

## A premise that did not survive measurement

The ExpansionRx group was added at a supervisor's suggestion, framed as *"what
happens with small molecules and small datasets"*. Measurement contradicted the
premise: these datasets average **28.6 atoms**, more than HIV (25.5) or MUV (24.3),
and far more than FreeSolv (8.7).

What distinguishes them is not size but a **tight distribution** — median 28,
maximum 78, against HIV's maximum of 214. We kept them for that property and
reported the corrected characterisation. They later served as a control for the
cost-tail question in E4.

A second supervisor expectation — that ring-rich molecules would show a
*"very large speedup"* — was also tested, on a purpose-built 9 998-molecule
ring-rich set (34.7 atoms, 4.9 rings): the result was **8.19×**, in line with any
other 34-atom dataset (BACE 7.87×, SIDER 8.97×). The expectation was corrected by
measurement rather than carried forward.

## Presentation choice

An earlier version of this chart showed 15 individual bars and was unreadable
("roughly 7-11× everywhere"). Aggregating to three family means is a presentation
decision, not a measurement one; the per-dataset numbers are above and in the CSVs.
