# E3 — Scaling with molecule size

Poster element: the **Molecule size scaling** log-log panel pair.

## Question

How does the advantage change as molecules grow from drug-like to peptide scale?

## Data and preprocessing

Same Phase A runs as E2, plus the two largest peptide points that E2's cut excludes.

**Synthetic peptides** (`synthpep_L`, L = 5…160 residues): sequences drawn
uniformly from the 20 standard amino acids with a seeded RNG, built with
`Chem.MolFromSequence`. Generation retries until the target count of parseable
molecules is reached.

**Real peptides** from the **PeptideReactor** collection, read from FASTA
(`ace_vaxinpad`, `hiv_lpv`, `hiv_nvp`). These are plain peptide sequences, not
conjugates.

### Exclusions and size choices

**Truncation happens before filtering, deliberately.** `peptide_reactor` loads the
entire FASTA file. Filtering first would have made `hiv_nvp` 733 molecules of
~2000 atoms rather than the **two** we intend to measure — a difference between
minutes and days of compute.

**Counts decrease as molecules grow**: 200 → 100 → 50 → 20 → 10 → 5 → 2. At that
scale cost is dominated by deterministic O(n³) spectral work, so between-molecule
variance is small and few molecules suffice. One 1994-atom molecule costs about
19 minutes across both implementations; `preactor_hiv_nvp` is the single longest
process in the entire benchmark.

**The chart cuts at 840 atoms (100 residues).** `synthpep_160` (1332 atoms, 14.6×)
and `preactor_hiv_nvp` (1994 atoms, 18.0×) are measured and in the CSVs, but
excluded from a chart labelled "peptides", because above 100 residues these are
proteins.

## Axis choice

The x-axis is **molecular mass**, not atom count, at a supervisor's request —
chemists reason in daltons. The two cannot be interconverted without elemental
composition, so mean masses are computed separately (`benchmark_mordred/dataset_masses.py`
→ `results/dataset_masses.csv`) and joined per dataset.

## Results

| dataset | n | atoms | mass [Da] | speedup |
|---|---|---|---|---|
| freesolv | 642 | 8.7 | **138** | **5.63** |
| esol | 1128 | 13.3 | 206 | 6.83 |
| bace | 1513 | 34.1 | 483 | 7.87 |
| synthpep_5 | 200 | 42.6 | 610 | 7.64 |
| synthpep_10 | 100 | 84.1 | 1 203 | 9.24 |
| preactor_ace_vaxinpad | 200 | 101.1 | 1 341 | 10.63 |
| synthpep_20 | 50 | 167.2 | 2 387 | 10.90 |
| synthpep_40 | 20 | 333.8 | 4 758 | 12.76 |
| synthpep_80 | 10 | 666.5 | 9 498 | 11.82 |
| **preactor_hiv_lpv** | 10 | 764.3 | **10 886** | **13.86** |
| synthpep_100 | 5 | 831.8 | 11 877 | 12.43 |
| _synthpep_160_ | 5 | 1 332 | 18 979 | _14.59_ |
| _preactor_hiv_nvp_ | 2 | 1 994 | 28 109 | _18.04_ |

Italic rows are outside the chart. The poster's endpoints are **5.6× at 138 Da**
(FreeSolv) and **13.9× at 10 886 Da** (`preactor_hiv_lpv`, the real peptide
lopinavir inhibitor set).

Fitted exponents over the charted range: mordred ~n^1.85, ours ~n^1.67.

## Threats to validity, stated

**We do not claim better asymptotic complexity.** The fitted exponents come from a
range dominated by small molecules. Across the full ladder both implementations are
bounded by the same O(n³) spectral decompositions, which neither avoids. An earlier
two-point fit suggested n^2.07 against n^2.83; we withdrew that claim when more
points became available, and will not repeat it.

**The trend is not monotonic.** `synthpep_80` (11.8×) sits below `synthpep_40`
(12.8×), and `synthpep_100` (12.4×) below `hiv_lpv` (13.9×). At n = 5-10 this is
within noise. The chart therefore draws a fitted trend and **deliberately does not
connect the points** — a polyline would imply a trajectory the data does not
support.

**A plateau is not excluded.** Earlier measurements, taken before the optimisation
stack, suggested the speedup plateaued near 3.5×. It now rises to 18× at 1994 atoms
with no plateau visible. That is evidence of absence only within the measured range.

**Small-n points carry real uncertainty.** The two largest datasets have n = 5 and
n = 2. We accept this because the cost is deterministic at that size, but it should
not be read as a tight estimate.
