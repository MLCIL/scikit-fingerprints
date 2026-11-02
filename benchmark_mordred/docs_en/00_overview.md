# Benchmark methodology

Companion document to the poster *Optimizing Mordred Descriptors Calculation for
Scalable Molecular Feature Extraction*. It specifies how every number on the poster
was produced, and states plainly what each measurement does and does not support.

---

## 1. Setup common to all experiments

### Hardware

All measurements were taken on a single machine, `bitai2`: AMD EPYC 7763,
**64 physical cores / 128 hardware threads**, 503 GB RAM. Under full 64-core load
the CPU drops from ~3.5 GHz to ~2.45 GHz, which is visible in the parallel
efficiency figures and is accounted for in their interpretation.

GPU figures quoted for neural baselines come from an **RTX 4000 Ada**, inherited
from the reference benchmark described in §7.

### Software

| | |
|---|---|
| our implementation | `skfp.fingerprints.NewMordredFingerprint`, branch `mordred-benchmarks`, commit `0bec710` |
| reference implementation | **mordredcommunity 2.0.7** (`JacksonBurns/mordred-community`), imported as `mordred` |
| RDKit | 2026.03.5 |

We benchmark against **mordred-community**, not Moriwaki's original `mordred` 1.2.0.
The original package is unmaintained and does not run on current RDKit or Python;
mordred-community is its maintained successor and the package people actually
install today. Our implementation is a port of it, and the licence is honoured in
the source headers.

The commit hash is written into every row of every result CSV, so each measurement
can be traced to the exact code that produced it.

### Scope

Everything on the poster concerns the **1613 two-dimensional descriptors**.
3D descriptors are deliberately out of scope: they require a conformer, and
conformer generation is a separate methodological problem that would confound the
comparison. This is stated in the benchmark README, not discovered after the fact.

### BLAS thread pinning — the single most important control

`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` are all set to `1` **before
NumPy is imported** (`bench_run.py:64-71`). OpenBLAS reads these at library load
time, so setting them later has no effect.

This matters because the two implementations are not symmetric without it. Our code
performs spectral decompositions through NumPy, which hands them to OpenBLAS;
left unpinned, OpenBLAS claims every core. The original mordred is largely pure
Python and claims one. A nominally "single-threaded" comparison would then be
20+ cores against one.

Conversely, in *parallel* runs the asymmetry reverses and favours us: our code is
driven by `joblib`/`loky`, which pins each worker automatically
(`OMP_NUM_THREADS = cpu_count // n_jobs`), whereas mordred uses a raw
`multiprocessing.Pool`, which does not. Unpinned, the original is **7.3× slower
than it should be, with bit-identical output**. We measured both configurations
and report both (§7).

### Two-phase execution protocol

**Phase A** produces *ratios*. Machine contention affects both implementations
alike, so a ratio survives it, and many processes may run concurrently. We capped
concurrency at 32-45 threads so processes would not fight over resources.

**Phase B** produces *absolute* figures (seconds, molecules per second). These are
invalid under contention, so Phase B runs on an otherwise idle machine, serialised
behind an `flock` lock.

Every experiment below states which phase it belongs to.

### Repetitions and sampling

Three repetitions by default, two for the thread-scaling suite (where the
single-worker point alone costs half the sweep, and run-to-run spread on an idle
machine is under 1%). Repetitions of the two implementations are **interleaved**
(`ours, mordred, ours, mordred, ...`) so that both see comparable machine
conditions within each round, rather than one running entirely before the other.

HIV and MUV are subsampled to a random 10 000 molecules with a fixed seed in the
per-dataset suites; both implementations receive **the same molecules**.

---

## 2. Individual descriptor families (poster table)

**Question.** Where does the speedup come from, family by family?

**Design.** Both sides compute a descriptor family *together with its
prerequisites*, from scratch:

- mordred: `Calculator(module)` evaluates only that family's dependency subtree;
- ours: we construct exactly the objects that `calculator.compute` would hand the
  family, and time them together with it.

**Why not a profiler.** `cProfile` attributes only *self* time. Our families would
look artificially thin, because shared preparation — distance matrices, subgraph
enumeration, atomic properties — lives in other modules, while mordred's families
look thick because it repeats that work inside each one. The ratio would then
report *where the code lives*, not how much it computes. That is not defensible,
so we discarded it.

**These ratios are a lower bound.** Here prerequisites are paid once *per family*;
in a real run they are paid once *per molecule* for all families at once. The table
therefore **excludes** the benefit of shared precomputation, which is exactly what
the third bullet on the poster says.

**Data.** BACE from MoleculeNet, all 1513 molecules, mean 34.1 atoms. Single
thread, best of 3.

| family | features | mordred | ours | speedup |
|---|---|---|---|---|
| ExtendedTopochemicalAtom | 45 | 24.75 s | 2.65 s | 9.3× |
| MolecularDistanceEdge | 19 | 9.75 s | 1.35 s | 7.2× |
| Autocorrelation | 606 | 17.11 s | 2.61 s | 6.6× |
| EState | 316 | 18.18 s | 3.53 s | 5.1× |
| BaryszMatrix | 104 | 19.76 s | 4.01 s | 4.9× |
| Chi | 56 | 49.47 s | 10.13 s | 4.9× |
| InformationContent | 42 | 21.93 s | 4.83 s | 4.5× |
| MolecularId | 12 | 13.39 s | 3.43 s | 3.9× |
| PathCount | 21 | 35.36 s | 9.42 s | 3.8× |
| DetourMatrix | 14 | 5.98 s | 2.03 s | 3.0× |

**Two families are omitted, and we say so.** Under this protocol KappaShapeIndex
scores 0.30× and AdjacencyMatrix 0.93×. That is an artefact of the protocol, not a
regression: in isolation our side pays for a full `Subgraphs` enumeration and
eigendecomposition in order to produce three features, whereas in a real run that
cost is already paid by other families. The omission is deliberate and we can
justify it; it is not a suppressed regression.

**Note the consistency check.** No single family reaches the 7.5× that the whole
fingerprint achieves on the same dataset. The difference *is* the shared
precomputation.

---

## 3. Speedup across QSAR/QSPR datasets (bar chart)

**Phase A**, single thread, three repetitions, ratio per dataset computed first and
only then averaged — otherwise large datasets would dominate.

| family | mean | min-max | datasets |
|---|---|---|---|
| Peptides | 11.2× | 7.6 - 13.9× | 8 |
| ExpansionRx-OpenADMET | 7.6× | 7.4 - 8.2× | 10 |
| MoleculeNet | 7.5× | 5.6 - 9.0× | 11 |

"At least 7.5×" on the poster refers to **the lowest of the three means**, not to
the lowest individual dataset. The weakest single dataset is FreeSolv at **5.6×**,
which also has the smallest molecules in the whole study (8.7 atoms). If asked for
the worst case, the answer is 5.6×.

MoleculeNet in full: FreeSolv 5.63, ESOL 6.83, Lipophilicity 7.05, ToxCast 7.17,
MUV 7.19, Tox21 7.22, BBBP 7.74, BACE 7.87, ClinTox 7.99, SIDER 8.97, HIV 9.04
(median 7.22).

**Peptides are cut at 840 atoms (100 residues).** Above that these are proteins,
and most definitions cut "peptide" at 50 residues. Without the cut the largest
point would be a 241-residue protein and the bar would stop describing peptides.

**On the ExpansionRx group.** These ten ADME datasets were added on a supervisor's
suggestion to show "what happens with small molecules and small datasets". The
premise did not survive measurement: they average **28.6 atoms**, more than HIV
(25.5) or MUV (24.3). What distinguishes them is a **tight distribution** —
median 28, maximum 78, against HIV's maximum of 214. We kept them for that reason,
and report the corrected characterisation rather than the original premise.

---

## 4. Scaling with molecule size (log-log panel pair)

Same Phase A runs as §3. The x-axis is **molecular mass** rather than atom count,
at a supervisor's request: chemists reason in daltons. The two cannot be converted
without elemental composition, so masses are computed separately
(`dataset_masses.py`) and joined per dataset.

Quoted endpoints: **5.6× at 138 Da** (FreeSolv) and **13.9× at 10 886 Da**
(`preactor_hiv_lpv`, a real peptide from PeptideReactor). The ladder runs from
8.7 to 832 atoms within the chart, with two further points (1332 and 1994 atoms)
measured but excluded by the 100-residue cut.

Dataset sizes **decrease as molecules grow** (200 → 100 → 50 → 20 → 10 → 5 → 2).
At that scale cost is dominated by deterministic O(n³) spectral work, so variance
is small and few molecules suffice. One 1994-atom molecule costs about 19 minutes
across both implementations.

### What this chart does not claim

**We do not claim better asymptotic complexity.** Fitting the measured range gives
exponents of n^1.85 for mordred and n^1.67 for ours, but that range is dominated by
small molecules. Across the full ladder both implementations are bounded by the
same O(n³) spectral decompositions, which neither can avoid. An earlier two-point
fit suggested n^2.07 against n^2.83; we discarded that claim when more points
became available.

**The trend is not monotonic.** `synthpep_80` (11.8×) sits below `synthpep_40`
(12.8×). At n = 5-10 molecules this is within noise, which is why the chart draws
a fitted trend rather than connecting the points — a polyline would imply a
trajectory that the data does not support.

**A plateau is not excluded.** Earlier measurements, before the optimisation stack,
suggested the speedup plateaued around 3.5×. It now rises to 18× at 1994 atoms with
no plateau visible in the measured range. That is not proof that none exists.

---

## 5. Parallel scaling (panel pair)

**Phase B**, idle machine, worker counts 1-64, two repetitions. Repeat-to-repeat
spread is below 1%.

| workers | HIV, ours | HIV, mordred | peptides, ours | peptides, mordred |
|---|---|---|---|---|
| 1 | 67.5 | 8.0 | 10.2 | 0.9 |
| 64 | 2158.1 | 42.0 | 279.6 | 33.0 |
| **scaling** | **32.0×** | 5.2× | **27.4×** | 35.1× |

### The two implementations are given different dataset sizes, on purpose

Ours receives 41 124 HIV molecules and 3 200 peptides; mordred receives 5 999 and
320. With *equal* sizes our workers finish faster than process startup takes: on
320 peptides at 64 workers mordred showed 67% efficiency and ours 19%, and the
chart would have suggested that *mordred* scales better. With sizes tuned, ours
reaches 49%, which is what the hardware allows. Throughput is a per-molecule
quantity, so the curves remain comparable.

### Things we state rather than hide

**On peptides mordred scales better than we do** (35.1× against 27.4×). The cause
is on our side: `skfp` splits work statically (one task per worker), so the
makespan is set by the slowest worker, while mordred dispatches per molecule and
balances better. In absolute terms it is still 8.5× slower at 64 workers.

**Our 32× at 64 workers is 50% efficiency, not 100%.** Part is the static split
above; part is physics — the CPU drops from 3.5 to 2.45 GHz under full load.
"Near-linear scaling" on the poster describes the shape of the curve on log axes,
not an efficiency claim.

**Do not read the 51× gap on the HIV panel as a parallelism advantage.** It is
correctly measured and repeatable for that sample, but it describes an interaction
between mordred and this particular dataset. Four candidate explanations were
tested and rejected (missing BLAS pinning; more expensive molecules in HIV, whose
single-thread rates are in fact nearly equal at 0.0885 vs 0.0832 s/mol; process
state at fork; a size-based tail cut). Measuring the **cost distribution directly**
settles it:

| | HIV | MUV |
|---|---|---|
| median s/mol | 0.0910 | 0.0850 |
| maximum | **6.152 s** | 0.223 s |
| share of cost in the top 1% | **18.0%** | 2.2% |
| max atoms / rings | 114 / 16 | 36 / 6 |

The *typical* molecule costs the same in both datasets; only the tail differs. MUV
is a curated, size-bounded set; HIV is raw screening data. Because mordred
dispatches per molecule, throughput on HIV depends on whether the few pathological
molecules land in the sample: re-drawing 6 000 molecules moves it between
**34.8 and 83.1 mol/s**. Our side computes the whole dataset and is therefore
stable; mordred's subsample is unbiased in expectation but noisy.

The figures the poster quotes — 32× and 27× — are our own scaling, measured on one
sample at both worker counts, and are unaffected by this.

---

## 6. Pretraining-scale throughput (MolPILE)

MolPILE, 222 M molecules. We took the first 1 000 000; **996 819** remained after
filtering.

**3 181 molecules (0.32%) were removed** because our implementation either raises
or does not terminate on them — fullerenes and graphene fragments, where the detour
matrix requires a longest simple path, an NP-hard problem. The original mordred has
a timeout guard there and returns a placeholder; we do not. The filter uses a hard
`SIGALRM` budget and the excluded indices are recorded.

**This section reports our implementation only.** It demonstrates scale, not a
ratio; there is no mordred bar by design.

### Two protocols, two numbers

| protocol | work split | throughput | 1 M molecules |
|---|---|---|---|
| reference-repository protocol | batches of 10 000 | 605 mol/s | 27.3 min |
| maximum | `batch_size=250` | **3 800 mol/s** | **4.4 min** |

The 6.3× difference comes entirely from how work is divided. `skfp` defaults to
`batch_size = len(data) // n_jobs`, i.e. one task per worker, so the makespan is
set by the slowest chunk and molecule cost has a heavy tail. At `batch_size=250`
there are ~40 tasks per worker, enough to balance.

The poster quotes the maximum. Three repetitions at 96 workers: 3809.2, 3796.8,
3793.4 mol/s — **3 800 ± 8**. At 64 workers: ~3 175, so 96 workers add 19.7%.

The headline figure is `seconds_descriptors`, i.e. descriptor computation alone. It
excludes SMILES parsing (~21.8 s) and writing 12.9 GB to disk (~33 s). This matches
the reference harness, whose timer also wraps only `model.embed(batch)`, with
persistence outside the loop — the exclusion is a condition of comparability, not a
flattering choice. The raw timings are published so any variant can be recomputed.

**222 M in ~16 h is an extrapolation**, not a measurement: 222e6 / 3800 = 16.2 h.
We measured 1 M, a random slice of the same corpus.

---

## 7. Throughput against other molecular embeddings

Base: the "mean time to compute embeddings for 10 000 molecules" comparison from an
independent benchmark of 61 model configurations on **ogbg-molmuv** (93 087
molecules, nine full batches). We added three Mordred bars measured **inside that
same harness**, through its own wrapper interface.

| bar | s / 10k | mol/s | provenance |
|---|---|---|---|
| Mordred (as published) | 187.37 | **53** | the reference repository's own measurement, BLAS unpinned |
| Mordred (BLAS pinned) | 25.07 | 399 | our control, same binary |
| Our Mordred | 4.02 | **2 485** | this work |

### The caveat we volunteer before being asked

**2485 / 53 = 47× is inflated.** The "53" comes from a run in which mordred's BLAS
threads were unset, making it ~7.3× slower than it should be with bit-identical
output. **Against the same binary properly configured the honest ratio is 6.2×.**

We keep the "as published" bar because all other models in that comparison were
measured in the same harness and the same configuration. Removing the handicap for
mordred alone would make the comparison asymmetric with respect to every other
model. So both bars are shown.

Two supporting observations. Our implementation is insensitive to pinning (2534 vs
2485 mol/s) because `joblib`/`loky` pins workers itself. And our own unpinned
measurement of mordred (182.4 s) reproduces the published one (187.4 s) to within
3%, which confirms we replicated their protocol rather than measuring a different
machine.

### What "better throughput than all neural networks" means

Verified across all 61 configurations. Faster than our 4.02 s are only classical
hashed fingerprints — ECFP (0.71 s), Topological Torsion (0.73 s), their count
variants, Atom Pair (1.27 s) — which are a different category. The fastest neural
model is `mole` at 6.37 s; CheMeleon is 9.88 s, MoLFormer-XL 12.35 s, ChemBERTa
~16 s, COATI 116 s.

This is a claim about **throughput only**. We make no claim that Mordred descriptors
are better representations than learned embeddings.

---

## 8. Summary of what we do not claim

- not better asymptotic complexity — both implementations are O(n³)-bound on large
  molecules
- not a general parallelism advantage — the 51× HIV gap is dataset-specific, and on
  peptides mordred scales better than we do
- not 47× over the original — that figure carries a configuration handicap; 6.2× is
  the defensible number against the same binary
- not better representation quality — this work is about computation time
- not coverage of 3D descriptors — deliberately out of scope
- the 222 M figure is an extrapolation from a measured 1 M

## 9. Reproducibility

Scripts and result CSVs live on the `mordred-benchmarks` branch of our
scikit-fingerprints fork. The repository's `.gitignore` excludes `*.csv` and
`*.png`, so results were committed with `git add -f`; every row carries its
`branch@commit` and host.

```
benchmark_mordred/bench_run.py            measurement harness, all suites
benchmark_mordred/figures_final.py        charts 1-4
benchmark_mordred/profil/isolated_families.py   per-family table
benchmark_mordred/molpile/                MolPILE preparation, throughput, metrics
benchmark_mordred/muv_replication/        reference-harness replication and context chart
results/                           every CSV and JSON behind the poster
```
