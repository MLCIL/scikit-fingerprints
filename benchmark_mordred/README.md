# Reproducing the RDKit UGM poster

Everything behind *Optimizing Mordred Descriptors Calculation for Scalable
Molecular Feature Extraction* lives on this branch: runners, result CSVs, figures,
methodology and the poster sources.

Everything lives in this one directory; nothing else in the repository is touched.

```
benchmark_mordred/
  bench_run.py          the measurement harness, all suites
  bench_analyze.py      charts and summary tables from the CSVs
  figures_final.py      the poster's charts 1-4
  run_all.sh            phase A + phase B, the full September run
  run_expansionrx.sh    the ADME scenario and the ring-rich set
  make_rings_dataset.py rebuilds the one input that is not committed
  dataset_masses.py     molecular masses for the size charts
  molpile/              corpus preparation, throughput, metrics
  muv_replication/      reference-harness replication, wrappers, context chart
  profil/               per-family timings and profiling
  docs_en/              methodology, one file per experiment
  docs_pl/              Polish working notes and background
  poster/               the poster as presented
```

**Results are not committed.** Everything here is what you need to *produce* them;
the CSVs from the September run live on the `mordred-benchmarks` branch, together
with the exact implementation they were measured against.

Start with **[docs_en/README.md](docs_en/README.md)** —
one file per experiment, stating the data, the preprocessing, what was excluded and
why, and what each measurement does not support.

---

## 0. Prerequisites

### This repository

```bash
git checkout mordred-benchmarks
uv sync                                              # or: pip install -e .
pip install -r benchmark_mordred/requirements-bench.txt     # pinned reference versions
```

The reference library's version is **part of the result**, not an environment
detail: this is a comparison between two implementations, so measuring against a
different mordred-community release will not reproduce these numbers.

Measurements were taken at commit `0bec710` on `bitai2` (AMD EPYC 7763,
64 physical cores / 128 threads, 503 GB RAM). Every CSV row records its
`branch@commit` and host, so results can always be traced to the code.

### External data — not vendored here

Three datasets are too large to commit (about 8 GB together) and belong to other
projects. Fetch them before running the experiments that need them.

| what | size | where from | needed by |
|---|---|---|---|
| **PeptideReactor** peptides | 88 MB | `git clone --depth 1 https://github.com/scikit-fingerprints/peptides_molecular_fingerprints_classification` | E3 |
| **MolPILE** 1 M subset | 6.1 GB with embeddings | arXiv:2509.18353 | E5 |
| **reference embedding benchmark** | 343 MB | `benchmarking_molecular_models_private` (private) | E6 |

MoleculeNet and ExpansionRx datasets download themselves through
`skfp.datasets` on first use.

### Paths the scripts expect

No path is hardcoded; every script takes its locations from the environment and
falls back to something sensible relative to the repository.

```bash
export BENCH_REPO="$PWD"                 # defaults to the script's own repo root
export PEPTIDE_REPO="$PWD/peptides_molecular_fingerprints_classification"
export PEPDIR="$PEPTIDE_REPO/data/PeptideReactor"
export MOLPILE_CSV="/path/to/molpile_1M.csv"
export MOLPILE_SCRATCH="/path/with/room"  # ~13 GB if embeddings are persisted
export BENCH_RINGS_CSV="$PWD/dataset_rings.csv"
```

### One input has to be rebuilt

The ring-rich set used in Phase B of the ExpansionRx scenario is not committed
(it is derived data, and the September file came from the full 222 M corpus):

```bash
python benchmark_mordred/make_rings_dataset.py --n 10000 --min-rings 4
```

This reproduces the construction rule — 10 000 molecules with at least four rings —
not the original file molecule for molecule. That is sufficient, because the
measurement characterises a class of molecules rather than specific ones.

### Always calibrate first

```bash
python benchmark_mordred/bench_run.py --suite calib --out results/calib.csv
```

A few minutes. It measures *this* machine and projects the budget for every chart,
with a verdict on whether it fits. **Only that projection is binding** — estimates
carried over from another machine are worthless.

---

## 1. Running the experiments

The protocol is two-phase. **Phase A** produces ratios, which survive contention,
so many processes may run concurrently (we capped at 32-45 threads). **Phase B**
produces absolute numbers and needs an idle machine; it is serialised behind an
`flock` lock.

### E1 — descriptor families

```bash
python benchmark_mordred/profil/isolated_families.py --dataset bace --n 1513 --repeats 3 \
    > results/tabela_rodziny_bace.txt
```

### E2, E3 — speedup per dataset and per molecule size (Phase A)

```bash
for d in freesolv esol sider clintox bace bbbp lipophilicity tox21 toxcast hiv muv; do
    python benchmark_mordred/bench_run.py --suite speedup --only $d --out results/res_speedup_$d.csv &
done
for L in 5 10 20 40 80 100 160; do
    python benchmark_mordred/bench_run.py --suite size --only synthpep_$L --out results/res_size_synthpep_$L.csv &
done
for P in ace_vaxinpad hiv_lpv hiv_nvp; do
    python benchmark_mordred/bench_run.py --suite size --only preactor_$P \
        --peptide-dir "$PEPDIR" --out results/res_size_preactor_$P.csv &
done
wait
bash benchmark_mordred/run_expansionrx.sh          # 10 ADME datasets + ring-rich set
python benchmark_mordred/dataset_masses.py         # molecular masses for the x-axis
```

`preactor_hiv_nvp` is the long pole — two molecules of ~2000 atoms, about 19
minutes each across both implementations. `--repeats 1` is legitimate there: the
cost is deterministic O(n³), so repetitions carry no information.

### E4 — parallel scaling (Phase B, idle machine)

```bash
python benchmark_mordred/bench_run.py --suite threads --impls ours    --out results/res_threads.csv
python benchmark_mordred/bench_run.py --suite threads --impls mordred --out results/res_threads.csv
```

Each implementation needs its own invocation; the suite refuses both at once, and
gives them different dataset sizes on purpose — see
[E4](docs_en/E4_parallel_scaling.md).

### E5 — MolPILE throughput

```bash
cd benchmark_mordred/molpile
python prepare_molpile.py 1000000      # filter; writes the rejected indices too
bash run_molpile.sh 1m                 # reference-repository protocol
python throughput_molpile.py --n-jobs 96 --batch-size 250
python metrics_molpile.py
```

### E6 — comparison against other embeddings

```bash
cd benchmark_mordred/muv_replication
python prepare_muv.py
bash run_both.sh defaults              # their configuration
bash run_both.sh pinned                # control, BLAS threads set to 1
python plot_poster_context.py --runtime-csv ~/benchmarking_muv_runtime.csv
```

---

## 2. Figures and the poster

```bash
python benchmark_mordred/figures_final.py          # charts 1-4 -> results/figures_finals/
cd poster && python poster.py               # the A0 PDF, fully vectorial
```

---

## 3. Things that will bite you

**Results are gitignored.** `.gitignore` drops `*.csv`, `*.png`, `*.pdf` and
`*.svg`. After twelve hours of compute it is easy to return with only the scripts.
Commit with `git add -f` and verify with `git show --stat HEAD`.

**Runs are resumable.** Restarting with the same `--out` skips combinations already
measured, and each row is flushed with `fsync()`, so Ctrl+C costs nothing.

**Truncate before filtering.** `peptide_reactor` reads an entire FASTA file;
filtering first would make `hiv_nvp` 733 molecules of ~2000 atoms instead of the
two actually measured.

**HIV and MUV look like they hang at startup.** The loader parses 10 000 molecules
before `--max-size` applies.

**Do not tune configuration on a subsample of MolPILE.** Twice a subsample ranked
options the opposite way to the full corpus.

**Phase B means idle.** Absolute numbers taken under load are not comparable with
anything, including our own earlier runs.
