# E5 — Pretraining-scale throughput (MolPILE)

Poster element: the **At pretraining scale** figures — 4.5 min, 3 800 mol/s, 16 h.

## Question

Can this implementation realistically featurise a pretraining-scale corpus?

**This experiment measures our implementation only.** There is no mordred bar by
design: the point is scale, not a ratio. That was an explicit decision — *"MolPILE
only on our implementation, do not run the original"*.

## Data and preprocessing

**MolPILE** (arXiv:2509.18353), 222 M molecules. We took the first **1 000 000**.

**1. Canonicalisation.** `Chem.CanonSmiles` on every SMILES, matching what the
reference harness does in `build_dataset`. Without it the parsing cost — and any
SMILES-list matching — would differ from the harness we compare against.

**2. The uncomputable filter.** **3 181 molecules (0.320%) removed**, leaving
**996 819**. Rejected indices are recorded in `odrzucone_indeksy.csv`.

### Why those 3 181 were removed

They are molecules on which our implementation either raises or **never
terminates** — fullerenes, perfluorinated C60/C70 derivatives and graphene
fragments. The detour matrix requires the longest simple path, which is NP-hard;
on a 126-atom cage with 37 fused rings the search does not finish.

This was not a precaution but a response to a failure. An unfiltered preparation
run occupied 95 workers for **8 h 35 min** — 793 CPU-hours — and would never have
completed. `py-spy` showed every worker in the same stack, inside
`_walk_every_path`.

The original mordred has a timeout guard at that point: it computes for ~3 minutes
and returns a `timed out` placeholder for the affected descriptors, then continues.
**We lack that guard**, and we report this as a genuine deviation rather than a
data-cleaning step. The offending molecules cluster in contiguous blocks in the
corpus, so a single one would stall an entire 500-molecule batch.

**3. The filter mechanism.** A hard `SIGALRM` budget in `prepare_molpile.py`:
0.2 s per molecule for a whole batch, and on overrun the batch is retried molecule
by molecule with a 5 s limit. The alarm works inside a `loky` worker because the
task runs in that process's main thread.

## Protocol

```bash
python prepare_molpile.py 1000000
bash run_molpile.sh 1m                                       # reference-repo protocol
python throughput_molpile.py --n-jobs 96 --batch-size 250    # maximum
python metrics_molpile.py
```

## Two protocols, two numbers — do not mix them

| protocol | work split | throughput | 1 M molecules |
|---|---|---|---|
| reference-repository protocol | batches of 10 000 | 605 mol/s | 27.3 min |
| maximum | `batch_size=250` | **3 800 mol/s** | **4.4 min** |

The 6.3× difference is entirely the work split. `skfp` defaults to
`batch_size = len(data) // n_jobs`, i.e. **one task per worker**, so the makespan is
set by the slowest chunk and molecule cost has a heavy tail. At `batch_size=250`
there are ~4 000 tasks over 96 workers, ~40 each, enough to balance.

The poster quotes the maximum, and the question *"how fast is it with skfp's
defaults?"* has the answer 605 mol/s.

## Results

Three repetitions at 96 workers, `batch_size=250`:

| run | `seconds_descriptors` | mol/s |
|---|---|---|
| 1 | 261.68 | 3 809.2 |
| 2 | 262.54 | 3 796.8 |
| 3 | 262.78 | 3 793.4 |
| **mean** | **262.33** | **3 800 ± 8** |

At 64 workers: ~3 175 mol/s, so 96 workers add 19.7%. Single-thread ceiling,
measured on a random 3 000-molecule sample rather than the head of the file:
**73.6 mol/s**, giving 51.6× at 96 workers, or 78% efficiency against 64 physical
cores.

## What the headline number includes

`seconds_descriptors` is **descriptor computation alone**. It excludes SMILES
parsing (~21.8 s) and writing 12.9 GB of `float64` output (~33 s).

This matched a challenge raised during the work — *"not saving the embeddings
sounds suspicious"*. The reason it is excluded is that the reference harness, which
produced every one of the 61 model timings we compare against, also excludes it:
its timer wraps only `model.embed(batch)`, with persistence after the loop
(`src/embedding/embedding.py:155`). Excluding it is a **condition of
comparability**, not a flattering choice. The raw timings are published, so
`mol_per_s_end_to_end` (~3 509) and `mol_per_s_with_persist` (~3 131) can be read
off directly.

## The 222 M figure is an extrapolation

222e6 / 3800 = 58 421 s = **16.2 h**. We measured 1 M, a slice of the same corpus,
and assume cost homogeneity. That assumption is reasonable but unverified at full
scale.

Note that `metryki.txt` quotes **4.25 days** for 222 M — that is the same
extrapolation from the *605 mol/s* batch protocol. Both are correct for their own
protocol and must not be mixed.

## A methodological warning from this experiment

**Do not tune configuration on a subsample of this corpus.** Twice a subsample
pointed the opposite way to the full run: at 100 k molecules `batch_size=1` won
(2693 vs 2258 mol/s) but lost at 1 M (3581 vs 3800); and a 150 k sample suggested
64 and 96 workers were equivalent, while the full million showed ~20% in favour of
96.
