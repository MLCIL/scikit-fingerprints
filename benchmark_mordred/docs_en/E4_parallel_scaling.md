# E4 — Parallel scaling

Poster element: the **Parallel scaling** panel pair.

## Question

How does throughput scale from 1 to 64 workers, on drug-like molecules and on
peptides?

## Data and preprocessing

**HIV** (MoleculeNet) for the drug-like panel and **synthpep_20** (20-residue
synthetic peptides, 168 atoms) for the peptide panel.

### Exclusions

**HIV: three organometallics removed by `drop_uncomputable`**, so the full set
appears as **41 124** rather than 41 127. See `00_common_setup.md` for why they are
removed from both implementations rather than only ours — and for the fact that all
three now compute correctly on the current branch, so this exclusion reflects the
state of the code at measurement time, not a standing limitation.

### The two implementations receive different dataset sizes, on purpose

```python
THREADS_COUNTS = {
    "ours":    {"hiv": 50_000, "synthpep_20": 3_200},   # 50k = the whole of HIV
    "mordred": {"hiv":  6_000, "synthpep_20":   320},
}
```

With *equal* sizes our workers finish faster than process startup takes. On 320
peptides at 64 workers mordred showed 67% efficiency and ours 19% — the chart would
have suggested that **mordred scales better than we do**, which is an artefact of
the dataset being too small for our implementation to stay busy. With sizes tuned,
ours reaches 49%, which is what this hardware allows. Throughput is a per-molecule
quantity, so the curves remain comparable across different n.

This is the most common question asked about this chart and it has a direct answer.

## Protocol

**Phase B**, idle machine, `flock`-serialised. Worker counts 1, 2, 4, 8, 16, 32, 64.
Two repetitions (the single-worker point alone is half the sweep's cost, and
run-to-run spread on an idle machine is below 1%). Each implementation requires a
separate invocation; the suite refuses to run both at once.

```bash
python benchmark_mordred/bench_run.py --suite threads --impls ours    --out results/res_threads.csv
python benchmark_mordred/bench_run.py --suite threads --impls mordred --out results/res_threads.csv
```

## Results

| workers | HIV ours | HIV mordred | peptides ours | peptides mordred |
|---|---|---|---|---|
| 1 | 67.5 | 8.0 | 10.2 | 0.9 |
| 8 | 478.1 | 27.0 | 70.3 | 6.5 |
| 32 | 1220.0 | 37.0 | 172.3 | 23.2 |
| 64 | **2158.1** | 42.0 | **279.6** | 33.0 |
| **scaling** | **32.0×** | 5.2× | **27.4×** | 35.1× |

## Things we state rather than hide

**On peptides the original scales better than we do** — 35.1× against 27.4×. The
cause is ours: `skfp` splits work statically (one task per worker), so the makespan
is set by the slowest worker, while mordred dispatches per molecule and balances
better. In absolute terms it remains 8.5× slower at 64 workers, but the scaling
claim belongs to them on this panel.

**Our 32× at 64 workers is 50% efficiency, not 100%.** Part is the static split;
part is physics, since the CPU drops from 3.5 to 2.45 GHz under full load.
"Near-linear scaling" describes the shape of the curve on log axes.

## The 51× gap on the HIV panel — read it correctly

At 64 workers the HIV panel shows 2158 against 42 mol/s. **That ratio is a property
of this dataset, not of parallelism**, and we investigated it rather than quoting it.

Four candidate explanations were tested and rejected:

| hypothesis | test | result |
|---|---|---|
| our BLAS not pinned | 137 threads, load 7.7 | rejected |
| HIV molecules intrinsically costlier | single-thread rates | rejected: 0.0885 vs 0.0832 s/mol |
| process state at fork | clean process vs our runner | rejected: 143.6 / 143.4 / 145.0 s |
| heavy **size** tail | cut 2% of molecules >60 atoms (23% of cost) | 41.0 → 44.2 mol/s — only 8% |

The size-based cut failed because it used the wrong variable. Measuring the **cost
distribution directly** settles it:

| | HIV | MUV |
|---|---|---|
| median s/mol | 0.0910 | 0.0850 |
| mean s/mol | 0.1440 | 0.0895 |
| maximum | **6.152 s** | 0.223 s |
| share of total cost in the top 1% | **18.0%** | 2.2% |
| max atoms / rings | 114 / 16 | 36 / 6 |

The *typical* molecule costs the same in both datasets — medians differ by 7%.
Only the tail differs. MUV is a curated, size-bounded set; HIV is raw screening
data. Cost here tracks ring entanglement rather than atom count, which is exactly
why a filter on atom count did not catch it.

Consequence: because mordred dispatches per molecule, its HIV throughput depends on
whether the few pathological molecules land in the sample. Re-drawing 6 000
molecules moves it between **34.8 and 83.1 mol/s** at identical mean molecular size.
Our side computes the whole dataset and is therefore stable; mordred's subsample is
unbiased in expectation but noisy.

**The figures the poster quotes — 32× and 27× — are our own scaling**, measured on
one sample at both worker counts, and are unaffected by this. If someone measures
the gap off the chart instead, the honest statement is that the gap is
sample-dependent, and that on MUV under the same conditions it would be ~6×.

In hindsight HIV was a poor choice for the drug-like panel: it is the only dataset
in the study where the reference implementation's measurement is unstable.
