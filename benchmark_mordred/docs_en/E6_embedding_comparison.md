# E6 — Throughput against other molecular embeddings

Poster element: the **Throughput vs other embeddings** chart.

## Question

Where does this implementation sit among the molecular representations people
actually use?

## Base comparison

An independent benchmark of **61 model configurations** on **ogbg-molmuv**,
measuring mean time to embed 10 000 molecules. We did not re-measure those models;
we added three Mordred bars **inside the same harness**, through its own wrapper
interface, so that the numbers are produced by the same timer and the same batching.

## Data and preprocessing

**ogbg-molmuv, all 93 087 molecules**, processed in nine full batches of 10 000.
The reported figure is the **mean time per 10 000-molecule batch**.

**Canonicalisation.** `prepare_muv.py` applies `Chem.CanonSmiles` to every SMILES,
reproducing the harness's `build_dataset`. This matters for parsing cost and for
list matching below.

### Exclusions in this pipeline — and what they actually do

The reference pipeline contains two filters. We reproduced both, then checked
whether they fire. **Neither does, on this dataset:**

| mechanism | where | effect on MUV |
|---|---|---|
| `illegal_smiles` list, 46 entries | `embed.py` → `filter_out_problematic_molecules` | **0 molecules removed** |
| `*` → `C` wildcard substitution | `src/common/types.py:269` | **0 molecules modified** |

The `illegal_smiles` list matches nothing in MUV, raw or canonicalised (verified:
MUV's SMILES are canonical, 500/500 sampled). `prepare_muv.py` writes all 93 087
molecules, the standard dataset size, so nothing is dropped upstream either.

This is worth knowing for two reasons. First, **no molecules were silently removed
or altered in this comparison** — a question that would otherwise be fair to ask.
Second, the 46 entries are organometallics — ferrocenes, aluminium and tungsten
complexes — and include the same iron complex that triggers the `LinAlgError` in
our own HIV runs. The list evidently targets a different dataset, most likely HIV,
and is inert here.

## The three Mordred bars

| bar | s / 10k | mol/s | provenance |
|---|---|---|---|
| Mordred (as published) | 187.37 | **53** | the reference repository's own measurement, BLAS unpinned |
| Mordred (BLAS pinned) | 25.07 | 399 | our control, same binary |
| Our Mordred | 4.02 | **2 485** | this work |

```bash
bash run_both.sh defaults    # replicates their configuration
bash run_both.sh pinned      # control with BLAS threads set to 1
```

## The caveat we state before being asked

**2485 / 53 = 47× is inflated.** The "53" comes from a run with mordred's BLAS
threads unset, which makes it ~7.3× slower than it should be, with **bit-identical
output**. Against the same binary properly configured, the honest ratio is
**6.2×**.

This is written into the plotting script itself, which prints it on every run:

> `WARNING: inflated, because the 'as published' measurement had BLAS threads unset.`
> `The corrected rate is 25.07 s / 399 mol/s, i.e. an honest ratio of 6.2×.`

**Why the "as published" bar stays.** All 60 other configurations were measured in
the same harness and the same configuration — that is, also unpinned. Removing the
handicap for mordred alone would make the comparison asymmetric with respect to
every other model on the chart. Showing both bars is the only symmetric option.

This investigation was not self-initiated: a supervisor did not believe the MUV
result and asked for the runners to be re-checked. The finding was that the error
lies in the reference repository, not in our wrapper — their published Mordred
figure is understated by ~7.3×, which is worth reporting to them.

Two supporting checks. Our implementation is insensitive to pinning (2534 vs 2485
mol/s) because `joblib`/`loky` pins workers itself, while mordred's raw
`multiprocessing.Pool` does not. And our own unpinned measurement (182.4 s)
reproduces their published one (187.4 s) to within 3%, confirming we replicated the
protocol rather than measuring a different machine.

## "Better throughput than all neural networks"

Verified across all 61 configurations. Faster than our 4.02 s are only classical
hashed fingerprints — ECFP 0.71 s, Topological Torsion 0.73 s, their count
variants 0.77 s, Atom Pair 1.27 s — a different category of method. The fastest
neural model is `mole` at 6.37 s; CheMeleon 9.88 s, MoLFormer-XL 12.35 s,
ChemBERTa ~16 s, COATI 116 s, ChemGPT 167 s.

**This is a claim about throughput only.** We make no claim that Mordred
descriptors are better representations than learned embeddings.

## Hardware caveat inherited from the base benchmark

CPU models were measured on 96 workers of the EPYC machine; GPU models on an
RTX 4000 Ada. Different models therefore ran on different hardware. This is a
property of the benchmark we extend, not a choice of ours, and it is why the poster
says "96 CPU workers **or** RTX 4000 Ada GPU".
