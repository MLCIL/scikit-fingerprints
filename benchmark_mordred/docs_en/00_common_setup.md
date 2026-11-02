# Common setup, preprocessing and exclusions

Applies to every experiment. Each experiment file states only what differs.

## Hardware and software

| | |
|---|---|
| machine | `bitai2`: AMD EPYC 7763, 64 physical cores / 128 threads, 503 GB RAM |
| clock behaviour | drops ~3.5 GHz → ~2.45 GHz under full 64-core load |
| our implementation | `skfp.fingerprints.NewMordredFingerprint`, branch `mordred-benchmarks`, commit `0bec710` |
| reference | **mordredcommunity 2.0.7** (`JacksonBurns/mordred-community`), imported as `mordred` |
| RDKit | 2026.03.5 |

Moriwaki's original `mordred` 1.2.0 is unmaintained and does not run on current
RDKit or Python, so mordred-community is the only meaningful reference. Every CSV
row carries `branch@commit` and hostname.

## Preprocessing pipeline

Applied identically to both implementations, in this order:

**1. Load.** MoleculeNet and ExpansionRx through `skfp.datasets`; PeptideReactor
peptides parsed from FASTA with `Chem.MolFromSequence`; synthetic peptides
generated from the 20 standard amino acids with a seeded RNG.

**2. Truncate to the target size, before any filtering.** This order matters.
`peptide_reactor` reads the whole FASTA file, so filtering first would have made
the `hiv_nvp` set 733 molecules of ~2000 atoms instead of the two we actually
intend to measure — hours of wasted compute.

**3. Parse SMILES to RDKit `Mol`, outside the timer.** We measure descriptor
computation, not SMILES parsing. Molecules RDKit cannot parse are dropped here; on
the curated datasets used for the poster this is zero or near-zero.

**4. Random subsample where capped**, seed 0. **Both implementations receive the
same molecules** — the sample is drawn once and passed to both.

**5. `drop_uncomputable`** — see below.

**6. Measure.**

## Exclusion 1: molecules that crash our implementation

`drop_uncomputable` (`bench_run.py:318`) removes molecules on which our
implementation raises. In HIV there are **three out of 41 127**:

| HIV index | atoms | error |
|---|---|---|
| 14052 | 35 | `OverflowError: cannot convert float infinity to integer` |
| 16559 | 38 | same |
| 35278 | 30 | `LinAlgError: Eigenvalues did not converge` |

All three are organometallics — platinum and iron complexes. In the first two the
metal's valence evaluates to infinity, so an `int()` conversion overflows; the
third fails to converge in LAPACK.

**Why they are removed from *both* sides rather than only ours.** The original
mordred does not abort on these: it substitutes an error placeholder for the
offending descriptor and continues. Leaving them in would mean the two
implementations were computing different work on the same nominal dataset. We
remove them from both so that both measure the same thing.

**Why this is disclosed rather than quiet.** At the time of measurement it was a
genuine deviation from the original — our implementation was less robust here — and
we report it as such. The molecule count after filtering is written to the CSV, so
the exclusion is visible in the data: HIV appears as **9 999** in the
10 000-molecule subsamples and **41 124** in the full set.

**Status today: fixed.** All three molecules compute on the current development
branch and agree with the reference implementation. Re-checked directly:

| HIV index | result | agreement with mordred |
|---|---|---|
| 14052 | 1613 features, 284 NaN | complete |
| 16559 | 1613 features, 217 NaN | complete |
| 35278 | 1613 features, 228 NaN | two ETA features differ |

The `OverflowError` cases were resolved by correcting implicit-hydrogen counting for
metals; the `LinAlgError` case by adding dative bonds to the bond-order table, whose
absence had put NaN into the off-diagonal of the Burden matrix and made all twelve
stacked eigendecompositions fail at once. A residual difference remains on 35278
(`ETA_epsilon_4` 0.5239 vs 0.5301, `ETA_dEpsilon_B` 0.00622 vs 0).

The filter stays in the benchmark code because **the published measurements were
taken with the code as it stood**, and removing it now would make the recorded
molecule counts irreproducible.

One of the three falls inside the seed-0 10 000-molecule subsample, so without this
filter the HIV bar would simply have been missing from two charts.

**Related, not excluded:** `cpsa.py` produces NaN on many HIV molecules when the
Gasteiger charges sum to zero. It does not abort, so it does not affect the
benchmark, and those molecules stay in.

## Exclusion 2: peptides above 100 residues

All size-dependent charts cut peptides at **840 atoms (100 residues)**. Above that
these are proteins; most definitions cut "peptide" at 50 residues. Two measured
points — `synthpep_160` (1332 atoms) and `preactor_hiv_nvp` (1994 atoms) — sit in
the CSVs but outside the charts. Without the cut, the largest point on a chart
labelled "peptides" would be a 241-residue protein.

## Exclusion 3: MolPILE molecules that never terminate

Specific to the MolPILE experiment; see `E5_molpile_throughput.md`.

## Measurement protocol

**Two phases.** Phase A yields *ratios*, which survive machine contention because
both implementations suffer equally; many processes run concurrently, capped at
32-45 threads. Phase B yields *absolute* numbers and runs on an idle machine behind
an `flock` lock.

**BLAS pinning.** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` are set to `1` **before NumPy is
imported** (`bench_run.py:64-71`); OpenBLAS reads them at load time. Without this
the comparison is not symmetric: our spectral decompositions go through OpenBLAS
and would claim every core, while the original mordred is largely pure Python and
would claim one. In parallel runs the asymmetry reverses and would favour us,
because `joblib`/`loky` pins workers automatically while mordred's raw
`multiprocessing.Pool` does not — unpinned, the original runs **7.3× slower with
bit-identical output**.

**Repetitions.** Three by default, two for thread sweeps. The two implementations
are interleaved within each round so both see comparable machine conditions.

**Resumability.** Re-running with the same `--out` skips combinations already
measured; each row is flushed with `fsync()`, so an interrupted run loses nothing.

## Scope

All poster numbers concern the **1613 two-dimensional descriptors**. 3D descriptors
are deliberately excluded: they require a conformer, and conformer generation is a
separate problem that would confound the comparison rather than inform it.
