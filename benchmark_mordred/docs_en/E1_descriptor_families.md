# E1 — Individual descriptor families

Poster element: the **Individual descriptors speedup** table.

## Question

Where does the speedup come from, family by family — and how much of it is *not*
attributable to shared precomputation?

## Data and preprocessing

**BACE** from MoleculeNet, **all 1513 molecules**, mean 34.1 atoms. No subsampling
and no `drop_uncomputable` pass: BACE contains none of the organometallics that
break our implementation, and all 1513 molecules compute on both sides.

Earlier revisions of this table used 100 BBBP molecules, then 200 MUV molecules.
The final version was recomputed on the whole of BACE at a supervisor's request,
so that the table rests on a complete standard dataset rather than a sample.

## Protocol

```bash
python benchmark_mordred/profil/isolated_families.py --dataset bace --n 1513 --repeats 3
```

Single thread, BLAS pinned, **best of 3** (not mean — we want the cleanest run,
since this is a cost comparison, not a distribution estimate).

Both sides compute a family **together with its prerequisites**, from scratch:

- mordred: `Calculator(module)` evaluates only that family's dependency subtree;
- ours: we construct exactly the objects `calculator.compute` would hand the family,
  and time them together with it.

## Why not a profiler — the central methodological point

We first tried `cProfile`, and discarded it. A profiler attributes only *self*
time. Our families would appear artificially thin, because shared preparation —
distance matrices, subgraph enumeration, atomic properties — lives in other
modules; mordred's families would appear thick, because it repeats that work inside
each one. The resulting ratio would describe **where the code lives**, not how much
it computes, and could not be defended.

The symmetric design above costs us something, and we accept it: prerequisites are
paid once *per family* here, whereas a real run pays them once *per molecule* for
all families together. **The table therefore excludes the benefit of shared
precomputation and is a lower bound.** That is what the poster's third bullet means.

The internal consistency check: no single family reaches the 7.5× the whole
fingerprint achieves on the same dataset. The gap *is* the sharing.

## Results

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

`Autocorrelation` alone accounts for 606 of the 1613 features. `Chi` is the most
expensive family in absolute terms on both sides.

## Exclusion: two families are omitted, and why

**KappaShapeIndex scores 0.30× and AdjacencyMatrix 0.93× under this protocol.**

This is an artefact of the protocol, not a regression. In isolation our side pays
for a complete `Subgraphs` enumeration and eigendecomposition in order to produce
three features; in a real run that cost is already paid by other families and
shared. Mordred, which computes those prerequisites inside each family anyway, is
not penalised the same way.

We omit them from the poster table because a 0.30× entry would be read as a
regression, which it is not. The omission is deliberate and we can justify it on
request — it is not a suppressed result, and the numbers are in the repository.

## What this table does not show

- it does not show the full speedup available — by construction it excludes sharing
- it does not decompose the *whole* fingerprint; 10 of ~39 descriptor modules are
  listed, chosen as the largest contributors
- family timings do not sum to the whole-fingerprint timing, for the same reason
