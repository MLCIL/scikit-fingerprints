# Benchmark methodology — RDKit UGM poster

Companion to *Optimizing Mordred Descriptors Calculation for Scalable Molecular
Feature Extraction*. One file per experiment, each specifying the data,
the preprocessing, **what was excluded and why**, the protocol, and what the
measurement does not support.

| file | poster element | headline |
|---|---|---|
| [00_common_setup](00_common_setup.md) | — | hardware, preprocessing pipeline, exclusions shared by all |
| [E1](E1_descriptor_families.md) | Individual descriptors speedup | 3.0 - 9.3× per family |
| [E2](E2_qsar_qspr_speedup.md) | QSAR/QSPR datasets | 7.5× / 7.6× / 11.2× |
| [E3](E3_molecule_size_scaling.md) | Molecule size scaling | 5.6× at 140 Da → 13.9× at 11 kDa |
| [E4](E4_parallel_scaling.md) | Parallel scaling | 32× drug-like, 27× peptides |
| [E5](E5_molpile_throughput.md) | At pretraining scale | 3 800 mol/s, 16 h for 222 M |
| [E6](E6_embedding_comparison.md) | Throughput vs other embeddings | 2 485 mol/s |
| [00_overview](00_overview.md) | — | all six in one narrative, for reading end to end |

## Molecules excluded, in one table

| experiment | removed | of | why |
|---|---|---|---|
| E2, E3 (HIV subsample) | 1 | 10 000 | organometallic, `LinAlgError` in our code |
| E4 (HIV full) | 3 | 41 127 | two `OverflowError`, one `LinAlgError`, all organometallics |
| E2, E3 (peptides) | 2 datasets | 10 | above 100 residues — proteins, not peptides |
| E1 | 2 families | 12 | protocol artefact, not a regression; disclosed in E1 |
| E5 (MolPILE) | 3 181 | 1 000 000 | never terminate — fullerenes, NP-hard longest path |
| E6 (MUV) | **0** | 93 087 | both reference filters verified inert on this dataset |

Crash-related exclusions are removed from **both** implementations, so that both
measure the same work. At measurement time the original mordred survived those
molecules by substituting an error placeholder while we did not, and we report that
as a real deviation rather than hiding it behind a filter.

**The three HIV molecules have since been fixed** and now compute in agreement with
the reference implementation; the exclusion describes the code as it stood when the
published numbers were taken. See `00_common_setup.md`.

## The three caveats most likely to be challenged

1. **47× on the embeddings chart is inflated** — the reference figure was measured
   with BLAS threads unset. The honest ratio against the same binary is **6.2×**.
   See E6.
2. **The 51× gap on the HIV parallel panel is sample-dependent** — mordred's
   throughput there moves between 34.8 and 83.1 mol/s depending on which 6 000
   molecules are drawn, because HIV has a heavy cost tail. See E4.
3. **On peptides the original scales better than we do** (35.1× vs 27.4×), because
   our work split is static. See E4.
