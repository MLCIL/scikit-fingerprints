# Poster artefact

`poster_final.pdf` is the poster as presented, kept here as the reference for what
the numbers in `../docs_en/` describe.

The September rendering script is **not** included: it read result CSVs that are
deliberately not committed to this branch, so it would not run. The measurement
scripts that lived next to it have been moved to `../profil/`:

| script | what it measures |
|---|---|
| `count_nodes.py` | size of mordred's dependency graph — 1944 nodes, 4480 `_calculate_one` calls per molecule, against 39 module calls in ours |
| `family_bench.py` | earlier per-family protocol, superseded by `isolated_families.py` |
| `profile_families.py` | `cProfile` per source file; kept because it documents the approach we rejected and why |
