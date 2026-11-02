# MolPILE: przepustowosc naszej implementacji na skale korpusu pretreningowego

**Original Mordred tu NIE jest liczony.** To pomiar wewnetrzny: ile nasza
implementacja wyciaga na duzym korpusie. Porownania z oryginalem sa gdzie indziej
(`../bench_run.py` i `../muv_replication/`).

Zrodlo danych: `scikit-fingerprints/MolPILE` z HuggingFace, plik
`molpile_subset_1M.parquet`, raz skonwertowany do CSV. Caly MolPILE ma 222 mln
czasteczek (arXiv 2509.18353).

## Jak uruchomic

```bash
# 1. przygotowanie zbioru (filtruje czasteczki nieobliczalne, ~18 min na 1 mln)
python prepare_molpile.py 1000000

# 2a. przepustowosc maksymalna - liczba na poster
python throughput_molpile.py --n-jobs 96 --batch-size 250

# 2b. albo w protokole repo benchmarkowego (paczki 10 tys.), dla porownywalnosci
bash run_molpile.sh 1m
python metrics_molpile.py <sciezka>/runtime.csv
```

Skrypty maja **zaszyte sciezki absolutne** do `/home/dmarecik/mordred/...` i
zakladaja srodowisko z `bench_muv_replication` (harness `measure_embedding.py`
z repo benchmarkowego). Przy przenoszeniu na inna maszyne trzeba je poprawic.

## Co zmierzone (2026-09-05/06, bitai2, 96 workerow)

| konfiguracja | prob | przepustowosc | milion czasteczek |
|---|---|---|---|
| `batch_size=250` | 3 | **3800 +/- 8 mol/s** | 4.37 min |
| `batch_size=1` | 2 | 3581 +/- 11 mol/s | 4.64 min |
| paczki 10 tys. (protokol) | 1 | 605 mol/s | 27.3 min |

## Dwie pulapki, na ktorych mozna stracic dzien

**Fullereny wieszaja `DetourMatrix`.** MolPILE zawiera perfluorowane pochodne
C60/C70 (126 atomow, 37 skondensowanych pierscieni). Najdluzsza sciezka w takim
ukladzie to problem NP-trudny i liczenie nie konczy sie nigdy - 95 workerow stalo
8.5 h. Dlatego `prepare_molpile.py` ma twardy limit czasu przez `SIGALRM`
(0.2 s na czasteczke dla paczki, po przekroczeniu sztuka po sztuce z limitem 5 s).
Odfiltrowane 3181 z 1 000 000, czyli **0.318%**; indeksy laduja w
`data/prepared/<nazwa>-odrzucone.csv`.

**Konfiguracji NIE dobieraj na podprobce.** Ranking `batch_size` na 100 tys.
czasteczek jest ODWROTNY niz na pelnym milionie (tam wygrywa 1, tu 250). To samo
zdarzylo sie przy liczbie workerow: proba 150 tys. mowila, ze 64 i 96 daje to
samo, a pelny przebieg dal 15% roznicy. Koszt ksiegowania zadan i rozklad kosztow
czasteczek skaluja sie inaczej niz sama praca.
