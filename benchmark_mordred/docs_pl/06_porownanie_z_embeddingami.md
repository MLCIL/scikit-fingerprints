# 06. Throughput vs other embeddings (wykres slupkowy)

## Co jest na posterze

Slupki przepustowosci: nasz Mordred wsrod 28 modeli embeddingowych. Podpis:

> - better throughput than all neural networks
> - 2 485 mols/s against 53 for the original
> - MUV benchmark datasets, 96 CPU workers or RTX 4000 Ada GPU

## Runner

```bash
cd ~/mordred/scikit-fingerprints/benchmark_mordred/muv_replication
python prepare_muv.py                 # -> data/prepared/ogbg-molmuv.joblib
bash run_both.sh defaults             # proba 1: BEZ pinowania BLAS
bash run_both.sh pinned               # proba 2: Z pinowaniem BLAS
python plot_poster_context.py --runtime-csv ~/benchmarking_muv_runtime.csv
```

Wykres: `benchmark_mordred/muv_replication/plot_poster_context.py` ->
`5_context_models.png`. Nasz wrapper dla obcego harnessu:
`model_wrappers/skfp_mordred/wrapper.py`.

## Dane

| plik | co |
|---|---|
| `results/muv_replication/reference_all_models.csv` | 28 modeli z repo `benchmarking_molecular_models_private` |
| `results/muv_replication/proba1_defaults/runtime.csv` | mordred + my, BLAS **nie**pinowany |
| `results/muv_replication/proba2_blas_pinned/runtime.csv` | mordred + my, BLAS pinowany |
| `results/muv_replication/porownanie_prob.csv` | zestawienie obu prob |

Zbior: **ogbg-molmuv**, 93 087 czasteczek, mierzone w paczkach po 10 000
(9 pelnych paczek). Liczba na wykresie to **sredni czas na paczke 10 000**.

## Trzy slupki mordredowe - i dlaczego trzy

| slupek | czas / 10k | mol/s | skad |
|---|---|---|---|
| Mordred (as published) | 187.37 s | **53** | pomiar **repo referencyjnego**, BLAS niepinowany |
| Mordred (BLAS pinned) | 25.07 s | 399 | **nasza proba kontrolna**, ta sama binarka |
| Our Mordred | 4.02 s | **2 485** | nasza implementacja |

## NAJWAZNIEJSZE ZASTRZEZENIE

**Iloraz 2 485 / 53 = 47x jest zawyzony.** Liczba "53" pochodzi z pomiaru **as
published**, w ktorym mordred mial **nieustawione watki BLAS** i przez
nadsubskrypcje byl ~7.3x wolniejszy, niz powinien, przy bit-identycznym wyniku.

**Uczciwy iloraz wobec tej samej binarki to 2 485 / 399 = 6.2x.**

Zastrzezenie jest wpisane wprost w skrypt rysujacy wykres
(`plot_poster_context.py`, `main()`), ktory sam je drukuje na konsole:

> `UWAGA: zawyzony, bo pomiar 'as published' mial nieustawione watki BLAS.`
> `Poprawiona stawka to 25.07 s / 399 mol/s, czyli uczciwy iloraz 6.2x`

**Dlaczego mimo to na wykresie zostaje slupek "as published":** bo wszystkie
pozostale 26 modeli zmierzono w **tym samym harnessie i tej samej konfiguracji**,
czyli rowniez bez pinowania. Usuniecie tylko mordredowi tego handicapu zrobiloby
porownanie niesymetrycznym wobec reszty. Dlatego sa oba slupki - i dlatego trzeba
umiec to wytlumaczyc, jesli ktos zapyta o 47x.

To jest znalezisko metodologiczne warte zgloszenia autorom repo referencyjnego:
ich opublikowana liczba dla mordreda jest zanizona ~7.3x.

## Pelne dane z obu prob (`porownanie_prob.csv`)

| proba | BLAS | model | sredni czas / 10k | mol/s |
|---|---|---|---|---|
| 1 defaults | nie | mordred_2d | 182.43 s | 54.8 |
| 1 defaults | nie | skfp_mordred_2d | 3.95 s | 2 534.0 |
| 2 pinned | tak | mordred_2d | 25.07 s | 398.9 |
| 2 pinned | tak | skfp_mordred_2d | 4.02 s | 2 485.4 |

Dwie obserwacje:

**Nasza implementacja jest niewrazliwa na pinowanie** (2 534 vs 2 485 mol/s,
roznica w granicach szumu), bo `joblib`/`loky` sam pinuje workery
(`OMP_NUM_THREADS = cpu_count // n_jobs`). Mordred uzywa surowego
`multiprocessing.Pool`, ktory tego nie robi - stad jego 7.3x.

**Nasz wlasny pomiar mordreda niepinowanego (182.4 s) zgadza sie z opublikowanym
(187.4 s)** w granicach 3%. To potwierdza, ze odtworzylismy ich protokol wiernie, a
roznica nie bierze sie z innej maszyny.

## Sprzet

- CPU: **96 workerow** na bitai2 (AMD EPYC 7763)
- GPU dla modeli neuronowych: **RTX 4000 Ada**

Stad ostrozne sformulowanie na posterze: "96 CPU workers **or** RTX 4000 Ada GPU" -
rozne modele liczyly sie na roznym sprzecie, co jest wlasciwoscia odziedziczona po
repo referencyjnym, nie nasza decyzja.

## Pulapki

**"better throughput than all neural networks" to teza o przepustowosci, nie o
jakosci reprezentacji.** Poster nigdzie nie twierdzi, ze deskryptory Mordreda sa
lepsze predykcyjnie od CheMeleona czy COATI - tylko ze liczy sie je szybciej.

**Szybsze od nas sa proste fingerprinty** (ECFP ~0.71 s, TT ~0.73 s, AtomPair_count
~0.82 s na 10k) - to osobna kategoria i nie sa to siec neuronowe. Pozycja naszego
slupka w rankingu jest wypisywana przez skrypt (`nasz slupek: N z M od najszybszego`).
