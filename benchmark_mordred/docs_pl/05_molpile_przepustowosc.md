# 05. At pretraining scale (MolPILE)

## Co jest na posterze

Blok liczbowy bez wykresu:

> **4.5 min** - 1M mols, 96 workers
> **3 800 mol/s**
> **16 h** - 222M mols in MolPILE
> - our Mordred works well for large-scale computation
> - e.g. enables pretraining CheMeleon on MolPILE

## Runner

```bash
cd ~/mordred/scikit-fingerprints/benchmark_mordred/molpile
python prepare_molpile.py 1000000          # filtr + data/prepared/molpile-1m.joblib
bash run_molpile.sh 1m                     # przebieg w protokole repo referencyjnego
python throughput_molpile.py --n-jobs 96 --batch-size 250   # pomiar maksymalny
python metrics_molpile.py                  # -> metryki.txt
```

Dane: `results/molpile/1m/`, opis plikow w `results/molpile/1m/README.md`.

## Zbior

**MolPILE**, 222 mln czasteczek (arXiv:2509.18353). Pobrano pierwszy **1 000 000**,
po filtrze zostalo **996 819**.

**Odfiltrowano 3 181 czasteczek (0.320%)**, lista w `odrzucone_indeksy.csv`. Powod:
czasteczki, na ktorych nasza implementacja rzuca wyjatkiem albo **nie konczy
liczenia** - fullereny i fragmenty grafenowe, gdzie `DetourMatrix` szuka najdluzszej
sciezki prostej (problem NP-trudny). Oryginalny mordred ma tam bezpiecznik czasowy
i zwraca placeholder `timed out` po ~3 min; my go nie mamy.

Filtr: `prepare_molpile.py`, twardy limit przez `SIGALRM` - 0.2 s na czasteczke dla
calej paczki, po przekroczeniu sztuka po sztuce z limitem 5 s.

## Dwa rozne pomiary - to zrodlo najwiekszego nieporozumienia

| protokol | podzial pracy | przepustowosc | 1 mln |
|---|---|---|---|
| repo referencyjnego (`runtime.csv`) | paczki po 10 000 | **605 mol/s** | 27.3 min |
| maksymalny (`throughput_*.json`) | `batch_size=250` | **3 800 mol/s** | **4.5 min** |

Roznica **6.3x** bierze sie wylacznie z podzialu pracy. `skfp/utils/parallel.py`
ustawia domyslnie `batch_size = len(data) // n_jobs`, czyli **jedno zadanie na
workera**; czas calosci wyznacza wtedy najwolniejszy worker, a koszt czasteczki ma
ciezki ogon. Przy `batch_size=250` wychodzi ~4000 zadan na 96 workerow, czyli ~40
na workera - dosc do wyrownania.

**Poster cytuje liczbe maksymalna (3 800 mol/s).** Jesli ktos pyta "a ile w
domyslnej konfiguracji skfp", odpowiedz to 605 mol/s i trzeba umiec wyjasnic podzial.

## Wyniki, 3 powtorzenia

### 96 workerow, `batch_size=250`

| proba | `seconds_descriptors` | mol/s |
|---|---|---|
| 1 | 261.68 s | 3 809.2 |
| 2 | 262.54 s | 3 796.8 |
| 3 | 262.78 s | 3 793.4 |
| **srednia** | **262.33 s** | **3 800 +/- 8** |

### 64 workery, dla porownania

| proba | mol/s |
|---|---|
| 1 | 3 153.1 |
| 2 | 3 230.9 |
| 3 | 3 142.1 |
| **srednia** | **3 175** |

96 workerow daje **19.7% wiecej** niz 64 - hyperthreading cos tu wnosi, wbrew
wczesniejszemu pomiarowi na podprobce 150 tys., ktory sugerowal brak roznicy.

## Co obejmuje liczba glowna

`seconds_descriptors` = **samo liczenie deskryptorow**. Nie obejmuje:

- `seconds_parse` ~21.8 s - parsowanie SMILES do `Mol` (rownolegle)
- `seconds_persist` ~33 s - zapis 12.9 GB na dysk w `float64`

**Dlaczego bez zapisu:** harness, ktorym zmierzono wszystkie 28 modeli w
[06](06_porownanie_z_embeddingami.md), tez go nie obejmuje - stoper otacza wylacznie
`model.embed(batch)`, a zapis idzie po petli (`src/embedding/embedding.py:155`).
Zeby liczba byla porownywalna, musi byc liczona tak samo.

Warianty sa w JSON-ach, wiec kazda liczbe da sie odtworzyc:
`mol_per_s_end_to_end` ~3 509, `mol_per_s_with_persist` ~3 131.

## Ekstrapolacja do 222 mln

```
222 000 000 / 3 800 mol/s = 58 421 s = 16.2 h
```

**To jest ekstrapolacja, nie pomiar.** Zmierzono 1 mln. Zalozenie: korpus jest
jednorodny kosztowo, co jest uzasadnione, bo 1 mln to losowy wycinek tego samego
zbioru, ale nie zostalo zweryfikowane na pelnych 222 mln.

`metryki.txt` podaje **4.25 dnia** dla 222 mln - to przeliczenie z **605 mol/s**
(protokol paczkowy), nie z 3 800. Obie liczby sa poprawne dla swoich protokolow i
nie wolno ich mylic.

## Pulapki

**Nie decydowac o konfiguracji na podprobce.** Dwa razy podprobka wskazala
odwrotnie niz calosc: (1) przy 100 tys. `batch_size=1` wygrywal (2693 vs 2258
mol/s), a przy 1 mln przegrywal (3581 vs 3800); (2) proba 150 tys. pokazywala, ze
64 i 96 workerow daja to samo, a pelny milion dal ~20% roznicy. Dobor w
`batch_size_sweep.txt`.

**Sufit jednowatkowy: 73.6 mol/s** (losowa proba 3000, nie poczatek pliku).
Przyspieszenie 3800 / 73.6 = **51.6x** przy 96 workerach, czyli 78% efektywnosci
wzgledem 64 rdzeni fizycznych.
