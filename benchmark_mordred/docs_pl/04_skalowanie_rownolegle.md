# 04. Parallel scaling (para wykresow)

## Co jest na posterze

Dwa panele: przepustowosc wobec liczby workerow, osobno dla czasteczek
lekopodobnych i peptydow. Podpis:

> - near-linear scaling, on both drug-like molecules and peptides
> - 32x from one to 64 workers on drug-like, 27x on peptides
> - HIV dataset (MoleculeNet) and 20-mer peptides

## Runner

**Faza B - maszyna MUSI byc pusta.** To pomiary bezwzgledne (mol/s), nie ilorazy.

```bash
cd ~/mordred/scikit-fingerprints
python benchmark_mordred/bench_run.py --suite threads --impls ours    --out results/res_threads.csv
python benchmark_mordred/bench_run.py --suite threads --impls mordred --out results/res_threads.csv
```

Zestaw `threads` **wymaga osobnego wywolania na implementacje** i sam odmowi, jesli
dostanie obie naraz.

Wykres: `benchmark_mordred/figures_final.py`, `chart4_pair` -> `4_parallel_scaling_pair.png`
(warianty `_explained` i `_gap_only` w `results/figures_finals_v2/`).

## Konfiguracja

| | |
|---|---|
| liczby workerow | 1, 2, 4, 8, 16, 32, 64 (`THREAD_COUNTS`) |
| powtorzenia | 2 (`THREADS_REPEATS`) |
| zbiory | `hiv` (26 atomow), `synthpep_20` (168 atomow) |
| BLAS | przypiety do 1 watku na workera |

## Rozne liczebnosci na implementacje - to NIE jest blad

```python
THREADS_COUNTS = {
    "ours":    {"hiv": 50_000, "synthpep_20": 3_200},
    "mordred": {"hiv":  6_000, "synthpep_20":   320},
}
```

Przy **rownej** liczbie czasteczek nasze workery koncza prace szybciej, niz trwa
start procesow: na 320 peptydach i 64 workerach mordred wyciagal 67% efektywnosci,
a nasza implementacja 19% - i wykres sugerowalby, ze **mordred skaluje sie lepiej**.
Po dobraniu liczebnosci ta sama implementacja daje 49%, czyli tyle, ile pozwala
sprzet. Przepustowosc jest wielkoscia **na czasteczke**, wiec krzywe pozostaja
porownywalne mimo roznych n.

To jest najczestsze pytanie do tego wykresu i trzeba umiec na nie odpowiedziec.

## Wyniki (`results/res_threads.csv`)

### HIV, 26 atomow (lekopodobne)

| workerow | nasze [mol/s] | mordred [mol/s] |
|---|---|---|
| 1 | 67.5 | 8.0 |
| 2 | 131.3 | 13.2 |
| 4 | 252.6 | 20.0 |
| 8 | 478.1 | 27.0 |
| 16 | 641.3 | 33.0 |
| 32 | 1 220.0 | 37.0 |
| **64** | **2 158.1** | **42.0** |
| **skalowanie** | **32.0x** | **5.2x** |

### synthpep_20, 168 atomow (peptydy)

| workerow | nasze [mol/s] | mordred [mol/s] |
|---|---|---|
| 1 | 10.2 | 0.9 |
| 2 | 20.2 | 1.8 |
| 4 | 38.9 | 3.4 |
| 8 | 70.3 | 6.5 |
| 16 | 97.6 | 12.8 |
| 32 | 172.3 | 23.2 |
| **64** | **279.6** | **33.0** |
| **skalowanie** | **27.4x** | **35.1x** |

## Pulapki - czytac przed obrona

**Na peptydach mordred skaluje sie LEPIEJ od nas (35.1x vs 27.4x).** To jest na
wykresie i nie da sie tego ukryc. Przyczyna po naszej stronie: `skfp` dzieli prace
**statycznie** (`skfp/utils/parallel.py`: `batch_size = len(data) // n_jobs`, czyli
jedno zadanie na workera), wiec czas calosci wyznacza najwolniejszy worker. Mordred
dysponuje per czasteczke, wiec rownowazy lepiej. Mimo to jest w wartosciach
bezwzglednych **8.5x wolniejszy** przy 64 workerach (33.0 vs 279.6 mol/s).

**NIE uzywac ilorazu 51x z HIV jako "przewagi w zrownolegleniu".** Przy 64 workerach
na HIV wychodzi 2158 / 42 = **51x** i jest to poprawnie zmierzone oraz powtarzalne,
ale opisuje **interakcje mordreda z tanimi czasteczkami**, a nie jego zdolnosc do
zrownoleglania. Przyczyna ustalona: `mordred/_base/parallel.py:36-40` - `MolPool.submit`
robi `pool.apply_async` **na kazda czasteczke osobno**, a kalkulator przekazuje przez
proxy `Manager()`. Kazda czasteczka to komunikacja z procesem-zarzadca, ktory
serializuje caly ruch. Dlatego mordred skaluje sie zle **tylko na tanich
czasteczkach**, gdzie narzut dyspozycji dominuje.

Liczby do obrony: **9.0x jednowatkowo, 6.3x rownolegle** (na MUV przy tej samej
konfiguracji mordred osiaga 35% efektywnosci).

**Nasze 32x z 64 workerow to 50% efektywnosci, nie 100%.** Czesciowo to artefakt
statycznego podzialu (wyzej), czesciowo fizyka: EPYC 7763 przy wszystkich 64
rdzeniach zbija takt z ~3.5 GHz do ~2.45 GHz, czyli samo to zjada ~30%.
"Near-linear scaling" na posterze jest wiec opisem ksztaltu krzywej w skali log,
nie twierdzeniem o 100% efektywnosci.

## DOPISANE 2026-11-01: przepustowosc mordreda na HIV jest niestabilna

Pytanie "skoro na parallel wychodzi 51x, a na wykresie embeddingow 6.2x, to czy
BLAS byl ogarniety" doprowadzilo do pomiaru, ktorego wczesniej nie bylo.

**BLAS byl przypiety w obu eksperymentach** (`bench_run.py:64-71`, przed importem
numpy, jednakowo dla obu implementacji). Roznica bierze sie skadinad.

### Pomiar kontrolny: ten sam `calc.map`, `nproc=64`, jeden proces

| zbior | sr. atomow | mordred mol/s |
|---|---|---|
| MUV, 10 000 | 24.2 | **381.9** |
| HIV, 6 000 | 25.5 | **79.2** |

Czyli roznica **jest wlasciwoscia zbioru**, nie harnessu ani pinowania - oba
eksperymenty wolaja mordreda identycznie (`calc.map(mols, nproc=...)`:
`bench_run.py:371` oraz wrapper referencyjny `model_wrappers/mordred/wrapper.py:129`).

### Ale HIV sam w sobie daje niestabilny wynik

Ta sama konfiguracja, tylko inne losowanie 6 000 czasteczek z 41 127:

| ziarno | sr. atomow | mordred mol/s |
|---|---|---|
| 0 | 25.5 | **34.8** |
| 1 | 25.6 | **83.1** |

**Wahanie 2.4x przy identycznej sredniej wielkosci czasteczki.** Zapisane w CSV
42.0 mol/s jest powtarzalne (powtorzenia roznia sie <1%), ale powtarzalne **dla tej
jednej probki** - to nie to samo, co stabilne.

Mechanizm: mordred dysponuje zadania **per czasteczke** przez proxy `Manager()`
(`mordred/_base/parallel.py:36-40`), a HIV ma ciezki ogon kosztowy. Przy
serializowanej dyspozycji makespan wyznacza kilka najdrozszych czasteczek, wiec to,
czy trafia do probki, zmienia wynik dwukrotnie.

**Korekta wczesniejszego wniosku.** W notatkach bylo zapisane, ze hipoteze "ciezki
ogon HIV" **odrzucono** (odciecie 2% czasteczek >60 atomow dalo 41.0 -> 44.2 mol/s).
Tamten test ciał po **liczbie atomow**, a koszt nie idzie tu za liczba atomow, tylko
najpewniej za liczba pierscieni i sciezek. Test byl zle wycelowany i jego wniosek
nie obowiazuje.

### Co to znaczy dla postera

**Cytowane liczby 32x i 27x sa odporne** - to skalowanie *naszej* implementacji,
mierzone na tej samej probce przy 1 i 64 workerach, wiec wplyw doboru probki sie
skraca.

**Krucha jest krzywa mordreda na panelu HIV**, a wiec wizualna przepasc miedzy
krzywymi. Przy ziarnie 1 odstep przy 64 workerach wyniosłby ~26x zamiast 51x.
Poster tej liczby nie cytuje, wiec nic nie wymaga poprawki - ale gdyby ktos zmierzyl
odstep z wykresu i zapytal, odpowiedz brzmi: **ten odstep zalezy od probki**.

### Co konkretnie znaczy "inny zbior": ogon rozkladu kosztu

Zmierzone jednowatkowo, po 400 losowych czasteczek (ziarno 0), mordred:

| | HIV | MUV |
|---|---|---|
| mediana s/mol | 0.0910 | 0.0850 |
| srednia s/mol | 0.1440 | 0.0895 |
| p95 | 0.326 | 0.143 |
| p99 | 0.828 | 0.184 |
| **maksimum** | **6.152** | **0.223** |
| udzial najdrozszego 1% w koszcie | **18.0%** | 2.2% |
| udzial najdrozszych 5% | **31.6%** | 9.4% |
| max atomow / pierscieni | 114 / 16 | 36 / 6 |

**Typowa czasteczka kosztuje w obu zbiorach tyle samo** - mediany roznia sie o 7%,
a srednie wielkosci to 27.0 i 24.1 atomu. Cala roznica siedzi w ogonie: najdrozsza
czasteczka HIV liczy sie **28x dluzej** niz najdrozsza w MUV.

Powod jest w naturze zbiorow. **MUV** (Maximum Unbiased Validation) jest **kurowany**
i odgornie ograniczony - w probce maksimum 36 atomow i 6 pierscieni. **HIV** to surowe
dane przesiewowe, bez takiego ciecia, z czasteczkami do 114 atomow i 16 pierscieni.

To domyka lancuch przyczynowy: mordred dysponuje zadania per czasteczke, wiec przy
takim ogonie o wyniku decyduje, czy kilka najdrozszych sztuk trafi do wylosowanych
6 000 - stad zmierzone wahanie 2.4x miedzy ziarnami. W MUV ogona nie ma, pomiar jest
stabilny i skalowanie wychodzi dobrze.

### Jak rozne liczebnosci lacza sie z ogonem (dopisane po pytaniu)

Zmierzone liczebnosci w zestawie `threads`:

| zbior | nasze | mordred | wall przy 64w |
|---|---|---|---|
| HIV | **41 124** (caly zbior) | **5 999** | 19.1 s vs 143.0 s |
| synthpep_20 | 3 200 | 320 | 11.5 s vs 9.7 s |

Dobor roznych liczebnosci byl **konieczny** (patrz wyzej: bez niego wykres
sugerowalby, ze mordred skaluje sie lepiej). Ale w polaczeniu z ogonem HIV daje
skutek uboczny, ktorego nie bylo widac, dopoki nie zmierzono rozkladu kosztu:

- **nasza strona liczy CALY HIV** - zawiera wszystkie drogie czasteczki z
  pewnoscia, wiec pomiar jest stabilny
- **mordred liczy losowe ~15% zbioru** - drogie czasteczki trafiaja do probki albo
  nie, stad zmierzone wahanie 34.8 - 83.1 mol/s

**To nie jest obciazenie systematyczne.** Losowa podprobka ma te sama wartosc
oczekiwana kosztu na czasteczke co pelny zbior, a przepustowosc jest wielkoscia na
czasteczke, wiec porownanie pozostaje nieobciazone. Problemem jest wylacznie
**wariancja** po stronie mordreda - i zapisane 42.0 mol/s trafilo blisko
"pechowego" konca rozkladu.

Wniosek praktyczny: gdyby powtarzac ten wykres, HIV jest zlym wyborem na panel
"drug-like". MUV daje ten sam typ czasteczek bez ogona, wiec obie krzywe bylyby
stabilne - kosztem mniej efektownej przepasci (~6x zamiast 51x przy 64 workerach).
