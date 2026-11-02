# Wyniki i metodologia — wszystkie liczby z postera

## 1. Środowisko pomiarowe

| | |
|---|---|
| maszyna | **bitai2**, AMD EPYC 7763 |
| rdzenie | **64 fizyczne / 128 wątków sprzętowych** |
| pamięć | 503 GB |
| Python | 3.13 |
| RDKit | 2026.03 |
| mordredcommunity | 2.0.7 |
| NumPy | 2.5 |

**Rozróżnienie rdzeni fizycznych i wątków jest ważne.** Wykresy skalowalności
idą do 64, czyli do liczby rdzeni fizycznych. MolPILE liczyliśmy na 96 workerach,
czyli z wykorzystaniem hyperthreadingu — i zmierzyliśmy, że to daje 19,7% więcej
niż 64 workery (3 800 wobec 3 175 mol/s).

## 2. Protokół pomiarowy

Punkty, które promotor wymienił jako wymagania, i jak są spełnione:

- **Zapis do CSV natychmiast po każdym pomiarze.** Przerwany przebieg wznawia się
  bez utraty danych.
- **Analiza osobnym skryptem.** `bench_run.py` mierzy, `bench_analyze.py` i
  `figures_final.py` rysują. Pomiar nie wie nic o wykresach.
- **Trzy powtórzenia na punkt**, uśredniane. **Wyjątek: drabinka wątków ma dwa** —
  ze względu na budżet czasu (Faza B liczyła się 4 godziny).
- **Jednostka: 10 000 cząsteczek.**
- **Wszystko jednowątkowo**, poza zestawem `threads`. To była twarda umowa
  metodologiczna: dla danego eksperymentu jeden wątek ma nasza implementacja
  i jeden mordred.
- **BLAS pinowany do jednego wątku przed importem numpy.**
- **Blokada `flock`** na `~/.mordred-bench.lock`, żeby pomiary bezwzględne nie
  nakładały się na siebie.

### Podział na fazy

| faza | co | dlaczego |
|---|---|---|
| **A** | 11 zbiorów MoleculeNet + 10 punktów drabinki peptydowej, **równolegle** | mierzy **stosunki**, więc kontencja skraca się po obu stronach |
| **B** | `threads/ours`, `threads/mordred`, `reference`, **po kolei na wyłączność** | mierzy **wartości bezwzględne**, wymaga wyłączności maszyny |

Przebieg główny: 2026-09-04, **11:46:37 → 17:51:17, czyli 6 h 05 min**.

### Inwentarz pomiarów

| zestaw | zbiorów | pomiarów |
|---|---:|---:|
| speedup (MoleculeNet) | 11 | 66 |
| expansionrx (ADME + pierścieniowy) | 11 | 66 |
| size (drabinka rozmiaru) | 10 | 60 |
| threads (skalowalność) | 2 | 56 |
| reference (10 tys. HIV) | 1 | 6 |
| **razem** | **32** | **254, zero błędów** |

## 3. Wykres 1 — speedup na rodzinach zbiorów

Trzy słupki, każdy to średnia po zbiorach rodziny, jeden wątek:

| rodzina | zbiorów | średnia | zakres |
|---|---:|---:|---|
| **Peptides** (do 100 reszt) | 8 | **11,2×** | 7,6–13,9× |
| **ExpansionRx-OpenADMET** | 10 | 7,6× | 7,4–8,2× |
| **MoleculeNet** | 11 | **7,5×** | 5,6–9,0× |

Zbiór pierścieniowy (9 998 cząsteczek, 8,2×) został z tego wykresu usunięty na
prośbę promotora.

Pojedyncze zbiory MoleculeNet, gdyby ktoś dopytał:
HIV 9,04× · SIDER 8,97× · ClinTox 7,99× · BACE 7,87× · BBBP 7,74× ·
Tox21 7,22× · MUV 7,19× · ToxCast 7,17× · Lipophilicity 7,05× · ESOL 6,83× ·
FreeSolv 5,63×.

**Mediana wynosi 7,22×, średnia 7,52×.** Promotor wybrał średnią.

Po wszystkich 32 zmierzonych zbiorach: średnia **9,04×**, mediana 7,81×,
zakres 5,6–18,0×.

## 4. Wykres 2_3 — zależność od rozmiaru cząsteczki

Dwa panele, wspólna oś X w **masie cząsteczkowej (Da)**:

- lewy: **czas na cząsteczkę**, obie implementacje, skala log-log
- prawy: **speedup**, punkty plus dopasowanie

**20 zbiorów, od 138 Da (FreeSolv) do 11 877 Da (peptyd 100-resztowy).**
Peptydy ucięte na 100 resztach na prośbę promotora — powyżej to już białka.
Zbiory ADME usunięte z tego wykresu, bo dziesięć punktów o masie 378–425 Da
zagęszczało lewą stronę, nie wnosząc nic do zależności od rozmiaru.

### Liczby

- zakres przyspieszenia: **5,6× przy 138 Da → 13,9× przy 10,9 kDa**
- dopasowanie liniowe w log(masa): **+1,59× na e-krotność**, **R² = 0,916**
- dopasowanie liniowe w samej masie: R² = 0,707 — wyraźnie gorsze

**Uwaga na sformułowanie „prawie liniowy".** Jest prawdziwe **tylko względem
logarytmu masy**. Masa rośnie 86-krotnie, a przyspieszenie tylko 2,5-krotnie.
Ktoś, kto przeczyta „liniowo" i podstawi masę, oczekiwałby 86-krotnego wzrostu.

### Skalowanie asymptotyczne — czego NIE twierdzić

Krzywe kosztu są **równoległe**. Wykładniki dopasowane na zbiorach powyżej 100
atomów: mordred ~n^2,50, nasze ~n^2,36. Poniżej 40 atomów: n^1,03 i n^0,80.

**Nie twierdzimy lepszej złożoności asymptotycznej.** Przewaga to stały czynnik,
który sam rośnie z rozmiarem, a nie inny wykładnik. Koszt przy dużych
cząsteczkach jest zdominowany przez dekompozycje spektralne O(n³), których obie
implementacje muszą dokonać.

### Skala kosztu przy dużych cząsteczkach

Jedna cząsteczka 1982-atomowa (`hiv_nvp`, 241 reszt, 28 kDa) to **~46 minut**
liczenia dla obu implementacji razem: 611 s nasze + 2167 s mordred. Dlatego
drabinka peptydowa ma malejącą liczebność: 200 → 100 → 50 → 20 → 10 → 5 → 5.

## 5. Wykres 4_pair — skalowalność wielordzeniowa

Dwa panele: **Drug-like** (HIV, 25 atomów) i **Peptides** (20-mery, 168 atomów).
joblib/loky, 1 → 64 procesy robocze, średnia z 2 przebiegów.

| | HIV: mordred | HIV: nasze | peptydy: mordred | peptydy: nasze |
|---|---:|---:|---:|---:|
| 1 worker | 8,0 mol/s | 67,5 mol/s | 0,9 mol/s | 10,2 mol/s |
| 64 workery | 42,0 mol/s | **2 158 mol/s** | 33,0 mol/s | 279,6 mol/s |
| skalowanie | **5,3×** | **32×** | **36,7×** | 27,4× |
| przewaga | 8,4× → **51,4×** | | 10,8× → **8,5×** | |

### Dwie rzeczy do wyjaśnienia przy posterze

**Dlaczego na HIV przewaga rośnie tak drastycznie.** Mordred dyspozycjonuje
zadanie na każdą cząsteczkę przez proxy `Manager()`. Przy cząsteczce liczonej
15 ms narzut komunikacji dominuje i krzywa się wypłaszcza.

**Dlaczego na peptydach przewaga maleje.** Tam mordred skaluje się **lepiej**
(36,7× wobec naszych 27,4×), bo jego narzut amortyzuje się przy cząsteczce
liczonej sekundę, a nasz statyczny podział pracy zaczyna przeszkadzać. To jest
uczciwe i warto to powiedzieć samemu, zanim ktoś zauważy.

**Różne rozmiary próbek.** Nasze liczyliśmy na 41 124 cząsteczkach HIV, mordred
na 5 999; peptydy odpowiednio 3 200 i 320. Tak ustawia to runner, żeby mordred
nie liczył doby. **Porównanie pozostaje poprawne**, bo oś Y to przepustowość,
czyli tempo, a nie czas całkowity.

## 6. Wykres 5 — kontekst 28 modeli embeddingowych

Replika protokołu z wewnętrznego repozytorium benchmarkowego, zbiór
**`ogbg-molmuv`** z Open Graph Benchmark, paczki po 10 000 cząsteczek, 9 pełnych
paczek, 96 workerów CPU albo 1 GPU.

**Nasz słupek jest czwarty z 29.** Szybsze tylko: ECFP (14 109 mol/s),
Topological Torsion (13 743), AtomPair (12 167). My: **2 485 mol/s**.

Najszybsze modele neuronowe: MolE 1 570 (GPU), CLAMP 1 078 (GPU),
Chemeleon 1 012 (GPU), MoLFormer 810 (GPU). **Wyprzedzamy wszystkie**, w tym
liczone na GPU, licząc na CPU.

Original Mordred jak w cudzym repozytorium: **53 mol/s**.

### Uwaga krytyczna do tego wykresu

Słupek `Original Mordred` to pomiar **z niepinowanym BLAS-em**. Po pinowaniu
ta sama binarka daje **399 mol/s**. Z wykresu ktoś odczyta iloraz **46,6×**,
a uczciwa liczba to **6,2×**.

Podpis pod wykresem mówi „2 485 mols/s against 399 for the original" — czyli
podaje liczbę poprawioną, której na wykresie nie ma. **Jeśli ktoś to zauważy,
odpowiedź brzmi:** słupek pokazuje pomiar opublikowany, a 399 to ta sama
implementacja po ustawieniu wątków BLAS; różnica jest artefaktem pomiaru, nie
naszą zasługą.

## 7. Sekcja „At pretraining scale" — MolPILE

Zbiór: `scikit-fingerprints/MolPILE` z HuggingFace, podzbiór miliona cząsteczek.
Cały MolPILE ma **222 mln** cząsteczek (arXiv:2509.18353). Średnia cząsteczka:
23,9 atomu.

**Original Mordred nie jest tu liczony.** To pomiar wewnętrzny: ile nasza
implementacja wyciąga na korpusie pretreningowym.

| konfiguracja | prób | przepustowość | milion |
|---|---:|---:|---:|
| **96 workerów, `batch_size=250`** | 3 | **3 800 ± 8 mol/s** | **4,37 min** |
| 96 workerów, `batch_size=1` | 2 | 3 581 ± 11 mol/s | 4,64 min |
| 64 workery, `batch_size=250` | 3 | 3 175 ± 48 mol/s | 5,23 min |
| protokół, paczki 10 tys. | 1 | 605 mol/s | 27,3 min |
| jeden wątek | — | 73,6 mol/s | 3,76 h |

**Rozrzut 0,2% przy trzech próbach.** Dla kontrastu przebieg w cudzym protokole
miał odchylenie **79%** między paczkami.

Pochodne liczby:

- **6,1 mln wartości deskryptorów na sekundę** (3 800 × 1613)
- cały MolPILE, 222 mln cząsteczek: **16,2 h** na jednym węźle
- rozmiar wyniku: 1 mln × 1613 to **12,9 GB** w float64, 6,4 GB w float32
- parsowanie SMILES dokłada 22 s do miliona, zapis na dysk 28 s

**Przyspieszenie 51,6×** wobec jednego wątku (3 800 / 73,6).

### Powiązanie z CheMeleonem

CheMeleon jest pretrenowany na przewidywaniu **wszystkich 1613 deskryptorów
Mordreda** dla **1 miliona cząsteczek z PubChem**. My liczymy taki zbiór etykiet
w 4,4 minuty.

Porównanie kosztu na cząsteczkę, obie liczby z tego samego benchmarku:
**my 4,0 s na 10 tys. na CPU, CheMeleon 9,9 s na 10 tys. na GPU.**

## 8. Odfiltrowane cząsteczki

**3 181 z 1 000 000, czyli 0,318%** nie weszło do pomiaru MolPILE — nasza
implementacja albo rzuca na nich wyjątkiem, albo nie kończy liczenia.

Odsetek rośnie z „dzikością" zbioru:

| zbiór | odsetek nieobliczalnych |
|---|---|
| HIV | 0,007% |
| zbiór pierścieniowy | 0,02% |
| **MolPILE** | **0,318%** |

To trzeba powiedzieć samemu przy prezentacji, a nie czekać, aż ktoś zapyta.
Szczegóły w pliku o pułapkach.
