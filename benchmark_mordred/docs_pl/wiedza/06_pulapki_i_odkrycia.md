# Pułapki metodologiczne i odkrycia po drodze

Ten plik zawiera rzeczy, które **kosztowały nas czas albo zmieniły wnioski**.
Część z nich to najciekawszy materiał na rozmowę przy posterze, bo pokazuje, że
pomiar był robiony z głową, a nie „odpalone i zapisane".

## 1. Pułapka BLAS — artefakt 7,3×

### Co się stało

Odtwarzaliśmy protokół z cudzego repozytorium benchmarkowego, żeby postawić nasz
słupek obok 28 modeli. Wyszło, że jesteśmy **46× szybsi od Mordreda**. Promotor
nie uwierzył i kazał sprawdzić, czy warunki są identyczne.

### Przyczyna

W `model_wrappers/mordred/init.sh` tamtego repozytorium **nie ma ustawionych**
zmiennych `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`.

Mordred liczy dekompozycje spektralne przez NumPy, czyli przez OpenBLAS. Bez tych
zmiennych OpenBLAS rozpina pulę wątków na wszystkie rdzenie **w każdym z 96
procesów roboczych**. Procesy biją się o rdzenie i program chodzi wolniej, niż
gdyby był jednowątkowy.

### Pomiar

| | czas / 10 tys. | mol/s |
|---|---:|---:|
| bez pinowania | **182,4 s** | 53 |
| z pinowaniem | **25,1 s** | 399 |

**Artefakt 7,3×.** Wyjście obu przebiegów **bit w bit identyczne** — to kwestia
wydajności, nie poprawności. Po poprawce nasza przewaga to **6,2×**, nie 46×.

### Trzy rzeczy do zapamiętania

1. Zmienne muszą być ustawione **przed importem numpy**. OpenBLAS czyta je przy
   ładowaniu biblioteki; ustawienie po imporcie nie działa.
2. **joblib/loky pinuje automatycznie** (`OMP_NUM_THREADS = cpu_count // n_jobs`).
   Surowy `multiprocessing.Pool`, którego używa mordred, nie.
3. Nasza implementacja jest odporna: 3,95 s bez pinowania wobec 4,02 s z pinowaniem,
   czyli w granicach szumu.

### Jak to opowiadać

To jest **wkład metodologiczny**, nie atak na kogoś. Sformułowanie: *„porównując
implementacje CPU łatwo zmierzyć nie to, co się chce — bez pinowania wątków BLAS
baseline wychodzi 7,3× wolniejszy, niż jest naprawdę, przy identycznym wyniku
numerycznym"*. **Nie nazywaj repozytorium ani autora publicznie.**

## 2. Fullereny wieszają DetourMatrix

### Co się stało

Przygotowanie zbioru MolPILE chodziło **8 godzin 35 minut na 95 workerach**
(łącznie 793 godziny CPU) i nie skończyłoby się nigdy. Zrzut stosu (`py-spy`)
wszystkich workerów pokazał ten sam ślad:

```
_walk_every_path              (descriptors/detour_matrix.py:388)
_longest_paths_in_ring_system (detour_matrix.py:336)
_get_ring_system_correction   (detour_matrix.py:307)
_get_detour_matrix            (detour_matrix.py:125)
```

### Winowajcy

**Fullereny i fragmenty grafenowe**: 126 atomów, **37 skondensowanych pierścieni**,
perfluorowane pochodne C60/C70. Przykład z korpusu (indeks 62250):

```
FC(F)(F)C12C3=C4c5c6c7c8c9c%10c%11c%12c%13c%14c%15c%16c%17c%18c%19c%20c%21c%22...
```

Najdłuższa ścieżka prosta w takim układzie to problem **NP-trudny**. Nasza
optymalizacja ogranicza przeszukiwanie do układów pierścieni — ale gdy cała
cząsteczka jest jednym wielkim układem pierścieni, to nic nie daje.

W korpusie leżą **blokami** (indeksy 62250–62257 pod rząd), więc jedna sztuka
blokowała całą paczkę 500 cząsteczek, a ~95 takich paczek zajęło wszystkie
workery na stałe.

### Oryginał ma tam bezpiecznik, my nie

Sprawdzone: Original Mordred na tych samych cząsteczkach liczy **173–194 s**
i zwraca `timed out (SpAbs_Dt/.../DetourMatrixCache())` jako wartość deskryptora,
po czym idzie dalej.

**To jest trzeci zgłaszalny błąd naszej implementacji** — nie wyjątek, tylko brak
zakończenia, i brak guardu, który oryginał ma.

### Obejście

`prepare_molpile.py` ma twardy limit czasu przez `SIGALRM`: 0,2 s na cząsteczkę
dla całej paczki, a po przekroczeniu przejście sztuka po sztuce z limitem 5 s.
Działa w workerze loky, bo zadanie idzie tam w wątku głównym procesu.

Odfiltrowane: **3 181 z 1 000 000 = 0,318%**.

## 3. Trzy klasy cząsteczek, na których się wywracamy

To trzeba znać, bo ktoś może zapytać o odporność.

| błąd | przyczyna | przykład |
|---|---|---|
| `OverflowError` | nieskończona walencja metalu, `int(inf)` w `_implicit_hydrogen_count` | kompleksy Pt, Fe |
| `LinAlgError` | LAPACK nie zbiega przy dekompozycji | organometaliki |
| **brak zakończenia** | `DetourMatrix` na fullerenach | perfluorowane C60/C70 |

**Mordred na wszystkich trzech nie przerywa** — wstawia placeholder błędu dla
feralnego deskryptora i liczy dalej. To jest realne odstępstwo od oryginału.

Odsetki: HIV **0,007%** (3 z 41 127), zbiór pierścieniowy 0,02%, MolPILE 0,318%.

Dodatkowo, niezależnie od tego: `cpsa.py` produkuje NaN na wielu cząsteczkach HIV,
gdy suma ładunków wynosi zero (`q_max / np.sum(charges)`). To nie blokuje liczenia.

## 4. Artefakt domyślnego `batch_size` w scikit-fingerprints

### Co odkryliśmy

`skfp/utils/parallel.py` ustawia:

```python
batch_size = max(len(data) // n_jobs, 1)
```

czyli **każdy worker dostaje dokładnie jedno zadanie**. Czas całości wyznacza
wtedy najwolniejszy worker, a koszt cząsteczki ma ciężki ogon.

### Skala efektu

Na MolPILE, ten sam kod, zbiór i maszyna:

| podział | zadań na workera | przepustowość | rozrzut między paczkami |
|---|---:|---:|---:|
| paczki 10 tys. (protokół) | 1 | **605 mol/s** | **79%** |
| `batch_size=250` | ~40 | **3 800 mol/s** | 0,2% |

**Sześciokrotna różnica** i stabilność lepsza o dwa rzędy wielkości.

### Konsekwencje dla naszych własnych wyników

- **MUV liczyło się w tej patologicznej konfiguracji** (10 tys. cząsteczek na 96
  workerów to 104 na workera). Nasze liczby na MUV są więc **zaniżone**, czyli
  błąd działał przeciwko nam. Zapas szacuję na 1,5–1,7×.
- **Wykres skalowalności** ma ten sam artefakt, ale łagodniejszy — przy 64 wątkach
  wychodzi 642 cząsteczki na workera. Spadek efektywności z 89% (8 wątków) do 51%
  (64) to **częściowo artefakt domyślnego podziału**, nie granica implementacji.
- **Porównania z mordredem pozostają ważne**, bo 9,0× jednowątkowo pochodzi
  z pomiaru bez zrównoleglenia.

### To jest zgłoszalne do skfp

Domyślne `batch_size = len(data) // n_jobs` jest złe dla zbiorów o niejednorodnym
koszcie. Alternatywa: przekazać `batch_size` do `Parallel` i zostawić `"auto"`,
czyli adaptacyjne szeregowanie joblib.

## 5. Konfiguracji NIE wolno dobierać na podpróbce

To **dwukrotnie** nas złapało i jest chyba najbardziej praktyczną lekcją.

### Przypadek pierwszy — liczba workerów

Na próbce 150 tys. cząsteczek 64, 96 i 128 workerów dawały **to samo**
(1 780 / 1 794 / 1 797 mol/s). Wniosek brzmiał: hyperthreading nic nie wnosi.

Pełny milion: **64 workery 3 175 mol/s, 96 workerów 3 800 mol/s** — różnica
**19,7%**, dziesięciokrotnie większa niż rozrzut.

### Przypadek drugi — `batch_size`

| `batch_size` | 100 tys. cząsteczek | pełny 1 mln |
|---|---:|---:|
| 1 | **2 693 mol/s** (najlepszy) | 3 581 mol/s |
| 250 | 2 258 mol/s | **3 800 mol/s** (najlepszy) |
| domyślny | 1 978 mol/s | 605 mol/s |

**Ranking jest odwrotny.** Przy 100 tys. koszt księgowania zadań jest mały
i wygrywa lepsze równoważenie; przy milionie rośnie dziesięciokrotnie, a 250
daje i tak ~40 zadań na workera, czyli dość do wyrównania.

### Dlaczego `batch_size=1` nie jest „wyłączeniem batchowania"

`run_in_parallel` woła `Parallel(n_jobs=...)` **bez** `batch_size`, a joblib ma
tam domyślnie `batch_size="auto"`. Podanie 1 nie wyłącza grupowania — oddaje
decyzję adaptacyjnemu szeregowaniu joblib. Ale tworzy też **milion obiektów
`delayed()`**, a księgowanie ich wysyłki kosztuje w procesie rodzica.

### Trzeci przypadek — reprezentatywność wycinka

Pierwsze 150 tys. cząsteczek MolPILE jest **dwukrotnie droższe** niż średnia
korpusu (1 790 wobec 3 800 mol/s), bo zawiera rejon z fullerenami. Losowa próbka
3 000 cząsteczek dała 13,6 ms/cząsteczkę, zgodne ze średnią całości.

## 6. Pułapki narzędziowe przy mierzeniu obciążenia

Drobne, ale realnie kosztowały czas i mogą się przydać jako anegdota.

### `ps -eo pcpu` podaje średnią z życia procesu

Nie zużycie chwilowe. Świeżo uruchomiony proces, który żyje 20 ms i cały ten
czas liczy, raportuje ~100%, a na maszynie wielordzeniowej nawet 600%.

Suma `ps` po użytkowniku potrafi więc pokazać „5 zajętych rdzeni", gdy maszyna
jest bezczynna — bo **`ps` liczy sam siebie**. Zmierzone: `ps(600%, żyje 00:00)`.

Wiarygodne: `/proc/loadavg`, `top -bn2` (druga próbka), albo filtr odrzucający
procesy młodsze niż kilka sekund.

### `pgrep -f` dopasowuje własną linię poleceń

`pgrep -f "bench_run.py"` znajdzie sam siebie, bo wzorzec jest częścią linii
poleceń procesu `pgrep`. Stąd sztuczka z nawiasem: `pgrep -f "[b]ench_run.py"`.

### `kill -9` na rodzicu zostawia sieroty

Ubicie procesu-rodzica puli joblib zostawia workery z `PPID=1`. Zostało nam
w ten sposób **96 osieroconych procesów trzymających 22,3 GB** przez półtorej
godziny. Poprawnie: najpierw `pkill -9 -P <PID>`, potem `kill -9 <PID>`.

### `set_ylim` przed `set_yscale("log")`

Matplotlib: jeśli ustawisz granice osi, gdy jest jeszcze liniowa, dolna granica
wynosi 0. Po przejściu na skalę logarytmiczną zero rozciąga zakres na kilkanaście
dekad i wykres spłaszcza się do kreski. Granice ustawiaj **po** zmianie skali.

## 7. Rzeczy, których NIE zweryfikowaliśmy

Uczciwa lista, żeby nie dać się złapać:

- **Zgodność numeryczna z oryginałem na dużej próbce.** Twierdzimy wierność na
  podstawie tego, że przepisywaliśmy z kodu mordred-community formuła po formule.
  Nie mamy liczby typu „X% cech zgadza się co do wartości".
- **Porównanie z osmordredem.** Wiemy z ich README, że to port C++ obejmujący
  podzbiór 0D/1D/2D, wymagający Boost/Eigen/LAPACK. **Nie mierzyliśmy ani ich
  szybkości, ani zgodności numerycznej.**
- **Czy mordred dzieli pracę podobnie w cudzym runnerze MUV.** Używa
  `multiprocessing.Pool` z własnym schematem; jeśli dzieli lepiej niż my, nasze
  6,2× równolegle jest zaniżone bardziej niż jego wynik.
- **Poprawka błędu LogEE** — jest w notatkach projektu, ale nie weryfikowałem jej
  osobiście w kodzie.
- **Deskryptory 3D** — zaimplementowane (pełne 1826), ale niezbenchmarkowane.
