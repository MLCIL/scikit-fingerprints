# scikit-fingerprints — czym jest i dlaczego tam

## 1. Czym jest biblioteka

**scikit-fingerprints** to biblioteka do chemoinformatyki zgodna ze scikit-learn,
rozwijana w ML & Chemoinformatics Lab na Wydziale Informatyki AGH. Publikacja
źródłowa: **J. Adamczyk, P. Ludynia, *SoftwareX* 28, 101944 (2024)**. Licencja MIT,
czyli swobodne użycie akademickie i komercyjne.

Hasło z README: *„Go from SMILES to production-grade chemoinformatics ML pipelines
in a few lines of code."*

### Co zawiera

| moduł | zawartość |
|---|---|
| **fingerprints** | ponad 30 odcisków: ECFP, Avalon, MACCS, PubChem, Mordred, Atom Pair, Topological Torsion |
| **filters** | ponad 30 filtrów: Lipinski Rule of 5, PAINS, REOS, strukturalne i fizykochemiczne |
| **distances** | 14 miar podobieństwa i odległości: Tanimoto, Dice, MCS |
| **applicability_domain** | 11 metod oceny, czy model wolno stosować do nowej cząsteczki |
| **datasets** | MoleculeNet, Therapeutics Data Commons, MoleculeACE, LRGB, ExpansionRx — z gotowymi podziałami |
| **preprocessing, clustering, model_selection, metrics** | reszta pipeline'u |

W repozytorium jest **34 plików z fingerprintami** i 13 modułów pakietu.

### Idea przewodnia

Wszystko ma **jednolite API scikit-learn**: `fit`, `transform`,
`get_feature_names_out`, parametry `n_jobs`, `sparse`, `verbose`. Dzięki temu
dowolny odcisk wchodzi wprost do `Pipeline`, `FeatureUnion`, `GridSearchCV`,
`ColumnTransformer` i całej reszty ekosystemu.

To jest różnica wobec sytuacji sprzed biblioteki: każdy odcisk miał własne API,
własną konwencję zrównoleglenia i własny format wyjścia, a sklejanie ich w jeden
pipeline było pracą ręczną.

## 2. Dlaczego Mordred musiał trafić do tej biblioteki

### Argument pierwszy — spójność interfejsu

Mordred był jedną z najważniejszych rodzin cech w chemoinformatyce, a **jedyną
dużą, której w scikit-fingerprints brakowało w natywnej formie**. Użytkownik,
który chciał deskryptorów Mordreda obok ECFP, musiał wychodzić poza bibliotekę,
używać innego API (`Calculator(descriptors)(mol)`), innego formatu wyjścia
(obiekt `Result` albo DataFrame przez `calc.pandas()`) i innego mechanizmu
zrównoleglenia (`calc.map(mols, nproc=N)`).

Po przepisaniu wygląda to tak:

```python
from skfp.fingerprints import ECFPFingerprint, NewMordredFingerprint
from sklearn.pipeline import FeatureUnion

features = FeatureUnion([
    ("ecfp",    ECFPFingerprint(n_jobs=-1)),
    ("mordred", NewMordredFingerprint(n_jobs=-1)),
])
```

Dwie rodziny cech o zupełnie innym charakterze, jeden interfejs, jeden parametr
zrównoleglenia, jeden format wyjścia.

### Argument drugi — wydajność była realną barierą

To jest sedno postera. Biblioteka reklamuje się jako *„fast and efficient:
parallelized, sparse matrices support, C++ RDKit under the hood"*. Mordred w
oryginale tego nie spełniał: **0,1 s na cząsteczkę lekopodobną** i słabe
skalowanie przy tanich cząsteczkach (5,3× na 64 procesach).

Włączenie go jako cienkiego opakowania na oryginał oznaczałoby, że jedna pozycja
w bibliotece jest o dwa rzędy wielkości wolniejsza od reszty. Przy ECFP liczącym
14 tys. cząsteczek na sekundę i Mordredzie liczącym 53 to nie jest ta sama liga.

Po przepisaniu: **2 485 mol/s**, czyli czwarte miejsce w rankingu 29 metod,
przed wszystkimi modelami neuronowymi.

### Argument trzeci — kontrola nad kodem

Oryginalne repozytorium `mordred-descriptor/mordred` **nie jest aktywnie
rozwijane**. Utrzymanie przejął community fork (mordredcommunity, Jackson Burns).
Biblioteka, która buduje na nieutrzymywanej zależności, dziedziczy jej problemy:
brak reakcji na zmiany w RDKit, brak poprawek błędów, brak wsparcia dla nowych
wersji Pythona.

Mając własną implementację:

- poprawiamy błędy u siebie, w swoim tempie
- dopisujemy brakujące rodziny (kappa, information content)
- kontrolujemy zgodność z kolejnymi wersjami RDKit i NumPy
- nie ciągniemy zewnętrznej zależności do instalacji

### Argument czwarty — brak nowych zależności

Kluczowe ograniczenie projektowe: **czysty Python, RDKit i NumPy, bez rozszerzeń
kompilowanych**. scikit-fingerprints instaluje się przez `pip install` i działa
na każdej platformie.

Alternatywa w postaci portu C++ (osmordred) wymaga Boost, Eigen, LAPACK i Ninja
przy budowaniu, a obejmuje tylko podzbiór 0D/1D/2D. Dla biblioteki, której
obietnicą jest bezproblemowa instalacja, to byłby krok wstecz.

## 3. Co nasza implementacja dostaje od biblioteki za darmo

To warto wiedzieć, bo pokazuje, że nie pisaliśmy wszystkiego od zera.

- **Bazowy transformer** (`skfp/bases/base_fp_transformer.py`) — cała mechanika
  `fit`/`transform`, walidacja parametrów, obsługa `sparse`, integracja
  z `set_output(transform="pandas")`.
- **Zrównoleglenie** (`skfp/utils/parallel.py`) — `run_in_parallel` na joblib/loky.
  Nasz kod nie robi nic jawnego z procesami ani wątkami.
- **Walidacja wejścia** (`ensure_mols`) — akceptowanie SMILES, InChI i obiektów
  `Mol`, spójne komunikaty błędów z indeksem feralnej cząsteczki.
- **Obsługa konformerów** — `requires_conformers` i konwencja `conf_id` wspólna
  z innymi odciskami 3D w bibliotece.
- **Loadery zbiorów** — `skfp.datasets.moleculenet`, `skfp.datasets.expansionrx`.
  Wszystkie zbiory z naszego benchmarku ładowaliśmy właśnie przez nie.

Ciekawa konsekwencja uboczna: ponieważ zrównoleglenie idzie przez **joblib/loky**,
a nie przez surowy `multiprocessing.Pool`, **automatycznie unikamy pułapki BLAS**.
loky ustawia workerom `OMP_NUM_THREADS = cpu_count // n_jobs`; mordred z własną
pulą tego nie robi i dlatego w cudzym benchmarku wyszedł 7,3× wolniejszy, niż
jest naprawdę.

## 4. Co oddaliśmy bibliotece

- **1826 deskryptorów** w natywnym API, z `get_feature_names_out` zgodnym
  z konwencją nazw Mordreda.
- **Przyspieszenie 7–14×** względem oryginału na jednym wątku.
- **Znaleziony artefakt w domyślnym `batch_size`** (`skfp/utils/parallel.py`):
  `len(data) // n_jobs` daje jedno zadanie na workera, co przy niejednorodnym
  koszcie cząsteczki kosztuje sześciokrotność na MolPILE. To dotyczy **wszystkich**
  odcisków w bibliotece, nie tylko naszego.
- **Materiał benchmarkowy** — 254 pomiary, skrypty i protokół, które da się
  ponownie uruchomić dla innych odcisków.

## 5. Jak to opowiadać przy posterze

Gdy ktoś zapyta *„czemu nie zrobiliście z tego osobnego pakietu"*:

**PL:** Bo wartość jest w tym, że Mordred jest teraz **jednym z trzydziestu
odcisków o identycznym interfejsie**, a nie kolejną biblioteką do sklejenia.
Użytkownik pisze `FeatureUnion([ECFP, Mordred])` i to działa. Osobny pakiet
rozwiązywałby problem wydajności, ale nie problem spójności.

**EN:** Because the value is that Mordred is now **one of thirty fingerprints
with an identical interface**, not another library to glue in. You write
`FeatureUnion([ECFP, Mordred])` and it works. A separate package would solve the
performance problem but not the consistency problem.

Gdy ktoś zapyta *„czy to nie jest duplikowanie pracy"*:

**PL:** Oryginał pozostaje referencją i cytujemy go. Nasza wersja jest adaptacją
jego kodu na licencji BSD, formuła po formule. Różnica jest w architekturze
wykonania, nie w definicjach deskryptorów. To nie jest konkurencyjna definicja
Mordreda, tylko szybsza droga do tych samych liczb.

**EN:** The original remains the reference and we cite it. Our version is an
adaptation of its BSD-licensed code, formula by formula. The difference is in the
execution architecture, not in the descriptor definitions. This is not a competing
definition of Mordred, it is a faster path to the same numbers.
