# Nasza implementacja — co i jak przepisaliśmy

## 1. Gdzie to żyje

Pakiet `skfp/fingerprints/_new_mordred/` wewnątrz **scikit-fingerprints**,
wystawiony jako transformer `NewMordredFingerprint`. Struktura:

```
_new_mordred/
├── calculator.py            (344 linie)  — jedna funkcja compute() na cząsteczkę
├── descriptors/             (43 moduły)  — rodziny deskryptorów
└── utils/
    ├── feature_names.py     (1840 linii) — nazwy i kolejność 1826 cech
    ├── atomic_properties.py  (387)       — tablice właściwości atomów i wiązań
    ├── subgraphs.py          (367)       — enumeracja ESU podgrafów
    ├── matrix_attributes.py  (184)       — atrybuty spektralne macierzy
    ├── graph_matrix.py       (157)       — macierze sąsiedztwa i odległości
    ├── periodic_table.py     (105)       — tablice właściwości pierwiastków
    ├── mol_preprocess.py      (93)       — standaryzacja cząsteczki
    └── sasa.py, ragged.py, molecular_properties.py
```

API jest zgodne ze scikit-learn:

```python
from skfp.fingerprints import NewMordredFingerprint
fp = NewMordredFingerprint(use_3D=False, n_jobs=96, batch_size=250)
X = fp.transform(mols)          # (n, 1613) float32
```

## 2. Zmiana architektoniczna — jedno przejście zamiast 1613 obiektów

To jest **główna różnica** i pierwsze zdanie, które powinno paść przy posterze.

Mordred: 1613 obiektów `Descriptor`, każdy z własnym grafem zależności,
rozwiązywanym dla każdej cząsteczki.

My: funkcja `compute(mol, use_3D)` która:

1. buduje **19 obiektów wspólnego przygotowania**,
2. wykonuje **39 wywołań modułów** (dla 2D),
3. każdy moduł wpisuje swój wycinek do jednej, wcześniej zaalokowanej tablicy:

```python
result = np.full(n_features, np.nan, dtype=np.float32)
...
for module, values in descriptors_2d.items():
    result[_OUTPUT_IDXS[module]] = values
return result
```

Liczby są sprawdzalne: 39 wpisów w słowniku `descriptors_2d` w `calculator.py`,
19 przypisań w sekcji przygotowania.

**Skutek uboczny, który warto znać:** nasza implementacja liczy zawsze wszystko.
Nie da się poprosić o dziesięć wybranych deskryptorów i zapłacić tylko za nie —
Mordred to potrafi dzięki grafowi zależności. Przy pełnym zestawie to nasza
zaleta, przy podzbiorze wada.

## 3. Źródło pierwsze — wspólne przygotowanie („cache")

Pełna lista 19 obiektów budowanych raz na cząsteczkę w `compute()`:

| obiekt | co to |
|---|---|
| `n_frags` | liczba rozspójnionych fragmentów |
| `mol_regular` | cząsteczka po standaryzacji RDKit |
| `distance_matrix_regular` | macierz odległości topologicznych |
| `adjacency_matrix_regular` | macierz sąsiedztwa |
| `adjacency_eigendecomposition` | **dekompozycja spektralna macierzy sąsiedztwa** |
| `props_regular` | tablice właściwości atomów i wiązań |
| `rings_regular` | percepcja pierścieni |
| `subgraphs_regular` | enumeracja podgrafów rzędów 1–7 |
| `mol_properties` | wartości RDKit dla całej cząsteczki |
| `mol_hydrogens` | cząsteczka z jawnymi wodorami |
| `props_hydrogens` | właściwości tej wersji |
| `distance_matrix_hydrogens` | macierz odległości z wodorami |
| `gasteiger_charges_hydrogens` | ładunki cząstkowe Gasteigera |
| `cpsa_2d` | część CPSA wspólna dla 2D i 3D |
| `mol_kekulized` | cząsteczka kekulizowana |
| `kekulized_bond_types` | typy wiązań po kekulizacji |
| `estate_indices` | indeksy E-state |
| `graph_radius`, `graph_diameter` | promień i średnica grafu |

**Tablice właściwości czyta 30 z 42 modułów.** To jest ten obiekt, który najbardziej
się opłaca współdzielić.

### Trzy sztuczki wewnątrz przygotowania

**a) Tablica właściwości pierwiastków budowana raz przy imporcie**

```python
ELEMENT_PROPERTY_MATRIX = np.vstack([
    table.lookup(np.arange(119)) for table in ELEMENT_PROPERTY_TABLES.values()
])
```

To macierz **8 właściwości × 119 pierwiastków**, zbudowana w momencie importu
modułu. Cząsteczka pobiera wszystkie osiem właściwości dla wszystkich swoich
atomów **jedną operacją indeksowania**:

```python
self.element_properties = np.ascontiguousarray(ELEMENT_PROPERTY_MATRIX[:, atomic_nums])
```

Zamiast pętli po atomach i po właściwościach. Osiem właściwości to: liczba
atomowa, masa, objętość van der Waalsa, elektroujemności Sandersona, Paulinga
i Allreda–Rochowa, polaryzowalność, potencjał jonizacji.

**b) Wersja z wodorami wyprowadzana, nie odczytywana ponownie**

`AddHs` dokłada wodory w przewidywalnej kolejności: heavy atomy i ich wiązania
zostają nietknięte, a wodory są dopisywane na końcu, pogrupowane po atomie
macierzystym. Dzięki temu tablice dla cząsteczki z wodorami można **wyprowadzić
arytmetycznie** z tablic bez wodorów:

```python
parents = np.repeat(np.arange(props.num_atoms), props.total_num_hs)
atomic_nums=np.concatenate([props.atomic_nums, ones])
```

Przeliczane są tylko ładunki Gasteigera, bo te redystrybuują się po całej
cząsteczce. Oszczędza to drugie przejście po cząsteczce dwukrotnie większej.

**Ważne: jest zabezpieczenie poprawności.**

```python
if num_added != props.total_num_hs.sum():
    return cls.from_mol(mol_hydrogens)   # AddHs zachowal sie inaczej niz zakladamy
```

Czyli optymalizacja nie może po cichu zwrócić złych wartości — przy niespodziance
kod wraca do pełnego odczytu.

**c) Wielkości pochodne przez `bincount`, nie przez pętle**

```python
def _count_neighbors(self, atom_mask):
    counts  = np.bincount(begins, weights=atom_mask[ends], minlength=n)
    counts += np.bincount(ends,   weights=atom_mask[begins], minlength=n)
    return counts.astype(np.intp)
```

Tak liczone są: stopnie atomów, liczby wodorów, elektrony sigma, elektrony
walencyjne, stan wewnętrzny. Wszystko z tablic końców wiązań, bez pętli.

## 4. Źródło drugie — operacje tablicowe

### Autokorelacje: 606 cech z dwóch kontrakcji

To najbardziej efektowny przykład. Rodzina autokorelacji to iloczyn kartezjański
kilkunastu właściwości atomowych i ośmiu opóźnień topologicznych. Naiwnie to
jest podwójna pętla.

U nas:

```python
dist_masks = np.stack([...])                       # (8, n, n) maski odleglosci
weighted        = np.einsum("dij,pj->pdi", dist_masks, props)
quadratic_form  = np.einsum("pdi,pi->pd", weighted, props)
```

Dwie kontrakcje `einsum` po stosie masek i **wszystkie pary (właściwość, opóźnienie)
liczą się naraz**.

### Atrybuty spektralne całego stosu macierzy

`spectral_attributes()` liczy 13 atrybutów (suma wartości bezwzględnych
wartości własnych, największa i najmniejsza wartość własna, indeks Estrady,
rozrzut widma i inne) dla **stosu macierzy**, nie dla jednej:

```python
def spectral_attributes(matrices, props, hermitian, n_frags, eigendecomposition=None):
```

Deskryptory Barysza potrzebują ośmiu macierzy tej samej cząsteczki. Liczone razem,
redukcje idą po całym stosie.

**Ciekawy szczegół, który świadczy o tym, że to było mierzone, a nie zgadywane:**
w komentarzu stoi, że dekompozycje robi się **pojedynczo**, choć NumPy przyjąłby
cały stos — bo wersja stosowa okazała się wolniejsza i wchodziła w inną ścieżkę
LAPACK-a, dającą inne ostatnie cyfry.

### Ponowne użycie dekompozycji spektralnej

`compute()` liczy `np.linalg.eigh(adjacency_matrix)` raz i przekazuje wynik
zarówno do deskryptorów macierzy sąsiedztwa, jak i do liczb spacerów.
`spectral_attributes` ma jawny parametr `eigendecomposition`, żeby nie liczyć
tego drugi raz.

## 5. Źródło trzecie — lepsze algorytmy

### Liczby spacerów z wartości własnych

Liczba spacerów długości k w grafie to ślad k-tej potęgi macierzy sąsiedztwa.
Naiwnie: dziesięć mnożeń macierzy gęstych dla rzędów 1–10.

U nas (`walk_count.py`, `MAX_ORDER = 10`):

```python
orders = np.arange(1, MAX_ORDER + 1)[:, np.newaxis]
eigval_powers = eigvals ** orders
walk_counts = np.rint(eigval_powers @ weights)
```

Potęgi macierzy mają **te same wektory własne**, a wartości własne podnoszą się
do potęgi. Wystarczy więc podnieść wektor wartości własnych — który i tak jest
już policzony dla innych deskryptorów. Zamiast dziesięciu potęgowań O(n³) mamy
jedno potęgowanie wektora.

### Macierz detour z poprawką na układ pierścieni

Element (i,j) macierzy detour to **długość najdłuższej ścieżki prostej** między
atomami. W ogólności to problem NP-trudny.

Oryginał enumeruje wszystkie ścieżki proste w każdej składowej dwuspójnej.

U nas (`detour_matrix.py`), kluczowa obserwacja z docstringa:

> *Poza pierścieniami istnieje tylko jedna ścieżka prosta między dwoma atomami,
> więc odległość detour równa się zwykłej odległości; ścieżka może różnić się od
> najkrótszej tylko tam, gdzie przechodzi przez układ pierścieni. Każdy układ
> pierścieni dokłada się do macierzy odległości jako poprawka. Ścieżka prosta nie
> może opuścić układu pierścieni i do niego wrócić, bo oznaczałoby to odwiedzenie
> dwa razy atomu, przez który wyszła — dlatego poprawki są niezależne.*

Czyli: startujemy z macierzy najkrótszych ścieżek i dokładamy **jedną niezależną
poprawkę na układ pierścieni**. Przeszukiwanie ogranicza się do pierścieni.

**Uwaga do obrony:** to jest szybsze na normalnych cząsteczkach, ale na
fullerenach (37 skondensowanych pierścieni) i tak nie kończy się w rozsądnym
czasie. Mordred ma tam timeout, my nie. Patrz plik o pułapkach.

### Jedna enumeracja podgrafów dla trzech rodzin

`Subgraphs` wykonuje rekurencję **ESU** (Enumeration of Sub-graphs, Wernicke 2006)
dla wszystkich rzędów 1–7 **w jednym przebiegu** — każdy rząd wyrasta
z poprzedniego. Wynik to tablice indeksów wiązań plus pochodne informacje
o topologii (czy podgraf zamyka cykl, jaki ma maksymalny stopień) i ścieżkach.

Dzielą to trzy rodziny: **Chi, PathCount, KappaShapeIndex**. Komentarz w kodzie:
*„Ten algorytm jest sporo szybszy niż użycie RDKit"*.

## 6. Źródło czwarte — zrównoleglenie z konstrukcji

To nie jest osobna optymalizacja, tylko konsekwencja projektu, i tak trzeba to
opisywać.

`compute()` zwraca **jedną tablicę na cząsteczkę** i nie trzyma żadnego stanu
dzielonego. Worker dostaje listę cząsteczek, oddaje gęstą tablicę `float32`,
koniec. Nie ma procesu-serwera, nie ma proxy, nie ma koordynacji.

Zrównoleglenie realizuje bazowy transformer scikit-fingerprints przez
**joblib/loky**. Nasz kod nie robi nic jawnego z wątkami ani z BLAS-em — to jest
sprawdzone i warto to wiedzieć, żeby nie przypisywać sobie czegoś, czego nie ma.

Efekt uboczny, który wychodzi nam na dobre: **loky sam pinuje wątki BLAS**
w workerach, więc nie wpadamy w pułapkę, w którą wpada mordred z surowym
`multiprocessing.Pool`.

## 7. Rozszerzenia względem oryginału

Poza przyspieszeniem, nasza wersja wnosi:

- **Deskryptory kappa i information content** dopisane w osobnych gałęziach
  (`mordred-kappa`, `mordred-information-content`).
- **Poprawka błędu LogEE** znalezionego w oryginale — zbędny człon `exp(−a)`.
  *(Uwaga: to jest z wcześniejszych notatek projektu, nie weryfikowałem osobiście
  w tej sesji — sprawdź, zanim powiesz to publicznie.)*
- **API scikit-learn**: transformer z `fit`/`transform`, parametry `n_jobs`,
  `batch_size`, `sparse`, `use_3D`, zgodny z pipeline'ami sklearn.

## 8. Stan kodu

Pomiary powstały na branchu **`mordred-benchmarks`**, który zawiera cały stos
optymalizacji: gałęzie `mordred-opt-07` do `opt-17` plus `mordred-kappa`
i `mordred-information-content`. **Ten stos nie jest zmergowany do mastera** —
`_new_mordred` w ogóle nie istnieje na `origin/master`.

To warto powiedzieć uczciwie, jeśli ktoś zapyta „czy mogę tego użyć dzisiaj".
Odpowiedź: kod czeka na review, nie jest jeszcze w wydanej bibliotece.
