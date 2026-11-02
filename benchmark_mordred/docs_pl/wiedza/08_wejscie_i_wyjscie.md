# Format wejścia i wyjścia

Wszystko sprawdzone praktycznie na kodzie, nie z pamięci.

## 1. Wejście

Transformer przyjmuje **sekwencję cząsteczek** w jednej z trzech postaci:

| postać | przykład | uwagi |
|---|---|---|
| **SMILES** | `["CCO", "c1ccccc1"]` | najczęstsze, parsowane przez `MolFromSmiles` |
| **InChI** | `["InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"]` | rozpoznawane po prefiksie `InChI=` |
| **RDKit `Mol`** | `[Chem.MolFromSmiles("CCO")]` | gdy masz już sparsowane |

**Wszystkie elementy muszą być tego samego typu.** Mieszanka stringów i obiektów
`Mol` rzuca `TypeError` z wypisaniem znalezionych typów. Podobnie cząsteczka,
której nie da się sparsować, rzuca `TypeError` ze wskazaniem indeksu:

```
Could not parse 'XYZ' at index 3 as molecule
```

To jest zachowanie odziedziczone z `skfp.utils.ensure_mols` i wspólne dla
wszystkich fingerprintów w scikit-fingerprints.

### Ważne przy pomiarach

**Parsowanie SMILES odbywa się wewnątrz `transform()`**, więc jeśli mierzysz czas
i chcesz mierzyć wyłącznie deskryptory, podaj obiekty `Mol`, nie stringi.
Tak właśnie robiliśmy w benchmarku — na MolPILE parsowanie miliona SMILES-ów to
dodatkowe 22 sekundy wobec 4,4 minuty na same deskryptory.

### Tryb 3D wymaga konformerów

Przy `use_3D=True` transformer ustawia `requires_conformers=True` i oczekuje, że
każda cząsteczka ma **wygenerowany konformer oraz własność `conf_id`**:

```python
mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
cid = AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
mol.SetIntProp("conf_id", cid)
```

Bez tego leci `KeyError: 'conf_id'`. W normalnym użyciu robi to za ciebie
`ConformerGenerator` z scikit-fingerprints.

## 2. Wyjście

### Kształt i typ

| parametr | wynik |
|---|---|
| `use_3D=False` (domyślnie) | `np.ndarray` o kształcie **(n, 1613)**, `float32` |
| `use_3D=True` | `np.ndarray` o kształcie **(n, 1826)**, `float32` |
| `sparse=True` | `scipy.sparse.csr_array` o tym samym kształcie, `float32` |

Sprawdzone praktycznie:

```
wejscie SMILES  -> ndarray (2, 1613) float32
wejscie Mol     -> ndarray (1, 1613) float32
sparse=True     -> csr_array (1, 1613) float32
```

**Wiersz = cząsteczka, kolumna = deskryptor.** Kolejność wierszy odpowiada
kolejności wejścia; kolejność kolumn jest stała i deterministyczna.

### Nazwy cech

```python
fp.get_feature_names_out()      # ndarray, 1613 nazw
# ['ABC', 'ABCGG', 'nAcid', 'nBase', 'SpAbs_A', 'SpMax_A', ...]
```

Nazwy są zgodne z konwencją Mordreda, więc kolumny da się porównać jeden do
jednego z oryginałem. Interfejs `get_feature_names_out` jest standardem
scikit-learn, więc działa w `ColumnTransformer` i przy `set_output(transform="pandas")`.

**Pułapka historyczna projektu:** kolejność nazw musi zgadzać się z kolejnością
wartości wpisywanych do tablicy. To było źródłem błędów na wcześniejszym etapie —
`feature_names.py` ma 1840 linii właśnie dlatego, że ta kolejność jest ustalona
jawnie, a nie generowana.

### Wartości brakujące — NaN, nie zero

To jest **najważniejsza rzecz do powiedzenia o wyjściu** i pytanie o to padnie.

Tablica jest inicjalizowana jako `np.full(n_features, np.nan)`, a deskryptory
nadpisują swoje wycinki. NaN oznacza **wartość niezdefiniowaną dla tej cząsteczki**,
nie zero i nie błąd.

Zmierzone na 300 cząsteczkach BACE (typowy zbiór QSAR, ~34 atomy):

| | |
|---|---|
| NaN na cząsteczkę | średnio **142 z 1613** (mediana 141) |
| kolumny w całości NaN | **86** |
| kolumny bez żadnego NaN | **1391** |
| kolumny stałe, zerowa wariancja | **170** |

Dla porównania etanol (`CCO`, 3 atomy) ma **369 NaN** — im mniejsza cząsteczka,
tym więcej deskryptorów jest niezdefiniowanych.

**Typowe przyczyny NaN:**

- deskryptory pierścieniowe na cząsteczce bez pierścieni
- autokorelacje przy opóźnieniu większym niż średnica grafu
- atrybuty spektralne dla cząsteczek rozspójnionych (sole, kompleksy) —
  propagujemy NaN, tak jak oryginał
- dzielenie przez zero, np. `cpsa.py` liczy `q_max / sum(charges)`, a suma
  ładunków bywa zerowa

**Konsekwencja praktyczna:** pipeline QSAR musi mieć krok czyszczący —
usunięcie kolumn w całości NaN, kolumn stałych, imputację reszty. To nie jest
nasza specyfika, tak zachowuje się też oryginał.

### Cząsteczki, których nie policzymy

Osobna sprawa od NaN. Na trzech klasach cząsteczek nasza implementacja **rzuca
wyjątkiem albo nie kończy liczenia** — organometaliki z nieskończoną walencją,
przypadki braku zbieżności LAPACK, fullereny w `DetourMatrix`. Wyjątek zabija
całą paczkę joblib, więc przy dużych korpusach trzeba je odfiltrować wcześniej.

Odsetki: HIV 0,007%, zbiór pierścieniowy 0,02%, MolPILE 0,318%.

Oryginał w tych miejscach wstawia placeholder błędu i liczy dalej. To jest nasze
odstępstwo i mamy je zapisane jako błąd do zgłoszenia.

## 3. Pełny przykład użycia

```python
from skfp.fingerprints import NewMordredFingerprint

fp = NewMordredFingerprint(
    use_3D=False,      # 1613 cech; True daje 1826, ale wymaga konformerow
    sparse=False,      # True zwraca csr_array
    n_jobs=96,         # liczba procesow roboczych
    batch_size=250,    # cząsteczek na zadanie; None = len(X) // n_jobs
    verbose=0,
)

X = fp.transform(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
# ndarray (3, 1613) float32

names = fp.get_feature_names_out()   # 1613 nazw zgodnych z Mordredem
```

Jako transformer scikit-learn wchodzi wprost do pipeline'u:

```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

model = make_pipeline(
    NewMordredFingerprint(n_jobs=-1),
    SimpleImputer(strategy="median"),   # NaN-y trzeba czyms zastapic
    RandomForestRegressor(),
)
```

## 4. Parametry, o które mogą zapytać

| parametr | domyślnie | co robi |
|---|---|---|
| `use_3D` | `False` | 1613 kontra 1826 cech; `True` wymaga konformerów |
| `sparse` | `False` | zwraca `csr_array` zamiast `ndarray` |
| `n_jobs` | `None` | liczba procesów roboczych; `-1` to wszystkie rdzenie |
| `batch_size` | `None` | cząsteczek na zadanie; `None` daje `len(X) // n_jobs` |
| `verbose` | `0` | pasek postępu |

**`batch_size` jest tym parametrem, o którym mówi sekcja MolPILE.** Domyślne
`len(X) // n_jobs` daje jedno zadanie na workera i jest złe dla zbiorów
o niejednorodnym koszcie cząsteczki — zmierzyliśmy sześciokrotną różnicę na
MolPILE między domyślnym a `batch_size=250`.

## 5. Porównanie z interfejsem Mordreda

| | Original Mordred | nasze |
|---|---|---|
| wywołanie | `Calculator(descriptors)(mol)` | `fp.transform(mols)` |
| wynik pojedynczej cząsteczki | obiekt `Result`, iterowalny | wiersz tablicy NumPy |
| wynik wielu cząsteczek | `calc.pandas(mols)` → DataFrame | `ndarray` albo `csr_array` |
| podzbiór deskryptorów | **tak**, `Calculator(mordred.Chi)` | **nie**, zawsze cały odcisk |
| zrównoleglenie | `calc.map(mols, nproc=N)` | `n_jobs=N` w konstruktorze |
| API sklearn | brak | **pełne**, `fit`/`transform`/`get_feature_names_out` |
| błąd na cząsteczce | placeholder w wyniku, liczy dalej | wyjątek zabija paczkę |

**Możliwość policzenia podzbioru to realna zaleta oryginału.** Jeśli ktoś
potrzebuje dziesięciu deskryptorów, Mordred policzy tylko ich poddrzewo zależności,
a my i tak policzymy wszystkie 1613. Nasza architektura opłaca się przy pełnym
zestawie, bo współdzielenie przygotowania ma sens tylko wtedy, gdy liczysz wszystko.
