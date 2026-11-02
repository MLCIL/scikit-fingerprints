# Grupy deskryptorów — 1613 cech w rozbiciu

Liczby pochodzą wprost z `Calculator(descriptors, ignore_3D=True).descriptors`,
grupowane po module. Suma 2D: **1613**, z 3D: **1826**.

## 1. Pełna tabela

| moduł | 2D | +3D | czym jest |
|---|---:|---:|---|
| Autocorrelation | **606** | 0 | autokorelacje po odległości topologicznej |
| EState | **316** | 0 | indeksy elektrotopologiczne i kubełki VSA |
| RingCount | **138** | 0 | zliczenia pierścieni wg rozmiaru i typu |
| BaryszMatrix | **104** | 0 | atrybuty spektralne macierzy Barysza |
| Chi | 56 | 0 | indeksy połączalności Kiera–Halla |
| MoeType | 53 | 0 | deskryptory typu MOE (SlogP_VSA, SMR_VSA, PEOE_VSA) |
| ExtendedTopochemicalAtom | 45 | 0 | indeksy ETA Roya i Ghosha |
| InformationContent | 42 | 0 | entropia klas równoważności atomów |
| BCUT | 24 | 0 | wartości własne macierzy Burdena |
| PathCount | 21 | 0 | zliczenia ścieżek molekularnych |
| TopologicalCharge | 21 | 0 | indeksy przeniesienia ładunku Galveza |
| WalkCount | 21 | 0 | zliczenia spacerów |
| MolecularDistanceEdge | 19 | 0 | wektory MDE między typami atomów |
| AtomCount | 17 | 0 | zliczenia atomów wg pierwiastka i hybrydyzacji |
| Constitutional | 16 | 0 | deskryptory konstytucyjne |
| DetourMatrix | 14 | 0 | atrybuty spektralne macierzy detour |
| AdjacencyMatrix | 12 | 0 | atrybuty spektralne macierzy sąsiedztwa |
| DistanceMatrix | 12 | 0 | atrybuty spektralne macierzy odległości |
| MolecularId | 12 | 0 | identyfikatory molekularne Randicia |
| CarbonTypes | 11 | 0 | klasyfikacja atomów węgla |
| BondCount | 9 | 0 | zliczenia wiązań wg typu |
| TopologicalIndex, ZagrebIndex | 4+4 | 0 | indeksy klasyczne |
| KappaShapeIndex | 3 | 0 | indeksy kształtu Kiera |
| CPSA | 2 | **41** | powierzchnia ważona ładunkiem |
| ABCIndex, AcidBase, Aromatic, HydrogenBond, Lipinski, Polarizability, RotatableBond, SLogP, TopoPSA, Weight, WienerIndex | 2 każdy | 0 | drobne rodziny |
| BalabanJ, BertzCT, EccentricConnectivityIndex, FragmentComplexity, Framework, LogS, McGowanVolume, VdwVolumeABC, VertexAdjacencyInformation | 1 każdy | 0 | pojedyncze indeksy |
| MoRSE | 0 | **160** | 3D-MoRSE |
| GeometricalIndex, GravitationalIndex | 0 | 4+4 | indeksy geometryczne |
| MomentOfInertia | 0 | 3 | momenty bezwładności |
| PBF | 0 | 1 | plane of best fit |

**Trzy rodziny to 1060 z 1613 cech (66%):** Autocorrelation, EState, RingCount.

## 2. Co dokładnie liczą najważniejsze rodziny

### Autocorrelation — 606 cech, największa rodzina

Autokorelacja przestrzenna przeniesiona na graf molekularny. Idea: weź
właściwość atomową *p* (masa, elektroujemność, polaryzowalność…), weź opóźnienie
topologiczne *d* (ile wiązań dzieli parę atomów) i policz statystykę po wszystkich
parach atomów w odległości dokładnie *d*.

Trzy warianty klasyczne:

- **Moreau–Broto (ATS)** — suma iloczynów `p(i)·p(j)` po parach w odległości d.
- **Moran (MATS)** — wariant znormalizowany, analogiczny do współczynnika Morana.
- **Geary (GATS)** — oparty na kwadracie różnicy `(p(i) − p(j))²`.

Plus warianty uśrednione (AATS) i centrowane (ATSC, AATSC). Przy ~13
właściwościach atomowych, 8 opóźnieniach (d = 0…8) i kilku wariantach wychodzi 606.

**Dlaczego to u nas jest szybkie:** cała rodzina to dwie kontrakcje `einsum` po
stosie ośmiu masek odległości. Zmierzone na BACE: **6,6×** przewagi.

### EState — 316 cech

Indeksy elektrotopologiczne Kiera i Halla. Każdy atom dostaje wartość łączącą
jego stan elektronowy (elektrony walencyjne, wodory) z topologią (stopień,
odległości do pozostałych atomów). Cechy to sumy, maksima i minima po typach
atomów, plus warianty VSA — czyli powierzchnia van der Waalsa pocięta na kubełki
według wartości E-state.

Współdzielenie: `estate_indices` liczone raz w `compute()` i używane przez
całą rodzinę. Zmierzone na BACE: **5,1×**.

### RingCount — 138 cech

Zliczenia pierścieni w rozbiciu na: rozmiar (3–12 i „większe"), aromatyczność,
obecność heteroatomów, przynależność do układów skondensowanych. Iloczyn tych
kryteriów daje 138 kombinacji.

Wymaga percepcji pierścieni, która u nas jest w `rings_regular` i budowana raz.

### Chi — 56 cech

Indeksy połączalności Kiera–Halla, uogólnienie indeksu Randicia. Dla podgrafów
rzędu 0–7, w wariantach: ścieżki (path), klastry (cluster), łańcuchy (chain),
klastry ścieżkowe (path-cluster), każdy w wersji zwykłej i walencyjnej.

Wzór ma postać sumy po podgrafach z iloczynu odwrotności pierwiastków ze stopni
atomów wchodzących w podgraf.

**Dlaczego to jest drogie:** wymaga enumeracji wszystkich spójnych podgrafów do
rzędu 7. U nas ta enumeracja jest wspólna z PathCount i Kappa. Zmierzone na BACE:
**4,9×** przy liczeniu z prerekwizytami.

### BaryszMatrix — 104 cechy

Macierz Barysza to macierz odległości ważona rzędami wiązań i właściwościami
atomowymi. Dla każdej z ośmiu właściwości atomowych buduje się osobną macierz
i liczy 13 atrybutów spektralnych: sumę wartości bezwzględnych wartości własnych,
największą i najmniejszą, indeks Estrady, rozrzut widma, log wyznacznika i inne.

**8 macierzy × 13 atrybutów = 104.** U nas liczone jako **stos** — jedno wywołanie
`spectral_attributes` na wszystkie osiem. Zmierzone na BACE: **4,9×**.

### InformationContent — 42 cechy

Entropia Shannona rozkładu klas równoważności atomów. Atomy dzieli się na klasy
według otoczenia o rosnącym promieniu (0, 1, 2… wiązania), liczy entropię tego
podziału, a potem warianty: znormalizowany, strukturalny, komplementarny, bonding.

Wymaga cząsteczki z jawnymi wodorami i typów wiązań po kekulizacji — oba są
w naszym wspólnym przygotowaniu. Zmierzone na BACE: **4,5×**.

### BCUT — 24 cechy

Deskryptory Burdena–CAS–University of Texas. Buduje się zmodyfikowaną macierz
połączeń: na diagonali właściwość atomowa, poza diagonalą funkcja rzędu wiązania.
Bierze się **najwyższą i najniższą wartość własną**. Dla kilkunastu właściwości
daje to 24 cechy.

Klasyczne zastosowanie: opis przestrzeni chemicznej i różnorodności bibliotek.

### ExtendedTopochemicalAtom — 45 cech

Indeksy ETA (Roy i Ghosh). Każdemu atomowi przypisuje się „miarę rdzenia"
zależną od elektronów walencyjnych, a potem sumuje po parach z wagą zależną od
odległości topologicznej. Warianty uwzględniają rozgałęzienie, elektroujemność,
obecność heteroatomów.

Zmierzone na BACE: **9,3×** — najwyższa przewaga w tabeli.

### MolecularDistanceEdge — 19 cech

Wektory MDE: dla każdej pary typów atomów węgla (pierwszo-, drugo-, trzecio-,
czwartorzędowy) i dla par heteroatomów liczy się średnią geometryczną odległości
topologicznych między atomami tych typów.

Zmierzone na BACE: **7,2×**.

### DetourMatrix — 14 cech

Atrybuty spektralne macierzy detour, czyli macierzy **najdłuższych ścieżek
prostych**. To najdroższa koncepcyjnie rodzina, bo problem najdłuższej ścieżki
jest NP-trudny.

Zmierzone na BACE: **3,0×** — najniższa przewaga w tabeli. To rodzina, w której
nasza optymalizacja pomaga najmniej, bo obie strony muszą przeszukiwać pierścienie.

**I to jest dokładnie ta rodzina, na której nasza implementacja wiesza się na
fullerenach.**

### MolecularId — 12 cech

Identyfikatory molekularne Randicia: sumy po ścieżkach ważonych odwrotnościami
pierwiastków ze stopni. Zmierzone na BACE: **3,9×**.

### WalkCount i PathCount

**WalkCount (21)** — liczby spacerów o długości 1–10, gdzie spacer może odwiedzać
wierzchołki wielokrotnie. U nas liczone z wartości własnych.

**PathCount (21)** — liczby ścieżek prostych o zadanej długości, gdzie wierzchołki
nie mogą się powtarzać. Wymaga enumeracji podgrafów. Zmierzone na BACE: **3,8×**.

Różnica spacer/ścieżka to typowe pytanie egzaminacyjne z teorii grafów i może paść.

### Rodziny „cienkie" — nakładki na RDKit

`SLogP`, `TopoPSA`, `Lipinski`, `HydrogenBond`, `AcidBase`, `RotatableBond`,
`CarbonTypes`, `LogS`, `Weight` — to w dużej mierze wywołania RDKit opakowane
w interfejs Mordreda.

**Tu nasza przewaga jest bliska 1×** i to jest dobry znak, nie zły: obie strony
wołają ten sam kod C++ RDKita, więc nie ma czego przyspieszać. Gdyby wszędzie
wychodziło 100×, to byłby sygnał, że coś jest nie tak z pomiarem.

## 3. Tabela z postera — speedup per rodzina

Pomiar na **1513 cząsteczkach BACE**, jeden wątek, najlepszy z 3 przebiegów.
Każda strona liczy rodzinę **wraz z jej prerekwizytami, od zera**.

| rodzina | cech | mordred | nasze | speedup |
|---|---:|---:|---:|---:|
| ExtendedTopochemicalAtom | 45 | 24,75 s | 2,65 s | **9,3×** |
| MolecularDistanceEdge | 19 | 9,75 s | 1,35 s | 7,2× |
| Autocorrelation | 606 | 17,11 s | 2,61 s | 6,6× |
| EState | 316 | 18,18 s | 3,53 s | 5,1× |
| BaryszMatrix | 104 | 19,76 s | 4,01 s | 4,9× |
| Chi | 56 | 49,47 s | 10,13 s | 4,9× |
| InformationContent | 42 | 21,93 s | 4,83 s | 4,5× |
| MolecularId | 12 | 13,39 s | 3,43 s | 3,9× |
| PathCount | 21 | 35,36 s | 9,42 s | 3,8× |
| DetourMatrix | 14 | 5,98 s | 2,03 s | 3,0× |

### Jak to czytać i czego NIE mówić

**Metodologia:** mordred liczony przez `Calculator(moduł)`, który liczy tylko
poddrzewo zależności tej rodziny. My budujemy dokładnie te obiekty, które
`compute()` podaje danej rodzinie na wejściu, i mierzymy razem z nimi.

**Dlaczego tak, a nie profilowaniem:** profilowanie przez `tottime` przypisuje
funkcji tylko jej własny czas. Nasze rodziny byłyby wtedy „cienkie", bo wspólne
przygotowanie leży w innych plikach, a rodziny mordreda „grube", bo on tę pracę
wykonuje wewnątrz każdej z nich. Iloraz wychodziłby z tego, **gdzie kod leży**,
a nie ile liczy — i nie dałoby się tego obronić. Profilowaniem Chi wychodziło
67×, tą metodą 4,9×.

**Kluczowe zastrzeżenie, które musi paść:** w tym pomiarze prerekwizyty liczymy
**raz na rodzinę**, a w prawdziwym przebiegu raz na cząsteczkę dla wszystkich 39
modułów naraz. Te ilorazy są więc **dolnym oszacowaniem** — mediana po rodzinach
to 4,7×, a cały odcisk daje 7,2× na MUV. Różnica to właśnie zysk ze współdzielenia.

To jest dobra odpowiedź na pytanie „czemu tabela pokazuje mniej niż nagłówek".
