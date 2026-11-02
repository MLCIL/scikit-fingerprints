# Deskryptory molekularne — kontekst dziedzinowy

## 1. Definicja i miejsce w chemoinformatyce

Deskryptor molekularny to **liczba wyliczona deterministycznie ze struktury
cząsteczki**. Klasyczna definicja Todeschiniego i Consonniego: *„wynik logicznej
i matematycznej procedury przekształcającej informację chemiczną zakodowaną
w symbolicznej reprezentacji cząsteczki w liczbę użyteczną"*.

Trzy własności, które odróżniają deskryptor od reprezentacji uczonej:

1. **Determinizm** — ten sam graf zawsze daje tę samą wartość. Bez ziarna, bez wag,
   bez wersji modelu.
2. **Niezmienniczość względem numeracji atomów** — deskryptor nie może zależeć od
   tego, w jakiej kolejności RDKit ponumerował atomy. Stąd konstrukcje oparte na
   sumach, śladach macierzy, wartościach własnych i histogramach.
3. **Interpretowalność** — każda kolumna ma nazwę i sens. `nAromRing` to liczba
   pierścieni aromatycznych, `SLogP` to oszacowanie lipofilowości.

### Skąd się to wzięło

- **1947** — Harry Wiener publikuje indeks nazwany później jego nazwiskiem: suma
  odległości topologicznych między wszystkimi parami atomów. Koreluje z temperaturą
  wrzenia alkanów. To pierwszy indeks topologiczny.
- **1975** — Milan Randić wprowadza indeks połączalności: suma po wiązaniach
  z 1/√(δᵢ·δⱼ), gdzie δ to stopień atomu. Punkt wyjścia dla całej rodziny chi.
- **lata 80.–90.** — Kier i Hall uogólniają Randicia na podgrafy wyższych rzędów
  (ścieżki, klastry, łańcuchy), wprowadzają indeksy kappa opisujące kształt
  i indeksy E-state łączące topologię z elektroniką.
- **lata 90.** — Burden i Pearlman: deskryptory BCUT, oparte na wartościach
  własnych zmodyfikowanej macierzy połączeń.
- **2000** — Todeschini i Consonni wydają *Handbook of Molecular Descriptors*,
  katalog kilku tysięcy deskryptorów. To do dziś główne źródło definicji.
- **2018** — Moriwaki i wsp. publikują **Mordred**, który implementuje 1826
  z nich w jednym pakiecie Pythona. To jest ten program, który przepisaliśmy.

### Dlaczego akurat Mordred stał się standardem

Przed Mordredem liczenie szerokiego zestawu deskryptorów wymagało albo
komercyjnego DRAGON-a, albo sklejania kilku bibliotek. Mordred dał **jeden pakiet,
otwartoźródłowy, w Pythonie, z RDKit jako backendem**. Płaci za to wydajnością,
bo jest napisany w czystym Pythonie z ciężką warstwą obiektową.

## 2. Taksonomia — na jakie poziomy dzielą się deskryptory

### 0D — bez topologii

Liczone ze wzoru sumarycznego. Masa cząsteczkowa, liczba atomów, liczba
heteroatomów, liczba elektronów walencyjnych. U Mordreda to m.in. `Weight`,
`AtomCount` (17 cech), częściowo `Constitutional` (16 cech).

### 1D — listy fragmentów

Zliczenia grup funkcyjnych, liczba wiązań danego typu, liczba pierścieni.
`BondCount` (9), `RingCount` (138), `CarbonTypes` (11).

### 2D — graf bez geometrii

**To jest 1613 cech, o których mówi poster.** Wymagają grafu molekularnego, ale
nie współrzędnych. Dzielą się grubo na:

- **indeksy topologiczne** — Wiener, Balaban J, Zagreb, ABC, Bertz CT
- **indeksy połączalności** — rodzina chi (56 cech), kappa (3)
- **deskryptory macierzowe** — atrybuty spektralne macierzy sąsiedztwa,
  odległości, Barysza, detour
- **autokorelacje** — ATS/AATS/ATSC/MATS/GATS, 606 cech
- **elektrotopologiczne** — E-state i pochodne VSA, 316 cech
- **zawartość informacyjna** — entropia Shannona klas równoważności atomów, 42
- **właściwości fizykochemiczne szacowane** — SLogP, TPSA, McGowan volume

### 3D — wymagają konformacji

MoRSE (160), CPSA (41), GeometricalIndex (4), GravitationalIndex (4),
MomentOfInertia (3), PBF (1). Razem 213 cech, dających 1826 z 2D.

**Ważne dla obrony postera:** my mamy 3D zaimplementowane (pełne 1826), ale ich
**nie zbenchmarkowaliśmy**. Powód jest merytoryczny: żeby policzyć deskryptor 3D,
trzeba najpierw wygenerować konformer (np. algorytmem ETKDG w RDKit), a to
kosztuje zwykle więcej niż same deskryptory. Pomiar mierzyłby wtedy ETKDG, nie nas.

## 3. Deskryptory kontra fingerprinty kontra embeddingi uczone

To rozróżnienie trójdzielne warto mieć gotowe, bo poster stawia nas obok
wszystkich trzech rodzin.

| | deskryptory | fingerprinty | embeddingi uczone |
|---|---|---|---|
| czym jest | wektor liczb o nazwanych składowych | wektor bitów/zliczeń podstruktur | wektor z wytrenowanej sieci |
| przykład | Wiener, SLogP, nAromRing | ECFP, MACCS, AtomPair, TT | CheMeleon, MolE, ChemBERTa |
| determinizm | pełny | pełny | zależny od wag i wersji |
| interpretacja | nazwa i sens na kolumnę | bit = „jakiś fragment" | brak |
| typowy wymiar | 1613 | 1024–4096 | 300–2048 |
| koszt (nasz pomiar, MUV) | **2 485 mol/s** | ECFP 14 109 mol/s | MolE 1 570, Chemeleon 1 012 |
| potrzebuje GPU | nie | nie | zwykle tak |

Nasza pozycja na wykresie kontekstowym: **czwarci z 29**, za ECFP, Topological
Torsion i AtomPair, a **przed wszystkimi modelami neuronowymi**, łącznie z tymi
liczonymi na GPU.

### Kiedy co wybrać

- **Fingerprinty** — gdy liczy się tylko przepustowość i masz dużo danych.
  Przesiew miliardowych bibliotek.
- **Deskryptory** — gdy zbiór jest mały (setki–tysiące), potrzebujesz
  interpretowalności albo model ma być tabelaryczny.
- **Embeddingi uczone** — gdy masz dużo danych nieoznaczonych do pretreningu
  i zadanie, na którym transfer działa.

Praktyka: na małych zbiorach QSAR deskryptory z Random Forest bywają **lepsze**
od wytrenowanych GNN. Publikacja CheMeleona podaje win rate na benchmarkach
Polaris: Random Forest 68%, fastprop 36%, Chemprop 32%. Klasyczny model na
deskryptorach bije sieć grafową.

## 4. Gdzie się tego realnie używa

### QSAR / QSPR

*Quantitative Structure–Activity/Property Relationship*. Model przewidujący
aktywność biologiczną albo własność fizykochemiczną ze struktury. Typowy zbiór
to **setki do kilku tysięcy cząsteczek** — dlatego promotor zwrócił uwagę, że
BACE z 1513 cząsteczkami jest realistycznym rozmiarem do tabeli per rodzina.

### ADMET

Absorpcja, Dystrybucja, Metabolizm, Ekskrecja, Toksyczność. To jest rodzina
zbiorów ExpansionRx/OpenADMET z naszego wykresu 1:

| zbiór | co mierzy |
|---|---|
| `erx_logd` | współczynnik podziału oktanol/woda przy pH 7,4 |
| `erx_ksol` | rozpuszczalność kinetyczna |
| `erx_hlm_clint` / `mlm` / `rlm` | klirens mikrosomalny: człowiek, mysz, szczur |
| `erx_mppb` / `mbpb` | wiązanie z białkami osocza / mózgu (mysz) |
| `erx_caco2_papp` / `efflux` | przepuszczalność i efflux w modelu Caco-2 |
| `erx_mgmb` | najmniejszy zbiór rodziny, 431 cząsteczek |

Wszystkie mają **średnią masę 378–425 Da**, czyli typowe cząsteczki lekopodobne,
i **431–7 309 cząsteczek**. To wyjaśnia, dlaczego przyspieszenie jest tam bardzo
stabilne: 7,4–8,2×, czyli rozrzut mniejszy niż na MoleculeNet.

### MoleculeNet

Standardowy zestaw benchmarkowy (Wu i wsp., *Chem. Sci.* 2018). Nasze 11 zbiorów:

| zbiór | zadanie | ~masa |
|---|---|---|
| FreeSolv | energia solwatacji | 138 Da |
| ESOL | rozpuszczalność wodna | 206 Da |
| Tox21, ToxCast | toksyczność, wiele zadań | 268–270 Da |
| BBBP | przenikanie bariery krew–mózg | 339 Da |
| MUV | aktywność, zbiór trudny | 344 Da |
| HIV | aktywność przeciw HIV | 379 Da |
| Lipophilicity | lipofilowość | 384 Da |
| ClinTox | toksyczność kliniczna | 398 Da |
| BACE | inhibicja beta-sekretazy | 483 Da |
| SIDER | efekty uboczne | 491 Da |

Nasze przyspieszenie rośnie z masą w tej grupie: FreeSolv 5,6×, SIDER 9,0×.

### Pretrening modeli fundamentalnych — CheMeleon

To najnowsze i najciekawsze zastosowanie, i ono jest na posterze.

**CheMeleon** (Burns i wsp., arXiv:2506.15792) to model fundamentalny
pretrenowany na **1 milionie cząsteczek z PubChem**, którego zadaniem
pretreningowym jest **przewidzieć wszystkie 1613 deskryptorów Mordreda**.
Architektura to D-MPNN (directed message passing neural network) z Chemprop,
warstwa message passing o wymiarze 2048 i 8,7 mln parametrów, razem 12,9 mln.
Po pretreningu testowe RMSE wynosi 0,14 uśrednione po 1613 deskryptorach
(każdy wystandaryzowany i winsoryzowany do sześciu odchyleń).

**Dlaczego akurat deskryptory jako cel pretreningu:** są niskoszumowe. Dane
eksperymentalne mają szum pomiarowy i błąd systematyczny, symulacje kwantowe mają
błąd metody. Deskryptor jest deterministyczną funkcją grafu, więc sieć uczy się
czystej zależności struktura→własność bez zanieczyszczenia etykietą.

**Dlaczego to jest argument za naszą pracą:** zbiór etykiet pretreningowych dla
CheMeleona to dokładnie te 1613 wartości dla miliona cząsteczek. My liczymy taki
zbiór w **4,4 minuty**. To przesuwa skalę, w jakiej można takie modele trenować.

## 5. Problemy praktyczne, o które ktoś może zapytać

### Wartości brakujące i nieskończone

Deskryptory potrafią zwracać NaN albo inf. Typowe przyczyny:

- **dzielenie przez zero** — np. `cpsa.py` liczy `q_max / sum(charges)`, a suma
  ładunków bywa zerowa. To realnie występuje na wielu cząsteczkach HIV.
- **cząsteczki rozspójnione** — atrybuty spektralne wymagają jednego fragmentu;
  dla soli i kompleksów propagujemy NaN, tak jak oryginał.
- **logarytmy z wartości niedodatnich** — `matrix_attributes.py` ma jawne
  zabezpieczenie `np.where(vr1 > 0, np.log(...), np.nan)`.

W praktyce pipeline QSAR musi mieć krok czyszczący: usunięcie kolumn stałych,
kolumn z NaN i wysoko skorelowanych.

### Standaryzacja cząsteczki

Wynik zależy od tego, jak znormalizujesz strukturę przed liczeniem: kekulizacja,
usunięcie wodorów jawnych, wybór formy tautomerycznej, jonizacja. Nasza
implementacja przechodzi przez `preprocess_mol`, odtwarzając konwencje Mordreda.
**Odpowiedź, gdy ktoś zapyta o zgodność:** przepisywaliśmy formuła po formule
z kodu mordred-community, ta sama standaryzacja i te same konwencje brzegowe.

### Skala wartości

Deskryptory mają wartości rozciągnięte przez wiele rzędów wielkości: od zliczeń
rzędu jedności po indeksy rzędu 10⁶. Praktycznie zawsze wymagają skalowania przed
modelem. CheMeleon robi standaryzację plus winsoryzację do 6σ.

## 6. Najczęstsze zarzuty wobec deskryptorów i odpowiedzi

**„To przestarzałe, teraz są GNN-y."**
Na małych zbiorach deskryptory z Random Forest bywają lepsze — patrz win rate
z publikacji CheMeleona. Poza tym GNN-y i deskryptory nie wykluczają się:
CheMeleon uczy się właśnie deskryptorów.

**„1613 cech to za dużo, będzie przeuczenie."**
To realny problem i odpowiedź brzmi: selekcja cech albo regularyzacja. Ale koszt
liczenia nie jest argumentem przeciw — policz wszystko, wybierz potem.

**„Wiele z tych cech jest skorelowanych."**
Tak, i to jest znana własność. Rodzina autokorelacji to 606 cech będących wariantami
tej samej konstrukcji dla ośmiu opóźnień i kilkunastu właściwości atomowych.
Nadmiarowość jest wpisana w projekt: lepiej policzyć i odfiltrować, niż zgadywać.

**„Czemu nie osmordred, skoro jest w C++?"**
osmordred to port do C++ obejmujący **podzbiór 0D/1D/2D**. Wymaga Boost, Eigen,
LAPACK i Ninja przy budowaniu. Nasza wersja jest w czystym Pythonie, bez
kompilowanych rozszerzeń i bez nowych zależności, i obejmuje **pełne 1826 cech**.
*(Uwaga: nie mamy własnego pomiaru osmordreda ani porównania numerycznego —
nie twierdź niczego o dokładności ich wyników.)*

## 7. Słownik pojęć

- **Graf molekularny** — atomy jako wierzchołki, wiązania jako krawędzie. Zwykle
  *hydrogen-suppressed*, bez jawnych wodorów.
- **Macierz sąsiedztwa (A)** — 1 gdy atomy związane, 0 wpp.
- **Macierz odległości topologicznych (D)** — element (i,j) to liczba wiązań na
  najkrótszej ścieżce. Nie mylić z odległością w angstremach.
- **Macierz detour (Δ)** — element (i,j) to długość **najdłuższej ścieżki prostej**
  między atomami. Problem NP-trudny w ogólności.
- **Macierz Barysza** — macierz odległości ważona rzędami wiązań i liczbami
  atomowymi, parametryzowana właściwością atomową.
- **Stopień atomu (δ)** — liczba wiązań, w których uczestniczy.
- **Stopień walencyjny (δᵛ)** — liczba elektronów walencyjnych minus liczba wodorów.
- **Podgraf spójny rzędu k** — spójny podgraf o k wiązaniach. Podstawa indeksów chi.
- **Ścieżka prosta** — ścieżka nieodwiedzająca żadnego wierzchołka dwukrotnie.
- **Spacer (walk)** — ciąg wiązań, w którym wierzchołki mogą się powtarzać.
- **Wartości własne macierzy sąsiedztwa** — widmo grafu; wiele deskryptorów to
  funkcje widma (Estrada index, liczby spacerów, BCUT).
- **VSA** — *Van der Waals Surface Area*, deskryptory dzielące powierzchnię na
  kubełki po wartości innej właściwości (logP, MR, ładunek).
- **SMILES** — tekstowy zapis struktury.
- **Konformer** — konkretna geometria 3D. Generowana np. przez ETKDG.
- **Scaffold split** — podział po rdzeniach strukturalnych, trudniejszy niż losowy.
- **Dalt / Da** — jednostka masy cząsteczkowej. Peptyd 100-resztowy to ~11 kDa.
