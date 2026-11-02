# Pytania i odpowiedzi — PL / EN

Każde pytanie ma wersję polską i angielską. Odpowiedzi ułożone od
najprawdopodobniejszych do najbardziej niewygodnych.

---

## A. Pytania podstawowe

### A1. Co dokładnie zrobiliście?

**PL:** Przepisaliśmy Mordreda — bibliotekę liczącą 1826 deskryptorów
molekularnych — wewnątrz scikit-fingerprints. Ten sam zestaw cech, czysty Python
z RDKit i NumPy, bez kompilowanych rozszerzeń. Średnio 7,5× szybciej na
cząsteczkach lekopodobnych, do 14× na peptydach.

**EN:** We reimplemented Mordred — the library computing 1826 molecular
descriptors — inside scikit-fingerprints. Same feature set, pure Python with
RDKit and NumPy, no compiled extensions. On average 7.5× faster on drug-like
molecules, up to 14× on peptides.

### A2. Skąd bierze się przyspieszenie?

**PL:** Z czterech rzeczy. Po pierwsze, architektura: Mordred modeluje każdą
z 1613 wartości jako osobny obiekt z własnym grafem zależności, my liczymy cały
odcisk jednym przebiegiem z 39 wywołań modułów. Po drugie, wspólne przygotowanie:
19 obiektów budowanych raz na cząsteczkę zamiast osobno dla każdej rodziny.
Po trzecie, operacje tablicowe zamiast pętli w Pythonie. Po czwarte, lepsze
algorytmy w kilku miejscach.

**EN:** Four things. First, architecture: Mordred models each of the 1613 values
as a separate object with its own dependency graph; we compute the whole
fingerprint in one pass from 39 module calls. Second, shared preprocessing:
19 objects built once per molecule instead of rebuilt per family. Third, array
operations instead of Python loops. Fourth, better algorithms in several places.

### A3. Czy wyniki są identyczne z oryginałem?

**PL:** Przepisywaliśmy formuła po formule z kodu mordred-community, na licencji
BSD, zachowując konwencje brzegowe i standaryzację cząsteczki. To jest
twierdzenie o konstrukcji i sprawdzalne przez porównanie źródeł. Nie mamy
natomiast systematycznego pomiaru zgodności numerycznej na dużej próbce — jeśli
potrzebujesz liczby typu „ile procent cech zgadza się co do wartości", to jej
jeszcze nie policzyliśmy.

**EN:** We reimplemented it formula by formula from the BSD-licensed
mordred-community code, keeping the edge-case conventions and the molecule
standardisation. That is a claim about construction and it is checkable against
the sources. What we do not have is a systematic numerical agreement measurement
on a large sample — if you want a number like "what percentage of features match
exactly", we have not computed it yet.

### A4. Dlaczego nie w C++ albo Rust?

**PL:** Celem było, żeby to weszło do scikit-fingerprints bez nowych zależności
i bez kroku kompilacji. Instalujesz przez pip i działa, na każdej platformie.
Port do C++ istnieje — osmordred — ale wymaga Boost, Eigen, LAPACK i Ninja przy
budowaniu i obejmuje podzbiór 0D/1D/2D. My mamy pełne 1826 w czystym Pythonie.

**EN:** The goal was to land it in scikit-fingerprints with no new dependencies
and no build step. You pip install it and it works, on any platform. A C++ port
exists — osmordred — but it needs Boost, Eigen, LAPACK and Ninja at build time
and covers a 0D/1D/2D subset. We have the full 1826 in pure Python.

---

## B. Pytania o metodologię

### B1. Jak mierzyliście? Ile powtórzeń?

**PL:** Trzy powtórzenia na punkt, uśredniane, wyjątkiem jest drabinka wątków
z dwoma. Wszystko jednowątkowo poza eksperymentem skalowalności. BLAS pinowany
do jednego wątku przed importem numpy. Pomiary bezwzględne na wyłącznej maszynie,
z blokadą. Każdy pomiar zrzucany do CSV natychmiast, analiza osobnym skryptem.
Razem 254 pomiary, zero błędów, 6 godzin 5 minut czasu maszynowego.

**EN:** Three repeats per point, averaged; the thread ladder has two. Everything
single-threaded except the scaling study. BLAS pinned to one thread before NumPy
is imported. Absolute measurements on an exclusive machine, behind a lock. Every
measurement flushed to CSV as it is taken, analysis by a separate script.
254 measurements in total, zero failures, 6 hours 5 minutes of machine time.

### B2. Na jakim sprzęcie?

**PL:** AMD EPYC 7763, 64 rdzenie fizyczne i 128 wątków sprzętowych, 503 GB RAM.
To rozróżnienie ma znaczenie: wykresy skalowalności idą do 64, czyli do liczby
rdzeni fizycznych, a MolPILE liczyliśmy na 96 workerach i zmierzyliśmy, że to
daje 19,7% więcej niż 64.

**EN:** AMD EPYC 7763, 64 physical cores and 128 hardware threads, 503 GB RAM.
The distinction matters: the scaling charts go to 64, the number of physical
cores, while MolPILE ran on 96 workers, which we measured to be 19.7% faster
than 64.

### B3. Dlaczego akurat te zbiory?

**PL:** MoleculeNet, bo to standard porównań. ExpansionRx/OpenADMET, bo to realne
zbiory ADME o typowym rozmiarze QSAR, od 431 do 7 309 cząsteczek — odpowiadają na
pytanie, czy przewaga utrzymuje się na małych zbiorach. Peptydy, bo dają
kontrolowaną drabinkę rozmiaru od 5 do 100 reszt i pozwalają zbadać zależność od
wielkości cząsteczki.

**EN:** MoleculeNet because it is the standard comparison set. ExpansionRx /
OpenADMET because those are real ADME sets of typical QSAR size, 431 to 7309
molecules — they answer whether the advantage holds on small datasets. Peptides
because they give a controlled size ladder from 5 to 100 residues and let us
study the dependence on molecule size.

### B4. Czemu peptydy, a nie tylko leki?

**PL:** Z dwóch powodów. Po pierwsze, na peptydach da się pokazać skalowalność
z rozmiarem cząsteczki w kontrolowany sposób — generujemy sekwencje o zadanej
długości. Po drugie, peptydy modyfikowane chemicznie realnie potrzebują
deskryptorów atomowych, a Mordred na nich dobrze działa.

**EN:** Two reasons. First, peptides let us show scaling with molecule size in a
controlled way — we generate sequences of a chosen length. Second, chemically
modified peptides genuinely need atom-level descriptors, and Mordred works well
on them.

### B5. Skąd wiadomo, że nie mierzycie parsowania SMILES?

**PL:** SMILES-y są parsowane przed uruchomieniem stopera. Mierzymy wyłącznie
liczenie deskryptorów. Na MolPILE podajemy osobno, ile kosztuje parsowanie:
22 sekundy na milion cząsteczek, wobec 4,4 minuty na same deskryptory.

**EN:** SMILES are parsed before the clock starts. We time descriptor computation
only. For MolPILE we report parsing separately: 22 seconds per million molecules
against 4.4 minutes for the descriptors themselves.

---

## C. Pytania o wykresy

### C1. Czemu na wykresie skalowalności przewaga maleje na peptydach?

**PL:** Bo tam Mordred skaluje się lepiej od nas: 36,7× wobec naszych 27,4× na
64 procesach. Jego architektura dyspozycjonuje zadanie na każdą cząsteczkę, co
jest kosztowne przy tanich cząsteczkach, ale przy peptydzie liczonym sekundę ten
narzut się amortyzuje i daje idealne równoważenie obciążenia. My mamy statyczny
podział pracy, który przy drogich cząsteczkach zaczyna przeszkadzać.

**EN:** Because there Mordred scales better than we do: 36.7× against our 27.4×
on 64 processes. Its architecture dispatches one task per molecule, which is
expensive for cheap molecules, but for a peptide taking a second that overhead
amortises and gives perfect load balancing. We use a static work split, which
starts to hurt on expensive molecules.

### C2. Co oznacza „32× scaling"?

**PL:** To jest o ile przyspieszyliśmy **samych siebie**, przechodząc z jednego
procesu na 64. Nie jest to przewaga nad Mordredem — tę pokazują pionowe klamry
między krzywymi. Przy 64 rdzeniach osiągamy 32-krotność własnej przepustowości
jednordzeniowej, czyli połowę teoretycznego maksimum.

**EN:** It is how much we sped up **ourselves** going from one process to 64. It
is not the advantage over Mordred — that is what the vertical brackets between
the curves show. On 64 cores we reach 32× our own single-core throughput, that is
half of the theoretical maximum.

### C3. Czemu przewaga rośnie z rozmiarem cząsteczki?

**PL:** Bo nasza przewaga bierze się głównie z wektoryzacji, a ta opłaca się tym
bardziej, im większe są tablice. Przy cząsteczce dziesięcioatomowej narzut
wywołania NumPy jest porównywalny z pracą; przy stuatomowej już nie. Zmierzone:
+1,6× na każde e-krotne zwiększenie masy, z R² 0,92.

**EN:** Because our advantage comes mainly from vectorisation, and that pays off
more as the arrays grow. For a ten-atom molecule the NumPy call overhead is
comparable to the work; for a hundred-atom one it is not. Measured: +1.6× per
e-fold increase in mass, with R² 0.92.

### C4. Czy macie lepszą złożoność asymptotyczną?

**PL:** Nie i tego nie twierdzimy. Krzywe kosztu są równoległe — powyżej 100
atomów obie implementacje skalują się jak około n do potęgi 2,4. Przewaga to
stały czynnik, który sam rośnie z rozmiarem, a nie inny wykładnik. Koszt przy
dużych cząsteczkach jest zdominowany przez dekompozycje spektralne, których obie
strony muszą dokonać.

**EN:** No, and we do not claim it. The cost curves are parallel — above 100 atoms
both implementations scale as roughly n to the 2.4. The advantage is a constant
factor that itself grows with size, not a different exponent. At large molecules
the cost is dominated by spectral decompositions that both sides must perform.

### C5. Tabela per rodzina pokazuje 3–9×, a nagłówek 7,5×. Skąd rozbieżność?

**PL:** Bo tabela mierzy każdą rodzinę **w izolacji, wraz z jej prerekwizytami**.
W prawdziwym przebiegu te prerekwizyty liczy się raz na cząsteczkę dla wszystkich
39 modułów naraz. Te ilorazy są więc dolnym oszacowaniem — różnica między medianą
z tabeli a wynikiem całościowym to właśnie zysk ze współdzielenia.

**EN:** Because the table measures each family **in isolation, with its
prerequisites**. In a real run those prerequisites are computed once per molecule
for all 39 modules at once. So those ratios are a lower bound — the difference
between the table's median and the whole-fingerprint result is exactly what
sharing buys.

---

## D. Pytania niewygodne

### D1. Czemu na wykresie kontekstowym Mordred ma 53 mol/s, a w podpisie 399?

**PL:** Bo słupek pokazuje pomiar opublikowany w tamtym benchmarku, gdzie wątki
BLAS nie były ustawione. Ta sama binarka po ustawieniu `OMP_NUM_THREADS=1` daje
399 mol/s. Wyjście jest bit w bit identyczne — to artefakt pomiaru, nie różnica
w implementacji. Uczciwy iloraz to 6,2×, nie 46×, i dlatego podajemy 399.

**EN:** Because the bar shows the measurement as published in that benchmark,
where BLAS threads were not pinned. The same binary with `OMP_NUM_THREADS=1`
reaches 399 mol/s. The output is bit-identical — it is a measurement artefact,
not an implementation difference. The honest ratio is 6.2×, not 46×, which is why
we quote 399.

### D2. Czy 0,318% odfiltrowanych cząsteczek to nie jest zamiatanie pod dywan?

**PL:** Nie, i mówimy o tym sami. To cząsteczki, na których nasza implementacja
albo rzuca wyjątkiem, albo nie kończy liczenia — głównie fullereny, gdzie
najdłuższa ścieżka w 37 skondensowanych pierścieniach jest problemem NP-trudnym.
Oryginał tam też się poddaje, ale robi to kontrolowanie: po trzech minutach
zwraca placeholder i idzie dalej. To jest nasz błąd do naprawienia, brakujący
guard, i mamy go zapisany do zgłoszenia.

**EN:** No, and we say it ourselves. These are molecules where our implementation
either raises or does not terminate — mainly fullerenes, where the longest path
across 37 fused rings is NP-hard. The original gives up there too, but in a
controlled way: after three minutes it returns a placeholder and moves on. That
is a bug on our side, a missing guard, and we have it logged to report.

### D3. Czy `batch_size=250` to nie jest dostrajanie pod benchmark?

**PL:** To jest parametr publicznego API, który Mordred w ogóle nie wystawia.
Sekcja o MolPILE nie jest porównaniem z Mordredem, tylko pomiarem tego, ile nasza
implementacja wyciąga na dużym korpusie — i tam warto użyć jej sensownie.
Wszystkie porównania z Mordredem na pozostałych wykresach idą z ustawieniami
domyślnymi po obu stronach.

**EN:** It is a parameter of the public API that Mordred does not expose at all.
The MolPILE section is not a comparison with Mordred, it is a measurement of what
our implementation achieves on a large corpus — and there it makes sense to use
it properly. All comparisons against Mordred on the other charts use default
settings on both sides.

### D4. Czy to jest dostępne? Mogę tego dziś użyć?

**PL:** Kod jest na gałęzi benchmarkowej scikit-fingerprints i czeka na review.
Nie jest jeszcze w wydanej wersji biblioteki. Wszystkie pomiary na posterze
pochodzą z tej gałęzi.

**EN:** The code is on a benchmark branch of scikit-fingerprints and is awaiting
review. It is not in a released version of the library yet. All measurements on
the poster come from that branch.

### D5. Czy porównywaliście się z osmordredem?

**PL:** Nie. Wiemy z ich README, że to port do C++ obejmujący podzbiór 0D/1D/2D
i wymagający Boost, Eigen i LAPACK przy budowaniu. Nie mierzyliśmy ani ich
szybkości, ani zgodności numerycznej, więc nie powiem nic o tym, jak wypadają.
Nasza różnica to pełne pokrycie 1826 cech bez zależności kompilowanych.

**EN:** No. We know from their README that it is a C++ port covering a 0D/1D/2D
subset and requiring Boost, Eigen and LAPACK at build time. We measured neither
their speed nor their numerical agreement, so I will not claim anything about how
they compare. Our difference is full coverage of all 1826 features with no
compiled dependencies.

### D6. Deskryptory 3D — macie je?

**PL:** Tak, pełne 1826 z 3D. Nie zbenchmarkowaliśmy ich, bo żeby policzyć
deskryptor 3D, trzeba najpierw wygenerować konformer, a to zwykle kosztuje więcej
niż same deskryptory. Pomiar mierzyłby wtedy ETKDG, nie nas.

**EN:** Yes, the full 1826 including 3D. We did not benchmark them because
computing a 3D descriptor requires generating a conformer first, and that usually
costs more than the descriptors themselves. The measurement would be dominated by
ETKDG, not by us.

### D7. Skąd dane do wykresu z 28 modelami?

**PL:** To pomiary z wewnętrznego benchmarku grupy, na zbiorze ogbg-molmuv,
paczki po 10 tysięcy cząsteczek. Nasze są tylko dwa dołożone słupki, zmierzone
w identycznym protokole tym samym harnessem.

**EN:** Those are measurements from an internal group benchmark on the
ogbg-molmuv dataset, batches of ten thousand molecules. Only the two added bars
are ours, measured under the identical protocol with the same harness.

### D8. Czy ktoś tego używa? Jakie ma znaczenie praktyczne?

**PL:** Najciekawszy przypadek to CheMeleon — model fundamentalny pretrenowany
na przewidywaniu wszystkich 1613 deskryptorów Mordreda dla miliona cząsteczek
z PubChem. Zbiór etykiet do takiego pretreningu liczymy w 4,4 minuty. Poza tym
każdy pipeline QSAR na deskryptorach, gdzie dziś liczenie cech bywa dłuższe niż
trenowanie modelu.

**EN:** The most interesting case is CheMeleon — a foundation model pretrained to
predict all 1613 Mordred descriptors for a million PubChem molecules. We compute
that label set in 4.4 minutes. Beyond that, any descriptor-based QSAR pipeline,
where today computing the features often takes longer than training the model.

---

## E. Pytania techniczne dla znawców

### E1. Jak liczycie macierz detour?

**PL:** Startujemy z macierzy najkrótszych ścieżek i dokładamy jedną poprawkę na
układ pierścieni. Poza pierścieniami istnieje tylko jedna ścieżka prosta między
dwoma atomami, więc odległość detour równa się zwykłej. Ścieżka prosta nie może
opuścić układu pierścieni i do niego wrócić, bo oznaczałoby to odwiedzenie dwa
razy atomu, przez który wyszła — dlatego poprawki są niezależne.

**EN:** We start from the shortest-path matrix and add one correction per ring
system. Outside rings there is only one simple path between two atoms, so the
detour distance equals the ordinary one. A simple path cannot leave a ring system
and come back, since that would mean visiting the atom it left through twice —
which is what makes the corrections independent.

### E2. Skąd liczby spacerów bez potęgowania macierzy?

**PL:** Z wartości własnych macierzy sąsiedztwa. Potęgi macierzy mają te same
wektory własne, a wartości własne podnoszą się do potęgi — wystarczy więc podnieść
wektor wartości własnych, który i tak jest już policzony dla innych deskryptorów.
Zamiast dziesięciu potęgowań O(n³) mamy jedno potęgowanie wektora.

**EN:** From the eigenvalues of the adjacency matrix. Matrix powers share the same
eigenvectors, and the eigenvalues are raised to that power — so it is enough to
raise the eigenvalue vector, which is already computed for other descriptors.
Instead of ten O(n³) matrix powers we do one vector exponentiation.

### E3. Jak liczycie 606 cech autokorelacji?

**PL:** Jako dwie kontrakcje einsum po stosie ośmiu masek odległości o kształcie
(8, n, n). Wszystkie pary właściwość–opóźnienie liczą się naraz, bez pętli.

**EN:** As two einsum contractions over a stack of eight distance masks of shape
(8, n, n). All property–lag pairs are computed at once, with no loops.

### E4. Ile pamięci to potrzebuje?

**PL:** Wynik dla miliona cząsteczek to 12,9 GB w float64, 6,4 GB w float32.
Sam transformer zwraca float32. Przy liczeniu równoległym dochodzi pamięć
workerów, ale pojedyncza cząsteczka to tablice rzędu n×n, więc dla cząsteczek
lekopodobnych to kilobajty.

**EN:** The result for a million molecules is 12.9 GB in float64, 6.4 GB in
float32. The transformer itself returns float32. Parallel runs add worker memory,
but a single molecule means n×n arrays, so for drug-like molecules that is
kilobytes.

### E5. Czy da się policzyć podzbiór deskryptorów?

**PL:** Nie w obecnej wersji — liczymy zawsze cały odcisk. Mordred to potrafi
dzięki grafowi zależności i to jest jego zaleta, gdy potrzebujesz dziesięciu cech.
Przy pełnym zestawie nasze podejście wygrywa, bo współdzielenie przygotowania
opłaca się tylko wtedy, gdy liczysz wszystko.

**EN:** Not in the current version — we always compute the whole fingerprint.
Mordred can do it thanks to its dependency graph, and that is its advantage when
you need ten features. For the full set our approach wins, because sharing the
preprocessing only pays off when you compute everything.

---

## F. Gdy nie znasz odpowiedzi

Zdania, które warto mieć gotowe:

**PL:** *„Tego nie mierzyliśmy, więc nie chcę zgadywać. Mogę powiedzieć, co
wiemy: …"*

**EN:** *"We did not measure that, so I do not want to guess. What I can tell you
is what we do know: …"*

**PL:** *„Dobre pytanie — to jest dokładnie ten przypadek, którego nie sprawdziliśmy.
Zapiszę i sprawdzę."*

**EN:** *"Good question — that is exactly the case we did not check. Let me note
it down and look into it."*

Lista rzeczy niezweryfikowanych jest w pliku `06_pulapki_i_odkrycia.md`, sekcja 7.
