# Wiedza do obrony postera — RDKit UGM

Materiał przygotowany 2026-09-13. Wszystkie liczby pochodzą z naszych pomiarów
albo wprost z kodu; miejsca niezweryfikowane są **oznaczone jawnie**.

| plik | o czym |
|---|---|
| `01_deskryptory_i_chemoinformatyka.md` | czym są deskryptory, po co, czym różnią się od fingerprintów |
| `02_mordred_oryginalny.md` | historia, architektura, dlaczego jest wolny |
| `03_nasza_implementacja.md` | co przepisaliśmy i jak, cztery źródła przyspieszenia |
| `04_grupy_deskryptorow.md` | 1613 cech w rozbiciu na rodziny, co każda liczy |
| `05_wyniki_i_metodologia.md` | wszystkie liczby z postera i jak powstały |
| `06_pulapki_i_odkrycia.md` | BLAS, batch_size, fullereny, błędy naszej implementacji |
| `07_pytania_i_odpowiedzi.md` | Q&A po polsku i angielsku |
| `08_wejscie_i_wyjscie.md` | format wejścia i wyjścia, NaN-y, parametry, porównanie API |
| `09_scikit_fingerprints.md` | czym jest biblioteka i dlaczego Mordred trafił właśnie tam |

---

## Pitch na 30 sekund

Mordred liczy **1826 deskryptorów molekularnych** (1613 dwuwymiarowych) i jest
standardem w chemoinformatyce, ale kosztuje około **0,1 s na cząsteczkę leku**
na jednym rdzeniu. To wystarczy do zbioru na tysiąc cząsteczek, ale nie do
korpusu pretreningowego.

Przepisaliśmy go w całości wewnątrz **scikit-fingerprints**, w czystym Pythonie
z RDKit i NumPy. Ten sam zestaw cech, **7–14× szybciej** na jednym wątku,
z przewagą rosnącą wraz z rozmiarem cząsteczki. Milion cząsteczek MolPILE liczy
się w **4,4 minuty** na jednym węźle.

## Pitch na 2 minuty

Trzy rzeczy, które zmieniliśmy:

1. **Jedno przejście zamiast 1613 obiektów.** Mordred modeluje każdą wartość
   wyjściową jako obiekt `Descriptor` z własnym grafem zależności rozwiązywanym
   per cząsteczka. My liczymy cały odcisk jednym przebiegiem: 39 wywołań modułów
   wpisujących wyniki do jednej, wcześniej zaalokowanej tablicy.
2. **Wspólne przygotowanie.** 19 obiektów budowanych raz na cząsteczkę: macierze
   sąsiedztwa i odległości wraz z dekompozycją spektralną, tablice właściwości
   atomów, percepcja pierścieni, enumeracja podgrafów. Mordred odbudowuje takie
   rzeczy dla każdej rodziny osobno.
3. **Operacje tablicowe zamiast pętli.** 606 cech autokorelacji powstaje z dwóch
   kontrakcji `einsum`; zliczenia idą przez `bincount`; atrybuty spektralne liczy
   się dla całego stosu macierzy naraz.

Plus kilka lepszych algorytmów: liczby spacerów z wartości własnych zamiast
dziesięciu potęgowań macierzy, macierz detour z najkrótszych ścieżek plus
poprawka na układ pierścieni, jedna enumeracja podgrafów dzielona przez trzy
rodziny.

## Liczby, które musisz pamiętać

| co | wartość |
|---|---|
| deskryptory 2D / z 3D | **1613 / 1826** |
| MoleculeNet, jeden wątek | **średnio 7,5×** (5,6–9,0×) |
| ADME ExpansionRx | średnio 7,6× (7,4–8,2×) |
| peptydy do 100 reszt | średnio 11,2× (7,6–13,9×) |
| wzrost z masą | **+1,6× na e-krotność masy**, R² 0,92 |
| skalowalność, 64 workery | **32×** na lekach, 27× na peptydach |
| MolPILE, 96 workerów | **3 800 ± 8 mol/s**, milion w 4,4 min |
| w protokole MUV | **2 485 mol/s** wobec 399 dla poprawionego oryginału |
| pomiarów w benchmarku | **254, zero błędów** |
