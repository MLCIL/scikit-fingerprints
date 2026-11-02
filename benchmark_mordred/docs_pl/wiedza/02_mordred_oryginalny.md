# Oryginalny Mordred — co to jest i dlaczego jest wolny

## 1. Podstawowe fakty

- **Publikacja:** H. Moriwaki, Y.-S. Tian, N. Kawashita, T. Takagi,
  *Mordred: a molecular descriptor calculator*, J. Cheminformatics 10, 4 (2018).
- **Zawartość:** 1613 deskryptorów 2D, 1826 z 3D.
- **Język:** czysty Python, backend RDKit, bez rozszerzeń kompilowanych.
- **Stan:** oryginalne repozytorium `mordred-descriptor/mordred` nie jest już
  aktywnie rozwijane. Utrzymywany fork to **mordredcommunity** (Jackson Burns),
  i to jego używamy jako punktu odniesienia we wszystkich pomiarach.
- **Ciekawostka wartа wspomnienia:** Jackson Burns jest jednocześnie autorem
  CheMeleona, czyli modelu pretrenowanego na deskryptorach Mordreda. Ten sam
  człowiek utrzymuje bibliotekę i buduje na niej model fundamentalny.

## 2. Architektura — model obiektowy

To jest sedno różnicy między nami a oryginałem i **najważniejsza rzecz do
zrozumienia przed obroną postera**.

Mordred modeluje **każdą wartość wyjściową jako osobny obiekt klasy `Descriptor`**.
Sprawdzone wprost:

```python
from mordred import Calculator, descriptors
len(Calculator(descriptors, ignore_3D=True).descriptors)   # 1613
```

Każdy taki obiekt ma:

- metodę `calculate()` zwracającą swoją wartość,
- metodę `dependencies()` deklarującą, czego potrzebuje,
- własną tożsamość (`__eq__`, `__hash__`) pozwalającą go cache'ować.

`Calculator` przy liczeniu cząsteczki buduje **graf zależności**: rozwija
deskryptory w ich zależności, te w swoje, aż do liści (macierz sąsiedztwa,
właściwości atomów). Potem wykonuje to w kolejności topologicznej, cache'ując
wyniki pośrednie **w obrębie jednej cząsteczki**.

### Dlaczego to jest kosztowne

1. **Narzut dyspatchu per wartość.** 1613 obiektów oznacza 1613 wywołań
   `calculate()` plus rozwiązywanie zależności. Przy cząsteczce liczonej ~100 ms
   to jest zauważalna część budżetu.
2. **Brak współdzielenia między rodzinami na poziomie projektu.** Cache działa,
   ale architektura zachęca do tego, żeby każda rodzina deklarowała własne
   zależności. W efekcie ta sama praca bywa wykonywana wielokrotnie w różnych
   miejscach, bo rodziny nie wiedzą o sobie nawzajem.
3. **Pętle w Pythonie.** Formuły są pisane per atom albo per para atomów,
   w czystym Pythonie, zamiast jako operacje na tablicach.
4. **Zależności od bibliotek zewnętrznych.** W naszym profilowaniu mordred spędza
   **21% czasu w bibliotekach zewnętrznych**, z czego RDKit 7,5 s i **networkx
   2,4 s** na 300 cząsteczkach MUV. Konwersja grafu do obiektów networkx i z
   powrotem to czysty narzut.

## 3. Zrównoleglenie — `calc.map()` i proxy `Manager()`

Mordred udostępnia `calc.map(mols, nproc=N)`. Kluczowy kod jest w
`mordred/_base/parallel.py`:

```python
class MolPool:
    def __init__(self, calc, nproc):
        self.pool = Pool(nproc)
        self.mgr = Manager()
        self.calc_proxy = self.mgr.list([calc])      # kalkulator w proxy!

    def submit(self, mol, id):
        cxt = Context.from_calculator(self.calc, mol, id)
        return self.pool.apply_async(worker, (self.calc_proxy, cxt))
```

Czyli:

- **jedno zadanie na cząsteczkę** (`apply_async` per cząsteczka),
- z buforem `nproc*2 + 10` zadań w locie (`MolIterator`),
- kalkulator przekazywany przez **proxy `multiprocessing.Manager`**.

### Konsekwencje, które zmierzyliśmy

Proxy `Manager()` to osobny proces-serwer. Każde odwołanie do kalkulatora
w workerze to runda komunikacji międzyprocesowej z tym serwerem. Przy jednym
zadaniu na cząsteczkę płaci się ten koszt **na każdą cząsteczkę**.

Pomiar na HIV (cząsteczki tanie, ~25 atomów, ~15 ms każda):

| workery | Original Mordred | nasze |
|---|---|---|
| 1 | 8,0 mol/s | 67,5 mol/s |
| 64 | **42,0 mol/s** | 2 158 mol/s |
| skalowanie | **5,3×** | 32× |

Mordred na 64 procesach przyspiesza tylko 5,3×. To nie jest wada zrównoleglenia
jako takiego — to koszt dyspozycji, który dominuje, gdy cząsteczka jest tania.

Pomiar na peptydach 20-resztowych (cząsteczki drogie, ~168 atomów, ~1,1 s każda):

| workery | Original Mordred | nasze |
|---|---|---|
| 1 | 0,9 mol/s | 10,2 mol/s |
| 64 | 33,0 mol/s | 279,6 mol/s |
| skalowanie | **36,7×** | 27,4× |

Tutaj mordred skaluje się **lepiej od nas**. Jego narzut dyspozycji amortyzuje
się, bo jedna cząsteczka to sekunda pracy, a nasz statyczny podział zaczyna
przeszkadzać. Przewaga maleje z 10,8× na 8,5×.

**To jest uczciwy i ważny punkt do obrony.** Nie twierdzimy, że mordred źle się
zrównolegla w ogóle. Twierdzimy, że jego architektura dyspozycji per cząsteczka
jest kosztowna przy tanich cząsteczkach, a to jest dominujący przypadek użycia.

## 4. Znane słabe punkty oryginału

### Brak pinowania wątków BLAS

To jest nasze odkrycie metodologiczne i jest na posterze.

Mordred liczy dekompozycje spektralne przez NumPy, czyli przez OpenBLAS. OpenBLAS
domyślnie rozpina pulę wątków na wszystkie dostępne rdzenie. Gdy uruchamiasz
**96 procesów roboczych**, każdy z nich rozpina własną pulę i procesy biją się
o rdzenie.

Zmierzone na MUV, 96 workerów, identyczny protokół:

| konfiguracja | czas na 10 tys. cząsteczek | mol/s |
|---|---|---|
| bez pinowania (jak w cudzym runnerze) | **182,4 s** | 53 |
| z `OMP_NUM_THREADS=1` przed importem numpy | **25,1 s** | 399 |

**Artefakt 7,3×.** Wyjście obu przebiegów jest **bit w bit identyczne** — to
czysta kwestia wydajności, nie poprawności.

Trzy szczegóły, które trzeba znać:

1. Zmienne muszą być ustawione **przed importem numpy**, bo OpenBLAS czyta je
   przy ładowaniu biblioteki. Ustawienie ich później nie działa.
2. Dotyczy `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
   `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`.
3. **Nasza implementacja jest na to odporna**, bo joblib/loky sam pinuje workery
   (ustawia `OMP_NUM_THREADS = cpu_count // n_jobs`). Mordred używa surowego
   `multiprocessing.Pool`, który tego nie robi. Zmierzone: nasze 3,95 s bez
   pinowania wobec 4,02 s z pinowaniem, czyli w granicach szumu.

### Timeout w DetourMatrix

Mordred ma w macierzy detour mechanizm limitu czasu. Gdy trafi na cząsteczkę,
której nie jest w stanie policzyć, po ~173–194 s zwraca wartość
`timed out (SpAbs_Dt/.../DetourMatrixCache())` i **liczy dalej**.

To wygląda jak proteza, ale jest to **zabezpieczenie, którego my nie mamy** —
patrz plik o pułapkach. Na fullerenach nasza implementacja nie kończy liczenia
w ogóle.

### Ogólna filozofia obsługi błędów

Mordred na patologicznej cząsteczce wstawia placeholder błędu dla feralnego
deskryptora i kontynuuje. Nasza implementacja rzuca wyjątkiem, który zabija całą
paczkę. To jest realne odstępstwo od oryginału i mamy je zapisane jako błąd do
zgłoszenia.

## 5. Co Mordred robi dobrze

Dla uczciwości i żeby nie wyjść na kogoś, kto tylko krytykuje:

- **Kompletność.** 1826 deskryptorów w jednym pakiecie, z dokumentacją i
  odniesieniami do publikacji źródłowych dla każdego.
- **Poprawność.** To jest referencja, względem której wszyscy się porównują.
  My też — przepisywaliśmy formuła po formule z jego kodu.
- **Brak zależności kompilowanych.** Instaluje się przez pip i działa.
- **Graf zależności.** Architektura obiektowa jest elegancka i pozwala policzyć
  **dowolny podzbiór** deskryptorów, płacąc tylko za ich zależności. Nasza
  implementacja liczy wszystko naraz — to zaleta przy pełnym zestawie, ale wada,
  gdy potrzebujesz dziesięciu cech.
- **Dynamiczne równoważenie obciążenia.** Dyspozycja per cząsteczka jest kosztowna,
  ale daje idealny balans. Na drogich cząsteczkach wygrywa z naszym statycznym
  podziałem.

## 6. Licencja i pochodzenie naszego kodu

Nasza implementacja jest **adaptacją kodu mordred-community na licencji BSD**.
Każdy plik w `skfp/fingerprints/_new_mordred/` ma w nagłówku informację:

```
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community
See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
```

**To jest ważne przy posterze.** Gdy ktoś zapyta „czy to na pewno daje te same
wartości", odpowiedź brzmi: przepisywaliśmy z ich własnego kodu, formuła po
formule, zachowując konwencje brzegowe. To twierdzenie o konstrukcji, sprawdzalne
przez porównanie źródeł.

**Czego NIE mamy:** systematycznego pomiaru zgodności numerycznej na dużej
próbce. Jeśli ktoś zapyta „ile procent cech zgadza się co do wartości" —
nie mamy tej liczby i trzeba to powiedzieć wprost.
