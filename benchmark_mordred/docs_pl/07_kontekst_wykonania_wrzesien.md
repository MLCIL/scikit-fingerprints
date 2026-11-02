# 07. Kontekst wykonania benchmarkow (wrzesien 2026)

Odkopane z transkryptu rozmowy `3f64f335-...jsonl` (4-13 wrzesnia). Rzeczy, ktorych
nie ma w skryptach ani w README, a ktore tlumacza, **dlaczego** benchmark wyglada tak,
a nie inaczej.

## Przebieg

| kiedy | co |
|---|---|
| 04.09 06:51 | start, kalibracja; budzet mial sie zmiescic w ~8 h |
| 04.09 07:05 | ustalona zasada: **wszystkie eksperymenty jednowatkowe** poza badaniem skalowalnosci |
| 04.09 09:09 | limit **32-45 watkow** dla Fazy A, zeby procesy nie bily sie o zasoby |
| 04.09 09:23 | `bash benchmark_mordred/run_all.sh` w tmuxie |
| 04.09 09:47 | monitoring co 10 min (`/loop`): logi, procesy, liczba pomiarow i bledow |
| 04.09 16:03 | osobno uruchomiony benchmark MUV ("wykres kolegi") |
| 04.09 19:18 | `run_expansionrx.sh` - dorzucony pozniej, na sugestie promotora |
| 04.09 22:01 | MolPILE 1 mln, 96 watkow, **tylko nasza implementacja** |
| 05.09 | poster, nowe pomiary per rodzina |

Przebieg glowny: **188 pomiarow, 6 h 05 min**.

## Skad wziela sie proba z pinowanym BLAS

Nie z naszej inicjatywy. **Promotor nie uwierzyl w wynik na MUV** (04.09 16:59:
"promotorowi jakos sie wierzyc nie chce w taki wynik") i kazal sprawdzic, czy runner
skfp ma identyczne warunki co runner mordreda.

Sprawdzenie wykazalo, ze **nie my zrobilismy blad, tylko repo referencyjne**: ich
pomiar mordreda szedl bez `OMP_NUM_THREADS=1`, przez co mordred przez nadsubskrypcje
byl ~7.3x wolniejszy. Nasza implementacja jest na to niewrazliwa, bo `joblib`/`loky`
pinuje workery sam.

Stad **dwie proby** w `results/muv_replication/`: `proba1_defaults` (jak u nich) i
`proba2_blas_pinned` (kontrola). Obie sa w repo celowo - patrz
[06](06_porownanie_z_embeddingami.md).

Pytanie uzytkownika z 04.09 17:35 - "czy nasz runner skalowalnosci tez wpadl w ta
pulapke" - doprowadzilo do dodania pinowania do `bench_run.py` **przed importem
numpy** dla wszystkich zestawow.

## Sprawa wykresu 4: cztery odrzucone hipotezy i jedna, ktora przetrwala

Pytanie "czemu na wykresie 4 jest taka ogromna przebitka, skoro na MUV wyszlo raptem
6x" padlo juz **04.09 o 18:29** i zostalo wtedy zbadane czterema pomiarami:

| hipoteza | test | wynik wtedy |
|---|---|---|
| brak pinowania BLAS u nas | 137 watkow, load 7.7 | odrzucona |
| **ciezki ogon rozmiarow** | odciete 2% czasteczek >60 atomow (23% kosztu) | **odrzucona**: 41.0 -> 44.2 mol/s |
| drozsze czasteczki w HIV | stawki jednowatkowe | odrzucona: 0.0885 vs 0.0832 s/mol |
| stan procesu przy forku | czysty proces vs nasz runner | odrzucona: 143.6 / 143.4 / 145.0 s |

Hipoteza, ktora wtedy **przetrwala, ale nie zostala sprawdzona** (cytat z 18:55):

> Najbardziej prawdopodobne: **kilka czasteczek patologicznych dla mordreda, gdzie
> koszt nie zalezy od liczby atomow, a od splatania pierscieni** - liczniki sciezek
> i spacerow eksploduja na skondensowanych ukladach, wiec 40-atomowy policykl moze
> kosztowac minute. Odcialem ogon po atomach, wiec taka czasteczka by przetrwala.

### Rozstrzygniecie (1 listopada)

Zmierzony **rozklad kosztu wprost**, a nie przez posrednik w postaci liczby atomow:

| | HIV | MUV |
|---|---|---|
| mediana s/mol | 0.0910 | 0.0850 |
| srednia s/mol | 0.1440 | 0.0895 |
| maksimum | **6.152 s** | 0.223 s |
| udzial najdrozszego 1% w koszcie | **18.0%** | 2.2% |

Oraz wahanie miedzy losowaniami: **34.8 vs 83.1 mol/s** (ziarna 0 i 1, te same
6 000 czasteczek co do liczby, identyczna srednia wielkosc).

**Wrzesniowa hipoteza byla trafna, a test, ktory ja odrzucil, byl zle wycelowany.**
Ciecie po **liczbie atomow** nie lapie czasteczek, ktorych koszt bierze sie ze
splatania pierscieni. Dlatego usuniecie 23% kosztu dalo tylko 8% - zostaly te
najgorsze sztuki, ktorych filtr po atomach nie dotknal.

Zwiazek z metodyka: nasza strona liczy **caly** HIV (41 124), a mordred **losowe
5 999**, wiec nasza krzywa jest stabilna, a jego loteryjna. Szczegoly w
[04](04_skalowanie_rownolegle.md).

**Wniosek sformulowany juz we wrzesniu i nadal obowiazujacy:** nie uzywac 51x jako
miary przewagi w zrownolegleniu.

## Dlaczego doszly ExpansionRx i dataset_rings

Sugestia promotora (04.09 19:07): *"mozesz wziac czasteczki z OpenADMET, np.
ExpansionRx. To jest pokazanie w druga strone - co kiedy mamy male czasteczki"*.

**Zalozenie o malych czasteczkach sie nie potwierdzilo** i to byl wynik sam w sobie:
ExpansionRx ma srednio **28.6 atomu**, czyli wiecej niz HIV (25.5) i MUV (24.3).
Wyrozniaja je nie male rozmiary, tylko **ciasny rozklad** - mediana 28, maksimum 78,
przy 214 w HIV. Dlatego wlaczono je jako **test hipotezy o ogonie**.

Promotor spodziewal sie tez "bardzo duzego speedupu" na czasteczkach bogatych w
pierscienie. Zmierzono (`dataset_rings`, 34.7 atomu, 4.9 pierscienia): **8.19x** -
czyli tyle, ile dla kazdego zbioru 34-atomowego (`bace` 7.87x, `sider` 8.97x).
**Oczekiwanie promotora zostalo skorygowane pomiarem.**

Faze B scenariusza ExpansionRx (krzywa skalowania na LogD) **zastapiono** zbiorem
ring-rich, bo LogD ma tylko 7 309 czasteczek i powyzej 16 workerow mierzylaby sie
glownie narzut startu puli.

## MolPILE - co bylo intencja

04.09 22:02: *"molpile TYLKO na naszej implementacji, nie licz na orginalnym
mordredzie"*. 05.09 09:44: *"chcialem pokazac na przykladzie molpile 1m jak szybko
jestesmy w stanie to policzyc, bo tu nie porownujemy z og mordredem"*.

Stad brak slupka mordreda w tej sekcji postera - to **nie przeoczenie**, tylko
decyzja: MolPILE pokazuje skale, a nie iloraz.

05.09 09:42: *"na molpile chcialem maksymalna przepustowosc, przepustowosc
porownywalna z innymi modelami byla testowana na MUV"* - stad dwa rozne protokoly i
dwie rozne liczby (605 vs 3 800 mol/s), opisane w [05](05_molpile_przepustowosc.md).

Pytanie z 05.09 09:54 - *"nie zapisuje embeddingow - brzmi podejrzanie"* -
doprowadzilo do udokumentowania, ze harness referencyjny tez nie mierzy zapisu
(`src/embedding/embedding.py:155`), wiec pominiecie go jest warunkiem
porownywalnosci, a nie naciaganiem.

## Rzeczy proceduralne, ktore sie przydaly

- **tmux** do przebiegow nocnych; `Ctrl+B d` do odlaczenia
- `flock` na `~/.mordred-bench.lock` dla Fazy B, zeby nic sie nie zderzylo
- wyniki testow do **osobnego katalogu** (`results_tests/`), zeby nie zanieczyszczac
  kanonicznych CSV - zasada ustalona po 04.09 17:15 ("ZEBY NIE NADPISALO DOBRYCH CSV")
- commit CSV wymaga `git add -f`, bo `.gitignore` wyrzuca `*.csv` i `*.png`
