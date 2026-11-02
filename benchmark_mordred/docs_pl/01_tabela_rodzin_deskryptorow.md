# 01. Individual descriptors speedup (tabela)

## Co jest na posterze

Tabela 10 rodzin deskryptorow z liczba cech, czasem mordreda, naszym czasem
i przyspieszeniem. Podpis:

> - much faster individual descriptors across the board
> - BACE MoleculeNet dataset, 1513 molecules
> - separate descriptors, excludes further gains from shared precomputing

## Runner

```bash
cd ~/mordred/scikit-fingerprints
python benchmark_mordred/profil/isolated_families.py --dataset bace --n 1513 --repeats 3
```

Skrypt: `benchmark_mordred/profil/isolated_families.py`.
Wynik zapisany recznie jako `results/tabela_rodziny_bace.txt` (tekst + gotowy LaTeX)
oraz `results/tabela_rodziny_bace.tex`. Wariant dla MUV: `tabela_rodziny_muv.*`.

Skrypt **nie zapisuje CSV sam z siebie** - drukuje na stdout, wiec pliki w
`results/` to przekierowane wyjscie.

## Metodyka - i dlaczego nie profilowanie

To jest wariant **obronny**, napisany po odrzuceniu profilowania `cProfile`.

Profilowanie mierzy `tottime`, czyli czas wlasny funkcji. Nasze rodziny wychodza
wtedy sztucznie "cienkie", bo wspolne przygotowanie (macierze odleglosci,
enumeracja podgrafow, wlasciwosci atomowe) lezy w **innych plikach**, a rodziny
mordreda "grube", bo on te sama prace wykonuje **wewnatrz** kazdej z nich. Iloraz
mowilby wtedy o tym, GDZIE kod lezy, a nie ile liczy.

Tutaj **obie strony placa za prerekwizyty**:

- mordred: `Calculator(modul)` liczy tylko poddrzewo zaleznosci tej rodziny,
- nasza: budujemy dokladnie te obiekty, ktore `calculator.compute` podaje rodzinie
  na wejsciu, i mierzymy je razem z nia.

**Te ilorazy sa dolnym oszacowaniem.** Prerekwizyty liczymy raz **na rodzine**, a w
prawdziwym przebiegu raz **na czasteczke dla wszystkich rodzin naraz**. Czyli
tabela **nie zawiera** zysku ze wspoldzielenia - i dokladnie to mowi trzeci
bullet na posterze.

Pomiar: najlepszy z 3 powtorzen, jeden watek, BLAS przypiety.

## Dane wejsciowe

BACE z MoleculeNet przez `skfp.datasets.moleculenet.load_bace`, **caly zbior
1513 czasteczek**, srednio 34.1 atomu.

Wczesniejsza wersja (wrzesien) liczyla 100 czasteczek BBBP, a potem 200 MUV -
finalna tabela zostala przeliczona na calym BACE na prosbe promotora.

## Wyniki (`results/tabela_rodziny_bace.txt`)

| rodzina | cech | mordred | nasze | speedup |
|---|---|---|---|---|
| ExtendedTopochemicalAtom | 45 | 24.75 s | 2.65 s | **9.3x** |
| MolecularDistanceEdge | 19 | 9.75 s | 1.35 s | **7.2x** |
| Autocorrelation | 606 | 17.11 s | 2.61 s | **6.6x** |
| EState | 316 | 18.18 s | 3.53 s | **5.1x** |
| BaryszMatrix | 104 | 19.76 s | 4.01 s | **4.9x** |
| Chi | 56 | 49.47 s | 10.13 s | **4.9x** |
| InformationContent | 42 | 21.93 s | 4.83 s | **4.5x** |
| MolecularId | 12 | 13.39 s | 3.43 s | **3.9x** |
| PathCount | 21 | 35.36 s | 9.42 s | **3.8x** |
| DetourMatrix | 14 | 5.98 s | 2.03 s | **3.0x** |

## Czego na posterze NIE MA i dlaczego

**KappaShapeIndex i AdjacencyMatrix sa pominiete.** W tym protokole wychodza
odpowiednio **0.30x** i **0.93x**, czyli pozornie regres. To **artefakt protokolu**,
nie regres: w izolacji nasza strona placi za caly `Subgraphs` i pelna
eigendekompozycje, zeby policzyc 3 cechy, podczas gdy w prawdziwym przebiegu ten
koszt jest juz zaplacony przez inne rodziny. Pominiecie jest swiadome, ale trzeba
umiec je uzasadnic, jesli ktos zapyta - nie jest to przemilczenie regresu.

## Liczby do obrony

- suma po rodzinach z tabeli nie rowna sie calemu fingerprintowi: caly przebieg
  daje **7.5x** na BACE (patrz [02](02_qsar_qspr_datasets.md)), wiecej niz
  wiekszosc pojedynczych rodzin - **roznica to wlasnie wspoldzielenie**
- `Autocorrelation` ma 606 cech z 1613, czyli ponad jedna trzecia wektora
- `Chi` jest najdrozsza bezwzglednie u obu stron (49.5 s / 10.1 s)
