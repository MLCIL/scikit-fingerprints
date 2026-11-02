# 03. Molecule size scaling (para wykresow)

## Co jest na posterze

Dwa panele w skali log-log: koszt czasteczki wobec masy oraz przyspieszenie
wobec masy. Podpis:

> - very good scaling, bigger gains for larger molecules
> - 5.6x at 140 Da, 13.9x at 11 kDa
> - note the log-scales above

## Runner

Te same przebiegi, co w [02](02_qsar_qspr_datasets.md) - suite `speedup`,
`expansionrx` i `size`. Wykres rysuje `benchmark_mordred/figures_final.py`:

```bash
cd ~/mordred/scikit-fingerprints
python benchmark_mordred/dataset_masses.py          # -> results/dataset_masses.csv
python benchmark_mordred/figures_final.py           # chart2, chart3, chart_pair
```

- `chart2` -> `2_cost_per_molecule.png` (koszt)
- `chart3` -> `3_speedup_vs_size.png` (przyspieszenie)
- `chart_pair` -> `2_3_large_molecules.png` - **to jest para z postera**

## Dlaczego masa, a nie liczba atomow

Osi X zmieniono z liczby atomow na **mase czasteczkowa** na prosbe promotora:
chemicy operuja raczej na masie. Jednego nie da sie przeliczyc na drugie bez
skladu pierwiastkowego, wiec masy sa liczone osobno (`dataset_masses.py`, srednia
`mean_mw` na zbior) i podlaczane przez `p["mw"] = p.dataset.map(masses)`.

## Drabinka rozmiaru

| zbior | n | atomow | masa [Da] | speedup |
|---|---|---|---|---|
| freesolv | 642 | 8.7 | **138** | **5.63** |
| esol | 1128 | 13.3 | 206 | 6.83 |
| bace | 1513 | 34.1 | 483 | 7.87 |
| synthpep_5 | 200 | 42.6 | 610 | 7.64 |
| synthpep_10 | 100 | 84.1 | 1 203 | 9.24 |
| preactor_ace_vaxinpad | 200 | 101.1 | 1 341 | 10.63 |
| synthpep_20 | 50 | 167.2 | 2 387 | 10.90 |
| synthpep_40 | 20 | 333.8 | 4 758 | 12.76 |
| synthpep_80 | 10 | 666.5 | 9 498 | 11.82 |
| **preactor_hiv_lpv** | 10 | 764.3 | **10 886** | **13.86** |
| synthpep_100 | 5 | 831.8 | 11 877 | 12.43 |
| _synthpep_160_ | 5 | 1 332 | 18 979 | _14.59_ |
| _preactor_hiv_nvp_ | 2 | 1 994 | 28 109 | _18.04_ |

Kursywa: **poza wykresem** z postera, odciete progiem 840 atomow (100 AA).

Liczby z postera: **5.6x przy 138 Da** to `freesolv`, **13.9x przy 10 886 Da**
to `preactor_hiv_lpv` (realny peptyd z PeptideReactor, inhibitor HIV lopinawir).

## Metodyka

- jeden watek, BLAS przypiety, 3 powtorzenia (dla `hiv_nvp` mozliwe `--repeats 1`,
  bo koszt jest deterministyczny O(n^3) i powtorzenia nie niosa informacji)
- **liczebnosc maleje z rozmiarem** (200 -> 100 -> 50 -> 20 -> 10 -> 5 -> 2): przy
  duzych czasteczkach wariancja jest mala, bo koszt zdominowany przez
  deterministyczna dekompozycje spektralna
- peptydy syntetyczne `synthpep_L`: losowe sekwencje z 20 aminokwasow, dlugosc L
- peptydy realne: repozytorium **PeptideReactor**, katalog wskazany `--peptide-dir`

## Wykladniki skalowania (z `chart2`)

```
mordred: koszt ~ n^1.85
nasze  : koszt ~ n^1.67
```

## Pulapki - najwazniejsza sekcja tego pliku

**NIE twierdzic, ze mamy fundamentalnie lepsze skalowanie asymptotyczne.**
Dopasowanie do dwoch malych punktow sugerowalo kiedys n^2.07 vs n^2.83, ale pelna
drabinka 9-1994 atomow pokazuje, ze przy duzych czasteczkach **obie implementacje
skaluja sie podobnie** - koszt jest zdominowany przez dekompozycje spektralne
O(n^3), ktorych obie musza dokonac. Wykladniki 1.85 vs 1.67 sa dopasowane na
zakresie zdominowanym przez male czasteczki i nie sa asymptota.

**Trend nie jest monotoniczny.** `synthpep_80` (11.8x) wypada **nizej** niz
`synthpep_40` (12.8x), a `synthpep_100` (12.4x) nizej niz `hiv_lpv` (13.9x).
Przy n = 5-10 czasteczkach to mieści sie w szumie, ale wykres z linia laczaca
punkty sugerowalby trajektorie, ktorej nie ma - dlatego `chart3` **celowo nie
laczy punktow linia**, tylko rysuje dopasowanie.

**Teza "bigger gains for larger molecules" jest poprawna, ale plateau nie zostalo
wykluczone.** Stare pomiary z laptopa (bez stosu optymalizacji opt-06..opt-17)
sugerowaly plateau na ~3.5x; po optymalizacjach przyspieszenie rosnie do 18x przy
1994 atomach i plateau nie widac w zmierzonym zakresie. To nie dowodzi, ze go nie ma.

**Koszt duzych czasteczek jest brutalny.** Jedna czasteczka 1994-atomowa to ~19 min
(59.6 s nasze + 1074 s mordred). Stad malejaca liczebnosc i stad `hiv_nvp` jako
najdluzszy pojedynczy proces calego benchmarku.
