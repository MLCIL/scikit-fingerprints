# 02. QSAR/QSPR datasets (wykres slupkowy)

## Co jest na posterze

Trzy poziome slupki: MoleculeNet, ExpansionRx-OpenADMET, Peptides, z liczba
przyspieszenia na koncu kazdego. Podpis:

> - consistent gains, at least 7.5x faster than original
> - mean speedup per benchmark
> - varied dataset & molecule sizes, primarily small drug-like

## Runner

**Faza A**, rownolegle, jeden proces na zbior:

```bash
cd ~/mordred/scikit-fingerprints
for d in freesolv esol sider clintox bace bbbp lipophilicity tox21 toxcast hiv muv; do
    python benchmark_mordred/bench_run.py --suite speedup --only $d --out results/res_speedup_$d.csv &
done
bash benchmark_mordred/run_expansionrx.sh      # 10 zbiorow ADME
for L in 5 10 20 40 80 100; do
    python benchmark_mordred/bench_run.py --suite size --only synthpep_$L --out results/res_size_synthpep_$L.csv &
done
for P in ace_vaxinpad hiv_lpv hiv_nvp; do
    python benchmark_mordred/bench_run.py --suite size --only preactor_$P \
        --peptide-dir "$PEPDIR" --out results/res_size_preactor_$P.csv &
done
wait
```

Wykres: `benchmark_mordred/figures_final.py`, funkcja `chart1` -> `1_speedup_datasets.png`.

## Dane

| zrodlo | zbiorow | plik |
|---|---|---|
| MoleculeNet | 11 | `results/res_speedup_*.csv` |
| ExpansionRx (ADME) | 10 | `results/expansionrx/res_erx_*.csv` |
| peptydy | 8 (<= 840 atomow) | `results/res_size_*.csv` |

Masy czasteczkowe: `results/dataset_masses.csv` (skrypt `benchmark_mordred/dataset_masses.py`).

## Metodyka

- **jeden watek** (`n_jobs == 1`), BLAS przypiety do 1
- 3 powtorzenia, usredniane przez `seconds_per_mol.mean()`
- HIV i MUV przyciete do **losowych 10 000** czasteczek (`SUBSAMPLE_CAP`,
  `bench_run.py:112`, ziarno 0). Pelne zbiory maja odpowiednio 41 127 i **93 087**
  czasteczek. **Uwaga:** ten sam MUV wystepuje na wykresie 5
  ([06](06_porownanie_z_embeddingami.md)) w **pelnej liczebnosci 93 087** - tam
  liczebnosc nie byla przycinana, bo protokol repo referencyjnego mierzy caly zbior
- slupek to **srednia po zbiorach w rodzinie**, nie po czasteczkach
- speedup liczony **per zbior** jako `mordred.s_per_mol / ours.s_per_mol`, dopiero
  potem usredniany - inaczej duze zbiory zdominowalyby wynik

**Peptydy ucinane na 840 atomach (100 AA).** Powyzej tego to juz bialka, a wiekszosc
definicji tnie peptyd nawet na 50 AA. Bez tego cuta najwiekszym punktem bylby
241-resztowy `preactor_hiv_nvp` i slupek "Peptides" skoczylby z 11.2x na wyzej,
ale przestalby opisywac peptydy.

## Wyniki

| rodzina | srednia | min-max | zbiorow |
|---|---|---|---|
| **Peptides** | **11.2x** | 7.6 - 13.9x | 8 |
| **ExpansionRx-OpenADMET** | **7.6x** | 7.4 - 8.2x | 10 |
| **MoleculeNet** | **7.5x** | 5.6 - 9.0x | 11 |

Stad "at least 7.5x" na posterze - to **najnizsza ze srednich**, a nie najnizszy
pojedynczy pomiar.

### MoleculeNet w rozbiciu (mediana 7.22x)

| zbior | n | atomow | mordred s/mol | nasze s/mol | speedup |
|---|---|---|---|---|---|
| freesolv | 642 | 8.7 | 0.041 | 0.007 | 5.63 |
| esol | 1128 | 13.3 | 0.062 | 0.009 | 6.83 |
| lipophilicity | 4200 | 27.0 | 0.106 | 0.015 | 7.05 |
| toxcast | 8576 | 18.8 | 0.071 | 0.010 | 7.17 |
| muv | 10000 | 24.2 | 0.089 | 0.012 | 7.19 |
| tox21 | 7831 | 18.6 | 0.078 | 0.011 | 7.22 |
| bbbp | 2039 | 24.1 | 0.121 | 0.016 | 7.74 |
| bace | 1513 | 34.1 | 0.138 | 0.018 | 7.87 |
| clintox | 1477 | 26.2 | 0.124 | 0.016 | 7.99 |
| sider | 1427 | 33.6 | 0.246 | 0.027 | 8.97 |
| hiv | 9999 | 25.5 | 0.134 | 0.015 | 9.04 |

## Pulapki

**"at least 7.5x" dotyczy srednich rodzin, nie pojedynczych zbiorow.** Najgorszy
pojedynczy zbior to freesolv z **5.6x** - najmniejsze czasteczki w calym
zestawieniu (8.7 atomu). Jesli ktos zapyta "a najgorszy przypadek?", odpowiedz
brzmi 5.6x, nie 7.5x.

**Slupki wygladaja plasko (7.5 / 7.6 / 11.2), ale wasy sa szerokie.** Wczesniejsza
wersja wykresu miala 15 osobnych slupkow i byla nieczytelna ("wszedzie okolo
7-11x"); agregacja do rodzin jest decyzja prezentacyjna, nie pomiarowa.
