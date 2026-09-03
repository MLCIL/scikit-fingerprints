# Benchmark: przepisany Mordred vs mordred-community

Dwa skrypty: `bench_run.py` mierzy i zapisuje CSV, `bench_analyze.py` robi z tego
wykresy. Wszystkie pomiary są **jednowątkowe** poza zestawem `threads`.

## 0. Przygotowanie serwera

```bash
# repo skfp na branchu docelowym (opt-17 + kappa + IC)
cd scikit-fingerprints
git checkout <branch-benchmarkowy>
pip install -e .            # albo uv sync
pip install mordred matplotlib pandas

# dane peptydowe (136 MB)
git clone --depth 1 https://github.com/scikit-fingerprints/peptides_molecular_fingerprints_classification.git
export PEPDIR="$PWD/peptides_molecular_fingerprints_classification/data/PeptideReactor"

# skrypty benchmarkowe
mkdir -p ~/bench && cd ~/bench      # tu wrzuć bench_run.py i bench_analyze.py
```

Sprawdź, czy skfp się importuje i czy to właściwy branch — skrypt zapisuje
`branch@commit` do każdego wiersza CSV, więc później widać, co mierzyłeś.

## 1. Kalibracja — ZAWSZE NAJPIERW

```bash
python bench_run.py --suite calib --out calib.csv
```

Trwa kilka minut. Mierzy tempo **tej** maszyny i wypisuje projekcję budżetu dla
wszystkich wykresów wraz z werdyktem, czy mieści się w 12 h. Jeśli nie —
podaje listę, co przyciąć. **Dopiero ta projekcja jest wiążąca**; wszelkie
wcześniejsze szacunki z innych maszyn są bezwartościowe.

## 2. Faza A — równolegle

Kontencja nie psuje **ilorazów** (obie implementacje cierpią tak samo), więc tu
można puścić wiele procesów naraz.

```bash
# wykres 1: speedup na MoleculeNet
for d in freesolv esol sider clintox bace bbbp lipophilicity tox21 toxcast hiv muv; do
    python bench_run.py --suite speedup --only $d --out res_speedup_$d.csv &
done

# wykresy 2+3: drabinka rozmiaru - peptydy generowane
for L in 5 10 20 40 80 100; do
    python bench_run.py --suite size --only synthpep_$L --out res_size_$L.csv &
done

# wykresy 2+3: realne peptydy z papera
for P in ace_vaxinpad hiv_lpv hiv_nvp; do
    python bench_run.py --suite size --only preactor_$P \
        --peptide-dir "$PEPDIR" --out res_size_$P.csv &
done
wait
```

**Długi biegun:** `preactor_hiv_nvp` (2 cząsteczki po ~2000 atomów) to najdłuższy
pojedynczy proces. Jeśli budżet nie domyka, daj mu `--repeats 1` — koszt jest
deterministyczny (O(n³)), więc przy tym rozmiarze powtórzenia nie niosą informacji.

## 3. Faza B — maszyna MUSI być pusta

To pomiary **bezwzględne** (sekundy, nie ilorazy). Pod obciążeniem wychodzą
zaniżone i nieporównywalne z wykresem kolegi.

```bash
python bench_run.py --suite threads   --out res_threads.csv
python bench_run.py --suite reference --out res_reference.csv
```

## 4. Analiza

```bash
python bench_analyze.py --inputs "res_*.csv" --outdir figures
```

Generuje `figures/1_speedup_datasets.png` … `4_threads.png`, tabelę `summary.csv`
oraz wypisuje na konsolę liczbę do wykresu kolegi (sekundy / 10 000 cząsteczek).
Można odpalać **w trakcie** przebiegu — rysuje to, na co już są dane.

## Przydatne flagi

| flaga | do czego |
| --- | --- |
| `--max-size N` | twardy limit cząsteczek; **nadpisuje wszystko**, dobre do testów |
| `--repeats N` | domyślnie 3 |
| `--only DATASET` | jeden dataset na proces (faza A) |
| `--impls ours` | zmierz tylko naszą implementację |
| `--thread-counts 1 2 4 8` | nadpisz liczby wątków dla `threads` |
| `--seed N` | ziarno losowania podpróbek (domyślnie 0) |

## ⚠️ Commitowanie wyników

`.gitignore` tego repo wyrzuca `*.csv`, `*.png`, `*.svg` i `*.pdf`. Wyniki
przebiegu i wykresy **nie zostaną dodane zwykłym `git add`** — po 12 h liczenia
łatwo to przeoczyć i wrócić z serwera z samymi skryptami.

Przy commitowaniu wyników wymuś dodanie:

```bash
git add -f res_*.csv figures/*.png
git commit -m "Wyniki benchmarku z serwera"
```

Sprawdź przed pushem, że faktycznie weszły:

```bash
git show --stat HEAD
```

## Rzeczy, które mogą zaskoczyć

**Wznawialność.** Restart z tym samym `--out` pomija już zmierzone kombinacje.
Można bezpiecznie przerwać Ctrl+C i wrócić później. Zapis idzie z `fsync()`
po każdym pomiarze, więc nawet ubicie procesu nic nie gubi.

**Wolny start na HIV/MUV.** Skrypt parsuje 10 000 cząsteczek **zanim** zadziała
`--max-size`. Przy testach wygląda to jak zawieszenie — to normalne.

**Zbyt małe N w `threads` daje bezsens.** Przy kilkudziesięciu cząsteczkach
dominuje koszt startu procesów i krzywa idzie w dół. Dlatego domyślnie 2000.

**Zakres.** Deskryptory 3D / GEOM są świadomie poza zakresem — oryginalny Mordred
ma zepsute liczenie konformerów i uczciwy pomiar wymaga osobnego podejścia.
