# Specyfikacje eksperymentow z postera

**[methodology_en/](methodology_en/README.md)** - metodologia po angielsku, plik na
eksperyment, z preprocessingiem i wykluczeniami. To jest wersja do obrony na
konferencji. Pliki ponizej sa robocze, po polsku.

Poster: `~/poster mordred (5).pdf` (RDKit UGM).
Jeden plik na eksperyment, w kolejnosci wystepowania na posterze.

| plik | element postera | glowna liczba |
|---|---|---|
| [01](01_tabela_rodzin_deskryptorow.md) | tabela **Individual descriptors speedup** | 3.0x - 9.3x per rodzina |
| [02](02_qsar_qspr_datasets.md) | wykres **QSAR/QSPR datasets** | 7.5x / 7.6x / 11.2x |
| [03](03_skalowanie_rozmiarem.md) | wykres **Molecule size scaling** | 5.6x przy 140 Da, 13.9x przy 11 kDa |
| [04](04_skalowanie_rownolegle.md) | wykres **Parallel scaling** | 32x (leki), 27x (peptydy) |
| [05](05_molpile_przepustowosc.md) | sekcja **At pretraining scale** | 3 800 mol/s, 4.5 min / 1 mln |
| [06](06_porownanie_z_embeddingami.md) | wykres **Throughput vs other embeddings** | 2 485 mol/s vs 53 |
| [07](07_kontekst_wykonania_wrzesien.md) | **kontekst wykonania** (wrzesien 2026) | skad wziely sie decyzje |

## Wspolne dla wszystkich

**Maszyna.** `bitai2`: AMD EPYC 7763, 64 rdzenie fizyczne / 128 watkow, 503 GB RAM.
Przy obciazeniu wszystkich 64 rdzeni procesor zbija takt z ~3.5 GHz do ~2.45 GHz,
co widac w efektywnosci skalowania.

**Kod.** Branch `mordred-benchmarks`, commit `0bec710` (zapisany w kazdym wierszu
CSV w kolumnie `commit`). Oryginal: **mordredcommunity 2.0.7** - pakiet
`JacksonBurns/mordred-community`, importowany jako `mordred`. Oryginalny `mordred`
Moriwakiego (1.2.0) jest nierozwijany i nie dziala z aktualnym RDKit.

**Pinowanie BLAS.** Wszystkie pomiary poza wyraznie zaznaczonymi maja
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` ustawione **przed importem
numpy** (`bench_run.py:65-71`). Bez tego mordred jest sztucznie **7.3x wolniejszy**
przez nadsubskrypcje watkow, przy bit-identycznym wyniku. To jedyny najwiekszy
czynnik zakrzywiajacy porownania i wraca w pliku [06](06_porownanie_z_embeddingami.md).

**Protokol dwufazowy.** Faza A to **ilorazy** - kontencja nie psuje ilorazu, bo obie
strony cierpia tak samo, wiec mozna puszczac wiele procesow naraz. Faza B to pomiary
**bezwzgledne** (sekundy, przepustowosc) i wymaga pustej maszyny.

**Zakres.** Wszystko na posterze dotyczy **1613 deskryptorow 2D**. Deskryptory 3D sa
swiadomie poza zakresem benchmarku (wymagaja konformera, co jest osobnym problemem
metodologicznym - `benchmark_mordred/README.md`).

## Gdzie sa dane

Dane i skrypty **nie sa na branchu roboczym** - siedza na `mordred-benchmarks`:

```bash
cd ~/mordred/scikit-fingerprints
git worktree add /tmp/bench --detach mordred-benchmarks
ls /tmp/bench/results/ /tmp/bench/benchmark_mordred/
```

`.gitignore` repo wyrzuca `*.csv` i `*.png`, wiec wyniki weszly przez `git add -f`.

## Weryfikacja postera wobec danych (2026-11-01)

Kazda liczba z postera sprawdzona wprost w CSV. **Zgadzaja sie wszystkie poza
trzema ponizej.**

### 1. "4.5 min" i "3 800 mol/s" sa z roznych przebiegow

| zrodlo | mol/s | czas na 996 819 czasteczek |
|---|---|---|
| `throughput.json` (pierwsza proba) | 3 658 | **4.54 min** |
| srednia z `throughput_96w_run{1,2,3}.json` | **3 800** | **4.37 min** |

Poster bierze czas z pierwszej proby, a przepustowosc ze sredniej. Rozbieznosc 4%:
3 800 mol/s przez 4.5 min dawaloby 1 026 000 czasteczek zamiast 996 819.
Spojne byloby **4.4 min + 3 800 mol/s** albo **4.5 min + 3 658 mol/s**.

### 2. "1M mols" to 996 819

Odfiltrowano **3 181 czasteczek (0.32%)** - fullereny i fragmenty grafenowe, na
ktorych `DetourMatrix` nie konczy liczenia. Poster nie wspomina o filtrze.
Rachunkowo bez znaczenia, ale to pytanie, ktore moze paść.

### 3. "53 for the original" - pomiar bez pinowania BLAS

Uczciwy iloraz wobec tej samej binarki to **6.2x**, nie 47x. Pelny opis w
[06](06_porownanie_z_embeddingami.md).

### Zweryfikowane jako poprawne

1613 deskryptorow; "~0.1 s per molecule" (mediana mordreda na MoleculeNet 0.12 s);
BACE 1513; 7.5x jako najnizsza ze srednich rodzin; 5.6x przy 138 Da i 13.9x przy
10 886 Da; 32.0x i 27.4x; 3 800 mol/s; 16.2 h dla 222 mln; 2 485 mol/s.

**"better throughput than all neural networks"** - potwierdzone na 61 modelach:
szybsze od nas (4.02 s / 10k) sa wylacznie klasyczne fingerprinty haszowane
(ECFP 0.71, TT 0.73, TT_count 0.77, ECFP_count 0.77, AtomPair_count 0.82,
AtomPair 1.27 s). Najszybszy model neuronowy to `mole` - 6.37 s.
