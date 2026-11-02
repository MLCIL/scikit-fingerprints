#!/usr/bin/env bash
# Pelny przebieg benchmarku: faza A rownolegle, faza B na pustej maszynie, analiza.
# Odpalac z katalogu glownego repo:  nohup setsid bash benchmark_mordred/run_all.sh &
set -u

# repo i interpreter wyznaczamy jak w pozostalych runnerach, zeby skrypt dzialal
# z dowolnego katalogu i na dowolnej maszynie
REPO=${BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${BENCH_PY:-$([ -x "$REPO/.venv/bin/python" ] && echo "$REPO/.venv/bin/python" || command -v python)}
cd "$REPO"
# PEPDIR mozna podac wprost; domyslnie szukamy repo peptydowego obok tego repo
PEPTIDE_REPO=${PEPTIDE_REPO:-$REPO/../peptides_molecular_fingerprints_classification}
PEPDIR=${PEPDIR:-$PEPTIDE_REPO/data/PeptideReactor}
# BENCH_RES i BENCH_EXTRA sluza do przepuszczenia tego samego drivera na
# mikroskopijnych danych przed prawdziwym przebiegiem, np.
#   BENCH_RES=/tmp/dryrun BENCH_EXTRA="--max-size 30 --repeats 1" bash benchmark_mordred/run_all.sh
RES=${BENCH_RES:-results}
read -r -a EXTRA <<< "${BENCH_EXTRA:-}"
# BENCH_PREACTOR pozwala pominac najdrozszy zbior (hiv_nvp) w przebiegu probnym
read -r -a PREACTOR <<< "${BENCH_PREACTOR:-ace_vaxinpad hiv_lpv hiv_nvp}"
LOGS=$RES/logs
mkdir -p "$LOGS" "$RES/figures"

stamp() { date '+%H:%M:%S'; }
say() { echo "[$(stamp)] $*" | tee -a "$LOGS/run_all.log"; }

# Pomiary bezwzgledne (Faza B tutaj, oraz replika wykresu kolegi w
# bench_muv_replication) wykluczaja sie wzajemnie - puszczone razem daja liczby
# zanizone i nieporownywalne. Blokada na wspolnym pliku pilnuje tego zamiast
# uwagi w README: drugi przebieg czeka, zamiast psuc pierwszy.
LOCK=$HOME/.mordred-bench.lock
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(stamp)] inny przebieg benchmarku trzyma blokade $LOCK - czekam" | tee -a "$LOGS/run_all.log"
    flock 9
fi


say "START | host=$(hostname) | load=$(cut -d' ' -f1-3 /proc/loadavg)"

# ---------------------------------------------------------------- FAZA A
# Ilorazy, wiec kontencja ich nie psuje - 20 procesow jednowatkowych naraz.
say "FAZA A start (11 datasetow MoleculeNet + 10 punktow drabinki peptydowej)"

for d in freesolv esol sider clintox bace bbbp lipophilicity tox21 toxcast hiv muv; do
    $PY benchmark_mordred/bench_run.py --suite speedup --only "$d" \
        "${EXTRA[@]}" --out "$RES/res_speedup_$d.csv" > "$LOGS/speedup_$d.log" 2>&1 &
done

for L in 5 10 20 40 80 100 160; do
    $PY benchmark_mordred/bench_run.py --suite size --only "synthpep_$L" \
        "${EXTRA[@]}" --out "$RES/res_size_synthpep_$L.csv" > "$LOGS/size_synthpep_$L.log" 2>&1 &
done

for P in "${PREACTOR[@]}"; do
    $PY benchmark_mordred/bench_run.py --suite size --only "preactor_$P" \
        --peptide-dir "$PEPDIR" \
        "${EXTRA[@]}" --out "$RES/res_size_preactor_$P.csv" > "$LOGS/size_preactor_$P.log" 2>&1 &
done

wait
say "FAZA A koniec"

# ---------------------------------------------------------------- FAZA B
# Pomiary BEZWZGLEDNE - jeden proces, nic obok. Load zapisujemy do logu, bo
# jesli ktos w tym czasie wejdzie na maszyne, te liczby przestaja byc porownywalne
# z wykresem 10k czasteczek.
say "FAZA B start | load=$(cut -d' ' -f1-3 /proc/loadavg)"

# Osobne wywolanie na implementacje: kazda dostaje wlasna liczebnosc zbioru, tak
# zeby przy 64 workerach zaden nie glodowal (patrz THREADS_COUNTS w bench_run.py).
for impl in ours mordred; do
    $PY benchmark_mordred/bench_run.py --suite threads --impls "$impl" \
        "${EXTRA[@]}" --out "$RES/res_threads.csv" > "$LOGS/threads_$impl.log" 2>&1
    say "  threads/$impl gotowe | load=$(cut -d' ' -f1-3 /proc/loadavg)"
done

$PY benchmark_mordred/bench_run.py --suite reference \
    "${EXTRA[@]}" --out "$RES/res_reference.csv" > "$LOGS/reference.log" 2>&1
say "  reference gotowe | load=$(cut -d' ' -f1-3 /proc/loadavg)"

# ---------------------------------------------------------------- ANALIZA
$PY benchmark_mordred/bench_analyze.py --inputs "$RES/res_*.csv" \
    --outdir "$RES/figures" > "$LOGS/analyze.log" 2>&1
say "KONIEC | wykresy w $RES/figures"
