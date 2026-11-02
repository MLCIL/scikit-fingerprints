#!/usr/bin/env bash
# Benchmark na rodzinie ExpansionRx / OpenADMET - dziesiec realnych zbiorow ADME.
#
#   bash benchmark_mordred/run_expansionrx.sh
#
# PO CO TEN ZESTAW. Reszta benchmarku bada skrajnosci rozmiaru: MoleculeNet daje
# 9-34 atomy, drabinka peptydowa 43-1994. ExpansionRx to realne dane lekopodobne
# o srednio 28.6 atomu, ale z CIASNYM rozkladem - mediana 28, maksimum 78, gdy HIV
# ma 214. To wazne, bo na HIV mordred przy 64 workerach wyciaga 42 mol/s, a na MUV
# (maksimum 42 atomy) 400 mol/s. Jesli za ta roznica stoi ciezki ogon rozkladu, to
# ExpansionRx powinien zachowac sie jak MUV, mimo ze ma wieksze czasteczki.
#
# FAZA A - dziesiec zbiorow ADME, ilorazy, dziesiec procesow jednowatkowych
# rownolegle; kontencja ilorazow nie psuje. FAZA B - zbior ring-rich, dopiero po
# zakonczeniu Fazy A, zeby liczyl sie na wolnej maszynie.
set -u

# repo i interpreter wyznaczamy jak w pozostalych runnerach, zeby skrypt dzialal
# z dowolnego katalogu i na dowolnej maszynie
REPO=${BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${BENCH_PY:-$([ -x "$REPO/.venv/bin/python" ] && echo "$REPO/.venv/bin/python" || command -v python)}
cd "$REPO"
RES=${BENCH_RES:-results/expansionrx}
read -r -a EXTRA <<< "${BENCH_EXTRA:-}"
LOGS=$RES/logs
mkdir -p "$LOGS" "$RES/figures"

stamp() { date '+%H:%M:%S'; }
say() { echo "[$(stamp)] $*" | tee -a "$LOGS/run.log"; }

# Ta sama blokada, ktorej uzywaja run_all.sh i replika MUV - Faza B jest
# bezwzgledna i nie moze zachodzic z innymi pomiarami.
LOCK=$HOME/.mordred-bench.lock
exec 9>"$LOCK"
if ! flock -n 9; then
    say "inny przebieg benchmarku trzyma blokade $LOCK - czekam"
    flock 9
fi

say "START | host=$(hostname) | load=$(cut -d' ' -f1-3 /proc/loadavg)"

# ---------------------------------------------------------------- FAZA A
say "FAZA A: 10 zbiorow ExpansionRx, jednowatkowo, 10 procesow rownolegle"

for D in erx_logd erx_ksol erx_mlm_clint erx_hlm_clint erx_caco2_efflux \
         erx_caco2_papp erx_mppb erx_mbpb erx_rlm_clint erx_mgmb; do
    $PY benchmark_mordred/bench_run.py --suite expansionrx --only "$D" \
        "${EXTRA[@]}" --out "$RES/res_$D.csv" > "$LOGS/$D.log" 2>&1 &
done

wait
say "FAZA A koniec"

# ---------------------------------------------------------------- FAZA B
# Zbior ring-rich: 10 000 czasteczek z co najmniej 4 pierscieniami (CalcNumRings),
# pierwsze 10k takich z MolPILE. Jednowatkowo, po Fazie A, na wolnej maszynie.
# Probka 200 czasteczek dala 34.7 atomu, 4.9 pierscienia i 8.19x - czyli tyle, ile
# bace (34.1 at, 7.87x) i sider (33.6 at, 8.97x) o tej samej wielkosci, wiec samo
# bogactwo pierscieni nie daje wyraznie wiekszego przyspieszenia.
RINGS_CSV=${BENCH_RINGS_CSV:-$HOME/dataset_rings.csv}
if [ -f "$RINGS_CSV" ]; then
    say "FAZA B: ring-rich ($RINGS_CSV), jednowatkowo | load=$(cut -d' ' -f1-3 /proc/loadavg)"
    $PY benchmark_mordred/bench_run.py --suite expansionrx --only "csv:$RINGS_CSV" \
        "${EXTRA[@]}" --out "$RES/res_rings.csv" > "$LOGS/rings.log" 2>&1
    say "FAZA B koniec"
else
    say "FAZA B pominieta: nie ma pliku $RINGS_CSV"
fi

# ---------------------------------------------------------------- ANALIZA
$PY benchmark_mordred/bench_analyze.py --inputs "$RES/res_*.csv" \
    --outdir "$RES/figures" --label ExpansionRx > "$LOGS/analyze.log" 2>&1
say "KONIEC | wykresy w $RES/figures"
