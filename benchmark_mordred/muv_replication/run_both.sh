#!/usr/bin/env bash
# Pomiar na calym ogbg-molmuv pod protokolem repo benchmarking_molecular_models_private:
# mordred (kontrola) i nasza implementacja, sekwencyjnie, nice -n 10.
#
#   bash run_both.sh            # proba 1: jego protokol, BLAS niepinowany
#   bash run_both.sh pinned     # proba 2: BLAS pinowany do 1 watku na worker
#
# Kazdy tryb pisze do wlasnego katalogu w results/muv_replication/, zeby proby
# stanely obok siebie i bylo widac, ktora jest ktora. Istniejacego runtime.csv
# skrypt nie nadpisze - albo wskaz inny BENCH_OUT, albo BENCH_APPEND=1.
set -u

# repo wyznaczamy z polozenia tego skryptu, zeby dzialalo na dowolnej maszynie
REPO=${BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${BENCH_PY:-$([ -x "$REPO/.venv/bin/python" ] && echo "$REPO/.venv/bin/python" || command -v python)}

MODE=${1:-defaults}
case "$MODE" in
    defaults) PIN_DEF=0; OUT_DEF=$REPO/results/muv_replication/proba1_defaults ;;
    pinned)   PIN_DEF=1; OUT_DEF=$REPO/results/muv_replication/proba2_blas_pinned ;;
    *) echo "uzycie: bash run_both.sh [defaults|pinned]" >&2; exit 1 ;;
esac

PIN=${BENCH_PIN_BLAS:-$PIN_DEF}
OUT=${BENCH_OUT:-$OUT_DEF}
LOGS=${BENCH_LOGS:-$OUT/logs}
# embeddingi to produkt uboczny harnessu, 1.2 GB na model - trzymamy je poza repo
CLOCK=${BENCH_CLOCK:-$HOME/mordred/muv_embeddings_tmp/$MODE}

# BLAS czyta te zmienne RAZ, przy ladowaniu biblioteki, wiec musza byc ustawione
# przed startem Pythona. Ustawione w kodzie po "import numpy" nie robia nic -
# zmierzone: cpu/wall wychodzi wtedy 11.5 zamiast 1.0.
if [ "$PIN" = "1" ]; then
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
fi

if [ -f "$OUT/runtime.csv" ] && [ "${BENCH_APPEND:-0}" != "1" ]; then
    echo "$OUT/runtime.csv juz istnieje - to wyniki poprzedniego przebiegu." >&2
    echo "Odloz je, wskaz inny BENCH_OUT, albo ustaw BENCH_APPEND=1." >&2
    exit 1
fi

mkdir -p "$LOGS" "$OUT" "$CLOCK"
echo "tryb=$MODE | pinowanie BLAS=$PIN"
echo "  runtime.csv -> $OUT/runtime.csv"
echo "  logi        -> $LOGS"
echo "  embeddingi  -> $CLOCK"

# Ta sama blokada, ktorej uzywa benchmark_mordred/run_all.sh. Ten pomiar i Faza B tamtego
# wykluczaja sie wzajemnie - oba sa bezwzgledne.
LOCK=$HOME/.mordred-bench.lock
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "inny przebieg benchmarku trzyma blokade $LOCK - czekam"
    flock 9
fi

for MODEL in mordred skfp_mordred; do
    echo "=== $(date '+%H:%M:%S') $MODEL | load=$(cut -d' ' -f1-3 /proc/loadavg)"
    PYTHONPATH=".:model_wrappers/$MODEL" nice -n 10 $PY -u measure_embedding.py \
        --multirun "+experiment=$MODEL" ++cache=false \
        '+dataset=glob(clf_ogbg-molmuv)' \
        "++embedding.runtime_csv=$OUT/runtime.csv" \
        "++embedding.clock_directory=$CLOCK" \
        > "$LOGS/muv_$MODEL.log" 2>&1
    echo "=== $(date '+%H:%M:%S') $MODEL gotowe (exit $?)"
    grep -E "mean=" "$LOGS/muv_$MODEL.log" | tail -1
done

echo "=== $OUT/runtime.csv ==="
cat "$OUT/runtime.csv"
