#!/usr/bin/env bash
# Przeliczenie MolPILE TYLKO nasza implementacja, pod protokolem repo
# benchmarking_molecular_models_private - czyli tym samym, ktorym powstal wykres
# 28 modeli. Dzieki temu liczba jest wprost porownywalna z Chemeleonem, MolE i
# reszta, bez zadnych przeliczen.
#
#   bash run_molpile.sh            # 1 mln czasteczek
#   bash run_molpile.sh 30k        # dry run, 30 tys. (3 pelne paczki)
#
# Mordreda tu NIE liczymy - jego stawke mamy z repliki MUV (25.07 s / 10k z
# pinowanym BLAS-em), a na milionie kosztowalby ponad 4 godziny bez nowej wiedzy.
set -u

# repo wyznaczamy z polozenia tego skryptu, zeby dzialalo na dowolnej maszynie
REPO=${BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${BENCH_PY:-$([ -x "$REPO/.venv/bin/python" ] && echo "$REPO/.venv/bin/python" || command -v python)}

SIZE=${1:-1m}
DATASET="molpile-$SIZE"
OUT=${BENCH_OUT:-$REPO/results/molpile/$SIZE}
LOGS=${BENCH_LOGS:-$OUT/logs}
# 12.9 GB embeddingow na milion czasteczek - poza repo
CLOCK=${BENCH_CLOCK:-$HOME/mordred/molpile/embeddings_$SIZE}

if [ ! -f "data/prepared/$DATASET.joblib" ]; then
    echo "brak data/prepared/$DATASET.joblib - odpal najpierw:" >&2
    echo "  python prepare_molpile.py <liczba_czasteczek>" >&2
    exit 1
fi
if [ -f "$OUT/runtime.csv" ] && [ "${BENCH_APPEND:-0}" != "1" ]; then
    echo "$OUT/runtime.csv juz istnieje - odloz go, wskaz inny BENCH_OUT" >&2
    echo "albo ustaw BENCH_APPEND=1." >&2
    exit 1
fi

mkdir -p "$LOGS" "$OUT" "$CLOCK"
echo "zbior=$DATASET | workery=$($PY -c 'import os;print(int(os.cpu_count()*0.75))') (75% rdzeni, jak get_embed_cpu_njobs)"
echo "  runtime.csv -> $OUT/runtime.csv"
echo "  embeddingi  -> $CLOCK"

# Pomiar bezwzgledny, wiec ta sama blokada co reszta benchmarku.
LOCK=$HOME/.mordred-bench.lock
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "inny przebieg benchmarku trzyma blokade $LOCK - czekam"
    flock 9
fi

echo "=== $(date '+%H:%M:%S') start | load=$(cut -d' ' -f1-3 /proc/loadavg)"
PYTHONPATH=".:model_wrappers/skfp_mordred" nice -n 10 $PY -u measure_embedding.py \
    --multirun "+experiment=skfp_mordred" ++cache=false \
    "+dataset=glob(clf_$DATASET)" \
    "++embedding.runtime_csv=$OUT/runtime.csv" \
    "++embedding.clock_directory=$CLOCK" \
    > "$LOGS/molpile_$SIZE.log" 2>&1
echo "=== $(date '+%H:%M:%S') gotowe (exit $?)"
grep -E "mean=" "$LOGS/molpile_$SIZE.log" | tail -1

$PY metrics_molpile.py "$OUT/runtime.csv" | tee "$OUT/metryki.txt"
