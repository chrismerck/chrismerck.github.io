#!/bin/sh
# Everything, in order.  Roughly two hours on four cores, plus ~4 GB of
# model downloads into $EMBEDALIGN_DATA (default /tmp/embedalign).
#
#   ./run_all.sh [results-dir]
set -e
OUT="${1:-/tmp/embedalign-results}"
mkdir -p "$OUT"
cd "$(dirname "$0")"

echo "== sanity ladder (nothing below matters if this is not ~100%)"
python3 sanity.py --n 2000 --out "$OUT/sanity.json" 2>&1 | tee "$OUT/sanity.log"

echo "== survey, method comparison, sweeps, positional, open setting"
python3 run_all.py --out "$OUT" 2>&1 | tee "$OUT/run.log"

echo "== the whole shared vocabulary, up to 44k tokens"
python3 scale.py --out "$OUT/scale.json" 2>&1 | tee "$OUT/scale.log"

echo "== restart behaviour around the threshold"
python3 restarts.py --out "$OUT/restarts.json" 2>&1 | tee "$OUT/restarts.log"

echo "== script confound and postprocessing ablation"
python3 extras.py --out "$OUT/extras.json" 2>&1 | tee "$OUT/extras.log"

echo "== figures"
python3 figures.py --results "$OUT"
