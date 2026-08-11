#!/bin/sh
# Everything, in the order the results are argued about in the write-up.
# Roughly two hours on four cores.
set -e
cd "$(dirname "$0")"
OUT=${1:-/tmp/results}
mkdir -p "$OUT"

echo "=== 0. sanity ladder (synthetic + rotated real vectors) ==="
# If this doesn't come back at 100% for the rotation-invariant methods,
# stop here -- nothing below it means anything.
python3 test_sanity.py --n 2000 --real-n 2000 --out "$OUT/sanity.json"

echo "=== 1. the headline: off-the-shelf vectors, 20k vocabulary ==="
python3 run_embeddings.py --vocab 20000 --match-vocab 3000 --gw-vocab 2000 \
    --sl-cut 6000 --out "$OUT/embeddings-20k.json"

echo "=== 2. the variable that decided it: vocabulary size ==="
python3 vocab_sweep.py --results "$OUT"

echo "=== 3. how alike are the two shapes, and how much distortion survives ==="
python3 distortion.py --out "$OUT/distortion.json"

echo "=== 4. count vectors we build ourselves, four corpus conditions ==="
for cond in europarl-par europarl-comp ud-indep de-de-split; do
  python3 run_experiment.py "$cond" --vocab 2000 --dim 200 \
      --gw-outer 80 --gw-init 2 --restarts 3 \
      --out "$OUT/counts-$cond.json"
done

echo "=== 5. figures ==="
python3 figures.py --results "$OUT"

echo "all done"
