#!/bin/sh
# Everything, in the order the results are argued about in the write-up.
# Roughly 90 minutes on four cores.
set -e
cd "$(dirname "$0")"
OUT=${1:-/tmp/results}
mkdir -p "$OUT"

echo "=== 0. sanity ladder (synthetic + rotated real vectors) ==="
python3 test_sanity.py --n 2000 --real-n 2000 --out "$OUT/sanity.json"

echo "=== 1. distortion axis: where do the real cases land? ==="
python3 distortion.py --out "$OUT/distortion.json"

echo "=== 2. count vectors, four corpus conditions ==="
for cond in europarl-par europarl-comp ud-indep de-de-split; do
  python3 run_experiment.py "$cond" --vocab 2000 --dim 200 \
      --gw-outer 80 --gw-init 2 --restarts 3 \
      --out "$OUT/counts-$cond.json"
done

echo "=== 3. off-the-shelf vectors, large vocabulary ==="
python3 run_embeddings.py --vocab 20000 --match-vocab 3000 --gw-vocab 2000 \
    --sl-cut 6000 --out "$OUT/embeddings-20k.json"

echo "=== 4. how much text do you need? ==="
python3 sweep.py --out "$OUT/sweep.json"

echo "all done"
