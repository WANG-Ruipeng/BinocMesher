#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/../.." && pwd)"
OUTPUT=""
usage(){ echo "usage: $0 --output NEW_DIRECTORY [--repo PATH]"; }
while (($#)); do
  case "$1" in
    --repo) REPO="$(cd "$2" && pwd)"; shift 2;;
    --output) OUTPUT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$OUTPUT" ]] || { usage >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "output exists: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT/logs"
if ! python3 -c 'import gin' >/dev/null 2>&1; then
  export PYTHONPATH="$SELF_DIR/vendor${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$SELF_DIR:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
make -C "$REPO/binocmesher" -j2 2>&1 | tee "$OUTPUT/logs/build.log"
python3 "$SELF_DIR/run_lightweight_profile.py" \
  --repo "$REPO" --output "$OUTPUT/cache" --cameras 24 --ppc 180 \
  --profile demo --coarse 360 --outview 720 --bisection-iters 1 \
  --seed-stride 2 --fading-time 0.041666666666666664 \
  --n-coarse-nodes 200000 --medium-group 20000 --fine-group 5000 \
  --bisection-group 1000000 2>&1 | tee "$OUTPUT/logs/profile.log"
python3 "$SELF_DIR/run_tv0_tv4.py" \
  --cache-root "$OUTPUT/cache" --output "$OUTPUT/theory" \
  2>&1 | tee "$OUTPUT/logs/tv0_tv4.log"
python3 "$SELF_DIR/validate_tv0_tv4.py" \
  --results "$OUTPUT/theory" --cache-root "$OUTPUT/cache" \
  2>&1 | tee "$OUTPUT/logs/validate.log"
echo PASS_LIGHTWEIGHT_TV0_TV4_FROM_FRESH_CACHE
