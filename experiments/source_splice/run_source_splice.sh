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
  export PYTHONPATH="$REPO/experiments/tv0_tv4/vendor${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$SELF_DIR:$REPO/experiments/tv0_tv4:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
make -C "$REPO/binocmesher" -j2 2>&1 | tee "$OUTPUT/logs/build.log"
python3 "$REPO/experiments/tv0_tv4/run_lightweight_profile.py" \
  --repo "$REPO" --output "$OUTPUT/cache" --cameras 24 --ppc 180 \
  --profile demo --coarse 360 --outview 720 --bisection-iters 1 \
  --seed-stride 2 --fading-time 0.041666666666666664 \
  --n-coarse-nodes 200000 --medium-group 20000 --fine-group 5000 \
  --bisection-group 1000000 2>&1 | tee "$OUTPUT/logs/profile.log"
python3 "$SELF_DIR/compile_splice_plans.py" \
  --cache-root "$OUTPUT/cache" --output "$OUTPUT/plans" \
  2>&1 | tee "$OUTPUT/logs/compile_plans.log"
read -r TIME_NUM TIME_DEN < <(python3 - "$OUTPUT/plans/plan_metadata.json" "$OUTPUT/probe.json" <<'PYPROBE'
import json, sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())
num, den = (int(value) for value in metadata["probe_time"].split("/", 1))
Path(sys.argv[2]).write_text(json.dumps({
    "schema": "binoc-source-splice-probe-v1",
    "event_id": metadata["event_id"],
    "event_root": metadata["event_root"],
    "probe_side": metadata["probe_side"],
    "probe_time": {"numerator": num, "denominator": den},
}, indent=2, sort_keys=True) + "\n")
print(num, den)
PYPROBE
)
python3 "$SELF_DIR/run_splice_experiment.py" \
  --repo "$REPO" --cache-root "$OUTPUT/cache" \
  --plans "$OUTPUT/plans" --output "$OUTPUT/runtime" \
  --time-num "$TIME_NUM" --time-den "$TIME_DEN" \
  2>&1 | tee "$OUTPUT/logs/runtime.log"
python3 "$SELF_DIR/validate_source_splice.py" \
  --runtime-results "$OUTPUT/runtime" --plans "$OUTPUT/plans" \
  --output "$OUTPUT/source_splice_validation.json" \
  2>&1 | tee "$OUTPUT/logs/validate.log"
echo PASS_SOURCE_FACE_SUPPRESSION_AND_BOUNDARY_GLUING_FROM_FRESH_CACHE
