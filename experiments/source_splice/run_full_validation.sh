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
mkdir -p "$OUTPUT"
if ! python3 -c 'import gin' >/dev/null 2>&1; then
  export PYTHONPATH="$REPO/experiments/tv0_tv4/vendor${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$REPO/experiments/source_splice:$REPO/experiments/tv0_tv4:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
git -C "$REPO" diff --check
python3 -m compileall -q "$REPO/experiments/source_splice"
bash -n "$SELF_DIR/run_source_splice.sh" "$SELF_DIR/run_full_validation.sh"
g++ -O2 -std=c++17 -fPIC -fopenmp -Wall -Wextra -Wpedantic -Werror \
  -c "$REPO/binocmesher/source/source_splice.cpp" \
  -o "$OUTPUT/source_splice_strict.o"
make -C "$REPO/binocmesher" strict-focused \
  > "$OUTPUT/strict_focused.log" 2>&1
bash "$REPO/experiments/p1_p4/run_all.sh" \
  --repo "$REPO" --output "$OUTPUT/p1_p4" --full-official-smoke \
  > "$OUTPUT/p1_p4.console.log" 2>&1
bash "$REPO/experiments/tv0_tv4/run_lightweight_tv0_tv4.sh" \
  --repo "$REPO" --output "$OUTPUT/tv0_tv4" \
  > "$OUTPUT/tv0_tv4.console.log" 2>&1
bash "$SELF_DIR/run_source_splice.sh" \
  --repo "$REPO" --output "$OUTPUT/source_splice" \
  > "$OUTPUT/source_splice.console.log" 2>&1
python3 - "$REPO" "$OUTPUT" <<'PY'
import json, subprocess, sys
from pathlib import Path
repo=Path(sys.argv[1]); root=Path(sys.argv[2])
p1=json.loads((root/'p1_p4/results/validation.json').read_text())
tv=json.loads((root/'tv0_tv4/theory/summary.json').read_text())
splice=json.loads((root/'source_splice/source_splice_validation.json').read_text())
checks={
 'p1_p4': p1.get('pass') is True,
 'tv0_tv4': tv.get('verdict') == 'PASS_TV0_TV4_PRODUCTION_DERIVED_THEORY_VALIDATION',
 'source_face_suppression_boundary_gluing': splice.get('pass') is True,
}
payload={
 'schema':'binoc-source-splice-full-validation-v1',
 'pass':all(checks.values()),
 'verdict':'PASS_CERTIFIED_SOURCE_SPLICE_FULL_VALIDATION' if all(checks.values()) else 'STOP_CERTIFIED_SOURCE_SPLICE_FULL_VALIDATION',
 'checks':checks,
 'head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
 'tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
 'worktree_status':subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True).strip(),
 'source_splice':splice,
}
(root/'FULL_VALIDATION.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(payload['verdict'])
if not payload['pass']: raise SystemExit(2)
PY
echo PASS_CERTIFIED_SOURCE_SPLICE_FULL_VALIDATION
