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
python3 "$SELF_DIR/test_critical_beb1_event_ir.py" > "$OUTPUT/critical_beb1_event_ir_test.log" 2>&1
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
python3 "$SELF_DIR/compile_critical_beb1_event_ir.py" \
  --cache-root "$OUTPUT/tv0_tv4/cache" \
  --theory-root "$OUTPUT/tv0_tv4/theory" \
  --output "$OUTPUT/critical_beb1_event_ir.json" \
  --plan-output "$OUTPUT/critical_beb1.ssp1" \
  --expected-root 104/5 \
  --require-whole-mesh-ready \
  > "$OUTPUT/critical_beb1_event_ir.console.log" 2>&1
python3 "$SELF_DIR/run_critical_beb1_splice.py" \
  --repo "$REPO" \
  --cache-root "$OUTPUT/tv0_tv4/cache" \
  --event-ir "$OUTPUT/critical_beb1_event_ir.json" \
  --plan "$OUTPUT/critical_beb1.ssp1" \
  --output "$OUTPUT/critical_beb1_runtime" \
  > "$OUTPUT/critical_beb1_runtime.console.log" 2>&1
python3 "$SELF_DIR/validate_critical_beb1_splice.py" \
  --runtime-results "$OUTPUT/critical_beb1_runtime" \
  --event-ir "$OUTPUT/critical_beb1_event_ir.json" \
  --output "$OUTPUT/critical_beb1_validation.json" \
  > "$OUTPUT/critical_beb1_validation.console.log" 2>&1
python3 "$SELF_DIR/run_all_canonical_beb1_events.py" \
  --repo "$REPO" \
  --cache-root "$OUTPUT/tv0_tv4/cache" \
  --output "$OUTPUT/all_canonical_beb1" \
  --expected-events 4 \
  --expected-profile demo \
  > "$OUTPUT/all_canonical_beb1.console.log" 2>&1
bash "$SELF_DIR/run_source_splice.sh" \
  --repo "$REPO" --output "$OUTPUT/source_splice" \
  > "$OUTPUT/source_splice.console.log" 2>&1
python3 - "$REPO" "$OUTPUT" <<'PY'
import json, subprocess, sys
from pathlib import Path
repo=Path(sys.argv[1]); root=Path(sys.argv[2])
p1=json.loads((root/'p1_p4/results/validation.json').read_text())
tv=json.loads((root/'tv0_tv4/theory/summary.json').read_text())
beb1=json.loads((root/'critical_beb1_event_ir.json').read_text())
beb1_runtime=json.loads((root/'critical_beb1_validation.json').read_text())
all_beb1=json.loads((root/'all_canonical_beb1/all_canonical_beb1_summary.json').read_text())
splice=json.loads((root/'source_splice/source_splice_validation.json').read_text())
checks={
 'p1_p4': p1.get('pass') is True,
 'tv0_tv4': tv.get('verdict') == 'PASS_TV0_TV4_PRODUCTION_DERIVED_THEORY_VALIDATION',
 'critical_beb1_event_ir': beb1.get('verdict') == 'PASS_CRITICAL_BEB1_EVENT_IR',
 'critical_beb1_event_star_closed': (
     beb1.get('whole_mesh_splice_ready') is True
     and beb1.get('runtime_disposition') == 'READY_FOR_WHOLE_MESH_SPLICE'
     and beb1.get('admission', {}).get('mapping_cylinder_ready') is True
     and beb1.get('event_star_geometry', {}).get(
         'mapping_cylinder', {}).get('critical_side_edges_remaining') == 0
 ),
 'critical_beb1_whole_mesh_runtime': beb1_runtime.get('verdict') == 'PASS_CRITICAL_BEB1_WHOLE_MESH_SPLICE',
 'all_canonical_beb1_events': (
     all_beb1.get('verdict') == 'PASS_ALL_CANONICAL_BEB1_WHOLE_MESH'
     and all_beb1.get('coverage', {}).get(
         'canonical_event_fraction') == '4/4'
     and all_beb1.get('coverage', {}).get('runtime_validated') == 4
 ),
 'source_face_suppression_boundary_gluing': splice.get('pass') is True,
}
payload={
 'schema':'binoc-source-splice-full-validation-v3',
 'pass':all(checks.values()),
 'verdict':'PASS_CERTIFIED_SOURCE_SPLICE_FULL_VALIDATION' if all(checks.values()) else 'STOP_CERTIFIED_SOURCE_SPLICE_FULL_VALIDATION',
 'checks':checks,
 'head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
 'tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
 'worktree_status':subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True).strip(),
 'source_splice':splice,
 'critical_beb1_event_ir':beb1,
 'critical_beb1_whole_mesh_validation':beb1_runtime,
 'all_canonical_beb1_campaign':all_beb1,
}
(root/'FULL_VALIDATION.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(payload['verdict'])
if not payload['pass']: raise SystemExit(2)
PY
echo PASS_CRITICAL_BEB1_EVENT_STAR_CLOSURE
echo PASS_CRITICAL_BEB1_WHOLE_MESH_SPLICE
echo PASS_ALL_CANONICAL_BEB1_WHOLE_MESH
echo PASS_CERTIFIED_SOURCE_SPLICE_FULL_VALIDATION
