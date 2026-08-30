#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SELF_DIR/../.." && pwd)"
REPO="$DEFAULT_REPO"
OUTPUT=""
INSTALL_DEPS=0
FULL_OFFICIAL=0
ALLOW_DIRTY=0

usage() {
  cat <<'EOF'
Usage: run_all.sh [--repo PATH] --output NEW_PATH [--install-deps]
                  [--full-official-smoke] [--allow-dirty]

The output path must not already exist. The script never deletes a repository
or overwrites an existing result directory. Committed-branch validation
requires a clean worktree unless --allow-dirty is explicitly supplied for a
patch-application smoke.
EOF
}

while (($#)); do
  case "$1" in
    --repo) REPO="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    --full-official-smoke) FULL_OFFICIAL=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT" ]] || { echo "--output is required" >&2; exit 2; }
REPO="$(cd "$REPO" && pwd)"
OUTPUT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT")"
[[ ! -e "$OUTPUT" ]] || { echo "output exists; refusing to overwrite: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT/logs" "$OUTPUT/results" "$OUTPUT/results/focused_bin"

[[ -s "$REPO/binocmesher/source/event_registry.cpp" ]] || { echo "event_registry.cpp missing" >&2; exit 2; }
[[ -s "$REPO/binocmesher/source/event_registry.h" ]] || { echo "event_registry.h missing" >&2; exit 2; }
[[ -s "$REPO/binocmesher/source/hyperpoly_provenance.h" ]] || { echo "hyperpoly_provenance.h missing" >&2; exit 2; }
[[ -s "$REPO/experiments/p1_p4/validate_smoke.py" ]] || { echo "experiment payload missing" >&2; exit 2; }

if ((INSTALL_DEPS)); then
  python3 -m pip install -r "$REPO/experiments/p1_p4/requirements-smoke.txt" \
    2>&1 | tee "$OUTPUT/logs/pip_install.log"
fi

export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
{
  echo "repo=$REPO"
  echo "head=$(git -C "$REPO" rev-parse HEAD)"
  echo "tree=$(git -C "$REPO" rev-parse 'HEAD^{tree}')"
  echo "branch=$(git -C "$REPO" branch --show-current)"
  echo "python=$(python3 --version 2>&1)"
  echo "compiler=$(g++ --version | head -1)"
  git -C "$REPO" status --short
} > "$OUTPUT/logs/environment.log"

make -C "$REPO/binocmesher" clean 2>&1 | tee "$OUTPUT/logs/make_clean.log"
make -C "$REPO/binocmesher" -j2 2>&1 | tee "$OUTPUT/logs/build.log"
sha256sum "$REPO/binocmesher/lib/core.so" | tee "$OUTPUT/results/core_so.sha256"

# 0001 safety regressions.
g++ -O2 -std=c++17 -fopenmp -Wall -Wextra -Wpedantic -Werror \
  -I"$REPO/binocmesher/source" \
  "$REPO/experiments/p1_p4/test_safety_stream_reader.cpp" \
  -o "$OUTPUT/results/focused_bin/test_safety_stream_reader" \
  2>&1 | tee "$OUTPUT/logs/safety_stream_compile.log"
"$OUTPUT/results/focused_bin/test_safety_stream_reader" \
  2>&1 | tee "$OUTPUT/logs/safety_stream_run.log"

g++ -O2 -std=c++17 -fopenmp -Wall -Wextra -Wpedantic -Werror \
  -I"$REPO/binocmesher/source" \
  "$REPO/experiments/p1_p4/test_safety_cache_rational_cabi.cpp" \
  -L"$REPO/binocmesher/lib" -l:core.so \
  -Wl,-rpath,"$REPO/binocmesher/lib" \
  -o "$OUTPUT/results/focused_bin/test_safety_cache_rational_cabi" \
  2>&1 | tee "$OUTPUT/logs/safety_cache_compile.log"
"$OUTPUT/results/focused_bin/test_safety_cache_rational_cabi" \
  2>&1 | tee "$OUTPUT/logs/safety_cache_run.log"

# 0002 provenance schema/join regressions.
python3 "$REPO/experiments/p1_p4/test_provenance_v2.py" --self-test \
  2>&1 | tee "$OUTPUT/logs/provenance_self_test.log"

g++ -O2 -std=c++17 -fopenmp -Wall -Wextra -Wpedantic -Werror \
  -I"$REPO/binocmesher/source" \
  "$REPO/experiments/p1_p4/test_processed_triple_stream.cpp" \
  -o "$OUTPUT/results/focused_bin/test_processed_triple_stream" \
  2>&1 | tee "$OUTPUT/logs/processed_triple_compile.log"
"$OUTPUT/results/focused_bin/test_processed_triple_stream" \
  2>&1 | tee "$OUTPUT/logs/processed_triple_run.log"

python3 - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
logs = root / "logs"
checks = {
    "safety_stream": "PASS_SAFETY_STREAM_READER" in (logs / "safety_stream_run.log").read_text(),
    "safety_cache_rational_cabi": "PASS_SAFETY_CACHE_RATIONAL_C_ABI" in (logs / "safety_cache_run.log").read_text(),
    "provenance_v2_self_test": "PASS_PROVENANCE_V2_SELF_TEST" in (logs / "provenance_self_test.log").read_text(),
    "processed_triple_stream": "PASS_PROCESSED_PRIMARY_DISCON_BPM2_TRIPLE_STREAM" in (logs / "processed_triple_run.log").read_text(),
}
payload = {"checks": checks, "pass": all(checks.values())}
(root / "results/focused_checks.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
if not payload["pass"]:
    raise SystemExit(2)
PY

python3 "$REPO/experiments/p1_p4/run_core_fixture_smoke.py" \
  --core "$REPO/binocmesher/lib/core.so" \
  --output "$OUTPUT/results/core_fixtures" \
  2>&1 | tee "$OUTPUT/logs/core_fixture.log"

BINOC_P1_P4_REFERENCE_OUT="$OUTPUT/results/reference" \
  python3 "$REPO/experiments/p1_p4/reference/run_p1_p4_reference.py" \
  2>&1 | tee "$OUTPUT/logs/reference.log"
python3 "$REPO/experiments/p1_p4/reference/validate_reference.py" "$OUTPUT/results/reference" \
  2>&1 | tee "$OUTPUT/logs/reference_validate.log"

if ((FULL_OFFICIAL)); then
  python3 "$REPO/experiments/p1_p4/run_official_p1_smoke.py" \
    --repo "$REPO" \
    --output "$OUTPUT/results/official_p1" \
    2>&1 | tee "$OUTPUT/logs/official_p1.log"
fi

VALIDATE_ARGS=(--repo "$REPO" --results "$OUTPUT")
if ((FULL_OFFICIAL)); then VALIDATE_ARGS+=(--require-official); fi
if ((ALLOW_DIRTY)); then VALIDATE_ARGS+=(--allow-dirty); fi
python3 "$REPO/experiments/p1_p4/validate_smoke.py" "${VALIDATE_ARGS[@]}" \
  2>&1 | tee "$OUTPUT/logs/final_validate.log"
