#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SELF_DIR/../.." && pwd)"
REPO="$DEFAULT_REPO"
OUTPUT=""
INSTALL_DEPS=0
FULL_OFFICIAL=0

usage() {
  cat <<'EOF'
Usage: run_all.sh [--repo PATH] --output NEW_PATH [--install-deps] [--full-official-smoke]

The output path must not already exist. The script never deletes a repository
or overwrites an existing result directory.
EOF
}

while (($#)); do
  case "$1" in
    --repo) REPO="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    --full-official-smoke) FULL_OFFICIAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT" ]] || { echo "--output is required" >&2; exit 2; }
REPO="$(cd "$REPO" && pwd)"
OUTPUT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT")"
[[ ! -e "$OUTPUT" ]] || { echo "output exists; refusing to overwrite: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT/logs" "$OUTPUT/results"

[[ -s "$REPO/binocmesher/source/event_registry.cpp" ]] || { echo "event_registry.cpp missing" >&2; exit 2; }
[[ -s "$REPO/binocmesher/source/event_registry.h" ]] || { echo "event_registry.h missing" >&2; exit 2; }
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
  echo "python=$(python3 --version 2>&1)"
  echo "compiler=$(g++ --version | head -1)"
  git -C "$REPO" status --short
} > "$OUTPUT/logs/environment.log"

make -C "$REPO/binocmesher" clean 2>&1 | tee "$OUTPUT/logs/make_clean.log"
make -C "$REPO/binocmesher" -j2 2>&1 | tee "$OUTPUT/logs/build.log"
sha256sum "$REPO/binocmesher/lib/core.so" | tee "$OUTPUT/results/core_so.sha256"

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
python3 "$REPO/experiments/p1_p4/validate_smoke.py" "${VALIDATE_ARGS[@]}" \
  2>&1 | tee "$OUTPUT/logs/final_validate.log"
