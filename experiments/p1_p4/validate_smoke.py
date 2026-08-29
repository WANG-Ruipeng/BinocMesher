#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE_COMMIT = "8fae63707b6b128f1a4f9a35ec4d4a2bdc488e19"


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--require-official", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.results.resolve()

    fixture = load(root / "results/core_fixtures/fixture_smoke.json")
    reference = load(root / "results/reference/summary.json")
    reference_validation = load(root / "results/reference/independent_validation.json")
    core = repo / "binocmesher/lib/core.so"
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()

    checks = {
        "base_commit": head == BASE_COMMIT,
        "core_so_exists": core.is_file() and core.stat().st_size > 0,
        "fixture_pass": fixture.get("verdict") == "PASS_P2_P4_CORE_FIXTURE_SMOKE",
        "reference_pass": reference.get("overall_verdict") == "GO_P1_P4_THEORY_AND_LOCAL_UPSTREAM_SEAM",
        "reference_independent_pass": reference_validation.get("verdict") == "PASS_INDEPENDENT_P1_P4_VALIDATION",
        "event_registry_source_present": (repo / "binocmesher/source/event_registry.cpp").is_file(),
        "event_registry_header_present": (repo / "binocmesher/source/event_registry.h").is_file(),
    }
    official = None
    if args.require_official:
        official = load(root / "results/official_p1/p1_official_smoke.json")
        checks.update({
            "official_p1_pass": official.get("verdict") == "PASS_P1_OFFICIAL_PIPELINE_READ_ONLY",
            "sidecar_generated": bool(official.get("checks", {}).get("instrumented_has_sidecar")),
            "official_cache_unchanged": bool(official.get("checks", {}).get("cache_bytes_identical")),
            "official_mesh_unchanged": bool(official.get("checks", {}).get("mesh_hashes_identical")),
        })

    passed = all(checks.values())
    validation = {
        "schema": "binoc-p1-p4-worktree-smoke-v3",
        "pass": passed,
        "verdict": "PASS_P1_P4_OFFICIAL_WORKTREE_SMOKE" if passed else "STOP_P1_P4_OFFICIAL_WORKTREE_SMOKE",
        "checks": checks,
        "base_commit": BASE_COMMIT,
        "head": head,
        "core_so_sha256": hashlib.sha256(core.read_bytes()).hexdigest() if core.is_file() else None,
        "full_official_p1_requested": args.require_official,
        "scope": {
            "p1": "full official Python pipeline only when --require-official is used",
            "p2_p4": "deterministic seam fixtures compiled into official core.so plus source-local oracle",
            "not_claimed": "production P2/P3 intervention or Forest/Cave/Mountain significance",
        },
    }
    (root / "results/validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    print(validation["verdict"])
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
