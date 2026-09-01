#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from fractions import Fraction
from pathlib import Path

from runtime_common import exact_mesh, save_mesh_npz


def run_success(repo: Path, cache: Path, exact: Fraction, plan: Path | None,
                output: Path, name: str, omp: int) -> dict:
    audit = output / f"{name}.audit.json" if plan is not None else None
    vertices, faces, tags, metadata = exact_mesh(
        repo, cache, exact, plan=plan, audit=audit, omp_threads=omp)
    if audit is not None:
        metadata["audit"] = json.loads(audit.read_text())
    metadata["pass"] = True
    save_mesh_npz(output / f"{name}.npz", vertices, faces, tags, metadata)
    (output / f"{name}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def run_failure(repo: Path, cache: Path, exact: Fraction, plan: Path,
                output: Path, name: str, expected: tuple[str, ...]) -> dict:
    try:
        exact_mesh(repo, cache, exact, plan=plan,
                   audit=output / f"{name}.audit.json")
    except Exception as error:
        message = str(error)
        passed = any(fragment in message for fragment in expected)
        payload = {
            "pass": passed,
            "expected_fragments": list(expected),
            "error": message,
            "traceback": traceback.format_exc(),
        }
        (output / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if not passed:
            raise AssertionError(
                f"{name} failed for the wrong reason: {message}") from error
        return payload
    raise AssertionError(f"{name} unexpectedly succeeded")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-num", type=int, required=True)
    parser.add_argument("--time-den", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    exact = Fraction(args.time_num, args.time_den)
    repo = args.repo.resolve()
    cache = args.cache_root.resolve()
    plans = args.plans.resolve()

    results = {
        "baseline_omp1": run_success(
            repo, cache, exact, None, output, "baseline_omp1", 1),
        "baseline_omp8": run_success(
            repo, cache, exact, None, output, "baseline_omp8", 8),
        "identity_omp1": run_success(
            repo, cache, exact, plans / "identity.ssp1", output,
            "identity_omp1", 1),
        "identity_omp8": run_success(
            repo, cache, exact, plans / "identity.ssp1", output,
            "identity_omp8", 8),
        "star_omp1": run_success(
            repo, cache, exact, plans / "star.ssp1", output,
            "star_omp1", 1),
        "star_omp8": run_success(
            repo, cache, exact, plans / "star.ssp1", output,
            "star_omp8", 8),
    }
    failures = {
        "wrong_time": run_failure(
            repo, cache, exact, plans / "wrong_time.ssp1", output,
            "wrong_time", ("exact time does not match",)),
        "missing_ref": run_failure(
            repo, cache, exact, plans / "missing_ref.ssp1", output,
            "missing_ref", ("not suppressed exactly once", "suppression count mismatch")),
        "bad_boundary": run_failure(
            repo, cache, exact, plans / "bad_boundary.ssp1", output,
            "bad_boundary", ("was not produced by ordinary mesh", "was not emitted by the ordinary mesh")),
    }
    summary = {
        "schema": "binoc-source-splice-runtime-campaign-v1",
        "exact_time": {
            "numerator": exact.numerator,
            "denominator": exact.denominator,
        },
        "results": results,
        "negative_tests": failures,
        "pass": all(item["pass"] for item in results.values()) and
                all(item["pass"] for item in failures.values()),
    }
    (output / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("PASS_SOURCE_SPLICE_RUNTIME_CAMPAIGN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
