#!/usr/bin/env python3
'''Run one admitted BEB1 replacement in the complete ordinary mesh.'''
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from run_splice_experiment import run_success


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--event-ir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    event_ir = json.loads(args.event_ir.resolve().read_text())
    if (
        event_ir.get('whole_mesh_splice_ready') is not True or
        event_ir.get('runtime_disposition') != 'READY_FOR_WHOLE_MESH_SPLICE'
    ):
        raise RuntimeError('critical BEB1 Event IR was not admitted')
    root = event_ir['event']['root']
    exact = Fraction(int(root['numerator']), int(root['denominator']))
    repo = args.repo.resolve()
    cache = args.cache_root.resolve()
    plan = args.plan.resolve()
    if not plan.is_file():
        raise FileNotFoundError(plan)
    expected_plan = event_ir.get('whole_mesh_replacement_plan') or {}
    if (
        expected_plan.get('sha256') !=
        hashlib.sha256(plan.read_bytes()).hexdigest()
    ):
        raise RuntimeError('critical BEB1 SSP1 hash disagrees with Event IR')

    results = {
        'baseline_omp1': run_success(
            repo, cache, exact, None, output, 'baseline_omp1', 1),
        'baseline_omp8': run_success(
            repo, cache, exact, None, output, 'baseline_omp8', 8),
        'critical_omp1': run_success(
            repo, cache, exact, plan, output, 'critical_omp1', 1),
        'critical_omp8': run_success(
            repo, cache, exact, plan, output, 'critical_omp8', 8),
    }
    payload = {
        'schema': 'binoc-critical-beb1-runtime-campaign-v1',
        'pass': all(value['pass'] for value in results.values()),
        'exact_time': root,
        'event_ir': str(args.event_ir.resolve()),
        'plan': str(plan),
        'results': results,
    }
    (output / 'runtime_summary.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print('PASS_CRITICAL_BEB1_RUNTIME_CAMPAIGN')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
