#!/usr/bin/env python3
'''Run one admitted same-root BEB1 batch in the complete ordinary mesh.'''
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from compile_same_root_beb1_batch import BATCH_IR_SCHEMA, BATCH_PLAN_SCHEMA
from run_splice_experiment import run_success
from space_position_contract import space_position_contract_is_valid


def fraction_from_json(value: dict[str, Any]) -> Fraction:
    return Fraction(int(value['numerator']), int(value['denominator']))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--batch-ir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    batch_ir_path = args.batch_ir.resolve()
    batch_ir: dict[str, Any] = json.loads(batch_ir_path.read_text())
    plan_metadata = batch_ir.get('whole_mesh_replacement_plan') or {}
    contracts = [
        value.get('position_contract')
        for value in plan_metadata.get('internal_vertices', [])
    ]
    if (
        batch_ir.get('schema') != BATCH_IR_SCHEMA or
        batch_ir.get('pass') is not True or
        batch_ir.get('whole_mesh_splice_ready') is not True or
        batch_ir.get('runtime_disposition') !=
        'READY_FOR_ATOMIC_SAME_ROOT_SPLICE' or
        batch_ir.get('composition', {}).get('classification') !=
        'DISJOINT_CLOSED_SUPPORT' or
        plan_metadata.get('schema') != BATCH_PLAN_SCHEMA or
        len(contracts) != 2 or
        not all(space_position_contract_is_valid(value)
                for value in contracts)
    ):
        raise RuntimeError('same-root BEB1 batch IR was not admitted')

    root = fraction_from_json(batch_ir['exact_time'])
    repo = args.repo.resolve()
    cache = args.cache_root.resolve()
    plan = args.plan.resolve()
    if not plan.is_file():
        raise FileNotFoundError(plan)
    if (
        plan_metadata.get('sha256') !=
        hashlib.sha256(plan.read_bytes()).hexdigest()
    ):
        raise RuntimeError('same-root BEB1 SSP1 hash disagrees with batch IR')

    results = {
        'baseline_omp1': run_success(
            repo, cache, root, None, output, 'baseline_omp1', 1),
        'baseline_omp8': run_success(
            repo, cache, root, None, output, 'baseline_omp8', 8),
        'batch_omp1': run_success(
            repo, cache, root, plan, output, 'batch_omp1', 1),
        'batch_omp8': run_success(
            repo, cache, root, plan, output, 'batch_omp8', 8),
    }
    passed = all(value['pass'] for value in results.values())
    payload = {
        'schema': 'binoc-same-root-beb1-runtime-campaign-v1',
        'pass': passed,
        'verdict': (
            'PASS_SAME_ROOT_BEB1_RUNTIME_CAMPAIGN'
            if passed else 'STOP_SAME_ROOT_BEB1_RUNTIME_CAMPAIGN'
        ),
        'exact_time': batch_ir['exact_time'],
        'batch_ir': str(batch_ir_path),
        'plan': str(plan),
        'results': results,
    }
    (output / 'runtime_summary.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(payload['verdict'])
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
