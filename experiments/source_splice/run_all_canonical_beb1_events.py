#!/usr/bin/env python3
'''Run the BEB1 admission and whole-mesh campaign for every demo event.

Each canonical registry event is compiled and executed independently.  Events
sharing an exact root are deliberately not combined in this campaign; overlap
and simultaneous-batch conflict resolution remain a separate gate.
'''
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from space_position_contract import (
    EVENT_IR_SCHEMA,
    plan_position_contract_is_valid,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def fraction_json(value: Fraction) -> dict[str, int]:
    return {
        'numerator': value.numerator,
        'denominator': value.denominator,
    }


def discover_events(cache_root: Path) -> list[dict[str, Any]]:
    registry = cache_root / 'event_registry_p1.csv'
    if not registry.is_file():
        raise FileNotFoundError(registry)
    with registry.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row['canonical_event_id']].append(row)
    events = []
    for event_id, group in grouped.items():
        roots = {
            Fraction(int(row['root_num']), int(row['root_den']))
            for row in group
        }
        if len(roots) != 1:
            raise RuntimeError(
                f'canonical event has inconsistent roots: {event_id}')
        root = next(iter(roots))
        events.append({
            'event_id': event_id,
            'root_fraction': root,
            'raw_observations': len(group),
            'logical_incidences': len({
                row['logical_incidence_id'] for row in group
            }),
        })
    events.sort(key=lambda value: (
        value['root_fraction'], value['event_id']))
    for index, event in enumerate(events):
        digest = hashlib.sha256(
            event['event_id'].encode('utf-8')).hexdigest()[:12]
        event['index'] = index
        event['key'] = f'event-{index:02d}-{digest}'
    return events


def run_logged(
    command: list[str],
    log_path: Path,
    cwd: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write('$ ' + shlex.join(command) + '\n')
        handle.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f'\nEXIT_CODE={result.returncode}\n')
    return int(result.returncode)


def stage_record(
    exit_code: int | None,
    verdict: str | None,
    passed: bool | None,
    log_path: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    return {
        'exit_code': exit_code,
        'verdict': verdict,
        'pass': passed,
        'log': (
            log_path.relative_to(output_root).as_posix()
            if log_path is not None else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, default=4)
    parser.add_argument('--expected-profile', default='demo')
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()

    repo = args.repo.resolve()
    cache = args.cache_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    profile_path = cache / 'profile_result.json'
    profile = read_json(profile_path) if profile_path.is_file() else {}
    events = discover_events(cache)
    results: list[dict[str, Any]] = []

    tv_runner = repo / 'experiments' / 'tv0_tv4' / 'run_tv0_tv4.py'
    tv_validator = (
        repo / 'experiments' / 'tv0_tv4' / 'validate_tv0_tv4.py')
    event_compiler = (
        repo / 'experiments' / 'source_splice' /
        'compile_critical_beb1_event_ir.py')
    runtime_runner = (
        repo / 'experiments' / 'source_splice' /
        'run_critical_beb1_splice.py')
    runtime_validator = (
        repo / 'experiments' / 'source_splice' /
        'validate_critical_beb1_splice.py')

    for event in events:
        event_id = str(event['event_id'])
        root = event['root_fraction']
        event_root = output / str(event['key'])
        logs = event_root / 'logs'
        event_root.mkdir(parents=True)
        theory = event_root / 'theory'
        event_ir_path = event_root / 'critical_beb1_event_ir.json'
        plan_path = event_root / 'critical_beb1.ssp1'
        runtime_path = event_root / 'runtime'
        validation_path = event_root / 'whole_mesh_validation.json'

        record: dict[str, Any] = {
            'index': event['index'],
            'key': event['key'],
            'event_id': event_id,
            'root': fraction_json(root),
            'raw_observations': event['raw_observations'],
            'logical_incidences': event['logical_incidences'],
            'stages': {},
        }

        theory_log = logs / 'theory.log'
        theory_code = run_logged([
            sys.executable, str(tv_runner),
            '--cache-root', str(cache),
            '--output', str(theory),
            '--event-id', event_id,
        ], theory_log, repo)
        theory_summary_path = theory / 'summary.json'
        theory_summary = (
            read_json(theory_summary_path)
            if theory_summary_path.is_file() else {})
        theory_ok = (
            theory_code == 0 and
            theory_summary.get('pass') is True and
            theory_summary.get('event_id') == event_id and
            theory_summary.get('verdict') ==
            'PASS_TV0_TV4_PRODUCTION_DERIVED_THEORY_VALIDATION'
        )
        record['stages']['theory'] = stage_record(
            theory_code, theory_summary.get('verdict'), theory_ok,
            theory_log, output)

        independent_ok = False
        if theory_ok:
            independent_log = logs / 'independent_theory_validation.log'
            independent_code = run_logged([
                sys.executable, str(tv_validator),
                '--results', str(theory),
                '--cache-root', str(cache),
                '--event-id', event_id,
            ], independent_log, repo)
            independent_path = theory / 'independent_validation.json'
            independent = (
                read_json(independent_path)
                if independent_path.is_file() else {})
            independent_ok = (
                independent_code == 0 and
                independent.get('pass') is True and
                independent.get('event_id') == event_id and
                independent.get('verdict') ==
                'PASS_INDEPENDENT_TV0_TV4_VALIDATION'
            )
            record['stages']['independent_theory'] = stage_record(
                independent_code, independent.get('verdict'),
                independent_ok, independent_log, output)
        else:
            record['stages']['independent_theory'] = stage_record(
                None, 'SKIPPED_AFTER_THEORY_FAILURE', False, None, output)

        event_ir_ok = False
        event_ir: dict[str, Any] = {}
        if theory_ok and independent_ok:
            event_ir_log = logs / 'event_ir.log'
            event_ir_code = run_logged([
                sys.executable, str(event_compiler),
                '--cache-root', str(cache),
                '--theory-root', str(theory),
                '--output', str(event_ir_path),
                '--plan-output', str(plan_path),
                '--event-id', event_id,
                '--expected-root',
                f'{root.numerator}/{root.denominator}',
                '--require-whole-mesh-ready',
            ], event_ir_log, repo)
            event_ir = (
                read_json(event_ir_path)
                if event_ir_path.is_file() else {})
            mapping = (
                event_ir.get('event_star_geometry', {})
                .get('mapping_cylinder', {}))
            event_plan = event_ir.get('whole_mesh_replacement_plan') or {}
            position_contract = (
                event_plan.get('critical_position_contract') or {})
            event_ir_ok = (
                event_ir_code == 0 and
                event_ir.get('schema') == EVENT_IR_SCHEMA and
                event_ir.get('pass') is True and
                event_ir.get('event', {}).get('event_id') == event_id and
                event_ir.get('whole_mesh_splice_ready') is True and
                event_ir.get('runtime_disposition') ==
                'READY_FOR_WHOLE_MESH_SPLICE' and
                event_ir.get(
                    'admission', {}).get('mapping_cylinder_ready') is True and
                event_ir.get('admission', {}).get(
                    'runtime_space_position_contract_ready') is True and
                event_ir.get('admission', {}).get(
                    'critical_position_quantization_isotopy_ready') is True and
                plan_position_contract_is_valid(event_plan) and
                event_ir.get('event_star_geometry', {}).get(
                    'critical_position_quantization_audit', {}).get(
                        'pass') is True and
                mapping.get('pass') is True and
                mapping.get('critical_side_edges_remaining') == 0 and
                plan_path.is_file()
            )
            record['stages']['event_ir'] = stage_record(
                event_ir_code, event_ir.get('verdict'), event_ir_ok,
                event_ir_log, output)
            record['event_ir_metrics'] = {
                'mapping_vertices': len(mapping.get('vertices4', [])),
                'mapping_tetrahedra': len(mapping.get('tetrahedra', [])),
                'mapping_side_faces': len(
                    mapping.get('side_trace_faces', [])),
                'minimum_gram_volume':
                    mapping.get('minimum_gram_volume'),
                'critical_side_edges_remaining':
                    mapping.get('critical_side_edges_remaining'),
                'critical_position_exactly_representable':
                    position_contract.get('exactly_representable'),
                'critical_position_maximum_quantization_error':
                    position_contract.get(
                        'maximum_absolute_quantization_error'),
                'plan_sha256': (
                    hashlib.sha256(plan_path.read_bytes()).hexdigest()
                    if plan_path.is_file() else None
                ),
            }
        else:
            record['stages']['event_ir'] = stage_record(
                None, 'SKIPPED_AFTER_THEORY_VALIDATION_FAILURE',
                False, None, output)

        runtime_ok = False
        validation_ok = False
        if args.compile_only:
            record['stages']['runtime'] = stage_record(
                None, 'SKIPPED_COMPILE_ONLY', None, None, output)
            record['stages']['whole_mesh_validation'] = stage_record(
                None, 'SKIPPED_COMPILE_ONLY', None, None, output)
        elif event_ir_ok:
            runtime_log = logs / 'runtime.log'
            runtime_code = run_logged([
                sys.executable, str(runtime_runner),
                '--repo', str(repo),
                '--cache-root', str(cache),
                '--event-ir', str(event_ir_path),
                '--plan', str(plan_path),
                '--output', str(runtime_path),
            ], runtime_log, repo)
            runtime_summary_path = runtime_path / 'runtime_summary.json'
            runtime_summary = (
                read_json(runtime_summary_path)
                if runtime_summary_path.is_file() else {})
            runtime_ok = (
                runtime_code == 0 and
                runtime_summary.get('pass') is True and
                len(runtime_summary.get('results', {})) == 4 and
                all(
                    value.get('pass') is True
                    for value in runtime_summary.get(
                        'results', {}).values()
                )
            )
            record['stages']['runtime'] = stage_record(
                runtime_code,
                (
                    'PASS_CRITICAL_BEB1_RUNTIME_CAMPAIGN'
                    if runtime_ok else
                    'STOP_CRITICAL_BEB1_RUNTIME_CAMPAIGN'
                ),
                runtime_ok, runtime_log, output)

            if runtime_ok:
                validation_log = logs / 'whole_mesh_validation.log'
                validation_code = run_logged([
                    sys.executable, str(runtime_validator),
                    '--runtime-results', str(runtime_path),
                    '--event-ir', str(event_ir_path),
                    '--output', str(validation_path),
                ], validation_log, repo)
                validation = (
                    read_json(validation_path)
                    if validation_path.is_file() else {})
                validation_ok = (
                    validation_code == 0 and
                    validation.get('pass') is True and
                    validation.get('verdict') ==
                    'PASS_CRITICAL_BEB1_WHOLE_MESH_SPLICE'
                )
                record['stages']['whole_mesh_validation'] = stage_record(
                    validation_code, validation.get('verdict'),
                    validation_ok, validation_log, output)
                record['whole_mesh_counts'] = validation.get('counts')
            else:
                record['stages']['whole_mesh_validation'] = stage_record(
                    None, 'SKIPPED_AFTER_RUNTIME_FAILURE',
                    False, None, output)
        else:
            record['stages']['runtime'] = stage_record(
                None, 'SKIPPED_AFTER_EVENT_IR_FAILURE',
                False, None, output)
            record['stages']['whole_mesh_validation'] = stage_record(
                None, 'SKIPPED_AFTER_EVENT_IR_FAILURE',
                False, None, output)

        record_checks = {
            'theory_pass': theory_ok,
            'independent_theory_pass': independent_ok,
            'event_ir_and_plan_admitted': event_ir_ok,
        }
        if not args.compile_only:
            record_checks.update({
                'runtime_pass': runtime_ok,
                'whole_mesh_validation_pass': validation_ok,
            })
        record['checks'] = record_checks
        record['pass'] = all(record_checks.values())
        results.append(record)

    checks = {
        'expected_demo_profile':
            profile.get('profile') == args.expected_profile,
        'expected_canonical_event_count':
            len(events) == args.expected_events,
        'canonical_event_ids_unique':
            len({event['event_id'] for event in events}) == len(events),
        'all_event_theory_pass':
            all(value['checks']['theory_pass'] for value in results),
        'all_independent_theory_pass':
            all(
                value['checks']['independent_theory_pass']
                for value in results),
        'all_event_ir_and_plans_admitted':
            all(
                value['checks']['event_ir_and_plan_admitted']
                for value in results),
    }
    if not args.compile_only:
        checks.update({
            'all_runtime_campaigns_pass': all(
                value['checks']['runtime_pass'] for value in results),
            'all_whole_mesh_validations_pass': all(
                value['checks']['whole_mesh_validation_pass']
                for value in results),
        })
    passed = bool(results) and all(checks.values())
    if args.compile_only:
        verdict = (
            'PASS_ALL_CANONICAL_BEB1_COMPILE_ONLY'
            if passed else
            'STOP_ALL_CANONICAL_BEB1_COMPILE_ONLY'
        )
    else:
        verdict = (
            'PASS_ALL_CANONICAL_BEB1_WHOLE_MESH'
            if passed else
            'STOP_ALL_CANONICAL_BEB1_WHOLE_MESH'
        )
    payload = {
        'schema': 'binoc-all-canonical-beb1-campaign-v1',
        'pass': passed,
        'verdict': verdict,
        'mode': 'compile_only' if args.compile_only else 'full_runtime',
        'cache_root': str(cache),
        'profile': profile.get('profile'),
        'checks': checks,
        'coverage': {
            'canonical_events_discovered': len(events),
            'theory_passed': sum(
                value['checks']['theory_pass'] for value in results),
            'event_ir_and_plans_admitted': sum(
                value['checks']['event_ir_and_plan_admitted']
                for value in results),
            'runtime_validated': (
                0 if args.compile_only else
                sum(
                    value['checks']['whole_mesh_validation_pass']
                    for value in results)
            ),
            'canonical_event_fraction': (
                '0/0' if not events else
                f'{sum(value["pass"] for value in results)}/{len(events)}'
            ),
        },
        'events': results,
        'scope': {
            'independent_per_event': True,
            'simultaneous_same_root_batch': False,
            'overlap_conflict_resolution': False,
            'paper_scenes': False,
        },
    }
    summary_path = output / 'all_canonical_beb1_summary.json'
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(verdict)
    print(json.dumps(payload['coverage'], indent=2, sort_keys=True))
    if not passed:
        for name, value in checks.items():
            if not value:
                print('FAILED:', name)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
