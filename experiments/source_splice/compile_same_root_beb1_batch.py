#!/usr/bin/env python3
'''Compile the two certified 104/5 BEB1 events into one atomic SSP1 plan.

This compiler is intentionally narrow. It consumes the independently
certified artifacts produced by run_all_canonical_beb1_events.py and only
admits their composition when the two closed event supports are disjoint. It
is not an overlapping-event solver.
'''
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shlex
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from compile_splice_plans import write_plan
from processed_mesh import HVID, SourceVID, TriangleRef
from runtime_common import oriented_face_multiset
from space_position_contract import (
    EVENT_IR_SCHEMA,
    SSP1_COORDINATE_FORMAT,
    plan_position_contract_is_valid,
    position_float64,
)


BATCH_IR_SCHEMA = 'binoc-same-root-beb1-batch-ir-v1'
BATCH_PLAN_SCHEMA = 'binoc-same-root-beb1-atomic-plan-v1'


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_json(value: Fraction) -> dict[str, int]:
    return {'numerator': value.numerator, 'denominator': value.denominator}


def fraction_from_json(value: dict[str, Any]) -> Fraction:
    return Fraction(int(value['numerator']), int(value['denominator']))


def parse_fraction(value: str) -> Fraction:
    numerator, denominator = value.split('/', 1)
    return Fraction(int(numerator), int(denominator))


def parse_source_vid(value: str) -> SourceVID:
    halves = value.split('|')
    if len(halves) != 2:
        raise ValueError(f'invalid SourceVID: {value!r}')
    hvids = []
    for half in halves:
        fields = half.split(':')
        if len(fields) != 2:
            raise ValueError(f'invalid HVID in SourceVID: {value!r}')
        hvids.append(HVID(int(fields[0]), int(fields[1])))
    result = SourceVID.canonical(hvids[0], hvids[1])
    if result.text() != value:
        raise ValueError(f'noncanonical SourceVID: {value!r}')
    return result


def parse_ssp1(path: Path) -> dict[str, Any]:
    rows = [
        shlex.split(line, comments=False, posix=True)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(rows) < 6 or rows[0] != ['SSP1'] or rows[-1] != ['END']:
        raise RuntimeError(f'invalid SSP1 framing: {path}')
    if len(rows[1]) != 2 or rows[1][0] != 'PLAN':
        raise RuntimeError(f'invalid SSP1 PLAN row: {path}')
    if len(rows[2]) != 3 or rows[2][0] != 'TIME':
        raise RuntimeError(f'invalid SSP1 TIME row: {path}')
    if rows[3] != ['COORDINATES', SSP1_COORDINATE_FORMAT]:
        raise RuntimeError(f'invalid SSP1 coordinate contract: {path}')
    if len(rows[4]) != 5 or rows[4][0] != 'EXPECT':
        raise RuntimeError(f'invalid SSP1 EXPECT row: {path}')
    result: dict[str, Any] = {
        'time': Fraction(int(rows[2][1]), int(rows[2][2])),
        'expect': tuple(map(int, rows[4][1:])),
        'suppressions': [],
        'source_vertices': [],
        'internal_vertices': [],
        'faces': [],
    }
    for row in rows[5:-1]:
        if row[0] == 'SUPPRESS' and len(row) == 8:
            result['suppressions'].append(tuple(map(int, row[1:])))
        elif row[0] == 'VERTEX_SOURCE' and len(row) == 7:
            result['source_vertices'].append(tuple(map(int, row[1:])))
        elif row[0] == 'VERTEX_INTERNAL' and len(row) == 7:
            result['internal_vertices'].append({
                'local_id': int(row[1]),
                'position': [float(value) for value in row[3:6]],
                'in_view': bool(int(row[6])),
            })
        elif row[0] == 'FACE' and len(row) == 5:
            result['faces'].append(tuple(map(int, row[1:])))
        else:
            raise RuntimeError(f'invalid SSP1 command in {path}: {row!r}')
    actual = (
        len(result['suppressions']),
        len(result['source_vertices']),
        len(result['internal_vertices']),
        len(result['faces']),
    )
    if actual != result['expect']:
        raise RuntimeError(
            f'SSP1 EXPECT mismatch in {path}: {actual} != {result["expect"]}')
    return result


def counter_rows(
    counter: collections.Counter[tuple[int, int, int]],
) -> list[list[int]]:
    return [list(value) for value, count in sorted(counter.items())
            for _ in range(count)]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload['vertices']),
            np.asarray(payload['faces']),
            np.asarray(payload['tags']),
        )


def separating_axis(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any] | None:
    '''Return a strict spatial AABB separating-axis certificate.'''
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (first.ndim != 2 or second.ndim != 2 or first.shape[1] != 4 or
            second.shape[1] != 4 or not np.all(np.isfinite(first)) or
            not np.all(np.isfinite(second))):
        return None
    first_min, first_max = np.min(first, axis=0), np.max(first, axis=0)
    second_min, second_max = np.min(second, axis=0), np.max(second, axis=0)
    candidates = []
    for axis in range(3):
        if first_max[axis] < second_min[axis]:
            candidates.append((
                float(second_min[axis] - first_max[axis]), axis,
                'first_before_second'))
        elif second_max[axis] < first_min[axis]:
            candidates.append((
                float(first_min[axis] - second_max[axis]), axis,
                'second_before_first'))
    if not candidates:
        return None
    margin, axis, order = max(candidates)
    return {
        'axis': axis,
        'axis_name': ('x', 'y', 'z')[axis],
        'order': order,
        'strict_margin': margin,
        'first_interval': [float(first_min[axis]), float(first_max[axis])],
        'second_interval': [float(second_min[axis]), float(second_max[axis])],
        'first_aabb4': [first_min.tolist(), first_max.tolist()],
        'second_aabb4': [second_min.tolist(), second_max.tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--events-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-output', type=Path, required=True)
    parser.add_argument('--expected-root', default='104/5')
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args()

    events_root = args.events_root.resolve()
    output = args.output.resolve()
    plan_output = args.plan_output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if plan_output.exists():
        raise FileExistsError(plan_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_root = parse_fraction(args.expected_root)
    expected_counts = {
        'events': 2,
        'raw_suppressions': 18,
        'boundary_vertices': 8,
        'internal_vertices': 2,
        'replacement_faces': 8,
        'removed_logical_faces': 4,
    }

    campaign_path = events_root / 'all_canonical_beb1_summary.json'
    campaign = read_json(campaign_path)
    selected = [
        record for record in campaign.get('events', [])
        if fraction_from_json(record['root']) == expected_root
    ]
    selected.sort(key=lambda value: str(value['event_id']))
    artifacts: list[dict[str, Any]] = []
    load_errors: list[str] = []
    for record in selected:
        event_root = events_root / str(record['key'])
        try:
            ir_path = event_root / 'critical_beb1_event_ir.json'
            plan_path = event_root / 'critical_beb1.ssp1'
            validation_path = event_root / 'whole_mesh_validation.json'
            runtime_root = event_root / 'runtime'
            ir = read_json(ir_path)
            validation = read_json(validation_path)
            parsed_plan = parse_ssp1(plan_path)
            baseline = load_npz(runtime_root / 'baseline_omp1.npz')
            critical = load_npz(runtime_root / 'critical_omp1.npz')
            baseline_faces = oriented_face_multiset(baseline[1])
            critical_faces = oriented_face_multiset(critical[1])
            artifacts.append({
                'record': record,
                'ir_path': ir_path,
                'ir': ir,
                'plan_path': plan_path,
                'parsed_plan': parsed_plan,
                'validation_path': validation_path,
                'validation': validation,
                'baseline': baseline,
                'critical': critical,
                'removed': baseline_faces - critical_faces,
                'added': critical_faces - baseline_faces,
            })
        except Exception as error:
            load_errors.append(f'{record.get("key")}: {error!r}')

    artifacts_admitted = bool(artifacts) and all(
        item['record'].get('pass') is True and
        item['ir'].get('schema') == EVENT_IR_SCHEMA and
        item['ir'].get('pass') is True and
        item['ir'].get('whole_mesh_splice_ready') is True and
        item['ir'].get('admission', {}).get(
            'mapping_cylinder_ready') is True and
        item['ir'].get('event_star_geometry', {}).get(
            'mapping_cylinder', {}).get('pass') is True and
        item['ir'].get('event_star_geometry', {}).get(
            'mapping_cylinder', {}).get(
                'critical_side_edges_remaining') == 0 and
        plan_position_contract_is_valid(
            item['ir'].get('whole_mesh_replacement_plan')) and
        item['validation'].get('verdict') ==
        'PASS_CRITICAL_BEB1_WHOLE_MESH_SPLICE'
        for item in artifacts
    )
    input_plans_exact = bool(artifacts) and all(
        item['ir']['whole_mesh_replacement_plan'].get('sha256') ==
        sha256_file(item['plan_path']) and
        item['parsed_plan']['time'] == expected_root and
        item['parsed_plan']['expect'] == (
            item['ir']['whole_mesh_replacement_plan'][
                'raw_suppression_count'],
            len(item['ir']['whole_mesh_replacement_plan']['boundary_cycle']),
            1,
            item['ir']['whole_mesh_replacement_plan'][
                'replacement_face_count'],
        ) and
        len(item['parsed_plan']['internal_vertices']) == 1 and
        np.array_equal(
            position_float64(
                item['parsed_plan']['internal_vertices'][0]['position']),
            position_float64(item['ir']['whole_mesh_replacement_plan'][
                'critical_position']))
        for item in artifacts
    )
    suppressions = [
        tuple(map(int, row))
        for item in artifacts
        for row in item['ir']['whole_mesh_replacement_plan'].get(
            'raw_suppressions', [])
    ]
    boundary_tokens = [
        str(token)
        for item in artifacts
        for token in item['ir']['whole_mesh_replacement_plan'].get(
            'boundary_cycle', [])
    ]
    event_hvid_sets = [
        {str(value) for value in item['ir'].get('kernel', {}).get(
            'block_vertex_source_hvids', []) if value is not None}
        for item in artifacts
    ]
    global_boundary_sets = [
        {tuple(map(int, edge)) for edge in
         item['validation'].get('patch_boundary_edges', [])}
        for item in artifacts
    ]
    global_boundary_vertex_sets = [
        {vertex for edge in edges for vertex in edge}
        for edges in global_boundary_sets
    ]
    separation = None
    if len(artifacts) == 2:
        separation = separating_axis(
            np.asarray(artifacts[0]['ir']['event_star_geometry'][
                'mapping_cylinder']['vertices4'], dtype=np.float64),
            np.asarray(artifacts[1]['ir']['event_star_geometry'][
                'mapping_cylinder']['vertices4'], dtype=np.float64),
        )
    baselines_identical = len(artifacts) == 2 and all(
        all(np.array_equal(
            artifacts[0]['baseline'][index], item['baseline'][index])
            for index in range(3))
        for item in artifacts[1:]
    )
    elements = {
        int(item['ir']['whole_mesh_replacement_plan'].get('element', -1))
        for item in artifacts
    }
    counts = {
        'events': len(artifacts),
        'raw_suppressions': len(suppressions),
        'boundary_vertices': len(boundary_tokens),
        'internal_vertices': len(artifacts),
        'replacement_faces': sum(
            int(item['ir']['whole_mesh_replacement_plan'].get(
                'replacement_face_count', -1))
            for item in artifacts),
        'removed_logical_faces': sum(
            sum(item['removed'].values()) for item in artifacts),
    }
    checks = {
        'source_campaign_passed': (
            campaign.get('verdict') ==
            'PASS_ALL_CANONICAL_BEB1_WHOLE_MESH'),
        'exactly_two_events_at_104_over_5': (
            len(selected) == 2 and len(artifacts) == 2 and not load_errors),
        'event_artifacts_independently_admitted': artifacts_admitted,
        'input_ssp1_contracts_and_hashes_exact': input_plans_exact,
        'same_exact_root': bool(artifacts) and all(
            fraction_from_json(item['ir']['event']['root']) == expected_root
            for item in artifacts),
        'common_baseline_byte_exact': baselines_identical,
        'common_runtime_element': len(elements) == 1,
        'suppression_owners_pairwise_disjoint': (
            len(suppressions) == len(set(suppressions))),
        'boundary_sourcevids_pairwise_disjoint': (
            len(boundary_tokens) == len(set(boundary_tokens))),
        'event_hvids_pairwise_disjoint': (
            len(event_hvid_sets) == 2 and
            not (event_hvid_sets[0] & event_hvid_sets[1])),
        'global_patch_boundary_edges_pairwise_disjoint': (
            len(global_boundary_sets) == 2 and
            not (global_boundary_sets[0] & global_boundary_sets[1])),
        'global_patch_boundary_vertices_pairwise_disjoint': (
            len(global_boundary_vertex_sets) == 2 and
            not (global_boundary_vertex_sets[0] &
                 global_boundary_vertex_sets[1])),
        'closed_mapping_cylinders_have_strict_spatial_separation': (
            separation is not None and separation['strict_margin'] > 0.0),
        'each_independent_delta_is_two_to_four': bool(artifacts) and all(
            sum(item['removed'].values()) == 2 and
            sum(item['added'].values()) == 4 for item in artifacts),
        'aggregate_counts_match_fixed_e2_e3_contract': (
            counts == expected_counts),
    }
    ready = all(checks.values())

    plan_metadata: dict[str, Any] | None = None
    event_records: list[dict[str, Any]] = []
    if artifacts:
        total_boundary = len(boundary_tokens)
        boundary_vertices = [
            parse_source_vid(value) for value in boundary_tokens]
        triangle_refs = [
            TriangleRef(*value) for value in suppressions]
        internal_vertices = []
        replacement_faces: list[tuple[int, int, int]] = []
        boundary_offset = 0
        for event_index, item in enumerate(artifacts):
            original = item['ir']['whole_mesh_replacement_plan']
            parsed = item['parsed_plan']
            boundary_count = len(original['boundary_cycle'])
            old_internal_id = parsed['internal_vertices'][0]['local_id']
            new_internal_id = total_boundary + event_index
            remap = {
                local_id: boundary_offset + local_id
                for local_id in range(boundary_count)
            }
            remap[old_internal_id] = new_internal_id
            event_faces = []
            for face in original['replacement_faces']:
                mapped = tuple(remap[int(vertex)] for vertex in face)
                replacement_faces.append(mapped)
                event_faces.append(list(mapped))
            internal_vertices.append((
                position_float64(original['critical_position']),
                parsed['internal_vertices'][0]['in_view'],
            ))
            event_records.append({
                'event_index': event_index,
                'event_key': item['record']['key'],
                'event_id': item['record']['event_id'],
                'event_ir': str(item['ir_path']),
                'event_ir_sha256': sha256_file(item['ir_path']),
                'input_plan': str(item['plan_path']),
                'input_plan_sha256': sha256_file(item['plan_path']),
                'whole_mesh_validation': str(item['validation_path']),
                'whole_mesh_validation_sha256': sha256_file(
                    item['validation_path']),
                'boundary_sourcevids': list(original['boundary_cycle']),
                'boundary_local_ids': list(range(
                    boundary_offset, boundary_offset + boundary_count)),
                'internal_local_id': new_internal_id,
                'critical_position_contract':
                    original['critical_position_contract'],
                'expected_internal_tag': int(
                    item['critical'][2][len(item['baseline'][0])]),
                'raw_suppressions': list(original['raw_suppressions']),
                'replacement_faces': event_faces,
                'removed_oriented_faces': counter_rows(item['removed']),
                'independent_added_oriented_faces':
                    counter_rows(item['added']),
                'global_patch_boundary_edges': sorted(
                    list(edge) for edge in global_boundary_sets[event_index]),
            })
            boundary_offset += boundary_count

        plan_metadata = {
            'schema': BATCH_PLAN_SCHEMA,
            'plan_id':
                'production-critical-beb1-same-root-disjoint-batch',
            'composition': 'ATOMIC_DISJOINT_UNION',
            'exact_time': fraction_json(expected_root),
            'element': next(iter(elements)) if len(elements) == 1 else None,
            'counts': counts,
            'boundary_sourcevids': boundary_tokens,
            'internal_vertices': [
                {
                    'event_id': event_records[index]['event_id'],
                    'local_id': total_boundary + index,
                    'position': position_float64(value[0]).tolist(),
                    'in_view': bool(value[1]),
                    'position_contract': event_records[index][
                        'critical_position_contract'],
                }
                for index, value in enumerate(internal_vertices)
            ],
            'raw_suppressions': [list(value) for value in suppressions],
            'replacement_faces': [
                list(value) for value in replacement_faces],
            'per_event': event_records,
        }
        if ready:
            write_plan(
                plan_output, plan_metadata['plan_id'], expected_root,
                triangle_refs, boundary_vertices, internal_vertices,
                replacement_faces, int(plan_metadata['element']))
            plan_metadata['path'] = str(plan_output)
            plan_metadata['sha256'] = sha256_file(plan_output)

    baseline_counts = None
    predicted_counts = None
    if artifacts and baselines_identical:
        baseline_counts = {
            'vertices': len(artifacts[0]['baseline'][0]),
            'faces': len(artifacts[0]['baseline'][1]),
        }
        predicted_counts = {
            'vertices': (
                baseline_counts['vertices'] + counts['internal_vertices']),
            'faces': (
                baseline_counts['faces'] -
                counts['removed_logical_faces'] +
                counts['replacement_faces']),
        }
    payload = {
        'schema': BATCH_IR_SCHEMA,
        'pass': ready,
        'verdict': (
            'PASS_SAME_ROOT_BEB1_BATCH_IR'
            if ready else 'STOP_SAME_ROOT_BEB1_BATCH_IR'),
        'runtime_disposition': (
            'READY_FOR_ATOMIC_SAME_ROOT_SPLICE'
            if ready else 'SAME_ROOT_BATCH_CONFLICT_OR_UNRESOLVED'),
        'whole_mesh_splice_ready': ready,
        'exact_time': fraction_json(expected_root),
        'source_campaign': {
            'path': str(campaign_path),
            'sha256': sha256_file(campaign_path),
            'verdict': campaign.get('verdict'),
        },
        'composition': {
            'classification': (
                'DISJOINT_CLOSED_SUPPORT' if ready else 'UNRESOLVED'),
            'mapping_cylinder_separation': separation,
            'event_count': len(artifacts),
            'event_order': [
                item['record']['event_id'] for item in artifacts],
            'atomicity': (
                'Compile both certified deltas against their byte-identical '
                'common baseline, admit their complete union before mutation, '
                'then emit once for their common runtime element.'
            ),
        },
        'events': event_records,
        'counts': counts,
        'expected_counts': expected_counts,
        'baseline_counts': baseline_counts,
        'predicted_batch_counts': predicted_counts,
        'whole_mesh_replacement_plan': plan_metadata,
        'checks': checks,
        'load_errors': load_errors,
        'scope': {
            'completed': (
                'atomic composition of the two independently certified '
                'canonical BEB1 events at exact root 104/5 when their closed '
                '4D supports have a strict spatial separating axis'
            ),
            'not_claimed': (
                'shared-boundary, overlapping-support, arbitrary-order, or '
                'general simultaneous-event conflict resolution'
            ),
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(payload['verdict'])
    print(payload['runtime_disposition'])
    if not ready:
        for name, passed in checks.items():
            if not passed:
                print('FAILED:', name)
        for error in load_errors:
            print('LOAD_ERROR:', error)
        return 3 if args.require_ready else 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
