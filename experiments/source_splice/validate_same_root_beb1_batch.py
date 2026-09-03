#!/usr/bin/env python3
'''Validate the atomic E2+E3 same-root whole-mesh replacement.'''
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from compile_same_root_beb1_batch import BATCH_IR_SCHEMA, BATCH_PLAN_SCHEMA
from runtime_common import (
    canonical_face,
    face_set_nonincident_intersections,
    mesh_topology,
    oriented_face_multiset,
    patch_nonincident_intersections,
)
from space_position_contract import (
    SSP1_COORDINATE_FORMAT,
    binary32_hex,
    position_float64,
    space_position_contract_is_valid,
)


def load_case(root: Path, name: str):
    payload = json.loads((root / f'{name}.json').read_text())
    arrays = np.load(root / f'{name}.npz')
    return payload, arrays


def counter_from_rows(rows: list[list[int]]) -> collections.Counter:
    return collections.Counter(tuple(map(int, row)) for row in rows)


def external_partner_keys(
    faces: np.ndarray,
    patch_indices: list[int],
    pairs: list[tuple[int, int]],
) -> set[tuple[int, int, int]]:
    patch = set(patch_indices)
    result = set()
    for first, second in pairs:
        other = second if first in patch else first
        result.add(tuple(sorted(map(int, faces[other]))))
    return result


def boundary_edges(
    faces: np.ndarray,
    indices: list[int],
) -> list[tuple[int, int]]:
    incidence: collections.Counter[tuple[int, int]] = collections.Counter()
    for index in indices:
        a, b, c = map(int, faces[index])
        for edge in ((a, b), (b, c), (c, a)):
            incidence[tuple(sorted(edge))] += 1
    return sorted(edge for edge, count in incidence.items() if count == 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-results', type=Path, required=True)
    parser.add_argument('--batch-ir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    runtime = args.runtime_results.resolve()
    batch_ir = json.loads(args.batch_ir.resolve().read_text())
    plan = batch_ir.get('whole_mesh_replacement_plan') or {}
    events = plan.get('per_event', [])
    internal_rows = plan.get('internal_vertices', [])
    contracts_valid = (
        batch_ir.get('schema') == BATCH_IR_SCHEMA and
        plan.get('schema') == BATCH_PLAN_SCHEMA and
        len(events) == 2 and len(internal_rows) == 2 and
        all(space_position_contract_is_valid(
            value.get('position_contract')) for value in internal_rows)
    )
    expected_positions = (
        [position_float64(value['position_contract'][
            'canonical_position_float64']) for value in internal_rows]
        if contracts_valid else [])

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline1, baseline1_npz = load_case(runtime, 'baseline_omp1')
    baseline8, baseline8_npz = load_case(runtime, 'baseline_omp8')
    batch1, batch1_npz = load_case(runtime, 'batch_omp1')
    batch8, batch8_npz = load_case(runtime, 'batch_omp8')
    baseline_v = baseline1_npz['vertices']
    baseline_f = baseline1_npz['faces']
    baseline_t = baseline1_npz['tags']
    batch_v = batch1_npz['vertices']
    batch_f = batch1_npz['faces']
    batch_t = batch1_npz['tags']

    baseline_faces = oriented_face_multiset(baseline_f)
    batch_faces = oriented_face_multiset(batch_f)
    removed = baseline_faces - batch_faces
    added = batch_faces - baseline_faces
    removed_values = [
        value for value, count in removed.items() for _ in range(count)]
    added_values = [
        value for value, count in added.items() for _ in range(count)]
    new_vertices = list(range(len(baseline_v), len(batch_v)))
    new_vertex_set = set(new_vertices)
    added_face_indices = [
        index for index, face in enumerate(batch_f)
        if set(map(int, face)) & new_vertex_set
    ]
    per_event_face_indices = [
        [index for index in added_face_indices
         if vertex in set(map(int, batch_f[index]))]
        for vertex in new_vertices
    ]
    per_event_boundaries = [
        boundary_edges(batch_f, indices)
        for indices in per_event_face_indices
    ]
    aggregate_boundary = boundary_edges(batch_f, added_face_indices)
    whole_incidence: collections.Counter[tuple[int, int]] = (
        collections.Counter())
    for face in batch_f:
        a, b, c = map(int, face)
        for edge in ((a, b), (b, c), (c, a)):
            whole_incidence[tuple(sorted(edge))] += 1

    expected_removed_per_event = [
        counter_from_rows(value.get('removed_oriented_faces', []))
        for value in events
    ]
    expected_removed = sum(
        expected_removed_per_event, collections.Counter())
    expected_final_faces = baseline_faces - expected_removed
    local_to_global = {
        int(reuse['local_id']): int(reuse['global_id'])
        for reuse in (batch1.get('audit') or {}).get(
            'boundary_reuses', [])
    }
    for event_index, event in enumerate(events):
        internal_local_id = int(event['internal_local_id'])
        local_to_global[internal_local_id] = len(baseline_v) + event_index
        for face in event.get('replacement_faces', []):
            expected_final_faces[canonical_face(np.asarray([
                local_to_global[int(vertex)] for vertex in face
            ], dtype=np.int64))] += 1

    removed_keys = set(removed_values)
    removed_indices = [
        index for index, face in enumerate(baseline_f)
        if canonical_face(face) in removed_keys
    ]
    baseline_pairs = patch_nonincident_intersections(
        baseline_v, baseline_f, removed_indices)
    batch_pairs = patch_nonincident_intersections(
        batch_v, batch_f, added_face_indices)
    baseline_partners = external_partner_keys(
        baseline_f, removed_indices, baseline_pairs)
    batch_partners = external_partner_keys(
        batch_f, added_face_indices, batch_pairs)
    new_partners = sorted(batch_partners - baseline_partners)
    cross_event_pairs = (
        face_set_nonincident_intersections(
            batch_v, batch_f,
            per_event_face_indices[0], per_event_face_indices[1])
        if len(per_event_face_indices) == 2 else [])

    baseline_topology = mesh_topology(baseline_v, baseline_f)
    batch_topology = mesh_topology(batch_v, batch_f)
    topology_keys = (
        'components', 'chi', 'boundary_edges', 'boundary_components')
    audit = batch1.get('audit') or {}
    expected_counts = plan.get('counts', {})
    expected_audit = {
        'expected_suppressions': expected_counts.get('raw_suppressions'),
        'suppressed_triangles': expected_counts.get('raw_suppressions'),
        'expected_boundary_vertices':
            expected_counts.get('boundary_vertices'),
        'resolved_boundary_vertices':
            expected_counts.get('boundary_vertices'),
        'expected_internal_vertices':
            expected_counts.get('internal_vertices'),
        'emitted_internal_vertices':
            expected_counts.get('internal_vertices'),
        'expected_replacement_faces':
            expected_counts.get('replacement_faces'),
        'emitted_replacement_faces':
            expected_counts.get('replacement_faces'),
        'replacement_emissions': 1,
    }
    actual_positions = [
        position_float64(batch_v[index]) for index in new_vertices]
    expected_boundary_sets = [
        {tuple(map(int, edge)) for edge in event.get(
            'global_patch_boundary_edges', [])}
        for event in events
    ]
    checks = {
        'batch_ir_admitted_as_disjoint_closed_support': (
            batch_ir.get('pass') is True and
            batch_ir.get('whole_mesh_splice_ready') is True and
            batch_ir.get('runtime_disposition') ==
            'READY_FOR_ATOMIC_SAME_ROOT_SPLICE' and
            batch_ir.get('composition', {}).get('classification') ==
            'DISJOINT_CLOSED_SUPPORT'
        ),
        'two_position_contracts_valid': contracts_valid,
        'runtime_cases_succeeded': all(
            value.get('pass') is True for value in (
                baseline1, baseline8, batch1, batch8)),
        'baseline_omp_deterministic': all(np.array_equal(
            baseline1_npz[name], baseline8_npz[name])
            for name in ('vertices', 'faces', 'tags')),
        'batch_omp_deterministic': all(np.array_equal(
            batch1_npz[name], batch8_npz[name])
            for name in ('vertices', 'faces', 'tags')),
        'batch_audit_omp_deterministic': (
            batch1.get('audit') == batch8.get('audit')),
        'runtime_coordinate_format_exact': all(
            value.get('audit', {}).get('coordinate_format') ==
            SSP1_COORDINATE_FORMAT for value in (batch1, batch8)),
        'ordinary_vertex_prefix_exact': np.array_equal(
            batch_v[:len(baseline_v)], baseline_v),
        'ordinary_tag_prefix_exact': np.array_equal(
            batch_t[:len(baseline_t)], baseline_t),
        'two_critical_vertices_added': (
            len(batch_v) == len(baseline_v) + 2 and len(new_vertices) == 2),
        'both_critical_positions_bit_exact': (
            len(actual_positions) == len(expected_positions) == 2 and
            all(np.array_equal(actual, expected)
                for actual, expected in zip(
                    actual_positions, expected_positions))),
        'both_internal_tags_match_independent_controls': (
            len(new_vertices) == len(events) == 2 and
            [int(batch_t[index]) for index in new_vertices] ==
            [int(event['expected_internal_tag']) for event in events]),
        'four_source_faces_removed': (
            len(removed_values) == 4 and removed == expected_removed),
        'eight_event_star_faces_added': (
            len(added_values) == 8 and len(added_face_indices) == 8),
        'four_faces_per_event_internal_vertex': (
            len(per_event_face_indices) == 2 and
            all(len(value) == 4 for value in per_event_face_indices) and
            all(all(
                len(set(map(int, batch_f[index])) & new_vertex_set) == 1
                for index in indices)
                for indices in per_event_face_indices)),
        'eight_distinct_source_boundary_vertices_reused': (
            len((batch1.get('audit') or {}).get(
                'boundary_reuses', [])) == 8 and
            len({int(value['global_id']) for value in
                 (batch1.get('audit') or {}).get(
                     'boundary_reuses', [])}) == 8),
        'two_disjoint_four_edge_patch_boundaries': (
            len(per_event_boundaries) == 2 and
            all(len(value) == 4 for value in per_event_boundaries) and
            not (set(per_event_boundaries[0]) &
                 set(per_event_boundaries[1])) and
            len(aggregate_boundary) == 8),
        'runtime_boundaries_match_independent_certificates': (
            len(expected_boundary_sets) ==
            len(per_event_boundaries) == 2 and
            all(set(actual) == expected for actual, expected in
                zip(per_event_boundaries, expected_boundary_sets))),
        'patch_boundary_global_incidence_two': (
            len(aggregate_boundary) == 8 and
            all(whole_incidence[edge] == 2 for edge in aggregate_boundary)),
        'outside_oriented_faces_unchanged': (
            oriented_face_multiset(np.asarray([
                face for face in batch_f
                if not (set(map(int, face)) & new_vertex_set)
            ], dtype=np.int32)) == baseline_faces - removed),
        'atomic_mesh_equals_canonical_union_of_independent_deltas': (
            batch_faces == expected_final_faces),
        'whole_mesh_topology_invariants': all(
            baseline_topology[key] == batch_topology[key]
            for key in topology_keys),
        'no_new_topology_failures': all(
            batch_topology[key] <= baseline_topology[key]
            for key in (
                'nonmanifold_edges', 'duplicate_faces', 'degenerate_faces')),
        'no_cross_event_nonincident_intersections': not cross_event_pairs,
        'no_new_external_intersection_partners': not new_partners,
        'exact_once_runtime_audit': all(
            value is not None and
            int(audit.get(key, -1)) == int(value)
            for key, value in expected_audit.items()),
    }
    passed = all(checks.values())
    result = {
        'schema': 'binoc-same-root-beb1-whole-mesh-validation-v1',
        'pass': passed,
        'verdict': (
            'PASS_SAME_ROOT_BEB1_ATOMIC_BATCH'
            if passed else 'STOP_SAME_ROOT_BEB1_ATOMIC_BATCH'),
        'checks': checks,
        'counts': {
            'baseline_vertices': len(baseline_v),
            'baseline_faces': len(baseline_f),
            'batch_vertices': len(batch_v),
            'batch_faces': len(batch_f),
            'raw_suppressions': expected_counts.get('raw_suppressions'),
            'removed_source_faces': len(removed_values),
            'replacement_faces': len(added_values),
            'reused_boundary_vertices': len(aggregate_boundary),
            'new_internal_vertices': len(new_vertices),
            'new_external_intersection_partners': len(new_partners),
            'cross_event_intersections': len(cross_event_pairs),
        },
        'critical_position_audits': [
            {
                'event_id': events[index]['event_id'],
                'expected_runtime_position':
                    expected_positions[index].tolist(),
                'actual_runtime_position': actual_positions[index].tolist(),
                'actual_runtime_binary32_hex':
                    binary32_hex(actual_positions[index]),
                'maximum_runtime_contract_error': float(np.max(np.abs(
                    actual_positions[index] - expected_positions[index]))),
            }
            for index in range(min(
                len(events), len(expected_positions), len(actual_positions)))
        ],
        'baseline_topology': baseline_topology,
        'batch_topology': batch_topology,
        'per_event_patch_boundary_edges': per_event_boundaries,
        'aggregate_patch_boundary_edges': aggregate_boundary,
        'cross_event_intersection_pairs': cross_event_pairs,
        'new_external_intersection_partners': new_partners,
        'scope': {
            'certified': (
                'one atomic whole-mesh execution of the two disjoint '
                'canonical BEB1 events at exact root 104/5'
            ),
            'not_claimed': (
                'overlapping or shared-boundary simultaneous events'
            ),
        },
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(result['verdict'])
    if not passed:
        for name, value in checks.items():
            if not value:
                print('FAILED:', name)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
