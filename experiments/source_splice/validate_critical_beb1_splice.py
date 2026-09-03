#!/usr/bin/env python3
'''Validate the admitted BEB1 root slice as a whole-mesh replacement.'''
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from runtime_common import (
    mesh_topology,
    oriented_face_multiset,
    patch_nonincident_intersections,
)
from space_position_contract import (
    EVENT_IR_SCHEMA,
    SSP1_COORDINATE_FORMAT,
    binary32_hex,
    plan_position_contract_is_valid,
    position_float64,
)


def load_case(root: Path, name: str):
    payload = json.loads((root / f'{name}.json').read_text())
    arrays = np.load(root / f'{name}.npz')
    return payload, arrays


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-results', type=Path, required=True)
    parser.add_argument('--event-ir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_results.resolve()
    event_ir = json.loads(args.event_ir.resolve().read_text())
    plan = event_ir.get('whole_mesh_replacement_plan') or {}
    position_contract = plan.get('critical_position_contract') or {}
    position_contract_valid = (
        event_ir.get('schema') == EVENT_IR_SCHEMA and
        plan_position_contract_is_valid(plan) and
        event_ir.get('event_star_geometry', {}).get(
            'critical_position_contract') == position_contract and
        event_ir.get('admission', {}).get(
            'runtime_space_position_contract_ready') is True
    )
    if position_contract_valid:
        expected_critical_position = position_float64(
            position_contract['canonical_position_float64'])
        theory_critical_position = position_float64(
            position_contract['theory_position_float64'])
    else:
        expected_critical_position = None
        theory_critical_position = None
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    baseline1, baseline1_npz = load_case(runtime, 'baseline_omp1')
    baseline8, baseline8_npz = load_case(runtime, 'baseline_omp8')
    critical1, critical1_npz = load_case(runtime, 'critical_omp1')
    critical8, critical8_npz = load_case(runtime, 'critical_omp8')
    baseline_v = baseline1_npz['vertices']
    baseline_f = baseline1_npz['faces']
    baseline_t = baseline1_npz['tags']
    critical_v = critical1_npz['vertices']
    critical_f = critical1_npz['faces']
    critical_t = critical1_npz['tags']
    runtime_critical_position = (
        position_float64(critical_v[len(baseline_v)])
        if len(critical_v) == len(baseline_v) + 1 else None
    )

    baseline_faces = oriented_face_multiset(baseline_f)
    critical_faces = oriented_face_multiset(critical_f)
    removed = baseline_faces - critical_faces
    added = critical_faces - baseline_faces
    removed_values = [
        value for value, count in removed.items() for _ in range(count)]
    added_values = [
        value for value, count in added.items() for _ in range(count)]
    new_vertex = len(baseline_v)
    added_face_indices = [
        index for index, face in enumerate(critical_f)
        if new_vertex in set(map(int, face))
    ]
    added_edges: collections.Counter[tuple[int, int]] = collections.Counter()
    for index in added_face_indices:
        a, b, c = map(int, critical_f[index])
        for edge in ((a, b), (b, c), (c, a)):
            added_edges[tuple(sorted(edge))] += 1
    patch_boundary = sorted(
        edge for edge, count in added_edges.items() if count == 1)
    whole_incidence: collections.Counter[tuple[int, int]] = collections.Counter()
    for face in critical_f:
        a, b, c = map(int, face)
        for edge in ((a, b), (b, c), (c, a)):
            whole_incidence[tuple(sorted(edge))] += 1

    removed_keys = set(removed_values)
    removed_indices = [
        index for index, face in enumerate(baseline_f)
        if min(
            tuple(map(int, face[offset:].tolist() + face[:offset].tolist()))
            for offset in range(3)
        ) in removed_keys
    ]
    baseline_pairs = patch_nonincident_intersections(
        baseline_v, baseline_f, removed_indices)
    critical_pairs = patch_nonincident_intersections(
        critical_v, critical_f, added_face_indices)
    baseline_partners = external_partner_keys(
        baseline_f, removed_indices, baseline_pairs)
    critical_partners = external_partner_keys(
        critical_f, added_face_indices, critical_pairs)
    new_partners = sorted(critical_partners - baseline_partners)

    baseline_topology = mesh_topology(baseline_v, baseline_f)
    critical_topology = mesh_topology(critical_v, critical_f)
    topology_keys = (
        'components', 'chi', 'boundary_edges', 'boundary_components')
    expected_suppressions = int(plan.get('raw_suppression_count', -1))
    expected_boundary = len(plan.get('boundary_cycle', []))
    audit = critical1.get('audit') or {}
    expected_audit = {
        'expected_suppressions': expected_suppressions,
        'suppressed_triangles': expected_suppressions,
        'expected_boundary_vertices': expected_boundary,
        'resolved_boundary_vertices': expected_boundary,
        'expected_internal_vertices': 1,
        'emitted_internal_vertices': 1,
        'expected_replacement_faces': 4,
        'emitted_replacement_faces': 4,
        'replacement_emissions': 1,
    }
    checks = {
        'event_ir_admitted': (
            event_ir.get('whole_mesh_splice_ready') is True and
            event_ir.get('runtime_disposition') ==
            'READY_FOR_WHOLE_MESH_SPLICE' and
            event_ir.get(
                'admission', {}).get('mapping_cylinder_ready') is True and
            event_ir.get('admission', {}).get(
                'runtime_space_position_contract_ready') is True and
            event_ir.get('admission', {}).get(
                'critical_position_quantization_isotopy_ready') is True and
            event_ir.get('event_star_geometry', {}).get(
                'critical_position_quantization_audit', {}).get(
                    'pass') is True and
            event_ir.get('event_star_geometry', {}).get(
                'mapping_cylinder', {}).get(
                    'critical_side_edges_remaining') == 0
        ),
        'critical_position_contract_valid': position_contract_valid,
        'runtime_cases_succeeded': all(value['pass'] for value in (
            baseline1, baseline8, critical1, critical8)),
        'baseline_omp_deterministic': all(np.array_equal(
            baseline1_npz[name], baseline8_npz[name])
            for name in ('vertices', 'faces', 'tags')),
        'critical_omp_deterministic': all(np.array_equal(
            critical1_npz[name], critical8_npz[name])
            for name in ('vertices', 'faces', 'tags')),
        'critical_audit_omp_deterministic': (
            critical1.get('audit') == critical8.get('audit')),
        'runtime_coordinate_format_exact': all(
            value.get('audit', {}).get('coordinate_format') ==
            SSP1_COORDINATE_FORMAT
            for value in (critical1, critical8)
        ),
        'ordinary_vertex_prefix_exact': np.array_equal(
            critical_v[:len(baseline_v)], baseline_v),
        'ordinary_tag_prefix_exact': np.array_equal(
            critical_t[:len(baseline_t)], baseline_t),
        'one_critical_vertex_added': len(critical_v) == len(baseline_v) + 1,
        'critical_position_exact': (
            position_contract_valid and
            runtime_critical_position is not None and
            expected_critical_position is not None and
            np.array_equal(
                runtime_critical_position,
                expected_critical_position)
        ),
        'two_source_faces_removed': len(removed_values) == 2,
        'four_event_star_faces_added': (
            len(added_values) == 4 and len(added_face_indices) == 4),
        'all_added_faces_use_critical_vertex': (
            len(added_face_indices) == 4 and
            all(new_vertex in set(map(int, critical_f[index]))
                for index in added_face_indices)
        ),
        'four_source_boundary_vertices_reused': (
            len(patch_boundary) == 4 and
            len({
                value for face in added_values for value in face
                if value != new_vertex
            }) == 4
        ),
        'critical_vertex_is_internal': (
            all(new_vertex not in edge for edge in patch_boundary)
        ),
        'patch_boundary_global_incidence_two': (
            len(patch_boundary) == 4 and
            all(whole_incidence[edge] == 2 for edge in patch_boundary)
        ),
        'outside_oriented_faces_unchanged': (
            oriented_face_multiset(np.asarray([
                face for face in critical_f
                if new_vertex not in set(map(int, face))
            ], dtype=np.int32)) == baseline_faces - removed
        ),
        'whole_mesh_topology_invariants': all(
            baseline_topology[key] == critical_topology[key]
            for key in topology_keys
        ),
        'no_new_topology_failures': all(
            critical_topology[key] <= baseline_topology[key]
            for key in (
                'nonmanifold_edges', 'duplicate_faces', 'degenerate_faces')
        ),
        'no_new_nonincident_intersections': not new_partners,
        'exact_once_runtime_audit': all(
            int(audit.get(key, -1)) == value
            for key, value in expected_audit.items()
        ),
    }
    result = {
        'schema': 'binoc-critical-beb1-whole-mesh-validation-v2',
        'pass': all(checks.values()),
        'verdict': (
            'PASS_CRITICAL_BEB1_WHOLE_MESH_SPLICE'
            if all(checks.values()) else
            'STOP_CRITICAL_BEB1_WHOLE_MESH_SPLICE'
        ),
        'checks': checks,
        'critical_position_audit': {
            'contract': position_contract,
            'theory_position_float64': (
                theory_critical_position.tolist()
                if theory_critical_position is not None else None
            ),
            'expected_runtime_position_float64': (
                expected_critical_position.tolist()
                if expected_critical_position is not None else None
            ),
            'actual_runtime_position_float64': (
                runtime_critical_position.tolist()
                if runtime_critical_position is not None else None
            ),
            'actual_runtime_binary32_hex': (
                binary32_hex(runtime_critical_position)
                if runtime_critical_position is not None else None
            ),
            'maximum_runtime_contract_error': (
                float(np.max(np.abs(
                    runtime_critical_position - expected_critical_position)))
                if runtime_critical_position is not None and
                expected_critical_position is not None else None
            ),
        },
        'counts': {
            'baseline_vertices': len(baseline_v),
            'baseline_faces': len(baseline_f),
            'critical_vertices': len(critical_v),
            'critical_faces': len(critical_f),
            'raw_suppressions': expected_suppressions,
            'removed_source_faces': len(removed_values),
            'replacement_faces': len(added_values),
            'reused_boundary_vertices': expected_boundary,
            'new_internal_vertices': 1,
            'new_external_intersection_partners': len(new_partners),
        },
        'baseline_topology': baseline_topology,
        'critical_topology': critical_topology,
        'patch_boundary_edges': patch_boundary,
        'new_external_intersection_partners': new_partners,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(result['verdict'])
    if not result['pass']:
        for name, passed in checks.items():
            if not passed:
                print('FAILED:', name)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
