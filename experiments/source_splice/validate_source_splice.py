#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from runtime_common import (
    mesh_topology, oriented_face_multiset,
    patch_nonincident_intersections,
)
from space_position_contract import SSP1_COORDINATE_FORMAT


def load_case(root: Path, name: str):
    payload = json.loads((root / f'{name}.json').read_text())
    arrays = np.load(root / f'{name}.npz') if (root / f'{name}.npz').is_file() else None
    return payload, arrays


def counter_delta(
    first: collections.Counter, second: collections.Counter
) -> collections.Counter:
    return first - second


def audit_counts(payload: dict, expected: dict[str, int]) -> bool:
    audit = payload.get('audit') or {}
    return all(int(audit.get(name, -1)) == value for name, value in expected.items())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-results', type=Path, required=True)
    parser.add_argument('--plans', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_results.resolve()
    plan_summary = json.loads((args.plans / 'plan_summary.json').read_text())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    baseline1, baseline1_npz = load_case(runtime, 'baseline_omp1')
    identity1, identity1_npz = load_case(runtime, 'identity_omp1')
    star1, star1_npz = load_case(runtime, 'star_omp1')
    baseline8, baseline8_npz = load_case(runtime, 'baseline_omp8')
    identity8, identity8_npz = load_case(runtime, 'identity_omp8')
    star8, star8_npz = load_case(runtime, 'star_omp8')
    negative = {
        name: json.loads((runtime / f'{name}.json').read_text())
        for name in (
            'wrong_time', 'missing_ref', 'bad_boundary', 'bad_topology',
            'bad_coordinate_format')
    }
    assert all(value is not None for value in (
        baseline1_npz, identity1_npz, star1_npz,
        baseline8_npz, identity8_npz, star8_npz,
    ))
    baseline_v = baseline1_npz['vertices']
    baseline_f = baseline1_npz['faces']
    baseline_t = baseline1_npz['tags']
    identity_v = identity1_npz['vertices']
    identity_f = identity1_npz['faces']
    identity_t = identity1_npz['tags']
    star_v = star1_npz['vertices']
    star_f = star1_npz['faces']
    star_t = star1_npz['tags']

    expected_suppressions = int(plan_summary['selected_suppression_occurrences'])
    expected_source_vertices = len(plan_summary['boundary_cycle'])
    baseline_faces = oriented_face_multiset(baseline_f)
    identity_faces = oriented_face_multiset(identity_f)
    star_faces = oriented_face_multiset(star_f)
    removed = counter_delta(baseline_faces, star_faces)
    added = counter_delta(star_faces, baseline_faces)
    removed_values = [value for value, count in removed.items() for _ in range(count)]
    added_values = [value for value, count in added.items() for _ in range(count)]
    new_vertex = len(baseline_v)
    added_face_indices = [
        index for index, face in enumerate(star_f)
        if new_vertex in set(map(int, face))
    ]
    added_boundary_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    for index in added_face_indices:
        a, b, c = map(int, star_f[index])
        for edge in ((a, b), (b, c), (c, a)):
            added_boundary_counts[tuple(sorted(edge))] += 1
    patch_boundary = [edge for edge, count in added_boundary_counts.items() if count == 1]
    whole_incidence: collections.Counter[tuple[int, int]] = collections.Counter()
    for face in star_f:
        a, b, c = map(int, face)
        for edge in ((a, b), (b, c), (c, a)):
            whole_incidence[tuple(sorted(edge))] += 1
    intersection_pairs = patch_nonincident_intersections(
        star_v, star_f, added_face_indices)
    removed_face_indices = []
    removed_face_keys = {canonical for canonical in removed_values}
    for index, face in enumerate(baseline_f):
        canonical = tuple(min(
            (tuple(map(int, face[offset:].tolist() + face[:offset].tolist()))
             for offset in range(3))
        ))
        if canonical in removed_face_keys:
            removed_face_indices.append(index)
    baseline_intersection_pairs = patch_nonincident_intersections(
        baseline_v, baseline_f, removed_face_indices)

    def external_partner_keys(
        faces: np.ndarray, patch_indices: list[int],
        pairs: list[tuple[int, int]],
    ) -> set[tuple[int, int, int]]:
        patch = set(patch_indices)
        result = set()
        for first, second in pairs:
            other = second if first in patch else first
            result.add(tuple(sorted(map(int, faces[other]))))
        return result

    baseline_intersection_partners = external_partner_keys(
        baseline_f, removed_face_indices, baseline_intersection_pairs)
    star_intersection_partners = external_partner_keys(
        star_f, added_face_indices, intersection_pairs)
    new_intersection_partners = sorted(
        star_intersection_partners - baseline_intersection_partners)

    baseline_topology = mesh_topology(baseline_v, baseline_f)
    star_topology = mesh_topology(star_v, star_f)
    topology_invariant_keys = (
        'components', 'chi', 'boundary_edges', 'boundary_components',
    )
    checks = {
        'runtime_cases_succeeded': all(
            value['pass'] for value in (
                baseline1, identity1, star1, baseline8, identity8, star8,
            )
        ),
        'identity_vertices_exact': np.array_equal(identity_v, baseline_v),
        'identity_faces_exact': np.array_equal(identity_f, baseline_f),
        'identity_tags_exact': np.array_equal(identity_t, baseline_t),
        'identity_hash_exact': identity1['mesh_sha256'] == baseline1['mesh_sha256'],
        'identity_oriented_face_roundtrip': identity_faces == baseline_faces,
        'identity_exact_once_audit': audit_counts(identity1, {
            'expected_suppressions': expected_suppressions,
            'suppressed_triangles': expected_suppressions,
            'expected_boundary_vertices': expected_source_vertices,
            'resolved_boundary_vertices': expected_source_vertices,
            'expected_internal_vertices': 0,
            'emitted_internal_vertices': 0,
            'expected_replacement_faces': 2,
            'emitted_replacement_faces': 2,
            'replacement_emissions': 1,
        }),
        'star_ordinary_vertex_prefix_exact': np.array_equal(
            star_v[:len(baseline_v)], baseline_v),
        'star_ordinary_tag_prefix_exact': np.array_equal(
            star_t[:len(baseline_t)], baseline_t),
        'star_one_internal_vertex': len(star_v) == len(baseline_v) + 1,
        'star_two_to_four_face_count': len(star_f) == len(baseline_f) + 2,
        'star_removed_exactly_two': len(removed_values) == 2,
        'star_added_exactly_four': len(added_values) == 4,
        'star_new_faces_share_one_internal_vertex': (
            len(added_face_indices) == 4 and
            all(new_vertex in set(map(int, star_f[index])) for index in added_face_indices)
        ),
        'star_boundary_ids_reuse_ordinary_vertices': (
            len({value for face in added_values for value in face if value != new_vertex}) == 4
            and all(value < new_vertex for face in added_values for value in face if value != new_vertex)
        ),
        'star_outside_oriented_faces_unchanged': (
            oriented_face_multiset(np.asarray([
                face for face in star_f if new_vertex not in set(map(int, face))
            ], dtype=np.int32)) == counter_delta(baseline_faces, removed)
        ),
        'star_four_patch_boundary_edges': len(patch_boundary) == 4,
        'star_patch_boundary_global_incidence_two': (
            len(patch_boundary) == 4 and
            all(whole_incidence[edge] == 2 for edge in patch_boundary)
        ),
        'star_topology_invariants': all(
            baseline_topology[key] == star_topology[key]
            for key in topology_invariant_keys
        ),
        'star_no_new_topology_failures': all(
            star_topology[key] <= baseline_topology[key]
            for key in ('nonmanifold_edges', 'duplicate_faces', 'degenerate_faces')
        ),
        'star_no_new_nonincident_intersections': not new_intersection_partners,
        'star_exact_once_audit': audit_counts(star1, {
            'expected_suppressions': expected_suppressions,
            'suppressed_triangles': expected_suppressions,
            'expected_boundary_vertices': expected_source_vertices,
            'resolved_boundary_vertices': expected_source_vertices,
            'expected_internal_vertices': 1,
            'emitted_internal_vertices': 1,
            'expected_replacement_faces': 4,
            'emitted_replacement_faces': 4,
            'replacement_emissions': 1,
        }),
        'baseline_omp_deterministic': all(np.array_equal(
            baseline1_npz[name], baseline8_npz[name])
            for name in ('vertices', 'faces', 'tags')),
        'identity_omp_deterministic': all(np.array_equal(
            identity1_npz[name], identity8_npz[name])
            for name in ('vertices', 'faces', 'tags')),
        'star_omp_deterministic': all(np.array_equal(
            star1_npz[name], star8_npz[name])
            for name in ('vertices', 'faces', 'tags')),
        'identity_audit_omp_deterministic': identity1['audit'] == identity8['audit'],
        'star_audit_omp_deterministic': star1['audit'] == star8['audit'],
        'ssp1_coordinate_format_exact': (
            plan_summary.get('ssp1_coordinate_format') ==
            SSP1_COORDINATE_FORMAT and
            all(
                value.get('audit', {}).get('coordinate_format') ==
                SSP1_COORDINATE_FORMAT
                for value in (identity1, identity8, star1, star8)
            )
        ),
        'bad_topology_fail_closed': (
            negative['bad_topology']['pass'] and
            'validate_plan_topology' in negative['bad_topology']['error']
        ),
        'bad_coordinate_format_fail_closed': (
            negative['bad_coordinate_format']['pass'] and
            'coordinate format' in
            negative['bad_coordinate_format']['error']
        ),
        'wrong_time_fail_closed': (
            negative['wrong_time']['pass'] and
            'exact time does not match' in negative['wrong_time']['error']
        ),
        'missing_owner_fail_closed': (
            negative['missing_ref']['pass'] and
            ('suppression count mismatch' in negative['missing_ref']['error'] or
             'source triangle was not suppressed' in negative['missing_ref']['error'])
        ),
        'unknown_boundary_fail_closed': (
            negative['bad_boundary']['pass'] and
            ('was not emitted by the ordinary mesh' in negative['bad_boundary']['error'] or
             'was not produced by ordinary mesh' in negative['bad_boundary']['error'])
        ),
        'all_raw_owners_suppressed_once': (
            expected_suppressions == len(plan_summary['suppression_refs']) and
            identity1['audit']['suppressed_triangles'] == expected_suppressions and
            star1['audit']['suppressed_triangles'] == expected_suppressions
        ),
        'beb1_boundary_interface_frozen': (
            (args.plans / 'beb1_slice_interface.json').is_file()
            and json.loads((args.plans / 'beb1_slice_interface.json').read_text())[
                'suppressed_source_owners'] == expected_suppressions
        ),
    }
    result = {
        'schema': 'binoc-source-face-suppression-boundary-gluing-validation-v1',
        'pass': all(checks.values()),
        'verdict': (
            'PASS_SOURCE_FACE_SUPPRESSION_AND_BOUNDARY_GLUING'
            if all(checks.values()) else
            'STOP_SOURCE_FACE_SUPPRESSION_AND_BOUNDARY_GLUING'
        ),
        'checks': checks,
        'counts': {
            'baseline_vertices': len(baseline_v),
            'baseline_faces': len(baseline_f),
            'star_vertices': len(star_v),
            'star_faces': len(star_f),
            'raw_suppression_owners': expected_suppressions,
            'unique_source_faces_suppressed': len(removed_values),
            'replacement_faces': len(added_values),
            'reused_global_boundary_vertices': 4,
            'new_internal_vertices': 1,
            'baseline_patch_intersection_pairs': len(baseline_intersection_pairs),
            'star_patch_intersection_pairs': len(intersection_pairs),
            'new_external_intersection_partners': len(new_intersection_partners),
        },
        'baseline_topology': baseline_topology,
        'identity_topology': mesh_topology(identity_v, identity_f),
        'star_topology': star_topology,
        'removed_oriented_faces': removed_values,
        'added_oriented_faces': added_values,
        'patch_boundary_edges': patch_boundary,
        'patch_boundary_incidence': {
            str(edge): whole_incidence[edge] for edge in patch_boundary
        },
        'baseline_nonincident_intersection_pairs': baseline_intersection_pairs,
        'star_nonincident_intersection_pairs': intersection_pairs,
        'baseline_external_intersection_partners': sorted(baseline_intersection_partners),
        'star_external_intersection_partners': sorted(star_intersection_partners),
        'new_external_intersection_partners': new_intersection_partners,
        'scope': {
            'completed': (
                'one production-derived event-star disk in a full ordinary mesh; '
                'all raw owners are suppressed exactly once and replacement '
                'boundary vertices reuse ordinary global source-edge IDs'
            ),
            'identity_case': 'byte-exact suppress-and-reemit full-mesh round trip',
            'nontrivial_case': (
                '2-to-4 PL subdivision by a midpoint on the old shared source edge'
            ),
            'not_claimed': (
                'critical-time BEB1 half-handle source replacement, production '
                'endpoint block, natural mixed batch, or arbitrary global theorem'
            ),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(result['verdict'])
    if not result['pass']:
        for key, value in checks.items():
            if not value:
                print(f'FAILED: {key}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
