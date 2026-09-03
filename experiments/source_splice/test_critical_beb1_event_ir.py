#!/usr/bin/env python3
'''Pure combinatorial regression for the source-labelled BEB1 half-handle.'''
from __future__ import annotations

import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

from compile_critical_beb1_event_ir import (
    audit_critical_position_quantization,
    block_source_boundary_segments,
    build_event_star_mapping_cylinder,
    build_whole_mesh_patch_slice,
    patch_boundary_segments,
    registry_face_trace,
    side_trace_affine_audit,
    slice_block,
)
from compile_splice_plans import write_plan
from processed_mesh import HVID, SourceVID
from space_position_contract import (
    SSP1_COORDINATE_FORMAT,
    binary32_hex,
    build_space_position_contract,
    canonical_space_position,
    space_position_contract_is_valid,
)
from theory_audit import complete_production_event_star


def main() -> int:
    vertices4 = np.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    tets = np.asarray([
        [0, 3, 1, 4],
        [0, 3, 4, 2],
    ], dtype=np.int64)
    exact_times = [
        Fraction(0), Fraction(-1), Fraction(-1),
        Fraction(1), Fraction(1),
    ]
    sources = [None, '0:0', '1:0', '2:0', '3:0']
    slices = {
        'lower': slice_block(
            vertices4, tets, exact_times, sources,
            Fraction(-1, 2), 'fixture'),
        'critical': slice_block(
            vertices4, tets, exact_times, sources,
            Fraction(0), 'fixture'),
        'upper': slice_block(
            vertices4, tets, exact_times, sources,
            Fraction(1, 2), 'fixture'),
    }
    event_star_tets, completion = complete_production_event_star(tets)
    complement_tets = np.asarray(
        completion['added_tetrahedra'], dtype=np.int64)
    completed_slices = {
        name: slice_block(
            vertices4, event_star_tets, exact_times, sources, tau, 'fixture')
        for name, tau in (
            ('lower', Fraction(-1, 2)),
            ('critical', Fraction(0)),
            ('upper', Fraction(1, 2)),
        )
    }

    production_root = Fraction(104, 5)
    production_epsilon = Fraction(2, 5)
    production_vertices = np.asarray([
        [-1.181640625, 8.8515625, 0.0, float(production_root)],
        [-1.611328125, 8.59375, 0.0, 16.0],
        [-1.07421875, 9.130859375, 0.0, 20.0],
        [-1.07421875, 8.59375, 0.0, 22.0],
        [-1.611328125, 8.59375, 0.0, 24.0],
    ])
    production_times = [
        production_root, Fraction(16), Fraction(20),
        Fraction(22), Fraction(24),
    ]
    production_sources = [
        None, '243:0', '325:8', '7:10', '155:8']
    row = {
        'logical_incidence_id': 'fixture-incidence',
        'h0': '7:10', 'h1': '243:0',
        'h2': '155:8', 'h3': '325:8',
        't0': '22', 't1': '16', 't2': '24', 't3': '20',
    }
    production_slices = {
        name: slice_block(
            production_vertices, tets, production_times,
            production_sources, tau, 'production-fixture')
        for name, tau in (
            ('lower', production_root - production_epsilon),
            ('upper', production_root + production_epsilon),
        )
    }
    production_completed = {
        name: slice_block(
            production_vertices, event_star_tets, production_times,
            production_sources, tau, 'production-fixture')
        for name, tau in (
            ('lower', production_root - production_epsilon),
            ('critical', production_root),
            ('upper', production_root + production_epsilon),
        )
    }
    production_complement = {
        name: slice_block(
            production_vertices, complement_tets, production_times,
            production_sources, tau, 'production-fixture')
        for name, tau in (
            ('lower', production_root - production_epsilon),
            ('upper', production_root + production_epsilon),
        )
    }
    production_traces = {
        name: registry_face_trace([row], tau)
        for name, tau in (
            ('lower', production_root - production_epsilon),
            ('upper', production_root + production_epsilon),
        )
    }

    # Exact outer boundary recovered by the already validated 102/5 source
    # patch.  Unlike the four-branch critical core it contains 321:8, proving
    # that S_B must be frozen from the ordinary event star rather than guessed
    # from the local half-handle.
    outer_cycle = (
        SourceVID.canonical(HVID(7, 10), HVID(321, 8)),
        SourceVID.canonical(HVID(7, 10), HVID(325, 8)),
        SourceVID.canonical(HVID(155, 8), HVID(325, 8)),
        SourceVID.canonical(HVID(155, 8), HVID(243, 0)),
    )
    hvid_geometry = {
        HVID(7, 10): (np.asarray([-1.07421875, 8.59375, 0.0]), Fraction(22)),
        HVID(321, 8): (np.asarray([-1.07421875, 8.59375, 0.0]), Fraction(18)),
        HVID(325, 8): (np.asarray([-1.07421875, 9.130859375, 0.0]), Fraction(20)),
        HVID(155, 8): (np.asarray([-1.611328125, 8.59375, 0.0]), Fraction(24)),
        HVID(243, 0): (np.asarray([-1.611328125, 8.59375, 0.0]), Fraction(16)),
    }

    def source_position(vertex: SourceVID, tau: Fraction) -> np.ndarray:
        first_position, first_time = hvid_geometry[vertex.first]
        second_position, second_time = hvid_geometry[vertex.second]
        weight = float((tau - first_time) / (second_time - first_time))
        return (1.0 - weight) * first_position + weight * second_position

    outer_patches = {}
    outer_runtime = {}
    for name, tau in (
        ('lower', production_root - production_epsilon),
        ('critical', production_root),
        ('upper', production_root + production_epsilon),
    ):
        positions = {
            vertex: source_position(vertex, tau) for vertex in outer_cycle
        }
        runtime = {
            'cycle': outer_cycle,
            'positions': positions,
            'in_view': {vertex: False for vertex in outer_cycle},
            'faces': (
                (outer_cycle[0], outer_cycle[1], outer_cycle[2]),
                (outer_cycle[0], outer_cycle[2], outer_cycle[3]),
            ),
        }
        patch = {
            'time': {
                'numerator': tau.numerator, 'denominator': tau.denominator},
            'boundary_segments': patch_boundary_segments(outer_cycle),
            'boundary_positions': {
                vertex.text(): positions[vertex].tolist()
                for vertex in outer_cycle
            },
        }
        outer_patches[name] = patch
        outer_runtime[name] = runtime
    side_trace = side_trace_affine_audit(
        outer_patches, {
            'lower': production_root - production_epsilon,
            'critical': production_root,
            'upper': production_root + production_epsilon,
        })
    whole_root = build_whole_mesh_patch_slice(
        outer_patches['critical'], outer_runtime['critical'],
        'production-fixture',
        np.asarray([-1.181640625, 8.8515625, 0.0]))
    mapping_cylinder = build_event_star_mapping_cylinder(
        outer_patches,
        outer_runtime,
        {
            'lower': production_root - production_epsilon,
            'critical': production_root,
            'upper': production_root + production_epsilon,
        },
        'production-fixture',
        np.asarray([-1.181640625, 8.8515625, 0.0]),
    )

    perturbed_theory_center = np.asarray([
        -1.181640626, 8.851562503, 0.0])
    perturbed_contract = build_space_position_contract(
        perturbed_theory_center)
    perturbed_runtime_center = canonical_space_position(
        perturbed_theory_center)
    theory_root = build_whole_mesh_patch_slice(
        outer_patches['critical'], outer_runtime['critical'],
        'production-fixture', perturbed_theory_center)
    runtime_root = build_whole_mesh_patch_slice(
        outer_patches['critical'], outer_runtime['critical'],
        'production-fixture', perturbed_runtime_center)
    quantization_isotopy = audit_critical_position_quantization(
        theory_root, runtime_root, perturbed_contract)

    nondyadic_theory_position = np.asarray([
        0.5859375, -0.05326704545454547, 0.3373579545454546])
    nondyadic_runtime_position = canonical_space_position(
        nondyadic_theory_position)
    position_contract = build_space_position_contract(
        nondyadic_theory_position)
    serialized_contract = json.loads(json.dumps(position_contract))
    corrupted_contract = json.loads(json.dumps(position_contract))
    corrupted_contract['canonical_position_float64'][1] = float(
        nondyadic_theory_position[1])
    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / 'canonical.ssp1'
        write_plan(
            plan_path,
            'position-contract-fixture',
            Fraction(0),
            [],
            outer_cycle,
            [(nondyadic_theory_position, False)],
            [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
            0,
        )
        plan_rows = plan_path.read_text().splitlines()
    coordinate_row = next(
        row for row in plan_rows if row.startswith('COORDINATES '))
    internal_row = next(
        row for row in plan_rows if row.startswith('VERTEX_INTERNAL '))
    serialized_position = np.asarray(
        list(map(float, internal_row.split()[3:6])), dtype=np.float64)

    lower = slices['lower']['topology']
    critical = slices['critical']['topology']
    upper = slices['upper']['topology']
    checks = {
        'lower_two_components': (
            lower['V'], lower['E'], lower['F'],
            lower['chi'], lower['components'],
        ) == (6, 6, 2, 2, 2),
        'upper_one_component': (
            upper['V'], upper['E'], upper['F'],
            upper['chi'], upper['components'],
        ) == (6, 9, 4, 1, 1),
        'critical_pinched_slice': (
            critical['V'], critical['F'], critical['components'],
        ) == (5, 2, 1),
        'regular_sides_manifold_edges': all(
            slices[name]['topology']['nonmanifold_edges'] == 0
            for name in ('lower', 'upper')
        ),
        'source_vid_labels_present': all(
            any(vertex['label']['kind'] == 'source_vid'
                for vertex in slices[name]['vertices'])
            for name in slices
        ),
        'critical_vertex_present_only_at_root': (
            not any(vertex['label']['kind'] == 'critical'
                    for vertex in slices['lower']['vertices']) and
            any(vertex['label']['kind'] == 'critical'
                for vertex in slices['critical']['vertices']) and
            not any(vertex['label']['kind'] == 'critical'
                    for vertex in slices['upper']['vertices'])
        ),
        'half_handle_side_trace_is_not_falsely_admitted': all(
            not slices[name]['source_boundary_complete']
            for name in slices
        ),
        'four_unresolved_side_edges_per_slice': all(
            len(slices[name]['unresolved_boundary_edges']) == 4
            for name in slices
        ),
        'lower_source_interface_matches_registry': (
            block_source_boundary_segments(production_slices['lower']) ==
            production_traces['lower']['canonical_segments']
        ),
        'upper_source_interface_mismatch_is_visible': (
            block_source_boundary_segments(production_slices['upper']) !=
            production_traces['upper']['canonical_segments']
        ),
        'complement_supplies_upper_registry_interface': (
            block_source_boundary_segments(production_complement['upper']) ==
            production_traces['upper']['canonical_segments']
        ),
        'critical_link_disk_completed_to_sphere': (
            completion['critical_link']['is_sphere'] is True and
            completion['critical_side_faces_remaining'] == 0
        ),
        'completed_event_star_has_four_tetrahedra': (
            event_star_tets.shape == (4, 4)
        ),
        'completed_slices_are_disks': all(
            value['topology']['components'] == 1 and
            value['topology']['chi'] == 1 and
            value['topology']['boundary_loops'] == 1 and
            value['topology']['boundary_edges'] == 4
            for value in completed_slices.values()
        ),
        'four_critical_side_edges_eliminated': all(
            value['source_boundary_complete'] and
            not value['unresolved_boundary_edges']
            for value in completed_slices.values()
        ),
        'completed_boundary_is_source_quadrilateral': all(
            len(block_source_boundary_segments(value)) == 4
            for value in production_completed.values()
        ),
        'ordinary_side_trace_is_fixed_and_affine': (
            side_trace['regular'] is True and
            side_trace['affine_trajectory_error'] <=
            side_trace['tolerance']
        ),
        'ordinary_side_trace_is_not_core_boundary_guess': (
            outer_patches['critical']['boundary_segments'] !=
            block_source_boundary_segments(production_completed['critical'])
        ),
        'whole_mesh_root_fan_is_nondegenerate_disk': (
            whole_root['topology']['components'] == 1 and
            whole_root['topology']['chi'] == 1 and
            whole_root['topology']['boundary_edges'] == 4 and
            whole_root['topology']['boundary_loops'] == 1 and
            whole_root['minimum_double_area'] > 1e-12 and
            whole_root['orientation_coherent'] and
            len(whole_root['faces']) == 4 and
            whole_root['source_boundary_complete']
        ),
        'explicit_event_star_mapping_cylinder_passes': (
            mapping_cylinder['pass'] is True and
            mapping_cylinder['verdict'] ==
            'PASS_BEB1_DOUBLE_MAPPING_CYLINDER' and
            len(mapping_cylinder['vertices4']) == 15 and
            len(mapping_cylinder['tetrahedra']) == 24 and
            mapping_cylinder['minimum_gram_volume'] > 1e-12 and
            mapping_cylinder[
                'volume_audit']['valid_relative_3_manifold'] is True and
            len(mapping_cylinder['side_trace_faces']) == 16 and
            mapping_cylinder['checks'][
                'side_trace_avoids_all_centers'] is True and
            mapping_cylinder['checks'][
                'middle_critical_vertex_not_on_boundary'] is True and
            mapping_cylinder['critical_side_edges_remaining'] == 0
        ),
        'spaceT_quantization_has_fixed_boundary_disk_isotopy': (
            quantization_isotopy['pass'] is True and
            quantization_isotopy['verdict'] ==
            'PASS_CRITICAL_POSITION_QUANTIZATION_ISOTOPY' and
            quantization_isotopy['displacement_norm'] > 0.0 and
            quantization_isotopy['checks'][
                'straight_line_center_motion_nondegenerate'] is True
        ),
        'nondyadic_space_position_is_canonical_binary32': (
            not np.array_equal(
                nondyadic_theory_position, nondyadic_runtime_position) and
            np.array_equal(
                nondyadic_runtime_position,
                np.asarray([
                    0.5859375,
                    -0.05326704680919647,
                    0.3373579680919647,
                ])) and
            binary32_hex(nondyadic_runtime_position) == [
                '3f160000', 'bd5a2e8c', '3eacba2f']
        ),
        'space_position_contract_round_trips_and_fails_closed': (
            space_position_contract_is_valid(serialized_contract) and
            not space_position_contract_is_valid(corrupted_contract)
        ),
        'ssp1_serializes_only_canonical_space_position': (
            coordinate_row == f'COORDINATES {SSP1_COORDINATE_FORMAT}' and
            np.array_equal(
                serialized_position, nondyadic_runtime_position)
        ),
    }
    if not all(checks.values()):
        for name, passed in checks.items():
            if not passed:
                print('FAILED:', name)
        return 2
    print('PASS_CRITICAL_BEB1_EVENT_STAR_CLOSURE_COMBINATORICS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
