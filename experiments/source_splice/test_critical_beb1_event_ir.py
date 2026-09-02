#!/usr/bin/env python3
'''Pure combinatorial regression for the source-labelled BEB1 half-handle.'''
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

from compile_critical_beb1_event_ir import (
    block_source_boundary_segments,
    registry_face_trace,
    slice_block,
)


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

    production_root = Fraction(104, 5)
    production_epsilon = Fraction(2, 5)
    production_vertices = vertices4.copy()
    production_vertices[:, 3] = [
        float(production_root), 16.0, 20.0, 22.0, 24.0]
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
    production_traces = {
        name: registry_face_trace([row], tau)
        for name, tau in (
            ('lower', production_root - production_epsilon),
            ('upper', production_root + production_epsilon),
        )
    }

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
    }
    if not all(checks.values()):
        for name, passed in checks.items():
            if not passed:
                print('FAILED:', name)
        return 2
    print('PASS_CRITICAL_BEB1_EVENT_IR_COMBINATORICS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
