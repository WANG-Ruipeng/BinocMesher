#!/usr/bin/env python3
"""Fail-fast, offline C1-lite fan preflight. Never emits runtime plans.

C1-lite is a route name, not a claim of temporal C^1 smoothness. Source
positions are obtained anew from the ordinary cache at each exact probe.
Finite probes cannot certify owner stability or whole-window embedding.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'source_splice'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from run_geometry_gate import GateStop, require, heights_at_xy, error_stats


def rational(value):
    if isinstance(value, dict):
        return Fraction(int(value['numerator']), int(value['denominator']))
    require(isinstance(value, (str, int)), 'Times must be exact rationals, not floats.')
    return Fraction(value)


def fraction_json(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def exact_vector(values):
    return tuple(Fraction.from_float(float(value)) for value in values)


def point_on_segment_exact(point, first, second):
    p, a, b = map(exact_vector, (point, first, second))
    direction = tuple(y-x for x, y in zip(a, b))
    offset = tuple(y-x for x, y in zip(a, p))
    nonzero = [i for i in range(3) if direction[i]]
    if not nonzero:
        return p == a
    weight = offset[nonzero[0]] / direction[nonzero[0]]
    return 0 <= weight <= 1 and all(offset[i] == weight*direction[i] for i in range(3))


def endpoint_contract(first, second):
    """Derive each midpoint from the source diagonal in its own precision."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    midpoint_exact = tuple((x+y)/2 for x, y in zip(exact_vector(a), exact_vector(b)))
    center64 = np.asarray([float(value) for value in midpoint_exact])
    source32 = np.asarray([a, b]).astype(np.float32).astype(np.float64)
    runtime_a, runtime_b = source32
    midpoint32_exact = tuple((x+y)/2 for x, y in
                             zip(exact_vector(runtime_a), exact_vector(runtime_b)))
    center32 = np.asarray([float(value) for value in midpoint32_exact],
                          dtype=np.float32).astype(np.float64)
    direction = runtime_b-runtime_a
    require(float(direction @ direction) > 0, 'Endpoint shared diagonal is degenerate.')
    weight = float(np.clip((center32-runtime_a) @ direction / (direction @ direction), 0, 1))
    return {
        'ideal_midpoint_exact': [fraction_json(value) for value in midpoint_exact],
        'quantized_source_midpoint_exact': [fraction_json(value) for value in midpoint32_exact],
        'ideal_midpoint_on_shared_diagonal_exact': True,
        'center_binary64': center64.tolist(),
        'center_binary32': center32.tolist(),
        'source_diagonal_binary64': [a.tolist(), b.tolist()],
        'source_diagonal_binary32': source32.tolist(),
        'binary64_center_on_shared_diagonal_exact': point_on_segment_exact(center64, a, b),
        'binary32_center_on_ideal_diagonal_exact': point_on_segment_exact(center32, a, b),
        'binary32_center_on_shared_diagonal_exact': point_on_segment_exact(center32, runtime_a, runtime_b),
        'binary32_distance_to_shared_diagonal': float(np.linalg.norm(center32-(runtime_a+weight*direction))),
        'meaning': 'Binary64 center uses binary64 source; binary32 center is computed after source quantization. Each is checked against its own diagonal. No C++ runtime execution was performed.',
    }


def center_at(tau, levels, anchors, critical=True):
    lower, root, upper = levels
    require(lower < root < upper and lower <= tau <= upper, 'Invalid or out-of-window time.')
    if critical:
        left, right = (0, 1) if tau <= root else (1, 2)
    else:
        left, right = 0, 2
    if tau == levels[left]:
        return np.asarray(anchors[left], dtype=float).copy()
    if tau == levels[right]:
        return np.asarray(anchors[right], dtype=float).copy()
    weight = float((tau-levels[left])/(levels[right]-levels[left]))
    return (1-weight)*np.asarray(anchors[left]) + weight*np.asarray(anchors[right])


def fan_triangles(boundary, center):
    boundary = np.asarray(boundary, dtype=float)
    require(boundary.shape == (4, 3), 'Pilot supports exactly four source boundary vertices.')
    return np.asarray([[boundary[i], boundary[(i+1) % 4], center] for i in range(4)])


def require_convex_graph(boundary, triangles):
    """Sufficient, deliberately restricted projected-graph condition."""
    boundary = np.asarray(boundary, dtype=float)
    edges = np.roll(boundary[:, :2], -1, axis=0)-boundary[:, :2]
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0]*following[:, 1]-edges[:, 1]*following[:, 0]
    floor = 1e-12 * max(float(np.max(np.sum(edges**2, axis=1))), 1e-30)
    require(bool(np.all(turns > floor) or np.all(turns < -floor)),
            'Projected boundary is not a strictly convex quadrilateral; graph pilot is unsupported.')
    a = triangles[:, 1, :2]-triangles[:, 0, :2]
    b = triangles[:, 2, :2]-triangles[:, 0, :2]
    areas = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    require(bool(np.all(areas*np.sign(turns[0]) > floor)),
            'Patch has projected degeneracy, reversed sectors, or a center outside the graph footprint.')


def compare_heights(boundary, baseline, c1, naive, terrain, grid):
    for triangles in (baseline, c1, naive):
        require_convex_graph(boundary, triangles)
    low, high = np.min(boundary, axis=0), np.max(boundary, axis=0)
    tolerance = 1e-9 * max(1., float(np.linalg.norm(high-low)))
    axis = (np.arange(grid)+.5)/grid
    gx, gy = np.meshgrid(low[0]+axis*(high[0]-low[0]), low[1]+axis*(high[1]-low[1]))
    xy = np.column_stack((gx.ravel(), gy.ravel()))
    heights = {name: heights_at_xy(triangles, xy, tolerance)
               for name, triangles in [('baseline', baseline), ('c1', c1), ('naive', naive)]}
    coverage = {name: np.isfinite(values) for name, values in heights.items()}
    require(all(np.array_equal(coverage['baseline'], mask) for mask in coverage.values()),
            'Projected coverage differs; never compare only the intersection.')
    mask = coverage['baseline']
    require(int(mask.sum()) >= 100, 'Too few reference samples.')
    reference = -np.asarray(terrain(np.column_stack((xy[mask], np.zeros(int(mask.sum()))))))
    require(reference.shape == (int(mask.sum()),) and np.all(np.isfinite(reference)),
            'Invalid independent height-field reference.')
    stats = {name: error_stats(np.abs(values[mask]-reference)) for name, values in heights.items()}
    return {'grid': grid, 'samples': int(mask.sum()), 'same_sampled_coverage': True,
            'tolerance': tolerance, **stats}


def stable_cycle(reference, candidate):
    return len(reference) == len(candidate) and any(
        list(reference) == list(candidate[i:])+list(candidate[:i]) for i in range(len(candidate)))


def additional_times(path, lower, upper):
    if path is None:
        return [], None
    payload = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(payload.get('mapping_source'), str) and bool(payload['mapping_source'].strip()),
            'Natural-frame times need an explicit independently verified mapping_source.')
    values = [rational(value) for value in payload['times']]
    require(all(lower <= value <= upper for value in values), 'Natural-frame input has out-of-window times.')
    return values, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('event-root', 'cache-root', 'profile-source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--times-json', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {'schema': 'binoc-c1-lite-reference-preflight-v2', 'status': 'RUNNING',
              'endpoint_evaluation_order': 'Quantize source first, then compute prospective binary32 center; binary64 reference remains separate.',
              'measurement_coordinate_model': 'Binary64 parser boundary and binary64 center trajectory; NOT a binary32 runtime quality evaluation.',
              'runtime_plan_emitted': False, 'render_started': False,
              'continuous_window_certified': False, 'endpoint_audits': {}, 'measurements': [],
              'limitations': ['Finite source samples do not prove interval owner or branch stability.',
                             'No patch/exterior continuous intersection certificate.',
                             'No production whole-mesh splice, attribute, or OMP execution.',
                             'Height error is not Hausdorff distance or warped SSIM.',
                             'C1-lite denotes a route, not temporal derivative continuity.'],
              'selection': 'First existing canonical event; original window, fixed grids 32/64/128; no retuning.',
              'quality_gate': 'At each interior sample/grid, C1 mean improves over baseline and naive beyond numeric tolerance; sampled maximum does not regress against either.'}

    def save():
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    save()
    try:
        from compile_critical_beb1_event_ir import compile_ordinary_patch
        event = args.event_root.resolve()
        campaign_path = event.parent/'all_canonical_beb1_summary.json'
        ir_path = event/'critical_beb1_event_ir.json'
        validation_path = event/'whole_mesh_validation.json'
        inputs = [campaign_path, ir_path, validation_path, args.profile_source.resolve(), Path(__file__).resolve()]
        if args.times_json:
            inputs.append(args.times_json.resolve())
        report['inputs_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        campaign = json.loads(campaign_path.read_text())
        require(campaign['events'][0]['key'] == event.name, 'Not the preregistered first canonical event.')
        validation = json.loads(validation_path.read_text())
        require(validation['pass'] is True and all(validation['checks'].values()), 'Frozen C0 safety checks failed.')
        ir = json.loads(ir_path.read_text())
        require(ir['whole_mesh_splice_ready'] is True, 'Frozen C0 exact-root splice was not admitted.')
        levels = tuple(rational(ir['one_sided_window'][name]) for name in ('lower', 'critical', 'upper'))
        lower, root, upper = levels
        require(lower < root < upper, 'Invalid recorded window.')
        extra, time_mapping = additional_times(args.times_json, lower, upper)
        times = sorted(set([lower, (lower+root)/2, root, (root+upper)/2, upper, *extra]))
        report['times'] = [fraction_json(tau) for tau in times]
        report['natural_frame_mapping'] = time_mapping
        report['event_id'] = ir['event']['event_id']
        report['cache_root'] = str(args.cache_root.resolve())
        patches = {}
        root_patch, root_runtime = compile_ordinary_patch(args.cache_root.resolve(), root, event_id=ir['event']['event_id'])
        patches[root] = root_patch, root_runtime
        cycle = root_patch['boundary_cycle']
        required = frozenset(root_runtime['cycle'])
        require(len(cycle) == 4, 'Only quadrilateral pilot patches are supported.')
        report['fixed_boundary_cycle'] = cycle
        for tau in (lower, upper):
            patches[tau] = compile_ordinary_patch(args.cache_root.resolve(), tau,
                                                 required_boundary=required, event_id=ir['event']['event_id'])
        anchors = []
        contract = ir['event_star_geometry']['critical_position_contract']
        for name, tau in [('lower', lower), ('upper', upper)]:
            patch, _ = patches[tau]
            require(stable_cycle(cycle, patch['boundary_cycle']), 'Endpoint boundary identity/orientation changes.')
            diagonal = sorted(set(patch['source_faces'][0]) & set(patch['source_faces'][1]))
            require(len(diagonal) == 2, 'Endpoint ordinary patch lacks a shared diagonal.')
            audit = endpoint_contract(*(patch['boundary_positions'][key] for key in diagonal))
            audit['source_diagonal'] = diagonal
            report['endpoint_audits'][name] = audit
            save()
            require(audit['binary64_center_on_shared_diagonal_exact'],
                    name+': binary64 reference center leaves its own source diagonal; offline endpoint identity is not satisfied.')
            require(audit['binary32_center_on_shared_diagonal_exact'],
                    name+': quantized center leaves the quantized source diagonal; prospective binary32 endpoint identity is not satisfied.')
            anchors.append(np.asarray(audit['center_binary64']))
        # The root anchor deliberately reuses the admitted C0 binary32 point.
        anchors.insert(1, np.asarray(contract['canonical_position_float64'], dtype=float))
        root_vertices = ir['event_star_geometry']['critical_patch']['vertices']
        old_root_center = np.asarray(root_vertices[-1]['position'], dtype=float)
        require(np.array_equal(anchors[1], old_root_center), 'C0 root center contract differs from saved geometry.')
        old_boundary = {item['label']['id']: item['position'] for item in root_vertices[:-1]}
        require(all(key in old_boundary and np.array_equal(root_patch['boundary_positions'][key], old_boundary[key])
                    for key in cycle), 'Newly read source root boundary differs from frozen C0.')
        report['root_c0_geometry_identical'] = True
        report['root_anchor'] = 'Frozen C0 canonical binary32 center, promoted to binary64; not the unrounded ideal critical point.'
        spec = importlib.util.spec_from_file_location('c1_original_profile', args.profile_source)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        for tau in times:
            if tau not in patches:
                patches[tau] = compile_ordinary_patch(args.cache_root.resolve(), tau,
                                                     required_boundary=required, event_id=ir['event']['event_id'])
            patch, runtime = patches[tau]
            require(stable_cycle(cycle, patch['boundary_cycle']), 'Source cycle changes at an interior probe.')
            boundary = np.asarray([patch['boundary_positions'][key] for key in cycle], dtype=float)
            positions = patch['boundary_positions']
            baseline = np.asarray([[positions[key] for key in face] for face in patch['source_faces']])
            c1 = fan_triangles(boundary, center_at(tau, levels, anchors))
            naive = fan_triangles(boundary, center_at(tau, levels, anchors, critical=False))
            measurement = {'time': fraction_json(tau), 'source_owner_refs': [list(ref.values()) for ref in runtime['suppressions']],
                           'source_boundary_shared_by_construction': True, 'boundary_gap_to_same_source_input': 0.,
                           'source_faces': patch['source_faces'], 'c1_center': c1[0, 2].tolist(),
                           'naive_center': naive[0, 2].tolist(), 'grids': []}
            report['measurements'].append(measurement)
            save()
            for grid in (32, 64, 128):
                values = compare_heights(boundary, baseline, c1, naive, profile.terrain, grid)
                interior = lower < tau < upper
                if interior:
                    tol = values['tolerance']
                    accepted = all(values[opponent]['mean_absolute_height_error']-values['c1']['mean_absolute_height_error'] > tol
                                   and values['c1']['maximum_absolute_height_error']-values[opponent]['maximum_absolute_height_error'] <= tol
                                   for opponent in ('baseline', 'naive'))
                    values['quality_gate_pass'] = bool(accepted)
                else:
                    accepted = True
                    values['quality_gate_pass'] = None
                measurement['grids'].append(values)
                save()
                require(accepted, 'No preregistered C1 improvement over both baseline and naive at this interior probe/grid.')
        report['status'] = 'PASS_PREFLIGHT_ONLY'
        report['next_step'] = 'Review evidence; do not emit runtime plans before interval and exterior certificates.'
        save()
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        report['status'] = 'STOP_PREFLIGHT'
        report['stop_reason'] = str(error)
        save()
        print(json.dumps(report, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
