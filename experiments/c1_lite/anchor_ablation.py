#!/usr/bin/env python3
"""Fixed four-demo, offline center-placement ablation; no runtime or rendering.

Quality losses are data, not rejection criteria. Unsupported geometry stops
that method's remaining probes, while other methods/events remain scheduled.
All integrals are finite trapezoidal estimates, not continuous certificates.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from reference import (GateStop, center_at, endpoint_contract, error_stats,
                       fan_triangles, fraction_json, heights_at_xy, rational,
                       require, require_convex_graph, stable_cycle)

METHODS = ('baseline', 'naive', 'centroid', 'beb1', 'c0_root_only')
GRIDS = (32, 64, 128)
MAX_REPORT_BYTES = 16 * 1024 * 1024


def fixed_times(levels):
    lower, root, upper = levels
    require(lower < root < upper, 'Invalid original event window.')
    return ([lower+(root-lower)*Fraction(i, 16) for i in range(17)] +
            [root+(upper-root)*Fraction(i, 16) for i in range(1, 17)])


def normalized_integral(times, values):
    """Trapezoidal integral divided by duration; exact rational time weights."""
    require(len(times) == len(values) and len(times) >= 2, 'Incomplete quadrature series.')
    require(all(b > a for a, b in zip(times, times[1:])), 'Times must strictly increase.')
    array = np.asarray(values, dtype=float)
    require(np.all(np.isfinite(array)), 'Nonfinite quadrature values.')
    duration = times[-1]-times[0]
    return float(math.fsum(float((b-a)/(2*duration))*(float(x)+float(y))
                     for a, b, x, y in zip(times, times[1:], array, array[1:])))


def candidate_centers(tau, levels, endpoint_anchors, root_boundary, beb1_root):
    left, right = endpoint_anchors
    event_anchors = (left, np.asarray(beb1_root, dtype=float), right)
    centroid = np.mean(np.asarray(root_boundary, dtype=float), axis=0)
    return {
        'naive': center_at(tau, levels, event_anchors, critical=False),
        'centroid': center_at(tau, levels, (left, centroid, right)),
        'beb1': center_at(tau, levels, event_anchors),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def baseline_grid(boundary, baseline, terrain, grid):
    """One shared independent-reference query batch, never method-specific."""
    require_convex_graph(boundary, baseline)
    low, high = np.min(boundary, axis=0), np.max(boundary, axis=0)
    tolerance = 1e-9*max(1., float(np.linalg.norm(high-low)))
    axis = (np.arange(grid)+.5)/grid
    gx, gy = np.meshgrid(low[0]+axis*(high[0]-low[0]), low[1]+axis*(high[1]-low[1]))
    xy = np.column_stack((gx.ravel(), gy.ravel()))
    heights = heights_at_xy(baseline, xy, tolerance)
    mask = np.isfinite(heights)
    require(int(mask.sum()) >= 100, 'Too few baseline reference samples.')
    reference = -np.asarray(terrain(np.column_stack((xy[mask], np.zeros(int(mask.sum()))))))
    require(reference.shape == (int(mask.sum()),) and np.all(np.isfinite(reference)),
            'Independent reference returned invalid heights.')
    record = {'grid': grid, 'samples': int(mask.sum()), 'xy_bbox': [low[:2].tolist(), high[:2].tolist()],
              'numeric_tolerance': tolerance, **error_stats(np.abs(heights[mask]-reference))}
    return record, (xy, mask, reference, tolerance)


def evaluate_candidate(boundary, triangles, shared, grid):
    require_convex_graph(boundary, triangles)
    xy, mask, reference, tolerance = shared
    heights = heights_at_xy(triangles, xy, tolerance)
    require(np.array_equal(mask, np.isfinite(heights)),
            'Candidate coverage differs; intersection-only comparison is forbidden.')
    return {'grid': grid, 'samples': int(mask.sum()), 'same_sampled_coverage': True,
            **error_stats(np.abs(heights[mask]-reference))}


def pairwise_loss(candidate, baseline):
    """Positive values mean the candidate is worse; never filters records."""
    result = {}
    for key in ('mean_absolute_height_error', 'p95_absolute_height_error', 'maximum_absolute_height_error'):
        old, new = float(baseline[key]), float(candidate[key])
        result[key+'_difference'] = new-old
        result[key+'_relative_change'] = (new/old-1) if old else None
    return result


def geometry_record(boundary, triangles):
    require_convex_graph(boundary, triangles)
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                    triangles[:, 2]-triangles[:, 0]), axis=1)/2
    require(np.all(np.isfinite(areas)) and np.min(areas) > 0, 'Nonfinite or degenerate triangle.')
    return {'sampled_graph_valid': True, 'minimum_triangle_area': float(np.min(areas)),
            'triangles': int(len(triangles)), 'shared_source_boundary_input': True,
            'continuous_embedding_certified': False, 'exterior_intersections_checked': False}


def endpoint_audits(patches, levels, cycle):
    result = {}
    for name, tau in (('lower', levels[0]), ('upper', levels[2])):
        patch, _ = patches[tau]
        require(stable_cycle(cycle, patch['boundary_cycle']), 'Endpoint source cycle differs from root.')
        require(len(patch['source_faces']) == 2, 'Expected two ordinary endpoint source faces.')
        diagonal = sorted(set(patch['source_faces'][0]) & set(patch['source_faces'][1]))
        require(len(diagonal) == 2, 'Endpoint source patch lacks exactly one shared diagonal.')
        result[name] = endpoint_contract(*(patch['boundary_positions'][key] for key in diagonal))
        result[name]['source_diagonal'] = diagonal
    return result


def summarize_method(records, times, levels, duration_seconds, method, baseline_summary=None):
    expected = {tau for tau in times}
    by_time = {rational(row['time']): row for row in records}
    if set(by_time) != expected or any(len(row['grids']) != len(GRIDS) for row in records):
        return {'status': 'INCOMPLETE_NO_WINDOW_INTEGRAL', 'completed_times': len(records)}
    output = {'status': 'COMPLETE_FINITE_QUADRATURE', 'completed_times': len(records), 'grids': []}
    for grid in GRIDS:
        series = [next(item for item in by_time[tau]['grids'] if item['grid'] == grid) for tau in times]
        mean_values = [row['mean_absolute_height_error'] for row in series]
        max_values = [row['maximum_absolute_height_error'] for row in series]
        p95_values = [row['p95_absolute_height_error'] for row in series]
        ordinary_integral = normalized_integral(times, mean_values)
        integral = ordinary_integral
        if method == 'c0_root_only':
            require(baseline_summary is not None, 'C0 measure-zero summary needs baseline.')
            integral = next(row for row in baseline_summary['grids'] if row['grid'] == grid)['normalized_mean_height_error_integral']
        output['grids'].append({
            'grid': grid, 'normalized_mean_height_error_integral': integral,
            'mean_height_error_integral_seconds': integral*duration_seconds,
            'maximum_sampled_height_error_over_window': float(max(max_values)),
            'maximum_sampled_mean_height_error_over_window': float(max(mean_values)),
            'maximum_sampled_p95_height_error_over_window': float(max(p95_values)),
            'minimum_sampled_mean_height_error_over_window': float(min(mean_values)),
            'root_mean_height_error': series[times.index(levels[1])]['mean_absolute_height_error'],
            'quadrature_note': ('C0 differs only at a measure-zero root: its time integral equals baseline; '
                                'no artificial trapezoidal root-spike gain is credited.' if method == 'c0_root_only'
                                else 'Trapezoidal estimate with actual elapsed-time weights, not a continuous maximum/integral certificate.'),
            'diagnostic_naive_trapezoidal_mean': ordinary_integral if method == 'c0_root_only' else None,
        })
    return output


def run_event(event_path, cache, terrain, mapping, compiler, progress=None):
    started = time.monotonic()
    ir_path = event_path/'critical_beb1_event_ir.json'
    event = {'key': event_path.name, 'status': 'RUNNING', 'runtime_admitted': False,
             'continuous_window_certified': False, 'measurement_coordinate_model': 'parser binary64 reference only',
             'methods': {name: {'status': 'SCHEDULED', 'measurements': []} for name in METHODS},
             'query_cost': {'ordinary_source_parse_calls_shared': 0, 'independent_reference_points_shared': 0,
                            'independent_reference_batches_shared': 0, 'construction_occupancy_queries_per_method': {name: 0 for name in METHODS},
                            'historical_BEB1_anchor_construction_cost': 'Not measured; center is loaded from existing frozen C0 IR.',
                            'scope': 'Same incremental 1-center/4-face budget for three fan controls; shared reference queries are evaluation cost, not method construction.'}}

    def publish():
        event['elapsed_seconds'] = time.monotonic()-started
        if progress is not None:
            progress(event)

    try:
        ir = json.loads(ir_path.read_text(encoding='utf-8'))
        validation_path = event_path/'whole_mesh_validation.json'
        validation = json.loads(validation_path.read_text(encoding='utf-8'))
        event['inputs_sha256'] = {str(path): sha256(path) for path in (ir_path, validation_path)}
        require(ir['whole_mesh_splice_ready'] is True and validation['pass'] is True and all(validation['checks'].values()),
                'Existing frozen C0 event safety/IR is not admitted.')
        levels = tuple(rational(ir['one_sided_window'][key]) for key in ('lower', 'critical', 'upper'))
        times = fixed_times(levels)
        event['levels_internal'] = [fraction_json(tau) for tau in levels]
        event['fixed_times_internal'] = [fraction_json(tau) for tau in times]
        delta, origin = rational(mapping['delta_seconds']), rational(mapping['origin_seconds'])
        require(delta > 0, 'Nonpositive reconstructed time scale.')
        event['window_seconds'] = float((levels[2]-levels[0])*delta)
        event['event_id'] = ir['event']['event_id']
        patches = {}

        def get_patch(tau, required=None):
            if tau not in patches:
                patches[tau] = compiler(cache, tau, required_boundary=required, event_id=ir['event']['event_id'])
                event['query_cost']['ordinary_source_parse_calls_shared'] += 1
            return patches[tau]

        root_patch, root_runtime = get_patch(levels[1])
        cycle = root_patch['boundary_cycle']
        require(len(cycle) == 4 and len(set(cycle)) == 4, 'Only four-source-vertex pilot patches are supported.')
        required = frozenset(root_runtime['cycle'])
        for tau in (levels[0], levels[2]):
            get_patch(tau, required)
        event['boundary_cycle'] = cycle
        event['endpoint_audits'] = endpoint_audits(patches, levels, cycle)
        ideal_endpoints = all(audit['binary64_center_on_shared_diagonal_exact'] for audit in event['endpoint_audits'].values())
        runtime_endpoints = all(audit['binary32_center_on_shared_diagonal_exact'] for audit in event['endpoint_audits'].values())
        event['binary64_endpoint_contract_pass'] = ideal_endpoints
        event['prospective_binary32_endpoint_contract_pass'] = runtime_endpoints
        event['runtime_disposition'] = ('NOT_ADMITTED_INTERVAL_AND_RUNTIME_UNTESTED' if runtime_endpoints
                                        else 'REJECTED_PROSPECTIVE_BINARY32_ENDPOINT_CONTRACT')
        publish()
        require(ideal_endpoints, 'Binary64 midpoint leaves its source diagonal; no endpoint-contract relaxation allowed.')
        endpoints = tuple(np.asarray(event['endpoint_audits'][name]['center_binary64']) for name in ('lower', 'upper'))
        root_boundary = np.asarray([root_patch['boundary_positions'][key] for key in cycle])
        beb1_root = np.asarray(ir['event_star_geometry']['critical_position_contract']['canonical_position_float64'])
        c0_vertices = ir['event_star_geometry']['critical_patch']['vertices']
        old_boundary = {row['label']['id']: row['position'] for row in c0_vertices[:-1]}
        require(np.array_equal(beb1_root, c0_vertices[-1]['position']) and all(
            key in old_boundary and np.array_equal(root_patch['boundary_positions'][key], old_boundary[key]) for key in cycle),
            'Re-read root source or BEB1 center differs from frozen C0 local geometry.')
        event['root_local_input_geometry_matches_C0'] = True
        event['center_anchors'] = {'lower': endpoints[0].tolist(), 'upper': endpoints[1].tolist(),
                                    'centroid_root': root_boundary.mean(axis=0).tolist(), 'beb1_root': beb1_root.tolist()}
        event['root_agreement_scope'] = 'BEB1 local input geometry only; centroid deliberately need not reproduce C0; no whole-mesh arrays executed.'
        active = set(METHODS)
        for name in active:
            event['methods'][name]['status'] = 'RUNNING'
        for tau in times:
            patch, runtime = get_patch(tau, required)
            require(stable_cycle(cycle, patch['boundary_cycle']), 'Source boundary cycle changed; event reference unsupported.')
            boundary = np.asarray([patch['boundary_positions'][key] for key in cycle])
            baseline = np.asarray([[patch['boundary_positions'][key] for key in face] for face in patch['source_faces']])
            baseline_validity = geometry_record(boundary, baseline)
            centers = candidate_centers(tau, levels, endpoints, root_boundary, beb1_root)
            triangles = {'baseline': baseline, **{name: fan_triangles(boundary, center) for name, center in centers.items()}}
            triangles['c0_root_only'] = triangles['beb1'] if tau == levels[1] else baseline
            rows = {}
            source_refs = [list(ref.values()) for ref in runtime['suppressions']]
            for name in METHODS:
                if name not in active:
                    continue
                try:
                    validity = baseline_validity if name == 'baseline' else geometry_record(boundary, triangles[name])
                    row = {'time': fraction_json(tau), 'seconds': float(origin+tau*delta),
                           'validity': validity, 'source_owner_count': len(source_refs),
                           'source_owner_sha256': source_hash(source_refs), 'source_faces_sha256': source_hash(patch['source_faces']),
                           'center': centers[name].tolist() if name in centers else None, 'grids': []}
                    event['methods'][name]['measurements'].append(row)
                    rows[name] = row
                except GateStop as error:
                    active.remove(name)
                    event['methods'][name].update(status='UNSUPPORTED_GEOMETRY', stop_time=fraction_json(tau), stop_reason=str(error))
            for grid in GRIDS:
                ordinary, shared = baseline_grid(boundary, baseline, terrain, grid)
                event['query_cost']['independent_reference_points_shared'] += ordinary['samples']
                event['query_cost']['independent_reference_batches_shared'] += 1
                for name in METHODS:
                    if name not in active:
                        continue
                    try:
                        values = dict(ordinary) if name == 'baseline' else evaluate_candidate(boundary, triangles[name], shared, grid)
                        values['difference_from_baseline'] = pairwise_loss(values, ordinary)
                        rows[name]['grids'].append(values)
                    except GateStop as error:
                        active.remove(name)
                        event['methods'][name].update(status='UNSUPPORTED_GEOMETRY', stop_time=fraction_json(tau),
                                                     stop_grid=grid, stop_reason=str(error))
            publish()
        for name in METHODS:
            data = event['methods'][name]
            if name in active:
                data['status'] = 'COMPLETE_REFERENCE_ONLY'
            data['summary'] = summarize_method(data['measurements'], times, levels, event['window_seconds'], name,
                event['methods']['baseline'].get('summary'))
        event['comparison_summaries'] = []
        for opponent in ('baseline', 'naive', 'centroid'):
            ours = event['methods']['beb1']['summary']
            other = event['methods'][opponent]['summary']
            if ours['status'] != 'COMPLETE_FINITE_QUADRATURE' or other['status'] != 'COMPLETE_FINITE_QUADRATURE':
                event['comparison_summaries'].append({'opponent': opponent, 'status': 'UNAVAILABLE_INCOMPLETE_CURVE'})
                continue
            for current, previous in zip(ours['grids'], other['grids']):
                before, after = previous['normalized_mean_height_error_integral'], current['normalized_mean_height_error_integral']
                ours_rows = event['methods']['beb1']['measurements']
                other_rows = event['methods'][opponent]['measurements']
                ours_values = [next(g for g in row['grids'] if g['grid'] == current['grid']) for row in ours_rows]
                other_values = [next(g for g in row['grids'] if g['grid'] == current['grid']) for row in other_rows]
                mean_regret = [a['mean_absolute_height_error']-b['mean_absolute_height_error'] for a, b in zip(ours_values, other_values)]
                max_regret = [a['maximum_absolute_height_error']-b['maximum_absolute_height_error'] for a, b in zip(ours_values, other_values)]
                event['comparison_summaries'].append({'opponent': opponent, 'grid': current['grid'],
                    'normalized_integral_difference': after-before,
                    'normalized_integral_relative_change': (after/before-1) if before else None,
                    'worst_sampled_error_difference': current['maximum_sampled_height_error_over_window']-previous['maximum_sampled_height_error_over_window'],
                    'maximum_sampled_mean_error_regret': max(mean_regret),
                    'minimum_sampled_mean_error_regret': min(mean_regret),
                    'maximum_sampled_max_error_regret': max(max_regret),
                    'worst_mean_regret_time': ours_rows[int(np.argmax(mean_regret))]['time'],
                    'sign_convention': 'Positive is worse for BEB1; every sign is retained.'})
        event['status'] = 'COMPLETE_REFERENCE_ONLY' if len(active) == len(METHODS) else 'COMPLETE_WITH_UNSUPPORTED_METHODS'
    except Exception as error:
        event['status'] = 'UNSUPPORTED_EVENT_REFERENCE' if isinstance(error, GateStop) else 'ERROR_EVENT_REFERENCE'
        event['stop_reason'] = str(error)
        for data in event['methods'].values():
            if data['status'] in ('RUNNING', 'SCHEDULED'):
                data['status'] = 'NOT_COMPLETED_EVENT_STOP'
    publish()
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign-root', 'cache-root', 'profile-source', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    campaign, cache, output = args.campaign_root.resolve(), args.cache_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(output == root or root in output.parents for root in (campaign, cache)):
        raise ValueError('Output cannot be inside the immutable campaign/cache.')
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {'schema': 'binoc-c1-lite-anchor-ablation-v1', 'status': 'RUNNING', 'events': [],
              'fixed_design': {'events': 'All four original canonical demos, no outcome selection.',
                               'times_per_event': 33, 'points_per_half_including_endpoints': 17, 'grids': list(GRIDS),
                               'controls': {'baseline': 'ordinary two-triangle source patch',
                                            'naive': 'four-fan center interpolates directly between original endpoint diagonal midpoints',
                                            'centroid': 'same endpoints, root anchor is arithmetic mean of four root boundary positions',
                                            'beb1': 'same endpoints, root anchor is frozen C0 canonical binary32 BEB1 center',
                                            'c0_root_only': 'ordinary baseline except at the exact root'},
                               'acceptance': 'No quality-win filtering. Geometry/coordinate contract failures are retained as unsupported.'},
              'runtime_started': False, 'runtime_admitted': False, 'render_started': False,
              'storage': 'One compact JSON report; no meshes, per-pixel arrays or renders.',
              'limitations': ['Existing demo cache, not four independent scenes.',
                             'Finite projected-graph probes; no full-window owner/nondegeneracy/exterior certificate.',
                             'Vertical height error, not Hausdorff, topology repair or warped SSIM.',
                             'Prospective binary32 endpoint audit is not an actual C++ runtime execution.',
                             'Common evaluation queries are separated from incremental method construction cost.']}

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        text = json.dumps(report, indent=2, allow_nan=False)+'\n'
        if len(text.encode('utf-8')) > MAX_REPORT_BYTES:
            raise RuntimeError('Compact report exceeded the fixed 16 MiB storage budget.')
        (output/'result.json').write_text(text, encoding='utf-8')

    save()
    try:
        from census import infer_cache_shape, reconstruct_demo_time
        from compile_critical_beb1_event_ir import compile_ordinary_patch
        campaign_file = campaign/'all_canonical_beb1_summary.json'
        summary = json.loads(campaign_file.read_text(encoding='utf-8'))
        keys = [row['key'] for row in summary['events']]
        require(len(keys) == 4 and len(set(keys)) == 4, 'Expected exactly the four original canonical event keys.')
        require(all(Path(key).name == key and key.startswith('event-') for key in keys), 'Invalid canonical event path.')
        profile_file = cache/'profile_result.json'
        groups, maximum = infer_cache_shape(cache)
        mapping = reconstruct_demo_time(json.loads(profile_file.read_text()), groups, maximum)
        report['time_mapping'] = mapping
        report['inputs_sha256'] = {str(path.resolve()): sha256(path) for path in
                                  (campaign_file, profile_file, args.profile_source, Path(__file__), HERE/'reference.py')}
        report['campaign_root'], report['cache_root'] = str(campaign), str(cache)
        spec = importlib.util.spec_from_file_location('anchor_ablation_original_profile', args.profile_source)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        for key in keys:
            position = len(report['events'])
            report['events'].append({'key': key, 'status': 'SCHEDULED'})

            def progress(event, index=position):
                report['events'][index] = event
                save()

            result = run_event(campaign/key, cache, profile.terrain, mapping, compile_ordinary_patch, progress)
            report['events'][position] = result
            save()
            print(json.dumps({'event': key, 'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        statuses = [event['status'] for event in report['events']]
        report['status'] = 'COMPLETE_REFERENCE_STUDY' if all(status == 'COMPLETE_REFERENCE_ONLY' for status in statuses) else 'COMPLETE_WITH_UNSUPPORTED_OR_ERRORS'
        report['counts'] = {'scheduled_events': 4, 'completed_events': statuses.count('COMPLETE_REFERENCE_ONLY'),
                            'events_with_unsupported_methods': statuses.count('COMPLETE_WITH_UNSUPPORTED_METHODS'),
                            'unsupported_events': statuses.count('UNSUPPORTED_EVENT_REFERENCE'),
                            'error_events': statuses.count('ERROR_EVENT_REFERENCE')}
        save()
        print(json.dumps({'status': report['status'], 'counts': report['counts'], 'output': str(output)}))
        return 2 if report['counts']['error_events'] else 0
    except Exception as error:
        report['status'] = 'STOP_INPUT_OR_EXECUTION_ERROR'
        report['stop_reason'] = str(error)
        save()
        print(json.dumps({'status': report['status'], 'stop_reason': str(error)}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
