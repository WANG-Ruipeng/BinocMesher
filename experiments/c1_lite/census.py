#!/usr/bin/env python3
"""Read-only four-demo C1-lite screen, never a window admission certificate.

Inputs are old campaign/cache files. Only --output (which must not exist) is
written. No mesher runtime, cache update, renderer, or SSP1 writer is invoked.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'source_splice'
sys.path.insert(0, str(SOURCE))
from check_window_boundary import slice_triangle, segment_distance
from compile_critical_beb1_event_ir import compile_ordinary_patch
from processed_mesh import (HVID, SourceVID, _read_counted_vector,
    _parse_processed_metadata, parse_serialized_vid, parse_hypervertices,
    infer_cache_shape, selected_event_rows)
from sidewall_contract import audit_sidewall


def fj(value):
    value = Fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def fr(value):
    return Fraction(value['numerator'], value['denominator'])


def parse_vid(value):
    first, second = value.split('|')
    return SourceVID.canonical(HVID(*map(int, first.split(':'))),
                               HVID(*map(int, second.split(':'))))


def reconstruct_demo_time(profile, groups, maximum):
    """Mirror profile cameras, core.__init__, and load_parameters without loading C++."""
    if profile.get('profile') != 'demo' or profile.get('cameras') != 24:
        raise ValueError('This pilot only reconstructs the documented 24-camera demo.')
    camera_times = [(0.5+i)/24.0 for i in range(24)]
    origin = min(camera_times)  # core.py min_t_offset=0
    shifted = [t-origin for t in camera_times]
    duration = max(shifted)+1e-5  # core.py use_alignment=False
    fading = max(float(profile['fading_time']),
                 min(b-a for a, b in zip(shifted, shifted[1:]) if b != a))
    level = 0
    while fading*(1 << (level+1)) < duration:
        level += 1
    if groups != (1 << level) or maximum != (2 << level):
        raise ValueError('Reconstructed temporal scale disagrees with cache shape.')
    delta = duration/maximum
    return {'status': 'RECONSTRUCTED_DEMO_PROFILE_NOT_SERIALIZED_RUNTIME_PARAMETERS',
        'formula': 'global_seconds = origin_seconds + internal_exact_time * delta_seconds',
        'origin_seconds': fj(Fraction.from_float(origin)),
        'delta_seconds': fj(Fraction.from_float(delta)),
        'duration_seconds': fj(Fraction.from_float(duration)),
        'max_temporal_level': level, 'cache_group_count': groups,
        'maximum_discrete_time': maximum,
        'origin_seconds_float': origin, 'delta_seconds_float': delta,
        'assumptions': ['Existing run_lightweight_profile.make_cameras: (i+0.5)/24.',
            'core.py defaults min_t_offset=0 and use_alignment=False.',
            'Parameters reconstructed from saved profile and verified against cache shape; no C++ runtime initialized.',
            'Fractions represent actual binary64 parameter values; final C++ long-double arithmetic/rounding is not modelled.'],
        'source_references': ['experiments/tv0_tv4/run_lightweight_profile.py:make_cameras',
            'binocmesher/core.py:BinocMesher.__init__',
            'binocmesher/source/coarse_step.cpp:load_parameters',
            'binocmesher/source/slicing.cpp:run_slicing_exact']}


def sample_hits(lower_seconds, root_seconds, upper_seconds, fps, phase=Fraction(0)):
    """Count predeclared global grid t=(n+phase)/fps; endpoints explicitly separate."""
    lower_seconds, root_seconds, upper_seconds = map(Fraction,
        (lower_seconds, root_seconds, upper_seconds))
    lo = math.ceil(lower_seconds*fps-phase)
    hi = math.floor(upper_seconds*fps-phase)
    hits = []
    for index in range(max(0, lo), hi+1):
        t = (index+phase)/fps
        hits.append({'index_zero_based': index, 'seconds': fj(t),
            'location': ('lower_endpoint' if t == lower_seconds else
                         'upper_endpoint' if t == upper_seconds else
                         'exact_root' if t == root_seconds else
                         'left_open_slab' if t < root_seconds else 'right_open_slab')})
    return {'fps': fps, 'phase_in_frames': fj(phase),
        'formula': '(index + phase_in_frames) / fps',
        'closed_window_count': len(hits),
        'open_window_count': sum(lower_seconds < fr(h['seconds']) < upper_seconds for h in hits),
        'left_open_slab_count': sum(h['location'] == 'left_open_slab' for h in hits),
        'right_open_slab_count': sum(h['location'] == 'right_open_slab' for h in hits),
        'hits': hits}


def effective_interval(first, second):
    if first.time > second.time:
        first, second = second, first
    lower, upper = first.time, second.time
    if first.in_view != second.in_view and lower+first.halfspan <= upper-second.halfspan:
        if second.in_view:
            upper -= second.halfspan
        else:
            lower += first.halfspan
    return lower, upper


def relevant_records(cache, event_rows):
    """Read complete processed intervals for registry-linked raw owner records."""
    wanted = defaultdict(set)
    for row in event_rows:
        wanted[(int(row['t_group']), int(row['t_start']))].add(int(row['sorted_record_index']))
    result = []
    for (group, start), indices in sorted(wanted.items()):
        folder = cache / 'processed_hyperpolys'
        payload = (folder / f'{group}_{start}.bin').read_bytes()
        metadata = _parse_processed_metadata(
            (folder / f'{group}_{start}_hpmeta.bin').read_bytes(), group, start)
        offset = 0
        record_index = 0
        while offset < len(payload):
            element = struct.unpack_from('<b', payload, offset)[0]
            offset += 1
            blobs, offset = _read_counted_vector(payload, offset, 1)
            times = [struct.unpack_from('<b', blob, 0)[0] for blob in blobs]
            if len(times) < 2 or any(a >= b for a, b in zip(times, times[1:])):
                raise ValueError('Invalid processed interval list.')
            vids = set()
            interval_faces = []
            for interval in range(len(times)-1):
                count = 0
                while True:
                    faces, offset = _read_counted_vector(payload, offset, 16)
                    if not faces:
                        break
                    vids.update(parse_serialized_vid(blob) for blob in faces)
                    count += 1
                interval_faces.append(count)
            if record_index >= len(metadata):
                raise ValueError('Processed stream outlives metadata.')
            index = metadata[record_index]
            if index in indices:
                expanded = times.copy()
                expanded[0] -= 1
                expanded[-1] += 1
                result.append({'group': group, 'start': start, 'sorted_index': index,
                    'element': element, 'times': times, 'expanded_times': expanded,
                    'interval_face_counts': interval_faces, 'source_vids': vids})
            record_index += 1
        if offset != len(payload) or record_index != len(metadata):
            raise ValueError('Processed stream length mismatch.')
        if indices != {r['sorted_index'] for r in result if r['group'] == group and r['start'] == start}:
            raise ValueError('Missing registry-linked processed owner.')
    return result


def source_breakpoints(cache, event_id, levels, boundary_cycle, hypervertices, groups, maximum):
    _, rows = selected_event_rows(cache, event_id)
    records = relevant_records(cache, rows)
    candidates = defaultdict(set)
    for n in range(groups+1):
        candidates[Fraction(n*maximum, groups)].add('active_cache_group_boundary')
    vids = {parse_vid(text) for text in boundary_cycle}
    for record in records:
        owner = f"{record['group']}:{record['start']}:{record['sorted_index']}"
        candidates[Fraction(record['start']-1)].add('record_load_threshold:'+owner)
        for time in record['expanded_times']:
            candidates[Fraction(time)].add('processed_owner_interval:'+owner)
        vids.update(record['source_vids'])
    source_intervals = []
    for vid in sorted(vids):
        first, second = hypervertices[vid.first], hypervertices[vid.second]
        effective = effective_interval(first, second)
        for value in (first.time, second.time):
            candidates[Fraction(value)].add('raw_source_clamp_or_time:'+vid.text())
        for value in effective:
            candidates[Fraction(value)].add('effective_clamp_or_SourceVID_collapse:'+vid.text())
        if vid.text() in boundary_cycle:
            source_intervals.append({'source_vid': vid.text(),
                'raw_times': [first.time, second.time],
                'effective_interpolation_interval': list(effective),
                'endpoint_halfspans': [first.halfspan, second.halfspan],
                'endpoint_in_view': [first.in_view, second.in_view]})
    points = [{'time': fj(t), 'reasons': sorted(reasons)} for t, reasons in sorted(candidates.items())
              if levels[0] <= t <= levels[-1]]
    return {'interval_stability_certificate': 'UNKNOWN',
        'reason': 'Candidate branch boundaries are extracted analytically for registry-linked records and observed boundary VIDs. Full exterior incidence, selected-owner equivalence, and all-source ownership over the interval remain uncertified.',
        'registry_record_count': len(records), 'source_edge_count': len(vids),
        'boundary_source_intervals': source_intervals,
        'candidate_breakpoints_in_closed_window': points,
        'slabs': [{'slab': slab, 'internal_candidates': [p for p in points
                   if levels[slab] < fr(p['time']) < levels[slab+1]],
                   'stability': 'UNKNOWN'} for slab in (0, 1)]}


def legacy_gap_samples(ir, cache):
    cylinder = ir['event_star_geometry']['mapping_cylinder']
    levels = [fr(cylinder['levels'][key]) for key in ('lower', 'critical', 'upper')]
    times = [t for t in levels for _ in range(5)]
    vertices = np.asarray(cylinder['vertices4'], dtype=float)
    cycle = cylinder['boundary_cycle']
    required = frozenset(parse_vid(text) for text in cycle)
    results = []
    for slab in (0, 1):
        for alpha in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            tau = (1-alpha)*levels[slab]+alpha*levels[slab+1]
            patch, runtime = compile_ordinary_patch(cache, tau,
                required_boundary=required, event_id=ir['event']['event_id'])
            positions = {k: np.asarray(v, float) for k, v in patch['boundary_positions'].items()}
            distances = []
            for face in cylinder['side_trace_faces']:
                points = slice_triangle(vertices, face, times, tau)
                if not points:
                    continue
                edge = sorted({index % 5 for index in face})
                if len(edge) != 2 or max(edge) >= 4:
                    raise ValueError('Unsupported old side-wall face.')
                a, b = (positions[cycle[index]] for index in edge)
                # A directed diagnostic on old wall vertices; not Hausdorff.
                distances.extend(segment_distance(point, a, b) for point in points)
            if not distances:
                raise ValueError('Old side-wall sample has no slice vertices.')
            results.append({'slab': slab, 'alpha': fj(alpha), 'internal_time': fj(tau),
                'sampled_max_vertex_to_source_edge_gap': max(distances),
                'sampled_rms_vertex_to_source_edge_gap': math.sqrt(sum(x*x for x in distances)/len(distances)),
                'sampled_vertex_count': len(distances),
                'boundary_cycle': patch['boundary_cycle'],
                'source_faces': patch['source_faces'],
                'raw_owner_references': [list(r.values()) for r in runtime['suppressions']]})
    return {'status': 'FINITE_DIAGNOSTIC_NOT_A_CERTIFICATE',
        'sampling': 'Fixed alpha=1/4,1/2,3/4 in each slab; duplicate shared wall-vertices remain in RMS.',
        'distance': 'One-way sliced old-wall vertices to corresponding ordinary boundary edge; not Hausdorff and not the continuous-time maximum.',
        'maximum_over_samples': max(r['sampled_max_vertex_to_source_edge_gap'] for r in results),
        'samples': results}


def run(campaign, cache):
    groups, maximum = infer_cache_shape(cache)
    profile = json.loads((cache/'profile_result.json').read_text())
    mapping = reconstruct_demo_time(profile, groups, maximum)
    origin, delta = fr(mapping['origin_seconds']), fr(mapping['delta_seconds'])
    hypervertices = parse_hypervertices(cache)
    paths = sorted(campaign.glob('event-*/critical_beb1_event_ir.json'))
    if len(paths) != 4:
        raise ValueError('Expected exactly four canonical demo IRs.')
    events = []
    for path in paths:
        ir = json.loads(path.read_text())
        cylinder = ir['event_star_geometry']['mapping_cylinder']
        levels = [fr(cylinder['levels'][key]) for key in ('lower', 'critical', 'upper')]
        seconds = [origin+t*delta for t in levels]
        event = {'key': path.parent.name, 'event_id': ir['event']['event_id'],
            'source_ir_path': str(path), 'source_ir_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'levels_internal': dict(zip(('lower', 'root', 'upper'), map(fj, levels))),
            'levels_global_seconds': dict(zip(('lower', 'root', 'upper'), map(fj, seconds))),
            'window_seconds_float': float(seconds[2]-seconds[0]),
            'original_24fps_half_phase': sample_hits(*seconds, 24, phase=Fraction(1, 2)),
            'predeclared_zero_phase_grids': [sample_hits(*seconds, fps) for fps in (24, 48, 96, 192)],
            'exact_legacy_rotation_contract': audit_sidewall(cylinder['vertices4']),
            'source_interval_audit': source_breakpoints(cache, ir['event']['event_id'], levels,
                cylinder['boundary_cycle'], hypervertices, groups, maximum)}
        try:
            event['legacy_sidewall_samples'] = legacy_gap_samples(ir, cache)
        except Exception as error:
            event['legacy_sidewall_samples'] = {'status': 'STOP_SAMPLE_FAILURE',
                'reason': str(error)}
            events.append(event)
            return {'schema': 'c1-lite-demo-census-v1', 'status': 'STOP_SAMPLE_FAILURE',
                'time_mapping': mapping, 'events': events, 'unprocessed_events': len(paths)-len(events)}
        events.append(event)
    return {'schema': 'c1-lite-demo-census-v1', 'status': 'SCREEN_COMPLETE_NO_WINDOW_ADMISSION',
        'campaign_root': str(campaign), 'cache_root': str(cache), 'time_mapping': mapping,
        'source_interval_stability': 'UNKNOWN', 'continuous_embedding_certificate': 'UNKNOWN',
        'runtime_started': False, 'rendering_started': False,
        'events': events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', required=True, type=Path)
    parser.add_argument('--cache-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output, campaign, cache = args.output.resolve(), args.campaign_root.resolve(), args.cache_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    if campaign == output or campaign in output.parents or cache == output or cache in output.parents:
        raise ValueError('Output must not modify input campaign/cache trees.')
    report = run(campaign, cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'events': len(report['events'])}))
    return 0 if report['status'] == 'SCREEN_COMPLETE_NO_WINDOW_ADMISSION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
