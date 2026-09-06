#!/usr/bin/env python3
"""Compact, read-only four-demo A audit: source + graph fan + exterior.

PASS denotes these conditional ideal-rational source/geometry certificates,
never event isolation, numerical compiler acceptance, runtime admission, or
binary32 quality. Every source singleton is audited in addition to every open
branch. Retained processed triangles are a conservative pre-filter superset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from window_source import (audit_event, read_inventory, iter_ordinary_triangles,
                           file_signatures, input_files, fj, fr)
from window_geometry import certify_segment
from window_exterior import certify_exterior_separation

SCHEMA = 'c1-lite-four-demo-window-audit-v1'
EVENT_KEYS = ('event-00-6d2d6dc2cdd7', 'event-01-e40703e638f3',
              'event-02-88ade47aa4fa', 'event-03-0c61aa0982ef')
OUTPUT_BUDGET = 10*1024*1024


def stable_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest_value(value):
    return hashlib.sha256(stable_bytes(value)).hexdigest()


def vectors(values):
    return tuple(tuple(fr(value) if isinstance(value, dict) else Fraction(value)
                       for value in point) for point in values)


def center_at(source, tau):
    levels = source['levels']
    lower, root, upper = (fr(levels[name]) for name in ('lower', 'root', 'upper'))
    if not lower <= tau <= upper or not lower < root < upper:
        raise ValueError('Center query outside the ordered source window.')
    if tau <= root:
        left, right, t0, t1 = 'lower', 'root', lower, root
    else:
        left, right, t0, t1 = 'root', 'upper', root, upper
    a = vectors([source['anchors'][left]['position']])[0]
    b = vectors([source['anchors'][right]['position']])[0]
    weight = (tau-t0)/(t1-t0)
    return [fj((1-weight)*x+weight*y) for x, y in zip(a, b)]


def qualified_vid(element, source_vid):
    if isinstance(element, bool) or not isinstance(element, int):
        raise ValueError('Element identity must be an integer.')
    if not isinstance(source_vid, str) or not source_vid:
        raise ValueError('SourceVID must be a nonempty string.')
    return f'element={element};{source_vid}'


def aggregate_status(statuses):
    values = list(statuses)
    if 'REJECT' in values:
        return 'REJECT'
    if not values or any(value != 'PASS' for value in values):
        return 'UNKNOWN'
    return 'PASS'


def check_junctions(source):
    """Check actual singleton coordinates and ordered boundary on both sides."""
    cycle = source['boundary_cycle']
    points = {fr(point['time']): point for point in source['breakpoint_points']}
    if len(points) != len(source['breakpoint_points']):
        raise ValueError('Duplicated source breakpoint singleton.')
    lower, root, upper = (fr(source['levels'][name]) for name in ('lower', 'root', 'upper'))
    if set(points) != {fr(segment[key]) for segment in source['segments'] for key in ('t0', 't1')}:
        raise ValueError('Singletons do not exactly cover all branch endpoints.')
    previous = lower
    for segment in source['segments']:
        t0, t1 = fr(segment['t0']), fr(segment['t1'])
        if t0 != previous or not t0 < t1 or (t0 < root < t1):
            raise ValueError('Source branches have a gap, overlap, or unpartitioned root.')
        previous = t1
        if segment['boundary_cycle'] != cycle or [row['source_vid'] for row in segment['boundary']] != cycle:
            raise ValueError('An affine branch changes the ordered boundary cycle.')
        for key, tau in (('position_t0', t0), ('position_t1', t1)):
            point = points[tau]
            if point['boundary_cycle'] != cycle or [row['source_vid'] for row in point['boundary']] != cycle:
                raise ValueError('A singleton changes the ordered boundary cycle.')
            actual = vectors([row['position'] for row in point['boundary']])
            endpoint = vectors([row[key] for row in segment['boundary']])
            if actual != endpoint:
                raise ValueError('Exact source boundary differs between singleton and branch limit.')
            # center_at is evaluated from the same exact global anchors here;
            # the local geometry certificate receives those exact values too.
            center_at(source, tau)
    if previous != upper or root not in points:
        raise ValueError('Branch union does not cover the recorded whole window.')
    if source.get('junction_boundary_cycle_and_orientation_equal') is not True:
        raise ValueError('Source orientation certificate is absent.')
    return {'status': 'PASS', 'exact_boundary_positions_match_all_junctions': True,
            'ordered_boundary_cycle_matches_all_junctions': True,
            'center_rule': 'One piecewise-affine interpolation through the exact source endpoint midpoints and frozen binary32 root anchor.',
            'source_face_orientation_basis': 'Frozen source certificate, not a new C++ runtime test.'}


def units_for(source):
    for index, segment in enumerate(source['segments']):
        yield {'kind': 'affine_branch', 'index': index,
               't0': segment['t0'], 't1': segment['t1'], 'owners': segment['owners'],
               'boundary_start': [row['position_t0'] for row in segment['boundary']],
               'boundary_end': [row['position_t1'] for row in segment['boundary']]}
    for index, point in enumerate(source['breakpoint_points']):
        boundary = [row['position'] for row in point['boundary']]
        yield {'kind': 'actual_singleton', 'index': index,
               't0': point['time'], 't1': point['time'], 'owners': point['owners'],
               'boundary_start': boundary, 'boundary_end': boundary}


def compact_certificate(certificate):
    return {key: certificate[key] for key in
            ('schema', 'status', 'scope', 'reason', 'orientation_xy', 'witness',
             'separation_certificate', 'minimum_3d_fan_area_lower_bound')
            if key in certificate}


def certify_degenerate_superset_axis(boundary_start, boundary_end, center_start,
                                    center_end, boundary_ids, triangle, triangle_ids):
    """Prove strict separation of convex hulls without dropping point/line faces."""
    report = {'schema': 'c1-lite-degenerate-superset-axis-v1', 'status': 'UNKNOWN',
              'classification': 'SUPERSET_DEGENERATE_UNRESOLVED',
              'reason': 'Repeated SourceVID superset geometry has no strict axis certificate; this is not a new collision.'}
    try:
        p0 = vectors([*boundary_start, center_start])
        p1 = vectors([*boundary_end, center_end])
        q0, q1 = vectors(triangle['positions_t0']), vectors(triangle['positions_t1'])
        if len(p0) != 5 or len(p1) != 5 or len(q0) != 3 or len(q1) != 3:
            raise ValueError('Malformed fan or retained superset coordinates.')
        seen = {}
        for index, source_id in enumerate(triangle_ids):
            actual = (q0[index], q1[index])
            if source_id in seen and seen[source_id] != actual:
                raise ValueError('Repeated retained SourceVID has contradictory exact trajectories.')
            seen[source_id] = actual
            if source_id in boundary_ids:
                j = boundary_ids.index(source_id)
                if actual != (p0[j], p1[j]):
                    raise ValueError('Shared SourceVID has contradictory exact trajectories.')
        for axis in range(3):
            for sign in (1, -1):
                differences = [sign*(patch[i][axis]-retained[j][axis])
                               for patch, retained in ((p0, q0), (p1, q1))
                               for i in range(5) for j in range(3)]
                if any(max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 4096
                       for value in differences):
                    report['reason'] = 'Exact axis arithmetic exceeds the fixed 4096-bit budget.'
                    return report
                maximum = max(differences)
                if maximum < 0:
                    report.update({'status': 'PASS',
                        'classification': 'SUPERSET_DEGENERATE_STRICT_AXIS',
                        'reason': 'Every signed fan/retained vertex-pair difference is strictly negative at both endpoints; affine interpolation certifies disjoint convex hulls, including degenerate geometry.',
                        'separation_certificate': {'kind': 'SUPERSET_DEGENERATE_STRICT_AXIS',
                            'axis': 'xyz'[axis], 'sign_of_patch_minus_retained': sign,
                            'exact_minimum_gap_lower_bound': fj(-maximum),
                            'endpoint_pair_count': len(differences),
                            'includes_all_four_boundary_vertices_and_center': True,
                            'retained_geometry_discarded': False}})
                    return report
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        report.update({'status': 'REJECT', 'reason': str(error),
                       'witness': {'kind': 'invalid_or_contradictory_source_identity', 'diagnostic': str(error)}})
    return report


def audit_source_report(source, triangle_provider, *, deadline=None, max_diagnostics=8,
                        progress=None):
    """Pure integration API; provider(t0,t1,excluded_owners) streams triangles."""
    report = {'schema': SCHEMA, 'event_id': source.get('event_id'), 'status': 'UNKNOWN',
        'coordinate_model': source.get('coordinate_model'),
        'source_report_sha256': digest_value(source), 'units': [], 'diagnostics': [],
        'diagnostic_limit': max_diagnostics, 'diagnostics_total': 0,
        'event_isolation_certificate': 'UNKNOWN', 'numerical_selector_conditioning': 'UNKNOWN',
        'runtime_admission': False, 'continuous_window_admitted': False,
        'binary32_runtime_certificate': 'UNKNOWN',
        'exterior_scope': 'Every retained canonical processed-source triangle in the bounded input cache, including a pre-filter superset; no runtime-filter equivalence assumed.',
        'not_claimed': ['No new whole-mesh runtime output, OMP execution, or C0 mesh-array equality.',
                        'No event isolation or arbitrary same-root window composition.',
                        'No natural-frame, visibility, rendering, SSIM, or binary32 quality result.',
                        'Graph-class REJECT is not a general 3D impossibility; separator UNKNOWN is not a collision.']}
    overall_digest = hashlib.sha256()

    def diagnostic(unit, kind, detail):
        report['diagnostics_total'] += 1
        if len(report['diagnostics']) < max_diagnostics:
            report['diagnostics'].append({'unit': unit, 'kind': kind, 'detail': detail})

    if source.get('status') != 'PASS_SOURCE_BRANCH_CONTRACT':
        report['reason'] = 'Source certificate did not pass; no geometry/exterior admission attempted.'
        report['source_status'] = source.get('status', 'UNKNOWN')
        return report
    report['source_status'] = 'PASS'
    try:
        if source.get('coordinate_model') != 'ideal rational interpolation of serialized HV binary32':
            raise ValueError('Wrong source coordinate model for this A audit.')
        if source.get('piecewise_source_ownership_certificate') is not True:
            raise ValueError('Missing complete source ownership branch certificate.')
        if source.get('half_window_owner_sets_constant') != {'left': True, 'right': True}:
            raise ValueError('Half-window owner constancy was not established.')
        report['junctions'] = check_junctions(source)
        cycle = source['boundary_cycle']
        # Frozen source owners carry the actual element; insist it is unique
        # throughout this event instead of silently taking a first triangle.
        elements = {owner[0] for unit in units_for(source) for owner in unit['owners']}
        if len(elements) != 1:
            raise ValueError('Event support does not have one consistent element identity.')
        element = next(iter(elements))
        boundary_ids = [qualified_vid(element, value) for value in cycle]
        for unit in units_for(source):
            key = f"{unit['kind']}:{unit['index']}"
            t0, t1 = fr(unit['t0']), fr(unit['t1'])
            item = {k: unit[k] for k in ('kind', 'index', 't0', 't1')}
            item.update({'source_owners_sha256': digest_value(unit['owners']),
                         'retained_counts': {'PASS': 0, 'UNKNOWN': 0, 'REJECT': 0},
                         'retained_certificate_kinds': {},
                         'retained_scan_complete': False})
            report['units'].append(item)
            if deadline is not None and time.monotonic() >= deadline:
                item.update({'status': 'UNKNOWN', 'reason': 'Wall-time budget exhausted before this unit.',
                             'geometry_status': 'UNKNOWN', 'exterior_status': 'UNKNOWN'})
                continue
            center0, center1 = center_at(source, t0), center_at(source, t1)
            geometry = certify_segment(unit['boundary_start'], unit['boundary_end'], center0, center1)
            item['geometry_status'] = geometry['status']
            item['geometry_certificate_sha256'] = digest_value(geometry)
            item['geometry_summary'] = compact_certificate(geometry)
            if geometry['status'] != 'PASS':
                item.update({'exterior_status': 'UNKNOWN', 'status': geometry['status'],
                             'reason': 'Exterior precondition not certified; no collision inference.'})
                diagnostic(key, 'local_geometry_not_admitted', compact_certificate(geometry))
                continue
            stream_hash = hashlib.sha256()
            processed = 0
            try:
                for triangle in triangle_provider(t0, t1, unit['owners']):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError('Wall-time budget exhausted during retained scan.')
                    triangle_ids = [qualified_vid(triangle['element'], value) for value in triangle['source_vertices']]
                    if len(triangle_ids) != 3:
                        certificate = {'status': 'UNKNOWN', 'reason': 'Malformed processed superset triangle arity.',
                                       'classification': 'SUPERSET_INPUT_UNRESOLVED'}
                    elif len(set(triangle_ids)) < 3:
                        certificate = certify_degenerate_superset_axis(
                            unit['boundary_start'], unit['boundary_end'], center0, center1,
                            boundary_ids, triangle, triangle_ids)
                    else:
                        certificate = certify_exterior_separation(
                            unit['boundary_start'], unit['boundary_end'], boundary_ids,
                            triangle['positions_t0'], triangle['positions_t1'], triangle_ids,
                            center_start=center0, center_end=center1,
                            orientation_xy=geometry['orientation_xy'])
                    status = certificate['status']
                    item['retained_counts'][status] += 1
                    kind = certificate.get('classification',
                        (certificate.get('separation_certificate') or {}).get('kind', status))
                    item['retained_certificate_kinds'][kind] = item['retained_certificate_kinds'].get(kind, 0)+1
                    processed += 1
                    stream_hash.update(stable_bytes({'triangle': triangle, 'certificate': certificate})+b'\n')
                    if status != 'PASS':
                        diagnostic(key, certificate.get('classification', 'exterior_not_certified'),
                            {'element': triangle['element'], 'source_vertices': triangle['source_vertices'],
                             'owners': triangle.get('owners', []),
                             'triangle_sha256': digest_value(triangle),
                             'certificate_sha256': digest_value(certificate),
                             'certificate': compact_certificate(certificate)})
                item['retained_scan_complete'] = True
            except Exception as error:
                item['retained_scan_error'] = str(error)
                diagnostic(key, 'retained_scan_incomplete', str(error))
            item['retained_triangles_checked'] = processed
            item['exterior_certificate_stream_sha256'] = stream_hash.hexdigest()
            if item['retained_counts']['REJECT']:
                exterior_status = 'REJECT'
            elif not item['retained_scan_complete'] or item['retained_counts']['UNKNOWN']:
                exterior_status = 'UNKNOWN'
            else:
                exterior_status = 'PASS'
            item['exterior_status'] = exterior_status
            item['status'] = aggregate_status((geometry['status'], exterior_status))
            overall_digest.update(stable_bytes(item)+b'\n')
            if progress is not None:
                progress({'event_id': source.get('event_id'), 'unit': key, 'status': item['status'],
                          'retained_checked': processed, 'counts': item['retained_counts']})
        report['status'] = aggregate_status(unit['status'] for unit in report['units'])
        report['local_graph_family_certificate'] = aggregate_status(unit['geometry_status'] for unit in report['units'])
        report['retained_exterior_certificate'] = aggregate_status(unit['exterior_status'] for unit in report['units'])
        report['unit_status_counts'] = dict(Counter(unit['status'] for unit in report['units']))
        report['certificate_stream_sha256'] = digest_value({
            'source_report_sha256': report['source_report_sha256'], 'all_units': report['units']})
        report['all_affine_branches_and_actual_singletons_attempted'] = len(report['units']) == len(source['segments'])+len(source['breakpoint_points'])
        report['reason'] = ('Conditional ideal source/graph/exterior checks complete; runtime and event isolation remain unadmitted.'
                            if report['status'] == 'PASS' else
                            'One or more exact graph restrictions, conservative exterior certificates, or scan budgets did not pass. No production change is authorized.')
    except Exception as error:
        report['status'] = 'REJECT'
        report['reason'] = 'Integration contract failure: '+str(error)
        diagnostic('integration', 'source_or_junction_contract_mismatch', str(error))
    return report


def write_fresh(path, value, budget_state):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    size = len(payload.encode('utf-8'))
    if budget_state[0]+size > OUTPUT_BUDGET:
        raise ValueError('Refusing to exceed the 10 MiB output budget.')
    with path.open('x', encoding='utf-8') as handle:
        handle.write(payload)
    budget_state[0] += size
    return {'path': path.name, 'bytes': size, 'sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=7200.)
    args = parser.parse_args()
    campaign, cache, output = args.campaign_root.resolve(), args.cache_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.max_seconds <= 0:
        raise ValueError('A positive bounded wall time is required.')
    for protected in (campaign, cache):
        if output == protected or protected in output.parents:
            raise ValueError('Output must not be inside original campaign/cache trees.')
    summary_path = campaign/'all_canonical_beb1_summary.json'
    campaign_summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if tuple(event['key'] for event in campaign_summary['events']) != EVENT_KEYS:
        raise ValueError('Expected the frozen four-demo event order; no event selection is permitted.')
    output.mkdir(parents=True)
    start = time.monotonic()
    deadline = start+args.max_seconds
    budget_state = [0]
    sources = [Path(__file__).resolve(), Path(__file__).with_name('window_source.py'),
               Path(__file__).with_name('window_geometry.py'), Path(__file__).with_name('window_exterior.py'),
               summary_path]
    summary = {'schema': SCHEMA, 'status': 'RUNNING', 'event_order': list(EVENT_KEYS),
        'python': platform.python_version(), 'events': [], 'max_seconds': args.max_seconds,
        'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'output_budget_bytes': OUTPUT_BUDGET, 'runtime_admission': False,
        'event_isolation_certificate': 'UNKNOWN', 'render_started': False}
    inventory = read_inventory(cache)
    for key in EVENT_KEYS:
        folder = output/key
        folder.mkdir()
        source = audit_event(campaign/key, cache)
        source_artifact = write_fresh(folder/'source.json', source, budget_state)
        if source.get('status') == 'PASS_SOURCE_BRANCH_CONTRACT' and source.get('cache_input_sha256') != inventory.digest:
            raise ValueError('Source and retained provider cache digests differ; stopping before exterior scan.')
        if file_signatures(input_files(cache)) != inventory.signatures:
            raise ValueError('Input cache changed before exterior scan.')
        report = audit_source_report(source,
            lambda t0, t1, owners: iter_ordinary_triangles(inventory, t0, t1, owners),
            deadline=deadline, progress=lambda item: print(json.dumps(item), flush=True))
        report['source_exterior_cache_bound'] = source.get('cache_input_sha256') == inventory.digest
        if file_signatures(input_files(cache)) != inventory.signatures:
            report['status'] = 'REJECT'
            report['reason'] = 'Input cache changed during exterior scan; no current-cache admission.'
        audit_artifact = write_fresh(folder/'audit.json', report, budget_state)
        summary['events'].append({'key': key, 'status': report['status'],
            'source_status': source['status'],
            'local_geometry': report.get('local_graph_family_certificate', 'UNKNOWN'),
            'exterior': report.get('retained_exterior_certificate', 'UNKNOWN'),
            'source_artifact': source_artifact, 'audit_artifact': audit_artifact,
            'diagnostics_total': report['diagnostics_total']})
        print(json.dumps({'event': key, 'status': report['status'], 'elapsed_seconds': time.monotonic()-start}), flush=True)
    summary['status'] = aggregate_status(event['status'] for event in summary['events'])
    summary['elapsed_seconds'] = time.monotonic()-start
    summary['bytes_written_before_summary'] = budget_state[0]
    write_fresh(output/'summary.json', summary, budget_state)
    print(json.dumps({'status': summary['status'], 'output': str(output),
                      'bytes_written': budget_state[0], 'events': len(summary['events'])}), flush=True)
    return 0 if summary['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
