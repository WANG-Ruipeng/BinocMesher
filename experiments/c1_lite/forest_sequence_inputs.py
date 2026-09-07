"""Hash-bound inputs shared by component certification and sequence playback."""
from fractions import Fraction as F
import json
from pathlib import Path

from forest_native_campaign import file_sha
from forest_component_graph import digest

HERE = Path(__file__).resolve().parent
FORMAL = HERE/'artifacts/forest_formal_20260906'
CAMERA = HERE/'artifacts/repair_20260906/forest/stage64/camera_inputs.json'
EFFECTIVE = HERE/'artifacts/repair_20260906/forest/stage64/effective_inputs.json'
CACHE = Path('/home/warpwang/binoc-runs/forest-census64-repair-20260906/HyperMesh/OpaqueTerrain')
BUILD = Path('/home/warpwang/binoc-runs/forest-observer-sourcevid-20260906/build')
LIBRARY_SHA = '3a0bab2f77142708e83251520dd11937de20cd198a887bc4cc8e8ce6bd543197'
AUDIT_SHA = 'eda0f0144150296651cf7bac3127b43843b1fc469256799622c49b1b6f7d34e7'
EVENTS_SHA = 'f6425b58e74b4f12cb4f8030b5e5be1d5d27ea99513e761b0ec4044ac9e826bc'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_certified_inputs():
    audit_path = FORMAL/'native04_independent_audit_v2.json'
    if file_sha(audit_path) != AUDIT_SHA:
        raise ValueError('Independent native04 certificate audit changed.')
    audit = read(audit_path)
    bound = dict(audit['audited_files_sha256'])
    bound[str(audit_path)] = AUDIT_SHA
    for path, expected in bound.items():
        if file_sha(path) != expected:
            raise ValueError('Audited event/query/input artifact changed: '+path)
    native = FORMAL/'native04'
    summary = read(native/'summary.json')
    if (summary['status'] != 'COMPLETE_REAL_FIXED_POLICY_RATE' or summary['admitted'] != 31
            or summary['unknown'] != 0 or summary['native_library_sha256'] != LIBRARY_SHA):
        raise ValueError('Frozen per-event campaign is not the audited 31/131 result.')
    for path, expected in summary['executed_sources_sha256'].items():
        if file_sha(path) != expected:
            raise ValueError('Frozen certificate implementation changed: '+path)
        bound[path] = expected
    verification = read(FORMAL/'attempt01/input_verification.json')
    for path, expected in verification['input_full_sha256'].items():
        if file_sha(path) != expected:
            raise ValueError('Original source cache differs from certificate: '+path)
        bound[path] = expected
    events = []
    for row in read(native/'events_index.json'):
        path = native/row['artifact']['path']
        if file_sha(path) != row['artifact']['sha256']:
            raise ValueError('Native event index binding differs.')
        event = read(path)
        if row['event_id'] != event['event_id'] or row['decision'] != event['decision']:
            raise ValueError('Native event identity or decision differs.')
        events.append(event)
    if len(events) != 131 or digest(sorted(e['event_id'] for e in events)) != EVENTS_SHA:
        raise ValueError('Fixed full 131-event denominator changed.')
    queries = {}
    for path in sorted((native/'queries').glob('*.json')):
        if bound.get(str(path)) != file_sha(path):
            raise ValueError('Native query lacks independent audit hash binding.')
        query = read(path)
        key = query['query']['key']
        if key in queries:
            raise ValueError('Duplicate native query.')
        queries[key] = query
    if len(queries) != 22:
        raise ValueError('Incomplete audited query set.')
    return {'events': events, 'queries': queries, 'camera': read(CAMERA),
            'effective': read(EFFECTIVE), 'summary': summary, 'bindings': bound}


def all_natural_queries(camera, actual_delta):
    times = tuple(map(float, camera['times_seconds']))
    origin = min(times)
    if len(times) != 64 or float(actual_delta).hex() != '0x1.500053e2d6239p-1':
        raise ValueError('Unexpected frozen 64-frame schedule or actual delta.')
    return [{'key': f'frame_{i+1:04d}', 'kind': 'natural', 'frame_number': i+1,
        'frame_index_zero_based': i, 'time_mode': 'physical',
        'physical_time_hex': float(t-origin).hex(), 'global_camera_time_hex': t.hex(),
        'evaluation_tau': str(F.from_float(float(float(t-origin)/actual_delta)))}
        for i, t in enumerate(times)]


def bound_plan(event, query, baseline, spec, footprint):
    """Inherit a certificate only for its exact original query and baseline."""
    import numpy as np
    if event['decision'] != 'ADMITTED_REQUESTED_SCHEDULE':
        raise ValueError('Rejected event cannot contribute a replacement.')
    cases = [c for c in event['runtime']['cases'] if c['query']['key'] == query['key']]
    if len(cases) != 1:
        raise ValueError('Missing or duplicated certified event query.')
    case = cases[0]
    if any(case['query'][key] != query[key] for key in ('evaluation_tau', 'time_mode', 'physical_time_hex')):
        raise ValueError('Certificate/query time mismatch.')
    if case['audit']['status'] != 'PASS' or footprint['status'] != 'COMPLETE_ACTUAL_REQUESTED_SUPPORT':
        raise ValueError('Uncertified or unresolved actual support.')
    plan = case['plan']
    if plan['source_digest'] != spec.source_digest or case['audit']['source_digest'] != spec.source_digest:
        raise ValueError('Certificate/source contract mismatch.')
    if (plan['element'] != spec.element
            or sorted(plan['removed_face_rows']) != sorted(footprint['source_face_rows'])
            or list(plan['boundary_actual_ids']) != footprint['boundary_actual_ids']
            or sorted(map(tuple, plan['consumed_owners'])) != sorted(map(tuple, footprint['consumed_owners']))
            or not np.array_equal(plan['center'], footprint['center'])):
        raise ValueError('Certified plan differs from freshly resolved actual support.')
    if (plan['baseline_vertex_count'] != baseline[spec.element]['vertices']
            or plan['baseline_face_count'] != baseline[spec.element]['faces']):
        raise ValueError('Certificate baseline array sizes differ.')
    # Full five-element baseline hash parity is a mandatory caller precondition;
    # it is repeated here when callers attach the original bound query receipt.
    if footprint.get('certified_baseline') != baseline:
        raise ValueError('Missing complete five-element certificate baseline binding.')
    return plan
