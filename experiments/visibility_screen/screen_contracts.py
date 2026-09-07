"""Pure contracts for the preregistered three-segment visibility experiment."""
from __future__ import annotations
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C1 = HERE.parent / 'c1_lite'
for folder in (ROOT, C1, HERE.parent / 'source_splice'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return {'path': str(path.resolve()), 'sha256': file_sha(path)}


def load_protocol(path, segment_id=None):
    document = read(path)
    require(document['schema'] == 'preregistered-paper-scene-visibility-screen-v1', 'WRONG_PROTOCOL_SCHEMA')
    require(document['registration_state'].startswith('LOCKED_BEFORE_'), 'PROTOCOL_NOT_PREREGISTERED')
    rows = document['segments']
    require(len(rows) == 3 and {r['segment_id'] for r in rows} == {'forest_b', 'cave', 'mountain'},
            'FIXED_THREE_SEGMENTS_REQUIRED')
    for row in rows:
        require(row['last_frame'] - row['first_frame'] + 1 == 64 and row['fps'] == 24
                and row['pixels_per_cube'] == 6, 'FIXED_64_FRAME_PROFILE_REQUIRED')
    if segment_id is None:
        return document
    selected = [r for r in rows if r['segment_id'] == segment_id]
    require(len(selected) == 1, 'SEGMENT_NOT_PREREGISTERED')
    return document, selected[0]


def query_token(query):
    return (query['time_mode'], query['evaluation_tau'] if query['time_mode'] == 'exact'
            else query['physical_time_hex'])


def natural_queries(camera, segment, delta):
    times = camera['times_seconds']
    require(len(times) == 64, 'EXPECTED_64_NATURAL_QUERIES')
    origin = min(times)
    return [{'key': f'frame_{i + 1:04d}', 'kind': 'natural', 'frame_number': i + 1,
             'frame_index_zero_based': i, 'absolute_frame_number': segment['first_frame'] + i,
             'time_mode': 'physical', 'physical_time_hex': float(t-origin).hex(),
             'global_camera_time_hex': float(t).hex(),
             'evaluation_tau': str(F.from_float(float(float(t-origin)/delta)))}
            for i, t in enumerate(times)]


def validate_schedule_domain(events):
    """Do not silently extrapolate the frozen root-separated graph's domain."""
    require(len({e['event_id'] for e in events}) == len(events), 'DUPLICATE_EVENT_POPULATION')
    roots = {}
    for event in events:
        schedule = event['schedule']
        bounds = tuple(F(schedule['bounds'][k]) for k in ('lower', 'root', 'upper'))
        require(F(event['root']) == bounds[1], 'ROOT_SCHEDULE_DISAGREEMENT')
        for query in schedule['all_queries']:
            tau = F(query['evaluation_tau'])
            active = bounds[0] < tau < bounds[2]
            require(bool(query['active']) == active, 'PHYSICAL_BOUND_VS_EFFECTIVE_TAU_PHASE_MISMATCH')
        projection = (bounds, tuple(sorted((q['key'], query_token(q), q['evaluation_tau'], q['active'])
                                          for q in schedule['all_queries'])))
        previous = roots.setdefault(event['root'], projection)
        require(previous == projection, 'UNSUPPORTED_HETEROGENEOUS_SAME_ROOT_SCHEDULE')
    ordered = sorted(roots, key=F)
    for i, first in enumerate(ordered):
        for second in ordered[i+1:]:
            a, _, b = roots[first][0]
            c, _, d = roots[second][0]
            require(b <= c or d <= a, 'UNSUPPORTED_CROSS_ROOT_WINDOW_OVERLAP')
    return {'status': 'PASS_FROZEN_ROOT_SEPARATION_AND_QUERY_PHASE_DOMAIN',
            'root_count': len(roots), 'all_same_root_schedules_equal': True,
            'cross_root_open_windows_disjoint': True,
            'effective_tau_and_physical_hit_selection_agree': True}


def all_queries(camera, segment, delta, events):
    result = natural_queries(camera, segment, delta)
    by_key = {q['key']: q for q in result}
    roots = {}
    for event in events:
        for q in event['schedule']['all_queries']:
            if q['kind'] == 'exact_root':
                prior = roots.setdefault(q['key'], q)
            else:
                prior = by_key[q['key']]
            require(query_token(prior) == query_token(q) and prior['evaluation_tau'] == q['evaluation_tau'],
                    'QUERY_KEY_COLLISION_OR_NATIVE_TIME_DISAGREEMENT')
    return result + sorted(roots.values(), key=lambda q: F(q['evaluation_tau']))


def read_source_stage(root):
    root = Path(root)
    summary = read(root/'summary.json')
    require(summary['status'] == 'COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION', 'INCOMPLETE_SOURCE_STAGE')
    index_ref = summary['events_index']
    index_path = Path(index_ref['path'])
    if not index_path.is_absolute():
        index_path = root/index_path
    require(file_sha(index_path) == index_ref['sha256'], 'SOURCE_INDEX_HASH_CHANGED')
    events, bindings = [], {str((root/'summary.json').resolve()): file_sha(root/'summary.json'),
                           str(index_path.resolve()): file_sha(index_path)}
    for row in read(index_path):
        path = Path(row['artifact']['path'])
        if not path.is_absolute():
            path = root/path
        require(file_sha(path) == row['artifact']['sha256'], 'SOURCE_EVENT_HASH_CHANGED')
        event = read(path)
        require(event['event_id'] == row['event_id'], 'SOURCE_INDEX_EVENT_ID_CHANGED')
        events.append(event)
        bindings[str(path.resolve())] = file_sha(path)
    inv_ref = summary['source_support_inventory']
    inv_path = Path(inv_ref['path'])
    if not inv_path.is_absolute():
        inv_path = root/inv_path
    require(file_sha(inv_path) == inv_ref['sha256'], 'SOURCE_INVENTORY_HASH_CHANGED')
    inventory = read(inv_path)
    require(inventory['status'] == 'COMPLETE_FIXED_SOURCE_SUPPORT_INVENTORY'
            and len(events) == summary['canonical_event_denominator']
            and {e['event_id'] for e in events} == {e['event_id'] for e in inventory['events']},
            'SOURCE_POPULATION_INVENTORY_MISMATCH')
    bindings[str(inv_path.resolve())] = file_sha(inv_path)
    bindings.update(summary['executed_sources_sha256'])
    bindings.update(summary['input_verification'].get('input_full_sha256', {}))
    for path, expected in bindings.items():
        require(file_sha(path) == expected, 'SOURCE_BINDING_CHANGED:'+path)
    return summary, events, inventory, bindings


def summarize_visibility(events, components, frame_rows, segment, rule):
    """Count distinct events and unique per-component mask pixels, never fan sums."""
    population = {e['event_id'] for e in events}
    candidates, admitted, replacement, unknown = set(), set(), set(), set()
    compiled, qualified = set(), []
    component_by_id = {c['component_id']: c for c in components}
    component_frames = {}
    seen_components = set()
    observed = set()
    for frame in frame_rows:
        absolute = frame['query']['absolute_frame_number']
        for row in frame['events']:
            eid = row['event_id']
            require(eid in population and (eid, absolute) not in observed, 'DUPLICATE_OR_UNKNOWN_VISIBILITY_ROW')
            observed.add((eid, absolute))
            if row['domain_kind'] == 'FIXED_SOURCE_AND_REPLACEMENT':
                compiled.add(eid)
            if row['visible_baseline_source_pixels'] is None:
                unknown.add(eid)
            elif row['visible_baseline_source_pixels'] > 0:
                candidates.add(eid)
                if row['jointly_admitted']:
                    admitted.add(eid)
            if (row['visible_replacement_pixels'] or 0) > 0:
                replacement.add(eid)
        for row in frame['components']:
            cid = row['component_id']
            require(cid in component_by_id, 'UNKNOWN_VISIBILITY_COMPONENT')
            require((cid, absolute) not in seen_components, 'DUPLICATE_COMPONENT_FRAME')
            seen_components.add((cid, absolute))
            require(row['decision'] == component_by_id[cid]['decision'], 'COMPONENT_DECISION_CHANGED')
            if (row['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE'
                    and row['visible_baseline_union_pixels'] >= rule['minimum_baseline_support_pixels_per_frame']
                    and row['visible_replacement_union_pixels'] >= rule['minimum_actual_replacement_pixels_on_each_qualifying_frame']):
                component_frames.setdefault(cid, set()).add(absolute)
    expected = {(e['event_id'], segment['first_frame'] + q['frame_index_zero_based'])
                for e in events for q in e['schedule']['natural'] if q['active']}
    require(observed == expected, 'INCOMPLETE_ACTIVE_NATURAL_VISIBILITY_DENOMINATOR')
    for cid, frames in component_frames.items():
        if len(frames) >= rule['minimum_natural_frames']:
            qualified.append({'segment_id': segment['segment_id'], 'component_id': cid,
                              'earliest_qualifying_absolute_frame': min(frames),
                              'qualifying_absolute_frames': sorted(frames),
                              'events': component_by_id[cid]['events']})
    qualified.sort(key=lambda x: (x['segment_id'], x['earliest_qualifying_absolute_frame'], x['component_id']))
    return {'canonical_events': len(population), 'natural_event_frame_queries': len(observed),
            'visible_candidate_events': len(candidates), 'visible_admitted_source_events': len(admitted),
            'visible_replacement_events': len(replacement), 'unknown_support_visibility_events': len(unknown),
            'visible_source_admission_rate': len(admitted)/len(candidates) if candidates and not unknown else None,
            'compiled_source_visible_candidates': len(compiled & candidates),
            'compiled_source_visible_admitted': len(compiled & admitted),
            'qualified_components': qualified, 'first_qualifying_component': qualified[0] if qualified else None,
            'visible_artifact_impact_coverage': None,
            'visibility_scope': 'Original-camera baseline source-support visibility, not counterfactual artifact impact.',
            'zero_denominator_is_not_zero_percent': True}


def source_decision_counts(events):
    return dict(Counter(e.get('decisive_rejection', {}).get('reason_code',
                        e.get('source_policy_witness', {}).get('reason_code', e['decision'])) for e in events))
