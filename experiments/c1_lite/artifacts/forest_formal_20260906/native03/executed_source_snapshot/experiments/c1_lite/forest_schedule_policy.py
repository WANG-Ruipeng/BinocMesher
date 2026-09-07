"""Fixed, explicit-denominator Forest requested-schedule policy helpers."""
from fractions import Fraction as F
import math

import numpy as np


def fraction(value):
    if isinstance(value, dict):
        return F(value['numerator'], value['denominator'])
    return F(value)


def physical_bound(tau, delta):
    tau = F(tau)
    return float(np.longdouble(tau.numerator)*np.longdouble(delta)/np.longdouble(tau.denominator))


def make_schedule(camera, lower, root, upper, *, actual_delta=None):
    lower, root, upper = map(F, (lower, root, upper))
    if not 0 <= lower < root < upper:
        raise ValueError('Invalid fixed source window.')
    times = tuple(map(float, camera['times_seconds']))
    if not times or not all(math.isfinite(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Expected strictly increasing finite original camera times.')
    origin = min(times)
    reconstructed = float(fraction(camera['time_mapping']['delta_seconds']))
    delta = reconstructed if actual_delta is None else float(actual_delta)
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError('Invalid actual discrete time scale.')
    if actual_delta is not None and delta.hex() != reconstructed.hex():
        raise ValueError('Actual time scale differs from original camera-input reconstruction.')
    a, b = physical_bound(lower, delta), physical_bound(upper, delta)
    hit = [i for i, t in enumerate(times) if a < float(t-origin) < b]
    selected = set(hit)
    if hit:
        if min(hit) > 0:
            selected.add(min(hit)-1)
        if max(hit)+1 < len(times):
            selected.add(max(hit)+1)
    natural = []
    for index in sorted(selected):
        local = float(times[index]-origin)
        tau = F.from_float(float(local/delta))
        natural.append({'key': f'frame_{index+1:04d}', 'kind': 'natural',
            'frame_number': index+1, 'frame_index_zero_based': index,
            'time_mode': 'physical', 'physical_time_hex': local.hex(),
            'global_camera_time_hex': times[index].hex(), 'evaluation_tau': str(tau),
            'active': index in hit})
    exact = {'key': 'root_'+str(root).replace('/', '_'), 'kind': 'exact_root',
        'frame_number': None, 'time_mode': 'exact', 'evaluation_tau': str(root),
        'physical_time_hex': physical_bound(root, delta).hex(), 'active': True}
    return {'natural': natural, 'exact_root': exact,
        'all_queries': natural+[exact], 'hit_frame_numbers': [i+1 for i in hit],
        'all_camera_frames': len(times), 'natural_hit_count': len(hit),
        'selected_natural_count': len(natural), 'delta_t_hex': delta.hex(),
        'origin_seconds_hex': origin.hex(),
        'time_mapping_status': 'ACTUAL_NATIVE_INITIALIZATION_VERIFIED' if actual_delta is not None else 'RECONSTRUCTED_INPUT_ONLY',
        'bounds': {'lower': str(lower), 'root': str(root), 'upper': str(upper)},
        'root_excluded_from_natural_rates': True,
        'other_events_policy': 'UNCHANGED_SAME_ORDINARY_BASELINE'}


def summarize_decisions(events, expected_count=131):
    """Admission point estimate is absent while any event remains unresolved."""
    if len(events) != expected_count or len({e['event_id'] for e in events}) != expected_count:
        raise ValueError('Population omitted or duplicated canonical event IDs.')
    allowed = {'ADMITTED_REQUESTED_SCHEDULE', 'REJECTED_FIXED_POLICY', 'UNKNOWN'}
    if any(e.get('decision') not in allowed for e in events):
        raise ValueError('Malformed final event decision.')
    admitted = [e for e in events if e['decision'] == 'ADMITTED_REQUESTED_SCHEDULE']
    rejected = [e for e in events if e['decision'] == 'REJECTED_FIXED_POLICY']
    unknown = [e for e in events if e['decision'] == 'UNKNOWN']
    for event in admitted:
        if event.get('runtime', {}).get('status') != 'COMMITTED_REQUESTED_SCHEDULE':
            raise ValueError('Source/ideal PASS cannot authorize runtime admission.')
    for event in rejected:
        if not event.get('decisive_rejection', {}).get('certified_necessary_policy_failure'):
            raise ValueError('Policy rejection lacks a decisive necessary-gate witness.')
    counts = {'admitted': len(admitted), 'rejected_fixed_policy': len(rejected), 'unknown': len(unknown)}
    return {'canonical_event_denominator': expected_count, **counts,
        'policy_admission_rate': len(admitted)/expected_count if not unknown else None,
        'policy_admission_rate_bounds': [len(admitted)/expected_count, (len(admitted)+len(unknown))/expected_count],
        'runtime_events_attempted': sum(bool(e.get('runtime_attempted')) for e in events),
        'conditional_runtime_commit_rate': (len(admitted)/sum(bool(e.get('runtime_attempted')) for e in events)
                                            if any(e.get('runtime_attempted') for e in events) else None),
        'rate_scope': 'Fixed construction-policy, individual-event requested-schedule admission against the same baseline; not same-root union or all-time admission.',
        'same_root_union_rate': None,
        'all_events_resolved': not unknown}
