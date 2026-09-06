#!/usr/bin/env python3
"""Raw ordinary emission ledger over exact source-time cells.

This adds slicing.cpp's original-time rejection to the frozen processed-source
model. It does NOT reproduce binary32 interpolation, C++ array numbering, or
extra_smooth special identities. The exported faces are a canonical SourceVID
geometric quotient; every contributing raw owner remains in the ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import window_source as source

SCHEMA = 'c1-lite-raw-retained-v1'
COORDINATE_MODEL = 'ideal rational interpolation of serialized HV binary32'
EVENT_KEYS = ('event-00-6d2d6dc2cdd7', 'event-01-e40703e638f3',
              'event-02-88ade47aa4fa', 'event-03-0c61aa0982ef')
MAX_OUTPUT_BYTES = 2 * 1024 * 1024


class RetainedStop(source.SourceStop):
    pass


def require(condition, message):
    if not condition:
        raise RetainedStop(message)


def stable_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(stable_bytes(value)).hexdigest()


def partition_points(inventory, lower, root, upper):
    """Complete old source partition plus unexpanded record-time predicates."""
    reasons = defaultdict(set)
    for time, why in source.partition_points(inventory, lower, root, upper):
        reasons[time].update(why)
    for record in inventory.records:
        for time in record.times:
            if lower <= time <= upper:
                reasons[Fraction(time)].add('raw_original_record_time')
    return [(time, sorted(why)) for time, why in sorted(reasons.items())]


def required_splits(inventory, t0, t1):
    t0, t1 = Fraction(t0), Fraction(t1)
    require(t0 <= t1, 'Reversed raw source-time cell.')
    if t0 == t1:
        return []
    midpoint = (t0 + t1) / 2
    return [(time, why) for time, why in
            partition_points(inventory, t0, midpoint, t1)
            if t0 < time < t1 and any(w != 'critical_root' for w in why)]


@dataclass
class RetainedUnit:
    t0: Fraction
    t1: Fraction
    ledger: tuple[dict, ...]
    triangles: tuple[dict, ...]
    cache_digest: str


def build_retained_unit(inventory, t0, t1, excluded_owners=(), *, mode='raw'):
    """Build one full raw-owner ledger and its retained geometric quotient.

    A non-singleton labels the OPEN cell. Coordinates at t0/t1 are its limits;
    exact endpoints require separate singleton calls. Enumerating every discrete
    predicate first, not sampling agreement, establishes branch membership.
    """
    require(mode == 'raw', 'UNSUPPORTED_EXTRA_SMOOTH: special VID/coordinate semantics are not modeled.')
    t0, t1 = Fraction(t0), Fraction(t1)
    splits = required_splits(inventory, t0, t1)
    require(not splits, 'UNPARTITIONED_RAW_TIME_CELL: required splits ' +
            ', '.join(str(time) for time, _ in splits))
    tau = (t0 + t1) / 2
    active = set(source._active_groups(tau, inventory.groups, inventory.maximum))
    requested_rows = [tuple(row) for row in excluded_owners]
    require(all(len(row) == 7 for row in requested_rows), 'Malformed suppression owner.')
    excluded = set(requested_rows)
    require(len(excluded) == len(requested_rows), 'Duplicated suppression owner.')
    ledger, groups, formulae = [], defaultdict(list), {}
    known_owners, emitted_owners = set(), set()
    for record in inventory.records:
        selected = -1
        for index, threshold in enumerate(record.expanded):
            if tau >= threshold:
                selected = index
            else:
                break
        for interval_index, polygons in enumerate(record.intervals):
            for face_index, polygon in enumerate(polygons):
                for fan_index in range(max(0, len(polygon) - 2)):
                    owner = (record.element, record.group, record.start,
                             record.sorted_index, interval_index, face_index, fan_index)
                    require(owner not in known_owners, 'Duplicated inventory raw owner.')
                    known_owners.add(owner)
                    row = {'owner': list(owner), 'state': 'skip'}
                    if record.group not in active:
                        row['reason'] = 'inactive_cache_group'
                    elif tau < record.start - 1:
                        row['reason'] = 'record_not_loaded'
                    elif selected != interval_index:
                        row['reason'] = 'interval_not_selected'
                    elif interval_index == 0 and tau < record.times[0]:
                        row['reason'] = 'before_original_start'
                    elif interval_index == len(record.intervals) - 1 and tau > record.times[-1]:
                        row['reason'] = 'after_original_end'
                    else:
                        # raw: start_discon_face/end_discon_face are empty.
                        # Equality at the unexpanded endpoints is NOT skipped.
                        emitted_owners.add(owner)
                        raw = (polygon[0], polygon[fan_index + 1], polygon[fan_index + 2])
                        output = []
                        for vid in raw:
                            effective, coeff, visible, branch = source.source_formula(
                                vid, tau, inventory.hypervertices)
                            key = record.element, effective
                            require(key not in formulae or formulae[key][0] == coeff,
                                    'Same element/SourceVID has inconsistent ideal trajectories.')
                            formulae[key] = coeff, visible, branch
                            output.append(effective)
                        face = source.canonical_face(output)
                        group_key = record.element, face
                        row.update(state='suppress' if owner in excluded else 'retained',
                                   reason='requested_exact_owner' if owner in excluded else 'raw_emitted',
                                   source_vertices=[vid.text() for vid in output])
                        groups[group_key].append((owner, row))
                    ledger.append(row)
    require(excluded <= emitted_owners,
            'SUPPRESSION_NOT_RAW_EMITTED: ' + repr(sorted(excluded - emitted_owners)[:8]))
    triangles = []
    for (element, face), contributors in sorted(groups.items()):
        owners = {owner for owner, _ in contributors}
        require(not (owners & excluded) or owners <= excluded,
                'PARTIAL_SUPPRESSION_OF_REPLICATED_FACE: ' + repr(sorted(owners)))
        if owners <= excluded:
            continue
        semantic_id = digest({'element': element, 'oriented_source_face': [v.text() for v in face]})
        for _, row in contributors:
            row['retained_source_face_id'] = semantic_id
        coefficients = [formulae[element, vid][0] for vid in face]
        triangles.append({'element': element,
            'source_vertices': [vid.text() for vid in face],
            'owners': [list(owner) for owner in sorted(owners)],
            'positions_t0': [source.vj(source.evaluate(c, t0)) for c in coefficients],
            'positions_t1': [source.vj(source.evaluate(c, t1)) for c in coefficients],
            'in_view': [formulae[element, vid][1] for vid in face],
            'source_branches': [formulae[element, vid][2] for vid in face],
            'retained_source_face_id': semantic_id,
            'raw_occurrence_count': len(contributors),
            'coordinate_model': COORDINATE_MODEL})
    return RetainedUnit(t0, t1, tuple(ledger), tuple(triangles), inventory.digest)


def iter_retained_triangles(inventory, t0, t1, excluded_owners=(), *, mode='raw'):
    """A-runner adapter; raises before yielding on unsupported/partial cells."""
    yield from build_retained_unit(inventory, t0, t1, excluded_owners, mode=mode).triangles


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _identically_degenerate(triangle):
    p = [tuple(source.fr(x) for x in row) for row in triangle['positions_t0']]
    q = [tuple(source.fr(x) for x in row) for row in triangle['positions_t1']]
    a, b = _sub(p[1], p[0]), _sub(p[2], p[0])
    da, db = _sub(_sub(q[1], q[0]), a), _sub(_sub(q[2], q[0]), b)
    mixed = tuple(x+y for x, y in zip(_cross(a, db), _cross(da, b)))
    return all(x == 0 for coefficient in (_cross(a, b), mixed, _cross(da, db)) for x in coefficient)


def compact_manifest(unit, *, boundary_cycle=(), element=None, max_witnesses=8):
    """Compact complete counts/hashes, bounded witnesses; never final-array IDs."""
    require(max_witnesses >= 0, 'Negative witness limit.')
    require(not boundary_cycle or isinstance(element, int), 'Boundary requires an element namespace.')
    boundary = set(boundary_cycle)
    witnesses = []
    touching = 0
    repeated = identically_degenerate = 0
    for triangle in unit.triangles:
        ids = triangle['source_vertices']
        is_repeated = len(set(ids)) < 3
        zero_area = _identically_degenerate(triangle)
        repeated += is_repeated
        identically_degenerate += zero_area
        shared = sorted(boundary & set(ids)) if triangle['element'] == element else []
        if zero_area and shared:
            touching += 1
            if len(witnesses) < max_witnesses:
                witnesses.append({'kind': 'BASELINE_DEGENERACY_TOUCHES_SOURCE_INTERFACE',
                    'shared_source_vertices': shared,
                    'repeated_effective_SourceVID': is_repeated,
                    'exact_zero_area_on_cell': True, **triangle})
    rows = sorted(unit.ledger, key=lambda row: tuple(row['owner']))
    return {'schema': SCHEMA, 'status': 'PASS_RAW_EMISSION_MODEL',
        'mode': 'raw', 'coordinate_model': COORDINATE_MODEL,
        'time_domain': 'actual_singleton' if unit.t0 == unit.t1 else 'open_cell_with_endpoint_limits',
        't0': source.fj(unit.t0), 't1': source.fj(unit.t1),
        'cache_input_sha256': unit.cache_digest,
        'owner_universe': 'Every serialized polygon fan occurrence in the inventory; skipped owners are retained in the ledger denominator.',
        'raw_owner_count': len(rows),
        'owner_state_counts': dict(sorted(Counter(row['state'] for row in rows).items())),
        'owner_reason_counts': dict(sorted(Counter(row['reason'] for row in rows).items())),
        'owner_ledger_sha256': digest(rows),
        'retained_source_face_count': len(unit.triangles),
        'retained_raw_occurrence_count': sum(t['raw_occurrence_count'] for t in unit.triangles),
        'retained_manifest_sha256': digest(unit.triangles),
        'repeated_effective_SourceVID_faces': repeated,
        'identically_zero_area_source_faces': identically_degenerate,
        'interface_degenerate_source_faces': touching,
        'interface_degenerate_witnesses': witnesses,
        'interface_gate': 'REJECT_BASELINE_DEGENERACY_AT_INTERFACE' if touching else 'UNKNOWN_NO_COMPLETE_INTERFACE_CERTIFICATE',
        'runtime_final_array_identity': 'UNKNOWN_CANONICAL_SOURCE_GEOMETRIC_QUOTIENT_ONLY',
        'binary32_runtime_certificate': 'UNKNOWN', 'extra_smooth': 'UNSUPPORTED',
        'runtime_admission': False, 'scan_complete': True}


def audit_retained_unit(inventory, t0, t1, excluded_owners=(), *, boundary_cycle=(),
                        element=None, mode='raw', max_witnesses=8):
    try:
        unit = build_retained_unit(inventory, t0, t1, excluded_owners, mode=mode)
        return compact_manifest(unit, boundary_cycle=boundary_cycle, element=element,
                                max_witnesses=max_witnesses)
    except RetainedStop as error:
        splits = required_splits(inventory, t0, t1) if Fraction(t0) <= Fraction(t1) else []
        return {'schema': SCHEMA, 'status': 'UNKNOWN' if mode != 'raw' or splits else 'REJECT',
            'reason': str(error), 'required_splits': [source.fj(t) for t, _ in splits],
            'mode': mode, 'scan_complete': False, 'runtime_admission': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='Frozen four-event window_audit directory containing source.json files.')
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory; never overwritten.')
    args = parser.parse_args()
    output, source_root, cache = args.output.resolve(), args.source_root.resolve(), args.cache_root.resolve()
    require(not output.exists(), 'Output already exists.')
    require(output != cache and cache not in output.parents and
            output != source_root and source_root not in output.parents, 'Output overlaps frozen inputs.')
    inventory = source.read_inventory(cache)
    results = []
    # No source re-audit or mesher initialization; consume the frozen protocol.
    for key in EVENT_KEYS:
        path = source_root/key/'source.json'
        report = json.loads(path.read_text())
        require(report['cache_input_sha256'] == inventory.digest, 'Frozen source/cache digest mismatch.')
        units = []
        for index, segment in enumerate(report['segments']):
            units.append(('affine_branch', index, source.fr(segment['t0']), source.fr(segment['t1']), segment['owners']))
        for index, point in enumerate(report['breakpoint_points']):
            tau = source.fr(point['time'])
            units.append(('actual_singleton', index, tau, tau, point['owners']))
        case = {'schema': SCHEMA, 'event': key, 'source_report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'units': [], 'runtime_admission': False}
        elements = {owner[0] for _, _, _, _, owners in units for owner in owners}
        require(len(elements) == 1, 'Event source element is ambiguous.')
        for kind, index, t0, t1, owners in units:
            row = audit_retained_unit(inventory, t0, t1, owners,
                boundary_cycle=report['boundary_cycle'], element=next(iter(elements)))
            row.update(kind=kind, index=index)
            case['units'].append(row)
        case['raw_emission_models_complete'] = all(u['status'] == 'PASS_RAW_EMISSION_MODEL' for u in case['units'])
        case['interface_degenerate_witnesses_total'] = sum(u.get('interface_degenerate_source_faces', 0) for u in case['units'])
        results.append(case)
        print(json.dumps({'event': key, 'raw_models_complete': case['raw_emission_models_complete'],
                          'interface_degenerate_witnesses_total': case['interface_degenerate_witnesses_total']}), flush=True)
    require(source.file_signatures(source.input_files(cache)) == inventory.signatures, 'Cache changed during raw audit.')
    payloads = {case['event']+'.json': (json.dumps(case, indent=2, sort_keys=True)+'\n').encode() for case in results}
    summary = {'schema': SCHEMA, 'event_order': list(EVENT_KEYS),
        'status': 'COMPLETE_RAW_SOURCE_EMISSION_DIAGNOSTIC',
        'runtime_admission': False, 'extra_smooth': 'UNSUPPORTED',
        'cache_input_sha256': inventory.digest,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'artifacts': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in payloads.items()}}
    payloads['summary.json'] = (json.dumps(summary, indent=2, sort_keys=True)+'\n').encode()
    require(sum(map(len, payloads.values())) <= MAX_OUTPUT_BYTES, 'Compact output exceeds two MiB.')
    output.mkdir(parents=True)
    for name, data in payloads.items():
        with (output/name).open('xb') as handle:
            handle.write(data)
    print(json.dumps({'output': str(output), 'bytes': sum(map(len, payloads.values())), 'events': len(results)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
