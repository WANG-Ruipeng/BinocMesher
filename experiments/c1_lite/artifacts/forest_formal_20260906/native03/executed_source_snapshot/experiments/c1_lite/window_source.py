#!/usr/bin/env python3
"""Finite-branch certificate for a fixed four-SourceVID source patch.

This is NOT a runtime binary32 or exterior-embedding certificate. Full demo
processed streams are read so replicated raw owners cannot silently be omitted.
All time predicates of that parser are partitioned at their exact thresholds;
one representative labels a proven constant branch, not a sampled trajectory.
Numerical best-conditioned selection is not certified: only a unique
combinatorial disk with the frozen boundary is accepted, with its geometric
conditioning explicitly left to the separate geometry certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

SOURCE = Path(__file__).resolve().parents[1] / 'source_splice'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from processed_mesh import (HVID, SourceVID, Hypervertex, _active_groups,
    _parse_processed_metadata, _read_counted_vector, infer_cache_shape,
    parse_hypervertices, parse_serialized_vid, selected_event_rows)
from compile_splice_plans import directed_boundary_cycle

SCHEMA = 'c1-lite-window-source-v1'
MAX_JSON_BYTES = 2*1024*1024
MAX_CACHE_BYTES = 256*1024*1024


class SourceStop(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise SourceStop(message)


def fj(value):
    value = Fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def fr(value):
    return Fraction(int(value['numerator']), int(value['denominator']))


def vj(values):
    return [fj(value) for value in values]


def vf(values):
    require(len(values) == 3 and all(math.isfinite(float(x)) for x in values),
            'Nonfinite or malformed spatial coordinate.')
    return tuple(Fraction.from_float(float(value)) for value in values)


def vid(value):
    first, second = value.split('|')
    return SourceVID.canonical(HVID(*map(int, first.split(':'))),
                               HVID(*map(int, second.split(':'))))


def canonical_face(face):
    face = tuple(face)
    return min(face[i:]+face[:i] for i in range(3))


def same_cycle(first, second):
    return len(first) == len(second) and any(
        tuple(first) == tuple(second[i:])+tuple(second[:i]) for i in range(len(second)))


def evaluate(coefficients, tau):
    intercept, slope = coefficients
    return tuple(a+b*tau for a, b in zip(intercept, slope))


def source_thresholds(source, hypervertices):
    a, b = hypervertices[source.first], hypervertices[source.second]
    if a.time > b.time:
        a, b = b, a
    lower, upper = a.time, b.time
    effective_lower, effective_upper = lower, upper
    if a.in_view != b.in_view and lower+a.halfspan <= upper-b.halfspan:
        if b.in_view:
            effective_upper -= b.halfspan
        else:
            effective_lower += a.halfspan
    return lower, upper, effective_lower, effective_upper


def effective_source_id(source, tau, hypervertices):
    ah, bh = source.first, source.second
    a, b = hypervertices[ah], hypervertices[bh]
    if a.time > b.time:
        a, b, ah, bh = b, a, bh, ah
    lower, upper, e0, e1 = source_thresholds(source, hypervertices)
    clamped = max(Fraction(lower), min(tau, Fraction(upper)))
    effective = max(Fraction(e0), min(clamped, Fraction(e1)))
    if e0 == e1:
        key = ah if clamped < e0 else bh
        return SourceVID.canonical(key, key)
    if effective == e0:
        return SourceVID.canonical(ah, ah)
    if effective == e1:
        return SourceVID.canonical(bh, bh)
    return SourceVID.canonical(ah, bh)


def source_formula(source, tau, hypervertices):
    """Exact affine formula on the branch containing tau (singleton allowed)."""
    ah, bh = source.first, source.second
    a, b = hypervertices[ah], hypervertices[bh]
    if a.time > b.time:
        a, b, ah, bh = b, a, bh, ah
    lower, upper, e0, e1 = source_thresholds(source, hypervertices)
    clamped = max(Fraction(lower), min(tau, Fraction(upper)))
    effective = max(Fraction(e0), min(clamped, Fraction(e1)))
    pa, pb = vf(a.position), vf(b.position)
    zero = (Fraction(0),)*3
    if e0 == e1:
        coeff = (pa if clamped < e0 else pb, zero)
        branch = 'zero_effective_span_left' if clamped < e0 else 'zero_effective_span_right'
    elif effective == e0:
        coeff, branch = (pa, zero), 'lower_clamped'
    elif effective == e1:
        coeff, branch = (pb, zero), 'upper_clamped'
    else:
        slope = tuple((y-x)/(e1-e0) for x, y in zip(pa, pb))
        coeff = (tuple(x-m*e0 for x, m in zip(pa, slope)), slope)
        branch = 'affine'
    visible = not ((not a.in_view and clamped < e1) or
                   (not b.in_view and clamped > e0) or (not a.in_view and not b.in_view))
    return effective_source_id(source, tau, hypervertices), coeff, bool(visible), branch


@dataclass(frozen=True)
class Record:
    group: int
    start: int
    sorted_index: int
    element: int
    times: tuple[int, ...]
    intervals: tuple[tuple[tuple[SourceVID, ...], ...], ...]

    @property
    def identity(self):
        return self.group, self.start, self.sorted_index

    @property
    def expanded(self):
        return (self.times[0]-1, *self.times[1:-1], self.times[-1]+1)


@dataclass
class Inventory:
    records: tuple[Record, ...]
    hypervertices: dict
    groups: int
    maximum: int
    sources: tuple[SourceVID, ...]
    digest: str
    signatures: tuple
    bytes_read: int


def input_files(cache):
    primary = sorted((cache/'processed_hyperpolys').glob('*.bin'))
    hypervertices = sorted((cache/'hypervertices').glob('*.bin'))
    return primary+hypervertices+[cache/'event_registry_p1.csv']


def file_signatures(files):
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files)


def read_inventory(cache):
    cache = Path(cache).resolve()
    files = input_files(cache)
    signatures = file_signatures(files)
    size = sum(row[1] for row in signatures)
    require(size <= MAX_CACHE_BYTES, 'Demo source audit refuses caches above 256 MiB.')
    inventory = _read_inventory(str(cache), signatures)
    require(file_signatures(files) == signatures, 'Cache changed while reading.')
    return inventory


@lru_cache(maxsize=1)
def _read_inventory(cache_text, signatures):
    cache = Path(cache_text)
    groups, maximum = infer_cache_shape(cache)
    require(groups > 0 and maximum == 2*groups, 'Unsupported temporal cache shape.')
    folder = cache/'processed_hyperpolys'
    primaries = sorted(path for path in folder.glob('*.bin')
                       if re.fullmatch(r'\d+_\d+\.bin', path.name))
    require(bool(primaries), 'No processed primary streams.')
    expected_metadata = {path.with_name(path.stem+'_hpmeta.bin').name for path in primaries}
    require(expected_metadata == {p.name for p in folder.glob('*_hpmeta.bin')},
            'Primary/metadata filename sets differ.')
    digest = hashlib.sha256()
    for path_text, _, _ in signatures:
        path = Path(path_text)
        digest.update(path.relative_to(cache).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    hypervertices = parse_hypervertices(cache)
    records, sources = [], set()
    for path in primaries:
        group, start = map(int, path.stem.split('_'))
        require(0 <= group < groups and 0 <= start < maximum, 'Stream identity outside cache shape.')
        payload = path.read_bytes()
        metadata = _parse_processed_metadata(path.with_name(path.stem+'_hpmeta.bin').read_bytes(), group, start)
        offset, record_index = 0, 0
        seen_indices = set()
        while offset < len(payload):
            element = struct.unpack_from('<b', payload, offset)[0]
            offset += 1
            blobs, offset = _read_counted_vector(payload, offset, 1)
            times = tuple(struct.unpack_from('<b', blob, 0)[0] for blob in blobs)
            require(len(times) >= 2 and all(a < b for a, b in zip(times, times[1:])), 'Invalid owner time sequence.')
            intervals = []
            for _ in range(len(times)-1):
                faces = []
                while True:
                    blobs, offset = _read_counted_vector(payload, offset, 16)
                    if not blobs:
                        break
                    face = tuple(parse_serialized_vid(blob) for blob in blobs)
                    # Point/edge polygons parse normally and emit no fan faces.
                    faces.append(face)
                    sources.update(face)
                intervals.append(tuple(faces))
            require(record_index < len(metadata), 'Primary stream outlives metadata.')
            index = metadata[record_index]
            require(index not in seen_indices, 'Duplicated raw owner record identity.')
            seen_indices.add(index)
            records.append(Record(group, start, index, element, times, tuple(intervals)))
            record_index += 1
        require(offset == len(payload) and record_index == len(metadata), 'Primary/metadata length mismatch.')
    for source in sources:
        require(source.first in hypervertices and source.second in hypervertices, 'Missing source HVID.')
    return Inventory(tuple(records), hypervertices, groups, maximum,
                     tuple(sorted(sources)), digest.hexdigest(), signatures,
                     sum(row[1] for row in signatures))


def partition_points(inventory, lower, root, upper):
    reasons = defaultdict(set)
    for tau, name in ((lower, 'window_lower'), (root, 'critical_root'), (upper, 'window_upper')):
        reasons[tau].add(name)
    for group in range(inventory.groups+1):
        reasons[Fraction(group*inventory.maximum, inventory.groups)].add('active_cache_group')
    for record in inventory.records:
        reasons[Fraction(record.start-1)].add('record_load_threshold')
        for tau in record.expanded:
            reasons[Fraction(tau)].add('expanded_owner_interval')
    for source in inventory.sources:
        raw0, raw1, e0, e1 = source_thresholds(source, inventory.hypervertices)
        for tau in (raw0, raw1):
            reasons[Fraction(tau)].add('raw_source_time_or_clamp')
        for tau in (e0, e1):
            reasons[Fraction(tau)].add('effective_clamp_SourceVID_visibility')
    return [(tau, sorted(why)) for tau, why in sorted(reasons.items()) if lower <= tau <= upper]


def topology_at(inventory, tau, event_record_ids):
    active = set(_active_groups(tau, inventory.groups, inventory.maximum))
    effective = {source: effective_source_id(source, tau, inventory.hypervertices)
                 for source in inventory.sources}
    groups = defaultdict(list)
    event_keys = set()
    for record in inventory.records:
        if record.group not in active or tau < record.start-1:
            continue
        interval = -1
        for i, threshold in enumerate(record.expanded):
            if tau >= threshold:
                interval = i
            else:
                break
        if interval < 0 or interval >= len(record.intervals):
            continue
        for face_index, polygon in enumerate(record.intervals[interval]):
            for fan_index in range(len(polygon)-2):
                raw = (polygon[0], polygon[fan_index+1], polygon[fan_index+2])
                output = tuple(effective[source] for source in raw)
                key = record.element, canonical_face(output)
                owner = (record.element, record.group, record.start,
                         record.sorted_index, interval, face_index, fan_index)
                groups[key].append((owner, raw, output))
                if record.identity in event_record_ids:
                    event_keys.add(key)
    incidence = Counter()
    for (element, face) in groups:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            incidence[(element, *sorted((face[a], face[b])))] += 1
    return groups, event_keys, incidence, sorted(active)


def unique_support(groups, event_keys, incidence, cycle, element):
    """Unique topological candidate; no float score or conditioning claim."""
    boundary = frozenset(cycle)
    eligible = sorted(key for key in event_keys if key[0] == element and
                      len(set(key[1])) == 3 and set(key[1]) <= boundary and
                      (element, canonical_face(tuple(reversed(key[1])))) not in groups)
    candidates = []
    for index, first in enumerate(eligible):
        for second in eligible[index+1:]:
            shared = set(first[1]) & set(second[1])
            if len(shared) != 2 or set(first[1]) | set(second[1]) != boundary:
                continue
            try:
                found_cycle = directed_boundary_cycle((first[1], second[1]))
            except RuntimeError:
                continue
            if not same_cycle(cycle, found_cycle):
                continue
            edges = [tuple(sorted((found_cycle[i], found_cycle[(i+1)%4]))) for i in range(4)]
            edges.append(tuple(sorted(shared)))
            if any(incidence[(element, *edge)] != 2 for edge in edges):
                continue
            candidates.append((first, second))
    require(len(candidates) == 1,
            f'Expected unique same-boundary combinatorial disk; found {len(candidates)}. Float best-conditioned selection remains UNKNOWN.')
    return candidates[0]


def branch_support(inventory, tau, event_ids, cycle, element):
    groups, event_keys, incidence, active = topology_at(inventory, tau, event_ids)
    selected = unique_support(groups, event_keys, incidence, cycle, element)
    formulae, branches, owners = {}, defaultdict(set), []
    visible = defaultdict(set)
    for key in selected:
        for owner, raw, output in groups[key]:
            owners.append(owner)
            for source, target in zip(raw, output):
                effective, coeff, view, branch = source_formula(source, tau, inventory.hypervertices)
                require(effective == target, 'Source state/formula identity mismatch.')
                if target in formulae:
                    require(formulae[target] == coeff, 'Replicated boundary owners disagree in exact affine coefficients.')
                else:
                    formulae[target] = coeff
                branches[target].add(branch)
                visible[target].add(view)
    require(set(formulae) == set(cycle), 'Support includes unexpected boundary labels.')
    require(len(set(owners)) == len(owners), 'A raw owner occurs more than once in selected patch.')
    return {'selected': selected, 'formulae': formulae, 'owners': sorted(owners),
            'active_groups': active, 'branches': branches, 'visible': visible}


def source_payload(support, cycle):
    return {'source_faces': [[source.text() for source in key[1]] for key in support['selected']],
            'owners': [list(owner) for owner in support['owners']],
            'active_groups': support['active_groups'],
            'boundary_cycle': [source.text() for source in cycle],
            'unique_combinatorial_support': True,
            'numerical_selector_conditioning': 'DEFERRED_TO_GEOMETRY; no alternative combinatorial candidate'}


def iter_ordinary_triangles(inventory, t0, t1, excluded_owners=()):
    """Yield exact retained-source triangle trajectories for a cell or singleton.

    This is the processed source-face superset, not a claim about runtime
    post-merge filtering. No giant exterior arrays are written to source JSON.
    Equal t0/t1 evaluates exact breakpoint geometry with identical endpoints.
    """
    t0, t1 = Fraction(t0), Fraction(t1)
    require(t0 <= t1, 'Expected an ordered cell or exact singleton.')
    midpoint = (t0+t1)/2
    if t0 < t1:
        thresholds = partition_points(inventory, t0, midpoint, t1)
        require(all(t in (t0, midpoint, t1) for t, _ in thresholds),
                'Exterior segment crosses an unpartitioned source threshold.')
        require(all(not (t == midpoint and any(reason != 'critical_root' for reason in why))
                    for t, why in thresholds), 'Exterior midpoint is a source threshold.')
    groups, _, _, _ = topology_at(inventory, midpoint, set())
    excluded = {tuple(owner) for owner in excluded_owners}
    for (element, face), values in sorted(groups.items()):
        owners = {tuple(value[0]) for value in values}
        if owners & excluded:
            require(owners <= excluded, 'Partial exclusion of replicated face owners.')
            continue
        formulae = {}
        for _, raw, output in values:
            for source, target in zip(raw, output):
                effective, coeff, _, _ = source_formula(source, midpoint, inventory.hypervertices)
                require(effective == target, 'Exterior source state mismatch.')
                require(target not in formulae or formulae[target] == coeff,
                        'Replicated exterior coordinates disagree exactly.')
                formulae[target] = coeff
        yield {'element': element, 'source_vertices': [source.text() for source in face],
               'owners': [list(owner) for owner in sorted(owners)],
               'positions_t0': [vj(evaluate(formulae[source], t0)) for source in face],
               'positions_t1': [vj(evaluate(formulae[source], t1)) for source in face],
               'coordinate_model': 'ideal rational interpolation of serialized HV binary32'}


def audit_event(event_root, cache_root):
    event_root, cache_root = Path(event_root).resolve(), Path(cache_root).resolve()
    report = {'schema': SCHEMA, 'status': 'UNKNOWN',
        'event_root': str(event_root), 'cache_root': str(cache_root),
        'coordinate_model': 'ideal rational interpolation of serialized HV binary32',
        'actual_runtime_binary32_certificate': 'UNKNOWN',
        'global_exterior_embedding_certificate': 'UNKNOWN',
        'continuous_window_admitted': False, 'segments': [], 'breakpoint_points': []}
    try:
        path = event_root/'critical_beb1_event_ir.json'
        ir = json.loads(path.read_text(encoding='utf-8'))
        require(ir.get('whole_mesh_splice_ready') is True, 'C0 exact-root input was not admitted.')
        plan = ir['whole_mesh_replacement_plan']
        levels = tuple(fr(ir['one_sided_window'][name]) for name in ('lower', 'critical', 'upper'))
        lower, root, upper = levels
        require(lower < root < upper, 'Invalid frozen window.')
        require(fr(plan['exact_time']) == root == fr(ir['event']['root']), 'Inconsistent frozen root.')
        cycle = tuple(vid(value) for value in plan['boundary_cycle'])
        require(len(cycle) == len(set(cycle)) == 4, 'Exactly four distinct source boundary VIDs required.')
        element = int(plan['element'])
        inventory = read_inventory(cache_root)
        require(0 <= lower < upper <= inventory.maximum, 'Window outside cache.')
        event_id, rows = selected_event_rows(cache_root, ir['event']['event_id'])
        event_ids = {(int(row['t_group']), int(row['t_start']), int(row['sorted_record_index'])) for row in rows}
        require(event_ids <= {record.identity for record in inventory.records}, 'Registry references missing raw processed records.')
        points = partition_points(inventory, lower, root, upper)
        report.update({'event_id': event_id, 'source_ir_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'cache_input_sha256': inventory.digest, 'cache_bytes': inventory.bytes_read,
            'cache_record_count': len(inventory.records), 'cache_source_edge_count': len(inventory.sources),
            'cache_scope': 'All processed primary/BPM2 records and all source HV files in this bounded cache, including replicas.',
            'associated_registry_record_count': len(event_ids),
            'levels': dict(zip(('lower', 'root', 'upper'), map(fj, levels))),
            'boundary_cycle': [source.text() for source in cycle],
            'partition_proof': {'complete_for': 'All time predicates in processed_mesh.py traversal and _compute_slice_vertex identity/affine branches.',
                'rule': 'No listed predicate threshold lies inside an open partition cell; active groups, owners, effective SourceVIDs, event marking, and source-face incidence are constant there. The representative labels this constant combinatorial branch; it is not used to infer constancy from samples.',
                'excluded': ['Floating best-conditioned score/area threshold: unique combinatorial candidate only; conditioning deferred.',
                             'Geometric embedding, exterior collision, C++ rounding, cache-building provenance, and unregistered source event completeness.'],
                'thresholds': [{'time': fj(tau), 'reasons': why} for tau, why in points]}})
        supports = {}
        for tau, why in points:
            support = branch_support(inventory, tau, event_ids, cycle, element)
            supports[tau] = support
            payload = {**source_payload(support, cycle), 'time': fj(tau), 'reasons': why,
                'boundary': [{'source_vid': source.text(),
                              'position': vj(evaluate(support['formulae'][source], tau))} for source in cycle]}
            report['breakpoint_points'].append(payload)
        for (t0, _), (t1, _) in zip(points, points[1:]):
            tau = (t0+t1)/2
            support = branch_support(inventory, tau, event_ids, cycle, element)
            boundaries = []
            for source in cycle:
                coeff = support['formulae'][source]
                p0, p1 = evaluate(coeff, t0), evaluate(coeff, t1)
                require(p0 == evaluate(supports[t0]['formulae'][source], t0) and
                        p1 == evaluate(supports[t1]['formulae'][source], t1),
                        'Exact boundary trajectory is discontinuous at a partition endpoint.')
                boundaries.append({'source_vid': source.text(), 'position_t0': vj(p0), 'position_t1': vj(p1),
                    'intercept': vj(coeff[0]), 'slope': vj(coeff[1]),
                    'source_branches': sorted(support['branches'][source]),
                    'in_view_values': sorted(support['visible'][source])})
            report['segments'].append({**source_payload(support, cycle), 't0': fj(t0), 't1': fj(t1),
                'slab': 'left' if t1 <= root else 'right', 'branch_representative': fj(tau),
                'source_branch_certificate': True, 'boundary': boundaries})
        root_support = supports[root]
        root_owners_equal = [list(value) for value in root_support['owners']] == sorted(plan['raw_suppressions'])
        require(root_owners_equal, 'Root source owners disagree with frozen C0 plan.')
        old_patch = ir['event_star_geometry']['critical_patch']
        old_boundary = {item['label']['id']: vf(item['position']) for item in old_patch['vertices']
                        if item['label']['kind'] == 'source_vid'}
        center = vf(plan['critical_position'])
        c0_centers = [vf(item['position']) for item in old_patch['vertices'] if item['label']['kind'] == 'critical']
        require(c0_centers == [center], 'Frozen C0 critical center inconsistency.')
        old_labels = [item['label']['id'] for item in old_patch['vertices']]
        center_label = next(item['label']['id'] for item in old_patch['vertices']
                            if item['label']['kind'] == 'critical')
        old_fan = sorted(canonical_face(tuple(old_labels[i] for i in face))
                         for face in old_patch['faces'])
        expected_fan = sorted(canonical_face((cycle[i].text(),
            cycle[(i+1)%4].text(), center_label)) for i in range(4))
        require(old_fan == expected_fan,
                'Frozen C0 fan connectivity/orientation differs from fixed source cycle.')
        root_errors = {source.text(): max(abs(a-b) for a, b in zip(
            evaluate(root_support['formulae'][source], root), old_boundary[source.text()])) for source in cycle}
        anchors = {'root': {'position': vj(center), 'kind': 'frozen_c0_binary32',
                            'binary32_words': plan['critical_position_contract']['canonical_binary32_hex']}}
        endpoint_models = {}
        for name, tau in (('lower', lower), ('upper', upper)):
            support = supports[tau]
            diagonal = sorted(set(support['selected'][0][1]) & set(support['selected'][1][1]))
            require(len(diagonal) == 2, 'Endpoint support has no unique shared diagonal.')
            positions = [evaluate(support['formulae'][source], tau) for source in diagonal]
            midpoint = tuple((a+b)/2 for a, b in zip(*positions))
            require(positions[0] != positions[1], 'Endpoint diagonal exactly collapses.')
            anchors[name] = {'position': vj(midpoint), 'kind': 'exact_source_diagonal_midpoint',
                'source_diagonal': [source.text() for source in diagonal]}
            stored = ir['ordinary_whole_mesh_patches'][name]['boundary_positions']
            differences = {source.text(): max(abs(a-b) for a, b in zip(
                evaluate(support['formulae'][source], tau), vf(stored[source.text()]))) for source in cycle}
            midpoint64 = vf([float(value) for value in midpoint])
            endpoint_models[name] = {'maximum_source_error_vs_saved_parser_binary64': fj(max(differences.values())),
                'source_error_by_vid': {k: fj(v) for k, v in differences.items()},
                'midpoint_exactly_representable_binary64': midpoint64 == midpoint,
                'note': 'Ideal exact interpolation and saved parser binary64 are distinct coordinate models; differences are not silently welded.'}
        half_stability = {}
        for side in ('left', 'right'):
            signatures = []
            for segment in report['segments']:
                if segment['slab'] == side:
                    signatures.append((segment['source_faces'], segment['owners']))
            for point in report['breakpoint_points']:
                tau = fr(point['time'])
                if (lower <= tau < root and side == 'left') or (root < tau <= upper and side == 'right'):
                    signatures.append((point['source_faces'], point['owners']))
            half_stability[side] = bool(signatures) and all(value == signatures[0] for value in signatures)
        report.update({'anchors': anchors,
            'root_contract': {'source_owners_equal_frozen_c0': root_owners_equal,
                'critical_center_equal_frozen_c0': True,
                'maximum_source_error_vs_saved_parser_binary64': fj(max(root_errors.values())),
                'source_error_by_vid': {k: fj(v) for k, v in root_errors.items()},
                'whole_mesh_arrays_equal_frozen_c0': 'NOT_TESTED'},
            'endpoint_coordinate_model_audits': endpoint_models,
            'piecewise_source_ownership_certificate': True,
            'half_window_owner_sets_constant': half_stability,
            'half_window_owner_scope': {'left': '[lower,root)', 'right': '(root,upper]',
                                        'root': 'separate frozen C0 owners'},
            'boundary_continuity_exact': True,
            'junction_boundary_cycle_and_orientation_equal': True,
            'junction_orientation_basis': 'Every singleton and open-cell support has the frozen oriented boundary cycle; exact positions agree at every junction.',
            'root_fan_connectivity_and_orientation_equal_frozen_c0': True,
            'status': 'PASS_SOURCE_BRANCH_CONTRACT' if all(half_stability.values())
                      else 'UNSUPPORTED_EXTRA_OWNER_BREAKPOINT',
            'window_event_isolation_certificate': 'UNKNOWN; event-domain scope requires separate analysis',
            'not_claimed': ['No floating area/clearance acceptance certificate.',
                'No actual runtime binary32 boundary/anchor output proof.',
                'No global/exterior intersection or continuous embedded-disk certificate.']})
        require(file_signatures(input_files(cache_root)) == inventory.signatures, 'Cache changed during audit.')
    except Exception as error:
        report['status'] = 'STOP_SOURCE_CONTRACT'
        report['stop_reason'] = str(error)
        report['piecewise_source_ownership_certificate'] = False
    require(len(json.dumps(report, indent=2, sort_keys=True).encode('utf-8')) < MAX_JSON_BYTES,
            'Audit exceeds 2 MiB; refusing oversized JSON.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-root', required=True, type=Path)
    parser.add_argument('--cache-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    for protected in (args.event_root.resolve(), args.cache_root.resolve()):
        require(output != protected and protected not in output.parents,
                'Output must be outside original event/cache trees.')
    require(not output.exists(), 'Output already exists.')
    report = audit_event(args.event_root, args.cache_root)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n'
    require(len(payload.encode('utf-8')) < MAX_JSON_BYTES, 'Indented report exceeds 2 MiB.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        handle.write(payload)
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'segments': len(report['segments'])}))
    return 0 if report['status'] == 'PASS_SOURCE_BRANCH_CONTRACT' else 2


if __name__ == '__main__':
    raise SystemExit(main())
