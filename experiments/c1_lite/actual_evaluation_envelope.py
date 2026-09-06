#!/usr/bin/env python3
"""Ordered raw-occurrence audit and analytic C++ interpolation error envelope.

This is not a production admission decision. Arithmetic assumptions and the
actual binary/source association must be bound by the caller. No time samples
are used: exact record predicates partition the complete original E2 window.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import struct

import window_source as s
from runtime_retained import build_retained_unit
from run_window_audit import units_for, center_at

U64, U32 = F(1, 2**53), F(1, 2**24)
TARGET_EPSILON = F(1, 1024)


def gamma(n, unit=U64):
    if not isinstance(n, int) or n < 0 or n*unit >= 1:
        raise ValueError('Invalid rounding-operation budget.')
    return n*unit/(1-n*unit)


def analytic_bounds(max_coordinate, max_slope, tau_upper, *, delta_t=None):
    """Absolute per-coordinate bound, under the explicitly exported profile.

    Exact rational entry: four actual arithmetic stages, conservatively gamma8
    in binary64 (also dominates long-double stages and integer conversions).
    Physical entry: reference time is the actual computed binary64 tau_hat,
    NOT an independently requested rational time or physical-real time.
    """
    m, slope, upper = map(F, (max_coordinate, max_slope, tau_upper))
    if not (0 <= m <= 2**40 and slope >= 0 and 1 <= upper <= 64):
        raise ValueError('Magnitude outside the bounded arithmetic profile.')
    if delta_t is not None and not F(1, 2**100) <= F(delta_t) <= 2**100:
        raise ValueError('deltaT outside the normal finite arithmetic profile.')
    time_error = upper*gamma(8)
    # Every product term follows <=4 roundings. gamma8 is a conservative
    # overestimate. Positive barycentric weights have sum one, so absolute
    # cancellation cannot defeat this M-scaled forward error bound.
    double_error = m*gamma(8)
    # Include an absolute half-subnormal ulp, including for a zero coordinate.
    cast_error = U32*(m+double_error) + F(1, 2**150)
    discrete_error = double_error+cast_error
    exact_error = discrete_error+slope*time_error
    return {'maximum_coordinate': s.fj(m), 'maximum_slope': s.fj(slope),
        'tau_upper': s.fj(upper), 'gamma8_binary64': s.fj(gamma(8)),
        'rational_entry_discrete_time_error': s.fj(time_error),
        'interpolation_binary64_error': s.fj(double_error),
        'final_binary32_cast_error': s.fj(cast_error),
        'physical_entry_actual_discrete_double_coordinate_error': s.fj(discrete_error),
        'exact_rational_entry_coordinate_error': s.fj(exact_error),
        'target_epsilon': s.fj(TARGET_EPSILON),
        'exact_error_below_target': exact_error < TARGET_EPSILON,
        'physical_error_below_target': discrete_error < TARGET_EPSILON,
        'delta_t': None if delta_t is None else s.fj(F(delta_t)),
        'delta_t_supplied_and_in_profile': delta_t is not None,
        'delta_t_runtime_binding_verified': False,
        'arithmetic_assumptions': [
            'T is IEEE binary64; spaceT is IEEE binary32; RNE ties-to-even; gradual underflow; no fast-math reassociation.',
            'GCC Linux long double has precision at least binary64; rational int64 numerator/denominator are positive and <=2^63-1.',
            'deltaT is a fixed binary64 in [2^-100,2^100]; no overflow/underflow in the time conversion.',
            'All emitted HV coordinates are binary32, abs <=2^40; nonzero magnitudes >=2^-64; effective time endpoints are int8 integers.',
            'E2 tau lies in [102/5,106/5]; every emitted source pair has ordered times and a positive effective span, or identical endpoint coordinates.',
            'A positive span is an exact integer >=1. Binary64 interpolation intermediates are normal or zero under these exponent bounds.',
        ],
        'models': {
            'exact_rational': 'Compare actual geometry to the ideal source at the requested exact tau; rational selectors are separate from rounded interpolation time.',
            'physical_actual_discrete_double': 'Compare to ideal at Fraction(actual binary64 physical/deltaT); this does not prove the physical cache-group or owner selector matches the rational-entry ledger.',
        }}


def ordered_vid(blob):
    if len(blob) != 16:
        raise ValueError('Malformed original VID.')
    return s.SourceVID(s.HVID(struct.unpack_from('<i', blob, 0)[0], struct.unpack_from('<b', blob, 4)[0]),
                       s.HVID(struct.unpack_from('<i', blob, 8)[0], struct.unpack_from('<b', blob, 12)[0]))


def read_ordered_records(cache, inventory):
    """Reparse original bytes; never infer original endpoint order from quotient."""
    records = []
    canonical = {record.identity: record for record in inventory.records}
    for path in sorted((Path(cache)/'processed_hyperpolys').glob('*.bin')):
        if not re.fullmatch(r'\d+_\d+\.bin', path.name):
            continue
        group, start = map(int, path.stem.split('_'))
        payload = path.read_bytes()
        metadata = s._parse_processed_metadata(path.with_name(path.stem+'_hpmeta.bin').read_bytes(), group, start)
        offset = index = 0
        while offset < len(payload):
            element = struct.unpack_from('<b', payload, offset)[0]
            offset += 1
            blobs, offset = s._read_counted_vector(payload, offset, 1)
            times = tuple(struct.unpack_from('<b', blob)[0] for blob in blobs)
            intervals = []
            for _ in range(len(times)-1):
                faces = []
                while True:
                    blobs, offset = s._read_counted_vector(payload, offset, 16)
                    if not blobs:
                        break
                    faces.append(tuple(ordered_vid(blob) for blob in blobs))
                intervals.append(tuple(faces))
            record = s.Record(group, start, metadata[index], element, times, tuple(intervals))
            reference = canonical[record.identity]
            mapped = tuple(tuple(tuple(s.SourceVID.canonical(v.first, v.second) for v in face)
                                 for face in interval) for interval in intervals)
            if (record.element, record.times, mapped) != (reference.element, reference.times, reference.intervals):
                raise ValueError('Ordered parser disagrees with canonical source inventory.')
            records.append(record)
            index += 1
        if offset != len(payload) or index != len(metadata):
            raise ValueError('Ordered primary/metadata length mismatch.')
    if len(records) != len(inventory.records):
        raise ValueError('Incomplete ordered record coverage.')
    return tuple(records)


def selected_interval(record, tau, active):
    if record.group not in active or tau < record.start-1:
        return None
    selected = -1
    for i, threshold in enumerate(record.expanded):
        if tau >= threshold:
            selected = i
        else:
            break
    if not 0 <= selected < len(record.intervals):
        return None
    if selected == 0 and tau < record.times[0]:
        return None
    if selected == len(record.intervals)-1 and tau > record.times[-1]:
        return None
    return selected


def inspect_pair(vid, hypervertices):
    a, b = hypervertices[vid.first], hypervertices[vid.second]
    _, _, lo, hi = s.source_thresholds(vid, hypervertices)
    pa, pb = s.vf(a.position), s.vf(b.position)
    if a.time > b.time:
        return {'status': 'UNKNOWN_REVERSED_ORIGINAL_ENDPOINTS'}
    if hi < lo:
        return {'status': 'UNKNOWN_REVERSED_EFFECTIVE_SPAN'}
    if hi == lo and pa != pb:
        return {'status': 'UNKNOWN_DISCONTINUOUS_ZERO_EFFECTIVE_SPAN'}
    values = pa+pb
    if any(x and abs(x) < F(1, 2**64) for x in values) or any(abs(x) > 2**40 for x in values):
        return {'status': 'UNKNOWN_COORDINATE_EXPONENT_PROFILE'}
    slope = F(0) if hi == lo else max(abs(y-x)/(hi-lo) for x, y in zip(pa, pb))
    return {'status': 'PASS_CONTINUOUS_CLAMPED_SOURCE', 'lower': lo, 'upper': hi,
            'maximum_coordinate': max(map(abs, values)), 'maximum_slope': slope,
            'canonical_original_order': vid == s.SourceVID.canonical(vid.first, vid.second),
            'zero_span_constant': lo == hi}


def audit(source, inventory, records, *, delta_t=None):
    if source['cache_input_sha256'] != inventory.digest:
        raise ValueError('Source/cache digest mismatch.')
    if [s.fr(source['levels'][k]) for k in ('lower', 'root', 'upper')] != [F(102, 5), F(104, 5), F(106, 5)]:
        raise ValueError('Only the original unchanged E2 window is in this audit scope.')
    specs = list(units_for(source))
    if Counter(v['kind'] for v in specs) != Counter({'affine_branch': 3, 'actual_singleton': 4}):
        raise ValueError('Missing source branch or singleton.')
    boundary = set(source['boundary_cycle'])
    element_set = {owner[0] for spec in specs for owner in spec['owners']}
    if len(element_set) != 1:
        raise ValueError('Ambiguous source element.')
    element = next(iter(element_set))
    pair_reports, diagnostics, unit_rows = {}, [], []
    stream = hashlib.sha256()
    all_ordered_sources = set()
    canonical_to_ordered = {}
    m = slope = F(0)
    for spec in specs:
        lo, hi = s.fr(spec['t0']), s.fr(spec['t1'])
        tau = (lo+hi)/2
        raw = build_retained_unit(inventory, lo, hi, spec['owners'])
        expected_owners = {tuple(row['owner']) for row in raw.ledger if row['state'] != 'skip'}
        active = set(s._active_groups(tau, inventory.groups, inventory.maximum))
        found, counts, aliases = set(), Counter(), {}
        boundaries_found = set()
        for record in records:
            ii = selected_interval(record, tau, active)
            if ii is None:
                continue
            for fi, polygon in enumerate(record.intervals[ii]):
                counts['emitted_polygons'] += 1
                counts['no_fan_polygons'] += len(polygon) < 3
                for fan in range(max(0, len(polygon)-2)):
                    found.add((record.element, record.group, record.start, record.sorted_index, ii, fi, fan))
                for vertex_index, vid in enumerate(polygon):
                    counts['all_emitted_polygon_vertex_occurrences'] += 1
                    all_ordered_sources.add(vid)
                    canonical_vid = s.SourceVID.canonical(vid.first, vid.second)
                    if canonical_vid in canonical_to_ordered and canonical_to_ordered[canonical_vid] != vid:
                        raise ValueError('Canonical source has multiple original endpoint orders.')
                    canonical_to_ordered[canonical_vid] = vid
                    if vid not in pair_reports:
                        pair_reports[vid] = inspect_pair(vid, inventory.hypervertices)
                    checked = pair_reports[vid]
                    if checked['status'] != 'PASS_CONTINUOUS_CLAMPED_SOURCE':
                        if len(diagnostics) < 12:
                            diagnostics.append({'kind': checked['status'], 'original_source_vid': vid.text()})
                        continue
                    m = max(m, checked['maximum_coordinate'])
                    slope = max(slope, checked['maximum_slope'])
                    counts['noncanonical_original_order'] += not checked['canonical_original_order']
                    counts['zero_span_constant_occurrences'] += checked['zero_span_constant']
                    effective, coeff, _, branch = s.source_formula(vid, tau, inventory.hypervertices)
                    key = record.element, effective
                    if key in aliases and aliases[key] != coeff:
                        raise ValueError('Every-original-occurrence ideal replica trajectory mismatch.')
                    aliases[key] = coeff
                    if record.element == element and effective.text() in boundary:
                        boundaries_found.add(effective.text())
                        if canonical_vid.text() != effective.text():
                            diagnostics.append({'kind': 'INTERFACE_ORIGINAL_ID_MISMATCH', 'source': vid.text()})
                    stream.update(json.dumps([record.element, record.group, record.start,
                        record.sorted_index, ii, fi, vertex_index, vid.text(), effective.text(),
                        [s.vj(c) for c in coeff]], separators=(',', ':')).encode()+b'\n')
        if found != expected_owners:
            raise ValueError('Ordered actual fan owner set differs from complete raw emission ledger.')
        if boundaries_found != boundary:
            raise ValueError('Interface occurrence missing from an actual singleton or branch.')
        unit_rows.append({'kind': spec['kind'], 'index': spec['index'], 't0': s.fj(lo), 't1': s.fj(hi),
            'counts': dict(counts), 'emitted_raw_fan_owners': len(found),
            'exact_raw_owner_ledger_set_match': True, 'all_original_replicas_share_ideal_trajectory': True,
            'complete_exact_partition_checked': True,
            'partition_proof': 'build_retained_unit requires no internal source/global-group/expanded-or-original record-time threshold; source partition includes all serialized polygon vertices, including no-fan polygons. The midpoint labels this proven constant open cell, not an empirical sample.',
            'interface_source_ids_present': sorted(boundaries_found),
            'merged_ideal_id_count_including_no_fan_polygons': len(aliases)})
    bounds = analytic_bounds(m, slope, F(106, 5), delta_t=delta_t)
    time_error = s.fr(bounds['rational_entry_discrete_time_error'])
    interfaces = []
    for text in source['boundary_cycle']:
        vid = canonical_to_ordered[s.vid(text)]
        item = inspect_pair(vid, inventory.hypervertices)
        clearance = min(F(102, 5)-item['lower'], item['upper']-F(106, 5))
        interfaces.append({'source_vid': text, 'original_ordered_source_vid': vid.text(),
            'effective_lower': item['lower'], 'effective_upper': item['upper'],
            'minimum_clamp_clearance': s.fj(clearance),
            'stable_actual_original_vid_entire_window': clearance > time_error})
    # Replacement center is a separate exact-anchor evaluation contract. This
    # only supplies its final RNE32 bound, not an implementation audit.
    center_m = max(abs(s.fr(x)) for spec in specs for tau in (s.fr(spec['t0']), s.fr(spec['t1']))
                   for x in center_at(source, tau))
    center_error = U32*center_m+F(1, 2**150)
    success = (not diagnostics and bounds['exact_error_below_target'] and center_error < TARGET_EPSILON
        and all(i['stable_actual_original_vid_entire_window'] for i in interfaces))
    return {'schema': 'c1-lite-actual-evaluation-envelope-v1',
        'status': 'PASS_ANALYTIC_ENVELOPE_UNDER_DECLARED_ARITHMETIC' if success else 'UNKNOWN',
        'event_key': 'event-02-88ade47aa4fa', 'runtime_admitted': False,
        'sampling_used_as_proof': False, 'production_modified': False,
        'source_cache_sha256': inventory.digest, 'arithmetic_binary_binding_verified': False,
        'exact_rational_source_occurrence_model_complete': True,
        'physical_entry_owner_equivalence_proven': False,
        'actual_source_suppressor_set_equivalence_proven': False,
        'full_window_rollback_proven': False,
        'unique_original_ordered_source_pairs': len(all_ordered_sources),
        'canonical_to_original_order_relation_unique': True,
        'all_original_occurrence_certificate_stream_sha256': stream.hexdigest(),
        'units': unit_rows, 'source_envelope': bounds, 'interface_identity': interfaces,
        'center_exact_anchor_then_RNE32_error': s.fj(center_error),
        'center_error_below_target': center_error < TARGET_EPSILON,
        'center_runtime_implementation_verified': False, 'diagnostics': diagnostics,
        'merge_argument': [
            'Every emitted polygon vertex is covered, including polygons that emit no fan faces and vertices of suppressed faces.',
            'Noncollapsed actual pairs are the original ordered VID; identical VID evaluations use the same HV endpoints and the same actual double time.',
            'At a collapsed endpoint, binary32 HV times an int8 integer span fits exactly in binary64; dividing by that exact integer rounds to the original representable HV coordinate. All representatives of the collapsed ID coincide.',
            'A remote ID may clamp slightly before an exact threshold. Its continuous clamped coordinate trajectory is Lipschitz, so the time-error envelope applies on both sides; exact effective-ID equality is not claimed for remote entities.',
            'Four interface IDs remain noncollapsed by the strict all-window clearance; no collapsed endpoint ID can alias one of these noncollapsed two-endpoint IDs.',
        ],
        'remaining_external_obligations': [
            'Bind runtime compiled library, IEEE mode/flags, HV cache digest and actual deltaT to this declared arithmetic profile.',
            'Bind exact rational raw selector and source suppressor to the owner ledger, including singleton boundary policy.',
            'Verify the replacement center follows the original exact anchors then one RNE32 conversion; preserve all shared actual final-array coordinates.',
            'Compose with conditional perturbation, endpoint contract, topology and transactional whole-window admission; this report alone never admits runtime output.',
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delta-t-hex')
    args = parser.parse_args()
    source_path, cache, output = args.source_report.resolve(), args.cache_root.resolve(), args.output.resolve()
    if output.exists() or cache in output.parents or source_path.parent in output.parents:
        raise ValueError('Fresh output outside frozen inputs required.')
    original = source_path.read_bytes()
    report = json.loads(original)
    inventory = s.read_inventory(cache)
    records = read_ordered_records(cache, inventory)
    delta = F.from_float(float.fromhex(args.delta_t_hex)) if args.delta_t_hex else None
    result = audit(report, inventory, records, delta_t=delta)
    if original != source_path.read_bytes() or s.file_signatures(s.input_files(cache)) != inventory.signatures:
        raise ValueError('Frozen inputs changed.')
    result.update(original_inputs_unchanged=True, source_report_sha256=hashlib.sha256(original).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    data = (json.dumps(result, indent=2, sort_keys=True)+'\n').encode()
    if len(data) > 128*1024:
        raise ValueError('Output size budget exceeded.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(data)
    print(json.dumps({'status': result['status'], 'bytes': len(data),
        'source_coordinate_error_upper': float(s.fr(result['source_envelope']['exact_rational_entry_coordinate_error'])),
        'units': len(result['units']), 'diagnostics': result['diagnostics']}))


if __name__ == '__main__':
    main()
