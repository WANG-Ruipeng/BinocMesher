#!/usr/bin/env python3
"""Fixed E2 all-pair conditional perturbation audit, not a runtime emitter."""
from collections import Counter
from fractions import Fraction as F
import argparse
import hashlib
import json
from pathlib import Path
import time

from conditional_perturbation import certify_local_perturbation, certify_pair_perturbation
from runtime_retained import build_retained_unit, compact_manifest
from window_source import read_inventory, file_signatures, input_files, fr, fj
from run_window_audit import units_for, center_at, qualified_vid, check_junctions

EPSILONS = (F(1, 1024), F(1, 1 << 20))
OUTPUT_LIMIT = 1024 * 1024


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def status(values):
    values = list(values)
    return 'REJECT' if 'REJECT' in values else 'UNKNOWN' if any(v != 'PASS_CONDITIONAL' for v in values) else 'PASS_CONDITIONAL'


def audit(source, inventory, deadline, *, max_diagnostics=12):
    if source.get('status') != 'PASS_SOURCE_BRANCH_CONTRACT' or source['cache_input_sha256'] != inventory.digest:
        raise ValueError('Frozen source/cache contract mismatch.')
    if [fr(source['levels'][k]) for k in ('lower', 'root', 'upper')] != [F(102, 5), F(104, 5), F(106, 5)]:
        raise ValueError('Only the unchanged original E2 window is supported.')
    specs = list(units_for(source))
    if Counter(s['kind'] for s in specs) != Counter({'affine_branch': 3, 'actual_singleton': 4}):
        raise ValueError('Expected all three original branches and all four actual singleton states.')
    elements = {owner[0] for s in specs for owner in s['owners']}
    if len(elements) != 1:
        raise ValueError('Ambiguous source element.')
    element = next(iter(elements))
    boundary_ids = [qualified_vid(element, vid) for vid in source['boundary_cycle']]
    report = {'schema': 'c1-lite-e2-conditional-perturbation-audit-v1',
        'event_key': 'event-02-88ade47aa4fa', 'runtime_admitted': False,
        'production_modified': False, 'render_started': False,
        'coordinate_model': source['coordinate_model'],
        'source_cache_sha256': inventory.digest, 'ideal_junctions': check_junctions(source),
        'epsilon_schedule': [fj(e) for e in EPSILONS],
        'epsilon_schedule_policy': 'Both fixed envelopes are tested; neither the window nor anchors are changed. A smaller envelope is a stronger unproven runtime requirement, not a measured error bound.',
        'actual_error_bound_proven': False, 'actual_identity_mapping_proven': False,
        'actual_time_selector_proven': False,
        'coverage_contract': [
            'Model includes every retained raw owner; geometric-quotient replicas retain all owner references.',
            'For runtime use, each actual emitted raw occurrence must correspond to a certified model occurrence, and every chosen merged coordinate must satisfy that occurrence coordinate envelope.',
            'Unshared remote effective IDs need not be identical to ideal IDs, but no actual face may escape the covered occurrence/envelope family.',
            'Shared interface entities must reuse one common actual coordinate, and actual identity/incidence, exact-once suppression, endpoints and unchanged exterior need independent proof.',
            'No claim that rational-to-double timing preserves every remote clamp branch; the actual all-time envelope and occurrence coverage must handle it.',
        ],
        'units': [], 'diagnostics': [], 'diagnostics_total': 0,
        'margin_notice': 'Stored margins are unnormalized functional values within each certificate kind, not Euclidean distances or comparable quality scores.'}

    def diagnostic(unit, epsilon, triangle, certificate):
        report['diagnostics_total'] += 1
        if len(report['diagnostics']) < max_diagnostics:
            report['diagnostics'].append({'unit': unit, 'epsilon': fj(epsilon),
                'triangle': triangle, 'certificate': certificate})

    for spec in specs:
        if time.monotonic() >= deadline:
            raise TimeoutError('Conditional audit budget exhausted before a source unit.')
        lo, hi = fr(spec['t0']), fr(spec['t1'])
        raw = build_retained_unit(inventory, lo, hi, spec['owners'])
        owners = [tuple(owner) for triangle in raw.triangles for owner in triangle['owners']]
        expected = {tuple(row['owner']) for row in raw.ledger if row['state'] == 'retained'}
        coverage = len(owners) == len(set(owners)) and set(owners) == expected
        if not coverage:
            raise ValueError('Model retained raw-owner coverage is incomplete or duplicated.')
        c0, c1 = center_at(source, lo), center_at(source, hi)
        name = f"{spec['kind']}:{spec['index']}"
        unit = {key: spec[key] for key in ('kind', 'index', 't0', 't1')}
        unit['raw_manifest'] = compact_manifest(raw, boundary_cycle=source['boundary_cycle'], element=element)
        unit['complete_model_raw_owner_coverage'] = coverage
        unit['envelopes'] = []
        report['units'].append(unit)
        for epsilon in EPSILONS:
            local = certify_local_perturbation(spec['boundary_start'], spec['boundary_end'], c0, c1, epsilon)
            counts, kinds, minima = Counter(), Counter(), {}
            stream = hashlib.sha256()
            row = {'epsilon': fj(epsilon), 'local': local, 'scan_complete': False}
            unit['envelopes'].append(row)
            for triangle in raw.triangles:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Conditional audit budget exhausted during a source unit.')
                ids = [qualified_vid(triangle['element'], vid) for vid in triangle['source_vertices']]
                certificate = certify_pair_perturbation(
                    spec['boundary_start'], spec['boundary_end'], boundary_ids, c0, c1,
                    triangle['positions_t0'], triangle['positions_t1'], ids, epsilon)
                counts[certificate['status']] += 1
                proof = certificate.get('proof') or {}
                kind = proof.get('kind', certificate['status'])
                kinds[kind] += 1
                for item in (proof, certificate.get('regularity') or {}):
                    if 'remaining_margin' in item:
                        label = item['kind']
                        value = fr(item['remaining_margin'])
                        minima[label] = min(minima.get(label, value), value)
                stream.update(stable({'triangle': triangle, 'certificate': certificate})+b'\n')
                if certificate['status'] != 'PASS_CONDITIONAL':
                    diagnostic(name, epsilon, triangle, certificate)
            row.update(status=status([local['status'], *counts]), scan_complete=True,
                pair_counts=dict(counts), certificate_kinds=dict(kinds),
                minimum_remaining_margin_by_kind={k: fj(v) for k, v in sorted(minima.items())},
                complete_certificate_stream_sha256=stream.hexdigest())
            print(json.dumps({'unit': name, 'epsilon': str(epsilon), 'local': local['status'],
                              'pair_counts': dict(counts), 'status': row['status']}), flush=True)
    report['envelope_summaries'] = []
    for index, epsilon in enumerate(EPSILONS):
        rows = [unit['envelopes'][index] for unit in report['units']]
        report['envelope_summaries'].append({'epsilon': fj(epsilon), 'status': status(row['status'] for row in rows),
            'all_units_scanned': all(row['scan_complete'] for row in rows),
            'pair_counts': dict(sum((Counter(row['pair_counts']) for row in rows), Counter())),
            'certificate_kinds': dict(sum((Counter(row['certificate_kinds']) for row in rows), Counter())),
            'local_status_counts': dict(Counter(row['local']['status'] for row in rows)),
            'runtime_admitted': False})
    report['status'] = 'COMPLETE_CONDITIONAL_AUDIT'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-report', 'cache-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args()
    source_path, cache, output = args.source_report.resolve(), args.cache_root.resolve(), args.output.resolve()
    if (output.exists() or cache == output or cache in output.parents
            or source_path.parent == output or source_path.parent in output.parents or output.suffix != '.json'):
        raise ValueError('Expected a fresh JSON output outside frozen inputs.')
    if not 0 < args.max_seconds <= 900:
        raise ValueError('Expected a bounded wall time <=900 seconds.')
    original = source_path.read_bytes()
    source = json.loads(original)
    inventory = read_inventory(cache)
    started = time.monotonic()
    result = audit(source, inventory, started+args.max_seconds)
    if source_path.read_bytes() != original or file_signatures(input_files(cache)) != inventory.signatures:
        raise ValueError('Frozen inputs changed; refusing publication.')
    result.update(source_report_sha256=hashlib.sha256(original).hexdigest(),
                  original_inputs_unchanged=True, elapsed_seconds=time.monotonic()-started)
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ('conditional_perturbation.py', 'audit_relative_plane_feasibility.py',
         'window_contact_v2.py', 'window_contact_v3.py', 'window_geometry.py',
         'window_exterior.py', 'window_source.py', 'runtime_retained.py', 'run_window_audit.py')]]
    result['script_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if len(payload) > OUTPUT_LIMIT:
        raise ValueError('Conditional report exceeds one MiB output budget.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(payload)
    print(json.dumps({'output': str(output), 'bytes': len(payload), 'envelopes': result['envelope_summaries'],
                      'elapsed_seconds': result['elapsed_seconds'], 'runtime_admitted': False}), flush=True)


if __name__ == '__main__':
    main()
