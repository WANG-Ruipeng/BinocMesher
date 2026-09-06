#!/usr/bin/env python3
"""Fixed-demo contact-v3 follow-up audit, never a production splice.

This versioned entry retains the v2 engine while changing only the exact contact
kernel. No frozen v2 script/result is overwritten; geometry/inputs are identical.

Old source data are immutable inputs. New filtered candidate counts, source
interface topology and exact contact proofs have separate scopes; binary32
runtime admission cannot be inferred from any of them.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from run_window_audit import (EVENT_KEYS, aggregate_status, center_at,
    check_junctions, compact_certificate, digest_value, qualified_vid,
    stable_bytes, units_for, vectors, certify_degenerate_superset_axis)
from window_source import read_inventory, file_signatures, input_files, fr
from window_geometry import certify_segment
from window_contact_v3 import certify_contact
from runtime_retained import build_retained_unit, compact_manifest, _identically_degenerate
from interface_topology import audit_interface, original_disk
from window_admission import decide_admission, select_action

OUTPUT_BUDGET = 12 * 1024 * 1024


def gate(status, reason, **detail):
    return {'status': status, 'reason': reason, **detail}


def audit(source, provider, *, endpoint_evidence=None, deadline=None, max_diagnostics=12, progress=None):
    report = {'schema': 'c1-lite-repaired-raw-window-audit-v3',
        'event_id': source.get('event_id'), 'status': 'UNKNOWN', 'units': [],
        'diagnostics': [], 'diagnostics_total': 0, 'runtime_started': False,
        'runtime_plan_emitted': False, 'render_started': False,
        'coordinate_model': 'ideal rational interpolation of serialized HV binary32',
        'exterior_scope': 'Raw original-time emission filtered canonical SourceVID geometric quotient, NOT actual C++ array identity.',
        'source_report_sha256': digest_value(source)}
    evidence = {}

    def diagnostic(unit, kind, detail):
        report['diagnostics_total'] += 1
        if len(report['diagnostics']) < max_diagnostics:
            report['diagnostics'].append({'unit': unit, 'kind': kind, 'detail': detail})

    try:
        if source.get('status') != 'PASS_SOURCE_BRANCH_CONTRACT':
            raise ValueError('Frozen source contract is not PASS.')
        if source.get('coordinate_model') != report['coordinate_model']:
            raise ValueError('Unsupported source coordinate model.')
        if source.get('piecewise_source_ownership_certificate') is not True or source.get('half_window_owner_sets_constant') != {'left': True, 'right': True}:
            raise ValueError('Missing whole-branch ownership certificate.')
        report['junctions'] = check_junctions(source)
        specs = list(units_for(source))
        source_units = [*source['segments'], *source['breakpoint_points']]
        elements = {owner[0] for spec in specs for owner in spec['owners']}
        if len(elements) != 1:
            raise ValueError('Source element is ambiguous.')
        element = next(iter(elements))
        cycle = source['boundary_cycle']
        boundary_ids = [qualified_vid(element, vid) for vid in cycle]
        for spec, source_unit in zip(specs, source_units):
            name = f"{spec['kind']}:{spec['index']}"
            t0, t1 = fr(spec['t0']), fr(spec['t1'])
            row = {key: spec[key] for key in ('kind', 'index', 't0', 't1')}
            row.update(status='UNKNOWN', scan_complete=False, retained_counts={'PASS': 0, 'UNKNOWN': 0, 'REJECT': 0},
                       retained_certificate_kinds={}, original_disk_status='UNKNOWN', interface_topology_status='UNKNOWN',
                       interface_degeneracy_status='UNKNOWN', geometry_status='UNKNOWN', exterior_status='UNKNOWN')
            report['units'].append(row)
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Audit wall-time budget exhausted before this cell.')
                raw = provider(t0, t1, spec['owners'])
                row['raw_manifest'] = compact_manifest(raw, boundary_cycle=cycle, element=element)
                row['interface_degeneracy_status'] = 'REJECT' if row['raw_manifest']['interface_degenerate_source_faces'] else 'UNKNOWN'
                source_faces = source_unit['source_faces']
                row['original_disk_status'] = original_disk(cycle, source_faces)['status']
                row['interface_topology'] = audit_interface(cycle, source_faces, raw.triangles, element=element)
                row['interface_topology_status'] = row['interface_topology']['status']
                c0, c1 = center_at(source, t0), center_at(source, t1)
                geometry = certify_segment(spec['boundary_start'], spec['boundary_end'], c0, c1)
                row['geometry_status'] = geometry['status']
                row['geometry_certificate'] = compact_certificate(geometry)
                if geometry['status'] != 'PASS':
                    diagnostic(name, 'LOCAL_GRAPH_NOT_CERTIFIED', compact_certificate(geometry))
                    continue
                stream = hashlib.sha256()
                for triangle in raw.triangles:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError('Audit wall-time budget exhausted during this cell.')
                    ids = [qualified_vid(triangle['element'], vid) for vid in triangle['source_vertices']]
                    degenerate = _identically_degenerate(triangle)
                    if degenerate and set(ids) & set(boundary_ids):
                        certificate = {'schema': 'c1-lite-interface-degeneracy-policy-v1', 'status': 'REJECT',
                            'reason': 'Pre-existing retained zero-area support touches the interface; the fixed conservative policy refuses this window. This is NOT a new collision.',
                            'classification': 'BASELINE_INTERFACE_DEGENERACY_POLICY',
                            'witness': {'shared_source_vids': sorted(set(ids) & set(boundary_ids)),
                                        'kind': 'inherited_interface_degeneracy'}}
                    elif len(set(ids)) < 3:
                        certificate = certify_degenerate_superset_axis(
                            spec['boundary_start'], spec['boundary_end'], c0, c1, boundary_ids, triangle, ids)
                    else:
                        certificate = certify_contact(spec['boundary_start'], spec['boundary_end'], boundary_ids,
                            triangle['positions_t0'], triangle['positions_t1'], ids,
                            center_start=c0, center_end=c1, orientation_xy=geometry['orientation_xy'])
                    status = certificate['status']
                    row['retained_counts'][status] += 1
                    kind = certificate.get('classification') or (certificate.get('separation_certificate') or {}).get('kind', status)
                    row['retained_certificate_kinds'][kind] = row['retained_certificate_kinds'].get(kind, 0)+1
                    stream.update(stable_bytes({'triangle': triangle, 'certificate': certificate})+b'\n')
                    if status != 'PASS':
                        diagnostic(name, kind, {'source_vertices': triangle['source_vertices'],
                            'owners': triangle['owners'], 'certificate': compact_certificate(certificate)})
                row['scan_complete'] = True
                row['retained_certificate_stream_sha256'] = stream.hexdigest()
                row['exterior_status'] = aggregate_status(
                    status for status, count in row['retained_counts'].items() if count)
                # A complete empty retained set has no pairwise contacts, but
                # cannot pass the independent interface attachment gate.
                if not raw.triangles:
                    row['exterior_status'] = 'PASS'
                # A contact PASS required nondegeneracy for every shared face.
                # If an unresolved shared face remains, do not infer its safety.
                if row['interface_degeneracy_status'] != 'REJECT' and row['exterior_status'] == 'PASS':
                    row['interface_degeneracy_status'] = 'PASS'
                row['status'] = aggregate_status(row[key] for key in
                    ('original_disk_status', 'geometry_status', 'exterior_status', 'interface_topology_status', 'interface_degeneracy_status'))
            except Exception as error:
                row['error'] = type(error).__name__+': '+str(error)
                diagnostic(name, 'CELL_INCOMPLETE', row['error'])
            if progress:
                progress({'unit': name, 'status': row['status'], 'counts': row['retained_counts'], 'scan_complete': row['scan_complete']})
        rows = report['units']
        complete = len(rows) == len(specs) and all(row['scan_complete'] for row in rows)
        report['all_units_scanned'] = complete
        report['pair_counts'] = dict(sum((Counter(row['retained_counts']) for row in rows), Counter()))
        report['certificate_kinds'] = dict(sum((Counter(row['retained_certificate_kinds']) for row in rows), Counter()))
        report['status'] = aggregate_status(row['status'] for row in rows)
        for name, field in (('original_patch_disk', 'original_disk_status'), ('replacement_disk', 'geometry_status'),
                            ('source_incidence_contact', 'exterior_status'), ('interface_topology', 'interface_topology_status'),
                            ('interface_degeneracy_policy', 'interface_degeneracy_status')):
            evidence[name] = gate(aggregate_status(row[field] for row in rows),
                'Whole fixed partition; source-identity quotient / ideal rational scope only. Runtime equivalence is a separate gate.')
        evidence['source_ownership'] = gate('PASS' if complete else 'UNKNOWN',
            'Frozen source-owner contract plus every raw owner-emission/suppression model completed; not runtime array identity.')
        evidence['complete_temporal_partition'] = gate('PASS' if complete else 'UNKNOWN',
            'Every original source branch and actual singleton attempted, with raw-original-time split guards.')
        evidence['ideal_temporal_continuity'] = gate('PASS',
            'One exact piecewise-affine center and identical source boundaries at all junctions; no float32 continuity claim.')
    except Exception as error:
        report['status'] = 'REJECT'
        report['error'] = type(error).__name__+': '+str(error)
        evidence['source_ownership'] = gate('REJECT', report['error'])
    if endpoint_evidence is not None:
        numerical = endpoint_evidence.get('prospective_binary32_endpoint_contract_pass')
        evidence['endpoint_realization'] = gate('REJECT' if numerical is False else 'UNKNOWN',
            'Frozen B prospective endpoint audit: a failure is retained; a prospective PASS is not an actual C++ endpoint-equivalence proof.',
            prospective_binary32_pass=numerical,
            binary64_pass=endpoint_evidence.get('binary64_endpoint_contract_pass'))
    report['admission'] = decide_admission(evidence)
    report['runtime_admitted'] = report['admission']['runtime_plan_authorized']
    report['production_started'] = False
    levels = source.get('levels', {})
    if set(('lower', 'root', 'upper')) <= set(levels):
        lower, root, upper = (fr(levels[key]) for key in ('lower', 'root', 'upper'))
        report['fallback_selection_checks'] = {key: select_action(t, lower, root, upper, report['admission'])
            for key, t in (('lower', lower), ('left', (lower+root)/2), ('root', root), ('right', (root+upper)/2), ('upper', upper))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'cache-root', 'ab-result', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=900.)
    args = parser.parse_args()
    source_root, cache, output = args.source_root.resolve(), args.cache_root.resolve(), args.output.resolve()
    if output.exists() or any(p == output or p in output.parents for p in (source_root, cache)):
        raise ValueError('Output must be fresh and outside frozen inputs.')
    if not 0 < args.max_seconds <= 3600:
        raise ValueError('Expected a bounded positive wall time <=3600 seconds.')
    old_b = json.loads(args.ab_result.read_text())
    if tuple(row['key'] for row in old_b['events']) != EVENT_KEYS:
        raise ValueError('B evidence does not match the fixed four-event order.')
    numerical = {row['key']: row for row in old_b['events']}
    inventory = read_inventory(cache)
    start = time.monotonic()
    results = {}
    for key in EVENT_KEYS:
        source = json.loads((source_root/key/'source.json').read_text())
        if source['cache_input_sha256'] != inventory.digest:
            raise ValueError('Frozen source and raw inventory cache hashes differ.')
        result = audit(source, lambda t0, t1, owners: build_retained_unit(inventory, t0, t1, owners),
            endpoint_evidence=numerical[key], deadline=start+args.max_seconds,
            progress=lambda item: print(json.dumps({'event': key, **item}), flush=True))
        old = json.loads((source_root/key/'audit.json').read_text())
        result['historical_v1_superset_pair_counts'] = dict(sum((Counter(row['retained_counts']) for row in old['units']), Counter()))
        results[key] = result
        print(json.dumps({'event': key, 'status': result['status'], 'pair_counts': result.get('pair_counts'),
                          'admission': result['admission']['status'], 'elapsed_seconds': time.monotonic()-start}), flush=True)
    if file_signatures(input_files(cache)) != inventory.signatures:
        raise ValueError('Frozen cache changed; refusing publication of current-cache results.')
    scripts = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ('runtime_retained.py', 'window_contact_v2.py', 'window_contact_v3.py', 'window_admission.py', 'interface_topology.py',
         'run_window_audit.py', 'window_source.py', 'window_exterior.py', 'window_geometry.py')]]
    summary = {'schema': 'c1-lite-repair-summary-v3', 'status': 'COMPLETED_WITH_SCIENTIFIC_REJECTIONS_OR_UNRESOLVED_GATES',
        'runtime_admitted': False, 'production_started': False, 'render_started': False,
        'elapsed_seconds': time.monotonic()-start, 'cache_input_sha256': inventory.digest,
        'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
        'ab_result_sha256': hashlib.sha256(args.ab_result.read_bytes()).hexdigest(),
        'events': [{'key': key, 'ideal_audit_status': r['status'], 'pair_counts': r.get('pair_counts'),
                    'admission': r['admission']['status'], 'rejected_gates': r['admission']['rejected_gates'],
                    'unresolved_gates': r['admission']['unresolved_gates'], 'all_units_scanned': r.get('all_units_scanned', False)}
                   for key, r in results.items()]}
    payloads = {key+'.json': (json.dumps(value, indent=2, sort_keys=True)+'\n').encode() for key, value in results.items()}
    summary['artifacts'] = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in payloads.items()}
    payloads['summary.json'] = (json.dumps(summary, indent=2, sort_keys=True)+'\n').encode()
    total = sum(map(len, payloads.values()))
    if total > OUTPUT_BUDGET:
        raise ValueError('Refuse outputs larger than the 12 MiB budget.')
    output.mkdir(parents=True)
    for name, data in payloads.items():
        with (output/name).open('xb') as handle:
            handle.write(data)
    print(json.dumps({'output': str(output), 'bytes': total, 'status': summary['status']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
