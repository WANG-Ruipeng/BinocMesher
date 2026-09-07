"""Actually construct 64 combined Forest scenes plus two root diagnostics.

Arrays are streamed in memory and discarded after exact receipts. All frame
artifacts remain staged until complete input verification commits the manifest.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import gc
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

from forest_component_graph import actual_footprints, digest
from forest_native_campaign import PrivateCache, file_sha, mesh_receipt
from forest_native_reader import initialize_forest, FROZEN_SO_SHA256
from forest_sequence_inputs import (HERE, CAMERA, EFFECTIVE, CACHE, BUILD, LIBRARY_SHA,
    read, load_certified_inputs, bound_plan, all_natural_queries)
from run_forest_formal import Reports, message, REGISTRY_SHA


def load_component_certificate(root):
    root = Path(root).resolve()
    summary = read(root/'summary.json')
    if (summary['status'] != 'PASS_FOREST_COMPONENT_REQUESTED_CERTIFICATION'
            or summary.get('final_input_verification', {}).get('status') != 'PASS'
            or not summary.get('private_cache_removed')):
        raise ValueError('Component schedule certification has not completed cleanly.')
    bindings = {str(root/'summary.json'): file_sha(root/'summary.json')}
    for item in [summary['support_graph'], *summary['query_artifacts']]:
        path = root/item['path']
        if file_sha(path) != item['sha256']:
            raise ValueError('Component graph/query certificate changed.')
        bindings[str(path)] = item['sha256']
    for path, expected in {**summary['input_bindings_sha256'], **summary['executed_sources_sha256']}.items():
        if file_sha(path) != expected:
            raise ValueError('Component certificate source/input changed: '+path)
        bindings[path] = expected
    graph = read(root/summary['support_graph']['path'])
    queries = {q['query']['key']: q for q in (read(root/r['path']) for r in summary['query_artifacts'])}
    if len(queries) != 18 or len(queries) != summary['query_count']:
        raise ValueError('Incomplete active component schedule.')
    return summary, graph, queries, bindings


def reference_frames(root, certification_sha):
    if root is None:
        return {}, {}
    root = Path(root).resolve(); summary = read(root/'summary.json')
    if (summary['status'] != 'PASS_FOREST_ATOMIC_REQUESTED_SEQUENCE'
            or summary['certification_sha256'] != certification_sha
            or summary.get('final_input_verification', {}).get('status') != 'PASS'
            or not summary.get('private_cache_removed')):
        raise ValueError('Reference sequence is not a committed matching run.')
    frames, bindings = {}, {str(root/'summary.json'): file_sha(root/'summary.json')}
    for item in summary['frame_artifacts']:
        path = root/item['path']
        if file_sha(path) != item['sha256']:
            raise ValueError('Reference sequence frame changed.')
        bindings[str(path)] = item['sha256']
        row = read(path)
        if row['query']['key'] in frames:
            raise ValueError('Duplicate reference sequence timestamp.')
        frames[row['query']['key']] = row
    if len(frames) != 66:
        raise ValueError('Incomplete reference sequence.')
    return frames, bindings


def frozen_reference(query, private, reports):
    path = reports.root/'references'/('frozen-'+query['key']+'.json')
    path.parent.mkdir(exist_ok=True)
    value = query['evaluation_tau'] if query['time_mode'] == 'exact' else query['physical_time_hex']
    command = [sys.executable, str(HERE/'forest_frozen_query.py'), '--cache-copy', str(private.path),
        '--build-repo', '/home/warpwang/binoc-runs/e2-runtime-loop-20260906/build',
        '--camera-inputs', str(CAMERA), '--effective-inputs', str(EFFECTIVE),
        '--time-mode', query['time_mode'], '--value', value, '--output', str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise RuntimeError('Frozen ordinary query failed: '+(completed.stdout+completed.stderr)[-4000:])
    row = read(path)
    if (row['status'] != 'FROZEN_ORDINARY_QUERY_RECEIPT'
            or row['query_token'] != [query['time_mode'], value]
            or row['library_sha256'] != FROZEN_SO_SHA256 or row['identity_enabled'] is not False
            or row['registry_sha256'] != REGISTRY_SHA
            or row['camera_inputs_sha256'] != file_sha(CAMERA)
            or row['effective_inputs_sha256'] != file_sha(EFFECTIVE)
            or row['actual_delta_t_hex'] != '0x1.500053e2d6239p-1'):
        raise ValueError('Unbound ordinary frozen frame reference.')
    reports.bytes += path.stat().st_size
    if reports.bytes > 100*1024**2:
        raise MemoryError('Sequence report budget exceeded.')
    return row['meshes'], {'path': str(path.relative_to(reports.root)), 'sha256': file_sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-sequence', type=Path)
    args = parser.parse_args()
    reports = Reports(args.output)
    started, cpu = time.monotonic(), time.process_time()
    reader, private, calls = None, None, 0
    summary = {'status': 'STOP_SEQUENCE_NOT_COMMITTED', 'published_partial_outputs': False}
    try:
        inputs = load_certified_inputs()
        certificate, graph, cert_queries, bindings = load_component_certificate(args.certification)
        from audit_forest_component_certificate import verify_component_certificate
        supplemental = verify_component_certificate(args.certification)
        supplemental_ref = reports.save('component_certificate_audit.json', supplemental)
        bindings.update(supplemental['audited_files_sha256'])
        cert_sha = file_sha(args.certification/'summary.json')
        references, reference_bindings = reference_frames(args.reference_sequence, cert_sha)
        bindings.update(reference_bindings)
        code = {str(HERE/name): file_sha(HERE/name) for name in (
            'run_forest_atomic_sequence.py', 'forest_component_graph.py', 'forest_component_union.py',
            'forest_sequence_inputs.py', 'forest_native_patch.py', 'forest_native_campaign.py', 'forest_native_reader.py')}
        admitted = [c for c in graph['components'] if c['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE']
        component_by_event = {eid: c['component_id'] for c in admitted for eid in c['events']}
        events = {e['event_id']: e for e in inputs['events']}
        private = PrivateCache(CACHE)
        reader = initialize_forest(BUILD, private.path, inputs['camera'], inputs['effective'], expected_so_sha256=LIBRARY_SHA)
        from forest_native_patch import spec_from_source
        from forest_component_union import compile_union
        from run_forest_native_admission import validate_query_binding
        specs = {eid: spec_from_source(e['compiler']['source']) for eid, e in events.items() if e['compiler'].get('source')}
        queries = all_natural_queries(inputs['camera'], reader.delta_t)
        queries += [cert_queries[key]['query'] for key in sorted(cert_queries) if key.startswith('root_')]
        frame_artifacts, reference_artifacts, modified, frame_costs = [], [], [], []
        active_event_frames = 0
        for query in queries:
            before = time.monotonic()
            key, tau = query['key'], F(query['evaluation_tau'])
            cq = cert_queries.get(key)
            selected = sorted(eid for eid, spec in specs.items() if eid in component_by_event and spec.lower < tau < spec.upper)
            expected_components = {c['component_id']: c['events'] for c in admitted
                                   if cq and c['root'] == cq['root']}
            if set(selected) != {e for members in expected_components.values() for e in members}:
                raise ValueError('Playback would partially apply a graph component.')
            audit_start = time.monotonic()
            if key in inputs['queries']:
                expected_baseline = inputs['queries'][key]['baseline']
            elif key in references:
                expected_baseline = references[key]['baseline']
            else:
                expected_baseline, ref = frozen_reference(query, private, reports); calls += 1
                reference_artifacts.append(ref)
            frozen_reference_seconds = time.monotonic()-audit_start
            t_slice = time.monotonic()
            value = tau if query['time_mode'] == 'exact' else float.fromhex(query['physical_time_hex'])
            snapshot = reader.slice_query(value, mode=query['time_mode'], ledger=bool(cq)); calls += 1
            slicing_seconds = time.monotonic()-t_slice
            validate_query_binding(query, snapshot)
            audit_start = time.monotonic()
            baseline = mesh_receipt(snapshot['meshes'])
            if baseline != expected_baseline or (cq and baseline != cq['baseline']):
                raise ValueError('Sequence ordinary geometry differs from its frozen/certified baseline.')
            baseline_hash_audit_seconds = time.monotonic()-audit_start
            t_resolve = time.monotonic()
            requests = []
            if cq:
                for old in cq['events']:
                    eid = old['event_id']
                    request = {'event_id': eid, 'element': old['element'], 'spec': specs.get(eid)}
                    if eid not in specs:
                        request['candidate_actual_owners'] = old['consumed_owners']
                    requests.append(request)
            footprints = actual_footprints(snapshot, requests, tau) if cq else []
            if cq and digest(footprints) != digest(cq['events']):
                raise ValueError('Complete root support replay differs from component graph evidence.')
            by_id = {r['event_id']: r for r in footprints}
            records = []
            for eid in selected:
                plan = bound_plan(events[eid], query, baseline, specs[eid],
                    {**by_id[eid], 'certified_baseline': baseline})
                original = next(r for r in cq['certified_independent_plans'] if r['event_id'] == eid)['plan']
                if digest(plan) != digest(original):
                    raise ValueError('Sequence plan differs from joint-certified plan.')
                records.append({'event_id': eid, 'component_id': component_by_event[eid], 'plan': plan})
            resolve_and_interface_audit_seconds = time.monotonic()-t_resolve
            t_union = time.monotonic()
            outputs, receipt = compile_union(snapshot['meshes'], records, expected_components=expected_components)
            verified_union_seconds = time.monotonic()-t_union
            if outputs is None or receipt['status'] != 'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED':
                raise ValueError('Pre-certified component failed actual atomic construction: '+str(receipt))
            if receipt['baseline'] != baseline:
                raise ValueError('Union bound a different baseline.')
            # Rejected current-domain source faces remain ordinary, not just
            # the far exterior. All original vertex/tag rows are already audited.
            rejected_domains_checked = 0
            if cq:
                for footprint in cq['events']:
                    if footprint['event_id'] in component_by_event or footprint['status'] != 'COMPLETE_ACTUAL_REQUESTED_SUPPORT':
                        continue
                    element, rows = footprint['element'], footprint['source_face_rows']
                    if not np.array_equal(outputs[element][1][rows], snapshot['meshes'][element][1][rows]):
                        raise ValueError('A rejected component source/candidate domain changed.')
                    rejected_domains_checked += 1
            changed = receipt['output'] != baseline
            if changed != bool(records):
                raise ValueError('Nonempty union/change accounting disagrees.')
            if query['kind'] == 'natural' and changed:
                modified.append(query['frame_number']); active_event_frames += len(records)
            cost = {'native_slice_seconds': slicing_seconds,
                'frozen_baseline_reference_audit_seconds': frozen_reference_seconds,
                'baseline_hash_audit_seconds': baseline_hash_audit_seconds,
                'resolve_with_complete_interface_audit_seconds': resolve_and_interface_audit_seconds,
                'verified_union_seconds_including_exact_recheck_and_array_audit': verified_union_seconds,
                'frame_harness_wall_seconds': time.monotonic()-before}
            frame_costs.append(cost)
            row = {'query': query, 'baseline': baseline, 'union': receipt,
                'records': records,
                'root_support_replay_sha256': digest(footprints) if cq else None,
                'records_sha256': digest(records), 'certification_sha256': cert_sha,
                'publication': 'STAGED_UNTIL_FULL_SEQUENCE_VERIFIED',
                'changed': changed, 'rejected_actual_domains_unchanged': rejected_domains_checked,
                'single_scene_output_elements': 5, 'fresh_baseline_for_each_timestamp': True,
                'full_scene_arrays_actually_constructed': True, 'persistent_mesh_files': 0,
                'cost': cost}
            frame_artifacts.append(reports.save('frames/'+key+'.json', row))
            message('atomic_sequence_frame', query=key, events=len(records), components=len(expected_components),
                    changed=changed, wall_seconds=cost['frame_harness_wall_seconds'])
            del outputs, snapshot
            gc.collect()
        reader.close(); reader = None
        for path, expected in {**bindings, **code}.items():
            if file_sha(path) != expected:
                raise ValueError('Sequence input or executed code changed: '+path)
        verification = private.verify(calls)
        summary = {'schema': 'forest-atomic-sequence-v1', 'status': 'PASS_FOREST_ATOMIC_REQUESTED_SEQUENCE',
            'publication': 'COMMITTED_FULL_REQUESTED_SEQUENCE_MANIFEST', 'published_partial_outputs': False,
            'partial_output_published': False,
            'component_supplemental_audit': supplemental_ref,
            'natural_frame_count': 64, 'exact_root_diagnostic_count': 2, 'actual_scene_outputs': len(queries),
            'actual_modified_natural_frames': modified, 'natural_frame_change_rate': len(modified)/64,
            'actual_natural_event_frame_replacements': active_event_frames,
            'jointly_admitted_events': len(component_by_event), 'admitted_components': len(admitted),
            'nonempty_treated_roots': len({c['root'] for c in admitted}), 'root_denominator': 2,
            'frame_artifacts': frame_artifacts, 'frozen_reference_artifacts': reference_artifacts,
            'certification_sha256': cert_sha, 'support_graph_sha256': certificate['support_graph']['sha256'],
            'actual_delta_t_hex': certificate['actual_delta_t_hex'],
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS', 'UNSPECIFIED'),
            'native_slice_calls': calls, 'input_verification': verification,
            'input_bindings_sha256': bindings, 'executed_sources_sha256': code,
            'cost_partition_totals_seconds': {key: sum(c[key] for c in frame_costs) for key in frame_costs[0]},
            'performance_scope': 'Certification is separate. Verified playback includes explicit local/exact rechecks and full-array audits; not pure production frame latency.',
            'scope': 'Actual five-element pre-displacement combined finite sequence, not all-time, post-displacement or visible quality improvement.'}
    except Exception as error:
        summary.update(status='STOP_SEQUENCE_NOT_COMMITTED', reason=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        if reader is not None:
            reader.close()
        if private is not None:
            try:
                summary['final_input_verification'] = private.verify(calls)
            except Exception as error:
                summary.update(status='INPUT_VERIFICATION_FAILED', verification_error=str(error))
            private.remove(); summary['private_cache_removed'] = True
        summary['cost'] = {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
                           'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        reports.save('summary.json', summary)
    message('atomic_sequence_summary', **{k: summary.get(k) for k in
        ('status', 'jointly_admitted_events', 'actual_scene_outputs', 'actual_modified_natural_frames', 'reason')})
    return 0 if summary['status'] == 'PASS_FOREST_ATOMIC_REQUESTED_SEQUENCE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
