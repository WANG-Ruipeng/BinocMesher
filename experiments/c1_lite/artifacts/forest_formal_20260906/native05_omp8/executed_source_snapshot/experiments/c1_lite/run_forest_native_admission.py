#!/usr/bin/env python3
"""Actual five-element, one-query-at-a-time Forest admission continuation."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as F
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forest_native_campaign import PrivateCache, file_sha, mesh_receipt, apply_and_verify
from forest_schedule_policy import make_schedule, summarize_decisions
from run_forest_formal import Reports, REGISTRY_SHA, message, sha
from window_source import fr

FROZEN_SO = 'f4263a2f47ba5283175a921e49b8867998bdd8124aac810793b34242ec43a3c9'


def query_token(query):
    return (query['time_mode'], query['evaluation_tau'] if query['time_mode'] == 'exact' else query['physical_time_hex'])

def validate_query_binding(query, snapshot):
    actual = snapshot['query']
    if actual['mode'] != query['time_mode']:
        raise ValueError('Actual native query mode differs from requested schedule.')
    tau = (fr(actual['exact_time']) if actual['mode'] == 'exact'
           else F.from_float(float.fromhex(actual['effective_discrete_time_hex'])))
    if tau != F(query['evaluation_tau']) or actual['physical_time_hex'] != query['physical_time_hex']:
        raise ValueError('Actual native query time differs from requested schedule.')




def decide_runtime(event):
    cases = event['runtime']['cases']
    failures = [row for row in cases if row['audit']['status'] == 'REJECT'
                and row['audit'].get('certified_necessary_policy_failure') is True]
    if failures:
        witness = failures[0]
        event.update(decision='REJECTED_FIXED_POLICY', decisive_rejection={
            'certified_necessary_policy_failure': True, 'gate': 'actual_requested_query',
            'reason_code': witness['audit'].get('reason_code', witness['audit'].get('reason', 'ACTUAL_PATCH_REJECT')),
            'witness': witness, 'scope': 'Actual native required-query failure of the declared graph-fan policy.'})
        event['runtime'].update(status='BASELINE_ENTIRE_REQUESTED_SCHEDULE', published_partial_results=False,
            discarded_proposals=sum(row['audit']['status'] == 'PASS' for row in cases))
        return
    event.pop('unresolved_reason', None)
    expected = {q['key'] for q in event['schedule']['all_queries']}
    covered = {row['query']['key'] for row in cases}
    if len(covered) != len(cases):
        raise ValueError('Duplicate actual query cannot substitute for complete schedule coverage.')
    if covered == expected and all(row['audit']['status'] in ('PASS', 'BASELINE_OUTSIDE_WINDOW') for row in cases):
        # All candidates remain staged until campaign input verification.
        event['runtime']['status'] = 'PREPARED_ENTIRE_REQUESTED_SCHEDULE'
    else:
        event['runtime']['status'] = 'UNKNOWN_REQUESTED_SCHEDULE'
        event['unresolved_reason'] = 'ACTUAL_REQUESTED_QUERY_NOT_CERTIFIED'
        event['runtime']['unknown_query_reasons'] = sorted({str(row['audit'].get('reason', 'Unwitnessed rejection')) for row in cases if row['audit']['status'] not in ('PASS', 'BASELINE_OUTSIDE_WINDOW')})


def load_population(root):
    summary = json.loads((root/'summary.json').read_text())
    if summary.get('registry_sha256') != REGISTRY_SHA or not summary.get('input_verification_pass') or summary.get('e2_regression') != 'PASS_FULL_PARSER_AND_FROZEN_E2_PARITY':
        raise ValueError('Source campaign lacks verified fixed-input/E2 calibration evidence.')
    index = json.loads((root/'events_index.json').read_text())
    events = []
    for row in index:
        path = root/row['artifact']['path']
        if file_sha(path) != row['artifact']['sha256']:
            raise ValueError('A source event artifact changed.')
        event = json.loads(path.read_text())
        if event['event_id'] != row['event_id']:
            raise ValueError('Source event/index identity mismatch.')
        event['source_artifact'] = row['artifact']
        events.append(event)
    if len(events) != 131 or sha(sorted(e['event_id'] for e in events)) != summary['expected_event_ids_sha256']:
        raise ValueError('Native continuation changed the fixed population.')
    return summary, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-attempt', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--build-repo', type=Path, required=True)
    parser.add_argument('--camera-inputs', type=Path, required=True)
    parser.add_argument('--effective-inputs', type=Path, required=True)
    parser.add_argument('--expected-so-sha256', default=FROZEN_SO)
    parser.add_argument('--reference-receipts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exact-contact-fallback', action='store_true')
    parser.add_argument('--frozen-build-repo', type=Path, default=Path('/home/warpwang/binoc-runs/e2-runtime-loop-20260906/build'))
    args = parser.parse_args()
    reports = Reports(args.output)
    private, reader, calls, events = None, None, 0, []
    started, cpu = time.monotonic(), time.process_time()
    summary = {'status': 'UNKNOWN', 'policy_admission_rate': None}
    try:
        from forest_native_reader import initialize_forest
        from forest_native_patch import spec_from_source, audit_query
        source_summary, events = load_population(args.source_attempt)
        camera = json.loads(args.camera_inputs.read_text())
        effective = json.loads(args.effective_inputs.read_text())
        if file_sha(args.camera_inputs) != source_summary['camera_inputs_sha256']:
            raise ValueError('Camera input differs from source campaign.')
        verification = json.loads((args.source_attempt/'input_verification.json').read_text())
        registry_paths = [Path(p).resolve() for p in verification['input_full_sha256'] if Path(p).name == 'event_registry_p1.csv']
        if len(registry_paths) != 1 or args.cache.resolve() != registry_paths[0].parent:
            raise ValueError('Actual native cache path is not the cache bound by the source compiler.')
        for path, expected in verification['input_full_sha256'].items():
            if file_sha(path) != expected:
                raise ValueError('Native source cache differs from the compiled source bytes: '+path)
        here = Path(__file__).resolve().parent
        source_files = [here/name for name in ('run_forest_native_admission.py', 'forest_native_reader.py',
            'forest_native_patch.py', 'forest_native_campaign.py', 'forest_schedule_policy.py',
            'run_forest_formal.py', 'window_source.py', 'forest_frozen_query.py', 'forest_observer_contract.py',
            'forest_fan_contact.py', 'forest_exact_contact.py')]
        source_files += sorted((here.parents[1]/'binocmesher').glob('*.py'))
        # Independent tests/unexecuted fallback research are not runtime inputs.
        code_hashes = {str(p): file_sha(p) for p in source_files}
        private = PrivateCache(args.cache)
        reader = initialize_forest(args.build_repo, private.path, camera, effective,
                                   expected_so_sha256=args.expected_so_sha256)
        specs = {}
        for event in events:
            if event['decision'] == 'REJECTED_FIXED_POLICY':
                continue
            if event.get('unresolved_reason') != 'ACTUAL_FIVE_ELEMENT_REQUESTED_SCHEDULE_ADAPTER_REQUIRED':
                raise ValueError('Unresolved source event needs a source-stage correction before native evaluation.')
            source = event['compiler']['source']
            if source['cache_input_sha256'] != source_summary['streamed_source_binding_sha256']:
                raise ValueError('Event source contract is not bound to this source campaign cache.')
            specs[event['event_id']] = spec_from_source(source)
            bounds = [fr(source['levels'][k]) for k in ('lower', 'root', 'upper')]
            event['schedule'] = make_schedule(camera, *bounds, actual_delta=reader.delta_t)
            event['schedule_sha256'] = sha(event['schedule'])
            event['runtime'] = {'status': 'PREPARING', 'cases': [], 'published_partial_results': False,
                'temporal_scope': 'ONE_EVENT_ALL_REQUESTED_QUERIES_AGAINST_SAME_BASELINE',
                'source_sha256': sha(source), 'schedule_sha256': event['schedule_sha256']}
        references = {}
        if args.reference_receipts:
            document = json.loads(args.reference_receipts.read_text())
            if (document.get('library_sha256') != FROZEN_SO or document.get('registry_sha256') != REGISTRY_SHA
                    or document.get('camera_inputs_sha256') != file_sha(args.camera_inputs)
                    or document.get('effective_inputs_sha256') != file_sha(args.effective_inputs)
                    or document.get('initialization', {}).get('actual_delta_t_hex') != float(reader.delta_t).hex()
                    or document.get('final_input_verification', {}).get('status') != 'PASS'
                    or not document.get('private_cache_removed')):
                raise ValueError('Frozen-library reference receipt lacks matching verified input provenance.')
            for row in document['queries']:
                if row.get('identity_enabled') is False:
                    key = ('exact', row['root']) if 'root' in row else tuple(row['query_token'])
                    references[key] = row['meshes']
        query_receipts = []

        def evaluate(query, selected):
            nonlocal calls
            key = query_token(query)
            value = F(query['evaluation_tau']) if query['time_mode'] == 'exact' else float.fromhex(query['physical_time_hex'])
            if args.expected_so_sha256 != FROZEN_SO and key not in references:
                destination = reports.root/'references'/('frozen-'+sha(key)[:16]+'.json')
                destination.parent.mkdir(exist_ok=True)
                command = [sys.executable, str(here/'forest_frozen_query.py'),
                    '--cache-copy', str(private.path), '--build-repo', str(args.frozen_build_repo),
                    '--camera-inputs', str(args.camera_inputs), '--effective-inputs', str(args.effective_inputs),
                    '--time-mode', query['time_mode'], '--value', key[1], '--output', str(destination)]
                message('frozen_query_reference_start', key=key)
                completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
                calls += 1
                if completed.returncode != 0:
                    raise RuntimeError('Frozen query worker failed: '+(completed.stdout+completed.stderr)[-6000:])
                receipt = json.loads(destination.read_text())
                if (receipt.get('status') != 'FROZEN_ORDINARY_QUERY_RECEIPT' or tuple(receipt.get('query_token', ())) != key
                        or receipt.get('library_sha256') != FROZEN_SO or receipt.get('registry_sha256') != REGISTRY_SHA
                        or receipt.get('camera_inputs_sha256') != file_sha(args.camera_inputs)
                        or receipt.get('effective_inputs_sha256') != file_sha(args.effective_inputs)
                        or receipt.get('initialization', {}).get('actual_delta_t_hex') != float(reader.delta_t).hex()
                        or receipt.get('identity_enabled') is not False):
                    raise ValueError('Frozen query worker receipt is not bound to this requested query and input.')
                references[key] = receipt['meshes']
                reports.bytes += destination.stat().st_size
                if reports.bytes > 100*1024**2:
                    raise MemoryError('Native campaign reference receipts exceed the report budget.')
                message('frozen_query_reference_pass', key=key)
            begin = time.monotonic()
            ordinary = reader.slice_query(value, mode=query['time_mode'], ledger=False); calls += 1
            baseline = mesh_receipt(ordinary['meshes'])
            del ordinary
            snapshot = reader.slice_query(value, mode=query['time_mode'], ledger=True); calls += 1
            if mesh_receipt(snapshot['meshes']) != baseline:
                raise ValueError('Identity observation changed an actual ordinary query.')
            validate_query_binding(query, snapshot)
            if args.expected_so_sha256 != FROZEN_SO and references.get(key) != baseline:
                raise ValueError('Isolated observer library lacks matching frozen-library baseline receipt for '+str(key))
            if snapshot['identity_status'] != 1:
                raise RuntimeError('Observer capacity/identity is not ready: '+str(snapshot.get('identity_error')))
            from forest_observer_contract import original_identity_receipt
            identity_receipt = original_identity_receipt(snapshot)
            query_report = {'query_token': key, 'query': query, 'baseline': baseline,
                'identity_encoding': identity_receipt,
                'observer_enabled_disabled_byte_equal': True,
                'frozen_library_baseline_parity': 'SAME_FROZEN_LIBRARY' if args.expected_so_sha256 == FROZEN_SO else 'EXACT_RECEIPT_MATCH',
                'counts': snapshot['counts'], 'native_cost': snapshot['cost'], 'events': []}
            for event in selected:
                if event['decision'] == 'REJECTED_FIXED_POLICY':
                    continue
                actual_query = next(q for q in event['schedule']['all_queries'] if query_token(q) == key)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 8*1024**3:
                    raise MemoryError('Native event processing exceeded the declared 8 GiB RSS budget.')
                event['runtime_attempted'] = True
                before = time.monotonic()
                if not actual_query['active']:
                    case = {'query': actual_query, 'audit': {'status': 'BASELINE_OUTSIDE_WINDOW'},
                            'baseline': baseline, 'output_equals_baseline': True}
                else:
                    plan, audit = audit_query(snapshot, specs[event['event_id']], F(actual_query['evaluation_tau']),
                        full_exterior=True, exact_contact_fallback=args.exact_contact_fallback)
                    case = {'query': actual_query, 'audit': audit}
                    if audit['status'] == 'PASS':
                        if plan is None:
                            raise ValueError('Native PASS lacks an actual patch plan.')
                        case.update(plan=plan, actual_array_receipt=apply_and_verify(snapshot['meshes'], plan))
                case['check_wall_seconds'] = time.monotonic()-before
                event['runtime']['cases'].append(case)
                decide_runtime(event)
                query_report['events'].append({'event_id': event['event_id'], 'status': case['audit']['status'],
                                              'wall_seconds': case['check_wall_seconds'], 'case': case})
            if mesh_receipt(snapshot['meshes']) != baseline:
                raise ValueError('An event evaluation mutated the shared ordinary baseline.')
            query_report.update(all_events_preserved_shared_baseline=True, wall_seconds=time.monotonic()-begin)
            query_receipts.append(query_report)
            reports.save('queries/'+f'{len(query_receipts):02d}-'+sha(key)[:12]+'.json', query_report)
            message('native_query_completed', key=key, checked=len(query_report['events']),
                    statuses=dict(Counter(e['status'] for e in query_report['events'])), wall_seconds=query_report['wall_seconds'])
            del snapshot
            gc.collect()

        # Exact root is mandatory but not part of natural-frame hit denominators.
        for root in sorted({e['root'] for e in events if e['event_id'] in specs}, key=F):
            selected = [e for e in events if e['root'] == root and e['event_id'] in specs]
            evaluate(selected[0]['schedule']['exact_root'], selected)
        remaining = [e for e in events if e['event_id'] in specs and e['decision'] != 'REJECTED_FIXED_POLICY']
        natural = {}
        for event in remaining:
            for query in event['schedule']['natural']:
                natural[query_token(query)] = query
        for key, query in sorted(natural.items(), key=lambda item: float.fromhex(item[1]['physical_time_hex'])):
            selected = [e for e in remaining if e['decision'] != 'REJECTED_FIXED_POLICY'
                        and any(query_token(q) == key for q in e['schedule']['natural'])]
            if selected:
                evaluate(query, selected)
        reader.close(); reader = None
        input_check = private.verify(calls)
        if code_hashes != {str(p): file_sha(p) for p in source_files}:
            raise ValueError('Executed native campaign source changed during the run.')
        for event in events:
            if event['runtime'].get('status') == 'PREPARED_ENTIRE_REQUESTED_SCHEDULE':
                event['decision'] = 'ADMITTED_REQUESTED_SCHEDULE'
                event['runtime'].update(status='COMMITTED_REQUESTED_SCHEDULE',
                    published_partial_results=False, publication_kind='COMPACT_VERIFIED_ACTUAL_ARRAY_SCHEDULE_MANIFEST',
                    whole_mesh_arrays_discarded_after_verification=True, source_inputs_unchanged=True)
                event.pop('unresolved_reason', None)
        decision = summarize_decisions(events)
        artifacts = []
        for index, event in enumerate(events):
            reference = reports.save('events/'+f'{index:03d}-'+hashlib.sha256(event['event_id'].encode()).hexdigest()[:12]+'.json', event)
            artifacts.append({'event_id': event['event_id'], 'decision': event['decision'],
                'runtime_status': event['runtime']['status'], 'artifact': reference,
                'reason': event.get('decisive_rejection', {}).get('reason_code', event.get('unresolved_reason'))})
        reports.save('events_index.json', artifacts)
        modified = sorted({q['frame_number'] for e in events if e['decision'] == 'ADMITTED_REQUESTED_SCHEDULE'
                           for q in e['schedule']['natural'] if q['active']})
        summary = {'schema': 'forest-actual-requested-admission-v1',
            'status': 'COMPLETE_REAL_FIXED_POLICY_RATE' if decision['all_events_resolved'] else 'UNRESOLVED_ACTUAL_QUERIES',
            **decision, 'source_campaign_sha256': file_sha(args.source_attempt/'summary.json'),
            'registry_sha256': REGISTRY_SHA, 'expected_event_ids_sha256': source_summary['expected_event_ids_sha256'],
            'native_library_sha256': args.expected_so_sha256,
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS', 'UNSPECIFIED'),
            'exact_contact_fallback_enabled': args.exact_contact_fallback,
            'actual_delta_t_hex': next(e['schedule']['delta_t_hex'] for e in events if e['event_id'] in specs),
            'input_verification': input_check, 'executed_sources_sha256': code_hashes,
            'source_gate_rejections': source_summary['rejected_fixed_policy'],
            'actual_query_rejections': sum(e.get('decisive_rejection', {}).get('gate') == 'actual_requested_query' for e in events),
            'actual_unique_modified_natural_frames': modified,
            'actual_event_frame_modifications': sum(sum(q['active'] for q in e['schedule']['natural']) for e in events if e['decision'] == 'ADMITTED_REQUESTED_SCHEDULE'),
            'candidate_natural_schedule': source_summary['candidate_schedule'],
            'native_slice_calls': calls, 'unique_native_queries': len(query_receipts),
            'cost': {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
                     'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
            'scope': 'Individual-event five-element native requested-schedule admission; not same-root union, all-time theorem, or image-quality improvement.'}
    except Exception as error:
        summary.update(status='NATIVE_HARNESS_OR_INSTRUMENTATION_STOP_NOT_ALGORITHM_REJECTION',
            reason=type(error).__name__+': '+str(error), traceback=traceback.format_exc(),
            policy_admission_rate=None, native_slice_calls=calls)
    finally:
        if reader is not None:
            reader.close()
        if private is not None:
            try:
                summary['final_input_verification'] = private.verify(calls)
            except Exception as error:
                summary.update(status='INPUT_VERIFICATION_FAILED', policy_admission_rate=None, verification_error=str(error))
            private.remove(); summary['private_cache_removed'] = True
        summary['total_wall_seconds_including_cleanup'] = time.monotonic()-started
        reports.save('summary.json', summary)
    message('native_admission_summary', **{k: summary.get(k) for k in ('status', 'admitted', 'rejected_fixed_policy', 'unknown', 'policy_admission_rate')})
    return 0 if summary.get('all_events_resolved') else 2


if __name__ == '__main__':
    raise SystemExit(main())
