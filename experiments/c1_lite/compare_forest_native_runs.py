#!/usr/bin/env python3
"""Independent, order-insensitive OMP1/8 Forest receipt comparison.

Only completed reports are read. No native library, rendering, or admission
code is imported. Equal unresolved reports are not certified admission parity.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

REGISTRY_SHA = '0ce1a439bf33b785db9dbb3ea55dbc548933477aca64d2fca64829f660609b52'
IGNORED = {'cost', 'native_cost', 'peak_rss_bytes', 'array_bytes_transient',
           'wall_seconds', 'cpu_seconds', 'total_wall_seconds_including_cleanup'}


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(stable(value)).hexdigest()


def semantic(value):
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items()
                if k not in IGNORED and not k.endswith(('_wall_seconds', '_cpu_seconds'))}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def token(query):
    mode = query['time_mode']
    if mode == 'exact':
        return ('exact', str(Fraction(query['evaluation_tau'])))
    if mode == 'physical':
        return ('physical', float.fromhex(query['physical_time_hex']).hex())
    raise ValueError('Unsupported query time mode.')


def keyed(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError('Duplicate semantic identity: '+str(identity))
        result[identity] = row
    return result


def event_semantic(event):
    result = semantic(event)
    if 'runtime' in result:
        result['runtime']['cases'] = {str(k): semantic(v) for k, v in
            keyed(event['runtime'].get('cases', []), lambda r: token(r['query'])).items()}
    if 'schedule' in result:
        for name in ('natural', 'all_queries'):
            if name in event['schedule']:
                result['schedule'][name] = {str(k): semantic(v) for k, v in
                    keyed(event['schedule'][name], token).items()}
    return result


def _read(path, receipts):
    path = Path(path)
    data = path.read_bytes()
    value = json.loads(data)
    receipts[str(path)] = hashlib.sha256(data).hexdigest()
    return value


def load_attempt(path):
    root = Path(path).resolve()
    receipts = {}
    summary = _read(root/'summary.json', receipts)
    index = _read(root/'events_index.json', receipts)
    events = {}
    for row in index:
        artifact = (root/row['artifact']['path']).resolve()
        if root not in artifact.parents or artifact.is_symlink():
            raise ValueError('Event artifact escapes its attempt directory.')
        event = _read(artifact, receipts)
        if receipts[str(artifact)] != row['artifact']['sha256']:
            raise ValueError('Event artifact SHA256 mismatch: '+row['event_id'])
        if event['event_id'] != row['event_id'] or event['decision'] != row['decision']:
            raise ValueError('Event index and artifact disagree.')
        if event['event_id'] in events:
            raise ValueError('Duplicate event ID in artifact index.')
        events[event['event_id']] = event
    queries = {}
    for path in sorted((root/'queries').glob('*.json')):
        query = _read(path, receipts)
        key = token(query['query'])
        if tuple(query['query_token']) != key or key in queries:
            raise ValueError('Duplicate or inconsistent actual query token.')
        queries[key] = query
    for filename, expected in receipts.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('A compared report changed while being read: '+filename)
    return {'root': str(root), 'summary': summary, 'events': events, 'queries': queries,
            'input_receipts': receipts}


def compare_loaded(left, right, *, expected_events=131, expected_queries=22, expected_schedule_queries=11):
    differences, incomplete = [], []
    def difference(kind, key, a, b):
        if a != b:
            differences.append({'kind': kind, 'key': str(key), 'left_sha256': sha(a),
                                'right_sha256': sha(b)})
    costs = {}
    for label, run in (('left', left), ('right', right)):
        summary, events, queries = run['summary'], run['events'], run['queries']
        costs[label] = {'omp_num_threads': summary.get('omp_num_threads'),
                       'cost': summary.get('cost'),
                       'total_wall_seconds_including_cleanup': summary.get('total_wall_seconds_including_cleanup')}
        def require(condition, reason):
            if not condition:
                incomplete.append({'attempt': label, 'reason': reason})
        require(summary.get('status') == 'COMPLETE_REAL_FIXED_POLICY_RATE'
                and summary.get('all_events_resolved') is True and summary.get('unknown') == 0,
                'Attempt is unfinished or has UNKNOWN outcomes.')
        require(summary.get('registry_sha256') == REGISTRY_SHA, 'Fixed Forest registry binding differs or is missing.')
        require(summary.get('private_cache_removed') is True, 'Private-cache cleanup is not verified.')
        for field in ('input_verification', 'final_input_verification'):
            require(summary.get(field, {}).get('status') == 'PASS'
                    and summary.get(field, {}).get('private_nonlog_inputs_unchanged') is True,
                    field+' did not verify original/nonlog inputs.')
        require(len(events) == expected_events and summary.get('canonical_event_denominator') == expected_events,
                'Fixed event population is incomplete.')
        require(summary.get('expected_event_ids_sha256') == sha(sorted(events)), 'Event-ID set digest is not verified.')
        require(len(queries) == expected_queries and summary.get('unique_native_queries') == expected_queries,
                'Complete actual-query receipt set is missing.')
        require(bool(summary.get('executed_sources_sha256')), 'Executed source hash manifest is missing.')
        for field in ('source_campaign_sha256', 'native_library_sha256', 'actual_delta_t_hex',
                      'exact_contact_fallback_enabled'):
            require(field in summary, 'Required input/protocol binding is missing: '+field)
        require(bool(summary.get('final_input_verification', {}).get('original_input_content_sha256')),
                'Original source-cache content binding is missing.')
        decisions = Counter(e['decision'] for e in events.values())
        require(decisions.get('UNKNOWN', 0) == 0
                and set(decisions) <= {'ADMITTED_REQUESTED_SCHEDULE', 'REJECTED_FIXED_POLICY'},
                'Event artifacts contain unresolved or unsupported decisions.')
        require(summary.get('admitted') == decisions.get('ADMITTED_REQUESTED_SCHEDULE', 0)
                and summary.get('rejected_fixed_policy') == decisions.get('REJECTED_FIXED_POLICY', 0),
                'Summary admission/rejection counts disagree with event artifacts.')
        for key, query in queries.items():
            require(query.get('observer_enabled_disabled_byte_equal') is True
                    and query.get('all_events_preserved_shared_baseline') is True
                    and len(query.get('baseline', [])) == 5,
                    'Query lacks complete unchanged five-element baseline evidence: '+str(key))
            per_event = keyed(query['events'], lambda r: r['event_id'])
            for eid, row in per_event.items():
                require(eid in events, 'Query references an event outside the fixed population.')
                if eid in events:
                    cases = keyed(events[eid].get('runtime', {}).get('cases', []), lambda r: token(r['query']))
                    require(key in cases and semantic(row['case']) == semantic(cases.get(key)),
                            'Query/event case evidence disagrees: '+eid+' '+str(key))
        for eid, event in events.items():
            cases = keyed(event.get('runtime', {}).get('cases', []), lambda r: token(r['query']))
            for key, case in cases.items():
                require(key in queries, 'Event case has no actual query receipt: '+eid+' '+str(key))
                if key in queries:
                    rows = keyed(queries[key]['events'], lambda r: r['event_id'])
                    require(eid in rows and semantic(rows[eid]['case']) == semantic(case),
                            'Event/query receipt coverage disagrees: '+eid+' '+str(key))
            if event['decision'] == 'REJECTED_FIXED_POLICY':
                require(event.get('decisive_rejection', {}).get('certified_necessary_policy_failure') is True
                        and bool(event.get('decisive_rejection', {}).get('gate')),
                        'Rejected event lacks a named certified policy witness: '+eid)
            if event['decision'] != 'ADMITTED_REQUESTED_SCHEDULE':
                continue
            requested = keyed(event['schedule']['all_queries'], token)
            require(event['runtime'].get('status') == 'COMMITTED_REQUESTED_SCHEDULE'
                    and event['runtime'].get('published_partial_results') is False
                    and event['runtime'].get('source_inputs_unchanged') is True,
                    'Admitted event lacks an atomic completed publication receipt: '+eid)
            require(len(cases) == expected_schedule_queries and set(cases) == set(requested),
                    'Admitted event is missing requested queries: '+eid)
            for key, case in cases.items():
                active = case['query']['active']
                if active:
                    receipt = case.get('actual_array_receipt', {})
                    require(case['audit']['status'] == 'PASS' and bool(case.get('plan'))
                            and receipt.get('status') == 'ACTUAL_ARRAYS_CONSTRUCTED_NOT_PUBLISHED'
                            and receipt.get('old_vertices_and_tags_byte_identical') is True
                            and receipt.get('all_retained_face_rows_byte_identical') is True
                            and receipt.get('other_elements_unchanged_by_object_identity') is True
                            and set(receipt.get('output', {}).get('sha256', {})) == {'vertices', 'faces', 'tags'},
                            'Active admitted case lacks a plan or verified actual output hashes: '+eid+' '+str(key))
                else:
                    require(case['audit']['status'] == 'BASELINE_OUTSIDE_WINDOW'
                            and case.get('output_equals_baseline') is True
                            and key in queries and case.get('baseline') == queries[key]['baseline'],
                            'Inactive admitted case lacks original baseline evidence: '+eid+' '+str(key))
    ls, rs = left['summary'], right['summary']
    if {str(ls.get('omp_num_threads')), str(rs.get('omp_num_threads'))} != {'1', '8'}:
        incomplete.append({'attempt': 'pair', 'reason': 'Thread metadata is not the declared OMP1/OMP8 pair.'})
    for field in ('registry_sha256', 'expected_event_ids_sha256', 'source_campaign_sha256',
                  'native_library_sha256', 'actual_delta_t_hex', 'executed_sources_sha256',
                  'exact_contact_fallback_enabled'):
        difference('input_or_code_binding', field, ls.get(field), rs.get(field))
    for field in ('input_verification', 'final_input_verification'):
        difference('source_cache_binding', field,
                   ls.get(field, {}).get('original_input_content_sha256'),
                   rs.get(field, {}).get('original_input_content_sha256'))
    difference('event_population', 'event IDs', sorted(left['events']), sorted(right['events']))
    difference('query_population', 'query tokens', sorted(left['queries']), sorted(right['queries']))
    compared_events = compared_queries = admitted_cases = 0
    for eid in sorted(set(left['events']) & set(right['events'])):
        a, b = left['events'][eid], right['events'][eid]
        difference('event_decision_witness_and_cases', eid, event_semantic(a), event_semantic(b))
        compared_events += 1
        if a['decision'] == b['decision'] == 'ADMITTED_REQUESTED_SCHEDULE':
            ca = keyed(a['runtime']['cases'], lambda r: token(r['query']))
            cb = keyed(b['runtime']['cases'], lambda r: token(r['query']))
            for key in sorted(set(ca) & set(cb)):
                difference('admitted_plan', (eid, key), semantic(ca[key].get('plan')), semantic(cb[key].get('plan')))
                difference('admitted_actual_output', (eid, key), semantic(ca[key].get('actual_array_receipt')),
                           semantic(cb[key].get('actual_array_receipt')))
                admitted_cases += 1
    for key in sorted(set(left['queries']) & set(right['queries'])):
        a, b = left['queries'][key], right['queries'][key]
        difference('five_element_baseline_hashes', key, a['baseline'], b['baseline'])
        difference('actual_query_metadata', key, semantic(a['query']), semantic(b['query']))
        difference('actual_identity_encoding', key, a.get('identity_encoding'), b.get('identity_encoding'))
        compared_queries += 1
    passed = not incomplete and not differences
    return {'schema': 'forest-omp1-omp8-receipt-comparison-v1',
            'status': ('PASS_COMPLETE_OMP1_OMP8_ADMISSION_PARITY' if passed else
                       'NOT_CERTIFIED_INCOMPLETE_OR_UNRESOLVED' if incomplete else 'DIFFERENT_COMPLETE_RUNS'),
            'consistent_complete_admission_certified': passed,
            'semantic_differences': differences, 'incomplete_or_unverified': incomplete,
            'event_pairs_compared': compared_events, 'query_pairs_compared': compared_queries,
            'admitted_schedule_case_pairs_compared': admitted_cases,
            'resources_and_threads': costs,
            'scope': 'One recorded OMP1 and one OMP8 run; not repeated-run statistics or all-time admission.'}


def compare_runs(left, right):
    runs = []
    for label, path in (('left', left), ('right', right)):
        try:
            runs.append(load_attempt(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {'schema': 'forest-omp1-omp8-receipt-comparison-v1',
                    'status': 'NOT_CERTIFIED_INVALID_OR_INCOMPLETE_REPORTS',
                    'consistent_complete_admission_certified': False,
                    'attempt': label, 'reason': type(error).__name__+': '+str(error)}
    try:
        result = compare_loaded(*runs)
    except (ValueError, KeyError, TypeError) as error:
        result = {'schema': 'forest-omp1-omp8-receipt-comparison-v1',
                  'status': 'NOT_CERTIFIED_INVALID_OR_INCOMPLETE_REPORTS',
                  'consistent_complete_admission_certified': False,
                  'reason': type(error).__name__+': '+str(error)}
    result['input_reports_sha256'] = {label: run['input_receipts'] for label, run in zip(('left', 'right'), runs)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if args.output.exists() or args.output.is_symlink() or not target.parent.is_dir():
        raise ValueError('Output must be a new file in an existing directory.')
    if any(target == p.resolve() or p.resolve() in target.parents for p in (args.left, args.right)):
        raise ValueError('Comparison output must be outside both frozen attempts.')
    result = compare_runs(args.left, args.right)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'consistent_complete_admission_certified')}, sort_keys=True))


if __name__ == '__main__':
    main()
