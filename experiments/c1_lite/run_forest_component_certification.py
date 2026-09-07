"""Complete Forest131 support graph and finite-schedule component certification.

Read-only original cache, isolated observer, compact actual support evidence.
Does not publish a sequence or substitute independent proposals for a union.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import gc
from itertools import combinations
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

from forest_component_graph import actual_footprints, build_graph, digest
from forest_native_campaign import PrivateCache, file_sha, mesh_receipt
from forest_native_reader import initialize_forest
from forest_sequence_inputs import (HERE, CAMERA, EFFECTIVE, CACHE, BUILD, LIBRARY_SHA,
    read, load_certified_inputs, bound_plan)
from run_forest_formal import Reports, message


def candidate_halo_owners(inventory_event, tau):
    def fraction(value):
        return F(value['numerator'], value['denominator'])
    matched = [c for c in inventory_event['cells'] if
        (c['kind'] == 'singleton' and fraction(c['t0']) == tau) or
        (c['kind'] == 'open_interval' and fraction(c['t0']) < tau < fraction(c['t1']))]
    if len(matched) != 1:
        raise ValueError('Candidate reservation lacks a unique exact source cell.')
    cell = matched[0]
    if not cell['candidate_vertex_halo_complete'] or not cell['event_candidates_complete']:
        raise ValueError('Candidate reservation has incomplete Vhalo.')
    owner_ids = sorted({o for fid in cell['actual_halo_face_ids']
        for o in inventory_event['face_class_catalog'][fid]['owner_ids']})
    return [inventory_event['owner_catalog'][i]['reference'] for i in owner_ids]


def pair_proofs(meshes, records, footprints):
    from forest_component_union import check_pair_interactions
    proofs = []
    for a, b in combinations(sorted(records, key=lambda x: x['event_id']), 2):
        x, y = footprints[a['event_id']], footprints[b['event_id']]
        same_element = x['element'] == y['element']
        shared = bool(same_element and (
            set(x['boundary_actual_ids']) & set(y['boundary_actual_ids'])
            or set(x['source_face_rows']) & set(y['source_face_rows'])))
        shared |= bool(set(map(tuple, x['consumed_owners'])) & set(map(tuple, y['consumed_owners'])))
        lo, hi = np.asarray(x['bounds']); lo2, hi2 = np.asarray(y['bounds'])
        if not shared and (np.any(lo > hi2) or np.any(lo2 > hi)):
            report = {'status': 'PASS_PAIR_INTERACTIONS',
                'proof': 'STRICT_ACTUAL_BINARY32_SUPPORT_AABB', 'triangle_pairs_excluded': 32}
        else:
            report = check_pair_interactions(meshes, [a, b])
        proofs.append({'events': [a['event_id'], b['event_id']], 'report': report})
    if len(proofs) != len(records)*(len(records)-1)//2:
        raise ValueError('Joint pair denominator mismatch.')
    return proofs


def decide_components(graph, queries):
    for component in graph['components']:
        if component['independently_rejected_members']:
            continue
        selected = [q for q in queries if q['root'] == component['root']]
        failures = []
        for query in selected:
            for pair in query['pair_proofs']:
                if set(pair['events']) <= set(component['events']) and pair['report']['status'] != 'PASS_PAIR_INTERACTIONS':
                    failures.append({'query': query['query']['key'], **pair})
        component['joint_failures'] = failures
        component['decision'] = 'FAIL_CLOSED_UNSUPPORTED_JOINT_COMPONENT' if failures else 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE'
        component['certified_query_keys'] = [q['query']['key'] for q in selected]
    admitted = [c for c in graph['components'] if c['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE']
    return {'component_denominator': len(graph['components']), 'admitted_components': len(admitted),
        'jointly_admitted_events': sum(len(c['events']) for c in admitted),
        'canonical_event_denominator': graph['event_count'],
        'component_admission_rate': len(admitted)/len(graph['components']),
        'joint_event_admission_rate': sum(len(c['events']) for c in admitted)/graph['event_count'],
        'root_denominator': len(graph['root_counts']),
        'resolved_roots': len(graph['root_counts']),
        'nonempty_treated_roots': len({c['root'] for c in admitted}),
        'per_root': {root: {'events': graph['root_counts'][root]['events'],
            'components': sum(c['root'] == root for c in graph['components']),
            'admitted_components': sum(c['root'] == root for c in admitted),
            'jointly_admitted_events': sum(len(c['events']) for c in admitted if c['root'] == root)}
            for root in graph['root_counts']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = Reports(args.output)
    started, cpu = time.monotonic(), time.process_time()
    reader, private, calls = None, None, 0
    summary = {'status': 'STOP_NOT_A_METHOD_DECISION', 'sequence_executed': False}
    try:
        inputs = load_certified_inputs()
        inventory = read(args.inventory)
        if inventory['status'] != 'PASS_COMPLETE_FIXED_SOURCE_CANDIDATE_INVENTORY':
            raise ValueError('Source support inventory is incomplete.')
        inventory_events = {e['event_id']: e for e in inventory['events']}
        events = inputs['events']
        if set(inventory_events) != {e['event_id'] for e in events}:
            raise ValueError('Source inventory changed the full event population.')
        bindings = {str(args.inventory.resolve()): file_sha(args.inventory), **inputs['bindings'],
                    **inventory['frozen_artifact_sha256'], **inventory['executed_sources_sha256']}
        for path, expected in bindings.items():
            if file_sha(path) != expected:
                raise ValueError('Composition input changed: '+path)
        code = {str(HERE/name): file_sha(HERE/name) for name in (
            'run_forest_component_certification.py', 'forest_component_graph.py',
            'forest_component_union.py', 'forest_sequence_inputs.py',
            'forest_native_patch.py', 'forest_native_campaign.py', 'forest_native_reader.py')}
        private = PrivateCache(CACHE)
        reader = initialize_forest(BUILD, private.path, inputs['camera'], inputs['effective'],
                                   expected_so_sha256=LIBRARY_SHA)
        from forest_native_patch import spec_from_source
        from run_forest_native_admission import validate_query_binding
        from forest_observer_contract import original_identity_receipt
        specs = {e['event_id']: spec_from_source(e['compiler']['source']) for e in events if e['compiler'].get('source')}
        if float(reader.delta_t).hex() != inputs['summary']['actual_delta_t_hex']:
            raise ValueError('Actual time mapping changed.')
        queries = []
        query_artifacts = []
        for root in sorted({e['root'] for e in events}, key=F):
            selected = [e for e in events if e['root'] == root]
            active = [q for q in selected[0]['schedule']['all_queries'] if q['active']]
            for query in sorted(active, key=lambda q: (q['kind'] != 'exact_root', F(q['evaluation_tau']))):
                before = time.monotonic()
                tau = F(query['evaluation_tau'])
                value = tau if query['time_mode'] == 'exact' else float.fromhex(query['physical_time_hex'])
                snapshot = reader.slice_query(value, mode=query['time_mode'], ledger=True); calls += 1
                validate_query_binding(query, snapshot)
                baseline = mesh_receipt(snapshot['meshes'])
                if baseline != inputs['queries'][query['key']]['baseline']:
                    raise ValueError('Actual five-element baseline differs from audited frozen ordinary output.')
                requests = []
                for event in selected:
                    eid = event['event_id']
                    row = {'event_id': eid, 'element': event['element'], 'spec': specs.get(eid)}
                    if eid not in specs:
                        row['candidate_actual_owners'] = candidate_halo_owners(inventory_events[eid], tau)
                    requests.append(row)
                footprints = actual_footprints(snapshot, requests, tau)
                by_id = {r['event_id']: r for r in footprints}
                records = []
                for event in selected:
                    if event['decision'] == 'ADMITTED_REQUESTED_SCHEDULE':
                        eid = event['event_id']
                        plan = bound_plan(event, query, baseline, specs[eid],
                                          {**by_id[eid], 'certified_baseline': baseline})
                        records.append({'event_id': eid, 'component_id': 'PROVISIONAL-'+eid, 'plan': plan})
                proofs = pair_proofs(snapshot['meshes'], records, by_id)
                if mesh_receipt(snapshot['meshes']) != baseline:
                    raise ValueError('Support/pair checks mutated baseline.')
                row = {'root': root, 'query': query, 'baseline': baseline,
                    'identity_encoding': original_identity_receipt(snapshot), 'events': footprints,
                    'certified_independent_plans': records, 'pair_proofs': proofs,
                    'native_cost': snapshot['cost'], 'wall_seconds': time.monotonic()-before}
                queries.append(row)
                query_artifacts.append(reports.save('queries/'+query['key']+'.json', row))
                message('component_support_query', query=query['key'], root=root, events=len(footprints),
                    unresolved_support=sum(r['status'] != 'COMPLETE_ACTUAL_REQUESTED_SUPPORT' for r in footprints),
                    pair_statuses=dict(Counter(p['report']['status'] for p in proofs)), wall_seconds=row['wall_seconds'])
                del snapshot
                gc.collect()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 8*1024**3:
                    raise MemoryError('Component graph exceeded 8 GiB RSS.')
        symbolic = list(inventory.get('symbolic_edges', []))
        # Any actual joint failure must be inside a component, even if a future
        # broad-phase implementation was incomplete. Never silently cross it.
        for query in queries:
            for pair in query['pair_proofs']:
                if pair['report']['status'] != 'PASS_PAIR_INTERACTIONS':
                    symbolic.append({'events': pair['events'], 'reason': 'ACTUAL_JOINT_PAIR_NOT_CERTIFIED',
                                     'cell': query['query']['key']})
        graph = build_graph(events, queries, symbolic_edges=symbolic)
        decision = decide_components(graph, queries)
        graph_ref = reports.save('support_graph.json', graph)
        reader.close(); reader = None
        verification = private.verify(calls)
        for path, expected in {**bindings, **code}.items():
            if file_sha(path) != expected:
                raise ValueError('Bound input or executed source changed during certification: '+path)
        summary = {'schema': 'forest-component-schedule-certification-v1',
            'status': 'PASS_FOREST_COMPONENT_REQUESTED_CERTIFICATION', **decision,
            'support_graph': graph_ref, 'query_count': len(queries), 'native_slice_calls': calls,
            'query_artifacts': query_artifacts,
            'unknown_support_rows_conservatively_joined': graph['unknown_actual_support_rows'],
            'source_support_inventory_sha256': file_sha(args.inventory),
            'input_bindings_sha256': bindings, 'executed_sources_sha256': code,
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS', 'UNSPECIFIED'),
            'actual_delta_t_hex': inputs['summary']['actual_delta_t_hex'],
            'input_verification': verification, 'sequence_executed': False,
            'scope': 'Finite requested component certification only; actual combined sequence and visibility remain separate gates.'}
    except Exception as error:
        summary.update(status='STOP_NOT_A_METHOD_DECISION', reason=type(error).__name__+': '+str(error),
                       traceback=traceback.format_exc())
    finally:
        if reader is not None:
            reader.close()
        if private is not None:
            try:
                summary['final_input_verification'] = private.verify(calls)
            except Exception as error:
                summary.update(status='INPUT_VERIFICATION_FAILED', verification_error=str(error))
            private.remove(); summary['private_cache_removed'] = True
        summary['cost'] = {'wall_seconds': time.monotonic()-started,
            'cpu_seconds': time.process_time()-cpu,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        reports.save('summary.json', summary)
    message('component_certification_summary', **{k: summary.get(k) for k in
        ('status', 'jointly_admitted_events', 'admitted_components', 'component_denominator', 'reason')})
    return 0 if summary['status'] == 'PASS_FOREST_COMPONENT_REQUESTED_CERTIFICATION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
