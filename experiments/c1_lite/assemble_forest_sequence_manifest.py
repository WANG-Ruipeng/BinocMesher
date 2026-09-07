"""JSON-only final manifest for the fixed Forest 64-frame + two-root schedule.

Reuses the independent sequence validator. Geometry and graph certification are
inherited from explicitly hash-bound supplemental audits, not recomputed here.
No native library, cache, image, mesh, or scientific experiment is opened.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import compare_forest_sequences as comparison

require = comparison.require
digest = comparison.digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path, value):
    return {'path': str(Path(path).resolve()), 'sha256': value}


def load_bound(path, bindings, expected=None, maximum=4*1024*1024):
    path = Path(path).resolve()
    value, actual, _ = comparison.read_json(path, maximum)
    require(expected is None or actual == expected, 'ARTIFACT_HASH_MISMATCH:'+str(path))
    require(str(path) not in bindings or bindings[str(path)] == actual, 'ARTIFACT_REBOUND')
    bindings[str(path)] = actual
    return value, reference(path, actual)


def child(directory, row):
    directory = Path(directory).resolve()
    relative = Path(row['path'])
    target = (directory / relative).resolve()
    require(not relative.is_absolute() and directory in target.parents, 'CHILD_ARTIFACT_ESCAPES_DIRECTORY')
    comparison.hash_value(row['sha256'], 'child_artifact')
    return target


def check_comparison(report, replay, bindings):
    require(replay['status'] == 'PASS_IDENTICAL_ATOMIC_FOREST_SEQUENCES', 'INDEPENDENT_RECOMPARISON_FAILED')
    require(all(report.get(k) == v for k, v in replay.items()), 'COMPARISON_REPORT_PAYLOAD_NOT_REPRODUCED')
    require(report.get('audited_files_sha256') == bindings, 'COMPARISON_NOT_BOUND_TO_THESE_134_FILES')
    require(len(bindings) == 134 and report.get('input_artifacts_unchanged') is True,
            'COMPARISON_INPUT_AUDIT_INCOMPLETE')
    require(report.get('executed_script_sha256') == sha(comparison.__file__), 'COMPARISON_SCRIPT_BINDING_MISMATCH')


def check_population(inventory, graph, certificate):
    require(inventory.get('status') == 'PASS_COMPLETE_FIXED_SOURCE_CANDIDATE_INVENTORY' and
            inventory.get('full_input_hash_set_equal_frozen') is True, 'SOURCE_INVENTORY_NOT_VERIFIED')
    events = {e['event_id']: e for e in inventory['events']}
    require(len(events) == len(inventory['events']) == inventory['event_count'] == 131, 'SOURCE_131_DENOMINATOR_CHANGED')
    require(graph.get('status') == 'COMPLETE_CONSERVATIVE_REQUESTED_SUPPORT_GRAPH' and
            graph.get('event_count') == 131 and graph.get('events_sha256') == digest(sorted(events)),
            'GRAPH_POPULATION_BINDING_MISMATCH')
    roots = dict(Counter(e['root'] for e in events.values()))
    require(roots == {'3/2': 66, '5/2': 65}, 'ROOT_POPULATION_CHANGED')
    members = {}; admitted = {}; per_root = {}
    for c in graph['components']:
        cid = c['component_id']; rows = c['events']
        require(rows == sorted(set(rows)) and rows and cid == 'component-'+digest(rows)[:16], 'INVALID_COMPONENT_IDENTITY')
        require(all(e in events and e not in members and events[e]['root'] == c['root'] for e in rows),
                'COMPONENT_POPULATION_OVERLAP_OR_ROOT_MISMATCH')
        members.update({e: cid for e in rows})
        rejected = sorted(e for e in rows if events[e]['native_individual_decision'] != 'ADMITTED_REQUESTED_SCHEDULE')
        require(c['independently_rejected_members'] == rejected, 'INDEPENDENT_REJECTED_MEMBERS_CHANGED')
        if c['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE':
            require(not rejected, 'REJECTED_MEMBER_WAS_ADMITTED')
            admitted.update({e: cid for e in rows})
        else:
            require(c['decision'] == 'FAIL_CLOSED_REJECTED_MEMBER' and rejected, 'UNEXPECTED_FIXED_COMPONENT_DECISION')
    require(set(members) == set(events) and len(graph['components']) == 79, 'GRAPH_DROPPED_EVENT_OR_COMPONENT')
    require(len(admitted) == 25 and len(set(admitted.values())) == 20, 'FIXED_JOINT_ADMISSION_COUNTS_CHANGED')
    for root in roots:
        cs = [c for c in graph['components'] if c['root'] == root]
        per_root[root] = {'events': roots[root], 'components': len(cs),
            'admitted_components': sum(c['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE' for c in cs),
            'jointly_admitted_events': sum(events[e]['root'] == root for e in admitted)}
    require(certificate['per_root'] == per_root, 'COMPONENT_PER_ROOT_COUNTS_CHANGED')
    for name, value in [('canonical_event_denominator',131), ('component_denominator',79),
                        ('admitted_components',20), ('jointly_admitted_events',25), ('query_count',18)]:
        require(certificate.get(name) == value, 'CERTIFICATE_FIXED_COUNT:'+name)
    return events, admitted, per_root


def check_frame_links(frames, queries, events, admitted):
    active = set(); modified = []; replacements = 0; root_events = {}
    event_frames = Counter()
    for frame in frames:
        q = frame['query']; key = q['key']; tau = F(q['evaluation_tau'])
        expected = {e: cid for e, cid in admitted.items()
                    if F(events[e]['window']['lower']) < tau < F(events[e]['window']['upper'])}
        actual = {r['event_id']: r['component_id'] for r in frame['records']}
        require(actual == expected and len(actual) == len(frame['records']), 'FRAME_COMPONENT_FULL_MEMBERSHIP_CHANGED:'+key)
        if frame['root_support_replay_sha256'] is not None:
            active.add(key); require(key in queries, 'ACTIVE_QUERY_MISSING_CERTIFICATE')
            cq = queries[key]
            require(frame['root_support_replay_sha256'] == digest(cq['events']), 'ROOT_SUPPORT_REPLAY_BINDING_MISMATCH')
            require(frame['baseline'] == cq['baseline'], 'COMPONENT_TO_SEQUENCE_BASELINE_MISMATCH')
            for k in ('key','kind','evaluation_tau','physical_time_hex','time_mode'):
                require(q[k] == cq['query'][k], 'COMPONENT_QUERY_TIME_MISMATCH')
            population = {e for e, row in events.items() if row['root'] == cq['root']}
            require({r['event_id'] for r in cq['events']} == population and len(cq['events']) == len(population),
                    'ROOT_SUPPORT_REPLAY_NOT_COMPLETE_POPULATION')
            require(all(r['status'] == 'COMPLETE_ACTUAL_REQUESTED_SUPPORT' and
                        r['full_interface_star_enumerated'] is True for r in cq['events']), 'INCOMPLETE_ACTUAL_SUPPORT')
            independent = {r['event_id']: r['plan'] for r in cq['certified_independent_plans']}
            require(all(independent.get(r['event_id']) == r['plan'] for r in frame['records']), 'FRESH_PLAN_CHANGED_FROM_CERTIFICATE')
        else:
            require(not expected and frame['union']['output'] == frame['baseline'], 'OUTSIDE_BASELINE_CHANGED')
        if q['kind'] == 'natural':
            i = q['frame_number']
            require(q['global_camera_time_hex'] == float((i-0.5)/24).hex(), 'ORIGINAL_24FPS_GRID_CHANGED')
            if actual: modified.append(i)
            replacements += len(actual); event_frames.update(actual.keys())
        else:
            root_events[str(tau)] = len(actual)
    require(active == set(queries) and len(active) == 18, 'INCOMPLETE_18_QUERY_ROOT_SUPPORT_REPLAY')
    require(sorted(modified) == list(range(21,29))+list(range(37,45)) and replacements == 200,
            'FIXED_NATURAL_FRAME_HIT_COUNTS_CHANGED')
    require(dict(root_events) == {'3/2':12,'5/2':13} and set(event_frames) == set(admitted) and
            set(event_frames.values()) == {8}, 'ROOT_OR_EVENT_SCHEDULE_COVERAGE_CHANGED')
    return {'natural_frames':64, 'exact_root_diagnostics':2, 'five_element_scene_outputs_per_omp':66,
        'modified_natural_frames':sorted(modified), 'modified_natural_frame_count':16,
        'unchanged_natural_frames':48, 'natural_event_frame_replacements':200,
        'natural_frames_per_jointly_admitted_event':8, 'treated_exact_roots':root_events,
        'nonempty_treated_root_count':2, 'complete_root_support_replay_queries':18}


def assemble(component, sequence_omp1, sequence_omp8, comparison_path, visibility=None):
    bindings = {}; refs = {}
    a, af, ab = comparison.load_sequence(sequence_omp1)
    b, bf, bb = comparison.load_sequence(sequence_omp8)
    require(str(a['omp_num_threads']) == '1' and str(b['omp_num_threads']) == '8', 'CLI_OMP_LABEL_MISMATCH')
    require(not set(ab) & set(bb), 'OMP_SEQUENCE_ARTIFACTS_ALIAS')
    bindings.update(ab); bindings.update(bb)
    replay = comparison.compare_sequences(a, af, b, bf)
    cr, refs['omp_comparison'] = load_bound(comparison_path, bindings)
    check_comparison(cr, replay, {**ab, **bb})
    component = Path(component).resolve()
    cert, refs['component_certification'] = load_bound(component/'summary.json', bindings)
    cert_sha = refs['component_certification']['sha256']
    require(cert.get('status') == 'PASS_FOREST_COMPONENT_REQUESTED_CERTIFICATION' and
            cert.get('private_cache_removed') is True, 'COMPONENT_CERTIFICATION_NOT_PASS')
    require(comparison.input_verification(cert) == comparison.input_verification(a), 'COMPONENT_CACHE_BINDING_CHANGED')
    graph, refs['support_graph'] = load_bound(child(component,cert['support_graph']), bindings, cert['support_graph']['sha256'])
    candidates = [p for p,h in cert['input_bindings_sha256'].items() if h == cert['source_support_inventory_sha256']]
    require(len(candidates) == 1, 'SOURCE_INVENTORY_BINDING_MISSING_OR_AMBIGUOUS')
    inv, refs['source_support_inventory'] = load_bound(candidates[0], bindings, cert['source_support_inventory_sha256'], 10*1024*1024)
    events, admitted, per_root = check_population(inv, graph, cert)
    queries = {}; component_bindings = dict(bindings)
    for row in cert['query_artifacts']:
        q, ref = load_bound(child(component,row), bindings, row['sha256'])
        require(q['query']['key'] not in queries, 'DUPLICATE_COMPONENT_QUERY')
        queries[q['query']['key']] = q
    require(len(queries) == 18, 'COMPONENT_QUERY_COUNT_CHANGED')
    component_bindings = {p:h for p,h in bindings.items() if p not in ab and p not in bb and p != refs['omp_comparison']['path']}
    refs['supplemental_component_audits'] = []
    for folder, summary, bound in ((sequence_omp1,a,ab),(sequence_omp8,b,bb)):
        directory = Path(folder).resolve()
        if directory.is_file(): directory = directory.parent
        summary_path = directory/'summary.json'
        refs['sequence_omp'+str(summary['omp_num_threads'])] = reference(summary_path,bound[str(summary_path)])
        require(summary['certification_sha256'] == cert_sha and summary['support_graph_sha256'] == refs['support_graph']['sha256'] and
                summary['actual_delta_t_hex'] == cert['actual_delta_t_hex'], 'SEQUENCE_CERTIFICATION_BINDING_CHANGED')
        row = summary['component_supplemental_audit']
        audit, ref = load_bound(child(directory,row), bindings, row['sha256'])
        require(audit.get('status') == 'PASS_SUPPLEMENTAL_COMPONENT_CERTIFICATE_AUDIT' and
                audit.get('component_summary_sha256') == cert_sha and
                all(audit.get(k) is True for k in ('graph_and_component_decisions_replayed',
                    'pair_matrix_complete_and_unique','rejected_nodes_preserved_and_propagated','strict_cross_root_window_disjointness')),
                'SUPPLEMENTAL_COMPONENT_AUDIT_NOT_PASS')
        require(all(audit['audited_files_sha256'].get(p) == h for p,h in component_bindings.items()),
                'SUPPLEMENTAL_AUDIT_NOT_BOUND_TO_COMPONENT_INPUTS')
        refs['supplemental_component_audits'].append(ref)
    schedule = check_frame_links(af, queries, events, admitted)
    require(check_frame_links(bf, queries, events, admitted) == schedule, 'OMP_SCHEDULE_SUMMARY_CHANGED')
    optional = {'status':'NOT_INCLUDED', 'visual_quality_improvement_claimed':False}
    if visibility is not None:
        vis, refs['visibility'] = load_bound(Path(visibility)/'summary.json', bindings)
        require(vis.get('schema') == 'forest-original-camera-visibility-v1' and
                vis.get('certification_sha256') == cert_sha and vis.get('sequence_sha256') in
                (refs['sequence_omp1']['sha256'],refs['sequence_omp8']['sha256']), 'VISIBILITY_PROVENANCE_MISMATCH')
        optional = {'status':vis['status'], 'artifact_scope':vis.get('scope'),
            'scope':'Hash-bound descriptive triage reference only; images and pixel metrics not re-audited by this JSON assembler.',
            'visual_quality_improvement_claimed':False}
    require(all(sha(p) == h for p,h in bindings.items()), 'INPUT_CHANGED_DURING_ASSEMBLY')
    return {'schema':'forest-schedule-structural-coherence-v1',
        'status':'PASS_REQUESTED_SCHEDULE_STRUCTURAL_COHERENCE',
        'scope':'Fixed Forest pre-displacement/raw five-element mesh; original 64 natural 24FPS queries plus two exact-root diagnostics.',
        'not_claimed':['all-real-time admission','post-displacement safety','six-scene generalization',
                       'visual quality improvement','warped SSIM improvement','pure production latency'],
        'population':{'canonical_events':131,'support_components':79,'jointly_admitted_events':25,
            'jointly_admitted_components':20,'fail_closed_events':106,'fail_closed_components':59,
            'event_admission_rate':25/131,'component_admission_rate':20/79,'per_root':per_root},
        'schedule':schedule, 'actual_delta_t_hex':cert['actual_delta_t_hex'],
        'atomic_and_identity_checks':{'omp1_omp8_all_66_receipts_identical':True,
            'canonical_records_and_source_owner_consumption_verified':True,
            'component_membership_all_or_none_verified':True,'outside_window_baseline_receipts_identical':True,
            'partial_output_published':False,'fresh_five_element_baseline_per_query':True,
            'complete_sequence_manifest_committed':True,'persistent_mesh_files':0,
            'evidence_boundary':'JSON receipts and hash-bound prior native/geometry audits, not a new array or geometry replay.'},
        'original_cache_binding':comparison.input_verification(a),
        'private_cache_removed_both_omp':True,'visibility':optional,
        'cost':{'component_certification':cert.get('cost'),'omp1_verified_sequence':a.get('cost'),
            'omp8_verified_sequence':b.get('cost'),
            'scope':'Measured verification harness costs; not pure production frame latency.'},
        'artifact_references':refs,'audited_json_file_count':len(bindings),
        'audited_json_files_sha256_digest':digest(bindings),'input_artifacts_unchanged':True,
        'executed_sources_sha256':{str(Path(__file__).resolve()):sha(__file__),
                                  str(Path(comparison.__file__).resolve()):sha(comparison.__file__)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('component','sequence-omp1','sequence-omp8','comparison','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--visibility',type=Path)
    args = parser.parse_args(); require(not args.output.exists(),'REFUSE_TO_OVERWRITE_MANIFEST')
    started = time.monotonic()
    try:
        report = assemble(args.component,args.sequence_omp1,args.sequence_omp8,args.comparison,args.visibility)
    except (ValueError,KeyError,TypeError,IndexError,ArithmeticError,OSError) as error:
        report = {'schema':'forest-schedule-structural-coherence-v1', 'status':'STOP_MANIFEST_INPUT_NOT_VERIFIED',
            'reason':type(error).__name__+': '+str(error),'structural_coherence_claimed':False}
    report['assembly_wall_seconds'] = time.monotonic()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({'status':report['status'],'output':str(args.output)}))
    return 0 if report['status']=='PASS_REQUESTED_SCHEDULE_STRUCTURAL_COHERENCE' else 2


if __name__ == '__main__': raise SystemExit(main())
