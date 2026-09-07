"""Final approved resource-only screen entry, copied from frozen run_screen.

Only experiment input wiring is new. The source selector, local/exact geometry
and component transaction are the frozen algorithms (dimension-adapted without
padding). Outputs stay staged until all input checks finish. No displacement,
RGB, quality-based sample selection or production source mutation happens here.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as F
import gc
from itertools import combinations
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from screen_contracts import (HERE, C1, ROOT, read, require, digest, file_sha, write_new,
    load_protocol, query_token, all_queries, validate_schedule_domain,
    read_source_stage, summarize_visibility, source_decision_counts)
from final_gate_resources import (PrivateSceneCache, verify_bindings, validated_budget,
    amendment_binding, amendment_inputs, validate_build_budget, require_remaining)
from forest_native_campaign import mesh_receipt, array_sha
from forest_schedule_policy import make_schedule
from run_forest_native_admission import validate_query_binding, decide_runtime
from run_forest_formal import message
from run_forest_component_certification import candidate_halo_owners, decide_components
from forest_component_graph import build_graph
from forest_sequence_inputs import bound_plan
from scene_native import initialize_scene
from scene_geometry import (spec_from_source, audit_query, actual_footprints,
    original_identity_receipt, compile_union, check_pair_interactions)
from run_forest_visibility_triage import compact_source, compare_buffers, write_images

import numpy as np


class Reports:
    def __init__(self, root):
        self.root = Path(root).resolve()
        require(not self.root.exists(), 'REFUSE_EXISTING_SCREEN_ATTEMPT')
        self.root.mkdir(parents=True)
        self.bytes = 0
        self.visibility_frame_artifacts = []

    def save(self, name, value):
        ref = write_new(self.root/name, value)
        self.bytes += (self.root/name).stat().st_size
        require(self.bytes <= 128*1024**2, 'SCREEN_COMPACT_REPORT_BUDGET_EXCEEDED')
        return {'path': name, 'sha256': ref['sha256']}


def bound_read(root, ref, bindings=None):
    path = Path(ref['path'])
    if not path.is_absolute():
        path = Path(root)/path
    require(file_sha(path) == ref['sha256'], 'BOUND_ARTIFACT_CHANGED:'+str(path))
    if bindings is not None:
        bindings[str(path.resolve())] = ref['sha256']
    return read(path)


def read_visibility_receipts(root, refs, events, components, segment, rule, bindings):
    """Recompute visibility counts from hash-bound rows, without re-rasterizing."""
    require(isinstance(refs, list), 'VISIBILITY_FRAME_RECEIPTS_REQUIRED')
    expected = {}
    for event in events:
        for query in event['schedule']['natural']:
            if query['active']:
                expected.setdefault(query['key'], query)
    seen, rows, image_paths = set(), [], set()
    for ref in refs:
        query = ref['query']; key = query['key']
        require(key in expected and key not in seen, 'DUPLICATE_OR_UNEXPECTED_VISIBILITY_RECEIPT')
        seen.add(key)
        row = bound_read(root, ref['artifact'], bindings)
        require(row['query'] == query, 'VISIBILITY_RECEIPT_QUERY_CHANGED')
        required = expected[key]
        require(all(query[k] == required[k] for k in
                    ('kind', 'time_mode', 'evaluation_tau', 'physical_time_hex', 'frame_index_zero_based'))
                and query['absolute_frame_number'] == segment['first_frame']+required['frame_index_zero_based'],
                'VISIBILITY_RECEIPT_SCHEDULE_CHANGED')
        for image in row.get('images', []):
            path = Path(image['path'])
            if not path.is_absolute():
                path = Path(root)/path
            path = path.resolve()
            require(str(path) not in image_paths, 'DUPLICATE_VISIBILITY_IMAGE_RECEIPT')
            image_paths.add(str(path))
            require(file_sha(path) == image['sha256'], 'VISIBILITY_IMAGE_CHANGED:'+str(path))
            bindings[str(path)] = image['sha256']
        rows.append(row)
    require(seen == set(expected), 'INCOMPLETE_VISIBILITY_FRAME_RECEIPTS')
    return rows, summarize_visibility(events, components, rows, segment, rule)

def executed_sources():
    names = ('final_gate_screen.py', 'final_gate_source.py', 'final_gate_resources.py',
             'final_gate_amendment.py', 'screen_contracts.py', 'screen_resources.py',
             'scene_native.py', 'scene_geometry.py', 'scene_source.py', 'run_screen.py')
    # Freeze executed adapters, not concurrently developed unexecuted pilot/tests.
    return {str((HERE/name).resolve()): file_sha(HERE/name) for name in names}


def validate_inputs(args):
    amendment = validated_budget(args.budget_amendment, args.protocol, args.seal, args.segment)
    require_remaining(amendment)
    budget_binding = amendment_binding(args.budget_amendment)
    budget_inputs = amendment_inputs(args.budget_amendment, amendment, args.protocol, args.seal)
    budget_inputs.update(validate_build_budget(args.build_report, args.budget_amendment,
                                               amendment, args.protocol, args.segment))
    protocol, segment = load_protocol(args.protocol, args.segment)
    seal = read(args.seal)
    require(seal['status'] == 'SEALED_BEFORE_NEW_SEGMENT_EXPERIMENTS'
            and seal['protocol_sha256'] == file_sha(args.protocol), 'PREREGISTRATION_SEAL_MISMATCH')
    verify_bindings(seal['input_and_method_sha256'])
    build = read(args.build_report/'summary.json')
    require(build['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'
            and build['segment_id'] == args.segment, 'NEW_SCENE_BUILD_INCOMPLETE')
    complete = read(args.build_report/'worker_complete.json')
    require(complete['source_inputs_unchanged'] and complete['status'] == build['status'], 'COARSE_INPUT_INTEGRITY_FAILED')
    camera = read(args.build_report/'camera_inputs.json')
    effective = read(args.build_report/'effective_inputs.json')
    image_camera = read(args.build_report/'original_render_camera.json')
    for name in ('camera_inputs', 'effective_inputs', 'original_render_camera'):
        require(file_sha(args.build_report/(name+'.json')) == complete[name+'_sha256'], 'BUILD_INPUT_HASH_MISMATCH:'+name)
    require(effective['segment_id'] == segment['segment_id']
            and effective['frame_range'] == [segment['first_frame'], segment['last_frame']]
            and effective['geometry_stage'] == 'PRE_SURFACE_DISPLACEMENT_OPAQUE', 'BUILD_SEGMENT_SCOPE_MISMATCH')
    require(image_camera['intrinsics_role'] == 'UNRELAXED_IMAGE_CAMERA'
            and image_camera['poses'] == camera['poses'], 'ORIGINAL_CAMERA_POSE_MISMATCH')
    source, events, inventory, bindings = read_source_stage(args.source)
    require(source.get('budget_amendment') == budget_binding, 'SOURCE_BUDGET_AMENDMENT_MISMATCH')
    require(source['resource_limits']['input_cache_bytes']
            == amendment['effective_budgets']['cache_per_scene_output_bytes'], 'SOURCE_CACHE_BUDGET_MISMATCH')
    bindings.update(source['amendment_input_sha256'])
    bindings.update(budget_inputs)
    require(source['segment_id'] == args.segment and Path(source['cache_root']).resolve() == Path(complete['cache']).resolve(),
            'SOURCE_BUILD_CACHE_MISMATCH')
    for name, value in (('camera_binding', camera), ('segment_spec_binding', segment)):
        binding = source[name]
        expected = digest(value) if binding.get('kind') == 'CANONICAL_JSON' else file_sha(binding['path'])
        require(binding['sha256'] == expected, 'SOURCE_DOCUMENT_BINDING_CHANGED:'+name)
        if binding.get('kind') != 'CANONICAL_JSON':
            require(read(binding['path']) == value, 'SOURCE_DOCUMENT_CONTENT_MISMATCH:'+name)
            bindings[str(Path(binding['path']).resolve())] = expected
    bindings.update(source.get('cache_completion_provenance_sha256', {}))
    cache = Path(complete['cache']).resolve()
    require(file_sha(cache/'event_registry_p1.csv') == source['registry_sha256'], 'SOURCE_REGISTRY_CHANGED')
    bindings.update(seal['input_and_method_sha256'])
    for p in (args.protocol, args.seal, args.build_report/'summary.json',
              args.build_report/'worker_complete.json', args.build_report/'camera_inputs.json',
              args.build_report/'effective_inputs.json', args.build_report/'original_render_camera.json'):
        bindings[str(p.resolve())] = file_sha(p)
    verify_bindings(bindings)
    return protocol, segment, camera, effective, image_camera, cache, source, events, inventory, bindings


def initialize(protocol, which, private, cache, camera, effective):
    native = protocol['frozen_native']
    return initialize_scene(native[which+'_build_repo'], private, camera, effective,
                            original_cache=cache, expected_so_sha256=native[which+'_sha256'])


def slice_bound(reader, query, baseline=None, *, ledger=True):
    value = F(query['evaluation_tau']) if query['time_mode'] == 'exact' else float.fromhex(query['physical_time_hex'])
    snapshot = reader.slice_query(value, mode=query['time_mode'], ledger=ledger)
    validate_query_binding(query, snapshot)
    receipt = mesh_receipt(snapshot['meshes'])
    if baseline is not None:
        require(receipt == baseline, 'FROZEN_ORDINARY_BASELINE_PARITY_FAILED:'+query['key'])
    if ledger:
        original_identity_receipt(snapshot)
    return snapshot, receipt


def reference_worker(args):
    """Own process/native globals, original library, no observer or proposals."""
    protocol, segment, camera, effective, _, cache, _, events, _, bindings = validate_inputs(args)
    code = executed_sources()
    amendment = validated_budget(args.budget_amendment, args.protocol, args.seal, args.segment)
    require_remaining(amendment)
    reader = initialize(protocol, 'ordinary', args.cache_copy, cache, camera, effective)
    started = time.monotonic()
    try:
        queries = all_queries(camera, segment, reader.delta_t, events)
        rows = []
        for query in queries:
            require_remaining(amendment)
            snapshot, receipt = slice_bound(reader, query, ledger=False)
            rows.append({'query': query, 'baseline': receipt})
            del snapshot
        verify_bindings(bindings); verify_bindings(code)
        require_remaining(amendment)
        result = {'status': 'PASS_ORIGINAL_ORDINARY_REFERENCE_SEQUENCE', 'segment_id': args.segment,
                  'protocol_sha256': file_sha(args.protocol), 'source_summary_sha256': file_sha(args.source/'summary.json'),
                  'library_sha256': protocol['frozen_native']['ordinary_sha256'], 'initialization': reader.initialization,
                  'queries': rows, 'identity_enabled': False, 'code_sha256': code,
                  'budget_amendment': amendment_binding(args.budget_amendment),
                  'wall_seconds': time.monotonic()-started}
        write_new(args.reference_output, result)
    finally:
        reader.close()
    return 0


def baseline_references(args, reports, private, reader, protocol, events, camera, segment):
    output = reports.root/'ordinary_reference.json'
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--reference-worker',
               '--protocol', str(args.protocol.resolve()), '--segment', args.segment,
               '--seal', str(args.seal.resolve()), '--build-report', str(args.build_report.resolve()),
               '--budget-amendment', str(args.budget_amendment.resolve()),
               '--source', str(args.source.resolve()), '--cache-copy', str(private.path),
               '--reference-output', str(output)]
    with (reports.root/'ordinary_reference.log').open('x', encoding='utf-8') as log:
        amendment = validated_budget(args.budget_amendment, args.protocol, args.seal, args.segment)
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=min(900, require_remaining(amendment)))
    require(completed.returncode == 0, 'ORIGINAL_REFERENCE_WORKER_FAILED_SEE_LOG')
    document = read(output)
    require(document['status'] == 'PASS_ORIGINAL_ORDINARY_REFERENCE_SEQUENCE'
            and document['segment_id'] == args.segment
            and document['protocol_sha256'] == file_sha(args.protocol)
            and document['source_summary_sha256'] == file_sha(args.source/'summary.json')
            and document['library_sha256'] == protocol['frozen_native']['ordinary_sha256']
            and document['initialization']['actual_delta_t_hex'] == reader.delta_t.hex()
            and document['identity_enabled'] is False, 'ORIGINAL_REFERENCE_BINDING_FAILED')
    require(document.get('budget_amendment') == amendment_binding(args.budget_amendment),
            'ORIGINAL_REFERENCE_AMENDMENT_BINDING_FAILED')
    require(document.get('code_sha256') == executed_sources(), 'ORIGINAL_REFERENCE_CODE_BINDING_FAILED')
    expected = {q['key']: query_token(q) for q in all_queries(camera, segment, reader.delta_t, events)}
    require({r['query']['key']: query_token(r['query']) for r in document['queries']} == expected
            and len(document['queries']) == len(expected), 'REFERENCE_QUERY_DENOMINATOR_CHANGED')
    return {r['query']['key']: r['baseline'] for r in document['queries']}, len(document['queries'])


def pair_proofs(meshes, records, footprints):
    result = []
    for first, second in combinations(sorted(records, key=lambda r: r['event_id']), 2):
        a, b = footprints[first['event_id']], footprints[second['event_id']]
        shared = (a['element'] == b['element'] and
                  (set(a['boundary_actual_ids']) & set(b['boundary_actual_ids'])
                   or set(a['source_face_rows']) & set(b['source_face_rows'])))
        shared = bool(shared or set(map(tuple, a['consumed_owners'])) & set(map(tuple, b['consumed_owners'])))
        if not shared and (np.any(np.asarray(a['bounds'][0]) > np.asarray(b['bounds'][1]))
                           or np.any(np.asarray(b['bounds'][0]) > np.asarray(a['bounds'][1]))):
            proof = {'status': 'PASS_PAIR_INTERACTIONS', 'proof': 'STRICT_ACTUAL_BINARY32_SUPPORT_AABB',
                     'triangle_pairs_excluded': 32}
        else:
            proof = check_pair_interactions(meshes, [first, second])
        require(proof['status'] != 'UNKNOWN_INPUT_OR_PROOF', 'UNEXPECTED_PAIR_INPUT_OR_PROOF_FAILURE:'+str(proof))
        result.append({'events': [first['event_id'], second['event_id']], 'report': proof})
    return result


def requests_for(events, inventory, specs, root, tau):
    result = []
    for event in events:
        if event['root'] != root:
            continue
        eid = event['event_id']
        row = {'event_id': eid, 'element': event['element'], 'spec': specs.get(eid)}
        if eid not in specs:
            row['candidate_actual_owners'] = candidate_halo_owners(inventory[eid], tau)
        result.append(row)
    return result


def certify(reader, events, inventory, specs, reference, reports, segment, checkpoint):
    by_query = {}
    for event in events:
        if event['source_status'] == 'SOURCE_READY':
            event['runtime'] = {'status': 'PREPARING', 'cases': [], 'published_partial_results': False}
        for query in event['schedule']['all_queries']:
            by_query.setdefault(query['key'], {'query': query, 'events': []})['events'].append(event)
    rows, calls = [], 0
    for key, group in sorted(by_query.items(), key=lambda kv: (kv[1]['query']['kind'] != 'exact_root', F(kv[1]['query']['evaluation_tau']))):
        checkpoint(); query = group['query']; tau = F(query['evaluation_tau'])
        snapshot, baseline = slice_bound(reader, query, reference[key]); calls += 1
        active_roots = {event['root'] for event in group['events']
                        if next(q for q in event['schedule']['all_queries'] if q['key'] == key)['active']}
        require(len(active_roots) <= 1, 'UNSUPPORTED_COACTIVE_ROOTS')
        footprints = []
        if active_roots:
            root = next(iter(active_roots))
            footprints = actual_footprints(snapshot, requests_for(events, inventory, specs, root, tau), tau)
        for event in group['events']:
            if event['source_status'] != 'SOURCE_READY' or event['decision'] == 'REJECTED_FIXED_POLICY':
                continue
            requested = next(q for q in event['schedule']['all_queries'] if q['key'] == key)
            event['runtime_attempted'] = True
            if not requested['active']:
                case = {'query': requested, 'audit': {'status': 'BASELINE_OUTSIDE_WINDOW'},
                        'baseline': baseline, 'output_equals_baseline': True}
            else:
                plan, audit = audit_query(snapshot, specs[event['event_id']], tau,
                                          full_exterior=True, exact_contact_fallback=True)
                require(not (audit['status'] == 'UNKNOWN' and audit.get('exception_type')),
                        'UNEXPECTED_NATIVE_AUDIT_EXCEPTION:'+str(audit))
                case = {'query': requested, 'audit': audit}
                if audit['status'] == 'PASS':
                    require(plan is not None, 'NATIVE_PASS_WITHOUT_PLAN')
                    case['plan'] = plan
            event['runtime']['cases'].append(case)
            decide_runtime(event)
        records = []
        by_id = {row['event_id']: row for row in footprints}
        for event in group['events']:
            if event['event_id'] not in by_id or event['source_status'] != 'SOURCE_READY':
                continue
            case = next((c for c in event['runtime']['cases'] if c['query']['key'] == key), None)
            if case and case['audit']['status'] == 'PASS':
                records.append({'event_id': event['event_id'], 'component_id': 'PROVISIONAL-'+event['event_id'], 'plan': case['plan']})
        pairs = pair_proofs(snapshot['meshes'], records, by_id) if active_roots else []
        require(mesh_receipt(snapshot['meshes']) == baseline, 'CERTIFICATION_MUTATED_BASELINE')
        row = {'query': query, 'root': next(iter(active_roots)) if active_roots else None,
               'baseline': baseline, 'events': footprints, 'pair_proofs': pairs,
               'identity_encoding': original_identity_receipt(snapshot), 'native_cost': snapshot['cost']}
        rows.append(row)
        reports.save('certification_queries/'+key+'.json', row)
        message('scene_actual_query_certified', segment=segment['segment_id'], query=key,
                events=len(footprints), current_decisions=dict(Counter(e['decision'] for e in events)))
        del snapshot; gc.collect()
    for event in events:
        if event.get('runtime', {}).get('status') == 'PREPARED_ENTIRE_REQUESTED_SCHEDULE':
            event['decision'] = 'ADMITTED_REQUESTED_SCHEDULE'
            event['runtime']['status'] = 'COMMITTED_REQUESTED_SCHEDULE'
            event['runtime']['publication_scope'] = 'CERTIFIED_PLAN_ONLY_SCENE_SEQUENCE_NOT_YET_PUBLISHED'
    admitted = {e['event_id'] for e in events if e['decision'] == 'ADMITTED_REQUESTED_SCHEDULE'}
    active_rows = [row for row in rows if row['root'] is not None]
    for row in active_rows:
        row['pair_proofs'] = [pair for pair in row['pair_proofs'] if set(pair['events']) <= admitted]
        k = sum(e['event_id'] in admitted and e['root'] == row['root'] for e in events)
        require(len(row['pair_proofs']) == k*(k-1)//2, 'FINAL_ADMITTED_PAIR_DENOMINATOR_INCOMPLETE')
    return active_rows, calls


def make_graph(events, queries, inventory):
    if not events:
        return {'status': 'COMPLETE_ZERO_REGISTRY_SUPPORT_GRAPH', 'event_count': 0,
                'components': [], 'edges': [], 'root_counts': {}, 'unknown_actual_support_rows': 0}
    symbolic = list(inventory.get('symbolic_edges', []))
    for row in queries:
        for pair in row['pair_proofs']:
            if pair['report']['status'] != 'PASS_PAIR_INTERACTIONS':
                symbolic.append({'events': pair['events'], 'reason': 'ACTUAL_JOINT_PAIR_NOT_CERTIFIED', 'cell': row['query']['key']})
    graph = build_graph(events, queries, symbolic_edges=symbolic)
    decide_components(graph, queries)
    return graph


def render_visibility(snapshot, outputs, union, query, footprints, components, segment, reports):
    from forest_raster import render_scene
    camera = reports.image_camera
    by_event = {eid: c for c in components for eid in c['events']}
    numbers = {c['component_id']: i for i, c in enumerate(sorted(components, key=lambda c: c['component_id']))}
    labels = {e: np.full(len(mesh[1]), -1, np.int64) for e, mesh in enumerate(snapshot['meshes'])}
    for row in footprints:
        if row['status'] != 'COMPLETE_ACTUAL_REQUESTED_SUPPORT':
            continue
        cid = numbers[by_event[row['event_id']]['component_id']]
        old = labels[row['element']][row['source_face_rows']]
        require(np.all((old == -1) | (old == cid)), 'SOURCE_FACE_LABEL_CROSSES_COMPONENTS')
        labels[row['element']][row['source_face_rows']] = cid
    out_labels = {e: np.concatenate((labels[e], np.full(len(mesh[1])-len(labels[e]), -1, np.int64)))
                  for e, mesh in enumerate(outputs)}
    replacement_labels = {e: np.full(len(mesh[1]), -1, np.int64) for e, mesh in enumerate(outputs)}
    for mapping in union['event_mapping'].values():
        number = numbers[mapping['component_id']]
        out_labels[mapping['element']][mapping['fan_face_rows_by_sector']] = number
        replacement_labels[mapping['element']][mapping['fan_face_rows_by_sector']] = number
    index = query['frame_index_zero_based']
    raw = render_scene(snapshot['meshes'], camera, index, labels)
    changed = render_scene(outputs, camera, index, out_labels)
    counts, out_counts = {}, {}
    for element in range(len(outputs)):
        for target, buffer in ((counts, raw), (out_counts, changed)):
            ids, ns = np.unique(buffer['face_id'][buffer['element_id'] == element], return_counts=True)
            target[element] = dict(zip(map(int, ids), map(int, ns)))
    replacement_pixel_labels = np.full(raw['mask'].shape, -1, np.int64)
    for element in range(len(outputs)):
        mask = changed['element_id'] == element
        replacement_pixel_labels[mask] = replacement_labels[element][changed['face_id'][mask]]
    event_rows = []
    for row in footprints:
        eid = row['event_id']; component = by_event[eid]; ele = row['element']
        mapping = union['event_mapping'].get(eid)
        visible_out = sum(out_counts[ele].get(i, 0) for i in mapping['fan_face_rows_by_sector']) if mapping else 0
        pixels, isolated = None, None
        if row['status'] == 'COMPLETE_ACTUAL_REQUESTED_SUPPORT':
            pixels = sum(counts[ele].get(i, 0) for i in row['source_face_rows'])
            patch = render_scene([compact_source(snapshot['meshes'][ele], row['source_face_rows'])], camera, index)
            isolated = int(patch['mask'].sum()); del patch
        event_rows.append({'event_id': eid, 'component_id': component['component_id'],
                           'jointly_admitted': component['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE',
                           'domain_kind': row.get('kind', 'UNKNOWN_SUPPORT'),
                           'visible_baseline_source_pixels': pixels, 'unoccluded_source_pixels': isolated,
                           'visible_replacement_pixels': visible_out,
                           'visibility': 'UNKNOWN_SUPPORT' if pixels is None else
                           ('VISIBLE_SOURCE_SUPPORT' if pixels else ('OCCLUDED_OR_DEPTH_TIE_NOT_SELECTED' if isolated else 'NO_ISOLATED_PIXEL_COVERAGE'))})
    component_rows = []
    present = {by_event[row['event_id']]['component_id'] for row in footprints}
    for component in components:
        cid = component['component_id']
        if cid not in present:
            continue
        number = numbers[cid]
        roi = (raw['label_id'] == number) | (changed['label_id'] == number)
        component_rows.append({'component_id': cid, 'decision': component['decision'],
                               'visible_baseline_union_pixels': int(np.count_nonzero(raw['label_id'] == number)),
                               'visible_replacement_union_pixels': int(np.count_nonzero(replacement_pixel_labels == number)),
                               'metrics': compare_buffers(raw, changed, roi)})
    row = {'query': query, 'events': event_rows, 'components': component_rows,
           'global_metrics': compare_buffers(raw, changed),
           'buffer_sha256': {name: {k: array_sha(buffers[k]) for k in
                                   ('depth', 'normals', 'mask', 'element_id', 'face_id', 'label_id')}
                             for name, buffers in (('baseline', raw), ('combined', changed))},
           'geometry_difference_is_not_quality_improvement': True,
           **write_images(reports, query['key'], raw, changed)}
    ref = reports.save('visibility/'+query['key']+'.json', row)
    reports.visibility_frame_artifacts.append({'query': query, 'artifact': ref})
    message('scene_visibility_frame', segment=segment['segment_id'], absolute_frame=query['absolute_frame_number'],
            visible_source_events=sum((r['visible_baseline_source_pixels'] or 0) > 0 for r in event_rows),
            replacement_pixels=sum(r['visible_replacement_pixels'] for r in event_rows))
    return row


def execute_sequence(reader, camera, segment, events, inventory, specs, graph, certified, reference,
                     reports, checkpoint, *, primary=None):
    admitted = [c for c in graph['components'] if c['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE']
    event_by_id = {e['event_id']: e for e in events}
    certified_by_key = {q['query']['key']: q for q in certified}
    queries = all_queries(camera, segment, reader.delta_t, events)
    frames, images, changed_frames, calls = [], [], [], 0
    replacements = 0
    for query in queries:
        checkpoint(); key = query['key']; tau = F(query['evaluation_tau'])
        snapshot, baseline = slice_bound(reader, query, reference[key]); calls += 1
        cq = certified_by_key.get(key)
        footprints, records = [], []
        if cq is not None:
            footprints = actual_footprints(snapshot, requests_for(events, inventory, specs, cq['root'], tau), tau)
            require(digest(footprints) == digest(cq['events']), 'COMPLETE_ROOT_SUPPORT_REPLAY_CHANGED:'+key)
            by_id = {row['event_id']: row for row in footprints}
            for component in admitted:
                if component['root'] != cq['root']:
                    continue
                for eid in component['events']:
                    plan = bound_plan(event_by_id[eid], query, baseline, specs[eid],
                                      {**by_id[eid], 'certified_baseline': baseline})
                    records.append({'event_id': eid, 'component_id': component['component_id'], 'plan': plan})
        records.sort(key=lambda r: r['event_id'])
        membership = {c['component_id']: c['events'] for c in admitted if cq and c['root'] == cq['root']}
        outputs, union = compile_union(snapshot['meshes'], records, expected_components=membership)
        require(outputs is not None and union['status'] == 'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED',
                'ATOMIC_SEQUENCE_UNION_FAILED:'+str(union))
        consumed = {r['event_id'] for r in records}
        for footprint in footprints:
            if footprint['event_id'] in consumed or footprint['status'] != 'COMPLETE_ACTUAL_REQUESTED_SUPPORT':
                continue
            ele = footprint['element']; ids = footprint['source_face_rows']
            require(np.array_equal(outputs[ele][1][ids], snapshot['meshes'][ele][1][ids]), 'FALLBACK_SOURCE_FACES_CHANGED')
        frame = {'query': query, 'baseline': baseline, 'records': records, 'union': union,
                 'root_support_sha256': digest(footprints) if cq else None,
                 'full_root_population_replayed': len(footprints), 'fallback_source_faces_unchanged': True}
        if primary is not None:
            expected = primary[key]
            for field in ('query', 'baseline', 'records', 'union', 'root_support_sha256', 'full_root_population_replayed'):
                require(frame[field] == expected[field], 'OMP_SEQUENCE_DIFFERENCE:'+key+':'+field)
        refs = reports.save('frames/'+key+'.json', frame)
        frames.append({'query': query, 'artifact': refs})
        if query['kind'] == 'natural' and union['output'] != baseline:
            changed_frames.append(query['absolute_frame_number'])
            replacements += len(records)
        if primary is None and query['kind'] == 'natural' and cq is not None:
            images.append(render_visibility(snapshot, outputs, union, query, footprints, graph['components'], segment, reports))
        message('scene_atomic_frame', segment=segment['segment_id'], query=key, replacements=len(records),
                omp=os.environ.get('OMP_NUM_THREADS'), output_equal_baseline=union['output'] == baseline)
        del snapshot, outputs; gc.collect()
    return frames, images, changed_frames, replacements, calls


def run(args):
    reports = Reports(args.output)
    started = time.monotonic(); cpu = time.process_time(); calls = 0
    private = reader = None
    summary = {'status': 'STOP_SCREEN_ATTEMPT_NOT_COMPLETE', 'segment_id': args.segment,
               'admission_rate': None, 'partial_output_published': False}
    primary_summary = None
    try:
        summary.update(protocol_sha256=file_sha(args.protocol), preregistration_seal_sha256=file_sha(args.seal),
                       supplied_input_hashes_do_not_imply_validation=True)
        protocol, segment, camera, effective, image_camera, cache, source, events, inventory, bindings = validate_inputs(args)
        summary['budget_amendment'] = amendment_binding(args.budget_amendment)
        amendment = validated_budget(args.budget_amendment, args.protocol, args.seal, args.segment)
        summary['effective_budgets'] = amendment['effective_budgets']
        summary['campaign_deadline_utc'] = amendment['campaign_deadline_utc']
        require(os.environ.get('OMP_NUM_THREADS') == ('8' if args.replay_from else '1'), 'PREREGISTERED_OMP_CONFIGURATION_REQUIRED')
        summary['source'] = {'canonical_events': len(events), 'exact_roots': len({e['root'] for e in events}),
                             'raw_observations': source['raw_registry_observations'], 'source_decision_counts': source_decision_counts(events)}
        code = executed_sources(); bindings.update(code)
        reports.image_camera = image_camera
        def checkpoint():
            require_remaining(amendment)
            require(time.monotonic()-started <= protocol['budgets']['certification_and_screen_per_scene_wall_seconds'], 'SCREEN_WALL_BUDGET')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= protocol['budgets']['native_query_peak_rss_bytes'], 'SCREEN_RSS_BUDGET')
            actual_bytes = sum(p.stat().st_size for p in reports.root.rglob('*') if p.is_file())
            require(actual_bytes <= 124*1024**2, 'SCREEN_ACTUAL_REPORT_BYTES_EXCEED_RESERVED_BUDGET')
            artifact_root = args.seal.resolve().parent
            campaign_bytes = sum(p.stat().st_size for p in artifact_root.rglob('*') if p.is_file())
            require(campaign_bytes <= protocol['budgets']['maximum_compact_reports_and_images_bytes'], 'CAMPAIGN_COMPACT_ARTIFACT_BUDGET')
        private = PrivateSceneCache(cache, budget_amendment=args.budget_amendment,
                                    protocol=args.protocol, seal=args.seal, segment_id=args.segment)
        reader = initialize(protocol, 'observer', private.path, cache, camera, effective)
        initialization = reader.initialization
        for event in events:
            bounds = [F(event['schedule']['bounds'][k]) for k in ('lower', 'root', 'upper')]
            event['schedule'] = make_schedule(camera, *bounds, actual_delta=reader.delta_t)
            event['schedule_sha256'] = digest(event['schedule'])
        domain = validate_schedule_domain(events)
        specs = {e['event_id']: spec_from_source(e['compiler']['source'], reader.n_elements)
                 for e in events if e.get('compiler', {}).get('source')}
        inventory_rows = {e['event_id']: e for e in inventory['events']}
        if args.replay_from:
            primary_summary = read(args.replay_from/'summary.json')
            require(primary_summary['status'] == 'PASS_PREREGISTERED_SEGMENT_SCREEN_OMP1_ONLY'
                    and primary_summary['protocol_sha256'] == file_sha(args.protocol)
                    and primary_summary['preregistration_seal_sha256'] == file_sha(args.seal)
                    and primary_summary['segment_id'] == args.segment, 'PRIMARY_SCREEN_NOT_VERIFIED')
            require(primary_summary.get('budget_amendment') == amendment_binding(args.budget_amendment)
                    and primary_summary.get('effective_budgets') == summary['effective_budgets'],
                    'PRIMARY_SCREEN_AMENDMENT_MISMATCH')
            require(primary_summary['omp_num_threads'] == '1'
                    and primary_summary['final_input_verification']['status'] == 'PASS'
                    and primary_summary['private_cache_removed'] is True,
                    'PRIMARY_OMP_CONFIGURATION_OR_CLEANUP_NOT_VERIFIED')
            verify_bindings(primary_summary['input_and_code_sha256'])
            reference_doc = bound_read(args.replay_from, primary_summary['ordinary_reference'], bindings)
            require(reference_doc.get('budget_amendment') == amendment_binding(args.budget_amendment)
                    and reference_doc.get('code_sha256') == executed_sources(),
                    'REPLAY_ORIGINAL_REFERENCE_AMENDMENT_OR_CODE_MISMATCH')
            reference = {r['query']['key']: r['baseline'] for r in reference_doc['queries']}
            events = bound_read(args.replay_from, primary_summary['certified_events'], bindings)
            certified = bound_read(args.replay_from, primary_summary['component_queries'], bindings)
            graph = bound_read(args.replay_from, primary_summary['support_graph'], bindings)
            replay_graph = make_graph(events, certified, inventory)
            require(replay_graph == graph, 'PRIMARY_GRAPH_DECISION_REPLAY_CHANGED')
            primary = {r['query']['key']: bound_read(args.replay_from, r['artifact'], bindings)
                       for r in primary_summary['frame_artifacts']}
            _, primary_visibility = read_visibility_receipts(
                args.replay_from, primary_summary['visibility_frame_artifacts'], events,
                graph['components'], segment, protocol['visibility']['qualifying_component_rule'], bindings)
            require(primary_visibility == primary_summary['visibility'], 'PRIMARY_VISIBILITY_SUMMARY_REPLAY_CHANGED')
        else:
            require(os.environ.get('OMP_NUM_THREADS') == '1', 'INITIAL_SCREEN_REQUIRES_PREREGISTERED_OMP1')
            reference, reference_calls = baseline_references(args, reports, private, reader, protocol, events, camera, segment)
            calls += reference_calls
            certified, certification_calls = certify(reader, events, inventory_rows, specs, reference, reports, segment, checkpoint)
            calls += certification_calls
            verify_bindings(bindings); private.verify(calls)
            graph = make_graph(events, certified, inventory)
            primary = None
        graph_ref = reports.save('support_graph.json', graph)
        events_ref = reports.save('certified_events.json', events)
        certified_ref = reports.save('component_queries.json', certified)
        frame_refs, images, modified, replacement_count, frame_calls = execute_sequence(
            reader, camera, segment, events, inventory_rows, specs, graph, certified, reference,
            reports, checkpoint, primary=primary)
        calls += frame_calls
        checkpoint()
        reader.close(); reader = None
        verification = private.verify(calls); verify_bindings(bindings)
        admitted = [c for c in graph['components'] if c['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE']
        admitted_events = sum(len(c['events']) for c in admitted)
        visibility_refs = (primary_summary['visibility_frame_artifacts'] if primary_summary
                           else reports.visibility_frame_artifacts)
        visibility_base = args.replay_from if primary_summary else reports.root
        _, visibility = read_visibility_receipts(
            visibility_base, visibility_refs, events, graph['components'], segment,
            protocol['visibility']['qualifying_component_rule'], bindings)
        if primary_summary:
            require(visibility == primary_summary['visibility'], 'PRIMARY_VISIBILITY_SUMMARY_REPLAY_CHANGED')
        verify_bindings(bindings)
        require_remaining(amendment)
        summary.update(status='PASS_OMP8_SEQUENCE_REPLAY' if primary_summary else 'PASS_PREREGISTERED_SEGMENT_SCREEN_OMP1_ONLY',
            protocol_sha256=file_sha(args.protocol), preregistration_seal_sha256=file_sha(args.seal),
            input_and_code_sha256=bindings, initialization=initialization,
            source={'canonical_events': len(events), 'exact_roots': len({e['root'] for e in events}),
                    'raw_observations': source['raw_registry_observations'], 'source_decision_counts': source_decision_counts(events),
                    'individually_admitted_events': sum(e['decision'] == 'ADMITTED_REQUESTED_SCHEDULE' for e in events),
                    'unresolved_certificate_events': sum(e['decision'] == 'UNKNOWN' for e in events)},
            composition={'components': len(graph['components']), 'admitted_components': len(admitted),
                         'admitted_events': admitted_events, 'fallback_components': len(graph['components'])-len(admitted),
                         'fallback_events': len(events)-admitted_events,
                         'joint_event_admission_rate': admitted_events/len(events) if events else None,
                         'component_admission_rate': len(admitted)/len(graph['components']) if graph['components'] else None,
                         'unknown_actual_support_rows': graph['unknown_actual_support_rows']},
            schedule={'natural_frames': 64, 'modified_natural_frames': modified,
                      'natural_event_frame_replacements': replacement_count, 'query_count': len(frame_refs),
                      'exact_root_diagnostics': len(frame_refs)-64, 'root_schedule_domain': domain},
            visibility=visibility, omp_confirmation={'status': 'NOT_YET_CONFIRMED'} if not primary_summary else {'status': 'REPLAY_COMPLETED'},
            support_graph=graph_ref, certified_events=events_ref, component_queries=certified_ref, frame_artifacts=frame_refs,
            visibility_frame_artifacts=visibility_refs, visibility_artifacts_base=str(visibility_base.resolve()),
            original_cache_verification=verification, native_slice_calls=calls,
            output_publication='ALL_REQUESTED_QUERIES_VALIDATED; COMPACT_RECEIPTS_ONLY',
            whole_mesh_video_files=0, scope='Fixed pre-displacement 64-frame scene; no quality or all-time theorem.',
            omp_num_threads=os.environ.get('OMP_NUM_THREADS'))
        if primary_summary:
            summary['primary_summary'] = {'path': str((args.replay_from/'summary.json').resolve()), 'sha256': file_sha(args.replay_from/'summary.json')}
        else:
            summary['ordinary_reference'] = {'path': 'ordinary_reference.json', 'sha256': file_sha(reports.root/'ordinary_reference.json')}
    except Exception as error:
        summary.update(status='STOP_SCREEN_ATTEMPT_NOT_COMPLETE', reason=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        if reader is not None:
            reader.close()
        if private is not None:
            try:
                summary['final_input_verification'] = private.verify(calls)
            except Exception as error:
                summary.update(status='STOP_INPUT_VERIFICATION_FAILED', verification_error=str(error))
            private.remove(); summary['private_cache_removed'] = True
        summary['cost'] = {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
                           'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                           'scope': 'Research certification/playback/image/audit harness, not production speed.'}
        reports.save('summary.json', summary)
    if primary_summary and summary['status'] == 'PASS_OMP8_SEQUENCE_REPLAY':
        require(os.environ.get('OMP_NUM_THREADS') == '8', 'CONFIRMATION_REQUIRES_PREREGISTERED_OMP8')
        confirmation = {'status': 'PASS_OMP1_OMP8_IDENTICAL', 'segment_id': args.segment,
                        'protocol_sha256': file_sha(args.protocol), 'preregistration_seal_sha256': file_sha(args.seal),
                        'budget_amendment': amendment_binding(args.budget_amendment),
                        'omp1_summary': summary['primary_summary'],
                        'omp8_summary': {'path': str(reports.root/'summary.json'), 'sha256': file_sha(reports.root/'summary.json')},
                        'queries_compared': summary['schedule']['query_count'],
                        'all_arrays_records_component_membership_and_complete_root_supports_identical': True,
                        'no_total_wall_time_speedup_claim': True}
        confirmation_ref = write_new(reports.root/'omp_confirmation.json', confirmation)
        combined = {**primary_summary, 'status': 'PASS_PREREGISTERED_SEGMENT_SCREEN',
                    'omp_confirmation': {'status': confirmation['status'], 'artifact': confirmation_ref},
                    'omp1_attempt_root': str(args.replay_from.resolve()),
                    'evidence_scope': 'OMP1 complete screen plus OMP8 actual-array/support replay. Visibility images inherited with hash-bound source frames.'}
        write_new(reports.root/'combined_summary.json', combined)
    message('registered_scene_screen_summary', **{k: summary.get(k) for k in ('status', 'segment_id', 'composition', 'reason')})
    return 0 if summary['status'].startswith('PASS_') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'seal', 'budget-amendment', 'build-report', 'source'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--segment', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay-from', type=Path)
    parser.add_argument('--reference-worker', action='store_true')
    parser.add_argument('--cache-copy', type=Path)
    parser.add_argument('--reference-output', type=Path)
    args = parser.parse_args()
    if args.reference_worker:
        require(args.cache_copy is not None and args.reference_output is not None, 'REFERENCE_WORKER_ARGUMENTS')
        return reference_worker(args)
    require(args.output is not None, 'FRESH_OUTPUT_REQUIRED')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
