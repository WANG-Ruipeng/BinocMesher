#!/usr/bin/env python3
"""Read-only Forest131 real-source funnel, with mandatory frozen E2 parity.

Outputs compact source contracts/witnesses. Native admission is deliberately
not fabricated by this stage: surviving events are handed to a later actual
requested-schedule adapter. All decisive rejections require a named fixed
construction gate and complete reader/compiler evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from fractions import Fraction as F
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'source_splice'))
from processed_mesh import SourceVID, HVID
import compile_critical_beb1_event_ir as old_ir
from forest_support_reader import read_event_candidates
from forest_source_compiler import compile_kernel, compile_event, required_times, levels, select_existing
from forest_schedule_policy import make_schedule, summarize_decisions
from forest_requested_source_gates import audit_requested_source
from window_source import fr

REGISTRY_SHA = '0ce1a439bf33b785db9dbb3ea55dbc548933477aca64d2fca64829f660609b52'
E2_ID = 'element=0;role=temporal_neighbour;face=46:10|311:0|159:8|391:8'
E2_SOURCE_SHA = '6b7f08fb7689ef8e62002c787e301a321fb343cda16d1ce37c54021b58c92dc2'
REPORT_BUDGET = 100*1024*1024


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(stable(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


class Reports:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if self.root.exists():
            raise ValueError('Refusing to overwrite an existing attempt directory.')
        self.root.mkdir(parents=True)
        self.bytes = 0

    def save(self, name, value):
        data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
        self.bytes += len(data.encode())
        if self.bytes > REPORT_BUDGET:
            raise MemoryError('Compact campaign reports exceeded 100 MiB.')
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(data)
        return {'path': str(path.relative_to(self.root)), 'sha256': file_sha(path)}


def message(stage, **values):
    print(json.dumps({'stage': stage, **values}, sort_keys=True), flush=True)


def reader_binding(dataset):
    # Full opened primary/BPM2/HV/registry hashes, plus exact selected BHP2
    # provenance. This is explicitly NOT the old <=256 MiB runtime digest.
    return sha({'full_files': {str(p.relative_to(dataset.cache)): h for p, h in dataset.reader.full_hashes.items()},
                'source_provenance': [(list(k), list(v)) for k, v in sorted(dataset._provenance_checked.items())]})


def source_projection(source):
    return {'levels': source['levels'], 'boundary_cycle': source['boundary_cycle'],
        'anchors': {k: {'position': v['position'], 'source_diagonal': v.get('source_diagonal'),
                        'binary32_words': v.get('binary32_words')} for k, v in source['anchors'].items()},
        'points': [{k: p[k] for k in ('time', 'source_faces', 'owners', 'boundary_cycle', 'boundary')}
                   for p in source['breakpoint_points']],
        'segments': [{**{k: p[k] for k in ('t0', 't1', 'source_faces', 'owners', 'boundary_cycle')},
                      'boundary': [{k: b[k] for k in ('source_vid', 'position_t0', 'position_t1', 'intercept', 'slope')}
                                   for b in p['boundary']]} for p in source['segments']]}


def e2_regression(cache, frozen_path):
    start = time.monotonic()
    if file_sha(frozen_path) != E2_SOURCE_SHA:
        raise ValueError('Frozen E2 source input differs from the declared baseline.')
    frozen = json.loads(Path(frozen_path).read_text())
    dataset = read_event_candidates(cache, [E2_ID])
    kernel = compile_kernel(dataset.registry[E2_ID], dataset.hypervertices)
    if kernel['status'] != 'PASS_EXISTING_LOCAL_KERNEL':
        raise ValueError('E2 local kernel adapter failed: '+json.dumps(kernel))
    root = dataset.roots[E2_ID]
    snapshots = dataset.load_halos({E2_ID: required_times(root, kernel['corner_times'])})
    comparisons, boundary = [], None
    for name, tau in zip(('lower', 'critical', 'upper'), levels(root, kernel['corner_times'])):
        full, full_runtime = old_ir.compile_ordinary_patch(Path(cache), tau, boundary, E2_ID)
        halo, halo_runtime = select_existing(snapshots, E2_ID, tau, boundary)
        fields = ('time', 'element', 'boundary_cycle', 'boundary_segments', 'boundary_positions',
                  'boundary_in_view', 'source_faces', 'raw_suppression_count')
        full_projection, halo_projection = ({k: item[k] for k in fields} for item in (full, halo))
        full_owners = sorted(x.values() for x in full_runtime['suppressions'])
        halo_owners = sorted(x.values() for x in halo_runtime['suppressions'])
        if full_projection != halo_projection or full_owners != halo_owners:
            raise ValueError('E2 adapter differs from existing full parser at '+name)
        comparisons.append({'name': name, 'tau': str(tau), 'status': 'PASS_EXACT_FIELDS_AND_OWNERS',
            'projection': full_projection, 'owners': full_owners, 'snapshot': snapshots[E2_ID, tau].summary()})
        if boundary is None:
            boundary = frozenset(full_runtime['cycle'])
    compiled = compile_event(E2_ID, root, kernel['corner_times'], snapshots, dataset.hypervertices,
        kernel['critical_position'], cache_digest=reader_binding(dataset), group_count=dataset.groups,
        maximum_discrete_time=dataset.maximum, element=0)
    if compiled['status'] != 'PASS_SOURCE_COMPILER':
        raise ValueError('E2 source adapter failed: '+json.dumps(compiled))
    if source_projection(compiled['source']) != source_projection(frozen):
        raise ValueError('E2 exact source trajectories/owners/anchors differ from frozen contract.')
    verification = dataset.verify_inputs()
    return {'status': 'PASS_FULL_PARSER_AND_FROZEN_E2_PARITY', 'event_id': E2_ID,
        'wall_seconds': time.monotonic()-start, 'frozen_source_sha256': E2_SOURCE_SHA,
        'support_comparisons': comparisons, 'source_semantics_sha256': sha(source_projection(frozen)),
        'source_semantics_equal_frozen': True, 'compiler': compiled, 'kernel': kernel,
        'reader': dataset.summary(), 'verification': verification}


def classify_compiler(compiled):
    """Explicit allow-list of fixed construction failures, never arbitrary errors."""
    if compiled['status'] != 'REJECT_FIXED_SOURCE_COMPILER':
        return None
    reason = compiled.get('reason', '')
    prefixes = ('LEGACY_EXHAUSTIVE_SELECTOR_NO_CANDIDATE:',
        'FIXED_UNIQUE_SAME_BOUNDARY_POLICY:', 'SELECTED_SUPERSET_OWNER_NOT_ACTUALLY_RAW_EMITTED:',
        'SELECTED_REPLICA_AFFINE_TRAJECTORY_DISAGREEMENT', 'SELECTED_SOURCE_LABEL_OR_OWNER_ARITY',
        'LOWER_ROOT_UPPER_ORIENTED_SOURCE_BOUNDARY_CHANGED',
        'EXACT_SOURCE_TRAJECTORY_JUMPS_AT_A_DECLARED_BREAKPOINT', 'EXACT_SOURCE_ENDPOINT_DIAGONAL_DEGENERATE')
    if not reason.startswith(prefixes):
        return None
    return {'certified_necessary_policy_failure': True, 'gate': compiled.get('rejected_stage'),
        'reason_code': reason.split(':')[0], 'witness': reason,
        'evidence_binding': sha(compiled),
        'scope': 'Declared existing source construction policy; not all possible supports or actual native collision.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forest-cache', type=Path, required=True)
    parser.add_argument('--e2-cache', type=Path, required=True)
    parser.add_argument('--camera-inputs', type=Path, required=True)
    parser.add_argument('--frozen-e2-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for protected in (args.forest_cache.resolve(), args.e2_cache.resolve()):
        if args.output.resolve() == protected or protected in args.output.resolve().parents:
            raise ValueError('Output cannot be inside original source caches.')
    reports = Reports(args.output)
    started, cpu = time.monotonic(), time.process_time()
    source_files = [p for folder in (HERE, HERE.parent/'source_splice', HERE.parent/'tv0_tv4')
                    for p in sorted(folder.glob('*.py'))]
    source_hashes = {str(p): file_sha(p) for p in source_files}
    camera_hash = file_sha(args.camera_inputs)
    try:
        registry_bytes = (args.forest_cache/'event_registry_p1.csv').read_bytes()
        if hashlib.sha256(registry_bytes).hexdigest() != REGISTRY_SHA:
            raise ValueError('Forest registry is not the fixed 131-event input.')
        rows = list(csv.DictReader(io.StringIO(registry_bytes.decode())))
        expected_ids = {r['canonical_event_id'] for r in rows}
        if len(rows) != 440 or len(expected_ids) != 131:
            raise ValueError('Frozen population size mismatch.')
        camera = json.loads(args.camera_inputs.read_text())
        message('e2_regression_start')
        e2 = e2_regression(args.e2_cache, args.frozen_e2_source)
        reports.save('e2_regression.json', e2)
        message('e2_regression_pass', wall_seconds=e2['wall_seconds'])
        dataset = read_event_candidates(args.forest_cache)
        if set(dataset.event_ids) != expected_ids:
            raise ValueError('Stream reader changed the fixed population identity set.')
        kernels = {eid: compile_kernel(dataset.registry[eid], dataset.hypervertices) for eid in dataset.event_ids}
        requests = {eid: required_times(dataset.roots[eid], k['corner_times'])
                    for eid, k in kernels.items() if k['status'] == 'PASS_EXISTING_LOCAL_KERNEL'}
        message('forest_candidates_read', events=len(dataset.event_ids), kernel_status=dict(Counter(k['status'] for k in kernels.values())))
        snapshots = dataset.load_halos(requests)
        binding = reader_binding(dataset)
        message('forest_complete_halos_read', snapshots=len(snapshots), reader=dataset.summary())
        events = []
        for index, eid in enumerate(dataset.event_ids):
            begin, begin_cpu = time.monotonic(), time.process_time()
            dataset.reader.check()
            kernel = kernels[eid]
            event = {'event_id': eid, 'root': str(dataset.roots[eid]), 'decision': 'UNKNOWN',
                'runtime_attempted': False, 'runtime': {'status': 'NOT_ATTEMPTED'},
                'kernel': kernel, 'element': int(dataset.registry[eid][0]['element'])}
            if eid in requests:
                bounds = levels(dataset.roots[eid], kernel['corner_times'])
                schedule = make_schedule(camera, *bounds)
                event['schedule'] = schedule
                event['schedule_sha256'] = sha(schedule)
                compiled = compile_event(eid, dataset.roots[eid], kernel['corner_times'], snapshots,
                    dataset.hypervertices, kernel['critical_position'], cache_digest=binding,
                    group_count=dataset.groups, maximum_discrete_time=dataset.maximum, element=event['element'])
                event['compiler'] = compiled
                event['snapshot_summaries'] = [snapshots[eid, tau].summary() for tau in requests[eid]]
                rejection = classify_compiler(compiled)
                if rejection is not None:
                    event.update(decision='REJECTED_FIXED_POLICY', decisive_rejection=rejection)
                elif compiled['status'] == 'PASS_SOURCE_COMPILER':
                    interface = audit_requested_source(compiled['source'], schedule, snapshots,
                                                       event_id=eid, element=event['element'])
                    event['requested_source_interface'] = interface
                    if interface['status'] == 'REJECT':
                        event.update(decision='REJECTED_FIXED_POLICY', decisive_rejection={
                            **interface['decisive_rejection'], 'gate': 'requested_source_interface',
                            'compiler_sha256': sha(compiled), 'schedule_sha256': sha(schedule)})
                    else:
                        event['unresolved_reason'] = ('ACTUAL_FIVE_ELEMENT_REQUESTED_SCHEDULE_ADAPTER_REQUIRED'
                            if interface['status'] == 'PASS_REQUESTED_SOURCE_INTERFACE' else 'SOURCE_INTERFACE_UNKNOWN')
                else:
                    event['unresolved_reason'] = compiled.get('reason', compiled['status'])
            else:
                event['unresolved_reason'] = kernel.get('reason', kernel['status'])
            event['cost'] = {'wall_seconds': time.monotonic()-begin, 'cpu_seconds': time.process_time()-begin_cpu}
            event['artifact'] = reports.save('events/'+f'{index:03d}-'+hashlib.sha256(eid.encode()).hexdigest()[:12]+'.json', event)
            events.append(event)
            message('event_completed', number=index+1, total=131, decision=event['decision'],
                    reason=event.get('decisive_rejection', {}).get('reason_code', event.get('unresolved_reason')))
        verification = dataset.verify_inputs()
        if source_hashes != {str(p): file_sha(p) for p in source_files} or file_sha(args.camera_inputs) != camera_hash:
            raise ValueError('Executed source or camera input changed during campaign.')
        if {e['event_id'] for e in events} != expected_ids:
            raise ValueError('Result identity set differs from frozen registry.')
        decision = summarize_decisions(events)
        counts = Counter(e.get('decisive_rejection', {}).get('reason_code', e.get('unresolved_reason', 'ADMITTED')) for e in events)
        union_frames = sorted({n for e in events for n in e.get('schedule', {}).get('hit_frame_numbers', [])})
        index_rows = [{k: e[k] for k in ('event_id', 'element', 'root', 'decision', 'runtime_attempted', 'artifact', 'cost')} |
            {'reason': e.get('decisive_rejection', {}).get('reason_code', e.get('unresolved_reason')),
             'natural_hit_count': e.get('schedule', {}).get('natural_hit_count', 0)} for e in events]
        reports.save('events_index.json', index_rows)
        reports.save('input_verification.json', verification)
        summary = {'schema': 'forest-formal-requested-admission-v1',
            'status': 'COMPLETE_REAL_FIXED_POLICY_RATE' if decision['all_events_resolved'] else 'INCOMPLETE_SURVIVORS_OR_UNKNOWN',
            **decision, 'reasons': dict(counts), 'registry_sha256': REGISTRY_SHA,
            'expected_event_ids_sha256': sha(sorted(expected_ids)), 'camera_inputs_sha256': camera_hash,
            'streamed_source_binding_sha256': binding, 'e2_regression': e2['status'],
            'raw_observations': len(rows), 'logical_incidences': len({r['logical_incidence_id'] for r in rows}),
            'root_population': dict(Counter(e['root'] for e in events)),
            'candidate_schedule': {'events_with_natural_hits': sum(bool(e.get('schedule', {}).get('natural_hit_count')) for e in events),
                'event_frame_incidences': sum(e.get('schedule', {}).get('natural_hit_count', 0) for e in events),
                'unique_natural_frame_numbers': union_frames, 'unique_natural_hit_rate': len(union_frames)/len(camera['times_seconds']),
                'time_mapping': 'INPUT_RECONSTRUCTED; native initialization not yet used unless a runtime event is attempted',
                'actual_modified_frames': 0, 'root_excluded_from_natural_denominators': True},
            'cost': {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'report_bytes_before_summary': reports.bytes, 'source_reader': dataset.summary()},
            'executed_sources_sha256': source_hashes, 'input_verification_pass': verification['pass'],
            'limitations': ['Forest 64-frame 6px pre-displacement cache only, not six-scene results.',
                'Per-event fixed source-support construction, all other events ordinary baseline.',
                'Source-gate rejection is not an actual runtime geometry failure or global impossibility.',
                'No same-root union, full video, RGB, or all-real-time window theorem claimed.']}
        reports.save('summary.json', summary)
        message('campaign_summary', **{k: summary[k] for k in ('status', 'admitted', 'rejected_fixed_policy', 'unknown', 'policy_admission_rate')})
        return 0 if decision['all_events_resolved'] else 2
    except Exception as error:
        reports.save('STOP.json', {'status': 'HARNESS_OR_INPUT_STOP_NOT_ALGORITHM_REJECTION',
            'reason': type(error).__name__+': '+str(error), 'traceback': traceback.format_exc(),
            'wall_seconds': time.monotonic()-started, 'policy_admission_rate': None})
        message('STOP', reason=type(error).__name__+': '+str(error))
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
