"""External binding of one observed terminal STOP, never screen certification.

This deliberately narrow reporting adapter supports the completed zero-registry
Forest B source and its failed first OMP1 screen. It preserves the raw STOP and
does not infer observer identity, runtime completion, or measured visibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex

from coverage_report import read_bound, require
from final_gate_amendment import file_sha, load_amendment
from screen_contracts import all_queries, digest

HERE = Path(__file__).resolve().parent
SCHEMA = 'externally-observed-screen-stop-binding-v1'
STATUS = 'EXTERNALLY_BOUND_TERMINAL_SCREEN_STOP'
REASON = 'ValueError: Original SourceVID observation is not ready.'
CODE_NAMES = ('final_gate_screen.py', 'final_gate_source.py', 'final_gate_resources.py',
              'final_gate_amendment.py', 'screen_contracts.py', 'screen_resources.py',
              'scene_native.py', 'scene_geometry.py', 'scene_source.py', 'run_screen.py')


def merge_bindings(target, values):
    for path, sha in values.items():
        require(Path(path).is_absolute() and isinstance(sha, str) and len(sha) == 64
                and all(c in '0123456789abcdef' for c in sha), 'INVALID_EXTERNAL_INPUT_BINDING')
        require(path not in target or target[path] == sha, 'CONFLICTING_EXTERNAL_INPUT_BINDING')
        target[path] = sha


def validate_observation(observation, paths, reason):
    """Check the explicitly firsthand tool record, not a script-produced claim."""
    require(observation.get('schema') == 'firsthand-tool-execution-observation-v1'
            and observation.get('observation_source') == 'PARENT_AGENT_FIRSTHAND_TOOL_EXECUTION'
            and observation.get('script_generated') is False
            and observation.get('exact_start_time_recorded') is False
            and observation.get('prelaunch_input_hashes_recorded') is False,
            'EXTERNAL_OBSERVATION_PROVENANCE_REQUIRED')
    from datetime import datetime
    before = datetime.fromisoformat(observation['started_no_later_than_utc'])
    require(before.tzinfo is not None and observation.get('launch_returned_running') is True
            and type(observation.get('session_id')) is int and observation['session_id'] > 0
            and isinstance(observation.get('launch_chunk_id'), str) and observation['launch_chunk_id']
            and isinstance(observation.get('terminal_chunk_id'), str) and observation['terminal_chunk_id']
            and type(observation.get('terminal_exit_code')) is int and observation['terminal_exit_code'] != 0
            and observation.get('observed_status') == 'STOP_SCREEN_ATTEMPT_NOT_COMPLETE'
            and observation.get('observed_reason') == reason, 'EXTERNAL_NONZERO_TERMINAL_OBSERVATION_REQUIRED')
    argv = shlex.split(observation['command'])
    require(argv[:5] == ['wsl', '-d', 'Ubuntu', '--cd', str(HERE)], 'OBSERVED_WORKING_DIRECTORY_CHANGED')
    expected_prefix = ['--', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                       'MKL_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1',
                       '/home/warpwang/miniforge3/envs/binoc-exp/bin/python', '-B', 'final_gate_screen.py']
    require(argv[5:14] == expected_prefix, 'OBSERVED_EXECUTABLE_OR_OMP_CHANGED')
    args = argv[14:]
    require(len(args) == 14, 'OBSERVED_SCREEN_ARGUMENT_SET_CHANGED')
    options = dict(zip(args[::2], args[1::2]))
    require(set(options) == {'--protocol', '--seal', '--budget-amendment', '--segment',
                            '--build-report', '--source', '--output'}
            and options['--segment'] == 'forest_b', 'OBSERVED_SCREEN_ARGUMENT_SET_CHANGED')
    for option, expected in paths.items():
        value = Path(options[option]); value = value if value.is_absolute() else HERE/value
        require(value.resolve() == Path(expected).resolve(), 'OBSERVED_ATTEMPT_PATH_CHANGED:'+option)
    return True


def validate_documents(documents, refs, amendment):
    """Pure semantic checks after hash-bound reads; no favorable missing defaults."""
    d = documents; stop, source, reference = d['stop'], d['source'], d['ordinary_reference']
    expected = {'path': amendment['amendment_path'], 'sha256': amendment['amendment_sha256']}
    require(stop.get('status') == 'STOP_SCREEN_ATTEMPT_NOT_COMPLETE' and stop.get('reason') == REASON
            and stop.get('segment_id') == 'forest_b' and stop.get('admission_rate') is None
            and stop.get('partial_output_published') is False and stop.get('private_cache_removed') is True
            and stop.get('final_input_verification', {}).get('status') == 'PASS'
            and stop['final_input_verification'].get('private_nonlog_inputs_unchanged') is True,
            'RAW_STOP_NOT_SUPPORTED_COMPLETE_FAILURE_RECEIPT')
    require('input_and_code_sha256' not in stop
            and not any(k in stop for k in ('composition', 'schedule', 'visibility')),
            'OUTER_BINDING_MUST_NOT_REWRITE_OR_PROMOTE_SCREEN_RESULTS')
    require(stop.get('budget_amendment') == source.get('budget_amendment') == reference.get('budget_amendment') == expected
            and stop.get('protocol_sha256') == reference.get('protocol_sha256') == amendment['protocol_sha256']
            and stop.get('preregistration_seal_sha256') == amendment['preregistration_seal_sha256']
            and stop.get('effective_budgets') == amendment['effective_budgets']
            and stop.get('campaign_deadline_utc') == amendment['campaign_deadline_utc'], 'OUTER_STOP_AMENDMENT_CHANGED')
    require(source.get('status') == 'COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION'
            and source.get('segment_id') == 'forest_b' and source.get('input_verification', {}).get('pass') is True
            and source.get('all_registry_identities_retained') is True
            and source.get('canonical_event_denominator') == source.get('raw_registry_observations') == 0
            and source.get('root_population') == source.get('source_status_counts') == {}
            and d['events_index'] == [] and d['source_inventory'].get('events') == []
            and d['source_inventory'].get('status') == 'COMPLETE_FIXED_SOURCE_SUPPORT_INVENTORY',
            'ONLY_MEASURED_COMPLETE_ZERO_SOURCE_SUPPORTED')
    measured = {'canonical_events': 0, 'exact_roots': 0, 'raw_observations': 0, 'source_decision_counts': {}}
    require(stop.get('source') == measured, 'STOP_SOURCE_COUNTS_NOT_EQUAL_MEASURED_SOURCE')
    require(all(source.get('amendment_input_sha256', {}).get(refs[k]['path']) == refs[k]['sha256']
                for k in ('build', 'build_contract')), 'SOURCE_NOT_BOUND_TO_EXACT_SECOND_BUILD')
    require(source['events_index']['sha256'] == refs['events_index']['sha256']
            and source['source_support_inventory']['sha256'] == refs['source_inventory']['sha256'],
            'SOURCE_POPULATION_ARTIFACT_CHANGED')
    segment = next(s for s in amendment['effective_protocol']['segments'] if s['segment_id'] == 'forest_b')
    require(source['segment_spec_binding'] == {'kind': 'CANONICAL_JSON', 'sha256': digest(segment)},
            'SOURCE_FIXED_SEGMENT_CHANGED')
    worker = d['worker_complete']; cache = Path(source['cache_root']).resolve()
    require(worker['status'] == d['build']['status'] and worker['source_inputs_unchanged'] is True
            and Path(worker['cache']).resolve() == cache
            and source['camera_binding'] == refs['camera']
            and all(worker[k+'_sha256'] == refs[k]['sha256'] for k in ('camera_inputs', 'effective_inputs')
                    if k in refs), 'SOURCE_BUILD_CAMERA_OR_CACHE_CHANGED')
    require(worker['camera_inputs_sha256'] == refs['camera']['sha256']
            and worker['effective_inputs_sha256'] == refs['effective']['sha256'], 'WORKER_INPUT_DOCUMENT_CHANGED')
    require(reference.get('status') == 'PASS_ORIGINAL_ORDINARY_REFERENCE_SEQUENCE'
            and reference.get('segment_id') == 'forest_b'
            and reference.get('identity_enabled') is False
            and reference.get('source_summary_sha256') == refs['source']['sha256']
            and reference.get('library_sha256') == amendment['effective_protocol']['frozen_native']['ordinary_sha256'],
            'ORDINARY_REFERENCE_NOT_BOUND_TO_THIS_SOURCE')
    init = reference['initialization']; profile = source['profile']
    require(init['actual_delta_t_hex'] == profile['expected_delta_t_hex']
            and init['origin_seconds_hex'] == profile['origin_seconds_hex']
            and init['duration_hex'] == profile['duration_seconds_hex']
            and init['time_groups'] == profile['group_count'] == 2
            and init['actual_elements'] == len(d['effective']['opaque_elements'])
            and init['opaque_element_names'] == d['effective']['opaque_elements']
            and init['extra_smooth'] is False
            and init['library_sha256'] == reference['library_sha256']
            and Path(init['original_cache_root']).resolve() == cache
            and Path(init['build_repository']).resolve() == Path(amendment['effective_protocol']['frozen_native']['ordinary_build_repo']).resolve(),
            'ORDINARY_REFERENCE_INITIALIZATION_CHANGED')
    expected_queries = all_queries(d['camera'], segment, float.fromhex(init['actual_delta_t_hex']), [])
    require(len(reference['queries']) == 64 and [r['query'] for r in reference['queries']] == expected_queries,
            'ORDINARY_REFERENCE_FIXED_64_QUERY_SEQUENCE_CHANGED')
    require(set(reference['code_sha256']) == {str(HERE/name) for name in CODE_NAMES},
            'ORDINARY_REFERENCE_EXECUTED_CODE_SET_CHANGED')
    for row in reference['queries']:
        require(len(row['baseline']) == init['actual_elements'], 'ORDINARY_REFERENCE_ELEMENT_COUNT_CHANGED')
        for mesh in row['baseline']:
            require(all(type(mesh[k]) is int and mesh[k] >= 0 for k in ('vertices', 'faces'))
                    and set(mesh['sha256']) == {'vertices', 'faces', 'tags'}
                    and all(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                            for h in mesh['sha256'].values()), 'INVALID_ORDINARY_ARRAY_RECEIPT')
    limits = source['resource_limits']
    require(limits['input_cache_bytes'] == amendment['effective_budgets']['cache_per_scene_output_bytes']
            and limits['wall_seconds'] == limits['requested_wall_seconds'] == 1200
            and limits['reader_rss_bytes'] == 2*1024**3 and limits['source_output_bytes'] == 100*1024**2,
            'SOURCE_RESOURCE_CONTRACT_CHANGED')
    return measured


def evaluate_evidence(evidence, amendment, bindings):
    """Re-read actual evidence and current inputs; never load a native library."""
    required = {'build', 'build_contract', 'worker_complete', 'source', 'events_index',
                'source_inventory', 'camera', 'effective', 'stop', 'ordinary_reference', 'execution_observation'}
    require(set(evidence) == required, 'EXTERNAL_STOP_EVIDENCE_SET_CHANGED')
    documents = {}
    for key, ref in evidence.items():
        documents[key], actual = read_bound(ref['path'], bindings, ref['sha256'])
        require(actual == ref, 'EXTERNAL_STOP_REFERENCE_MUST_BE_ABSOLUTE')
    from final_gate_verdict import validate_retry_build
    validate_retry_build(documents['build'], documents['build_contract'], 'forest_b', amendment)
    require(documents['build']['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'
            and documents['build']['build_protocol_sha256'] == evidence['build_contract']['sha256'],
            'SCREEN_STOP_REQUIRES_COMPLETED_SECOND_BUILD')
    stop_root = Path(evidence['stop']['path']).parent
    build_root = Path(evidence['build']['path']).parent
    source_root = Path(evidence['source']['path']).parent
    require(stop_root.name == 'screen01_omp1' and source_root.name == 'source02'
            and build_root.name == 'build02' and stop_root.parent == source_root.parent == build_root.parent,
            'STOP_SOURCE_BUILD_ATTEMPTS_NOT_SAME_SECOND_CHAIN')
    require(Path(evidence['ordinary_reference']['path']) == stop_root/'ordinary_reference.json',
            'ORDINARY_REFERENCE_NOT_THIS_OBSERVED_ATTEMPT')
    paths = {'--protocol': HERE/'protocol_20260906.json', '--seal': Path(amendment['amendment_path']).parent/'preregistration_seal.json',
             '--budget-amendment': amendment['amendment_path'], '--build-report': build_root,
             '--source': source_root, '--output': stop_root}
    validate_observation(documents['execution_observation'], paths, documents['stop'].get('reason'))
    measured = validate_documents(documents, evidence, amendment)
    maps = [documents['source'][k] for k in ('amendment_input_sha256', 'executed_sources_sha256',
                                           'cache_completion_provenance_sha256')]
    maps += [documents['build']['input_sha256'], documents['source']['input_verification']['input_full_sha256'],
             documents['ordinary_reference']['code_sha256'], amendment['input_sha256']]
    for values in maps:
        merge_bindings(bindings, values)
    for path, sha in bindings.items():
        require(file_sha(path) == sha, 'EXTERNAL_STOP_CURRENT_INPUT_CHANGED:'+path)
    # This recheck also ties finally's cache digest to the source's actual cache,
    # rather than claiming that cache-only verification covered all source code.
    from forest_native_campaign import cache_inventory, content_inventory
    inventory = cache_inventory(documents['source']['cache_root'])
    actual_digest = hashlib.sha256(json.dumps(content_inventory(inventory), sort_keys=True).encode()).hexdigest()
    verification = documents['stop']['final_input_verification']
    require(actual_digest == verification['original_input_content_sha256']
            and len(inventory) == verification['original_files_unchanged']
            and sum(r['bytes'] for r in inventory.values()) == verification['original_bytes'],
            'STOP_FINAL_CACHE_VERIFICATION_NOT_BOUND_TO_THIS_SOURCE')
    private = Path(documents['ordinary_reference']['initialization']['private_cache_root'])
    require(not private.exists(), 'OBSERVED_PRIVATE_CACHE_STILL_EXISTS')
    return {'measured_source_counts': measured, 'ordinary_queries_confirmed': 64,
            'original_cache_content_sha256': actual_digest,
            'current_input_hashes_verified': len(bindings), 'private_cache_absent': True}


def build_receipt(evidence, amendment):
    bindings = {}
    checked = evaluate_evidence(evidence, amendment, bindings)
    return {'schema': SCHEMA, 'status': STATUS, 'segment_id': 'forest_b',
        'budget_amendment': {'path': amendment['amendment_path'], 'sha256': amendment['amendment_sha256']},
        'protocol_sha256': amendment['protocol_sha256'], 'preregistration_seal_sha256': amendment['preregistration_seal_sha256'],
        'evidence': evidence, 'validated_external_evidence': checked, 'observed_input_sha256': bindings,
        'scope': {'screen_self_certification': False, 'precise_launch_time_or_prelaunch_hashes_claimed': False,
            'input_hash_observation_time': 'Reverified after termination, anchored by the independent reference and source receipts.',
            'tool_execution_observation': 'Firsthand parent-agent observation; not a cryptographically signed or script-generated execution receipt.',
            'ordinary_reference_is_observer_or_method_admission': False,
            'screen_complete': False, 'runtime_admission': None, 'visibility': None, 'omp1_omp8_confirmation': False,
            'reason_not_inferred_as_geometric_failure': True, 'native_queries_executed_by_this_adapter': 0,
            'images_rendered_by_this_adapter': 0, 'raw_stop_modified': False},
        'executed_reporting_code_sha256': file_sha(__file__)}


def verify_external_stop(receipt, outcome_ref, build_ref, contract_ref, amendment, bindings):
    require(receipt.get('schema') == SCHEMA and receipt.get('status') == STATUS, 'INVALID_EXTERNAL_STOP_SCHEMA')
    evidence = receipt['evidence']
    require(evidence['stop'] == outcome_ref and evidence['build'] == build_ref and evidence['build_contract'] == contract_ref,
            'EXTERNAL_STOP_ATTACHED_TO_DIFFERENT_OUTCOME')
    rebuilt = build_receipt(evidence, amendment)
    require(receipt == rebuilt, 'EXTERNAL_STOP_RECEIPT_NOT_EQUAL_REVALIDATED_EVIDENCE')
    merge_bindings(bindings, rebuilt['observed_input_sha256'])
    return rebuilt['validated_external_evidence']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'seal', 'budget-amendment', 'build-report', 'source', 'attempt', 'observation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'EXTERNAL_STOP_OUTPUT_MUST_BE_FRESH')
    amendment = load_amendment(args.budget_amendment, args.protocol, args.seal, 'forest_b')
    paths = {'build': args.build_report/'summary.json', 'build_contract': args.build_report/'build_protocol.json',
        'worker_complete': args.build_report/'worker_complete.json', 'camera': args.build_report/'camera_inputs.json',
        'effective': args.build_report/'effective_inputs.json', 'source': args.source/'summary.json',
        'events_index': args.source/'events_index.json', 'source_inventory': args.source/'source_support_inventory.json',
        'stop': args.attempt/'summary.json', 'ordinary_reference': args.attempt/'ordinary_reference.json',
        'execution_observation': args.observation}
    evidence = {key: {'path': str(path.resolve()), 'sha256': file_sha(path)} for key, path in paths.items()}
    receipt = build_receipt(evidence, amendment)
    require(args.output.resolve() not in {p.resolve() for p in paths.values()}, 'REFUSE_OVERWRITING_STOP_EVIDENCE')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'source': receipt['validated_external_evidence']['measured_source_counts'],
                      'screen_complete': False, 'visibility': None}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
