import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import final_stop_binding as m
from screen_contracts import all_queries, digest


def fixture():
    h = 'a'*64
    root = Path('/reports/forest_b')
    paths = {'build': root/'build02/summary.json', 'build_contract': root/'build02/build_protocol.json',
             'source': root/'source02/summary.json', 'events_index': root/'source02/events_index.json',
             'source_inventory': root/'source02/source_support_inventory.json',
             'camera': root/'build02/camera_inputs.json', 'effective': root/'build02/effective_inputs.json',
             'stop': root/'screen01_omp1/summary.json'}
    refs = {k: {'path': str(p), 'sha256': h} for k, p in paths.items()}
    a = {'amendment_path': '/reports/budget_amendment01.json', 'amendment_sha256': h,
         'protocol_sha256': h, 'preregistration_seal_sha256': h, 'effective_budgets': {'cache_per_scene_output_bytes': 8*1024**3},
         'campaign_deadline_utc': '2026-09-06T19:28:39+00:00',
         'effective_protocol': {'segments': [{'segment_id': 'forest_b', 'first_frame': 97, 'last_frame': 160}],
                               'frozen_native': {'ordinary_sha256': h, 'ordinary_build_repo': '/build/ordinary'}}}
    budget = {'path': a['amendment_path'], 'sha256': h}
    profile = {'expected_delta_t_hex': (0.6562525).hex(), 'origin_seconds_hex': float(96.5/24).hex(),
               'duration_seconds_hex': (2.62501).hex(), 'group_count': 2}
    source = {'status': 'COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION', 'segment_id': 'forest_b',
        'budget_amendment': budget, 'input_verification': {'pass': True}, 'all_registry_identities_retained': True,
        'canonical_event_denominator': 0, 'raw_registry_observations': 0, 'root_population': {}, 'source_status_counts': {},
        'amendment_input_sha256': {refs[k]['path']: h for k in ('build', 'build_contract')},
        'events_index': refs['events_index'], 'source_support_inventory': refs['source_inventory'],
        'segment_spec_binding': {'kind': 'CANONICAL_JSON', 'sha256': digest(a['effective_protocol']['segments'][0])},
        'cache_root': '/data/forest_b_attempt02/native/cache', 'camera_binding': refs['camera'], 'profile': profile,
        'resource_limits': {'input_cache_bytes': 8*1024**3, 'wall_seconds': 1200, 'requested_wall_seconds': 1200,
                            'reader_rss_bytes': 2*1024**3, 'source_output_bytes': 100*1024**2}}
    measured = {'canonical_events': 0, 'raw_observations': 0, 'exact_roots': 0, 'source_decision_counts': {}}
    stop = {'status': 'STOP_SCREEN_ATTEMPT_NOT_COMPLETE', 'segment_id': 'forest_b', 'reason': m.REASON,
        'source': measured, 'budget_amendment': budget, 'protocol_sha256': h, 'preregistration_seal_sha256': h,
        'admission_rate': None, 'private_cache_removed': True, 'partial_output_published': False,
        'final_input_verification': {'status': 'PASS', 'private_nonlog_inputs_unchanged': True},
        'effective_budgets': a['effective_budgets'], 'campaign_deadline_utc': a['campaign_deadline_utc']}
    camera = {'times_seconds': [(96.5+i)/24 for i in range(64)]}
    init = {'actual_delta_t_hex': profile['expected_delta_t_hex'], 'origin_seconds_hex': profile['origin_seconds_hex'],
        'duration_hex': profile['duration_seconds_hex'], 'time_groups': 2, 'actual_elements': 1,
        'opaque_element_names': ['ground'], 'extra_smooth': False, 'library_sha256': h,
        'original_cache_root': source['cache_root'], 'build_repository': '/build/ordinary'}
    reference = {'status': 'PASS_ORIGINAL_ORDINARY_REFERENCE_SEQUENCE', 'segment_id': 'forest_b',
        'budget_amendment': budget, 'protocol_sha256': h, 'identity_enabled': False, 'source_summary_sha256': h,
        'library_sha256': h, 'initialization': init, 'code_sha256': {str(m.HERE/k): h for k in m.CODE_NAMES},
        'queries': [{'query': q, 'baseline': [{'vertices': 3, 'faces': 1, 'sha256': dict(vertices=h, faces=h, tags=h)}]}
                    for q in all_queries(camera, a['effective_protocol']['segments'][0], float.fromhex(init['actual_delta_t_hex']), [])]}
    documents = {'stop': stop, 'source': source, 'ordinary_reference': reference, 'camera': camera,
        'effective': {'opaque_elements': ['ground']}, 'events_index': [],
        'source_inventory': {'status': 'COMPLETE_FIXED_SOURCE_SUPPORT_INVENTORY', 'events': []},
        'build': {'status': 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'},
        'worker_complete': {'status': 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE', 'source_inputs_unchanged': True,
                            'cache': source['cache_root'], 'camera_inputs_sha256': h, 'effective_inputs_sha256': h}}
    return documents, refs, a


def observation_fixture():
    paths = {'--protocol': m.HERE/'protocol_20260906.json', '--seal': Path('/reports/seal.json'),
             '--budget-amendment': Path('/reports/budget_amendment01.json'), '--build-report': Path('/reports/build02'),
             '--source': Path('/reports/source02'), '--output': Path('/reports/screen01_omp1')}
    command = ('wsl -d Ubuntu --cd '+str(m.HERE)+' -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
               'MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/warpwang/miniforge3/envs/binoc-exp/bin/python -B '
               'final_gate_screen.py --segment forest_b '+' '.join(k+' '+str(v) for k, v in paths.items()))
    observation = {'schema': 'firsthand-tool-execution-observation-v1',
        'observation_source': 'PARENT_AGENT_FIRSTHAND_TOOL_EXECUTION', 'script_generated': False,
        'exact_start_time_recorded': False, 'prelaunch_input_hashes_recorded': False,
        'started_no_later_than_utc': '2026-09-06T16:49:03+00:00', 'launch_returned_running': True,
        'session_id': 14234, 'launch_chunk_id': '16ef38', 'terminal_chunk_id': 'c4c6cd', 'terminal_exit_code': 1,
        'observed_status': 'STOP_SCREEN_ATTEMPT_NOT_COMPLETE', 'observed_reason': m.REASON, 'command': command}
    return observation, paths


class FinalStopBindingTests(unittest.TestCase):
    def test_zero_source_does_not_promote_screen_or_visibility(self):
        docs, refs, a = fixture()
        result = m.validate_documents(docs, refs, a)
        self.assertEqual(result['canonical_events'], 0)
        self.assertNotIn('visibility', result)

    def test_exact_observed_command_and_nonzero_exit(self):
        obs, paths = observation_fixture()
        self.assertTrue(m.validate_observation(obs, paths, m.REASON))

    def test_changed_source_build_link(self):
        docs, refs, a = fixture(); docs['source']['amendment_input_sha256'][refs['build']['path']] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'EXACT_SECOND_BUILD'): m.validate_documents(docs, refs, a)

    def test_reference_from_other_source(self):
        docs, refs, a = fixture(); docs['ordinary_reference']['source_summary_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'THIS_SOURCE'): m.validate_documents(docs, refs, a)

    def test_source_complete_is_required(self):
        docs, refs, a = fixture(); docs['source']['status'] = 'STOP_SOURCE_STAGE_NOT_COMPLETE'
        with self.assertRaisesRegex(ValueError, 'COMPLETE_ZERO_SOURCE'): m.validate_documents(docs, refs, a)

    def test_nonzero_source_not_generalized(self):
        docs, refs, a = fixture(); docs['source']['canonical_event_denominator'] = 1
        with self.assertRaisesRegex(ValueError, 'COMPLETE_ZERO_SOURCE'): m.validate_documents(docs, refs, a)

    def test_missing_or_duplicate_reference_query(self):
        for operation in ('drop', 'duplicate'):
            docs, refs, a = fixture(); queries = docs['ordinary_reference']['queries']
            if operation == 'drop': queries.pop()
            else: queries[-1] = copy.deepcopy(queries[0])
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, '64_QUERY'):
                m.validate_documents(docs, refs, a)

    def test_wrong_actual_delta(self):
        docs, refs, a = fixture(); docs['ordinary_reference']['initialization']['actual_delta_t_hex'] = (1.).hex()
        with self.assertRaisesRegex(ValueError, 'INITIALIZATION_CHANGED'): m.validate_documents(docs, refs, a)

    def test_input_verification_or_cleanup_failure(self):
        for key in ('private_cache_removed', 'partial_output_published'):
            docs, refs, a = fixture(); docs['stop'][key] = not docs['stop'][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'FAILURE_RECEIPT'):
                m.validate_documents(docs, refs, a)

    def test_ordinary_observer_conflation_rejected(self):
        docs, refs, a = fixture(); docs['ordinary_reference']['identity_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'THIS_SOURCE'): m.validate_documents(docs, refs, a)

    def test_fabricated_screen_visibility_rejected(self):
        docs, refs, a = fixture(); docs['stop']['visibility'] = {'visible_candidate_events': 0}
        with self.assertRaisesRegex(ValueError, 'PROMOTE_SCREEN'): m.validate_documents(docs, refs, a)

    def test_missing_executed_script_rejected(self):
        docs, refs, a = fixture(); docs['ordinary_reference']['code_sha256'].pop(str(m.HERE/'scene_native.py'))
        with self.assertRaisesRegex(ValueError, 'CODE_SET_CHANGED'): m.validate_documents(docs, refs, a)

    def test_final_resource_limits_cannot_be_reinterpreted(self):
        docs, refs, a = fixture(); docs['source']['resource_limits']['reader_rss_bytes'] *= 2
        with self.assertRaisesRegex(ValueError, 'RESOURCE_CONTRACT_CHANGED'): m.validate_documents(docs, refs, a)

    def test_wrong_output_path_or_omp_rejected(self):
        for change in ('screen01_omp1', 'OMP_NUM_THREADS=1'):
            obs, paths = observation_fixture(); obs['command'] = obs['command'].replace(change, change+'different')
            with self.subTest(change=change), self.assertRaises(ValueError): m.validate_observation(obs, paths, m.REASON)

    def test_zero_exit_or_claimed_prelaunch_hash_rejected(self):
        for key, value in [('terminal_exit_code', 0), ('prelaunch_input_hashes_recorded', True), ('script_generated', True)]:
            obs, paths = observation_fixture(); obs[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_observation(obs, paths, m.REASON)

    def test_conflicting_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CONFLICTING'):
            m.merge_bindings({'/input': 'a'*64}, {'/input': 'b'*64})

    def test_external_receipt_cannot_attach_to_different_stop(self):
        _, refs, a = fixture()
        receipt = {'schema': m.SCHEMA, 'status': m.STATUS, 'evidence': {'stop': refs['stop'], 'build': refs['build'], 'build_contract': refs['build_contract']}}
        with self.assertRaisesRegex(ValueError, 'DIFFERENT_OUTCOME'):
            m.verify_external_stop(receipt, {'path': '/other', 'sha256': 'a'*64}, refs['build'], refs['build_contract'], a, {})

    def test_external_receipt_promoted_or_tampered_fields_rejected(self):
        _, refs, a = fixture(); evidence = {'stop': refs['stop'], 'build': refs['build'], 'build_contract': refs['build_contract']}
        receipt = {'schema': m.SCHEMA, 'status': m.STATUS, 'evidence': evidence, 'scope': {'screen_complete': True}}
        with patch.object(m, 'build_receipt', return_value={**receipt, 'scope': {'screen_complete': False}}):
            with self.assertRaisesRegex(ValueError, 'REVALIDATED'):
                m.verify_external_stop(receipt, refs['stop'], refs['build'], refs['build_contract'], a, {})


if __name__ == '__main__': unittest.main()
