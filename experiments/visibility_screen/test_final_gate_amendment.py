"""Resource-contract tests only; no actual inventories, library or campaign run."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import final_gate_amendment as m


def fixture():
    protocol = {'registered_at_utc': '2026-09-06 10:28:39 UTC',
        'method_policy': 'unchanged strict XY and exact contact',
        'segments': [{'segment_id': 'cave', 'seed': 1, 'frames': [1, 64]},
                     {'segment_id': 'forest_b', 'seed': 0, 'frames': [97, 160]},
                     {'segment_id': 'mountain', 'seed': 1, 'frames': [1, 64]}],
        'visibility': {'minimum_frames': 2, 'pixels': 16},
        'budgets': {'cache_per_scene_output_bytes': 4*1024**3, 'cache_per_scene_wall_seconds': 2700,
            'maximum_campaign_wall_seconds': 32400, 'source_per_scene_wall_seconds': 1200,
            'certification_and_screen_per_scene_wall_seconds': 3600, 'native_query_peak_rss_bytes': 8*1024**3,
            'total_existing_and_new_storage_limit_bytes': 400000000000}}
    effective = deepcopy(protocol); effective['budgets'] = m.effective_budgets(protocol)
    receipt = {'schema': 'final-visibility-gate-resource-amendment-v1',
        'status': 'APPROVED_FINAL_RESOURCE_CEILING_AMENDMENT',
        'amendment_number': 1, 'supersedes_amendment': None, 'final_resource_increase': True,
        'protocol_sha256': 'p'*64, 'preregistration_seal_sha256': 's'*64,
        'authorization': {'path': '/tmp/authorization', 'sha256': m.AUTHORIZATION_SHA256},
        'changes': deepcopy(m.CHANGES), 'effective_budgets': effective['budgets'], 'effective_protocol': effective,
        'campaign_deadline_utc': m.DEADLINE, 'registered_at_utc': '2026-09-06T14:30:00+00:00',
        'run_order': ['cave', 'forest_b'], 'attempt_forest_b_regardless_of_cave_outcome': True,
        'retry_sources': {'cave': {'source_coarse': '/original/cave'}, 'forest_b': {'source_coarse': '/original/forest'}},
        'completed_segment_reused': 'mountain', 'on_resource_limit': 'RESOURCE_UNRESOLVED',
        'old_stop_artifacts_preserved': True, 'new_attempt_directories_required': True,
        'prohibited_until_both_retry_outcomes': ['RGB', 'SSIM', 'post-displacement', 'five-method-quality']}
    return protocol, receipt


class AmendmentContractTests(unittest.TestCase):
    def test_only_cache_and_build_limits_change(self):
        protocol, receipt = fixture(); original = deepcopy(protocol)
        for segment in (None, 'cave', 'forest_b'):
            self.assertIs(m.validate_contract(receipt, protocol, 'p'*64, 's'*64, segment), receipt)
        self.assertEqual(receipt['effective_budgets']['cache_per_scene_output_bytes'], 8*1024**3)
        self.assertEqual(receipt['effective_budgets']['cache_per_scene_wall_seconds'], 5400)
        self.assertEqual(receipt['effective_budgets']['source_per_scene_wall_seconds'], 1200)
        self.assertEqual(receipt['effective_budgets']['certification_and_screen_per_scene_wall_seconds'], 3600)
        self.assertEqual(protocol, original)

    def test_method_scene_seed_and_threshold_changes_rejected(self):
        for field in ('method', 'seed', 'threshold'):
            protocol, receipt = fixture()
            if field == 'method': receipt['effective_protocol']['method_policy'] = 'relaxed'
            elif field == 'seed': receipt['effective_protocol']['segments'][0]['seed'] = 2
            else: receipt['effective_protocol']['visibility']['pixels'] = 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'NONRESOURCE'):
                m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_extra_time_or_rss_increase_rejected(self):
        for key in ('source_per_scene_wall_seconds', 'certification_and_screen_per_scene_wall_seconds', 'native_query_peak_rss_bytes'):
            protocol, receipt = fixture()
            receipt['effective_budgets'][key] *= 2
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'NONRESOURCE'):
                m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_cannot_add_third_changed_path_or_exceed_ceiling(self):
        protocol, receipt = fixture(); receipt['changes']['new_limit'] = {'old': 1, 'new': 2}
        with self.assertRaisesRegex(ValueError, 'ONLY_TWO'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)
        protocol, receipt = fixture(); receipt['effective_budgets']['cache_per_scene_output_bytes'] = 9*1024**3
        with self.assertRaisesRegex(ValueError, 'NONRESOURCE'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_second_amendment_and_chained_amendment_rejected(self):
        for key, value in (('amendment_number', 2), ('supersedes_amendment', 'old'), ('final_resource_increase', False)):
            protocol, receipt = fixture(); receipt[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'SECOND_RESOURCE'):
                m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_authorization_and_base_seal_are_exactly_bound(self):
        protocol, receipt = fixture(); receipt['authorization']['sha256'] = 'x'*64
        with self.assertRaisesRegex(ValueError, 'AUTHORIZATION'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)
        protocol, receipt = fixture()
        with self.assertRaisesRegex(ValueError, 'BASE_PROTOCOL_OR_SEAL'): m.validate_contract(receipt, protocol, 'x'*64, 's'*64)

    def test_deadline_never_restarts_from_registration(self):
        protocol, receipt = fixture()
        self.assertEqual(m.original_deadline(protocol), '2026-09-06T19:28:39+00:00')
        receipt['campaign_deadline_utc'] = '2026-09-06T23:30:00+00:00'
        with self.assertRaisesRegex(ValueError, 'CLOCK_RESET'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_expired_registration_rejected(self):
        protocol, receipt = fixture(); receipt['registered_at_utc'] = m.DEADLINE
        with self.assertRaisesRegex(ValueError, 'REGISTERED_AFTER'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_late_historical_load_contract_does_not_extend_execution_deadline(self):
        protocol, receipt = fixture()
        # Pure validation is independent of current wall time for archival reading.
        m.validate_contract(receipt, protocol, 'p'*64, 's'*64)
        self.assertEqual(receipt['campaign_deadline_utc'], m.DEADLINE)

    def test_cave_positive_cannot_skip_forest(self):
        protocol, receipt = fixture(); receipt['attempt_forest_b_regardless_of_cave_outcome'] = False
        with self.assertRaisesRegex(ValueError, 'STOPPING_POLICY'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_order_and_no_quality_phase_gate_are_fixed(self):
        protocol, receipt = fixture(); receipt['run_order'].reverse()
        with self.assertRaisesRegex(ValueError, 'ORDER'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)
        protocol, receipt = fixture(); receipt['prohibited_until_both_retry_outcomes'].remove('RGB')
        with self.assertRaisesRegex(ValueError, 'QUALITY'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_mountain_legacy_completion_is_reused_not_retried(self):
        protocol, receipt = fixture()
        with self.assertRaisesRegex(ValueError, 'NOT_AUTHORIZED'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64, 'mountain')
        receipt['completed_segment_reused'] = 'cave'
        with self.assertRaisesRegex(ValueError, 'POPULATION'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_resource_unknown_never_becomes_zero(self):
        protocol, receipt = fixture(); receipt['on_resource_limit'] = 'ZERO_VISIBLE'
        with self.assertRaisesRegex(ValueError, 'RESOURCE_OR_EVIDENCE'): m.validate_contract(receipt, protocol, 'p'*64, 's'*64)

    def test_register_refuses_existing_or_second_receipt_without_reading_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'budget_amendment01.json'; output.touch()
            with patch.object(m, '_collect') as collect:
                with self.assertRaisesRegex(ValueError, 'SECOND_REGISTRATION'): m.register_amendment('p', 's', 'a', output)
                collect.assert_not_called()
            output.unlink(); (root/'budget_amendment02.json').touch()
            with self.assertRaisesRegex(ValueError, 'SECOND_REGISTRATION'): m.register_amendment('p', 's', 'a', output)

    def test_register_refuses_expired_global_budget_before_scanning(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, '_collect') as collect:
            with self.assertRaisesRegex(ValueError, 'DEADLINE_EXPIRED'):
                m.register_amendment('p', 's', 'a', Path(directory)/'budget_amendment01.json', now=datetime(2026, 9, 6, 20, tzinfo=timezone.utc))
            collect.assert_not_called()

    def test_load_rejects_retry_mapping_even_with_other_fields_valid(self):
        protocol, receipt = fixture()
        bindings = {'/p': 'p'*64}; retry = deepcopy(receipt['retry_sources'])
        receipt.update(input_sha256=bindings, previous_resource_stops={}, mountain_completed_evidence={}, registration_code_sha256='c'*64)
        receipt['retry_sources']['cave']['source_coarse'] = '/replacement/scene'
        collected = (protocol, 'p'*64, 's'*64, bindings, retry, {}, {})
        with patch.object(m, 'read', return_value=receipt), patch.object(m, '_collect', return_value=collected):
            with self.assertRaisesRegex(ValueError, 'RETRY_MAPPING_CHANGED'):
                m.load_amendment('/tmp/budget_amendment01.json', '/p', '/s', 'cave')

    def test_loaded_receipt_adds_own_hash_without_mutating_registered_input_map(self):
        protocol, receipt = fixture(); bindings = {'/p': 'p'*64}
        receipt.update(input_sha256=bindings, previous_resource_stops={}, mountain_completed_evidence={}, registration_code_sha256='c'*64)
        collected = (protocol, 'p'*64, 's'*64, bindings, receipt['retry_sources'], {}, {})
        with patch.object(m, 'read', return_value=receipt), patch.object(m, '_collect', return_value=collected), patch.object(m, 'file_sha', return_value='c'*64):
            result = m.load_amendment('/tmp/budget_amendment01.json', '/p', '/s', 'forest_b')
        self.assertEqual(result['amendment_sha256'], 'c'*64)
        self.assertIn('/tmp/budget_amendment01.json', result['input_sha256'])
        self.assertEqual(receipt['input_sha256'], {'/p': 'p'*64})


if __name__ == '__main__':
    unittest.main()
