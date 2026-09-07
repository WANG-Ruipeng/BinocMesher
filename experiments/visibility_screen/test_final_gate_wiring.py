"""Final-resource wiring tests: synthetic files/mocks, no native or raster."""
from contextlib import ExitStack
from copy import deepcopy
import ast
from datetime import datetime, timezone, timedelta
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import Mock, patch

import final_gate_resources as resources
import final_gate_source as source
import final_gate_screen as screen
import scene_source as old_source
import run_screen as old_screen
from screen_contracts import file_sha, read, load_protocol
from test_scene_source import profile, empty_cache
from test_run_screen import MemoryReports, empty_scene, query

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'protocol_20260906.json'
FROZEN = {
    'scene_source.py': '87b023010d63fa8176658c239145b8c80b1a37a58f683bd93bf098797280ee0b',
    'run_screen.py': 'a1ac0467e538dd751098abbcf87615a1ed574f42cf6c06fcdf388aa46524040d',
    'screen_resources.py': '2b122024a9b6d81ba2869fdae1bf60c2d406a29576ce2e13df63f73ff3922031',
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    root = Path(root)
    original, spec = load_protocol(PROTOCOL, 'cave')
    receipt = {'effective_budgets': {**original['budgets'], 'cache_per_scene_output_bytes': 8*1024**3,
                                   'cache_per_scene_wall_seconds': 5400}, 'input_sha256': {},
               'campaign_deadline_utc': (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}
    amendment, seal = root/'amendment.json', root/'seal.json'
    save(amendment, receipt); save(seal, {'synthetic': True})
    camera, _ = profile(spec['first_frame'])
    cache, build = root/'cache', root/'build'
    empty_cache(cache); build.mkdir()
    save(build/'camera_inputs.json', camera)
    complete = {'status': 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE', 'source_inputs_unchanged': True,
                'cache': str(cache), 'camera_inputs_sha256': file_sha(build/'camera_inputs.json')}
    save(build/'worker_complete.json', complete)
    binding = resources.amendment_binding(amendment)
    contract = {'segment': spec, 'budget_amendment': binding, 'effective_budgets': receipt['effective_budgets'],
                'worker_uses_original_protocol_unchanged': True,
                'only_supervisor_native_wall_and_output_ceiling_amended': True}
    save(build/'build_protocol.json', contract)
    summary = {'status': complete['status'], 'segment_id': 'cave', 'budget_amendment': binding,
               'build_protocol_sha256': file_sha(build/'build_protocol.json'), 'input_sha256': {}}
    save(build/'summary.json', summary)
    return NS(root=root, receipt=receipt, amendment=amendment, seal=seal, camera=camera,
              cache=cache, build=build, spec=spec, binding=binding, contract=contract, summary=summary)


def run_source(f, **kwargs):
    return source.run_source_stage(f.cache, f.camera, f.spec, f.root/'out',
        budget_amendment=f.amendment, protocol=PROTOCOL, seal=f.seal, build_report=f.build, **kwargs)


class MechanicalFreezeTests(unittest.TestCase):
    def test_original_modules_keep_the_exact_mountain_frozen_hashes(self):
        self.assertEqual({name: file_sha(HERE/name) for name in FROZEN}, FROZEN)

    def test_source_nonentry_functions_are_text_identical(self):
        for name, value in vars(old_source).items():
            if inspect.isfunction(value) and value.__module__ == old_source.__name__ and name not in ('main', 'run_source_stage'):
                with self.subTest(name=name):
                    self.assertEqual(inspect.getsource(value), inspect.getsource(getattr(source, name)))

    def test_screen_geometry_and_sequence_functions_are_text_identical(self):
        allowed = {'executed_sources', 'validate_inputs', 'reference_worker', 'baseline_references', 'run', 'main'}
        for name, value in vars(old_screen).items():
            if inspect.isfunction(value) and value.__module__ == old_screen.__name__ and name not in allowed:
                with self.subTest(name=name):
                    self.assertEqual(inspect.getsource(value), inspect.getsource(getattr(screen, name)))
        self.assertEqual(inspect.getsource(old_screen.Reports), inspect.getsource(screen.Reports))

    def test_executed_source_receipt_includes_new_entries_and_frozen_adapters(self):
        result = screen.executed_sources()
        required = {'final_gate_screen.py', 'final_gate_source.py', 'final_gate_resources.py',
                    'final_gate_amendment.py', 'scene_native.py', 'scene_geometry.py', 'screen_contracts.py'}
        self.assertTrue(required <= {Path(p).name for p in result})
        self.assertTrue(all(file_sha(p) == h for p, h in result.items()))

    def test_cli_requires_budget_amendment(self):
        for module in (source, screen):
            with self.subTest(module=module.__name__), patch('sys.argv', [module.__name__]):
                with self.assertRaises(SystemExit): module.main()


class ResourceQualificationTests(unittest.TestCase):
    def test_private_copy_requires_explicit_qualification_arguments(self):
        with self.assertRaises(TypeError): resources.PrivateSceneCache('/not-created')

    def test_mountain_is_not_a_new_retry(self):
        with patch.object(resources, 'load_amendment') as loader:
            with self.assertRaisesRegex(ValueError, 'ONLY_TWO_FIXED_RETRIES'):
                resources.validated_budget('a', PROTOCOL, 's', 'mountain')
            loader.assert_not_called()

    def test_invalid_amendment_stops_before_cache_inspection(self):
        with patch.object(resources, 'load_amendment', side_effect=ValueError('NOT_AUTHORIZED')), \
             patch.object(resources, 'cache_inventory') as inventory:
            with self.assertRaisesRegex(ValueError, 'NOT_AUTHORIZED'):
                resources.PrivateSceneCache('/not-created', budget_amendment='a', protocol=PROTOCOL,
                                            seal='s', segment_id='cave')
            inventory.assert_not_called()

    def test_only_two_budget_fields_may_change(self):
        original = load_protocol(PROTOCOL)['budgets']
        correct = {**original, 'cache_per_scene_output_bytes': 8*1024**3, 'cache_per_scene_wall_seconds': 5400}
        for key, value in [('cache_per_scene_output_bytes', 9*1024**3),
                           ('certification_and_screen_per_scene_wall_seconds', 999999),
                           ('cache_per_scene_wall_seconds', 5401)]:
            with self.subTest(key=key), patch.object(resources, 'load_amendment',
                return_value={'effective_budgets': {**correct, key: value}}):
                with self.assertRaisesRegex(ValueError, 'ONLY_FINAL_CACHE'):
                    resources.validated_budget('a', PROTOCOL, 's', 'cave')

    def test_above_eight_gib_stops_before_temp_copy(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root)
            with patch.object(resources, 'load_amendment', return_value=f.receipt), \
                 patch.object(resources, 'cache_inventory', return_value={'x': {'bytes': 8*1024**3+1}}), \
                 patch.object(resources.tempfile, 'mkdtemp') as fresh:
                with self.assertRaisesRegex(ValueError, 'EXCEEDS_8_GIB'):
                    resources.PrivateSceneCache(f.cache, budget_amendment=f.amendment, protocol=PROTOCOL,
                                                seal=f.seal, segment_id='cave', parent=f.root)
                fresh.assert_not_called()

    def test_real_tiny_private_copy_keeps_original_and_is_recoverably_removed(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root)
            before = resources.cache_inventory(f.cache)
            with patch.object(resources, 'load_amendment', return_value=f.receipt):
                private = resources.PrivateSceneCache(f.cache, budget_amendment=f.amendment, protocol=PROTOCOL,
                                                     seal=f.seal, segment_id='cave', parent=f.root)
            self.assertEqual(resources.content_inventory(before), resources.content_inventory(private.copy_before))
            temp = private.temporary; self.assertTrue(temp.is_dir())
            private.remove(); self.assertFalse(temp.exists())
            self.assertEqual(resources.cache_inventory(f.cache), before)

    def test_same_receipt_but_changed_build_envelope_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root); f.contract['effective_budgets']['cache_per_scene_output_bytes'] = 4*1024**3
            save(f.build/'build_protocol.json', f.contract)
            f.summary['build_protocol_sha256'] = file_sha(f.build/'build_protocol.json')
            save(f.build/'summary.json', f.summary)
            # The serialized amendment still binds the original exact 8 GiB receipt.
            with self.assertRaisesRegex(ValueError, 'BUILD_EFFECTIVE_BUDGETS'):
                resources.validate_build_budget(f.build, f.amendment, read(f.amendment), PROTOCOL, 'cave')

    def test_old_build_without_amendment_cannot_be_borrowed(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root); f.summary.pop('budget_amendment'); save(f.build/'summary.json', f.summary)
            with self.assertRaisesRegex(ValueError, 'BUILD_BUDGET_AMENDMENT'):
                resources.validate_build_budget(f.build, f.amendment, f.receipt, PROTOCOL, 'cave')


class SourceEntryTests(unittest.TestCase):
    def test_missing_amendment_does_not_make_output(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root)
            with patch.object(resources, 'load_amendment', side_effect=ValueError('MISSING_AUTHORIZATION')):
                with self.assertRaisesRegex(ValueError, 'MISSING_AUTHORIZATION'): run_source(f)
            self.assertFalse((f.root/'out').exists())

    def test_completed_synthetic_zero_registry_is_explicit_and_bound(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root)
            with patch.object(resources, 'load_amendment', return_value=f.receipt): result = run_source(f)
            summary = result['summary']
            self.assertEqual(summary['status'], 'COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION')
            self.assertEqual(summary['canonical_event_denominator'], 0)
            self.assertEqual(summary['budget_amendment'], f.binding)
            self.assertEqual(summary['resource_limits']['input_cache_bytes'], 8*1024**3)
            self.assertEqual(summary['resource_limits']['wall_seconds'], 1200)
            self.assertIsNone(summary['admission_rate']); self.assertFalse(summary['runtime_attempted'])
            resources.verify_bindings(summary['amendment_input_sha256'])

    def test_spec_cannot_change_seed_even_with_valid_budget(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root); f.spec = {**f.spec, 'seed': 999}
            with patch.object(resources, 'load_amendment', return_value=f.receipt):
                with self.assertRaisesRegex(ValueError, 'EXACT_REGISTERED_SPEC'): run_source(f)
            self.assertFalse((f.root/'out').exists())

    def test_cache_must_be_the_bound_new_build(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root); f.cache = f.root/'other'
            with patch.object(resources, 'load_amendment', return_value=f.receipt):
                with self.assertRaisesRegex(ValueError, 'SOURCE_CACHE_BINDING'): run_source(f)
            self.assertFalse((f.root/'out').exists())

    def test_source_compute_budget_is_not_increased(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root)
            with patch.object(resources, 'load_amendment', return_value=f.receipt):
                with self.assertRaisesRegex(ValueError, '1200_SECONDS'): run_source(f, seconds=1201)


class ScreenBindingTests(unittest.TestCase):
    def test_invalid_amendment_stops_before_native_and_records_no_partial_publication(self):
        args = NS(output=Path('/tmp/not-created'), segment='cave', replay_from=None,
                  protocol=PROTOCOL, seal=Path('/tmp/no-seal'), budget_amendment=Path('/tmp/no-amendment'))
        with patch.object(screen, 'Reports', MemoryReports), patch.object(screen, 'file_sha', return_value='b'*64), \
             patch.object(screen, 'validated_budget', side_effect=ValueError('INVALID_AMENDMENT')), \
             patch.object(screen, 'initialize') as native, patch.object(screen, 'PrivateSceneCache') as cache, \
             patch.object(screen, 'message'):
            result = screen.run(args)
        self.assertEqual(result, 2); native.assert_not_called(); cache.assert_not_called()
        summary = MemoryReports.latest.saved['summary.json']
        self.assertIn('INVALID_AMENDMENT', summary['reason']); self.assertFalse(summary['partial_output_published'])

    def test_reference_subprocess_receives_amendment_and_mismatch_is_not_consumed(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root)
            args = NS(protocol=PROTOCOL, seal=f.seal, budget_amendment=f.amendment,
                      build_report=f.build, source=f.root/'source', segment='cave')
            reports = NS(root=f.root); reader = NS(delta_t=1.)
            protocol = load_protocol(PROTOCOL)
            doc = {'status': 'PASS_ORIGINAL_ORDINARY_REFERENCE_SEQUENCE', 'segment_id': 'cave',
                   'protocol_sha256': 'x', 'source_summary_sha256': 'x',
                   'library_sha256': protocol['frozen_native']['ordinary_sha256'],
                   'initialization': {'actual_delta_t_hex': 1.0.hex()}, 'identity_enabled': False,
                   'budget_amendment': {'path': 'wrong', 'sha256': 'x'}, 'queries': []}
            with patch.object(screen.subprocess, 'run', return_value=NS(returncode=0)) as child, \
                 patch.object(screen, 'validated_budget', return_value=f.receipt), \
                 patch.object(screen, 'read', return_value=doc), patch.object(screen, 'file_sha', return_value='x'), \
                 patch.object(screen, 'all_queries') as queries:
                with self.assertRaisesRegex(ValueError, 'REFERENCE_AMENDMENT_BINDING'):
                    screen.baseline_references(args, reports, NS(path=f.cache), reader, protocol, [], {}, f.spec)
            argv = child.call_args.args[0]
            self.assertEqual(argv[argv.index('--budget-amendment')+1], str(f.amendment.resolve()))
            queries.assert_not_called()

    def test_reference_worker_validates_envelope_before_initializing_native(self):
        args = NS()
        with patch.object(screen, 'validate_inputs', side_effect=ValueError('BAD_REFERENCE_ENVELOPE')), \
             patch.object(screen, 'initialize') as native:
            with self.assertRaisesRegex(ValueError, 'BAD_REFERENCE_ENVELOPE'): screen.reference_worker(args)
            native.assert_not_called()

    def test_unchanged_mock_sequence_still_has_no_native_or_raster_call(self):
        q = query('f', '0', False, 'natural'); meshes = empty_scene(); baseline = screen.mesh_receipt(meshes)
        reader = NS(delta_t=1.); reports = MemoryReports()
        with patch.object(screen, 'all_queries', return_value=[q]), \
             patch.object(screen, 'slice_bound', return_value=({'meshes': meshes}, baseline)), \
             patch.object(screen, 'render_visibility') as raster, patch.object(screen, 'message'):
            refs, images, changed, replacements, calls = screen.execute_sequence(reader, {}, {'segment_id': 'cave'}, [], {}, {},
                {'components': []}, [], {'f': baseline}, reports, lambda: None)
        self.assertEqual((len(refs), images, changed, replacements, calls), (1, [], [], 0, 1))
        raster.assert_not_called()


class DeadlineTests(unittest.TestCase):
    def test_deadline_is_absolute_not_a_fresh_ninety_minutes(self):
        receipt = {'campaign_deadline_utc': '2026-09-06T19:28:39+00:00'}
        now = datetime(2026, 9, 6, 19, 28, 9, tzinfo=timezone.utc)
        self.assertEqual(resources.remaining_seconds(receipt, now), 30)
        self.assertEqual(resources.remaining_seconds(receipt, now+timedelta(minutes=1)), 0)

    def test_source_expired_campaign_stops_before_fresh_output(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root)
            f.receipt['campaign_deadline_utc'] = '2000-01-01T00:00:00+00:00'
            with patch.object(resources, 'load_amendment', return_value=f.receipt):
                with self.assertRaisesRegex(ValueError, 'CAMPAIGN_DEADLINE_REACHED'):
                    run_source(f)
            self.assertFalse((f.root/'out').exists())

    def test_private_copy_expired_campaign_stops_before_cache_inspection(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root)
            f.receipt['campaign_deadline_utc'] = '2000-01-01T00:00:00+00:00'
            with patch.object(resources, 'load_amendment', return_value=f.receipt), \
                 patch.object(resources, 'cache_inventory') as inventory:
                with self.assertRaisesRegex(ValueError, 'CAMPAIGN_DEADLINE_REACHED'):
                    resources.PrivateSceneCache(f.cache, budget_amendment=f.amendment, protocol=PROTOCOL,
                                                seal=f.seal, segment_id='cave')
            inventory.assert_not_called()

    def test_old_omp_primary_cannot_be_replayed_under_new_amendment(self):
        with tempfile.TemporaryDirectory() as root:
            f = fixture(root); protocol = load_protocol(PROTOCOL)
            args = NS(output=f.root/'out', segment='cave', replay_from=f.root/'old-primary',
                      protocol=PROTOCOL, seal=f.seal, budget_amendment=f.amendment)
            values = (protocol, f.spec, f.camera, {}, {}, f.cache, {'raw_registry_observations': 0},
                      [], {'events': []}, {})
            reader = NS(initialization={}, delta_t=1., n_elements=1, close=Mock())
            private = NS(path=f.root/'private', verify=Mock(return_value={'status': 'PASS'}), remove=Mock())
            primary = {'status': 'PASS_PREREGISTERED_SEGMENT_SCREEN_OMP1_ONLY', 'segment_id': 'cave',
                       'protocol_sha256': 'b'*64, 'preregistration_seal_sha256': 'b'*64,
                       'budget_amendment': {'path': 'wrong', 'sha256': '0'*64},
                       'effective_budgets': f.receipt['effective_budgets']}
            with patch.object(screen, 'Reports', MemoryReports), \
                 patch.object(screen, 'validate_inputs', return_value=values), \
                 patch.object(screen, 'validated_budget', return_value=f.receipt), \
                 patch.object(screen, 'PrivateSceneCache', return_value=private), \
                 patch.object(screen, 'initialize', return_value=reader), \
                 patch.object(screen, 'file_sha', return_value='b'*64), \
                 patch.object(screen, 'executed_sources', return_value={}), \
                 patch.object(screen, 'read', return_value=primary), \
                 patch.object(screen, 'message'), \
                 patch.object(screen, 'execute_sequence') as execute, \
                 patch.dict(screen.os.environ, {'OMP_NUM_THREADS': '8'}):
                code = screen.run(args)
            self.assertEqual(code, 2); execute.assert_not_called()
            self.assertIn('PRIMARY_SCREEN_AMENDMENT_MISMATCH', MemoryReports.latest.saved['summary.json']['reason'])
            private.remove.assert_called_once()


if __name__ == '__main__': unittest.main()
