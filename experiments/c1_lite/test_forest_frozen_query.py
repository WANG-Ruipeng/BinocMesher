"""Fake-reader single-query helper tests. Never loads native code."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import forest_frozen_query as m


class FrozenQueryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='frozen-query-unit-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root/'parent-cache'
        self.cache.mkdir()
        self.registry = self.cache/'event_registry_p1.csv'
        self.registry.write_bytes(b'synthetic registry only\n')
        self.camera = self.root/'camera.json'
        self.effective = self.root/'effective.json'
        self.camera.write_text('{}', encoding='utf-8')
        self.effective.write_text('{}', encoding='utf-8')
        self.library = self.root/'fake.so'
        self.library.write_bytes(b'never a real library')
        self.fake_hash = hashlib.sha256(self.library.read_bytes()).hexdigest()
        self.hash_patch = patch.object(m, 'FROZEN_SO_SHA256', self.fake_hash)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        mesh = (np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
                np.asarray([[0, 1, 2]], np.int32), np.ones(3, np.int32))
        self.snapshot = {'meshes': (mesh,)*5, 'library_sha256': self.fake_hash,
                         'ledger_requested': False, 'baseline_only': True, 'extra_smooth': False,
                         'query': {'mode': 'exact', 'value': '3/2'},
                         'counts': [{'element': i, 'vertices': 3, 'faces': 1} for i in range(5)],
                         'cost': {'wall_seconds': 0.01}, 'input_documents_sha256': {}}
        self.reader = SimpleNamespace(library_sha256=self.fake_hash, library_path=self.library,
                                      delta_t=0.5, initialization={}, close=Mock(),
                                      slice_query=Mock(return_value=self.snapshot))
        self.initializer = patch.object(m, 'initialize_forest', return_value=self.reader)
        self.initialize = self.initializer.start()
        self.addCleanup(self.initializer.stop)
        self.output = self.root/'receipt.json'

    def run_helper(self, **changes):
        args = dict(cache_copy=self.cache, build_repo=self.root, camera_inputs=self.camera,
                    effective_inputs=self.effective, time_mode='exact', value='3/2', output=self.output)
        args.update(changes)
        return m.run_query(**args)

    def test_exact_one_ordinary_query_closed_and_small(self):
        result = self.run_helper()
        self.reader.slice_query.assert_called_once_with(Fraction(3, 2), mode='exact', ledger=False)
        self.reader.close.assert_called_once_with()
        self.assertEqual(result['query_token'], ['exact', '3/2'])
        self.assertEqual(result['status'], 'FROZEN_ORDINARY_QUERY_RECEIPT')
        self.assertEqual(result['cost'], self.snapshot['cost'])
        self.assertEqual(len(result['meshes']), 5)
        self.assertLess(self.output.stat().st_size, 65536)
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertTrue(self.cache.is_dir())
        self.assertEqual(list(self.cache.iterdir()), [self.registry])
        self.assertEqual(self.initialize.call_args.kwargs, {})

    def test_physical_hex_is_preserved(self):
        value = (13/24).hex()
        result = self.run_helper(time_mode='physical', value=value)
        self.reader.slice_query.assert_called_once_with(float.fromhex(value), mode='physical', ledger=False)
        self.assertEqual(result['query_token'], ['physical', value])

    def test_decimal_physical_input_is_rejected_before_native(self):
        with self.assertRaisesRegex(ValueError, 'hexadecimal'):
            self.run_helper(time_mode='physical', value='0.5')
        self.initialize.assert_not_called()

    def test_existing_output_never_overwritten(self):
        self.output.write_bytes(b'preserve this')
        with self.assertRaisesRegex(ValueError, 'new'):
            self.run_helper()
        self.assertEqual(self.output.read_bytes(), b'preserve this')
        self.initialize.assert_not_called()

    def test_cache_output_is_rejected_before_native(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.run_helper(output=self.cache/'receipt.json')
        self.initialize.assert_not_called()

    def test_missing_output_parent_does_not_create_directory(self):
        with self.assertRaisesRegex(ValueError, 'parent'):
            self.run_helper(output=self.root/'absent'/'receipt.json')
        self.assertFalse((self.root/'absent').exists())
        self.initialize.assert_not_called()

    def test_native_failure_closes_without_receipt_or_cache_delete(self):
        self.reader.slice_query.side_effect = RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            self.run_helper()
        self.reader.close.assert_called_once_with()
        self.assertFalse(self.output.exists())
        self.assertTrue(self.registry.is_file())

    def test_wrong_library_closes_before_query(self):
        self.reader.library_sha256 = '0'*64
        with self.assertRaisesRegex(ValueError, 'frozen library'):
            self.run_helper()
        self.reader.close.assert_called_once_with()
        self.reader.slice_query.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_registry_change_is_not_given_receipt(self):
        def changed(*args, **kwargs):
            self.registry.write_bytes(b'changed synthetic registry')
            return self.snapshot
        self.reader.slice_query.side_effect = changed
        with self.assertRaisesRegex(ValueError, 'input changed'):
            self.run_helper()
        self.reader.close.assert_called_once_with()
        self.assertFalse(self.output.exists())

    def test_observer_enabled_or_wrong_mesh_count_not_ordinary_receipt(self):
        for changed in ({'ledger_requested': True}, {'meshes': self.snapshot['meshes'][:4]}):
            with self.subTest(changed=list(changed)):
                old = self.snapshot.copy()
                self.snapshot.update(changed)
                with self.assertRaisesRegex(ValueError, 'contract'):
                    self.run_helper()
                self.snapshot.clear()
                self.snapshot.update(old)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
