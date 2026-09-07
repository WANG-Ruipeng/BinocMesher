from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from forest_native_campaign import PrivateCache, array_sha, mesh_receipt, apply_and_verify


class NativeCampaignTests(unittest.TestCase):
    def test_transient_apply_keeps_original_arrays_bit_exact(self):
        v = np.asarray([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)])
        f = np.asarray([(0, 1, 2), (0, 2, 3)], np.int32)
        mesh = (v, f, np.ones(4, np.int32))
        empty = (np.empty((0, 3)), np.empty((0, 3), np.int32), np.empty(0, np.int32))
        meshes = (empty, empty, mesh, empty, empty)
        before = mesh_receipt(meshes)
        for part in mesh:
            part.setflags(write=False)
        result = apply_and_verify(meshes, {'element': 2, 'removed_face_rows': [0, 1],
            'boundary_actual_ids': [0, 1, 2, 3], 'center': [.5, .5, .25]})
        self.assertEqual(result['vertices_after'], 5)
        self.assertEqual(result['faces_after'], 4)
        self.assertEqual(mesh_receipt(meshes), before)
        self.assertTrue(result['old_vertices_and_tags_byte_identical'])
        self.assertEqual(result['status'], 'ACTUAL_ARRAYS_CONSTRUCTED_NOT_PUBLISHED')

    def test_hash_binds_dtype_shape_and_empty_arrays(self):
        self.assertNotEqual(array_sha(np.empty((0, 3))), array_sha(np.empty(0)))
        self.assertNotEqual(array_sha(np.asarray([1], np.int32)), array_sha(np.asarray([1], np.float32)))

    def test_only_private_append_log_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent/'original'; original.mkdir()
            (original/'source.bin').write_bytes(b'original')
            (original/'log.txt').write_bytes(b'prior\n')
            private = PrivateCache(original, parent=parent)
            try:
                with (private.path/'log.txt').open('ab') as stream:
                    stream.write(b'native query\n')
                self.assertEqual(private.verify(1)['status'], 'PASS')
                self.assertEqual((original/'log.txt').read_bytes(), b'prior\n')
                (private.path/'source.bin').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    private.verify(1)
            finally:
                target = private.temporary
                private.remove()
                self.assertFalse(target.exists())
                self.assertTrue(original.is_dir())

    def test_private_log_prefix_rewrite_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent/'original'; original.mkdir()
            (original/'log.txt').write_bytes(b'abc')
            private = PrivateCache(original, parent=parent)
            try:
                (private.path/'log.txt').write_bytes(b'xyz')
                with self.assertRaises(ValueError):
                    private.verify(1)
            finally:
                private.remove()


if __name__ == '__main__':
    unittest.main()
