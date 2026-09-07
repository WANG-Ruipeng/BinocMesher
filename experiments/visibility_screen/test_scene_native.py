"""Pure and fake-ABI tests; no production library or real cache is opened."""
from fractions import Fraction as F
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

import scene_native as m


def fj(value):
    value = F(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def documents(n=3, start_frame=97):
    times = [(start_frame+i-.5)/24 for i in range(64)]
    origin = min(times); duration = max(t-origin for t in times)+1e-5
    camera = {'poses': [np.eye(4).tolist() for _ in times], 'intrinsics': [np.eye(3).tolist() for _ in times],
        'widths': [960]*64, 'heights': [540]*64, 'times_seconds': times,
        'time_mapping': {'use_alignment': False, 'min_t_offset': 0, 'origin_seconds': fj(origin),
            'duration_seconds': fj(duration), 'delta_seconds': fj(duration/4),
            'temporal_group_count': 2, 'maximum_discrete_time': 4, 'effective_fading_seconds': 1.}}
    elements = ['opaque'+str(i) for i in range(n)]
    effective = {'opaque_elements': elements, 'terrain_elements': ['transparent_first', *elements, 'atmosphere'],
        'bounds': [-10, 10, -20, 20, -30, 30],
        'mesher_parameters': {'pixels_per_cube': 6, 'pixels_per_cube_coarse': 30, 'pixels_per_cube_outview': 120,
            'min_dist': .1, 'fading_time': 1, 'relax_iters': 6, 'use_alignment': False, 'min_t_offset': 0}}
    return camera, effective


class FakeDll:
    def __init__(self, n=3, version=2, identity=1):
        self.n, self.version, self.identity = n, version, identity
        self.requested = False; self.output_started = False; self.events = []
        self.failure_status = 0; self.bad_count = False; self.bad_owner = False
        self.bad_ledger_count = False; self.bad_face = False; self.bad_coordinate = False
        self.shift_status = 0; self.expected_smooth = False
        self.unused_shift_nonzero = False; self.shift_capacities = []
        self.shifts = np.arange(n*4, dtype=np.int32).reshape(n, 4)*13
    def slicing_identity_enable(self, value):
        self.requested = value; self.events.append(('enable', value))
    def _run(self, vc, fc, smooth):
        assert smooth is self.expected_smooth
        self.events.append(('run', smooth)); self.output_started = False
        np.ctypeslib.as_array(vc, shape=(self.n,))[:] = -1 if self.bad_count else 3
        np.ctypeslib.as_array(fc, shape=(self.n,))[:] = 1
        return self.failure_status
    def run_slicing_rational(self, num, den, vc, fc, smooth):
        self.events.append(('rational', num, den)); return self._run(vc, fc, smooth)
    def run_slicing(self, value, vc, fc, smooth):
        self.events.append(('physical', value)); return self._run(vc, fc, smooth)
    def slicing_last_error(self): return b'fake ordinary error'
    def slicing_identity_status(self): return self.identity if self.requested else 0
    def slicing_identity_last_error(self): return b'bounded capacity unavailable' if self.identity == 3 else b''
    def slicing_identity_vertex_count(self, e): return 2 if self.bad_ledger_count else 3
    def slicing_identity_owner_count(self, e): return 2
    def slicing_identity_output_vertices(self, e, pointer, capacity):
        assert not self.output_started and capacity == 12
        self.events.append(('ids', e))
        np.ctypeslib.as_array(pointer, shape=(capacity,)).reshape(-1, 4)[:] = [[101+e, 2, 203+e, 3], [105+e, 4, 206+e, 5], [109+e, 6, 210+e, 7]]
        return 0
    def slicing_identity_output_owners(self, e, pointer, capacity):
        assert not self.output_started and capacity == 22
        self.events.append(('owners', e))
        a = np.ctypeslib.as_array(pointer, shape=(capacity,)).reshape(-1, 11)
        a[:] = [[e, 0, 0, i, 0, 0, 0, 0, 0, 1, 2] for i in range(2)]
        if self.bad_owner: a[0, 10] = 1
        return 0
    def slicing_identity_source_vid_encoding_version(self): return self.version
    def slicing_identity_output_source_vid_shifts(self, pointer, capacity):
        assert not self.output_started
        self.shift_capacities.append(capacity)
        self.events.append(('shifts',))
        if self.shift_status: return self.shift_status
        # Literal frozen runtime_identity.h ABI: five rows, not actual n rows.
        if capacity < 20: return -1
        assert self.n <= 5
        target = np.ctypeslib.as_array(pointer, shape=(capacity,))
        target[:20] = 0
        target[:self.n*4] = self.shifts.reshape(-1)
        if self.unused_shift_nonzero and self.n < 5: target[self.n*4] = 1
        return 0
    def slicing_output(self, e, vertices, faces, tags):
        self.output_started = True; self.events.append(('mesh', e))
        np.ctypeslib.as_array(vertices, shape=(9,))[:] = [e, 0, 0, e+1, 0, 0, e, 1, 0]
        if self.bad_coordinate: np.ctypeslib.as_array(vertices, shape=(9,))[0] = .1
        np.ctypeslib.as_array(faces, shape=(3,))[:] = [99 if self.bad_face else 0, 1, 2]
        np.ctypeslib.as_array(tags, shape=(3,))[:] = [0, 1, 1]
    def slicing_discard_output(self): self.events.append(('discard',))
    def slicing_clean_up(self): self.events.append(('cleanup',))


def reader(dll, *, smooth=False):
    camera, effective = documents(dll.n)
    return m.NativeSceneReader(dll, m.prepare_parameters(camera, effective), Path('/tmp/fake_private'),
        Path('/tmp/fake_original'), Path('/tmp/fake.so'), '0'*64, {}, extra_smooth=smooth)


class SceneNativeTests(unittest.TestCase):
    def test_dynamic_opaque_order_not_forest_or_atmosphere_guess(self):
        camera, effective = documents(7)
        parameters = m.prepare_parameters(camera, effective)
        self.assertEqual(parameters['n_elements'], 7)
        self.assertEqual(parameters['opaque_elements'], tuple(effective['opaque_elements']))
        self.assertEqual(parameters['cameras'].shape, (64*27,))
        self.assertEqual(parameters['cameras'][23], 0.)
        self.assertEqual(parameters['origin_seconds'], (97-.5)/24)
        self.assertEqual(m.frozen.N_ELEMENTS, 5)

    def test_opaque_order_must_match_actual_terrain_sequence(self):
        camera, effective = documents()
        effective['opaque_elements'] = list(reversed(effective['opaque_elements']))
        with self.assertRaisesRegex(m.NativeReadError, 'OPAQUE_ORDER'): m.prepare_parameters(camera, effective)

    def test_duplicate_opaque_elements_not_counted_twice(self):
        camera, effective = documents(); effective['opaque_elements'].append(effective['opaque_elements'][0])
        with self.assertRaisesRegex(m.NativeReadError, 'ELEMENT_ORDER'): m.prepare_parameters(camera, effective)

    def test_frozen_profile_not_silently_changed(self):
        camera, effective = documents(); effective['mesher_parameters']['pixels_per_cube'] = 3
        with self.assertRaisesRegex(m.NativeReadError, 'PARAMETER_PROFILE'): m.prepare_parameters(camera, effective)
        camera, effective = documents(); camera['time_mapping']['maximum_discrete_time'] = 8
        with self.assertRaisesRegex(m.NativeReadError, 'TEMPORAL_PROFILE'): m.prepare_parameters(camera, effective)

    def test_duration_and_time_order_binding(self):
        camera, effective = documents(); camera['time_mapping']['duration_seconds'] = fj(3)
        with self.assertRaisesRegex(m.NativeReadError, 'DURATION_BITS'): m.prepare_parameters(camera, effective)
        camera, effective = documents(); camera['times_seconds'][1] = camera['times_seconds'][0]
        with self.assertRaisesRegex(m.NativeReadError, 'INVALID_CAMERA_OR_TIMES'): m.prepare_parameters(camera, effective)

    def test_all_dynamic_ledgers_before_any_mesh(self):
        for n in (1, 2, 3, 5):
            with self.subTest(n=n):
                dll = FakeDll(n); r = reader(dll); s = r.slice_query(F(7, 5))
                self.assertEqual(len(s['meshes']), n); self.assertEqual(s['n_elements'], n)
                first = dll.events.index(('mesh', 0))
                self.assertEqual(sum(e[0] in ('ids', 'owners') for e in dll.events[:first]), 2*n)
                self.assertLess(dll.events.index(('shifts',)), first)
                self.assertEqual(s['cost']['array_bytes'], 248*n)
                self.assertEqual(s['source_vid_shifts'].shape, (n, 4))
                self.assertEqual(dll.shift_capacities, [20])
                self.assertEqual(s['source_vid_shift_transport_capacity_ints'], 20)
                self.assertTrue(s['unused_source_vid_shift_slots_verified_zero'])
                self.assertTrue(s['identity_contract_verified']); self.assertFalse(s['extra_smooth'])
                for arrays in zip(s['meshes'], s['vertex_ledgers'], s['owner_ledgers']):
                    mesh, ids, owners = arrays
                    self.assertTrue(all(not a.flags.writeable for a in (*mesh, ids, owners)))
                r.close()

    def test_nonzero_shift_receipt_does_not_double_translate_vids(self):
        dll = FakeDll(5); r = reader(dll); first = r.slice_query(F(3, 2))
        self.assertEqual(first['vertex_ledgers'][4][0].tolist(), [105, 2, 207, 3])
        np.testing.assert_array_equal(first['source_vid_shifts'], dll.shifts)
        self.assertFalse(first['source_vid_shifts'].flags.writeable)
        dll.shifts[:] = 0; second = r.slice_query(F(5, 2))
        self.assertTrue(np.any(first['source_vid_shifts'])); self.assertFalse(np.any(second['source_vid_shifts']))
        r.close()

    def test_physical_entry_and_explicit_smooth_no_raw_admission(self):
        dll = FakeDll(2); dll.expected_smooth = True; r = reader(dll, smooth=True)
        s = r.slice_query(.25, mode='physical', ledger=False)
        self.assertTrue(s['extra_smooth']); self.assertFalse(s['raw_method_admission_claimed'])
        self.assertEqual(s['query']['effective_discrete_time_hex'], float(.25/r.delta_t).hex())
        self.assertFalse(s['identity_contract_verified']); r.close()

    def test_capacity_failure_preserves_complete_baseline_without_claim(self):
        dll = FakeDll(5, identity=3); r = reader(dll); s = r.slice_query(F(2))
        self.assertEqual(len(s['meshes']), 5); self.assertEqual(s['identity_status'], 3)
        self.assertFalse(s['identity_contract_verified']); self.assertIsNone(s['source_vid_shifts'])
        self.assertFalse(any(e[0] == 'owners' for e in dll.events)); r.close()

    def test_unrecognized_encoding_is_not_original(self):
        dll = FakeDll(3, version=7); r = reader(dll); s = r.slice_query(F(2))
        self.assertEqual(s['source_vid_encoding'], 'UNSUPPORTED_OBSERVER_SOURCE_VID_ENCODING')
        self.assertFalse(s['identity_contract_verified']); self.assertIsNone(s['source_vid_shifts']); r.close()

    def test_native_failure_cleans_and_can_retry(self):
        dll = FakeDll(); dll.failure_status = -1; r = reader(dll)
        with self.assertRaisesRegex(m.NativeReadError, 'SLICING_FAILED'): r.slice_query(F(2))
        self.assertEqual(dll.events[-3:], [('discard',), ('cleanup',), ('enable', False)])
        dll.failure_status = 0; self.assertEqual(r.slice_query(F(2))['identity_status'], 1); r.close()

    def test_negative_count_stops_before_output(self):
        dll = FakeDll(); dll.bad_count = True; r = reader(dll)
        with self.assertRaisesRegex(m.NativeReadError, 'MESH_COUNTS'): r.slice_query(F(2))
        self.assertFalse(any(e[0] == 'mesh' for e in dll.events)); r.close()

    def test_bad_identity_count_stops_before_output(self):
        dll = FakeDll(); dll.bad_ledger_count = True; r = reader(dll)
        with self.assertRaisesRegex(m.NativeReadError, 'IDENTITY_VERTEX_COUNT'): r.slice_query(F(2))
        self.assertFalse(any(e[0] == 'mesh' for e in dll.events)); r.close()

    def test_failed_shift_export_cleans_all_pending_state(self):
        dll = FakeDll(); dll.shift_status = -1; r = reader(dll)
        with self.assertRaisesRegex(m.NativeReadError, 'SHIFT_OUTPUT_FAILED'): r.slice_query(F(2))
        self.assertFalse(any(e[0] == 'mesh' for e in dll.events))
        self.assertEqual(dll.events[-3:], [('discard',), ('cleanup',), ('enable', False)]); r.close()

    def test_bad_owner_face_coordinate_never_yields_snapshot(self):
        for flag, reason in (('bad_owner', 'OWNER_ORIENTED_FACE'), ('bad_face', 'FACE_INDEX'), ('bad_coordinate', 'EXACT_BINARY32')):
            with self.subTest(flag=flag):
                dll = FakeDll(); setattr(dll, flag, True); r = reader(dll)
                with self.assertRaisesRegex(m.NativeReadError, reason): r.slice_query(F(2))
                r.close()

    def test_pinned_observer_more_than_five_elements_stops_before_native(self):
        for n in (6, 7):
            dll = FakeDll(n); r = reader(dll)
            with self.assertRaisesRegex(m.NativeReadError, 'ELEMENT_CAPACITY_UNSUPPORTED'): r.slice_query(F(2))
            self.assertFalse(dll.events); r.close()

    def test_unused_transport_slots_are_zero_or_no_output_is_published(self):
        dll = FakeDll(1); dll.unused_shift_nonzero = True; r = reader(dll)
        with self.assertRaisesRegex(m.NativeReadError, 'UNUSED_SOURCE_VID_SHIFT_SLOTS_NONZERO'): r.slice_query(F(2))
        self.assertFalse(any(e[0] == 'mesh' for e in dll.events))
        self.assertEqual(dll.events[-3:], [('discard',), ('cleanup',), ('enable', False)]); r.close()

    def test_literal_frozen_transport_rejects_small_capacity_without_writing(self):
        dll = FakeDll(1); buffer = np.full(4, -77, np.int32)
        self.assertEqual(dll.slicing_identity_output_source_vid_shifts(m.frozen._int_pointer(buffer), buffer.size), -1)
        np.testing.assert_array_equal(buffer, [-77]*4)

    def test_shift_transport_failure_preserves_native_error(self):
        dll = FakeDll(); dll.shift_status = -1; r = reader(dll)
        dll.slicing_identity_last_error = lambda: b'ready snapshot and 20-int capacity'
        with self.assertRaisesRegex(m.NativeReadError, '20-int capacity'): r.slice_query(F(2))
        self.assertFalse(any(e[0] == 'mesh' for e in dll.events)); r.close()

    def test_unused_transport_slots_are_not_exposed_by_public_array_base(self):
        dll = FakeDll(1); r = reader(dll); s = r.slice_query(F(2))
        self.assertEqual(s['source_vid_shifts'].shape, (1, 4))
        self.assertIsNone(s['source_vid_shifts'].base)
        self.assertEqual(s['source_vid_shifts'].nbytes, 16)
        self.assertEqual(len(s['meshes']), 1)
        r.close()

    def test_array_and_identity_capacity_budgets(self):
        with self.assertRaisesRegex(m.NativeReadError, 'EXCEED_2_GIB'): m.validate_counts([40_000_000]*2, [0]*2, 2)
        with self.assertRaisesRegex(m.NativeReadError, 'CAPACITY_OVERFLOW'): m.validate_counts([0], [0], 1, [300_000_000])
        with self.assertRaisesRegex(m.NativeReadError, 'SHAPE'): m.validate_counts([1]*5, [0]*5, 3)

    def test_closed_wrong_thread_and_out_of_range_stop(self):
        dll = FakeDll(); r = reader(dll)
        with self.assertRaisesRegex(m.NativeReadError, 'OUTSIDE'): r.slice_query(F(5))
        with self.assertRaisesRegex(m.NativeReadError, 'OUTSIDE'): r.slice_query(float('nan'), mode='physical')
        errors = []
        def other_thread():
            try: r.slice_query(F(2))
            except m.NativeReadError as error: errors.append(str(error))
        thread = threading.Thread(target=other_thread); thread.start(); thread.join()
        self.assertTrue(errors); self.assertFalse(dll.events)
        r.close(); r.close()
        with self.assertRaisesRegex(m.NativeReadError, 'CLOSED'): r.slice_query(F(2))

    def test_shared_active_guard_and_environment_restoration(self):
        dll = FakeDll(); r = reader(dll)
        with mock.patch.object(m.frozen, '_ACTIVE', r), mock.patch.dict(os.environ, {'BINOC_EVENT_MODE': '1'}):
            with self.assertRaisesRegex(m.NativeReadError, 'ONE_NATIVE_READER'):
                m.initialize_scene('not-opened', 'not-opened', {}, {}, original_cache='not-opened', expected_so_sha256='0'*64)
            r._saved_environment = {'BINOC_EVENT_MODE': 'previous'}
            r.close()
            self.assertEqual(os.environ['BINOC_EVENT_MODE'], 'previous'); self.assertIsNone(m.frozen._ACTIVE)

    def test_cache_paths_explicit_original_and_nonaliasing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original, private, build = root/'original', root/'private', root/'build'
            for path in (original, private, build): path.mkdir()
            manifest = {'provenance_enabled': True, 'bpm_version': 2, 'bhp_version': 2}
            for path in (original, private):
                (path/'slicing_preprocess.finish').write_bytes(b'')
                (path/'slicing_preprocess.manifest.json').write_text(json.dumps(manifest))
                (path/'input.bin').write_bytes(b'input')
            self.assertEqual(m.validate_cache_paths(private, original, build), (private, original, build))
            with self.assertRaisesRegex(m.NativeReadError, 'DISJOINT'): m.validate_cache_paths(original, original, build)
            with self.assertRaisesRegex(m.NativeReadError, 'OUTSIDE_CACHE'): m.validate_cache_paths(private, original, private)
            (private/'input.bin').unlink(); os.link(original/'input.bin', private/'input.bin')
            with self.assertRaisesRegex(m.NativeReadError, 'HARDLINK'): m.validate_cache_paths(private, original, build)


if __name__ == '__main__':
    unittest.main()
