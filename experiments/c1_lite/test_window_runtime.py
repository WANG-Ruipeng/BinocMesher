"""Independent synthetic runtime transaction and exact-contact regressions.

The slicer is fake: these tests never initialize native geometry or read a cache.
"""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binocmesher.window_runtime import WindowRuntime, WindowSpec, round_binary32
from binocmesher.window_geometry_runtime import check_actual_patch, _contact, _p


def _canon(row):
    row = tuple(row)
    return min(row[i:]+row[:i] for i in range(3))


def fixture():
    vertices = np.asarray([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
                           [-1, -1, 0], [3, -1, 0], [3, 3, 0], [-1, 3, 0]], float)
    rows = [(0, 1, 2), (0, 2, 3)]
    for i in range(4):
        j = (i+1) % 4
        rows += [(j, i, i+4), (j, i+4, j+4)]
    faces = np.asarray([_canon(row) for row in rows], np.int32)
    tags = np.ones(len(vertices), np.int32)
    vids = np.asarray([(i, 0, i+100, 0) for i in range(len(vertices))], np.int32)
    owners = np.asarray([(0, 0, 0, i, 0, 0, 0, i, *face)
                         for i, face in enumerate(faces)] +
                        [(0, 0, 0, 100, 0, 0, 0, 0, *faces[0])], np.int32)
    cycle = [f'{i}:0|{i+100}:0' for i in range(4)]
    cell = {'owners': [owners[i, :7].tolist() for i in (0, 1, len(owners)-1)],
            'source_faces': [[cycle[i] for i in face] for face in faces[:2]],
            'boundary_cycle': cycle}
    doc = {
        'levels': {'lower': 0, 'root': 1, 'upper': 2},
        'boundary_cycle': cycle,
        'anchors': {key: {'position': [1, 1, 0]} for key in ('lower', 'root', 'upper')},
        'segments': [{'t0': 0, 't1': 1, **cell}, {'t0': 1, 't1': 2, **cell}],
        'breakpoint_points': [{'time': t, **cell} for t in (0, 1, 2)],
        'cache_input_sha256': 'synthetic-no-cache',
    }
    return (vertices, faces, tags), vids, owners, WindowSpec.from_source_contract(doc), doc


def fake_runtime(*, bad_at=None):
    base, vids, owners, spec, _ = fixture()
    runtime = object.__new__(WindowRuntime)
    runtime.mesher = SimpleNamespace(_window_delta_t=0.5, _window_n_elements=1)
    runtime._cache_digest = Mock(return_value=spec.cache_digest)
    def call(value, mode, smooth, ledger):
        identity = 0 if value == bad_at else 1
        return tuple(a.copy() for a in base), vids.copy(), owners.copy(), identity
    runtime._slice = Mock(side_effect=call)
    return runtime, base, spec


def assert_bits(test, first, second):
    for a, b in zip(first, second):
        test.assertEqual(a.shape, b.shape)
        test.assertEqual(a.dtype, b.dtype)
        test.assertEqual(a.tobytes(), b.tobytes())


class WindowTransactionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', {'BINOC_SOURCE_SPLICE_PLAN': ''})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_successful_whole_schedule_including_unchanged_endpoints(self):
        runtime, base, spec = fake_runtime()
        output, report = runtime.run([F(0), F(1, 2), F(1), F(3, 2), F(2)], spec)
        self.assertEqual(report['status'], 'COMMITTED_REQUESTED_SCHEDULE', report)
        self.assertFalse(report['continuous_window_admitted'])
        self.assertFalse(report['published_partial_results'])
        assert_bits(self, output[0], base)
        assert_bits(self, output[-1], base)
        for mesh in output[1:-1]:
            self.assertEqual(len(mesh[0]), len(base[0])+1)
            self.assertEqual(len(mesh[1]), len(base[1])+2)
            assert_bits(self, (mesh[0][:-1], mesh[1][2:len(base[1])], mesh[2][:-1]),
                        (base[0], base[1][2:], base[2]))

    def test_late_identity_rejection_rolls_back_every_query(self):
        runtime, base, spec = fake_runtime(bad_at=F(3, 2))
        output, report = runtime.run([F(1, 2), F(3, 2), F(2)], spec)
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        self.assertEqual(runtime._slice.call_count, 3)
        self.assertGreaterEqual(report['discarded_proposals'], 1)
        for mesh in output:
            assert_bits(self, mesh, base)

    def test_late_geometry_rejection_rolls_back_earlier_proposal(self):
        runtime, base, spec = fake_runtime()
        from binocmesher import window_runtime as module
        original = module.propose_actual
        def proposal(mesh, ids, owners, proposal_spec, tau):
            if tau == F(3, 2):
                return None, {'status': 'REFUSED', 'geometry': {'status': 'UNKNOWN'}}
            return original(mesh, ids, owners, proposal_spec, tau)
        with patch.object(module, 'propose_actual', side_effect=proposal):
            output, report = runtime.run([F(1, 2), F(3, 2)], spec)
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        for mesh in output:
            assert_bits(self, mesh, base)

    def test_late_invalid_endpoint_rolls_back_interior(self):
        runtime, base, spec = fake_runtime()
        changed = replace(spec, anchors=(spec.anchors[0], spec.anchors[1], (F(1), F(1), F(1))))
        output, report = runtime.run([F(1, 2), F(2)], changed)
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        for mesh in output:
            assert_bits(self, mesh, base)

    def test_disabled_fallback_same_queries_modes_and_arrays(self):
        runtime, base, spec = fake_runtime()
        output, report = runtime.run([0.25, 0.75], spec, time_mode='physical', enabled=False)
        self.assertEqual(report['fallback_reason'], 'DISABLED')
        self.assertEqual(runtime._slice.call_args_list[0].args, (0.25, 'physical', False, False))
        for mesh in output:
            assert_bits(self, mesh, base)

    def test_smooth_fallback_does_not_silently_switch_to_raw(self):
        runtime, base, spec = fake_runtime()
        output, report = runtime.run([F(1, 2)], spec, extra_smooth=True)
        self.assertEqual(report['fallback_reason'], 'UNSUPPORTED_EXTRA_SMOOTH')
        self.assertEqual(runtime._slice.call_args.args, (F(1, 2), 'exact', True, False))
        assert_bits(self, output[0], base)

    def test_invalid_non_spec_fallback(self):
        runtime, base, _ = fake_runtime()
        output, report = runtime.run([F(1, 2)], {'pass': True})
        self.assertEqual(report['fallback_reason'], 'INVALID_SPEC')
        assert_bits(self, output[0], base)

    def test_direct_dataclass_invalid_layout_falls_back(self):
        runtime, base, spec = fake_runtime()
        bad = replace(spec, root=spec.lower)
        output, report = runtime.run([F(1, 2)], bad)
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        assert_bits(self, output[0], base)

    def test_direct_dataclass_missing_singleton_falls_back(self):
        runtime, base, spec = fake_runtime()
        output, report = runtime.run([F(1)], replace(spec, points=()))
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        assert_bits(self, output[0], base)

    def test_missing_physical_scale_falls_back(self):
        runtime, base, spec = fake_runtime()
        del runtime.mesher._window_delta_t
        output, report = runtime.run([0.25], spec, time_mode='physical')
        self.assertEqual(report['fallback_reason'], 'MISSING_ACTUAL_DELTA_T')
        assert_bits(self, output[0], base)

    def test_wrong_cache_falls_back_before_any_proposal(self):
        runtime, base, spec = fake_runtime()
        runtime._cache_digest.return_value = 'different-cache'
        with patch('binocmesher.window_runtime.propose_actual') as proposal:
            output, report = runtime.run([F(1, 2), F(3, 2)], spec)
        proposal.assert_not_called()
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        for mesh in output:
            assert_bits(self, mesh, base)

    def test_cache_changed_during_transaction_discards_every_proposal(self):
        runtime, base, spec = fake_runtime()
        runtime._cache_digest.side_effect = [spec.cache_digest, 'changed-after-preparation']
        output, report = runtime.run([F(1, 2), F(3, 2)], spec)
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        self.assertGreaterEqual(runtime._cache_digest.call_count, 2)
        for mesh in output:
            assert_bits(self, mesh, base)

    def test_cache_digest_failed_read_returns_baseline(self):
        runtime, base, spec = fake_runtime()
        runtime._cache_digest.side_effect = OSError('synthetic cache read failure')
        output, report = runtime.run([F(1, 2), F(3, 2)], spec)
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        for mesh in output:
            assert_bits(self, mesh, base)

    def test_cache_final_digest_failed_read_discards_proposals(self):
        runtime, base, spec = fake_runtime()
        runtime._cache_digest.side_effect = [spec.cache_digest, OSError('synthetic final read failure')]
        output, report = runtime.run([F(1, 2), F(3, 2)], spec)
        self.assertEqual(report['status'], 'BASELINE_ENTIRE_SCHEDULE')
        self.assertGreaterEqual(runtime._cache_digest.call_count, 2)
        for mesh in output:
            assert_bits(self, mesh, base)

    def test_physical_bounds_and_nextafter_outside_are_unchanged(self):
        runtime, base, spec = fake_runtime()
        queries = [np.nextafter(0., -np.inf), 0., 0.25, 1., np.nextafter(1., np.inf)]
        output, report = runtime.run(queries, spec, time_mode='physical')
        self.assertEqual(report['status'], 'COMMITTED_REQUESTED_SCHEDULE', report)
        for i in (0, 1, 3, 4):
            assert_bits(self, output[i], base)
        self.assertEqual(len(output[2][0]), len(base[0])+1)
        self.assertIn('physical_time_hex', report['cases'][2])

    def test_no_proposal_or_result_state_leaks_between_calls(self):
        runtime, base, spec = fake_runtime()
        first, _ = runtime.run([F(1, 2)], spec)
        first[0][0][0] = [999, 999, 999]
        fallback, report = runtime.run([F(1, 2)], None)
        assert_bits(self, fallback[0], base)
        self.assertEqual(report['fallback_reason'], 'INVALID_SPEC')
        third, report = runtime.run([F(1, 2)], spec)
        self.assertEqual(report['status'], 'COMMITTED_REQUESTED_SCHEDULE')
        self.assertTrue(np.array_equal(third[0][0][:len(base[0])], base[0]))

    def test_ordinary_slice_failure_is_not_fabricated_baseline(self):
        runtime, _, spec = fake_runtime()
        runtime._slice.side_effect = RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            runtime.run([F(1, 2)], spec)

    def test_old_ssp1_intervention_is_rejected_before_slice(self):
        runtime, _, spec = fake_runtime()
        with patch.dict('os.environ', {'BINOC_SOURCE_SPLICE_PLAN': 'old.ssp1'}):
            with self.assertRaises(ValueError):
                runtime.run([F(1)], spec)
        runtime._slice.assert_not_called()

    def test_constructor_requires_initialized_single_element(self):
        for value in (None, 0, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                WindowRuntime(SimpleNamespace(_window_n_elements=value))


class NativeOutputCleanupTests(unittest.TestCase):
    """Run the actual Python _slice body against fake ctypes callables only."""
    def fake_native_runtime(self):
        dll = SimpleNamespace(**{
            name: Mock() for name in (
                'slicing_identity_enable', 'slicing_identity_status',
                'slicing_identity_last_error', 'slicing_identity_vertex_count',
                'slicing_identity_owner_count', 'slicing_identity_output_vertices',
                'slicing_identity_output_owners', 'slicing_discard_output')})
        dll.slicing_identity_status.return_value = 1
        dll.slicing_identity_vertex_count.return_value = 8
        dll.slicing_identity_owner_count.return_value = 11
        dll.slicing_identity_output_vertices.return_value = 0
        dll.slicing_identity_output_owners.return_value = 0
        def exact(numerator, denominator, vc, fc, smooth):
            vc[0], fc[0] = 8, 10
            return 0
        mesher = SimpleNamespace(
            _runtime_library=dll, _window_n_elements=1,
            run_slicing_rational=Mock(side_effect=exact),
            slicing_output=Mock(), slicing_clean_up=Mock(),
            slicing_last_error=Mock(return_value=b'fake native failure'),
            AF=lambda array: array)
        return WindowRuntime(mesher), mesher, dll

    def assert_discarded(self, mesher, dll):
        dll.slicing_discard_output.assert_called_once_with()
        self.assertEqual(dll.slicing_identity_enable.call_args.args, (False,))
        self.assertEqual(mesher.run_slicing_rational.call_count, 1)

    def test_successful_native_slice_then_memory_budget_failure_discards_output(self):
        runtime, mesher, dll = self.fake_native_runtime()
        with patch('binocmesher.window_runtime.MAX_BATCH_BYTES', 8):
            with self.assertRaises(ValueError):
                runtime._slice(F(1), 'exact', False, True)
        self.assert_discarded(mesher, dll)
        mesher.slicing_output.assert_not_called()

    def test_successful_native_slice_then_invalid_owner_count_discards_output(self):
        runtime, mesher, dll = self.fake_native_runtime()
        dll.slicing_identity_owner_count.return_value = -1
        with self.assertRaises(ValueError):
            runtime._slice(F(1), 'exact', False, True)
        self.assert_discarded(mesher, dll)
        mesher.slicing_output.assert_not_called()

    def test_successful_native_slice_then_ledger_output_failure_discards_output(self):
        runtime, mesher, dll = self.fake_native_runtime()
        dll.slicing_identity_output_vertices.return_value = -1
        with self.assertRaises(ValueError):
            runtime._slice(F(1), 'exact', False, True)
        self.assert_discarded(mesher, dll)
        mesher.slicing_output.assert_not_called()

    def test_python_mesh_output_exception_discards_native_output(self):
        runtime, mesher, dll = self.fake_native_runtime()
        mesher.slicing_output.side_effect = RuntimeError('fake output copy failure')
        with self.assertRaisesRegex(RuntimeError, 'fake output copy failure'):
            runtime._slice(F(1), 'exact', False, True)
        self.assert_discarded(mesher, dll)


class ActualContactTests(unittest.TestCase):
    def setUp(self):
        self.mesh, _, _, _, _ = fixture()
        self.center = np.array([1., 1., 0.])

    def check(self, vertices=None, faces=None, center=None):
        return check_actual_patch(self.mesh[0] if vertices is None else vertices,
                                  self.mesh[1] if faces is None else faces,
                                  (0, 1), (0, 1, 2, 3),
                                  self.center if center is None else center)

    def test_complete_collar_and_all_retained_faces_checked(self):
        result = self.check()
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['retained_faces_checked'], len(self.mesh[1])-2)
        self.assertEqual(sum(result['certificate_counts'].values()), len(self.mesh[1])-2)

    def test_shared_edge_crossing_is_not_skipped(self):
        vertices = self.mesh[0].copy()
        vertices[4] = [1, 1, 0]
        result = self.check(vertices=vertices)
        self.assertNotEqual(result['status'], 'PASS', result)

    def test_shared_vertex_crossing_is_not_skipped(self):
        boundary = tuple(_p(p) for p in self.mesh[0][:4])
        triangle = tuple(_p(p) for p in ([0, 0, 0], [1, 1, 0], [1, 0.5, 0]))
        kind, _ = _contact(boundary, _p(self.center), (0, 1, 2, 3), triangle, (0, 100, 101), 1)
        self.assertIsNone(kind)

    def test_degenerate_shared_edge_neighbor_rejected(self):
        boundary = tuple(_p(p) for p in self.mesh[0][:4])
        triangle = tuple(_p(p) for p in ([0, 0, 0], [2, 0, 0], [1, 0, 0]))
        kind, reason = _contact(boundary, _p(self.center), (0, 1, 2, 3), triangle, (0, 1, 100), 1)
        self.assertIsNone(kind)
        self.assertIn('Degenerate', reason)

    def test_nonshared_triangle_hidden_at_end_of_retained_list(self):
        v = np.vstack((self.mesh[0], [[0.5, 0.5, 0], [1, 0.5, 0], [0.5, 1, 0]]))
        f = np.vstack((self.mesh[1], [8, 9, 10]))
        result = self.check(v, f)
        self.assertEqual(result['status'], 'UNKNOWN', result)
        self.assertEqual(result['failing_retained_face'], len(f)-1)
        self.assertEqual(result['retained_faces_checked'], len(self.mesh[1])-2)

    def test_one_subnormal_aabb_gap_is_exactly_separated(self):
        gap = float(np.nextafter(np.float32(0), np.float32(1)))
        v = np.vstack((self.mesh[0], [[0.5, 0.5, gap], [1, 0.5, gap], [0.5, 1, gap]]))
        f = np.vstack((self.mesh[1], [8, 9, 10]))
        result = self.check(v, f)
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['certificate_counts']['STRICT_ACTUAL_AABB'], 1)
        self.assertEqual(result['retained_faces_checked'], len(f)-2)

    def test_aabb_touch_not_strict_separation(self):
        v = np.vstack((self.mesh[0], [[2, 1, 0], [3, 1, 0], [3, 1.5, 0]]))
        f = np.vstack((self.mesh[1], [8, 9, 10]))
        result = self.check(v, f)
        self.assertNotEqual(result['status'], 'PASS', result)

    def test_center_is_included_in_exterior_checks(self):
        v = np.vstack((self.mesh[0], [[0.5, 0.5, 1], [1.5, 0.5, 1], [1, 1.5, 1]]))
        f = np.vstack((self.mesh[1], [8, 9, 10]))
        result = self.check(v, f, np.array([1., 1., 2.]))
        self.assertNotEqual(result['status'], 'PASS', result)

    def test_binary64_only_coordinate_cannot_pass_binary32_contract(self):
        v = self.mesh[0].copy()
        v[7, 0] = np.nextafter(v[7, 0], 0.)
        self.assertEqual(self.check(vertices=v)['status'], 'REJECT')

    def test_center_on_boundary_rejected(self):
        self.assertEqual(self.check(center=np.array([1., 0., 0.]))['status'], 'REJECT')


class Binary32RoundingTests(unittest.TestCase):
    def test_even_midpoint_and_tiny_rational_offset(self):
        middle = F(1)+F(1, 2**24)
        self.assertEqual(round_binary32(middle), 1.)
        self.assertEqual(round_binary32(middle+F(1, 2**80)),
                         float(np.nextafter(np.float32(1), np.float32(2))))
        self.assertEqual(round_binary32(-middle-F(1, 2**80)),
                         -float(np.nextafter(np.float32(1), np.float32(2))))

    def test_odd_midpoint_rounds_to_even_upper_float(self):
        a = np.nextafter(np.float32(1), np.float32(2))
        b = np.nextafter(a, np.float32(2))
        self.assertEqual(round_binary32((F.from_float(float(a))+F.from_float(float(b)))/2), float(b))

    def test_subnormal_tie_and_above(self):
        self.assertEqual(round_binary32(F(1, 2**150)), 0.)
        self.assertEqual(round_binary32(F(1, 2**150)+F(1, 2**200)),
                         float(np.nextafter(np.float32(0), np.float32(1))))

    def test_nonfinite_overflow_refused(self):
        with np.errstate(over='ignore'), self.assertRaises(ValueError):
            round_binary32(F(2)**128)


if __name__ == '__main__':
    unittest.main()
