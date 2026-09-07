from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_schedule_coherence import (natural_schedule, controlled_centroid_document,
    disk_boundary, source_relations, forced_late_failure, timed_runtime, same_mesh, fj,
    audit_cache_side_effects)


def source_fixture():
    cycle = ['0:0|100:0', '1:0|101:0', '2:0|102:0', '3:0|103:0']
    positions = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]
    cell = {'boundary_cycle': cycle, 'source_faces': [[cycle[i] for i in f] for f in ((0, 1, 2), (0, 2, 3))],
            'owners': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]}
    result = {'levels': {k: fj(t) for k, t in zip(('lower', 'root', 'upper'), (0, 1, 2))},
        'boundary_cycle': cycle, 'cache_input_sha256': 'synthetic',
        'junction_boundary_cycle_and_orientation_equal': True,
        'anchors': {k: {'position': [fj(x) for x in (1, 1, 0)]} for k in ('lower', 'root', 'upper')},
        'segments': [], 'breakpoint_points': []}
    for a, b in ((0, 1), (1, 2)):
        result['segments'].append({'t0': fj(a), 't1': fj(b), **deepcopy(cell),
            'boundary': [{'source_vid': vid, 'position_t0': [fj(x) for x in p], 'position_t1': [fj(x) for x in p],
                          'slope': [fj(0)]*3, 'intercept': [fj(x) for x in p]} for vid, p in zip(cycle, positions)]})
    for t in (0, 1, 2):
        result['breakpoint_points'].append({'time': fj(t), **deepcopy(cell),
            'boundary': [{'source_vid': vid, 'position': [fj(x) for x in p]} for vid, p in zip(cycle, positions)]})
    return result


class CoherenceTests(unittest.TestCase):
    def test_top_level_runtime_log_append_is_allowed(self):
        before = {'log.txt': 'old', 'slicing_preprocess.manifest.json': 'same', 'hypervertices/0.bin': 'source'}
        after = {**before, 'log.txt': 'new'}
        report = audit_cache_side_effects(before, after, b'old records\n',
            b'old records\nprocessing 8\nbefore merging faces, face count: 5300\nafter merging faces, face count: 3818\n')
        self.assertTrue(report['allowed_runtime_log_mutation'])
        self.assertEqual(report['changed_relative_paths'], ['log.txt'])
        self.assertFalse(report['all_cache_file_hashes_unchanged'])
        self.assertEqual(report['log_append_record_count'], 3)
        measured = audit_cache_side_effects(before, after, b'old records\n',
            b'old records\nload_vertices  - Time taken: 0.000220 seconds\n, RSS Memory Usage: 0.089188 GB\n')
        self.assertTrue(measured['log_syntax_is_not_an_input_integrity_gate'])

    def test_cache_manifest_source_or_new_path_change_is_rejected(self):
        before = {'log.txt': 'same', 'slicing_preprocess.manifest.json': 'old', 'hypervertices/0.bin': 'old'}
        for name in ('slicing_preprocess.manifest.json', 'hypervertices/0.bin', 'new-output.json'):
            with self.subTest(path=name), self.assertRaisesRegex(ValueError, 'non-log'):
                audit_cache_side_effects(before, {**before, name: 'new'}, b'', b'')

    def test_log_rewrite_deletion_and_oversize_are_rejected(self):
        before, after = {'log.txt': 'a'}, {'log.txt': 'b'}
        for old, new in ((b'old\n', b'changed\n'),
                         (b'', b'processing 1\n'*11000)):
            with self.subTest(size=len(new)), self.assertRaises(ValueError):
                audit_cache_side_effects(before, after, old, new)
        with self.assertRaisesRegex(ValueError, 'deleted'):
            audit_cache_side_effects(before, {}, b'', b'')

    def test_natural_time_phase_is_not_event_aligned(self):
        rows = natural_schedule(float.fromhex('0x1.eaabfa360338dp-6'))
        self.assertEqual([r['frame'] for r in rows[:5]], list(range(13, 18)))
        for row in rows[:5]:
            self.assertEqual(row['value'].hex(), (float((row['frame']+.5)/24)-float(.5/24)).hex())
        self.assertEqual(rows[-1]['value'], F(104, 5))
        self.assertEqual(rows[-1]['kind'], 'exact_root')

    def test_declared_natural_schedule_has_one_active_sample(self):
        rows = natural_schedule(float.fromhex('0x1.eaabfa360338dp-6'))
        active = [r['frame'] for r in rows[:5] if F(102, 5) < F(r['metadata']['evaluation_tau']) < F(106, 5)]
        self.assertEqual(active, [15])

    def test_disk_orientation_not_unordered_boundary(self):
        source = source_fixture()
        disk_boundary(source['segments'][0]['source_faces'], source['boundary_cycle'])
        with self.assertRaises(ValueError):
            disk_boundary(source['segments'][0]['source_faces'], list(reversed(source['boundary_cycle'])))

    def test_all_declared_cells_and_root_tie(self):
        report = source_relations(source_fixture())
        self.assertEqual(report['all_declared_cells_checked'], 5)
        self.assertTrue(report['root_tie_uses_one_exact_anchor'])
        self.assertFalse(report['raw_cache_owner_coverage_checked'])
        json.dumps(report, allow_nan=False)

    def test_unmatched_junction_refused(self):
        source = source_fixture()
        source['segments'][0]['boundary'][0]['position_t1'][0] = fj(1)
        with self.assertRaises(ValueError):
            source_relations(source)

    def test_wrong_affine_coefficient_refused_even_endpoints_unchanged(self):
        source = source_fixture()
        source['segments'][0]['boundary'][0]['slope'][0] = fj(1)
        with self.assertRaises(ValueError):
            source_relations(source)

    def test_endpoint_anchor_not_midpoint_refused(self):
        source = source_fixture()
        source['anchors']['upper']['position'][2] = fj(1)
        with self.assertRaises(ValueError):
            source_relations(source)

    def test_centroid_changes_only_root_anchor(self):
        source = source_fixture()
        source['anchors']['root']['position'][0] = fj(F(5, 4))
        control = controlled_centroid_document(source)
        self.assertEqual(control['anchors']['root']['position'], [fj(1), fj(1), fj(0)])
        source['anchors']['root'] = control['anchors']['root']
        self.assertEqual(source, control)

    def test_forced_late_rejection_returns_all_baselines(self):
        from test_window_runtime import fake_runtime
        runtime, base, spec = fake_runtime()
        report = forced_late_failure(runtime, [0.25, 0.75], spec, [base, base])
        self.assertTrue(report['pass'])
        self.assertEqual(report['report']['discarded_proposals'], 2)
        restored, valid = runtime.run([0.25, 0.75], spec, time_mode='physical')
        self.assertEqual(valid['status'], 'COMMITTED_REQUESTED_SCHEDULE')
        self.assertFalse(same_mesh(restored[0], base))

    def test_timing_wrappers_restore_on_exception(self):
        from test_window_runtime import fake_runtime
        from binocmesher import window_runtime as module
        runtime, _, _ = fake_runtime()
        original = runtime._slice
        support, geometry = module.resolve_support, module.check_actual_patch
        with self.assertRaisesRegex(ValueError, 'injected'):
            with timed_runtime(runtime, module, {}):
                raise ValueError('injected')
        self.assertIs(runtime._slice, original)
        self.assertIs(module.resolve_support, support)
        self.assertIs(module.check_actual_patch, geometry)


if __name__ == '__main__':
    unittest.main()
