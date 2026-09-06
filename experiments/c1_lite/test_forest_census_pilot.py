import csv
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import forest_census_pilot as pilot


class ForestCensusTests(unittest.TestCase):
    def test_cli_help_does_not_run_blender(self):
        result = subprocess.run([sys.executable, '-B', str(Path(pilot.__file__)), '--help'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('forest64', result.stdout)

    def test_predeclared_64_frames_have_two_temporal_groups(self):
        times = [(frame-.5)/24 for frame in range(1, 65)]
        actual = pilot.forest_time_mapping(times)
        self.assertEqual(actual['temporal_group_count'], 2)
        self.assertEqual(actual['maximum_discrete_time'], 4)
        self.assertEqual(actual['duration_seconds'],
                         pilot.fraction_json(Fraction.from_float(63/24+1e-5)))


    def test_time_mapping_uses_real_short_forest_inputs(self):
        times = [(frame-.5)/24 for frame in range(1, 25)]
        actual = pilot.forest_time_mapping(times)
        self.assertEqual(actual['temporal_group_count'], 1)
        self.assertEqual(actual['maximum_discrete_time'], 2)
        self.assertEqual(actual['origin_seconds'], pilot.fraction_json(Fraction.from_float(times[0])))
        self.assertEqual(actual['delta_seconds'], pilot.fraction_json(Fraction.from_float((times[-1]-times[0]+1e-5)/2)))

    def test_fresh_output_rejects_overlap_and_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, repo = root/'source', root/'repo'
            source.mkdir(); repo.mkdir()
            with self.assertRaises(FileExistsError):
                pilot.require_fresh_output(source, source, repo)
            with self.assertRaises(ValueError):
                pilot.require_fresh_output(source/'nested', source, repo)
            self.assertEqual(pilot.require_fresh_output(root/'fresh', source, repo), root/'fresh')

    def make_registry(self, path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['canonical_event_id', 'root_num', 'root_den', 'logical_incidence_id', 'raw_id'])
            writer.writeheader()
            writer.writerows(rows)

    def test_registry_deduplicates_normalized_roots_without_merging_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'registry.csv'
            self.make_registry(path, [
                dict(canonical_event_id='a', root_num=2, root_den=4, logical_incidence_id='x', raw_id='r1'),
                dict(canonical_event_id='a', root_num=1, root_den=2, logical_incidence_id='x', raw_id='r2'),
                dict(canonical_event_id='b', root_num=1, root_den=2, logical_incidence_id='y', raw_id='r3')])
            actual = pilot.summarize_registry(path)
            self.assertEqual((actual['raw_observations'], actual['exact_roots'], actual['canonical_events']), (3, 1, 2))
            self.assertEqual(actual['logical_incidences'], 2)
            self.assertEqual(actual['unique_raw_observation_ids'], 3)
            self.assertTrue(all(e['beb1_classification'] == 'NOT_COMPILED' for e in actual['events']))

    def test_empty_registry_is_valid_zero_not_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'registry.csv'
            self.make_registry(path, [])
            actual = pilot.summarize_registry(path)
            self.assertEqual(actual['status'], 'REGISTRY_CENSUS_COMPLETE')
            self.assertEqual(actual['canonical_events'], 0)
            self.assertEqual(pilot.summarize_registry(Path(tmp)/'missing')['status'], 'MISSING_REGISTRY')

    def test_inconsistent_canonical_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'registry.csv'
            self.make_registry(path, [
                dict(canonical_event_id='a', root_num=1, root_den=2, logical_incidence_id='x', raw_id='r1'),
                dict(canonical_event_id='a', root_num=2, root_den=3, logical_incidence_id='x', raw_id='r2')])
            with self.assertRaises(ValueError):
                pilot.summarize_registry(path)


if __name__ == '__main__':
    unittest.main()
