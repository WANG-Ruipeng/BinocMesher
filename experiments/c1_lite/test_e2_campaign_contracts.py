import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from e2_image_metrics import gain_check
from run_schedule_e2_images import preview_changes


class CampaignContractTests(unittest.TestCase):
    def test_missing_coverage_cannot_be_a_positive_gain(self):
        good = {'missing_reference_coverage_pixels': 0, 'extra_coverage_pixels': 0,
                'both_valid_pixels': 4, 'depth': {'mean': 1.}, 'normal_degrees': {'mean': 1.}}
        incomplete = dict(good, missing_reference_coverage_pixels=1, both_valid_pixels=3)
        for row in gain_check(good, incomplete, good, incomplete).values():
            self.assertEqual(row['status'], 'UNRESOLVED_COVERAGE_MISMATCH')
            self.assertFalse(row['positive_gain_exceeds_twice_refinement_change'])

    def test_preview_count_is_pixels_not_color_channels(self):
        with tempfile.TemporaryDirectory(prefix='e2-preview-test-') as temporary:
            first, second = Path(temporary)/'a', Path(temporary)/'b'
            first.mkdir(); second.mkdir()
            for name in ('depth', 'normal', 'silhouette'):
                a = np.zeros((2, 2, 3), np.uint8)
                b = a.copy(); b[0, 0, :] = 1
                Image.fromarray(a).save(first/(name+'.png'))
                Image.fromarray(b).save(second/(name+'.png'))
            counts = preview_changes(first, second)
            for name in ('depth', 'normal', 'silhouette'):
                self.assertEqual(counts[name+'_changed_pixels'], 1)


if __name__ == '__main__':
    unittest.main()
