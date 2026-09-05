#!/usr/bin/env python3
"""Regression checks for camera transport and honest completion status."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("PILOT_REPO", "/home/warpwang/src/BinocMesher"))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="binoc-render-regression-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root / "scene"
        self.images = self.scene / "frames/Image/camera_0"
        self.flows = self.scene / "frames/Flow/camera_0"
        self.images.mkdir(parents=True)
        self.flows.mkdir(parents=True)
        rgb = np.random.default_rng(42).integers(0, 256, (32, 40, 3), dtype=np.uint8)
        for frame in range(1, 4):
            cv2.imwrite(str(self.images / f"Image_0_0_{frame:04d}_0.png"), rgb)
        for frame in (1, 2):
            np.save(self.flows / f"Flow_0_0_{frame:04d}_0.npy", np.zeros((32, 40, 3), np.float32))

    def score(self):
        return subprocess.run([
            sys.executable, str(HERE / "warped_ssim.py"), str(self.scene),
            "--output", str(self.root / "score"), "--expected-frames", "3",
        ], capture_output=True, text=True, timeout=30)

    def test_complete_pair_set(self):
        result = self.score()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.root / "score/summary.json").read_text())
        self.assertEqual(summary["pairs_scored"], 2)
        self.assertAlmostEqual(summary["warped_valid"]["mean"], 1.0)

    def test_missing_flow_fails_instead_of_partial_success(self):
        (self.flows / "Flow_0_0_0002_0.npy").unlink()
        result = self.score()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required forward-flow frames: [2]", result.stderr)
        self.assertFalse((self.root / "score/summary.json").exists())

    def test_blender_flow_translation_convention(self):
        image = cv2.imread(str(self.images / "Image_0_0_0001_0.png"))
        flow = np.zeros((32, 40, 3), np.float32)
        flow[..., 0] = -4  # Blender export for a four-pixel rightward motion.
        flow[..., 1] = 3
        for frame in (1, 2):
            following = np.zeros_like(image)
            following[3:, 4:] = image[:-3, :-4]
            cv2.imwrite(str(self.images / f"Image_0_0_{frame+1:04d}_0.png"), following)
            np.save(self.flows / f"Flow_0_0_{frame:04d}_0.npy", flow)
            image = following
        result = self.score()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.root / "score/summary.json").read_text())
        self.assertAlmostEqual(summary["warped_valid"]["mean"], 1.0)

    def test_runner_failure_and_existing_output_protection(self):
        output = self.root / "failed-batch"
        env = dict(os.environ, PILOT_REPO=str(REPO), PILOT_PYTHON="/bin/false")
        command = ["bash", str(HERE / "run_forest96_pilot.sh"), str(output), "binoc"]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertTrue((output / "FAILED").is_file())
        self.assertFalse((output / "FINISHED").exists())
        status = (output / "status.tsv").read_bytes()
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertEqual((output / "status.tsv").read_bytes(), status)


class CameraAdapterTests(unittest.TestCase):
    def test_both_adapters_accept_24_and_96_timestamped_cameras(self):
        sys.path.insert(0, str(REPO / "infinigen_binocmesher"))
        from infinigen.terrain import core
        for count in (24, 96):
            full_info = (
                np.repeat(np.eye(4)[None], count, axis=0),
                np.repeat(np.array([[40, 0, 32], [0, 40, 16], [0, 0, 1]])[None], count, axis=0),
                [32] * count, [64] * count, np.arange(count) / 24,
            )
            for cls in (core.OcMesher, core.CollectiveOcMesher):
                with self.subTest(adapter=cls.__name__, cameras=count):
                    with patch.object(core, "get_caminfo", return_value=(full_info,)):
                        mesher = cls([], [-4, 4, -4, 4, -4, 4], pixels_per_cube=30)
                    self.assertEqual(mesher.n_cameras, count)
                    self.assertEqual(mesher.cameras.size, count * 23)


if __name__ == "__main__":
    unittest.main(verbosity=2)
