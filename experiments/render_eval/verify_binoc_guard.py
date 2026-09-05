#!/usr/bin/env python3
"""Check the real cached-scene guard and audit per-frame slicing logs."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
parser.add_argument("--repo", type=Path, default=Path("/home/warpwang/src/BinocMesher"))
parser.add_argument("--audit-only", action="store_true")
args = parser.parse_args()
scene = args.run_root / "binoc/0"
result = {}

if not args.audit_only:
    output = args.run_root / "reject_multiframe"
    output.mkdir()
    command = [
        sys.executable, "-m", "infinigen_examples.generate_nature",
        "--input_folder", str(scene / "fine"),
        "--output_folder", str(output),
        "--seed", "0", "--task", "render", "--task_uniqname", "invalidgroup",
        "-g", "mountain", "monocular", "simple", "no_assets",
        "-p", f"LOG_DIR='{output}'", "execute_tasks.frame_range=[25,26]",
        "execute_tasks.camera_id=[0,0]", "execute_tasks.generate_resolution=[960,540]",
        "Terrain.device='cpu'", "full/configure_render_cycles.num_samples=8",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8",
               PYTHONPATH=str(args.repo / "infinigen_binocmesher"))
    with (output / "console.log").open("w") as log:
        run = subprocess.run(command, cwd=args.repo / "infinigen_binocmesher",
                             env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
    log = (output / "console.log").read_text()
    assert run.returncode != 0, "Invalid multi-frame task unexpectedly succeeded"
    assert "BinocMesher rendering requires one frame per task" in log, log[-3000:]
    assert "BinocMesher slicing frame=" not in log
    assert not (output / "FINISH_invalidgroup").exists()
    result["multiframe_rejected_before_slicing"] = True
else:
    records = []
    for frame in (25, 26):
        for task in ("shortrender", "blendergt"):
            key = f"{task}_0_0_{frame:04d}_0"
            assert (scene / "logs" / f"FINISH_{key}").exists(), key
            log = (scene / "logs" / f"{key}.err").read_text()
            assert f"Processing frames {frame} through {frame} inclusive" in log
            rows = re.findall(r"BinocMesher slicing frame=(\d+) time=([0-9.e+-]+)", log)
            assert len(rows) == 1, (key, rows)
            slice_frame, slice_time = rows[0]
            assert int(slice_frame) == frame
            assert abs(float(slice_time) - (frame - 0.5) / 24) < 1e-12
            records.append({"task": key, "frame": frame, "slice_time": float(slice_time)})
    result["per_frame_slicing"] = records
    summary = json.loads((args.run_root / "binoc/ssim/summary.json").read_text())
    assert summary["pairs_scored"] == 1
    assert (summary["frame_start"], summary["frame_end"]) == (25, 26)
    result["ssim_pairs"] = summary["pairs_scored"]

target = "binoc_slicing_audit.json" if args.audit_only else "binoc_guard_check.json"
(args.run_root / target).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
