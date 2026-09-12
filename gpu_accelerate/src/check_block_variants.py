"""Finite correctness checks for L1_32 launch-block variants; never records time.

Execution belongs to the supervising process.  This module contains no retries,
performance gate, calibration, timing API, or cleanup calls after a failure.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

from check_gpu_solver import API, ARRAYS, PU64, compare, digest, pointer, require
from performance_suite import load_execution_inputs

from workspace_paths import ROOT
MATRIX = ROOT / "configs/FROZEN_MATRIX.json"
PARAMETERS = ROOT / "inputs/accepted_scene/field_parameters.npz"
BRIDGE = ROOT / "build/libfield_bridge.so"
CHECKS = ((0, "first"), (0, "repeat"), (1, "first"), (1, "repeat"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def load_frozen():
    matrix = json.loads(MATRIX.read_text(encoding="utf8"))
    require(matrix["K"] == [3, 6], "Only frozen K=3/6 are supported")
    require(len(matrix["cases"]) == 18, "Expected exactly 18 real natural cases")
    require({r["field"] for r in matrix["cases"]} == {"LandTiles", "SdfTrees"},
            "Unexpected field scope")
    # This reused helper loads and compares the frozen captured native inputs.
    # It does not run the performance gate or initialize CUDA.
    cases, params = load_execution_inputs(matrix)
    require(len(cases) == 18, "Natural case count changed")
    require(all(c.evidence["scope"] == "real_natural_native_oracle" for c in cases.values()),
            "Analytic cases cannot establish real-field correctness")
    for k in (3, 6):
        selected = [c for c in cases.values() if c.K == k]
        require(len(selected) == 9 and sum(len(c.centers) for c in selected) == 192
                and sum(len(c.endpoints) for c in selected) == 1053,
                "Frozen natural distribution changed")
    return matrix, cases, params


def save_case(destination, case):
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(destination / "inputs.npz", offsets=case.offsets,
             centers=case.centers, endpoints=case.endpoints)
    np.savez(destination / "oracle.npz", **case.oracle)
    write_json(destination / "evidence.json", case.evidence)


def audit_info(api, handle, case, debug, expected_runs, destination, row, save):
    info = np.zeros(40, np.uint64)
    rc = api.solver.bm_solver_get_info(handle, case.K, pointer(info, PU64), info.size)
    row["info_rc"], row["info"] = int(rc), info.tolist()
    write_json(destination / "info.json", {"rc": int(rc), "values": info.tolist()})
    save()
    api.check(rc, "solver_get_info; evidence saved")
    require(info[0] == 1 and info[1] == int(debug), "Unexpected solver ABI/debug build")
    require(info[2] == len(case.centers) and info[3] == len(case.endpoints)
            and info[4] == case.K and info[5] == case.K, "Solver geometry/K changed")
    require(info[31] == 0 and info[32] == 0, "Timing APIs must remain dormant")
    require(info[33] == expected_runs and info[34] == expected_runs,
            "Unexpected normal/completed solve count")
    require(info[21] == 0 and info[24] == 0 and info[35] == 0,
            "L1_32 must not create or run Graphs")
    if debug:
        guard = np.zeros(8, np.uint64)
        rc = api.solver.bm_solver_debug_check(handle, pointer(guard, PU64), guard.size)
        row["guard_rc"], row["guard"] = int(rc), guard.tolist()
        write_json(destination / "guard.json", {"rc": int(rc), "values": guard.tolist()})
        save()
        api.check(rc, "solver_debug_check; evidence saved")
        require(guard[0] == 1 and guard[1] == info[39],
                "Guard check did not cover every allocation")
        require(np.all(guard[2:] == 0), "Device redzone/final-production-count violation")
        row["guard_status"] = "PASS"
    else:
        row["guard_status"] = "NOT_ENABLED_RELEASE"


def check_resident(args, out, solver, report, save):
    _, cases, params = load_frozen()
    selected = [cases[key] for key in sorted(cases)]
    if args.phase == "representative":
        selected = [cases["K3:g00000_b00000_SdfTrees"]]
    report["case_count"] = len(selected)
    report["fixed_solves"] = len(selected) * len(CHECKS)
    report["checks"] = [{"trace": trace, "purpose": purpose} for trace, purpose in CHECKS]
    report["trace_handle_policy"] = "Fresh handle per trace mode; first/repeat share that handle"
    save()
    api = API(BRIDGE, solver)
    debug = args.phase == "debug"
    for case in selected:
        case_dir = out / f"K{case.K}" / case.name
        save_case(case_dir, case)
        report["active"] = {"case": case.name, "K": case.K, "operation": "field_create"}
        save()
        field_handle, view = api.create_field(case.kind, params[case.kind])
        for trace in (0, 1):
            report["active"] = {"case": case.name, "K": case.K, "trace": trace,
                                "operation": "solver_create"}
            save()
            handle = api.create_solver(case, view)
            first = None
            for repetition, purpose in enumerate(("first", "repeat"), 1):
                dest = case_dir / f"trace{trace}_{purpose}"
                dest.mkdir()
                row = {"case": case.name, "field": case.field, "K": case.K,
                       "N": len(case.centers), "M": len(case.endpoints),
                       "mode": "L1_32", "mode_id": 4, "GROUP": 32,
                       "block_threads": args.block, "trace": trace, "purpose": purpose,
                       "status": "RUNNING", "performance_test": False,
                       "scope": case.evidence["scope"]}
                report["rows"].append(row)
                report["active"] = {"case": case.name, "K": case.K, "trace": trace,
                                    "purpose": purpose, "operation": "solver_run_readback"}
                save()
                print(json.dumps({"action": "correctness_solve", **row}), flush=True)
                actual = api.solve_once(handle, case, 4, trace=trace, prepare_graph=False)
                np.savez(dest / "actual.npz", **actual)
                expected_keys = set(ARRAYS if trace else ARRAYS[:6])
                require(set(actual) == expected_keys, "Incomplete output contract")
                row["bitwise_compared"] = list(actual)
                row["oracle_mismatches"] = compare(case.oracle, actual)
                row["repeat_mismatches"] = compare(first, actual) if first is not None else {}
                write_json(dest / "comparison.json", row)
                save()
                # Detect an output error BEFORE issuing further CUDA/debug calls.
                require(not row["oracle_mismatches"] and not row["repeat_mismatches"],
                        "Bitwise output mismatch; actual and native oracle saved")
                audit_info(api, handle, case, debug, repetition, dest, row, save)
                row["status"] = "PASS"
                write_json(dest / "comparison.json", row)
                save()
                print(json.dumps({"status": "PASS", "case": case.name, "K": case.K,
                                  "block": args.block, "trace": trace, "purpose": purpose}), flush=True)
                if first is None:
                    first = actual
            api.check(api.solver.bm_solver_destroy(handle), "solver_destroy")
        api.check(api.field.bm_field_destroy(field_handle), "field_destroy")
    require(len(report["rows"]) == report["fixed_solves"]
            and all(row["status"] == "PASS" for row in report["rows"]), "Incomplete finite checks")
    report["status"] = "PASS_REPRESENTATIVE_CORRECTNESS" if args.phase == "representative" else "PASS_ALL_18_REAL_NATURAL_CASES"


def check_common(args, out, solver, report, save):
    from common_runner import run_common

    matrix, cases, _ = load_frozen()
    report["fixed_common_runs"] = 2
    report["complete_output_contract"] = ["xyz", "times", "tags", "vertex_map"]
    save()
    for k in (3, 6):
        dest = out / f"K{k}"
        dest.mkdir()
        oracle_dir = dest / "oracle"
        oracle_dir.mkdir()
        # Save all reference arrays before the target starts.  run_common saves
        # actual per-field and cumulative per-batch arrays before comparison.
        capture_dirs = {r["capture_dir"] for r in matrix["cases"] if r["K"] == k}
        require(len(capture_dirs) == 1, "Unexpected capture directory set")
        capture = Path(next(iter(capture_dirs)))
        for batch in range(5):
            for suffix in ("input.jsonl", "native_outputs.npz"):
                source = capture / f"g00000_b{batch:05d}_{suffix}"
                shutil.copy2(source, oracle_dir / source.name)
        for key in sorted(cases):
            case = cases[key]
            if case.K == k:
                save_case(oracle_dir / "fields" / case.name, case)
        checkpoint = dest / "checkpoint"
        shutil.copytree(ROOT / "inputs/checkpoint_seed", checkpoint)
        copied_files = []
        for source in sorted((ROOT / "inputs/checkpoint_seed").rglob("*")):
            if source.is_file():
                target = checkpoint / source.relative_to(ROOT / "inputs/checkpoint_seed")
                expected_hash = digest(source)
                require(digest(target) == expected_hash, "Private checkpoint copy changed")
                copied_files.append({"path": str(target.relative_to(checkpoint)), "sha256": expected_hash})
        write_json(dest / "checkpoint_manifest.json", copied_files)
        report["active"] = {"K": k, "mode": "L1_32", "operation": "full_common_entry"}
        save()
        actual, row = run_common("L1_32", k, checkpoint, validate=True, output_dir=dest,
                                 timing=False, solver_path=solver)
        np.savez(dest / "final_actual.npz", **actual)
        report["rows"].append(row)
        write_json(dest / "result.json", row)
        save()
        require(row["status"] == "PASS_FULL_COMMON_OUTPUT" and not row["stages"],
                "Common output validation failed or timing was enabled")
        require(set(actual) == set(report["complete_output_contract"]),
                "Incomplete cumulative native output contract")
        require(len(row["solver_infos"]) == 9, "Expected nine natural field-batches per K")
        for item in row["solver_infos"]:
            info = item["values"]
            require(info[1] == 0 and info[31] == 0 and info[32] == 0,
                    "Common correctness must use a release solver without timing")
            require(info[33] == 1 and info[34] == 1 and info[35] == 0,
                    "Unexpected common solve/Graph count")
        print(json.dumps({"status": row["status"], "K": k, "block": args.block,
                          "natural_batches": len(row["batches"])}), flush=True)
    require(len(report["rows"]) == 2, "Incomplete common runs")
    report["status"] = "PASS_COMMON_K3_K6_COMPLETE_NATIVE_OUTPUT"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("release", "debug", "common", "representative"), required=True)
    parser.add_argument("--block", type=int, choices=(256, 128, 64), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out != ROOT and out.is_relative_to(ROOT) and not out.exists(),
            "A new output directory inside this experiment is required")
    out.mkdir(parents=True, exist_ok=False)
    solver = ROOT / "build" / f"libgpu_solver_b{args.block}{'_debug' if args.phase == 'debug' else ''}.so"
    report = {"status": "RUNNING", "phase": args.phase, "block_threads": args.block,
              "GROUP": 32, "mode": "L1_32", "mode_id": 4, "correctness_only": True,
              "performance_test": False, "warmups": 0, "calibration": False,
              "automatic_retries": 0, "timing_events_expected": 0, "rows": [],
              "failure_policy": "Save CPU-side evidence and exit; no CUDA cleanup or retries after failure"}
    def save():
        write_json(out / "STATUS.json", report)
    save()
    try:
        for name, path in (("solver", solver), ("bridge", BRIDGE), ("matrix", MATRIX),
                           ("parameters", PARAMETERS), ("checker", Path(__file__).resolve())):
            require(path.is_file(), "Missing required file: " + str(path))
            report[name] = {"path": str(path), "sha256": digest(path)}
        save()
        if args.phase == "common":
            check_common(args, out, solver, report, save)
        else:
            check_resident(args, out, solver, report, save)
        report.pop("active", None)
        save()
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        save()
        write_json(out / "STOP.json", {"status": "STOPPED_ON_FIRST_ERROR", "error": repr(error),
                                      "GPU_retry": False, "additional_CUDA_calls": False})
        raise
    print(json.dumps({"status": report["status"], "result": str(out / "STATUS.json")}), flush=True)


if __name__ == "__main__":
    main()
