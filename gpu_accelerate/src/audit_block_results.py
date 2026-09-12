"""Independent stdlib-only audit of saved block-size benchmark evidence.

This script never imports application/CUDA modules, executes a target, or changes
the experiment's measured files.  Ratios are recomputed from raw paired arms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from workspace_paths import ROOT
MODES = ("b256", "b128", "b64")
COMPARISONS = ("b256_vs_b128", "b256_vs_b64")


class Audit:
    def __init__(self):
        self.assertions = 0
        self.counts = Counter()

    def need(self, condition, message):
        self.assertions += 1
        if not condition:
            raise AssertionError(message)

    def equal(self, actual, expected, label):
        if isinstance(expected, float):
            self.need(isinstance(actual, (int, float)) and math.isfinite(actual)
                      and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)
        elif isinstance(expected, dict):
            self.need(isinstance(actual, dict), label + ": expected mapping")
            for key, value in expected.items():
                self.need(key in actual, label + ": missing " + key)
                self.equal(actual[key], value, label + "." + key)
        elif isinstance(expected, list):
            self.need(isinstance(actual, list) and len(actual) == len(expected), label + ": length")
            for index, value in enumerate(expected):
                self.equal(actual[index], value, label + f"[{index}]")
        else:
            self.need(actual == expected, label)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def p90(values):
    values = sorted(values)
    offset = (len(values) - 1) * 0.9
    lower, upper = math.floor(offset), math.ceil(offset)
    return values[lower] + (values[upper] - values[lower]) * (offset - lower)


def key(row):
    return row["table"], row["case"], row["K"], row["comparison"]


def summaries(audit, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    output = {}
    for group, arms in groups.items():
        audit.need(len(arms) == 40, "Expected 40 arms per process comparison")
        pairs = defaultdict(dict)
        for row in arms:
            audit.need(row["pair"] in range(20) and row["arm"] in ("A", "B"), "Invalid pair/arm")
            audit.need(row["arm"] not in pairs[row["pair"]], "Duplicate raw pair arm")
            pairs[row["pair"]][row["arm"]] = row
        audit.need(set(pairs) == set(range(20)), "Incomplete pair indices")
        order_counts = Counter()
        av, bv, ratios = [], [], []
        for index in range(20):
            pair = pairs[index]
            audit.need(set(pair) == {"A", "B"}, "Incomplete pair")
            audit.need(pair["A"]["order"] == pair["B"]["order"] in ("AB", "BA"), "Pair order differs")
            order_counts[pair["A"]["order"]] += 1
            actual_order = "".join(row["arm"] for row in arms if row["pair"] == index)
            audit.need(actual_order == pair["A"]["order"], "Recorded execution order is not raw arm order")
            av.append(pair["A"]["per_solve_ms"])
            bv.append(pair["B"]["per_solve_ms"])
            ratios.append(av[-1] / bv[-1])
        audit.need(order_counts == {"AB": 10, "BA": 10}, "Unbalanced AB/BA")
        output[group] = {"pairs": 20, "median_A_ms": statistics.median(av), "p90_A_ms": p90(av),
                         "median_B_ms": statistics.median(bv), "p90_B_ms": p90(bv),
                         "paired_ratios_A_over_B": ratios, "median_paired_ratio": statistics.median(ratios),
                         "p90_paired_ratio": p90(ratios), "mode_A": "b256",
                         "mode_B": group[3].split("_vs_")[1], "primary": group[0] == "resident"}
    return output


def check_measure(audit, row, case, repeats):
    audit.need(row["repeats"] == repeats and row["completed_solves_delta"] == repeats,
               "Frozen repeats/solve delta differ")
    audit.need(row["event_records_delta"] == 2 and row["trace"] == 0, "Invalid event/trace contract")
    audit.need(row["group_threads"] == 32 and row["block_threads"] == int(row["mode"][1:]),
               "GROUP/block metadata mismatch")
    audit.need(row["logical_queries_per_solve"] == case["Q"]
               and row["logical_queries_in_block"] == repeats * case["Q"], "Logical query count differs")
    audit.need(row["internal_repeats_are_independent_inputs"] is False, "Repeat independence overstated")
    for name in ("event_ms", "host_submission_and_sync_ms", "per_solve_ms", "host_ms_per_solve"):
        audit.need(math.isfinite(row[name]) and row[name] > 0, "Invalid positive time: " + name)
    audit.equal(row["per_solve_ms"], row["event_ms"] / repeats, "GPU time normalization")
    audit.equal(row["host_ms_per_solve"], row["host_submission_and_sync_ms"] / repeats, "Host normalization")
    audit.need(row["full_output_check"] == "PASS_OUTSIDE_TIMING", "Missing output check")


def check_session(audit, path, case, mode, measures, purposes):
    checks = lines(path / "checks.jsonl")
    audit.equal(read(path / "output_checks.json"), checks, "Saved output-check log")
    expected_purposes = ["first", "repeat", "warmup_final"] + purposes
    audit.need([r["purpose"] for r in checks] == expected_purposes, "Missing/extra output checks")
    for row in checks:
        audit.equal(row, {"mode": mode, "status": "PASS", "outside_timing": True, "trace": 0}, "Output check")
    prep = read(path / "preparation.json")
    audit.equal(prep, {"mode": mode, "block_threads": int(mode[1:]), "group_threads": 32,
                      "graph_capture_instantiate": "NOT_USED"}, "Preparation metadata")
    before, after, final = prep["info_before_events"], prep["info_after_events"], prep["final_info"]
    for info in (before, after, final):
        audit.need(len(info) == 40 and info[0] == 1 and info[1] == 0 and info[38] == 1, "Ordinary solver ABI")
        audit.need(info[2:8] == [case["N"], case["M"], case["K"], case["K"], case["Q"], case["Q"]],
                   "Session N/M/K/Q mismatch")
        audit.need(info[21] == 0 and info[24] == 0 and info[35] == 0, "Graphs unexpectedly used")
    audit.need(before[31:36] == [0, 0, 0, 0, 0], "Session not fresh before events")
    audit.need(after[31:36] == [1, 0, 0, 0, 0], "Event creation did a solve")
    audit.need(final[31:36] == [1, 2 * len(measures), 10, 10 + sum(r["repeats"] for r in measures), 0],
               "Final solve/event counts differ from raw measurements plus ten warmups")
    audit.counts["resident_sessions"] += 1
    audit.counts["resident_output_checks"] += len(checks)


def run(audit, suite):
    matrix = read(ROOT / "configs/BLOCK_MATRIX.json")
    audit.need(matrix["blocks"] == [256, 128, 64] and matrix["K"] == [3, 6], "Frozen variants/K differ")
    audit.need(matrix["node_group"] == 32 and matrix["mode_id"] == 4, "Wrong node group or mode")
    audit.need(matrix["processes"] == 3 and matrix["pairs"] == 20 and matrix["warmups"] == 10, "Finite protocol differs")
    case_map = {(r["K"], r["id"]): r for r in matrix["cases"]}
    held = {k: r for k, r in case_map.items() if r["role"] == "held_out"}
    cal = {k: r for k, r in case_map.items() if r["role"] == "calibration"}
    audit.need(len(case_map) == 18 and len(held) == 14 and len(cal) == 4, "Real case inventory differs")
    selection = read(suite / "SELECTION.json")
    selection_hash = sha(suite / "SELECTION.json")
    audit.equal(selection, {"worker": 0, "matrix_sha256": sha(ROOT / "configs/BLOCK_MATRIX.json"),
                           "variant_selection": "NONE", "frozen_blocks": [256, 128, 64]}, "Selection provenance")
    audit.need(set(selection["fields"]) == {f'{r["field"]}:K{r["K"]}' for r in cal.values()}, "Calibration field set")
    for (k, name), case in cal.items():
        path = suite / "worker_0/calibration" / f"K{k}_{name}"
        raw = lines(path / "raw.jsonl")
        choice = selection["fields"][f'{case["field"]}:K{k}']
        audit.need(choice["calibration_case"] == f"K{k}:{name}", "Wrong calibration input")
        levels = sorted({row["repeats"] for row in raw})
        audit.need(levels == [2 ** i for i in range(len(levels))] and levels[-1] <= 4096, "Calibration is not bounded doubling")
        audit.need(choice["repeats"] == levels[-1] and choice["target_ms"] == 10 and choice["repeat_cap"] == 4096,
                   "Calibration final repeat differs")
        for repeats in levels:
            level = [row for row in raw if row["repeats"] == repeats]
            audit.need(len(level) == 3 and {row["mode"] for row in level} == set(MODES), "Calibration modes incomplete")
            durations = {row["mode"]: row["event_ms"] for row in level}
            for row in level:
                check_measure(audit, row, case, repeats)
                audit.need(row["case"] == f"K{k}:{name}" and row["phase"] == "repeat_calibration", "Calibration scope")
            if repeats != levels[-1]:
                audit.need(min(durations.values()) < 10, "Calibration continued after fixed target reached")
            else:
                audit.equal(choice["final_calibration_event_ms"], durations, "Final calibration durations")
                audit.need(choice["resolution_target_reached"] == (min(durations.values()) >= 10), "Target flag differs")
                audit.need(choice["resolution_target_reached"] or repeats == 4096, "Calibration stopped prematurely")
        for mode in MODES:
            measures = [row for row in raw if row["mode"] == mode]
            check_session(audit, path / mode, case, mode, measures,
                          ["calibration_r" + str(r["repeats"]) for r in measures])
        audit.counts["calibration_measurement_blocks"] += len(raw)
    expected_groups = {("resident", name, k, comp) for k, name in held for comp in COMPARISONS}
    expected_groups |= {("common_checkpoint_to_output", "whole_checkpoint", k, comp)
                        for k in (3, 6) for comp in COMPARISONS}
    process_results = []
    for worker in range(3):
        wd = suite / f"worker_{worker}"
        status = read(wd / "STATUS.json")
        audit.equal(status, {"status": "COMPLETED", "performance": "MEASURED", "worker": worker,
                             "raw_timed_arms": 1280, "selection_sha256": selection_hash}, "Worker completion")
        raw = lines(wd / "trials.jsonl")
        audit.need(len(raw) == 1280, "Worker raw arm count")
        for row in raw:
            audit.need(row["worker"] == worker and row["comparison"] in COMPARISONS, "Worker/comparison identity")
            candidate = row["comparison"].split("_vs_")[1]
            audit.need(row["mode_A"] == "b256" and row["mode_B"] == candidate
                       and row["mode"] == ("b256" if row["arm"] == "A" else candidate), "Paired variant identity")
            if row["table"] == "resident":
                case = held[(row["K"], row["case"])]
                repeats = selection["fields"][f'{case["field"]}:K{case["K"]}']["repeats"]
                check_measure(audit, row, case, repeats)
                audit.need(row["field"] == case["field"] and row["role"] == "held_out", "Formal case role")
            else:
                audit.need(row["table"] == "common_checkpoint_to_output" and row["case"] == "whole_checkpoint",
                           "Unexpected timing table")
                audit.need(row["repeats"] == 1 and row["full_output_check"] == "PASS_OUTSIDE_TIMING", "Common timing contract")
                audit.need(row["checkpoint_provisioning_timed"] is False
                           and row["checkpoint_read_prepare_solve_readback_cleanup_timed"] is True, "Common timing boundary")
                record = read(wd / row["stage_file"])
                audit.need(record["label"] == f'{row["comparison"]}:{row["pair"]}:{row["arm"]}', "Common raw stage association")
                root = [r for r in record["stages"] if r["stage"] == "checkpoint_to_output" and r["parent"] is None]
                audit.need(len(root) == 1 and root[0]["wall_ns"] > 0, "Common parent interval")
                audit.equal(row["per_solve_ms"], root[0]["wall_ns"] / 1e6, "Common measured duration")
        computed = summaries(audit, raw)
        audit.need(set(computed) == expected_groups, "Missing comparison group")
        saved = {key(row): row for row in status["summary"]}
        audit.need(len(status["summary"]) == 32 and set(saved) == expected_groups, "Worker summary inventory")
        for group in expected_groups:
            audit.equal(saved[group], computed[group], "Independently recomputed worker summary")
        process_results.append(computed)
        for (k, name), case in held.items():
            for mode in MODES:
                measures = [row for row in raw if row["table"] == "resident"
                            and row["K"] == k and row["case"] == name and row["mode"] == mode]
                purposes = [r["comparison"] + "_pair" + str(r["pair"]).zfill(2) + "_" + r["arm"] for r in measures]
                check_session(audit, wd / "resident" / f"K{k}_{name}" / mode, case, mode, measures, purposes)
        common_records = [read(path) for path in sorted((wd / "common").glob("*.json"))]
        audit.need(len(common_records) == 220, "Expected 60 warmup plus 160 formal common records")
        counts = Counter()
        for record in common_records:
            phase, k, mode = record["phase"], record["K"], record["block_variant"]
            audit.need(phase in ("warmup", "formal") and k in (3, 6) and mode in MODES, "Common record identity")
            audit.need(record["mode"] == "L1_32" and record["group_threads"] == 32
                       and record["block_threads"] == int(mode[1:]), "Common kernel identity")
            audit.need(record["complete_output_check"] == "PASS_OUTSIDE_TIMING", "Common output mismatch")
            audit.need(record["performance_test"] == (phase == "formal"), "Warmup/formal timing separation")
            audit.need(set(record["outputs"]) == {"xyz", "times", "tags", "vertex_map"}, "Full native outputs missing")
            stage_counts = Counter(r["stage"] for r in record["stages"])
            if phase == "formal":
                for stage in ("candidate_allocate_owner_map_upload", "gpu_reset_and_solve",
                              "readback_and_output_restore", "solver_cleanup"):
                    audit.need(stage_counts[stage] == 9, "Missing natural field batch in common path")
                audit.need(stage_counts["field_initialization_upload"] == 1
                           and stage_counts["input_and_instance_prepare"] == 1
                           and stage_counts["reset_checkpoint_read_native_graph_load"] == 1,
                           "Common preparation boundary missing")
            else:
                audit.need(not record["stages"], "Warmup unexpectedly recorded benchmark timings")
            counts[(phase, k, mode)] += 1
        for k in (3, 6):
            for mode in MODES:
                audit.need(counts[("warmup", k, mode)] == 10, "Common warmup imbalance")
                audit.need(counts[("formal", k, mode)] == (40 if mode == "b256" else 20), "Common formal count")
        audit.need(not any((wd / "private_checkpoints").iterdir()), "Temporary checkpoints remain")
        audit.counts["formal_timed_arms"] += len(raw)
        audit.counts["common_output_checks"] += len(common_records)
    summary = read(suite / "SUMMARY.json")
    audit.equal(summary, {"status": "COMPLETED", "performance": "MEASURED",
                          "formal_timed_arms": 3840, "selection_sha256": selection_hash}, "Suite completion")
    aggregate_rows = {key(row): row for row in summary["rows"]}
    audit.need(len(summary["rows"]) == 32 and set(aggregate_rows) == expected_groups, "Aggregate inventory")
    for group, row in aggregate_rows.items():
        computed = [process[group] for process in process_results]
        ratios = [r["median_paired_ratio"] for r in computed]
        audit.equal(row, {"median_of_process_paired_medians": statistics.median(ratios),
                          "process_ratio_range": [min(ratios), max(ratios)],
                          "median_of_process_median_A_ms": statistics.median(r["median_A_ms"] for r in computed),
                          "median_of_process_median_B_ms": statistics.median(r["median_B_ms"] for r in computed)},
                    "Independently recomputed aggregate")
        audit.need([r["worker"] for r in row["process_summaries"]] == [0, 1, 2], "Aggregate worker identities")
        for worker in range(3):
            audit.equal(row["process_summaries"][worker], computed[worker], "Aggregate per-process contents")
    with (suite / "PROCESS_TIMINGS.csv").open(encoding="utf8", newline="") as f:
        exported = list(csv.DictReader(f))
    audit.need(len(exported) == 96, "Process CSV inventory")
    seen = set()
    for row in exported:
        group = row["table"], row["case"], int(row["K"]), row["comparison"]
        worker = int(row["worker"])
        audit.need((worker, group) not in seen, "Duplicate CSV process row")
        seen.add((worker, group))
        expected = process_results[worker][group]
        for name in ("median_A_ms", "median_B_ms", "p90_A_ms", "p90_B_ms", "median_paired_ratio", "p90_paired_ratio"):
            audit.equal(float(row[name]), expected[name], "CSV statistic")
    audit.need(audit.counts["formal_timed_arms"] == 3840 and audit.counts["common_output_checks"] == 660,
               "Final count mismatch")
    suite_status = read(suite / "STATUS.json")
    audit.equal(suite_status, {"status": "COMPLETED", "performance": "MEASURED", "completed_workers": [0, 1, 2],
                              "formal_timed_arms": 3840}, "Supervisor status")
    return {"selection_sha256": selection_hash, "matrix_sha256": sha(ROOT / "configs/BLOCK_MATRIX.json"),
            "summary_sha256": sha(suite / "SUMMARY.json"), "process_csv_sha256": sha(suite / "PROCESS_TIMINGS.csv"),
            "raw_trial_sha256": {str(i): sha(suite / f"worker_{i}/trials.jsonl") for i in range(3)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    suite, out = args.suite.resolve(), args.out.resolve()
    if not suite.is_relative_to(ROOT) or not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Existing suite and new output file must be inside this experiment")
    audit = Audit()
    result = {"status": "RUNNING", "GPU_executed": False, "application_imports": False,
              "suite": str(suite), "issues": [], "method": "Independent stdlib raw-pair recomputation and evidence inventory"}
    try:
        result["evidence"] = run(audit, suite)
        result["status"] = "PASS_INDEPENDENT_BLOCK_AUDIT"
    except BaseException as error:
        result.update(status="FAIL", issues=[repr(error)], traceback=traceback.format_exc())
        raise
    finally:
        result.update(assertions=audit.assertions, counts=dict(audit.counts))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps({"status": result["status"], "assertions": audit.assertions, "counts": dict(audit.counts)}))


if __name__ == "__main__":
    main()
