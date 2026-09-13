"""Replay a saved D4 first difference from its self-contained geometry/parameters."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from check_d4_gpu import (D4API, REPOSITORY, Case, write_json, digest, compare, require,
                         closure_checks, workload_checks, parameter_signature)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difference", type=Path, required=True)
    p.add_argument("--variant", type=int, choices=(1, 2, 3))
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--reference-solver", type=Path, required=True)
    p.add_argument("--candidate-solver", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    require(out != REPOSITORY and REPOSITORY not in out.parents, "--out must be outside the source repository")
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.difference / "replay.json").read_text(encoding="utf8"))
    with np.load(args.difference / "inputs.npz", allow_pickle=False) as z:
        params = (int(z["meta"]), z["ip"].copy(), z["fp"].copy(), z["ip2"].copy(), z["fp2"].copy())
        case = Case(metadata["case"], "SdfTrees", 4, int(metadata["K"]),
                    z["offsets"].copy(), z["centers"].copy(), z["endpoints"].copy(), {},
                    {"scope": "saved_failure_replay_only"}, params)
    variant = args.variant
    if variant is None:
        for candidate in (1, 2, 3):
            if f"GPU_D{candidate}_" in metadata["stem"]:
                variant = candidate
                break
    require(variant in (1, 2, 3), "saved failure did not identify a D4 variant; supply --variant")
    report = {"schema": "binocmesher.d4.failure_replay.v1", "status": "RUNNING", "variant": variant,
              "K": case.K, "command": sys.argv, "input_sha256": digest(args.difference / "inputs.npz"),
              "parameter_instance_signature": parameter_signature(params), "original_failure": metadata,
              "fingerprints": {"bridge": digest(args.bridge), "reference_solver": digest(args.reference_solver),
                               "candidate_solver": digest(args.candidate_solver)},
              "candidate_inputs": "only saved geometry and parameters; original saved outputs are comparison-only"}
    def save():
        write_json(out / "result.json", report)
    save()
    try:
        api = D4API(args.bridge, args.reference_solver, args.candidate_solver)
        field, view = api.create_field(4, params)
        reference_handle = api.create(case, view, candidate=False, max_k=6)
        fresh_reference = api.solve_once(reference_handle, case, 0, trace=1)
        api.destroy(reference_handle)
        candidate = api.create(case, view, candidate=True, max_k=6)
        actual = api.candidate_once(candidate, case, variant, trace=1, diagnostic=1)
        diagnostics = api.diagnostics(candidate)
        raw = api.raw_diagnostics(candidate, case, diagnostics["memory"]["values"])
        original_field = api.field_once(field, actual["trace_xyz"])
        np.savez(out / "actual.npz", **actual)
        np.savez(out / "fresh_reference.npz", **fresh_reference)
        np.savez(out / "raw_keys_descriptors.npz", **raw)
        np.savez(out / "original_field.npz", **original_field)
        report["fresh_reference_mismatches"] = compare(fresh_reference, actual)
        report["field_mismatches"] = compare(original_field, {key: actual[key] for key in original_field})
        report["closure"] = closure_checks(case, actual)
        report["diagnostics"] = diagnostics
        report["workload"] = workload_checks(case, params, variant, diagnostics, True)
        report["status"] = ("DIFFERENCE_REPRODUCED" if report["fresh_reference_mismatches"] or
                            report["field_mismatches"] or report["closure"]["status"] != "PASS" or
                            report["workload"]["status"] != "PASS" else "NO_DIFFERENCE_ON_THIS_BUILD")
        save()
        api.destroy(candidate, candidate=True)
        api.check(api.field.bm_field_destroy(field), "field_destroy")
    except BaseException as exc:
        report["status"] = "REPLAY_RUNTIME_FAILURE"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        save()
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "result": str(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
