"""Finite D4 correctness gates; saved observations never enter candidate solves.

All candidate inputs are immutable FieldView parameters and ordered FP64 CSR
geometry. Candidate traces are consumed only after successful full solves.
This module deliberately does not bind either implementation's timing API.
"""
from __future__ import annotations

import argparse
import ctypes as C
import dataclasses
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from check_gpu_solver import (
    API, ARRAYS, Case, FieldView, PI, PF, PD, PU64, compare, digest,
    load_parameters, natural_cases, pointer, require, same,
)


REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE_MODES = ((0, "REF_R0"), (1, "REF_R1"), (4, "REF_L132_B128"))
CANDIDATES = ((1, "GPU_D1_SPLIT"), (2, "GPU_D2_ROUND"), (3, "GPU_D3_SOLVE"))
READ_TYPES = (PF, PI, PI, PD, PD, PF, PF, PF, PI, PD, PD)
FIXED_CHECKS = (("first_trace1", 1, 1), ("repeat_trace1", 1, 1),
                ("trace0_counters0", 0, 0))
COUNT_NAMES = ("points", "requests", "unique", "prior_hits", "new_misses", "actual_construct",
               "enabled_construct", "actual_center_landtiles", "classify", "point_landtiles",
               "point_perlin", "outputs", "logical_enabled", "independent_original_enabled",
               "independent_original_classify", "table_size", "mapping_writes", "independent_original_queries",
               "exact_key_matches", "exact_descriptor_matches")
VALIDATION_NAMES = ("total", "key_mismatch", "descriptor_mismatch", "sdf_mismatch", "aux_mismatch",
                    "independent_copy_vs_bm_eval", "invalid_mapping", "not_READY", "redzone",
                    "production_counts", "capacity", "call_sequence")
DEFAULT_PARAMETER_HASH = "29ae668dab58b5270ae0f589f98dd3e4865ad89111c9844e27fb62154f58c4f4"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def output_arrays(case):
    n, m, k = len(case.centers), len(case.endpoints), case.K
    q = n + (k + 1) * m
    return {
        "position": np.full((n, 3), np.nan, np.float32),
        "witness": np.full(n, -999, np.int32),
        "valid": np.full(n, -999, np.int32),
        "left": np.full(n, np.nan, np.float64),
        "right": np.full(n, np.nan, np.float64),
        "aux": np.full((q, 3), np.nan, np.float32),
        "trace_sdf": np.full(q, np.nan, np.float32),
        "trace_xyz": np.full((q, 3), np.nan, np.float32),
        "trace_sign": np.full(q, -999, np.int32),
        "trace_left": np.full((k + 1, n), np.nan, np.float64),
        "trace_right": np.full((k + 1, n), np.nan, np.float64),
    }


class D4API(API):
    def __init__(self, bridge, reference_solver, candidate_solver=None):
        super().__init__(bridge, reference_solver)
        self._bind(self.field, "bm_field_eval_host",
                   [C.c_void_p, C.c_size_t, PF, PF, PF, C.c_void_p])
        self.d4 = None
        if candidate_solver is None:
            return
        self.d4 = C.CDLL(str(candidate_solver))
        self._bind(self.d4, "d4_solver_create",
                   [FieldView, C.c_int, C.c_int, PI, PD, PD, C.c_int, C.POINTER(C.c_void_p)])
        self._bind(self.d4, "d4_solver_run", [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int])
        self._bind(self.d4, "d4_solver_readback", [C.c_void_p, *READ_TYPES])
        self._bind(self.d4, "d4_solver_destroy", [C.c_void_p])
        self._bind(self.d4, "d4_solver_get_info", [C.c_void_p, C.c_int, PU64, C.c_size_t])
        self._bind(self.d4, "d4_solver_debug_check", [C.c_void_p, PU64, C.c_size_t])
        self._bind(self.d4, "d4_solver_read_diagnostics",
                   [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
                    C.POINTER(C.c_uint32), C.POINTER(C.c_uint32), PF, PF, PF, C.c_size_t, C.c_size_t])
        for name in ("d4_solver_read_counts", "d4_solver_read_memory", "d4_solver_read_validation"):
            self._bind(self.d4, name, [C.c_void_p, PU64, C.c_size_t])

    def create(self, case, view, candidate=False, max_k=6):
        handle = C.c_void_p()
        lib, prefix = (self.d4, "d4") if candidate else (self.solver, "bm")
        self.check(getattr(lib, prefix + "_solver_create")(
            view, len(case.centers), len(case.endpoints), pointer(case.offsets, PI),
            pointer(case.centers, PD), pointer(case.endpoints, PD), max_k, C.byref(handle)),
            prefix + "_solver_create")
        return handle

    def candidate_once(self, handle, case, variant, trace=1, diagnostic=1):
        result = output_arrays(case)
        self.check(self.d4.d4_solver_run(handle, variant, case.K, trace, diagnostic), "d4_solver_run")
        pointers = [pointer(result[key], typ) if trace or index < 6 else typ()
                    for index, (key, typ) in enumerate(zip(ARRAYS, READ_TYPES))]
        self.check(self.d4.d4_solver_readback(handle, *pointers), "d4_solver_readback")
        return result if trace else {key: result[key] for key in ARRAYS[:6]}

    def field_once(self, handle, xyz):
        require(xyz.dtype == np.float32 and xyz.shape[1:] == (3,), "field fixture must be FP32 xyz")
        xyz = np.ascontiguousarray(xyz)
        sdf = np.full(len(xyz), np.nan, np.float32)
        aux = np.full((len(xyz), 3), np.nan, np.float32)
        self.check(self.field.bm_field_eval_host(handle, len(xyz), pointer(xyz, PF),
                                                pointer(sdf, PF), pointer(aux, PF), None),
                   "independent_original_bm_eval_host")
        return {"trace_sdf": sdf, "aux": aux}

    def info(self, handle, k, candidate=False):
        lib, prefix = (self.d4, "d4") if candidate else (self.solver, "bm")
        info = np.zeros(40, np.uint64)
        self.check(getattr(lib, prefix + "_solver_get_info")(handle, k, pointer(info, PU64), info.size),
                   prefix + "_solver_get_info")
        require(info[0] == 1, "unsupported solver info schema")
        require(info[31] == 0 and info[32] == 0, "correctness checker unexpectedly activated timing")
        return info

    def guard(self, handle, info, candidate=False):
        if not info[1]:
            return {"status": "NOT_ENABLED"}
        lib, prefix = (self.d4, "d4") if candidate else (self.solver, "bm")
        guard = np.zeros(8, np.uint64)
        rc = getattr(lib, prefix + "_solver_debug_check")(handle, pointer(guard, PU64), guard.size)
        # The caller writes these raw values before enforcing the gate.
        return {"rc": int(rc), "values": guard.tolist(), "status": "PENDING"}

    def diagnostics(self, handle, max_k=6):
        result = {}
        for suffix, shape in (("counts", (max_k + 2, 20)), ("memory", (16,)), ("validation", (12,))):
            data = np.zeros(shape, np.uint64)
            rc = getattr(self.d4, "d4_solver_read_" + suffix)(handle, pointer(data, PU64), data.size)
            result[suffix] = {"rc": int(rc), "values": data.tolist()}
        return result

    def raw_diagnostics(self, handle, case, memory):
        request_capacity, point_capacity, calls = map(int, (memory[1], memory[2], memory[4]))
        requests, points = request_capacity * calls, point_capacity * calls
        require((requests * 22 + points * 7) * 4 <= 256 * 1024 * 1024,
                "diagnostic raw export exceeds fixed 256 MiB cap; no partial claim")
        rk, ck = np.empty((requests, 5), np.uint32), np.empty((requests, 5), np.uint32)
        rd, cd = np.empty((requests, 6), np.uint32), np.empty((requests, 6), np.uint32)
        xyz, sdf, aux = np.empty((points, 3), np.float32), np.empty(points, np.float32), np.empty((points, 3), np.float32)
        pu32 = C.POINTER(C.c_uint32)
        self.check(self.d4.d4_solver_read_diagnostics(
            handle, pointer(rk, pu32), pointer(ck, pu32), pointer(rd, pu32), pointer(cd, pu32),
            pointer(xyz, PF), pointer(sdf, PF), pointer(aux, PF), requests, points), "d4_read_diagnostics")
        lattice_count = int(memory[3])
        query_counts = [len(case.centers)] + [len(case.endpoints)] * (case.K + 1)
        req_ix = np.concatenate([np.arange(q * lattice_count) + call * request_capacity for call, q in enumerate(query_counts)])
        point_ix = np.concatenate([np.arange(q) + call * point_capacity for call, q in enumerate(query_counts)])
        return {"reference_keys_u32": rk[req_ix], "candidate_keys_u32": ck[req_ix],
                "reference_descriptors_u32": rd[req_ix], "candidate_descriptors_u32": cd[req_ix],
                "original_xyz": xyz[point_ix], "original_sdf": sdf[point_ix], "original_aux": aux[point_ix]}

    def destroy(self, handle, candidate=False):
        lib, prefix = (self.d4, "d4") if candidate else (self.solver, "bm")
        self.check(getattr(lib, prefix + "_solver_destroy")(handle), prefix + "_solver_destroy")


def inputs_signature(case):
    import hashlib
    h = hashlib.sha256()
    for name in ("offsets", "centers", "endpoints"):
        a = getattr(case, name)
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def parameter_signature(params):
    import hashlib
    h = hashlib.sha256(str(params[0]).encode())
    for arr in params[1:]:
        h.update(arr.dtype.str.encode()); h.update(repr(arr.shape).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def make_diagnostics(base, params):
    """Bounded CSR/field fixtures evaluated by real upstream SdfTrees only."""
    cases = []
    def add(name, offsets, centers, endpoints, *, k=3, field_params=None, tags=()):
        cases.append(Case("diagnostic_" + name, "SdfTrees", 4, k,
                          np.ascontiguousarray(offsets, dtype=np.int32),
                          np.ascontiguousarray(centers, dtype=np.float64).reshape(-1, 3),
                          np.ascontiguousarray(endpoints, dtype=np.float64).reshape(-1, 3), {},
                          {"scope": "diagnostic_real_SdfTrees_only", "tags": list(tags),
                           "not_new_native_scene": True, "source_case": base.name,
                           "reference": "frozen_REF_R0_fresh_closed_loop_plus_independent_bm_eval",
                           "candidate_inputs": "parameters_and_source_CSR_or_explicit_diagnostic_geometry_only"},
                          field_params))
    add("empty", [0], [], [], tags=("empty_input",))
    add("empty_node", [0, 0], base.centers[:1], [], tags=("empty_node", "m_zero", "U_equals_capacity_equals_lattices", "field_only"))
    counts = np.asarray([0, 1, 7, 8, 9, 31, 32, 33], np.int32)
    offsets = np.r_[np.int32(0), np.cumsum(counts, dtype=np.int32)]
    centers = base.centers[np.arange(len(counts)) % len(base.centers)]
    endpoints = base.endpoints[np.arange(int(offsets[-1])) % len(base.endpoints)]
    add("csr_tails", offsets, centers, endpoints, tags=("m_tail", "empty_and_nonempty_nodes", "original_rank"))
    reordered = base.endpoints.copy()
    for first, last in zip(base.offsets[:-1], base.offsets[1:]):
        reordered[first:last] = reordered[first:last][::-1]
    add("rank_reverse", base.offsets, base.centers, reordered,
        tags=("different_original_rank", "query_request_order_restoration"))
    order = np.arange(len(base.centers))[::-1]
    reversed_counts = np.diff(base.offsets)[order]
    node_endpoints = np.concatenate([base.endpoints[base.offsets[i]:base.offsets[i+1]] for i in order])
    add("node_reverse", np.r_[np.int32(0), np.cumsum(reversed_counts, dtype=np.int32)],
        base.centers[order], node_endpoints, tags=("query_order_restoration",))
    # No recorded future query is used: these are source geometry or fixed coordinates.
    repeated = np.repeat(base.centers[:1], 129, axis=0)
    add("all_duplicate_queries", np.zeros(130, np.int32), repeated, [],
        tags=("same_key_repeat", "different_lattices", "capacity_tail_129", "field_only"))
    grid = base.centers[0] + np.column_stack((np.arange(129) * 4096.0,
                                             np.arange(129) * -3072.0,
                                             np.zeros(129)))
    add("wide_separated_queries", np.zeros(130, np.int32), grid, [],
        tags=("unique_key_capacity_bound", "field_only", "distinctness_must_be_measured"))
    for label, mask_shift in (("mask_disabled", 1e6), ("mask_enabled", -1e6)):
        changed = tuple(v.copy() if isinstance(v, np.ndarray) else v for v in params)
        changed[2][11] = np.float32(mask_shift)
        add(label, np.zeros(len(base.centers) + 1, np.int32), base.centers, [], field_params=changed,
            tags=("mask_" + label.removeprefix("mask_"), "same_kind_different_parameter_instance", "field_only"))
    changed = tuple(v.copy() if isinstance(v, np.ndarray) else v for v in params)
    changed[2][1] *= np.float32(1.125)
    changed[2][5] += np.float32(0.375)
    add("changed_tree_parameters", np.zeros(len(base.centers) + 1, np.int32), base.centers, [],
        field_params=changed, tags=("same_kind_different_parameter_instance", "field_only"))
    add("K3_to_K6", base.offsets, base.centers, base.endpoints, k=6,
        tags=("K3_to_K6_same_handle", "center_endpoint_cross_cell", "adjacent_round_cross_cell"))
    return cases


def load_cases(args, params):
    cases = []
    for k, capture, oracle_root in ((3, args.capture_k3, args.oracle_k3),
                                    (6, args.capture_k6, args.oracle_k6)):
        loaded = natural_cases(capture, [("SdfTrees", 4, 1, 1)], k)
        require(len(loaded) == 5, f"K{k} requires five natural SdfTrees batches, found {len(loaded)}")
        for case in loaded:
            original_name = case.name
            case.evidence["capture_case_name"] = original_name
            case.evidence["capture_directory"] = str(capture.resolve())
            if oracle_root is not None:
                path = oracle_root / original_name / "oracle.npz"
                require(path.exists(), f"requested independent sealed oracle missing: {path}")
                with np.load(path, allow_pickle=False) as z:
                    old = {key: z[key] for key in z.files}
                require(not compare(old, case.oracle), f"capture vs independent sealed oracle mismatch: {path}")
                case.evidence["independent_sealed_oracle"] = {"path": str(path), "sha256": digest(path), "status": "BITWISE_IDENTICAL"}
            case.name = f"K{k}_" + original_name
            cases.append(case)
    base = cases[0]
    diagnostic = make_diagnostics(base, params)
    if args.scope == "natural":
        selected = cases
    elif args.scope == "diagnostic":
        selected = diagnostic
    else:
        selected = cases + diagnostic
    if args.case_name:
        selected = [case for case in selected if case.name in args.case_name]
        require(len(selected) == len(args.case_name), "one or more --case-name entries were not found")
    if args.case_limit:
        selected = selected[:args.case_limit]
    require(selected, "no selected cases")
    return selected


def query_identity(case, query_index):
    n, m = len(case.centers), len(case.endpoints)
    if query_index < n:
        return {"stage": "center", "round": -1, "node": query_index, "original_rank": None}
    if not m:
        return {"stage": "unknown", "query_index": query_index}
    round_index, endpoint_index = divmod(query_index - n, m)
    node = int(np.searchsorted(case.offsets, endpoint_index, side="right") - 1)
    return {"stage": "final_right" if round_index == case.K else "midpoint",
            "round": round_index, "node": node,
            "original_rank": endpoint_index - int(case.offsets[node]), "endpoint_index": endpoint_index}


def closure_checks(case, actual):
    """Independently reconstruct control from this candidate's freshly read SDF."""
    if "trace_sdf" not in actual:
        return {}
    n, m, k = len(case.centers), len(case.endpoints), case.K
    owners = np.repeat(np.arange(n), np.diff(case.offsets))
    sdf, xyz = actual["trace_sdf"], actual["trace_xyz"]
    expected = {"trace_sign": (sdf >= 0).astype(np.int32)}
    left, right = np.zeros(n, np.float64), np.ones(n, np.float64)
    expected["trace_left"] = np.empty((k + 1, n), np.float64)
    expected["trace_right"] = np.empty((k + 1, n), np.float64)
    expected["trace_left"][0], expected["trace_right"][0] = left, right
    xyz_parts = [case.centers.astype(np.float32)]
    any_parts = []
    for r in range(k + 1):
        t = right if r == k else (left + right) / 2.0
        xyz_parts.append((case.centers[owners] * (1.0 - t[owners, None])
                          + case.endpoints * t[owners, None]).astype(np.float32))
        vals = sdf[n + r * m:n + (r + 1) * m]
        different = (vals >= 0) != (sdf[:n][owners] >= 0)
        any_diff = np.asarray([np.any(different[a:b]) for a, b in zip(case.offsets[:-1], case.offsets[1:])], bool)
        if r < k:
            any_parts.append(any_diff.astype(np.int32))
            middle = (left + right) / 2.0
            right, left = np.where(any_diff, middle, right), np.where(any_diff, left, middle)
            expected["trace_left"][r + 1], expected["trace_right"][r + 1] = left, right
    expected["trace_xyz"] = np.concatenate(xyz_parts)
    expected["left"], expected["right"] = left, right
    mismatches = compare(expected, {key: actual[key] for key in expected})
    return {"status": "FAIL" if mismatches else "PASS", "mismatches": mismatches,
            "OR_by_round_node": np.asarray(any_parts, dtype=np.int32).tolist()}


def workload_checks(case, params, variant, diagnostics, enabled):
    """Check measured execution counts rather than infer them from U."""
    errors = []
    validation = diagnostics["validation"]["values"]
    if any(validation):
        errors.append({"validation_errors": dict(zip(VALIDATION_NAMES, validation))})
    counts = np.asarray(diagnostics["counts"]["values"], np.uint64)
    if not enabled:
        if np.any(counts):
            errors.append({"diagnostic_counter_off_nonzero": True})
        return {"status": "FAIL" if errors else "PASS", "errors": errors, "counter_mode": "OFF", "calls": []}
    q_calls = ([len(case.centers)] + ([len(case.endpoints)] * (case.K + 1) if len(case.endpoints) else [])
               if len(case.centers) else [])
    lattice_count = int(params[1][1])
    table = 0
    calls = []
    for call, q in enumerate(q_calls):
        c = dict(zip(COUNT_NAMES, map(int, counts[call])))
        p = q * lattice_count
        expected = {"points": q, "requests": p, "classify": p, "point_landtiles": q,
                    "point_perlin": q, "outputs": q, "mapping_writes": p,
                    "independent_original_queries": q, "independent_original_classify": p,
                    "exact_key_matches": p, "exact_descriptor_matches": p}
        for name, value in expected.items():
            if c[name] != value:
                errors.append({"call": call, "count": name, "actual": c[name], "expected": value})
        if c["logical_enabled"] != c["independent_original_enabled"]:
            errors.append({"call": call, "logical_enabled_differs_from_original": True})
        if c["enabled_construct"] != c["actual_center_landtiles"] or c["enabled_construct"] > c["actual_construct"]:
            errors.append({"call": call, "construct_landtiles_conservation": False})
        if c["unique"] > p:
            errors.append({"call": call, "unique_exceeds_requests": True})
        if variant == 1:
            if c["actual_construct"] != p or c["new_misses"] != p or c["prior_hits"] != 0:
                errors.append({"call": call, "D1_construct_per_request": False})
        else:
            if c["actual_construct"] != c["new_misses"]:
                errors.append({"call": call, "construct_miss_conservation": False})
            if c["prior_hits"] + c["new_misses"] != c["unique"]:
                errors.append({"call": call, "hit_miss_unique_conservation": False})
            if variant == 2 and (c["prior_hits"] != 0 or c["actual_construct"] != c["unique"]):
                errors.append({"call": call, "D2_construct_per_unique": False})
        table = table + c["new_misses"] if variant == 3 else c["new_misses"]
        if c["table_size"] != table:
            errors.append({"call": call, "table_size": c["table_size"], "expected_table": table})
        c["stage"] = "center" if call == 0 else ("final_right" if call == case.K + 1 else "midpoint")
        c["round"] = call - 1
        calls.append(c)
    if np.any(counts[len(q_calls):]):
        errors.append({"inactive_call_tail_nonzero": True})
    if case.name == "diagnostic_empty_node" and calls[0]["unique"] != lattice_count:
        errors.append({"capacity_boundary_not_reached": True})
    if case.name == "diagnostic_all_duplicate_queries" and calls[0]["unique"] != lattice_count:
        errors.append({"same_key_duplicate_fixture_not_detected": True})
    return {"status": "FAIL" if errors else "PASS", "errors": errors, "counter_mode": "ON",
            "calls": calls, "totals": {name: sum(row[name] for row in calls) for name in COUNT_NAMES}}


class Runner:
    def __init__(self, args, report):
        self.args, self.report = args, report
        self.out = args.out.resolve()
        self.first_difference_saved = False

    def save(self):
        write_json(self.out / "result.json", self.report)

    def difference(self, case, params, stem, mismatches, expected, actual, origin):
        if not mismatches or self.first_difference_saved:
            return
        self.first_difference_saved = True
        path = self.out / "first_difference"
        path.mkdir(exist_ok=False)
        np.savez(path / "inputs.npz", offsets=case.offsets, centers=case.centers, endpoints=case.endpoints,
                 meta=np.asarray(params[0], np.int32), ip=params[1], fp=params[2], ip2=params[3], fp2=params[4])
        np.savez(path / "expected.npz", **expected)
        np.savez(path / "actual.npz", **actual)
        first_key = next(iter(mismatches))
        first_index = mismatches[first_key].get("first_flat_index")
        identity = None
        if first_index is not None and first_key in ("trace_xyz", "trace_sdf", "aux", "trace_sign"):
            query = first_index // 3 if first_key in ("trace_xyz", "aux") else first_index
            identity = query_identity(case, query)
            if "trace_xyz" in actual:
                identity["query_xyz_bits"] = actual["trace_xyz"][query].view(np.uint32).tolist()
        write_json(path / "replay.json", {"case": case.name, "K": case.K, "stem": stem,
                   "origin": origin, "mismatches": mismatches, "first_query": identity,
                   "parameters_sha256": parameter_signature(params), "fingerprints": self.report["fingerprints"],
                   "command": sys.argv, "trace_truncated": False,
                   "replay": "rerun recorded command with --case-name and a new --out; inputs.npz contains complete geometry and field parameters"})

    def enforce_guard(self, guard, info):
        if guard["status"] == "NOT_ENABLED":
            require(not self.args.expect_debug, "debug library requested but guards disabled")
            return
        self.api.check(guard["rc"], "debug_check; evidence saved")
        values = np.asarray(guard["values"], np.uint64)
        require(values[0] == 1 and values[1] == info[39], "debug allocation coverage mismatch")
        require(np.all(values[2:] == 0), "debug redzone/production-count/descriptor violation; evidence saved")
        guard["status"] = "PASS"

    def run_case(self, case, params, candidate):
        case_dir = self.out / case.name
        case_dir.mkdir(exist_ok=True)
        np.savez(case_dir / "inputs.npz", offsets=case.offsets, centers=case.centers, endpoints=case.endpoints)
        write_json(case_dir / "evidence.json", case.evidence)
        self.api = self.report_api
        fh, view = self.api.create_field(4, params)
        handle = self.api.create(case, view, candidate=candidate, max_k=6)
        arms = CANDIDATES if candidate else REFERENCE_MODES
        if candidate:
            arms = tuple(arm for arm in arms if arm[0] in self.args.variants)
        for mode, name in arms:
            first = None
            first_counts = None
            for purpose, trace, diagnostic in FIXED_CHECKS:
                stem = name + "_" + purpose
                row = {"case": case.name, "mode": name, "purpose": purpose, "K": case.K,
                       "N": len(case.centers), "M": len(case.endpoints),
                       "Q": len(case.centers) + (case.K + 1) * len(case.endpoints),
                       "trace": trace, "diagnostic": diagnostic if candidate else 0,
                       "scope": case.evidence["scope"], "status": "RUNNING",
                       "input_signature": inputs_signature(case), "parameter_instance_signature": parameter_signature(params)}
                self.report["rows"].append(row)
                self.report["active"] = {"case": case.name, "stem": stem}
                self.save()
                print(json.dumps({"action": "correctness_solve", **self.report["active"]}), flush=True)
                actual = (self.api.candidate_once(handle, case, mode, trace, diagnostic) if candidate else
                          self.api.solve_once(handle, case, mode, trace, prepare_graph=purpose != "repeat_trace1"))
                np.savez(case_dir / (stem + ".npz"), **actual)
                info = self.api.info(handle, case.K, candidate)
                row["info"] = info.tolist()
                row["guard"] = self.api.guard(handle, info, candidate)
                if candidate:
                    row["device_diagnostics"] = self.api.diagnostics(handle)
                self.save()
                if candidate and diagnostic:
                    device_values = row["device_diagnostics"]["validation"]["values"]
                    if purpose == "first_trace1" or any(device_values):
                        raw = self.api.raw_diagnostics(handle, case, row["device_diagnostics"]["memory"]["values"])
                        raw_path = case_dir / (stem + "_keys_descriptors.npz")
                        np.savez(raw_path, **raw)
                        row["raw_key_descriptor_file"] = str(raw_path.relative_to(self.out))
                        raw_comparison = {
                            "keys": not same(raw["reference_keys_u32"], raw["candidate_keys_u32"]),
                            "descriptors": not same(raw["reference_descriptors_u32"], raw["candidate_descriptors_u32"])}
                        row["raw_key_descriptor_comparison"] = raw_comparison
                        if any(device_values) or any(raw_comparison.values()):
                            self.difference(case, params, stem,
                                            {"device_validation": {"count": int(sum(device_values)), "first_flat_index": 0}},
                                            {"validation": np.zeros(12, np.uint64)}, actual, "device_key_descriptor_gate")
                            np.savez(self.out / "first_difference" / "raw_keys_descriptors.npz", **raw)
                        self.save()
                self.enforce_guard(row["guard"], info)
                if candidate:
                    for key, diagnostic_record in row["device_diagnostics"].items():
                        self.api.check(diagnostic_record["rc"], "d4_read_" + key + "; evidence saved")
                    row["workload"] = workload_checks(case, params, mode, row["device_diagnostics"], diagnostic)
                    current_counts = row["device_diagnostics"]["counts"]["values"]
                    if diagnostic:
                        row["cold_reset_counts_identical"] = first_counts is None or first_counts == current_counts
                        if first_counts is None:
                            first_counts = current_counts
                    self.save()
                    require(not any(row.get("raw_key_descriptor_comparison", {}).values()),
                            "host raw key/descriptor comparison failed; evidence saved")
                    require(row["workload"]["status"] == "PASS", "device field/descriptor/work-count gate failed; evidence saved")
                    require(row.get("cold_reset_counts_identical", True), "repeat solve did not reproduce cold table counts")
                if not case.oracle:
                    require(not candidate and mode == 0 and first is None,
                            "diagnostic candidate requires fresh frozen REF_R0 reference")
                    case.oracle = {key: arr.copy() for key, arr in actual.items()}
                reference = case.oracle if first is None else first
                mismatches = compare(reference, actual)
                row["mismatches"] = mismatches
                row["bitwise_compared"] = list(actual)
                row["closure"] = closure_checks(case, actual)
                self.difference(case, params, stem, mismatches, reference, actual, "closed_loop_oracle")
                if trace:
                    field = self.api.field_once(fh, actual["trace_xyz"])
                    field_actual = {key: actual[key] for key in field}
                    field_mismatches = compare(field, field_actual)
                    row["independent_bm_eval"] = {"status": "FAIL" if field_mismatches else "PASS",
                                                 "Q": len(actual["trace_xyz"]), "mismatches": field_mismatches,
                                                 "inputs": "candidate_actual_trace_xyz_after_full_solve"}
                    np.savez(case_dir / (stem + "_original_field.npz"), **field)
                    self.difference(case, params, stem, field_mismatches, field, actual, "independent_original_bm_eval")
                else:
                    field_mismatches = {}
                row["status"] = "FAIL" if mismatches or field_mismatches or row["closure"].get("mismatches") else "PASS"
                self.save()
                require(row["status"] == "PASS", f"FAIL_NUMERICAL {case.name}/{stem}; evidence saved")
                if first is None:
                    first = actual
            if "K3_to_K6_same_handle" in case.evidence.get("tags", []):
                # Explicit changes of k within a single immutable geometry/field handle.
                for sequence_index, sequence_k in enumerate((3, 6, 3, 6)):
                    small = dataclasses.replace(case, K=sequence_k)
                    actual = (self.api.candidate_once(handle, small, mode, 1, 1) if candidate else
                              self.api.solve_once(handle, small, mode, 1, prepare_graph=True))
                    fresh = self.api.create(small, view, candidate=False, max_k=6)
                    reference = self.api.solve_once(fresh, small, 0, 1, prepare_graph=False)
                    self.api.destroy(fresh, candidate=False)
                    mismatch = compare(reference, actual)
                    closure = closure_checks(small, actual)
                    field = self.api.field_once(fh, actual["trace_xyz"])
                    field_mismatch = compare(field, {key: actual[key] for key in field})
                    info = self.api.info(handle, sequence_k, candidate)
                    guard = self.api.guard(handle, info, candidate)
                    self.enforce_guard(guard, info)
                    sequence_stem = name + f"_K_sequence_{sequence_index}_K{sequence_k}"
                    np.savez(case_dir / (sequence_stem + ".npz"), **actual)
                    np.savez(case_dir / (sequence_stem + "_fresh_REF_R0.npz"), **reference)
                    self.difference(small, params, sequence_stem, mismatch, reference, actual, "K_sequence_fresh_REF_R0")
                    sequence_row = {"case": case.name, "mode": name, "purpose": "same_handle_K_sequence",
                                    "K": sequence_k, "status": "FAIL" if mismatch or closure["mismatches"] or field_mismatch else "PASS",
                                    "mismatches": mismatch, "closure": closure, "field_mismatches": field_mismatch,
                                    "info": info.tolist(), "guard": guard}
                    if candidate:
                        sequence_row["device_diagnostics"] = self.api.diagnostics(handle)
                        sequence_row["workload"] = workload_checks(small, params, mode, sequence_row["device_diagnostics"], True)
                        if sequence_row["workload"]["status"] != "PASS":
                            sequence_row["status"] = "FAIL"
                    self.report["rows"].append(sequence_row)
                    self.save()
                    require(sequence_row["status"] == "PASS", "K-change closure/field failure")
        np.savez(case_dir / "oracle.npz", **case.oracle)
        self.api.destroy(handle, candidate)
        self.api.check(self.api.field.bm_field_destroy(fh), "field_destroy")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("refs", "candidates", "all"), default="all")
    p.add_argument("--scope", choices=("natural", "diagnostic", "all"), default="all")
    p.add_argument("--capture-k3", type=Path, required=True)
    p.add_argument("--capture-k6", type=Path, required=True)
    p.add_argument("--oracle-k3", type=Path)
    p.add_argument("--oracle-k6", type=Path)
    p.add_argument("--parameters", type=Path, required=True)
    p.add_argument("--parameters-sha256", required=True, help="SHA256 of the caller-supplied frozen parameter NPZ")
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--reference-solver", type=Path, required=True)
    p.add_argument("--candidate-solver", type=Path, help="built D4 library; required unless --phase refs")
    p.add_argument("--reference-results", type=Path)
    p.add_argument("--expect-debug", action="store_true")
    p.add_argument("--variants", type=lambda v: tuple(map(int, v.split(","))), default=(1,), help="comma-separated variants; default 1 is D1")
    p.add_argument("--case-name", action="append")
    p.add_argument("--case-limit", type=int)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out.resolve()
    require(out != REPOSITORY and REPOSITORY not in out.parents, "--out must be outside the source repository")
    require(args.phase == "refs" or args.candidate_solver is not None, "--candidate-solver is required for candidate checks")
    require(set(args.variants) <= {1, 2, 3} and args.variants, "variants must be 1,2,3 subset")
    out.mkdir(parents=True, exist_ok=False)
    fingerprints = {"checker_sha256": digest(__file__), "bridge_sha256": digest(args.bridge),
                    "reference_solver_sha256": digest(args.reference_solver),
                    "parameters_file_sha256": digest(args.parameters)}
    if args.phase != "refs":
        fingerprints["candidate_solver_sha256"] = digest(args.candidate_solver)
    report = {"schema": "binocmesher.d4.correctness.v1", "status": "RUNNING", "phase": args.phase,
              "scope": args.scope, "pid": os.getpid(), "command": sys.argv, "fingerprints": fingerprints,
              "performance_tests": "NOT_RUN", "timing_apis_bound": False,
              "candidate_input_policy": "only immutable field parameters and original FP64 CSR geometry; no saved SDF/trace/future query is passed to candidate",
              "common_entry": {"status": "NOT_EVALUATED_BY_THIS_SOLVER_CHECKER"},
              "restricted_case_selection": bool(args.case_name or args.case_limit),
              "rows": [], "error_policy": "exit process after first failure; no CUDA retry or cleanup calls on failed context"}
    runner = Runner(args, report)
    runner.save()
    try:
        params = load_parameters(args.parameters, args.parameters_sha256)[4]
        cases = load_cases(args, params)
        report["case_count"] = len(cases)
        report["natural_case_count"] = sum(c.evidence["scope"] == "real_natural_native_oracle" for c in cases)
        report["diagnostic_case_count"] = len(cases) - report["natural_case_count"]
        report["parameter_lattice_count"] = int(params[1][1])
        if args.phase == "candidates":
            require(args.reference_results is not None, "--phase candidates requires --reference-results")
            ref_report = json.loads((args.reference_results / "result.json").read_text(encoding="utf8"))
            require(ref_report["status"] == "PASS_ON_EXECUTED_CASES", "reference correctness phase did not pass")
            require(ref_report["fingerprints"]["bridge_sha256"] == fingerprints["bridge_sha256"] and
                    ref_report["fingerprints"]["reference_solver_sha256"] == fingerprints["reference_solver_sha256"],
                    "reference phase library fingerprint changed")
            report["reference_gate"] = {"path": str(args.reference_results / "result.json"),
                                        "sha256": digest(args.reference_results / "result.json")}
            for case in cases:
                ref_case = args.reference_results / case.name
                with np.load(ref_case / "inputs.npz", allow_pickle=False) as z:
                    require(all(same(z[key], getattr(case, key)) for key in ("offsets", "centers", "endpoints")),
                            "reference phase CSR input differs")
                with np.load(ref_case / "oracle.npz", allow_pickle=False) as z:
                    sealed = {key: z[key] for key in z.files}
                if case.oracle:
                    require(not compare(case.oracle, sealed), "reference phase oracle differs from native capture")
                else:
                    case.oracle = sealed
        runner.report_api = D4API(args.bridge, args.reference_solver,
                                 args.candidate_solver if args.phase != "refs" else None)
        if args.phase in ("refs", "all"):
            for case in cases:
                runner.run_case(case, case.params if case.params is not None else params, False)
            report["frozen_reference_gate"] = "PASS_ON_EXECUTED_CASES"
            runner.save()
        if args.phase in ("candidates", "all"):
            for case in cases:
                runner.run_case(case, case.params if case.params is not None else params, True)
        report.pop("active", None)
        report["status"] = "PASS_ON_EXECUTED_CASES"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        runner.save()
        raise
    finally:
        runner.save()
    print(json.dumps({"status": report["status"], "result": str(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
