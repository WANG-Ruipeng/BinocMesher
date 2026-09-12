"""Correctness only: fixed reset and trace checks; no performance code.

Natural cases use original native capture as oracle. Analytic FP32 diagnostics
have a separately labelled CPU arithmetic oracle and never establish real-field
coverage. The supervisor, not this module's author, owns any GPU execution.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from workspace_paths import ROOT
PI, PF, PD = C.POINTER(C.c_int), C.POINTER(C.c_float), C.POINTER(C.c_double)
PU64 = C.POINTER(C.c_uint64)
MODES = ("R0", "R1", "L0", "L1_8", "L1_32")
ARRAYS = ("position", "witness", "valid", "left", "right", "aux",
          "trace_sdf", "trace_xyz", "trace_sign", "trace_left", "trace_right")


class CheckFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fp32(bits):
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def pointer(a, dtype):
    return a.ctypes.data_as(dtype) if a.size else dtype()


@dataclass
class Case:
    name: str
    field: str
    kind: int
    K: int
    offsets: np.ndarray
    centers: np.ndarray
    endpoints: np.ndarray
    oracle: dict
    evidence: dict
    params: tuple | None = None


class FieldView(C.Structure):
    _fields_ = [("kind", C.c_int), ("meta", C.c_int), ("ip", PI),
                ("fp", PF), ("ip2", PI), ("fp2", PF)]


class API:
    def __init__(self, bridge, solver):
        self.field = C.CDLL(str(bridge))
        self.solver = C.CDLL(str(solver))
        self._bind(self.field, "bm_field_create",
                   [C.c_int, C.c_int, C.c_size_t, PI, C.c_size_t, PF,
                    C.c_size_t, PI, C.c_size_t, PF, C.POINTER(C.c_void_p)])
        self._bind(self.field, "bm_field_get_view", [C.c_void_p, C.POINTER(FieldView)])
        self._bind(self.field, "bm_field_destroy", [C.c_void_p])
        self._bind(self.field, "bm_field_error_string", [C.c_int], C.c_char_p)
        self._bind(self.solver, "bm_solver_create",
                   [FieldView, C.c_int, C.c_int, PI, PD, PD, C.c_int, C.POINTER(C.c_void_p)])
        self._bind(self.solver, "bm_solver_prepare_graph", [C.c_void_p, C.c_int, C.c_int])
        self._bind(self.solver, "bm_solver_run", [C.c_void_p, C.c_int, C.c_int, C.c_int])
        self._bind(self.solver, "bm_solver_readback",
                   [C.c_void_p, PF, PI, PI, PD, PD, PF, PF, PF, PI, PD, PD])
        self._bind(self.solver, "bm_solver_destroy", [C.c_void_p])
        self._bind(self.solver, "bm_solver_get_info",
                   [C.c_void_p, C.c_int, PU64, C.c_size_t])
        self._bind(self.solver, "bm_solver_debug_check",
                   [C.c_void_p, PU64, C.c_size_t])
        # Deliberately do not bind or call either timing API in this checker.

    @staticmethod
    def _bind(lib, name, args, restype=C.c_int):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, restype

    def check(self, rc, operation):
        if rc:
            raw = self.field.bm_field_error_string(rc)
            raise CheckFailure(f"{operation}: CUDA {rc}: {raw.decode() if raw else 'unknown'}")

    def create_field(self, kind, params):
        meta, ip, fp, ip2, fp2 = params
        handle = C.c_void_p()
        self.check(self.field.bm_field_create(
            kind, meta, ip.size, pointer(ip, PI), fp.size, pointer(fp, PF),
            ip2.size, pointer(ip2, PI), fp2.size, pointer(fp2, PF), C.byref(handle)),
            "field_create")
        view = FieldView()
        self.check(self.field.bm_field_get_view(handle, C.byref(view)), "field_get_view")
        return handle, view

    def create_solver(self, case, view):
        handle = C.c_void_p()
        self.check(self.solver.bm_solver_create(
            view, len(case.centers), len(case.endpoints), pointer(case.offsets, PI),
            pointer(case.centers, PD), pointer(case.endpoints, PD), case.K, C.byref(handle)),
            "solver_create")
        return handle

    def solve_once(self, handle, case, mode, trace=1, prepare_graph=True):
        n, m, k = len(case.centers), len(case.endpoints), case.K
        q = n + (k + 1) * m
        result = {
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
        if mode == 1 and prepare_graph:
            self.check(self.solver.bm_solver_prepare_graph(handle, k, trace), "prepare_graph")
        self.check(self.solver.bm_solver_run(handle, mode, k, trace), "solver_run")
        ptr_types = (PF, PI, PI, PD, PD, PF, PF, PF, PI, PD, PD)
        pointers = [pointer(result[key], typ) if trace or index < 6 else typ()
                    for index, (key, typ) in enumerate(zip(ARRAYS, ptr_types))]
        self.check(self.solver.bm_solver_readback(handle, *pointers), "solver_readback")
        return result if trace else {key: result[key] for key in ARRAYS[:6]}


def validation_ok(records, path):
    for row in records:
        require(row.get("type") != "error", f"native capture recorded an error: {path}")
        if row.get("type") == "validation":
            for key, value in row.items():
                if "mismatch" in key or key in ("missing_witnesses", "nonfinite_nodes"):
                    require(value == 0, f"native capture failed {key}={value}: {path}")


def node_rows(path, kind):
    records = rows(path)
    validation_ok(records, path)
    found = [r for r in records if r.get("type") == kind]
    return {tuple(r["key"]): r for r in found}


def aux_columns(payload, n):
    keys = [key for key in payload.files if key.startswith("aux_")]
    # Driver preserves column index before original aux name.
    def rank(key):
        match = re.match(r"aux_column_(\d+)_", key)
        require(match is not None, f"aux key lacks explicit original column index: {key}")
        return int(match.group(1))
    keys.sort(key=rank)
    require([rank(k) for k in keys] == list(range(len(keys))), "noncontiguous aux columns")
    return np.column_stack([payload[k].reshape(n) for k in keys]).astype(np.float32, copy=False)


def actual_stage(capture_dir, events, group, batch, field, stage, native_xyz, expected_sdf, aux_count):
    selected = [e for e in events if e.get("type") == "field"
                and e.get("group") == group and e.get("batch") == batch
                and e.get("field") == field and e.get("query_stage") == stage
                and e.get("run_name") is None]
    require(selected, f"missing original field observation {group}/{batch}/{field}/{stage}")
    sdf_parts, xyz_parts, aux_parts = [], [], []
    for e in selected:
        path = capture_dir / e["file"]
        require(path.parent.resolve() == capture_dir.resolve(), "field file escaped capture directory")
        with np.load(path, allow_pickle=False) as data:
            before = data["xyz_fp64_before"]
            after = data["xyz_fp64_after"]
            xyz = data["xyz_fp32"]
            raw = data["raw_sdf"].reshape(-1)
            processed = data["processed_sdf"].reshape(-1)
            require(before.dtype == after.dtype == np.float64 and xyz.dtype == np.float32,
                    "unexpected captured geometry precision")
            require(raw.dtype == processed.dtype == np.float32, "unexpected SDF precision")
            require(same(before, after), "real closure has an unimplemented coordinate transform")
            require(same(xyz, np.ascontiguousarray(after, dtype=np.float32)),
                    "actual wrapper input differs from float64-to-float32 contract")
            require(same(raw, processed), "real closure has unimplemented SDF postprocessing")
            aux = aux_columns(data, len(raw))
            require(aux.shape == (len(raw), aux_count), "real native aux count changed")
            sdf_parts.append(raw.copy())
            xyz_parts.append(xyz.copy())
            aux_parts.append(aux.copy())
    xyz, sdf, aux = np.concatenate(xyz_parts), np.concatenate(sdf_parts), np.concatenate(aux_parts)
    require(same(xyz, np.ascontiguousarray(native_xyz, dtype=np.float32)),
            f"actual CUDA xyz does not match native query: {group}/{batch}/{field}/{stage}")
    require(same(sdf, expected_sdf), f"actual rawSDF does not match native bisection SDF: {stage}")
    require(np.isfinite(sdf).all() and np.isfinite(aux).all(), "nonfinite captured real SDF/aux")
    padded = np.zeros((len(sdf), 3), np.float32)
    padded[:, :aux_count] = aux
    return xyz, sdf, padded, [e["file"] for e in selected]


def natural_cases(capture_dir, fields, k):
    capture_dir = Path(capture_dir).resolve()
    events = rows(capture_dir / "events.jsonl")
    paths = sorted(capture_dir.glob("g*_b*_input.jsonl"))
    require(paths, f"no captured natural batches under {capture_dir}")
    cases = []
    for path in paths:
        match = re.fullmatch(r"g(\d+)_b(\d+)_input.jsonl", path.name)
        require(match is not None, "unsupported capture basename")
        group, batch = map(int, match.groups())
        prefix = path.name.removesuffix("input.jsonl")
        records = rows(path)
        validation_ok(records, path)
        header = next(r for r in records if r.get("type") == "header")
        require(header["schema"] == "binocmesher.native_capture.v2", "unsupported capture schema")
        all_nodes = [r for r in records if r.get("type") == "node"]
        finals_path = capture_dir / (prefix + "final.jsonl")
        finals = node_rows(finals_path, "final_node")
        before_rounds, after_rounds = [], []
        for r in range(k):
            before_rounds.append(node_rows(capture_dir / (prefix + f"r{r:03d}_before.jsonl"), "round_before_node"))
            after_rounds.append(node_rows(capture_dir / (prefix + f"r{r:03d}_after.jsonl"), "round_node"))
        require(not (capture_dir / (prefix + f"r{k:03d}_after.jsonl")).exists(),
                f"capture K exceeds requested K={k}")
        prefinals = node_rows(capture_dir / (prefix + "prefinish.jsonl"), "prefinish_node")
        for field, kind, element, aux_count in fields:
            nodes = sorted((r for r in all_nodes if r["element"] == element), key=lambda r: r["center_index"])
            if not nodes:
                continue
            keys = [tuple(r["key"]) for r in nodes]
            counts = np.asarray([len(r["endpoints"]) for r in nodes], np.int32)
            offsets = np.concatenate((np.zeros(1, np.int32), np.cumsum(counts, dtype=np.int32)))
            centers = np.ascontiguousarray([r["center"] for r in nodes], dtype=np.float64)
            endpoints = np.ascontiguousarray([ep["xyz"] for node in nodes for ep in node["endpoints"]], dtype=np.float64).reshape(-1, 3)
            center_indices = np.asarray([r["center_index"] for r in nodes], np.int64)
            endpoint_indices = np.asarray([ep["native_query_index"] for node in nodes for ep in node["endpoints"]], np.int64)
            require(all(r["initial_left"] == 0 and r["initial_right"] == 1 for r in nodes), "unexpected native initial bounds")
            require(np.array_equal(center_indices, np.arange(center_indices[0], center_indices[0] + len(nodes))),
                    "single-field centers are not contiguous native order")
            require(np.array_equal(endpoint_indices, np.arange(endpoint_indices[0], endpoint_indices[0] + len(endpoints))),
                    "single-field endpoints are not contiguous native order")
            n, m = len(nodes), len(endpoints)
            oracle = {"position": fp32([finals[key]["native_position_fp32_bits"] for key in keys]),
                      "witness": np.asarray([finals[key]["witness_rank"] for key in keys], np.int32),
                      "valid": np.ones(n, np.int32),
                      "left": np.asarray([finals[key]["left"] for key in keys], np.float64),
                      "right": np.asarray([finals[key]["right"] for key in keys], np.float64),
                      "trace_left": np.zeros((k + 1, n), np.float64),
                      "trace_right": np.ones((k + 1, n), np.float64)}
            for i, key in enumerate(keys):
                require(prefinals[key]["witness_rank"] == finals[key]["witness_rank"] >= 0,
                        "native pre/post finishing witness mismatch")
                for name in ("expected_vertex_map", "time_mid", "time_halfwidth", "inview"):
                    actual_name = "vertex_map" if name == "expected_vertex_map" else name
                    require(nodes[i][name] == finals[key][actual_name], "native identity metadata mismatch")
            center_sdf = fp32([before_rounds[0][key]["center_sdf_bits"] for key in keys])
            xyz_chunks, sdf_chunks, aux_chunks, field_files = [], [], [], []
            native_queries = [("center", "centers_fp64.npz", center_indices, center_sdf)]
            for r in range(k):
                sdfs = fp32([b for key in keys for b in before_rounds[r][key]["endpoint_sdf_bits"]])
                native_queries.append((f"round_{r}", f"r{r:03d}_positions_fp64.npz", endpoint_indices, sdfs))
                for i, key in enumerate(keys):
                    after = after_rounds[r][key]
                    require(after["bounds_match"], "native bounds mismatch")
                    require(after["endpoint_sdf_bits"] == before_rounds[r][key]["endpoint_sdf_bits"],
                            "native round pre/post SDF mismatch")
                    require(after["center_sdf_bits"] == int(center_sdf.view(np.uint32)[i]), "native center SDF changed across rounds")
                    oracle["trace_left"][r + 1, i] = after["left"]
                    oracle["trace_right"][r + 1, i] = after["right"]
            final_sdf = fp32([b for key in keys for b in prefinals[key]["endpoint_sdf_bits"]])
            native_queries.append(("final_right", "final_right_fp64.npz", endpoint_indices, final_sdf))
            for stage, basename, indices, sdf in native_queries:
                query_path = capture_dir / (prefix + basename)
                with np.load(query_path, allow_pickle=False) as data:
                    native_xyz = np.ascontiguousarray(data["positions"][indices])
                xyz, raw, aux, files = actual_stage(capture_dir, events, group, batch, field,
                                                   stage, native_xyz, sdf, aux_count)
                xyz_chunks.append(xyz); sdf_chunks.append(raw); aux_chunks.append(aux); field_files.extend(files)
            oracle["trace_xyz"] = np.concatenate(xyz_chunks)
            oracle["trace_sdf"] = np.concatenate(sdf_chunks)
            oracle["trace_sign"] = (oracle["trace_sdf"] >= 0).astype(np.int32)
            oracle["aux"] = np.concatenate(aux_chunks)
            # Check original geometry itself against the captured actual query.
            require(same(oracle["trace_xyz"][:n], centers.astype(np.float32)), "captured centers differ from native coordinates")
            evidence = {"scope": "real_natural_native_oracle", "group": group, "batch": batch,
                        "element": element, "field": field, "capture_input": str(path),
                        "capture_input_sha256": digest(path), "capture_final_sha256": digest(finals_path),
                        "field_observation_files": field_files, "counts": counts.tolist(),
                        "head_keys": [list(key) for key in keys],
                        "vertex_map": [finals[key]["vertex_map"] for key in keys],
                        "times": [[finals[key]["time_mid"], finals[key]["time_halfwidth"]] for key in keys],
                        "tags": [finals[key]["inview"] for key in keys],
                        "metadata_contract": "original identity metadata reused by ordered output adapter; no GPU metadata-generation claim"}
            cases.append(Case(f"g{group:05d}_b{batch:05d}_{field}", field, kind, k, offsets, centers, endpoints, oracle, evidence))
    require(cases, "no requested real-field natural nodes captured")
    return cases


def diagnostic_field(kind, params, xyz):
    p = params.astype(np.float32, copy=False)
    x, y, z = xyz.T
    if kind == 0:
        return (((p[0] * x + p[1] * y) + p[2] * z) + p[3]).astype(np.float32)
    dx, dy, dz = x - p[0], y - p[1], z - p[2]
    out = (((dx * dx + dy * dy) + dz * dz) - p[3] * p[3]).astype(np.float32)
    if kind == 2:
        dx, dy, dz = x - p[4], y - p[5], z - p[6]
        other = (((dx * dx + dy * dy) + dz * dz) - p[7] * p[7]).astype(np.float32)
        out = np.fmin(out, other)
    return out


def diagnostic_oracle(kind, params, offsets, centers, endpoints, k):
    n, m = len(centers), len(endpoints)
    owners = np.repeat(np.arange(n), np.diff(offsets))
    left, right = np.zeros(n), np.ones(n)
    center_xyz = centers.astype(np.float32)
    cs = diagnostic_field(kind, params, center_xyz)
    xyz_parts, sdf_parts = [center_xyz], [cs]
    lb, rb = [left.copy()], [right.copy()]
    for r in range(k + 1):
        t = right if r == k else (left + right) / 2
        xyz = (centers[owners] * (1 - t[owners, None]) + endpoints * t[owners, None]).astype(np.float32)
        sdf = diagnostic_field(kind, params, xyz)
        xyz_parts.append(xyz); sdf_parts.append(sdf)
        if r < k:
            any_diff = np.asarray([np.any((sdf[offsets[i]:offsets[i+1]] >= 0) != (cs[i] >= 0)) for i in range(n)])
            middle = (left + right) / 2
            right = np.where(any_diff, middle, right)
            left = np.where(any_diff, left, middle)
            lb.append(left.copy()); rb.append(right.copy())
    witness, valid = np.full(n, -1, np.int32), np.ones(n, np.int32)
    position = np.zeros((n, 3), np.float32)
    for i in range(n):
        matches = np.flatnonzero((sdf[offsets[i]:offsets[i+1]] >= 0) != (cs[i] >= 0))
        if not offsets[i+1] - offsets[i]:
            valid[i] = 0
        elif not len(matches):
            valid[i] = -1
        else:
            witness[i] = matches[0]
            position[i] = centers[i] * (1 - right[i]) + endpoints[offsets[i] + witness[i]] * right[i]
    all_sdf = np.concatenate(sdf_parts)
    return {"position": position, "witness": witness, "valid": valid,
            "left": left, "right": right, "aux": np.zeros((n + (k + 1) * m, 3), np.float32),
            "trace_sdf": all_sdf, "trace_xyz": np.concatenate(xyz_parts),
            "trace_sign": (all_sdf >= 0).astype(np.int32),
            "trace_left": np.asarray(lb, np.float64), "trace_right": np.asarray(rb, np.float64)}


def diagnostic_cases():
    result = []
    empty_i, empty_f = np.empty(0, np.int32), np.empty(0, np.float32)
    for kind, params in [(0, [1, 0, 0, 0]), (1, [0, 0, 0, 1]), (2, [0, 0, 0, 1, -3, 0, 0, 1])]:
        params = np.asarray(params, np.float32)
        counts = np.resize(np.asarray([0, 1, 7, 8, 9, 31, 32, 33, 63], np.int32), 33)
        offsets = np.concatenate((np.zeros(1, np.int32), np.cumsum(counts, dtype=np.int32)))
        centers = np.zeros((len(counts), 3), np.float64)
        centers[:, 0] = -1 if kind == 0 else 2
        endpoints = np.zeros((int(offsets[-1]), 3), np.float64)
        if kind == 0:
            endpoints[1::2, 0] = 1
        elif kind == 2:
            # Exercise either member of fmin without changing the diagnostic
            # endpoint-count distribution.
            for i in range(1, len(counts), 2):
                centers[i, 0] = -5
                endpoints[offsets[i]:offsets[i + 1], 0] = -3
        for k in range(1, 7):
            oracle = diagnostic_oracle(kind, params, offsets, centers, endpoints, k)
            base_name = f"diagnostic_kind{kind}_K{k}"
            result.append(Case(base_name, f"diagnostic_{kind}", kind, k,
                               offsets, centers, endpoints, oracle,
                               {"scope": "analytic_FP32_diagnostic_only", "not_real_field_coverage": True,
                                "counts": counts.tolist(), "tail_nodes": len(counts),
                                "field_float_params": params.tolist()},
                               (0, empty_i, params, empty_i, empty_f)))
            for suffix, scale, translation in (
                    ("scale_half_shift", 0.5, [4, -2, 1]),
                    ("scale_double_shift", 2.0, [-8, 4, -2])):
                shift = np.asarray(translation, np.float64)
                transformed_centers = np.ascontiguousarray(centers * scale + shift, dtype=np.float64)
                transformed_endpoints = np.ascontiguousarray(endpoints * scale + shift, dtype=np.float64)
                transformed_params = params.copy()
                if kind == 0:
                    # p' = scale*p + shift: keep n and set d'=scale*d-n.dot(shift).
                    transformed_params[3] = np.float32(
                        scale * float(params[3]) - np.dot(params[:3].astype(np.float64), shift))
                else:
                    # Transform both center and radius of each quadratic sphere.
                    for start in ((0, 4) if kind == 2 else (0,)):
                        transformed_params[start:start + 3] = (
                            params[start:start + 3].astype(np.float64) * scale + shift).astype(np.float32)
                        transformed_params[start + 3] = np.float32(scale * float(params[start + 3]))
                transformed_oracle = diagnostic_oracle(
                    kind, transformed_params, offsets, transformed_centers, transformed_endpoints, k)
                result.append(Case(base_name + "_" + suffix, f"diagnostic_{kind}", kind, k,
                                   offsets, transformed_centers, transformed_endpoints, transformed_oracle,
                                   {"scope": "analytic_FP32_diagnostic_only", "not_real_field_coverage": True,
                                    "counts": counts.tolist(), "tail_nodes": len(counts),
                                    "base_case": base_name,
                                    "geometry_transform": {"scale": scale, "shift": translation,
                                                           "coordinates": "p_prime = scale*p + shift",
                                                           "field_parameters_transformed": True},
                                    "field_float_params": transformed_params.tolist()},
                                   (0, empty_i, transformed_params, empty_i, empty_f)))
    for k in range(1, 7):
        offsets, centers, endpoints = np.zeros(1, np.int32), np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)
        params = np.asarray([1, 0, 0, 0], np.float32)
        result.append(Case(f"diagnostic_empty_K{k}", "diagnostic_0", 0, k, offsets, centers, endpoints,
                           diagnostic_oracle(0, params, offsets, centers, endpoints, k),
                           {"scope": "empty_input_diagnostic_only", "not_real_field_coverage": True},
                           (0, empty_i, params, empty_i, empty_f)))
    return result


def load_parameters(path, expected_hash):
    require(expected_hash and digest(path) == expected_hash, "real parameter SHA256 mismatch or missing expected hash")
    with np.load(path, allow_pickle=False) as data:
        def get(name, dtype):
            a = data[name]
            require(a.dtype == np.dtype(dtype), f"parameter precision changed: {name}")
            return np.ascontiguousarray(a)
        li, lf = get("land_int_params", np.int32), get("land_float_params", np.float32)
        ti, tf = get("trees_int_params", np.int32), get("trees_float_params", np.float32)
        meta = np.asarray(data["land_meta_params"]).ravel()
        require(np.all(meta == 0), "unsupported caved LandTiles closure")
    ei, ef = np.empty(0, np.int32), np.empty(0, np.float32)
    return {3: (0, li, lf, ei, ef), 4: (0, ti, tf, li, lf)}


def compare(expected, actual):
    mismatches = {}
    for key in actual:
        a, b = actual[key], expected[key]
        if not same(a, b):
            flat_a = a.view(np.uint32 if a.dtype == np.float32 else np.uint64).ravel() if a.dtype.kind == "f" else a.ravel()
            flat_b = b.view(np.uint32 if b.dtype == np.float32 else np.uint64).ravel() if b.dtype.kind == "f" else b.ravel()
            if a.shape == b.shape and a.dtype == b.dtype:
                indices = np.flatnonzero(flat_a != flat_b)
                first = int(indices[0]) if len(indices) else None
                mismatches[key] = {"count": int(len(indices)), "first_flat_index": first,
                                   "actual_bits_or_value": int(flat_a[first]) if first is not None else None,
                                   "expected_bits_or_value": int(flat_b[first]) if first is not None else None}
            else:
                mismatches[key] = {"actual_shape": list(a.shape), "expected_shape": list(b.shape),
                                   "actual_dtype": str(a.dtype), "expected_dtype": str(b.dtype)}
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("natural", "diagnostic", "all"), default="natural")
    parser.add_argument("--capture-dir", type=Path, default=ROOT / "runs/g0_observed_01/capture")
    parser.add_argument("--parameters", type=Path, default=ROOT / "inputs/accepted_scene/field_parameters.npz")
    parser.add_argument("--parameters-sha256")
    parser.add_argument("--field", choices=("all", "LandTiles", "SdfTrees"), default="all")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--bridge", type=Path, default=ROOT / "build/libfield_bridge.so")
    parser.add_argument("--solver", type=Path, default=ROOT / "build/libgpu_solver.so")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.K >= 1, "natural captures require K>=1")
    out = args.out.resolve()
    require(out != ROOT.resolve() and ROOT.resolve() in out.parents,
            "all checker outputs must remain inside the new experiment")
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": args.scope, "correctness_only": True,
              "performance_tests": "NOT_RUN", "warmup": False, "calibration": False,
              "timing_audit": "every solve records info[31]=events_prepared and info[32]=event_records; both must be 0",
              "debug_audit": "debug builds save and verify 8 guard/final-production-count values after every solve",
              "solves_per_case_mode": 3,
              "fixed_solve_purposes": ["native_or_diagnostic_oracle_trace1",
                                      "same_handle_reset_trace1_repeat",
                                      "trace0_noninterference"],
              "graph_repeat_policy": "second trace1 solve reuses same prepared graph; trace0 explicitly rebuilds",
              "rows": [], "cuda_error_cleanup": "process exit on first error; no further GPU calls or retry"}
    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        cases, real_params = [], {}
        if args.scope in ("natural", "all"):
            selected = [("LandTiles", 3, 0, 3), ("SdfTrees", 4, 1, 1)]
            if args.field != "all":
                selected = [row for row in selected if row[0] == args.field]
            cases.extend(natural_cases(args.capture_dir, selected, args.K))
            expected_hash = args.parameters_sha256
            if expected_hash is None:
                status = args.parameters.parent / "status.json"
                require(status.exists(), "supply --parameters-sha256 for a custom parameter file")
                expected_hash = json.loads(status.read_text())["field_parameters_sha256"]
            real_params = load_parameters(args.parameters, expected_hash)
            report["parameters_sha256"] = expected_hash
        if args.scope in ("diagnostic", "all"):
            cases.extend(diagnostic_cases())
        require(cases, "no cases")
        report["case_count"] = len(cases)
        distributions = {}
        for case in cases:
            if case.evidence["scope"] != "real_natural_native_oracle":
                continue
            entry = distributions.setdefault(case.field, {"N": 0, "M": 0, "endpoint_count_histogram": {}})
            entry["N"] += len(case.centers)
            entry["M"] += len(case.endpoints)
            for count in np.diff(case.offsets):
                key = str(int(count))
                entry["endpoint_count_histogram"][key] = entry["endpoint_count_histogram"].get(key, 0) + 1
        report["natural_endpoint_distributions"] = distributions
        report["total_fixed_solves"] = len(cases) * len(MODES) * 3
        report["bridge_sha256"], report["solver_sha256"] = digest(args.bridge), digest(args.solver)
        save()
        api = API(args.bridge, args.solver)
        for case in cases:
            case_dir = out / case.name
            case_dir.mkdir()
            np.savez(case_dir / "inputs.npz", offsets=case.offsets, centers=case.centers, endpoints=case.endpoints)
            np.savez(case_dir / "oracle.npz", **case.oracle)
            (case_dir / "evidence.json").write_text(json.dumps(case.evidence, indent=2) + "\n")
            params = case.params if case.params is not None else real_params[case.kind]
            field_handle, view = api.create_field(case.kind, params)
            handle = api.create_solver(case, view)
            for mode, mode_name in enumerate(MODES):
                first = None
                checks = (("oracle_trace1", 1, True),
                          ("reset_repeat_trace1", 1, False),
                          ("trace0_noninterference", 0, True))
                for purpose, trace, prepare in checks:
                    report["active"] = {"case": case.name, "mode": mode_name, "purpose": purpose}
                    save()
                    print(json.dumps({"action": "correctness_solve", "case": case.name,
                                      "mode": mode_name, "purpose": purpose, "trace": trace,
                                      "fixed_solves_per_case_mode": 3, "performance_test": False}), flush=True)
                    actual = api.solve_once(handle, case, mode, trace=trace, prepare_graph=prepare)
                    stem = mode_name + "_" + purpose
                    np.savez(case_dir / (stem + ".npz"), **actual)
                    row = {"case": case.name, "field": case.field, "mode": mode_name,
                           "purpose": purpose, "trace": trace,
                           "K": case.K, "N": len(case.centers), "M": len(case.endpoints),
                           "Q": len(case.oracle["trace_sdf"]), "status": "FAIL",
                           "bitwise_compared": list(actual), "mismatches": None,
                           "scope": case.evidence["scope"], "info": None, "guard": None}
                    report["rows"].append(row)
                    info = np.zeros(40, np.uint64)
                    info_rc = api.solver.bm_solver_get_info(handle, case.K, pointer(info, PU64), info.size)
                    row["info"], row["info_rc"] = info.tolist(), int(info_rc)
                    (case_dir / (stem + "_info.json")).write_text(
                        json.dumps({"rc": int(info_rc), "values": row["info"]}, indent=2) + "\n")
                    save()
                    api.check(info_rc, "solver_get_info; evidence saved")
                    require(info[0] == 1, "unsupported solver information ABI")
                    require(info[31] == 0 and info[32] == 0,
                            f"timing must remain dormant: {case.name}/{stem}; evidence saved")
                    if info[1]:
                        guard = np.zeros(8, np.uint64)
                        guard_rc = api.solver.bm_solver_debug_check(handle, pointer(guard, PU64), guard.size)
                        row["guard"], row["guard_rc"] = guard.tolist(), int(guard_rc)
                        (case_dir / (stem + "_guard.json")).write_text(
                            json.dumps({"rc": int(guard_rc), "values": row["guard"]}, indent=2) + "\n")
                        save()
                        api.check(guard_rc, "solver_debug_check; evidence saved")
                        require(guard[0] == 1 and guard[1] == info[39],
                                f"debug check did not cover all allocations: {case.name}/{stem}")
                        require(np.all(guard[2:] == 0),
                                f"device redzone/production-count violation: {case.name}/{stem}; evidence saved")
                        row["guard_status"] = "PASS"
                    else:
                        row["guard_status"] = "NOT_ENABLED"
                    reference = case.oracle if first is None else first
                    mismatches = compare(reference, actual)
                    row["mismatches"] = mismatches
                    row["status"] = "FAIL" if mismatches else "PASS"
                    save()
                    print(json.dumps(row), flush=True)
                    require(not mismatches, f"GPU correctness mismatch: {case.name}/{mode_name}/{purpose}; evidence saved")
                    if first is None:
                        first = actual
            api.check(api.solver.bm_solver_destroy(handle), "solver_destroy")
            api.check(api.field.bm_field_destroy(field_handle), "field_destroy")
        report.pop("active", None)
        report["status"] = "PASS_ON_EXECUTED_CASES"
        report["real_natural_coverage"] = sum(c.evidence["scope"] == "real_natural_native_oracle" for c in cases)
        report["performance_conclusion"] = "NOT_RUN"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        save()
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "result": str(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
