"""Bounded observers for one native BinocMesher instance; no CUDA initialization.

Compile native_capture.cpp into the SAME core.so as the original seven sources.
attach never imports a second BinocMesher namespace and verifies symbol identity.
All durations here are wall observations, never inferred CUDA/coordination costs.
"""
from __future__ import annotations

import contextlib
import ctypes as ct
import functools
import json
import sys
import time
from pathlib import Path
import numpy as np


class CaptureFailure(RuntimeError):
    pass


class CaptureHooks:
    def __init__(self, mesher, out_dir, library_path=None):
        if getattr(mesher, "_bm_capture_hook", None) is not None:
            raise RuntimeError("capture already attached")
        self.mesher = mesher
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=False)
        module = sys.modules[type(mesher).__module__]
        self.library_path = Path(library_path) if library_path else Path(module.__file__).parent / "lib/core.so"
        self.dll = ct.CDLL(str(self.library_path))
        original = mesher.bisection_hypermesh_verts
        if ct.cast(original, ct.c_void_p).value != ct.cast(self.dll.bisection_hypermesh_verts, ct.c_void_p).value:
            raise RuntimeError("observer core.so differs from mesher core.so; shared state would be wrong")
        fptr = ct.POINTER(ct.c_float)
        i32p = ct.POINTER(ct.c_int32)
        self._bind("bm_capture_input", [ct.c_char_p, ct.c_int])
        self._bind("bm_capture_round_before", [ct.c_char_p, ct.c_int, fptr, fptr])
        self._bind("bm_capture_round_after", [ct.c_char_p, ct.c_int, fptr, fptr])
        self._bind("bm_capture_prefinish", [ct.c_char_p, ct.c_int, fptr, fptr])
        self._bind("bm_capture_final", [ct.c_char_p, ct.c_int, fptr, fptr])
        self._bind("bm_replay_snapshot", [ct.c_int, ct.c_int])
        self._bind("bm_replay_restore", [ct.c_int])
        self._bind("bm_replay_invalidate", [], None)
        self._bind("bm_num_elements", [])
        self._bind("bm_output_sizes", [ct.POINTER(ct.c_int64)])
        self._bind("bm_copy_outputs", [fptr, i32p, ct.POINTER(ct.c_int8), i32p])
        self.original = {}
        self.group = None
        self.batch = -1
        self.round = 0
        self.active = False
        self.query_stage = "outside_bisection"
        self.query_index = 0
        self.replay_run_name = None
        self._inside = False
        self.counts = None
        self.center_counts = None
        self.on_batch_ready = None
        self.on_batch_finished = None
        self.records = []
        self.closed = False
        # Every existing native ctypes entry is wrapped once. Special observers
        # below add capture work, while ordinary entries get only a wall timer.
        for name, function in list(vars(mesher).items()):
            if isinstance(function, ct._CFuncPtr):
                self.original[name] = function
                setattr(mesher, name, self._wrap(name, function))
        mesher._bm_capture_hook = self
        self._event("attach", library=str(self.library_path), scalar_contract="geometry=float64,sdf=float32,output=float32",
                    observation_is_not_benchmark=True)

    def _bind(self, name, args, restype=ct.c_int):
        f = getattr(self.dll, name)
        f.argtypes, f.restype = args, restype

    def _event(self, kind, **values):
        row = {"type": kind, "group": self.group, "batch": self.batch,
               "round": self.round, "query_stage": self.query_stage,
               "run_name": self.replay_run_name, **values}
        self.records.append(row)
        with (self.out_dir / "events.jsonl").open("a", encoding="utf8") as f:
            f.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    def _path(self, suffix):
        return self.out_dir / f"g{self.group:05d}_b{self.batch:05d}_{suffix}"

    def _capture(self, function, suffix, *args):
        path = self._path(suffix)
        start = time.perf_counter_ns()
        rc = function(str(path).encode(), *args)
        self._event("capture", function=function.__name__, path=path.name, returncode=rc,
                    capture_wall_ns=time.perf_counter_ns() - start)
        if rc != 0:
            raise CaptureFailure(f"{function.__name__} returned {rc}; saved evidence: {path}")

    def _native(self, name, function, args):
        start = time.perf_counter_ns()
        try:
            return function(*args)  # exactly one native call
        finally:
            elapsed = time.perf_counter_ns() - start
            context = {}
            if name == "bisection_init_t":
                # The native call prepares the new group; self.group changes
                # only after it returns, so do not attribute this to the old one.
                context = {"group": int(args[0]), "batch": -1, "round": 0,
                           "query_stage": "group_load", "run_name": None}
            self._event("native_call", function=name, native_wall_ns=elapsed, **context)

    def _wrap(self, name, function):
        @functools.wraps(function)
        def wrapped(*args):
            if self._inside:
                raise RuntimeError(f"reentrant observer invocation: {name}")
            self._inside = True
            try:
                if name == "bisection_hypermesh_verts_iter":
                    self._capture(self.dll.bm_capture_round_before, f"r{self.round:03d}_before.jsonl", self.round, *args)
                elif name == "bisection_hypermesh_verts_finishing":
                    # This must execute before any native resize/assert/write.
                    self._capture(self.dll.bm_capture_prefinish, "prefinish.jsonl", *args)
                elif name in ("write_final_hypermesh", "bisection_clean_up"):
                    self.dll.bm_replay_invalidate()
                result = self._native(name, function, args)
                if name == "bisection_init_t":
                    self.group, self.batch, self.round = int(args[0]), -1, 0
                    self.replay_run_name = None
                    self.active = False
                    self.query_stage = "group_prepared"
                    if self.dll.bm_replay_snapshot(0, self.group):
                        raise CaptureFailure("prepared-group snapshot failed")
                    self._event("snapshot", slot=0, boundary="after_original_group_load_and_unique_counts",
                                costs_excluded=["checkpoint_read", "group_load", "unique_endpoint_counts"])
                elif name == "bisection_hypermesh_verts":
                    if result:
                        self.batch += 1
                        self.round = 0
                        self.active = True
                        n = self.dll.bm_num_elements()
                        self.counts = np.ctypeslib.as_array(args[1], shape=(n,)).copy()
                        self.center_counts = np.ctypeslib.as_array(args[2], shape=(n,)).copy()
                        self._capture(self.dll.bm_capture_input, "input.jsonl", int(args[0]))
                        if self.dll.bm_replay_snapshot(1, self.group):
                            raise CaptureFailure("prepared-batch snapshot failed")
                        self._event("snapshot", slot=1, boundary="after_original_batch_prepare_before_center_query",
                                    counts=self.counts.tolist(), center_counts=self.center_counts.tolist())
                        if self.on_batch_ready:
                            # Callback may use original[] for bounded replay.
                            # Driver must restore_batch() in try/finally before
                            # the original while-loop body resumes.
                            self.on_batch_ready(self)
                    else:
                        self.active = False
                elif name == "bisection_hypermesh_verts_output_center":
                    self.query_stage = "center"
                    self._save_query(args[0], int(self.center_counts.sum()), "centers_fp64.npz")
                elif name == "bisection_hypermesh_verts_output":
                    self.query_stage = "final_right" if int(args[1]) else f"round_{self.round}"
                    suffix = "final_right_fp64.npz" if int(args[1]) else f"r{self.round:03d}_positions_fp64.npz"
                    self._save_query(args[0], int(self.counts.sum()), suffix)
                elif name == "bisection_hypermesh_verts_iter":
                    self._capture(self.dll.bm_capture_round_after, f"r{self.round:03d}_after.jsonl", self.round, *args)
                    self.round += 1
                elif name == "bisection_hypermesh_verts_finishing":
                    self._capture(self.dll.bm_capture_final, "final.jsonl", *args)
                    np.savez(self._path("native_outputs.npz"), **self.copy_outputs())
                    self.active = False
                    self.query_stage = "batch_finished"
                    if self.on_batch_finished:
                        self.on_batch_finished(self)
                elif name == "write_final_hypermesh":
                    self.query_stage = "outside_bisection"
                return result
            finally:
                self._inside = False
        return wrapped

    def _save_query(self, ptr, n, suffix):
        values = np.ctypeslib.as_array(ptr, shape=(n * 3,)).reshape(n, 3).copy()
        if values.dtype != np.dtype("float64"):
            raise CaptureFailure("native query precision changed")
        np.savez(self._path(suffix), positions=values, position_bits=values.view(np.uint64))

    def record_field(self, field_name, *, xyz_fp64_before, xyz_fp64_after, xyz_fp32,
                     raw_sdf, aux, processed_sdf, timings_ns=None, metadata=None):
        """Call immediately after the genuine field callback; arrays are copied now.

        before/after refer to the field's real coordinate-offset transform, not
        an invented transform. aux is a name->array mapping of every real aux
        output (empty only if the genuine callback has none). Never recompute
        the field to observe it. Capture outside bisection may be omitted by
        caller; actual field callback should always execute exactly once.
        """
        if not self.active:
            return None
        arrays = {
            "xyz_fp64_before": np.array(xyz_fp64_before, copy=True),
            "xyz_fp64_after": np.array(xyz_fp64_after, copy=True),
            "xyz_fp32": np.array(xyz_fp32, copy=True),
            "raw_sdf": np.array(raw_sdf, copy=True),
            "processed_sdf": np.array(processed_sdf, copy=True),
        }
        for key, value in aux.items():
            if not key or "/" in key or "\\" in key:
                raise ValueError("aux names must be simple nonempty strings")
            arrays["aux_" + key] = np.array(value, copy=True)
        if arrays["xyz_fp64_before"].dtype != np.float64 or arrays["xyz_fp64_after"].dtype != np.float64:
            raise CaptureFailure("field must expose actual float64 before/after coordinates")
        if arrays["xyz_fp32"].dtype != np.float32 or arrays["raw_sdf"].dtype != np.float32 or arrays["processed_sdf"].dtype != np.float32:
            raise CaptureFailure("field precision changed")
        n = len(arrays["xyz_fp64_before"])
        if any(arrays[k].shape != (n, 3) for k in ("xyz_fp64_before", "xyz_fp64_after", "xyz_fp32")):
            raise CaptureFailure("field coordinate shape changed")
        if arrays["raw_sdf"].size != n or arrays["processed_sdf"].size != n:
            raise CaptureFailure("SDF query count changed")
        for key in ("xyz_fp64_before", "xyz_fp64_after"):
            arrays[key + "_bits"] = arrays[key].view(np.uint64)
        for key in ("xyz_fp32", "raw_sdf", "processed_sdf"):
            arrays[key + "_bits"] = arrays[key].view(np.uint32)
        self.query_index += 1
        path = self._path(f"field_{self.query_index:06d}.npz")
        np.savez(path, **arrays)
        self._event("field", field=field_name, file=path.name, count=n,
                    timings_ns=timings_ns or {}, metadata=metadata or {},
                    timing_rule="nested values are not additive; no host-minus-kernel attribution")
        return path

    @contextlib.contextmanager
    def span(self, name, **metadata):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self._event("span", name=name, wall_ns=time.perf_counter_ns() - start, **metadata)

    def copy_outputs(self):
        sizes = np.zeros(2, np.int64)
        if self.dll.bm_output_sizes(sizes.ctypes.data_as(ct.POINTER(ct.c_int64))):
            raise CaptureFailure("output sizes failed")
        n, m = map(int, sizes)
        xyz, times = np.empty((n, 3), np.float32), np.empty((n, 2), np.int32)
        tags, mapping = np.empty(n, np.int8), np.empty(m, np.int32)
        rc = self.dll.bm_copy_outputs(xyz.ctypes.data_as(ct.POINTER(ct.c_float)),
                                     times.ctypes.data_as(ct.POINTER(ct.c_int32)),
                                     tags.ctypes.data_as(ct.POINTER(ct.c_int8)),
                                     mapping.ctypes.data_as(ct.POINTER(ct.c_int32)))
        if rc:
            raise CaptureFailure("output copy failed")
        return {"xyz": xyz, "times": times, "tags": tags, "vertex_map": mapping}

    def restore_batch(self):
        if self.dll.bm_replay_restore(1):
            raise CaptureFailure("batch restore failed or graph lifetime ended")
        self.round = 0
        self.active = True
        self.query_stage = "batch_restored"
        self.replay_run_name = None

    def replay_batch(self, kernels, K=None, run_name="replay"):
        """Correctness replay, includes evidence IO; never use this for timing.

        Invoke from on_batch_ready with driver try/finally restore_batch()
        before normal caller continues, or from on_batch_finished while graph/
        nodes remain loaded.
        This adapter bypasses completion-file checks and observer wrappers,
        directly invokes each original ctypes function once per required step.
        """
        if len(kernels) != len(self.counts):
            raise ValueError("kernel order must equal native element order")
        K = self.mesher.bisection_iters if K is None else int(K)
        if K < 0 or "/" in run_name or "\\" in run_name:
            raise ValueError("invalid K or replay name")
        self.restore_batch()
        self.replay_run_name = run_name
        self.query_stage = "center"
        m, n = int(self.counts.sum()), int(self.center_counts.sum())
        positions, centers = np.zeros((m, 3), np.float64), np.zeros((n, 3), np.float64)
        f = self.original
        f["bisection_hypermesh_verts_output_center"](self.mesher.AF(centers))
        center_sdfs = []
        for e in range(len(kernels)):
            lo, hi = int(self.center_counts[:e].sum()), int(self.center_counts[:e+1].sum())
            center_sdfs.append(self.mesher.kernel_caller(kernels[e:e+1], centers[lo:hi]))
        center_sdfs = np.ascontiguousarray(np.concatenate(center_sdfs), dtype=np.float32)
        for r in range(K + 1):
            self.round = r
            final = int(r == K)
            self.query_stage = "final_right" if final else f"round_{r}"
            f["bisection_hypermesh_verts_output"](self.mesher.AF(positions), final)
            sdfs = np.zeros(m, np.float32)
            for e in range(len(kernels)):
                lo, hi = int(self.counts[:e].sum()), int(self.counts[:e+1].sum())
                sdfs[lo:hi] = self.mesher.kernel_caller(kernels[e:e+1], positions[lo:hi])[:, 0]
            sp, cp = self.mesher.sdf_AF(sdfs), self.mesher.sdf_AF(center_sdfs)
            if final:
                self._capture(self.dll.bm_capture_prefinish, f"{run_name}_prefinish.jsonl", self.group, sp, cp)
                f["bisection_hypermesh_verts_finishing"](self.group, sp, cp)
                self._capture(self.dll.bm_capture_final, f"{run_name}_final.jsonl", self.group, sp, cp)
            else:
                self._capture(self.dll.bm_capture_round_before, f"{run_name}_r{r:03d}_before.jsonl", r, sp, cp)
                f["bisection_hypermesh_verts_iter"](sp, cp)
                self._capture(self.dll.bm_capture_round_after, f"{run_name}_r{r:03d}_after.jsonl", r, sp, cp)
        outputs = self.copy_outputs()
        np.savez(self._path(f"{run_name}_outputs.npz"), **outputs)
        self.query_stage = "replay_finished"
        self._event("correctness_replay_complete", K=K, M=m, N=n, evidence_io_included=True,
                    boundary="prepared_batch_to_native_bisection_output")
        return outputs

    def close(self):
        if self.closed:
            return
        for name, function in self.original.items():
            setattr(self.mesher, name, function)
        self.dll.bm_replay_invalidate()
        self.mesher._bm_capture_hook = None
        self.closed = True
        self._event("detach")


def attach(mesher, out_dir, *, library_path=None):
    return CaptureHooks(mesher, out_dir, library_path)
