#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    lib = ctypes.CDLL(str(args.core.resolve()))
    lib.binoc_event_fixture_saddle.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
    lib.binoc_event_fixture_saddle.restype = ctypes.c_int32
    lib.binoc_event_fixture_endpoint.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.binoc_event_fixture_endpoint.restype = ctypes.c_int32
    lib.binoc_event_fixture_replay.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.binoc_event_fixture_replay.restype = ctypes.c_int32

    numerator = ctypes.c_int32()
    denominator = ctypes.c_int32()
    rc_saddle = int(lib.binoc_event_fixture_saddle(ctypes.byref(numerator), ctypes.byref(denominator)))

    official_gap = ctypes.c_double()
    shared_gap = ctypes.c_double()
    rc_endpoint = int(lib.binoc_event_fixture_endpoint(ctypes.byref(official_gap), ctypes.byref(shared_gap)))

    replay_rows = []
    for samples in (48, 96, 192, 480):
        registered = ctypes.c_int32()
        exact_hits = ctypes.c_int32()
        batch_size = ctypes.c_int32()
        rc = int(lib.binoc_event_fixture_replay(
            samples,
            ctypes.byref(registered),
            ctypes.byref(exact_hits),
            ctypes.byref(batch_size),
        ))
        replay_rows.append({
            "samples": samples,
            "return_code": rc,
            "registry_events": int(registered.value),
            "uniform_exact_hits": int(exact_hits.value),
            "simultaneous_batch_size": int(batch_size.value),
        })

    checks = {
        "saddle_return_code": rc_saddle == 0,
        "saddle_root_is_5_over_2": numerator.value == 5 and denominator.value == 2,
        "endpoint_return_code": rc_endpoint == 0,
        "official_per_face_gap_is_nonzero": official_gap.value > 1e-8,
        "shared_gap_is_zero": shared_gap.value == 0.0,
        "replay_return_codes": all(row["return_code"] == 0 for row in replay_rows),
        "registry_is_sampling_invariant": all(row["registry_events"] == 1 for row in replay_rows),
        "uniform_even_sampling_misses_exact_event": all(row["uniform_exact_hits"] == 0 for row in replay_rows),
        "simultaneous_batch_preserved": all(row["simultaneous_batch_size"] == 2 for row in replay_rows),
    }
    result = {
        "verdict": "PASS_P2_P4_CORE_FIXTURE_SMOKE" if all(checks.values()) else "STOP_P2_P4_CORE_FIXTURE_SMOKE",
        "core_so": str(args.core.resolve()),
        "core_so_sha256": hashlib.sha256(args.core.read_bytes()).hexdigest(),
        "saddle": {"return_code": rc_saddle, "numerator": numerator.value, "denominator": denominator.value},
        "endpoint": {"return_code": rc_endpoint, "official_per_face_gap": official_gap.value, "shared_gap": shared_gap.value},
        "replay": replay_rows,
        "checks": checks,
    }
    (args.output / "fixture_smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(result["verdict"])
    return 0 if result["verdict"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
