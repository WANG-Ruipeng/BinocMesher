#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from fractions import Fraction
from pathlib import Path
from runtime_common import exact_mesh, save_mesh_npz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--time-num", type=int, required=True)
    parser.add_argument("--time-den", type=int, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    exact = Fraction(args.time_num, args.time_den)
    vertices, faces, tags, metadata = exact_mesh(
        args.repo.resolve(), args.cache_root.resolve(), exact,
        trace=args.trace.resolve())
    save_mesh_npz(args.mesh, vertices, faces, tags, metadata)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("PASS_EXACT_SOURCE_TRIANGLE_TRACE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
