#!/usr/bin/env python3
"""Small actual-C++ raw/smoothed diagnostic; no C1 overlay or rendering.

The frozen input cache is copied before initialization because the ordinary
Python wrapper writes log/manifest files. Arrays stay in RAM; outputs are JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'source_splice'
sys.path.insert(0, str(SOURCE))
from runtime_common import initialize_mesher, canonical_mesh_hash
from compile_critical_beb1_event_ir import compile_ordinary_patch


def manifest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def point_key(point):
    return tuple(float(x) for x in point)


def oriented_coordinate_face(points):
    keys = tuple(point_key(p) for p in points)
    return min(keys[i:]+keys[:i] for i in range(3))


def patch_match(vertices, faces, patch):
    positions = np.asarray([patch['boundary_positions'][k]
                            for k in patch['boundary_cycle']], dtype=np.float32).astype(float)
    actual = defaultdict(list)
    for i, point in enumerate(vertices):
        actual[point_key(point)].append(i)
    matches = [actual[point_key(point)] for point in positions]
    source = dict(zip(patch['boundary_cycle'], positions))
    expected = [oriented_coordinate_face([source[k] for k in face])
                for face in patch['source_faces']]
    counts = Counter(oriented_coordinate_face(vertices[face]) for face in faces)
    return {
        'boundary_coordinate_match_counts': [len(m) for m in matches],
        'boundary_coordinate_match_ids': matches,
        'source_face_coordinate_match_counts': [counts[k] for k in expected],
        'unique_coordinate_match': all(len(m) == 1 for m in matches),
        'meaning': 'Diagnostic coordinate lookup, NOT SourceVID identity resolution or an exterior-intersection check.',
    }


def exact_call(mesher, tau, smooth):
    from binocmesher.utils.interface import AsInt
    vc, fc = np.zeros(1, np.int32), np.zeros(1, np.int32)
    started = time.monotonic()
    status = int(mesher.run_slicing_rational(tau.numerator, tau.denominator,
                                            AsInt(vc), AsInt(fc), bool(smooth)))
    if status != 0:
        detail = mesher.slicing_last_error()
        raise RuntimeError(detail.decode() if detail else f'exact slicer status {status}')
    try:
        vertices = np.zeros((int(vc[0]), 3), np.float64)
        faces = np.zeros((int(fc[0]), 3), np.int32)
        tags = np.zeros(int(vc[0]), np.int32)
        mesher.slicing_output(0, mesher.AF(vertices), AsInt(faces), AsInt(tags))
        if not np.all(np.isfinite(vertices)):
            raise RuntimeError('Runtime emitted non-finite coordinates.')
    finally:
        mesher.slicing_clean_up()
    return (vertices, faces, tags), time.monotonic()-started


def run(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    cache, event = args.cache_root.resolve(), args.event_root.resolve()
    if cache in out.parents or event in out.parents:
        raise ValueError('Output must not be inside a frozen input.')
    if any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError('Refuse a cache with symlinks for the isolated copy.')
    cache_bytes = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
    if cache_bytes > 100_000_000:
        raise ValueError('Diagnostic cache exceeds the 100 MB copy budget.')
    before = manifest(cache)
    out.mkdir(parents=True)
    copied = out/'mutable_cache_copy'
    shutil.copytree(cache, copied)
    report = {'schema': 'c1-runtime-baseline-audit-v1', 'status': 'RUNNING',
              'omp_threads': args.omp, 'cache_copy_bytes': cache_bytes,
              'original_cache': str(cache), 'runtime_plan_emitted': False,
              'render_started': False, 'cases': [],
              'limits': ['Five exact times of one fixed demo, not whole-window certification.',
                         'Raw/smoothed C++ comparison, not C1 runtime integration.',
                         'Coordinate matching is not an identity-splice proof.',
                         'No beauty, SSIM, or visibility conclusion.']}

    def save():
        (out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    save()
    try:
        if args.omp not in (1, 8):
            raise ValueError('Only the predeclared OMP 1/8 settings are supported.')
        os.environ['OMP_NUM_THREADS'] = str(args.omp)
        for key in ('BINOC_SOURCE_SPLICE_PLAN', 'BINOC_SOURCE_SPLICE_AUDIT',
                    'BINOC_SOURCE_SPLICE_TRACE'):
            os.environ.pop(key, None)
        ir_path = event/'critical_beb1_event_ir.json'
        ir = json.loads(ir_path.read_text())
        levels = [Fraction(**ir['one_sided_window'][k]) for k in ('lower', 'critical', 'upper')]
        lower, root, upper = levels
        times = [lower, (lower+root)/2, root, (root+upper)/2, upper]
        report['inputs_sha256'] = {
            str(ir_path): hashlib.sha256(ir_path.read_bytes()).hexdigest(),
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            str(args.repo/'binocmesher/lib/core.so'): hashlib.sha256((args.repo/'binocmesher/lib/core.so').read_bytes()).hexdigest(),
        }
        mesher = initialize_mesher(args.repo.resolve(), copied)
        for tau in times:
            patch, _ = compile_ordinary_patch(cache, tau, event_id=ir['event']['event_id'])
            variants = {}
            case = {'time': {'numerator': tau.numerator, 'denominator': tau.denominator},
                    'variants': {}}
            for name, smooth in [('raw', False), ('extra_smooth', True)]:
                arrays, elapsed = exact_call(mesher, tau, smooth)
                variants[name] = arrays
                v, f, tags = arrays
                case['variants'][name] = {'extra_smooth': smooth, 'vertices': len(v),
                    'faces': len(f), 'elapsed_seconds': elapsed,
                    'mesh_hash': canonical_mesh_hash(v, f, tags),
                    'patch_coordinate_lookup': patch_match(v, f, patch)}
            raw, smooth = variants['raw'], variants['extra_smooth']
            case['raw_smoothed_arrays_identical'] = {
                name: bool(np.array_equal(a, b))
                for name, a, b in zip(('vertices', 'faces', 'tags'), raw, smooth)}
            if tau == root:
                saved = event/'runtime'/f'baseline_omp{args.omp}.npz'
                with np.load(saved, allow_pickle=False) as frozen:
                    case['raw_matches_frozen_c0_baseline'] = {
                        name: bool(np.array_equal(a, frozen[name]))
                        for name, a in zip(('vertices', 'faces', 'tags'), raw)}
            report['cases'].append(case)
            save()
        report['status'] = 'COMPLETE_BASELINE_DIAGNOSTIC_ONLY'
    except Exception as error:
        report['status'] = 'STOP_RUNTIME_DIAGNOSTIC'
        report['stop_reason'] = str(error)
    finally:
        report['original_cache_unchanged'] = manifest(cache) == before
        if not report['original_cache_unchanged']:
            report['status'] = 'STOP_INPUT_MUTATION'
        report['output_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        save()
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']),
                      'output': str(out/'result.json'),
                      'original_cache_unchanged': report['original_cache_unchanged']}))
    return 0 if report['status'] == 'COMPLETE_BASELINE_DIAGNOSTIC_ONLY' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'cache-root', 'event-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--omp', type=int, choices=(1, 8), required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
