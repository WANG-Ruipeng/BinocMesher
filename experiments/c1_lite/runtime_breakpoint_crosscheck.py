#!/usr/bin/env python3
"""Actual baseline slices at the frozen four-demo breakpoint union.

No plan, renderer, compilation, or saved mesh arrays. Each OMP worker has an
isolated temporary cache and reports exact-coordinate diagnostics, not source
identity equivalence or a continuous runtime certificate.
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
import subprocess
import sys
import tempfile
import time

import numpy as np

import runtime_retained as retained
import window_source as source
from runtime_baseline_audit import exact_call, manifest, oriented_coordinate_face
from runtime_common import initialize_mesher, canonical_mesh_hash

SCHEMA = 'c1-lite-runtime-breakpoint-crosscheck-v1'
MAX_CACHE_BYTES = 10 * 1024 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fresh(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError('Crosscheck JSON exceeds two MiB.')
    with path.open('xb') as handle:
        handle.write(data)


def round_fraction_binary32(value):
    """Exact nearest binary32, ties-even; float64 is only a candidate seed."""
    value = Fraction(value)
    candidate = np.float32(float(value))
    if not np.isfinite(candidate):
        raise ValueError('Fraction exceeds supported finite binary32 range.')
    candidates = [candidate, np.nextafter(candidate, np.float32(-np.inf)),
                  np.nextafter(candidate, np.float32(np.inf))]
    finite = [x for x in candidates if np.isfinite(x)]
    def key(x):
        bits = int(np.asarray(x, dtype=np.float32).view(np.uint32))
        return abs(Fraction.from_float(float(x))-value), bits & 1
    return float(min(finite, key=key))


def exact_mesh_metrics(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if not np.all(np.isfinite(vertices)):
        raise ValueError('Nonfinite actual mesh.')
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Expected triangle arrays.')
    if len(faces) and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError('Actual mesh contains invalid vertex indices.')
    points = [tuple(Fraction.from_float(float(x)) for x in p) for p in vertices]
    repeated, zero_area = 0, 0
    for row in faces:
        ids = tuple(int(x) for x in row)
        repeated += len(set(ids)) < 3
        a, b, c = (points[i] for i in ids)
        u = tuple(y-x for x, y in zip(a, b))
        v = tuple(y-x for x, y in zip(a, c))
        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        zero_area += all(x == 0 for x in cross)
    return {'vertices': len(vertices), 'faces': len(faces),
        'repeated_index_faces': repeated, 'exact_zero_area_faces': zero_area,
        'distinct_index_exact_zero_area_faces': zero_area-repeated,
        'zero_area_arithmetic': 'Exact Fractions of actual finite binary64 output coordinates; no epsilon.',
        'coordinates_exactly_binary32': bool(np.array_equal(vertices, vertices.astype(np.float32).astype(np.float64)))}


def witness_matches(vertices, faces, witnesses):
    counts = Counter(oriented_coordinate_face(vertices[face]) for face in faces)
    coordinates = defaultdict(list)
    for i, point in enumerate(vertices):
        coordinates[tuple(float(x) for x in point)].append(i)
    result = []
    for witness in witnesses:
        points = [[round_fraction_binary32(source.fr(x)) for x in row]
                  for row in witness['positions_t0']]
        matches = [coordinates[tuple(p)] for p in points]
        count = counts[oriented_coordinate_face(points)]
        result.append({'event': witness['event'], 'source_vertices': witness['source_vertices'],
            'owners': witness['owners'], 'shared_source_vertices': witness['shared_source_vertices'],
            'source_model_repeated_effective_SourceVID': witness['repeated_effective_SourceVID'],
            'expected_coordinates_binary32_RNE': points,
            'vertex_coordinate_match_counts': [len(rows) for rows in matches],
            'vertex_coordinate_match_ids': matches, 'oriented_coordinate_face_match_count': count,
            'status': 'MATCH_COORDINATE_DIAGNOSTIC_ONLY' if count else 'UNKNOWN_COORDINATE_MODEL_MISMATCH',
            'identity_equivalence': 'UNKNOWN',
            'meaning': 'Exact RNE-coordinate lookup, not SourceVID/owner proof; zero matches never means the actual face is absent.'})
    return result


def load_protocol(source_root, retained_root):
    times, witnesses, inputs = set(), defaultdict(list), {}
    for key in retained.EVENT_KEYS:
        path = source_root/key/'source.json'
        report = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        times.update(source.fr(point['time']) for point in report['breakpoint_points'])
        if key not in (retained.EVENT_KEYS[0], retained.EVENT_KEYS[3]):
            continue
        path = retained_root/(key+'.json')
        diagnostic = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        if diagnostic['source_report_sha256'] != inputs[str(source_root/key/'source.json')]:
            raise ValueError('Raw diagnostic/source protocol digest mismatch.')
        for unit in diagnostic['units']:
            if unit['kind'] != 'actual_singleton':
                continue
            if unit['interface_degenerate_source_faces'] != len(unit['interface_degenerate_witnesses']):
                raise ValueError('Stored singleton interface witnesses are truncated; refuse incomplete crosscheck.')
            tau = source.fr(unit['t0'])
            for row in unit['interface_degenerate_witnesses']:
                witnesses[tau].append({'event': key, **row})
    return sorted(times), witnesses, inputs


def run_worker(args):
    output, cache = args.output.resolve(), args.cache_root.resolve()
    if output.exists() or cache == output or cache in output.parents:
        raise ValueError('Worker output must be fresh and outside the original cache.')
    if any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError('Cache symlinks are unsupported.')
    cache_bytes = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
    if cache_bytes > MAX_CACHE_BYTES:
        raise ValueError('Cache exceeds ten MiB isolation budget.')
    times, witnesses, inputs = load_protocol(args.source_root.resolve(), args.retained_root.resolve())
    before = manifest(cache)
    inventory = source.read_inventory(cache)
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {'schema': SCHEMA, 'status': 'RUNNING', 'omp_threads': args.omp,
        'times': [source.fj(t) for t in times], 'cases': [],
        'cache_copy_bytes': cache_bytes, 'inputs_sha256': inputs,
        'core_so_sha256': sha(args.repo/'binocmesher/lib/core.so'),
        'script_sha256': sha(__file__), 'runtime_plan_loaded': False,
        'render_started': False, 'runtime_admission': False,
        'limits': ['Fixed breakpoint diagnostics, not proof over every interior runtime time.',
                   'Counts/coordinate matches do not prove SourceVID or raw-owner equivalence.',
                   'Extra_smooth is observed, not represented by the raw source model.']}
    temporary = None
    try:
        os.environ['OMP_NUM_THREADS'] = str(args.omp)
        for name in ('BINOC_SOURCE_SPLICE_PLAN', 'BINOC_SOURCE_SPLICE_AUDIT', 'BINOC_SOURCE_SPLICE_TRACE'):
            os.environ.pop(name, None)
        with tempfile.TemporaryDirectory(prefix='c1-breakpoint-') as temporary:
            copied = Path(temporary)/'cache'
            shutil.copytree(cache, copied)
            mesher = initialize_mesher(args.repo.resolve(), copied)
            for tau in times:
                if time.monotonic()-started > args.max_seconds:
                    raise RuntimeError('Bounded worker time budget reached.')
                model = retained.build_retained_unit(inventory, tau, tau)
                model_summary = retained.compact_manifest(model)
                case = {'time': source.fj(tau), 'raw_model_baseline': {
                    name: model_summary[name] for name in ('retained_source_face_count',
                        'repeated_effective_SourceVID_faces', 'identically_zero_area_source_faces',
                        'owner_ledger_sha256', 'retained_manifest_sha256')}, 'variants': {}}
                arrays = {}
                for mode, smooth in (('raw', False), ('extra_smooth', True)):
                    (v, f, tags), elapsed = exact_call(mesher, tau, smooth)
                    arrays[mode] = (v, f, tags)
                    details = exact_mesh_metrics(v, f)
                    details.update(mesh_hash=canonical_mesh_hash(v, f, tags), elapsed_seconds=elapsed,
                        interface_witnesses=witness_matches(v, f, witnesses[tau]))
                    case['variants'][mode] = details
                raw, smooth = arrays['raw'], arrays['extra_smooth']
                case['raw_smooth_arrays_identical'] = {
                    name: bool(np.array_equal(a, b)) for name, a, b in zip(('vertices', 'faces', 'tags'), raw, smooth)}
                case['raw_source_face_count_matches_actual'] = len(raw[1]) == len(model.triangles)
                case['count_match_is_identity_proof'] = False
                report['cases'].append(case)
                print(json.dumps({'omp': args.omp, 'time': str(tau),
                    'raw_faces': len(raw[1]), 'raw_repeated': case['variants']['raw']['repeated_index_faces']}), flush=True)
        report['status'] = 'COMPLETE_BASELINE_BREAKPOINT_DIAGNOSTIC'
    except Exception as error:
        report['status'] = 'STOP_DIAGNOSTIC'
        report['reason'] = str(error)
    finally:
        report['original_cache_unchanged'] = manifest(cache) == before
        report['temporary_cache_removed'] = temporary is not None and not Path(temporary).exists()
        report['elapsed_seconds'] = time.monotonic()-started
        if not report['original_cache_unchanged'] or not report['temporary_cache_removed']:
            report['status'] = 'STOP_ISOLATION_CHECK'
        write_fresh(output/'result.json', report)
    return 0 if report['status'] == 'COMPLETE_BASELINE_BREAKPOINT_DIAGNOSTIC' else 2


def compare_workers(first, second):
    cases = []
    if first['times'] != second['times'] or len(first['cases']) != len(second['cases']):
        return {'status': 'UNKNOWN_INCOMPLETE_OR_DIFFERENT_PROTOCOL', 'cases': []}
    for a, b in zip(first['cases'], second['cases']):
        if a['time'] != b['time']:
            raise ValueError('Worker time order mismatch.')
        equal = {mode: a['variants'][mode]['mesh_hash'] == b['variants'][mode]['mesh_hash']
                 for mode in ('raw', 'extra_smooth')}
        cases.append({'time': a['time'], 'omp_mesh_hash_equal': equal})
    complete = all(r['status'] == 'COMPLETE_BASELINE_BREAKPOINT_DIAGNOSTIC' for r in (first, second))
    return {'status': 'PASS_OBSERVED_OMP_DETERMINISM' if complete and all(all(c['omp_mesh_hash_equal'].values()) for c in cases)
            else 'UNKNOWN_OR_MISMATCH', 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'cache-root', 'source-root', 'retained-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--omp', type=int, choices=(1, 8))
    parser.add_argument('--max-seconds', type=float, default=240)
    args = parser.parse_args()
    if args.omp:
        return run_worker(args)
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Crosscheck output must be fresh.')
    output.mkdir(parents=True)
    reports = []
    for omp in (1, 8):
        cmd = [sys.executable, '-B', str(Path(__file__).resolve())]
        for name in ('repo', 'cache_root', 'source_root', 'retained_root'):
            cmd += ['--'+name.replace('_', '-'), str(getattr(args, name).resolve())]
        cmd += ['--output', str(output/f'omp{omp}'), '--omp', str(omp), '--max-seconds', str(args.max_seconds)]
        env = dict(os.environ, OMP_NUM_THREADS=str(omp), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        process = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        path = output/f'omp{omp}'/'result.json'
        if not path.exists():
            raise RuntimeError('Worker produced no report: '+process.stdout[-3000:])
        reports.append(json.loads(path.read_text()))
        print(json.dumps({'omp': omp, 'exit_code': process.returncode, 'status': reports[-1]['status'],
                          'cases': len(reports[-1]['cases'])}), flush=True)
        if process.returncode:
            break
    comparison = compare_workers(*reports) if len(reports) == 2 else {'status': 'UNKNOWN_INCOMPLETE_WORKERS'}
    summary = {'schema': SCHEMA, 'omp_comparison': comparison, 'runtime_admission': False,
        'worker_statuses': [r['status'] for r in reports],
        'all_original_cache_unchanged': all(r['original_cache_unchanged'] for r in reports),
        'all_temporary_caches_removed': all(r['temporary_cache_removed'] for r in reports),
        'worker_artifacts': {str(p.relative_to(output)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                             for p in output.rglob('result.json')}}
    write_fresh(output/'summary.json', summary)
    total = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if total > MAX_OUTPUT_BYTES:
        raise RuntimeError('Aggregate report exceeds two MiB; no mesh arrays were saved.')
    print(json.dumps({'status': comparison['status'], 'bytes': total, 'output': str(output)}), flush=True)
    return 0 if comparison['status'] == 'PASS_OBSERVED_OMP_DETERMINISM' else 2


if __name__ == '__main__':
    raise SystemExit(main())
