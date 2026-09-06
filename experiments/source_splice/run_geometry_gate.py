#!/usr/bin/env python3
"""Fail-fast independent height-field fidelity gate for a saved demo event.

This is not a Hausdorff estimator, a renderer, or a topology-repair proof.
The reference is the original demo occupancy function, never the replacement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


class GateStop(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateStop(message)


def face_key(face):
    values = tuple(map(int, face))
    return min(values[i:] + values[:i] for i in range(3))


def heights_at_xy(triangles, xy, tolerance):
    """Interpolate a piecewise-linear graph; reject projected degeneracy/folds."""
    heights = np.full(len(xy), np.nan)
    for triangle in triangles:
        a, b, c = triangle
        basis = np.column_stack((b[:2] - a[:2], c[:2] - a[:2]))
        scale = float(np.linalg.norm(basis, ord=2))
        require(abs(np.linalg.det(basis)) > 1e-12 * max(scale**2, 1e-30),
                'Projected triangle is degenerate; height-field metric is inapplicable.')
        uv = np.linalg.solve(basis, (xy - a[:2]).T).T
        inside = (uv[:, 0] >= -1e-12) & (uv[:, 1] >= -1e-12) & (uv.sum(axis=1) <= 1 + 1e-12)
        candidate = a[2] + uv[:, 0] * (b[2] - a[2]) + uv[:, 1] * (c[2] - a[2])
        overlap = inside & np.isfinite(heights)
        require(not np.any(np.abs(heights[overlap] - candidate[overlap]) > tolerance),
                'Multiple different heights at a shared XY sample; stop this metric.')
        heights[inside] = candidate[inside]
    return heights


def error_stats(errors):
    return {'mean_absolute_height_error': float(np.mean(errors)),
            'maximum_absolute_height_error': float(np.max(errors)),
            'p95_absolute_height_error': float(np.percentile(errors, 95))}


def improvement_gate(baseline, ours, tolerance):
    mean_gain = baseline['mean_absolute_height_error'] - ours['mean_absolute_height_error']
    max_increase = ours['maximum_absolute_height_error'] - baseline['maximum_absolute_height_error']
    return mean_gain > tolerance and max_increase <= tolerance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-root', required=True, type=Path)
    parser.add_argument('--profile-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    event = args.event_root.resolve()
    started = time.monotonic()
    result = {
        'schema': 'binoc-independent-demo-geometry-gate-v1',
        'stage': 1, 'status': 'RUNNING', 'event_root': str(event),
        'reference': 'Original static demo terrain: z = 5 * Noise2(x/10,y/10,octaves=4).',
        'metric': 'Absolute vertical error on identical XY midpoint-grid samples in the patch.',
        'selection': 'First canonical event in the existing campaign (index 0); no outcome-based reselection.',
        'grids': [32, 64, 128],
        'acceptance': 'At every grid: mean height error improves beyond numeric tolerance, maximum does not regress, same sampled footprint, no pre-existing safety check fails.',
        'numeric_tolerance': '1e-9 * max(1, patch bounding-box diagonal); not a quality-tuning parameter.',
        'limitations': ['Not Euclidean surface distance or a global topology metric.',
                        'Saved demo meshes, not a newly built paper scene.',
                        'Finite quadrature, not a continuous maximum-error certificate.',
                        'A failed gate rejects this quality claim; it does not refute all BEB1 correctness contracts.'],
        'measurements': [], 'inputs': {},
        'stage_2_started': False, 'stage_3_started': False,
    }
    def save():
        result['elapsed_seconds'] = time.monotonic() - started
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        campaign_path = event.parent / 'all_canonical_beb1_summary.json'
        campaign = json.loads(campaign_path.read_text())
        require(campaign['events'][0]['key'] == event.name, 'Requested event is not the preregistered first event.')
        validation_path = event / 'whole_mesh_validation.json'
        validation = json.loads(validation_path.read_text())
        require(validation['pass'] is True and all(validation['checks'].values()), 'Existing whole-mesh safety validation failed.')
        input_paths = [campaign_path, validation_path, args.profile_source.resolve()]
        arrays = {}
        for name in ['baseline_omp1', 'baseline_omp8', 'critical_omp1', 'critical_omp8']:
            path = event / 'runtime' / (name + '.npz')
            input_paths.append(path)
            with np.load(path, allow_pickle=False) as data:
                arrays[name] = {key: data[key].copy() for key in ['vertices', 'faces', 'tags']}
        result['inputs'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
        for prefix in ['baseline', 'critical']:
            require(all(np.array_equal(arrays[prefix+'_omp1'][k], arrays[prefix+'_omp8'][k]) for k in ['vertices', 'faces', 'tags']), 'OMP 1/8 mismatch.')
        baseline, ours = arrays['baseline_omp1'], arrays['critical_omp1']
        bv, ov = baseline['vertices'].astype(float), ours['vertices'].astype(float)
        bf, of = baseline['faces'], ours['faces']
        require(np.array_equal(bv, ov[:len(bv)]), 'Original vertex positions changed.')
        before, after = Counter(map(face_key, bf)), Counter(map(face_key, of))
        removed = np.asarray(list((before - after).elements()))
        added = np.asarray(list((after - before).elements()))
        require(len(removed) == 2 and len(added) == 4 and len(ov) == len(bv)+1,
                'Unexpected patch structure.')
        bt, ot = bv[removed], ov[added]
        lo = bt.reshape(-1, 3).min(axis=0)
        hi = bt.reshape(-1, 3).max(axis=0)
        tolerance = 1e-9 * max(1., float(np.linalg.norm(hi-lo)))
        result['absolute_numeric_tolerance'] = tolerance
        result['root'] = campaign['events'][0]['root']
        spec = importlib.util.spec_from_file_location('original_demo_profile', args.profile_source)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        # z-h(x,y) has derivative 1 in z; its zero is the exact height reference.
        for n in result['grids']:
            axis = (np.arange(n) + .5) / n
            gx, gy = np.meshgrid(lo[0] + axis*(hi[0]-lo[0]), lo[1] + axis*(hi[1]-lo[1]))
            xy = np.column_stack((gx.ravel(), gy.ravel()))
            bh = heights_at_xy(bt, xy, tolerance)
            oh = heights_at_xy(ot, xy, tolerance)
            mask = np.isfinite(bh)
            require(np.array_equal(mask, np.isfinite(oh)), 'Projected patch coverage differs; do not compare only the intersection.')
            require(np.count_nonzero(mask) >= 100, 'Too few common reference samples.')
            positions = np.column_stack((xy[mask], np.zeros(np.count_nonzero(mask))))
            reference = -profile.terrain(positions)
            require(np.all(np.isfinite(reference)), 'Reference contains non-finite values.')
            bs = error_stats(np.abs(bh[mask]-reference))
            os = error_stats(np.abs(oh[mask]-reference))
            accepted = improvement_gate(bs, os, tolerance)
            result['measurements'].append({'grid': n, 'points': int(mask.sum()),
                'baseline': bs, 'ours': os, 'gate_pass': accepted,
                'mean_error_relative_change': (os['mean_absolute_height_error']/bs['mean_absolute_height_error']-1) if bs['mean_absolute_height_error'] else None})
            save()
            require(accepted, f'Geometry improvement gate failed at grid {n}; later grids and stages not run.')
        result['status'] = 'PASS_STAGE_1'
        result['next_step'] = 'Stage 2 requires a separately verified time-window intervention; this script does not launch it.'
        save()
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        result['status'] = 'STOP_STAGE_1'
        result['stop_reason'] = str(error)
        save()
        print(json.dumps(result, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
