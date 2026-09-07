"""Independent terrain reference and explicit-denominator image diagnostics."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tv0_tv4'))
from run_lightweight_profile import terrain


def heights(xy):
    xy = np.asarray(xy, np.float64)
    return -terrain(np.column_stack((xy, np.zeros(len(xy)))))


def terrain_reference(bounds, resolution):
    x0, x1, y0, y1 = bounds
    if not x0 < x1 or not y0 < y1 or not 2 <= resolution <= 512:
        raise ValueError('Invalid bounded reference grid.')
    x, y = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    xy = np.column_stack((x.ravel(), y.ravel()))
    vertices = np.column_stack((xy, heights(xy)))
    ids = np.arange(resolution*resolution).reshape(resolution, resolution)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, 1:].ravel(), ids[1:, :-1].ravel()
    faces = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    return vertices, faces.astype(np.int32)


def terrain_normals(xy, step=1e-4):
    xy = np.asarray(xy, np.float64)
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Finite positive difference step required.')
    dx, dy = np.asarray([step, 0.]), np.asarray([0., step])
    hx = (heights(xy+dx)-heights(xy-dx))/(2*step)
    hy = (heights(xy+dy)-heights(xy-dy))/(2*step)
    result = np.column_stack((-hx, -hy, np.ones(len(xy))))
    return result/np.linalg.norm(result, axis=1)[:, None]


def angles(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1)))


def stats(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {'count': 0, 'mean': None, 'p95': None, 'maximum': None, 'rmse': None}
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite values in an evaluated image metric.')
    return {'count': int(len(values)), 'mean': float(values.mean()),
            'p95': float(np.percentile(values, 95)), 'maximum': float(values.max()),
            'rmse': float(np.sqrt(np.mean(values*values)))}


def silhouette_boundary(mask):
    mask = np.asarray(mask, bool)
    p = np.pad(mask, 1)
    interior = p[1:-1, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:] & p[:-2, 1:-1] & p[2:, 1:-1]
    return mask & ~interior


def image_metrics(actual, reference, roi):
    roi = np.asarray(roi, bool)
    a, b = np.asarray(actual['mask'], bool), np.asarray(reference['mask'], bool)
    if a.shape != roi.shape or b.shape != roi.shape:
        raise ValueError('Image / ROI shape mismatch.')
    common = roi & a & b
    union = roi & (a | b)
    result = {'roi_pixels': int(roi.sum()), 'both_valid_pixels': int(common.sum()),
              'reference_valid_pixels': int((roi & b).sum()), 'actual_valid_pixels': int((roi & a).sum()),
              'missing_reference_coverage_pixels': int((roi & b & ~a).sum()),
              'extra_coverage_pixels': int((roi & a & ~b).sum()),
              'silhouette_xor_pixels': int((roi & (a ^ b)).sum()),
              'boundary_xor_pixels': int((roi & (silhouette_boundary(a) ^ silhouette_boundary(b))).sum()),
              'silhouette_iou': float(common.sum()/union.sum()) if union.any() else None,
              'depth': stats(np.abs(actual['depth'][common]-reference['depth'][common])),
              'normal_degrees': stats(angles(actual['normals'][common], reference['normals'][common])),
              'error_denominator': 'both-valid pixels; missing/extra coverage separately explicit',
              'normal_reference': 'original terrain central gradient, not candidate mesh'}
    return result


def image_change(first, second, roi=None, depth_atol=1e-9, normal_degrees_atol=1e-5):
    if roi is None:
        roi = np.ones(first['mask'].shape, bool)
    result = image_metrics(first, second, roi)
    common = roi & first['mask'] & second['mask']
    depth = np.abs(first['depth'][common]-second['depth'][common])
    normal = angles(first['normals'][common], second['normals'][common])
    result.update(depth_changed_pixels=int((depth > depth_atol).sum()),
                  normal_changed_pixels=int((normal > normal_degrees_atol).sum()),
                  depth_atol=depth_atol, normal_degrees_atol=normal_degrees_atol,
                  normal_reference='second actual buffer, no independent quality claim')
    return result


def reference_check(coarse, fine, roi):
    change = image_change(coarse, fine, roi)
    ok = (change['roi_pixels'] > 0 and change['both_valid_pixels'] == change['roi_pixels']
          and change['silhouette_xor_pixels'] == 0
          and change['depth']['mean'] <= 1e-4 and change['normal_degrees']['mean'] <= .1)
    return {'status': 'PASS_REFINEMENT_CHECK' if ok else 'STOP_REFERENCE_UNRESOLVED',
            'scope': 'finite-resolution convergence check, not an analytic surface-error bound',
            'depth_mean_limit': 1e-4, 'normal_mean_degree_limit': .1, 'difference': change}


def gain_check(baseline_fine, treatment_fine, baseline_coarse, treatment_coarse):
    records = (baseline_fine, treatment_fine, baseline_coarse, treatment_coarse)
    if (any(r['missing_reference_coverage_pixels'] or r['extra_coverage_pixels'] for r in records)
            or len({r['both_valid_pixels'] for r in records}) != 1):
        return {key: {'status': 'UNRESOLVED_COVERAGE_MISMATCH',
                      'positive_gain_exceeds_twice_refinement_change': False,
                      'negative_gain_exceeds_twice_refinement_change': False}
                for key in ('depth', 'normal_degrees')}
    result = {}
    for key in ('depth', 'normal_degrees'):
        b, t = baseline_fine[key]['mean'], treatment_fine[key]['mean']
        bc, tc = baseline_coarse[key]['mean'], treatment_coarse[key]['mean']
        if any(value is None for value in (b, t, bc, tc)):
            result[key] = {'status': 'NO_COMMON_REFERENCE_PIXELS'}
            continue
        gain = b-t
        uncertainty = abs(b-bc)+abs(t-tc)
        result[key] = {'absolute_mean_error_reduction': gain,
            'relative_reduction': gain/b if b else None,
            'refinement_uncertainty_sum': uncertainty,
            'positive_gain_exceeds_twice_refinement_change': gain > 2*uncertainty,
            'negative_gain_exceeds_twice_refinement_change': -gain > 2*uncertainty,
            'scope': 'empirical refinement sensitivity, not statistical significance'}
    return result
