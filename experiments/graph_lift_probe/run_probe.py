"""Supervised, append-only Stage 0/1 campaign. No scene/target/render imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WALL_SECONDS = 1800
ADDRESS_BYTES = 4*1024**3
OUTPUT_BYTES = 100*1024**2
REPO_BYTES = 400_000_000_000
LEVELS = ((16, 12), (32, 12), (32, 24))


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f'refusing to overwrite evidence: {path}')
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def bindings():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.md')}


def serial_method(method):
    return {k: v for k, v in method.items() if k not in ('vertices_array', 'faces_array', 'triangles')}


def measure(method, model, uv, uv_weights, analytic, reference):
    import numpy as np
    from metrics import TriangleSurface, pl_lift, weighted_stats
    xyz, normals, jacobian = pl_lift(uv, method['triangles'])
    model_xyz, model_normals, model_jacobian = analytic
    mesh = TriangleSurface(method['triangles'])
    outward, _ = reference.nearest(xyz)
    inward, _ = mesh.nearest(model_xyz)
    outstats = weighted_stats(outward, uv_weights*jacobian)
    instats = weighted_stats(inward, uv_weights*model_jacobian)
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(normals, model_normals), axis=1),
                                 np.sum(normals*model_normals, axis=1)))
    outnormal = weighted_stats(angle, uv_weights*jacobian)
    innormal = weighted_stats(angle, uv_weights*model_jacobian)
    return {
        'mesh_to_model_proxy': outstats,
        'analytic_model_to_mesh': instats,
        'symmetric_mean_distance': .5*(outstats['mean']+instats['mean']),
        'maximum_directed_p95': max(outstats['p95'], instats['p95']),
        'sampled_maximum_distance': max(outstats['sampled_max'], instats['sampled_max']),
        'normal_correspondence_symmetric_mean_degrees': .5*(outnormal['mean']+innormal['mean']),
        'normal_correspondence_max_directed_p95_degrees': max(outnormal['p95'], innormal['p95']),
        'mesh_area_quadrature': float(np.sum(uv_weights*jacobian)),
        'analytic_model_area_quadrature': float(np.sum(uv_weights*model_jacobian)),
        'normal_definition': 'same parameter location; not nearest-point correspondence',
    }


def metric_guard(records, family, scale):
    import numpy as np
    baseline = records['boundary_root0']
    for name in ('pure_pl_k1', 'pure_pl_k2', 'pure_pl_k4'):
        candidate = records[name]
        for key in ('symmetric_mean_distance', 'maximum_directed_p95',
                    'sampled_maximum_distance', 'mesh_area_quadrature'):
            if abs(baseline[key]-candidate[key]) > 1e-10*max(1., scale):
                raise RuntimeError(f'pure-subdivision metric invariant: {name}/{key}')
        key = 'normal_correspondence_symmetric_mean_degrees'
        if abs(baseline[key]-candidate[key]) > 1e-8:
            raise RuntimeError(f'pure-subdivision normal invariant: {name}')
    if family == 'affine':
        if max(r['sampled_maximum_distance'] for r in records.values()) > 1e-10*scale:
            raise RuntimeError('planar distance control failed')
        if max(r['normal_correspondence_symmetric_mean_degrees'] for r in records.values()) > 1e-8:
            raise RuntimeError('planar normal control failed')
    for record in records.values():
        for key, value in record.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise RuntimeError(f'nonfinite metric {key}')


def classify(method, baseline, scale):
    """Resolution-derived screen, not a rigorous error bound/confidence interval."""
    key = 'symmetric_mean_distance'
    m, b = method['metrics'], baseline['metrics']
    mf, bf = m[-1][key], b[-1][key]
    noise = 1e-10*max(1., scale)
    uncertainty = 2*(abs(m[1][key]-m[0][key])+abs(m[2][key]-m[1][key])+
                     abs(b[1][key]-b[0][key])+abs(b[2][key]-b[1][key]))+noise
    improvement = bf-mf
    if abs(improvement) <= noise:
        status = 'NUMERIC_TIE'
    elif improvement > uncertainty:
        status = 'RESOLVED_IMPROVEMENT'
    elif improvement < -uncertainty:
        status = 'RESOLVED_REGRESSION'
    else:
        status = 'RESOLUTION_UNRESOLVED'
    return {'status': status, 'baseline_error': bf, 'candidate_error': mf,
            'absolute_reduction': improvement,
            'relative_reduction_percent': None if bf <= 1e-9*max(1., scale) else 100*improvement/bf,
            'empirical_resolution_tolerance': uncertainty,
            'tolerance_is_certified_bound': False}


PAIRS = (
    ('rebuilt_lifted_vertex_mean', 'xyz_mean_fan', 'internal_information_same_faces'),
    ('rebuilt_lifted_area_centroid', 'rebuilt_lifted_vertex_mean', 'center_rule_same_faces_queries'),
    ('rebuilt_lifted_area_centroid', 'uniform_model_k1', 'placement_same_faces_queries'),
    ('greedy_model_k1', 'rebuilt_lifted_area_centroid', 'greedy_same_faces_more_queries'),
    ('greedy_model_k4', 'uniform_model_k4', 'refinement_same_faces_more_queries'),
    ('rebuilt_lifted_area_centroid', 'boundary_root0', 'density_information_vs_diagonal0'),
    ('rebuilt_lifted_area_centroid', 'boundary_root1', 'density_information_vs_diagonal1'),
)


def summarize(cases):
    import numpy as np
    summaries = []
    for candidate, baseline, meaning in PAIRS:
        for group in ('all', 'whole_square', 'cropped_asymmetric_hexagon'):
            selected = [case for case in cases if group == 'all' or case['domain'] == group]
            outcomes = [case['comparisons'][meaning] for case in selected]
            counts = {s: sum(o['status'] == s for o in outcomes) for s in (
                'RESOLVED_IMPROVEMENT', 'RESOLVED_REGRESSION', 'NUMERIC_TIE', 'RESOLUTION_UNRESOLVED')}
            relative = [o['relative_reduction_percent'] for o in outcomes
                        if o['relative_reduction_percent'] is not None]
            summaries.append({'candidate': candidate, 'baseline': baseline, 'meaning': meaning,
                              'group': group, 'cases': len(selected), **counts,
                              'relative_nonzero_cases': len(relative),
                              'median_relative_reduction_percent_nonzero': float(np.median(relative)) if relative else None,
                              'median_absolute_reduction': float(np.median([o['absolute_reduction'] for o in outcomes])),
                              'worst_absolute_reduction': float(min(o['absolute_reduction'] for o in outcomes)),
                              'median_baseline_error': float(np.median([o['baseline_error'] for o in outcomes])),
                              'median_candidate_error': float(np.median([o['candidate_error'] for o in outcomes]))})
    return summaries


def boundary_chord_gap(model, polygon):
    import numpy as np
    gaps = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        alpha = np.linspace(0, 1, 33)
        q = a[None, :]*(1-alpha[:, None])+b[None, :]*alpha[:, None]
        xyz = model.evaluate(q)[0]
        chord = xyz[0][None, :]*(1-alpha[:, None])+xyz[-1][None, :]*alpha[:, None]
        gaps.extend(np.linalg.norm(xyz-chord, axis=1).tolist())
    return {'maximum_sampled_same_parameter_chord_gap': float(max(gaps)),
            'evaluation_model_point_queries': len(gaps),
            'meaning': 'common boundary interpolation discrepancy, not a certified distance floor'}


def worker(out):
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_BYTES, ADDRESS_BYTES))
    import numpy as np
    import scipy
    from model import make_cases, all_methods, quadrature, reference_triangles
    from metrics import TriangleSurface
    initial = bindings()
    cases = make_cases()
    dump(out/'inputs.json', {
        'created_utc': utc(), 'source_bindings': initial, 'numpy': np.__version__,
        'scipy': scipy.__version__, 'python': sys.version, 'levels_ref_quadrature': LEVELS,
        'case_count': len(cases),
        'historical_gl_recovery_status': 'UNRECOVERED_WITHIN_BOUNDED_SEARCH',
        'historical_q_definition': 'UNRESOLVED',
        'historical_120_case_metrics': 'UNVERIFIED_REPORT_ONLY',
        'experiment_identity': 'NEW_RECONSTRUCTION_OF_SIMPLE_MODEL_SAMPLING',
        'input_cases': [{k: v for k, v in c.items() if k not in ('model', 'polygon')} |
                        {'polygon_uv': c['polygon'].tolist(), 'source': c['model'].receipt()}
                        for c in cases]})
    completed = []
    started = time.monotonic()
    for index, case in enumerate(cases):
        if time.monotonic()-started > WALL_SECONDS:
            raise RuntimeError('worker wall limit')
        if tree_bytes(HERE) > OUTPUT_BYTES:
            raise RuntimeError('experiment directory storage limit')
        begin = time.monotonic()
        model, polygon = case['model'], case['polygon']
        original_source = model.receipt()
        original_polygon = polygon.copy()
        methods = all_methods(model, polygon)
        records = {m['name']: serial_method(m) | {'metrics': []} for m in methods}
        scale = float(np.linalg.norm(np.ptp(methods[0]['vertices_array'][:len(polygon)], axis=0)))
        eval_count = 0
        reference_counts = {}
        # Each metric level is independent; none of these objects enters construction.
        for ref_n, quad_n in LEVELS:
            uv, weights = quadrature(polygon, quad_n)
            analytic = model.evaluate(uv)
            ref_tri = reference_triangles(model, polygon, ref_n)
            reference = TriangleSurface(ref_tri)
            eval_count += len(uv)+3*len(ref_tri)
            reference_counts[f'{ref_n}_{quad_n}'] = {
                'reference_triangles': len(ref_tri), 'quadrature_points_per_direction': len(uv),
                'model_point_evaluations': len(uv)+3*len(ref_tri)}
            level = {}
            for method in methods:
                result = measure(method, model, uv, weights, analytic, reference)
                result['reference_resolution'] = ref_n
                result['quadrature_resolution'] = quad_n
                records[method['name']]['metrics'].append(result)
                level[method['name']] = result
            metric_guard(level, case['family'], scale)
        comparisons = {meaning: classify(records[candidate], records[baseline], scale)
                       for candidate, baseline, meaning in PAIRS}
        if (original_source != model.receipt() or not np.array_equal(original_polygon, polygon)
                or bindings() != initial):
            raise RuntimeError('source or implementation mutated during campaign')
        chord = boundary_chord_gap(model, polygon)
        report = {k: v for k, v in case.items() if k not in ('model', 'polygon')} | {
            'status': 'MEASURED', 'model_only': True, 'independent_target_tested': False,
            'coordinate_scale_bbox_diagonal': scale, 'methods': records,
            'reference_evaluation': reference_counts,
            'reference_evaluation_model_queries': eval_count,
            'total_evaluation_model_queries': eval_count+chord['evaluation_model_point_queries'],
            'boundary_chord_diagnostic': chord,
            'comparisons': comparisons, 'all_metric_invariants_passed': True,
            'elapsed_seconds': time.monotonic()-begin}
        dump(out/f'case_{index:02d}.json', report)
        completed.append(report)
        print(json.dumps({'case': index+1, 'of': len(cases), 'id': case['id'],
                          'seconds': report['elapsed_seconds'], 'status': 'MEASURED'}), flush=True)
    result = {'status': 'STAGE_1_COMPLETE_STOP_BEFORE_STAGE_2', 'created_utc': utc(),
              'case_count': len(completed), 'method_count': len(completed[0]['methods']),
              'metric_levels': len(LEVELS), 'all_metric_invariants_passed': True,
              'summaries': summarize(completed), 'elapsed_seconds': time.monotonic()-started,
              'worker_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'claims_not_tested': ['historical_120_case_reproduction', 'original_target_accuracy',
                                    'temporal_coherence', 'render_quality', 'production_safety',
                                    'published_baseline_superiority', 'novel_algorithm'],
              'source_bindings': initial}
    dump(out/'summary.json', result)
    print(json.dumps({'status': result['status'], 'seconds': result['elapsed_seconds']}), flush=True)


def repo_size():
    text = subprocess.check_output(['du', '-sb', str(REPO)], text=True)
    return int(text.split()[0])


def supervise():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out = HERE/'artifacts'/f'run_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    initial_size = repo_size()
    initial_bindings = bindings()
    dump(out/'launch.json', {'created_utc': utc(), 'stage': '0_1_only',
         'wall_limit_seconds': WALL_SECONDS, 'address_space_limit_bytes': ADDRESS_BYTES,
         'experiment_directory_limit_bytes': OUTPUT_BYTES, 'repo_limit_bytes': REPO_BYTES,
         'repo_bytes_before': initial_size, 'source_bindings': initial_bindings,
         'command': [sys.executable, str(Path(__file__).resolve()), '--worker', str(out)]})
    if initial_size >= REPO_BYTES or tree_bytes(HERE) >= OUTPUT_BYTES:
        dump(out/'STOP.json', {'reason': 'preflight storage limit', 'created_utc': utc()})
        return 2
    env = dict(os.environ)
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    start = time.monotonic()
    stopped = None
    print(f'RUN_DIRECTORY={out}', flush=True)
    with (out/'worker.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(out)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while process.poll() is None:
            if time.monotonic()-start > WALL_SECONDS:
                stopped = 'wall limit'
            elif tree_bytes(HERE) > OUTPUT_BYTES:
                stopped = 'experiment directory storage limit'
            if stopped:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    final_size = repo_size()
    if bindings() != initial_bindings:
        stopped = stopped or 'source binding changed'
    if final_size >= REPO_BYTES:
        stopped = stopped or 'repository storage limit'
    okay = process.returncode == 0 and (out/'summary.json').exists() and stopped is None
    receipt = {'created_utc': utc(), 'status': 'COMPLETE' if okay else 'STOP',
               'worker_exit_code': process.returncode, 'reason': stopped,
               'elapsed_seconds': time.monotonic()-start, 'repo_bytes_after': final_size,
               'experiment_directory_bytes': tree_bytes(HERE), 'automatic_restart': False}
    dump(out/'supervisor.json', receipt)
    if not okay and not (out/'STOP.json').exists():
        dump(out/'STOP.json', receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if okay else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except Exception as exc:
            traceback.print_exc()
            dump(args.worker/'STOP.json', {'created_utc': utc(), 'status': 'STOP',
                                          'reason': str(exc), 'type': type(exc).__name__,
                                          'completed_cases': len(list(args.worker.glob('case_*.json')))})
            sys.exit(2)
    else:
        sys.exit(supervise())
