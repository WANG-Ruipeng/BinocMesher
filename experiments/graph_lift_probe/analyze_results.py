"""Post-measurement descriptive audit of saved JSON; no geometry is rerun."""
import argparse
import json
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(run):
    cases = [load(path) for path in sorted(run.glob('case_*.json'))]
    summary = load(run/'summary.json')
    assert len(cases) == summary['case_count'] == 24
    assert all(case['all_metric_invariants_passed'] for case in cases)
    pure_delta = {key: 0. for key in ('symmetric_mean_distance', 'maximum_directed_p95',
        'sampled_maximum_distance', 'normal_correspondence_symmetric_mean_degrees',
        'mesh_area_quadrature')}
    for case in cases:
        base = case['methods']['boundary_root0']['metrics']
        for name in ('pure_pl_k1', 'pure_pl_k2', 'pure_pl_k4'):
            for b, c in zip(base, case['methods'][name]['metrics']):
                for key in pure_delta:
                    pure_delta[key] = max(pure_delta[key], abs(b[key]-c[key]))
    affine = [case for case in cases if case['family'] == 'affine']
    flat_max = max(level['sampled_maximum_distance'] for case in affine
                   for method in case['methods'].values() for level in method['metrics'])
    flat_normal = max(level['normal_correspondence_symmetric_mean_degrees'] for case in affine
                      for method in case['methods'].values() for level in method['metrics'])
    budget_table = []
    for domain in ('whole_square', 'cropped_asymmetric_hexagon'):
        selected = [case for case in cases if case['domain'] == domain]
        for name in cases[0]['methods']:
            methods = [case['methods'][name] for case in selected]
            metric = [method['metrics'][-1]['symmetric_mean_distance'] for method in methods]
            budget_table.append({'domain': domain, 'method': name,
                'faces': methods[0]['checks']['faces'],
                'internal_vertices': methods[0]['checks']['internal_vertices'],
                'boundary_model_queries_each_run': methods[0]['budget']['boundary_model_point_evaluations'],
                'interior_model_queries': sorted(set(m['budget']['interior_model_point_evaluations'] for m in methods)),
                'candidate_queries_are_subset': True,
                'median_single_construction_microseconds': 1e6*median(m['budget']['method_construction_seconds'] for m in methods),
                'median_symmetric_distance_all_cases_including_flat': median(metric)})
    extra_pairs = (
        ('xyz_mean_fan', 'boundary_root0', 'xyz_fan_geometry_vs_original_diagonal'),
        ('rebuilt_lifted_vertex_mean', 'xyz_mean_fan', 'internal_model_information'),
    )
    descriptive = []
    for candidate, baseline, meaning in extra_pairs:
        for key in ('symmetric_mean_distance', 'maximum_directed_p95',
                    'sampled_maximum_distance', 'normal_correspondence_symmetric_mean_degrees'):
            rows = []
            for case in cases:
                a = case['methods'][candidate]['metrics'][-1][key]
                b = case['methods'][baseline]['metrics'][-1][key]
                threshold = 1e-8 if 'degrees' in key else 1e-9
                rows.append({'case': case['id'], 'reduction': b-a,
                             'relative_percent': None if b <= threshold else 100*(b-a)/b,
                             'sign': 'lower' if b-a > threshold else 'higher' if b-a < -threshold else 'tie'})
            relative = [row['relative_percent'] for row in rows if row['relative_percent'] is not None]
            descriptive.append({'meaning': meaning, 'candidate': candidate, 'baseline': baseline,
                'metric': key, 'signs_not_resolution_certified': True,
                'lower': sum(row['sign'] == 'lower' for row in rows),
                'higher': sum(row['sign'] == 'higher' for row in rows),
                'tie': sum(row['sign'] == 'tie' for row in rows),
                'nonzero_denominator_cases': len(relative),
                'median_relative_reduction_percent': median(relative) if relative else None,
                'median_absolute_reduction': median(row['reduction'] for row in rows),
                'worst_absolute_reduction': min(row['reduction'] for row in rows)})
    regressions = []
    for case in cases:
        for meaning, result in case['comparisons'].items():
            if result['status'] in ('RESOLVED_REGRESSION', 'RESOLUTION_UNRESOLVED'):
                regressions.append({'case': case['id'], 'comparison': meaning, **result})
    convergence = []
    for name in cases[0]['methods']:
        for key in ('symmetric_mean_distance', 'maximum_directed_p95',
                    'normal_correspondence_symmetric_mean_degrees'):
            refs, quads = [], []
            for case in cases:
                levels = case['methods'][name]['metrics']
                refs.append(abs(levels[1][key]-levels[0][key]))
                quads.append(abs(levels[2][key]-levels[1][key]))
            convergence.append({'method': name, 'metric': key,
                'median_reference_change': median(refs), 'max_reference_change': max(refs),
                'median_quadrature_change': median(quads), 'max_quadrature_change': max(quads)})
    return {'status': 'DESCRIPTIVE_AUDIT_ONLY_NO_NEW_MEASUREMENTS',
        'source_run': str(run), 'case_count': len(cases), 'distinct_source_models': 12,
        'flat_cases': len(affine), 'nonflat_cases': len(cases)-len(affine),
        'maximum_pure_subdivision_deltas_all_levels': pure_delta,
        'maximum_affine_sampled_distance': flat_max,
        'maximum_affine_mean_normal_degrees': flat_normal,
        'budget_table': budget_table, 'posthoc_descriptive_comparisons': descriptive,
        'regressions_and_unresolved': regressions, 'convergence': convergence,
        'interpretation_limits': [
            'Two domains reuse each source; do not treat 24 patches as independent source draws.',
            'Fan versus face-barycenter refinement changes interior edge layout as well as point position.',
            'Face-interior insertion preserves original diagonals, so it is not an unrestricted adaptive baseline.',
            'Square diagonal midpoint graph lift is algebraically the lifted center fan; no distinct GL strategy follows.',
            'All comparisons target the selected analytic model, not the original terrain or occupancy surface.',
            'Single Python timings are descriptive and exclude common mesh-validation guards; no production speed claim.',
        ]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({key: result[key] for key in (
        'status', 'case_count', 'distinct_source_models', 'maximum_pure_subdivision_deltas_all_levels',
        'maximum_affine_sampled_distance', 'maximum_affine_mean_normal_degrees',
        'posthoc_descriptive_comparisons')}, indent=2))
