#!/usr/bin/env python3
"""Small conditional relative-plane diagnostic; never a production admission.

The pure function supports arbitrary four-vertex graph-fan inputs. The CLI is
deliberately fixed to the frozen E2 seven-candidate, seven-unit experiment. It
reads original inputs and writes only a fresh, bounded diagnostic JSON file.
"""

from collections import Counter, defaultdict
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from window_contact_v2 import _cross_component
from window_exterior import _ids
from window_geometry import (
    DEFAULT_MAX_RATIONAL_BITS, _BudgetExceeded, _boundary, _check_budget,
    _cross_xy, _fraction, _json_fraction, _minimum_quadratic, _subtract,
    _trajectory, _vector,
)


EVENT_KEY = 'event-02-88ade47aa4fa'
BOUNDARY_CYCLE = ('46:10|376:8', '159:8|311:0', '159:8|391:8', '46:10|391:8')
CANDIDATES = (
    ('159:8|311:0', '160:8|312:0', '159:8|317:0'),
    ('159:8|317:0', '160:8|312:0', '160:8|318:0'),
    ('159:8|317:0', '160:8|318:0', '162:8|320:0'),
    ('159:8|317:0', '161:8|319:0', '161:8|392:8'),
    ('159:8|317:0', '161:8|392:8', '159:8|391:8'),
    ('159:8|317:0', '162:8|320:0', '161:8|319:0'),
    ('159:8|311:0', '159:8|317:0', '159:8|391:8'),
)
EPSILONS = (Fraction(1, 1024), Fraction(1, 128))
MAX_OUTPUT_BYTES = 512 * 1024


def epsilon_conditions(margin, bmax, tilt, epsilons=EPSILONS):
    """Uniform per-coordinate perturbation bound, NOT an actual error bound.

Each vertex (including the plane's two endpoints) may move by epsilon in each
coordinate. A shared vertex must use one common perturbed position everywhere.
For L=cross_XY(e,d)+tilt*cross_IJ(e,d), the error is bounded by
2*epsilon*B + 8*(1+abs(tilt))*epsilon**2.
"""
    rows = []
    for epsilon in epsilons:
        if epsilon < 0:
            raise ValueError('A coordinate error bound cannot be negative.')
        error = 2*epsilon*bmax+8*(1+abs(tilt))*epsilon*epsilon
        rows.append({
            'hypothetical_uniform_coordinate_error': _json_fraction(epsilon),
            'functional_error_upper_bound': _json_fraction(error),
            'remaining_signed_margin': _json_fraction(margin-error),
            'sufficient_if_actual_bound_and_common_identity_hold': error < margin,
            'actual_bound_proven': False,
        })
    return rows


def audit_relative_plane(boundary_start, boundary_end, boundary_vids,
                         center_start, center_end,
                         triangle_start, triangle_end, triangle_vids, *,
                         edge_index=0, tilt=Fraction(1), projection='yz',
                         max_rational_bits=DEFAULT_MAX_RATIONAL_BITS):
    """Certify relative half-space signs and a conditional perturbation bound.

The plane uses the declared boundary edge endpoints a,b: e=b-a, d=x-a,
L=orientation*cross_XY(e,d)+tilt*cross_XZ_or_YZ(e,d). At those endpoints L is
structurally zero even after perturbation, provided common actual coordinates
are used. Every other fan vertex, including the center, must have L>0; every
nonshared retained vertex must have L<0 on the entire affine branch.

PASS proves only these exact signs / allowed geometric contact. It does not
prove local graph embedding, manifold links, actual error bounds, emitted
identity/owner correspondence, time-selector correctness, or runtime admission.
"""
    report = {
        'schema': 'c1-lite-conditional-relative-plane-v1', 'status': 'UNKNOWN',
        'scope': 'ONE_IDEAL_AFFINE_FAN_VS_RETAINED_TRIANGLE_RELATIVE_PLANE_ONLY',
        'coordinate_model': 'exact rational affine trajectories',
        'runtime_admitted': False, 'constraints': [], 'structural_zero_checks': [],
        'actual_coordinate_error_bound_proven': False,
        'actual_source_vid_mapping_proven': False,
        'actual_double_time_selector_proven': False,
        'required_for_perturbed_geometry': [
            'Every actual vertex corresponds to the same ideal source/emitted vertex and has the stated coordinate error bound for every query time.',
            'Each shared SourceVID reuses exactly one common actual coordinate in patch and retained geometry.',
            'The two actual boundary endpoints define the perturbed plane symbolically; its zero is not independently approximated per face.',
        ],
        'not_certified': ['Local fan embedding or retained triangle regularity.',
                          'Owners, actual arrays, extra_smooth, topology, other triangle pairs, or endpoint realization.',
                          'Behavior near source breakpoints when physical time is converted to double.'],
        'witness': None,
    }
    try:
        if (isinstance(max_rational_bits, bool) or not isinstance(max_rational_bits, int)
                or max_rational_bits < 1):
            raise ValueError('max_rational_bits must be a positive integer.')
        if (isinstance(edge_index, bool) or not isinstance(edge_index, int)
                or not 0 <= edge_index < 4 or projection not in ('xz', 'yz')):
            raise ValueError('Expected a boundary edge index and xz or yz projection.')
        weight = _fraction(tilt, max_rational_bits)
        b0, b1 = _boundary(boundary_start, max_rational_bits), _boundary(boundary_end, max_rational_bits)
        c0, c1 = _vector(center_start, max_rational_bits), _vector(center_end, max_rational_bits)
        if len(triangle_start) != 3 or len(triangle_end) != 3:
            raise ValueError('Expected three retained vertices at each endpoint.')
        p0 = tuple(_vector(x, max_rational_bits) for x in triangle_start)
        p1 = tuple(_vector(x, max_rational_bits) for x in triangle_end)
        bv, tv = _ids(boundary_vids, 4, 'boundary_vids'), _ids(triangle_vids, 3, 'triangle_vids')
        following = (edge_index+1) % 4
        edge_ids = {bv[edge_index], bv[following]}
        shared = set(bv) & set(tv)
        if not shared <= edge_ids:
            report['reason'] = 'This chosen relative plane does not support all declared shared vertices.'
            return report
        for index, vid in enumerate(tv):
            if vid in shared:
                j = bv.index(vid)
                if p0[index] != b0[j] or p1[index] != b1[j]:
                    raise ValueError('Shared source identity has contradictory exact trajectories.')
        bt = [_trajectory(a, b) for a, b in zip(b0, b1)]
        pt = [_trajectory(a, b) for a, b in zip(p0, p1)]
        center = _trajectory(c0, c1)
        initial = _cross_xy(_subtract(bt[1], bt[0]), _subtract(bt[2], bt[0]))[0]
        if initial == 0:
            report['reason'] = 'No nonzero initial source-cycle XY orientation.'
            return report
        orientation = 1 if initial > 0 else -1
        axes = (0, 2) if projection == 'xz' else (1, 2)
        origin, edge = bt[edge_index], _subtract(bt[following], bt[edge_index])
        report.update(edge_index=edge_index, edge_source_vids=[bv[edge_index], bv[following]],
                      tilt=_json_fraction(weight), projection=projection, orientation_xy=orientation,
                      declared_shared_vids=sorted(shared))

        def plane(vertex):
            difference = _subtract(vertex, origin)
            h, g = _cross_xy(edge, difference), _cross_component(edge, difference, axes)
            return tuple(_check_budget(orientation*x+weight*y, max_rational_bits) for x, y in zip(h, g))

        for vid, vertex in [(bv[edge_index], bt[edge_index]), (bv[following], bt[following])]:
            coefficients = plane(vertex)
            if any(coefficients):
                raise ValueError('Boundary endpoint is not structurally on its own relative plane.')
            report['structural_zero_checks'].append({'source_vid': vid, 'role': 'plane_defining_boundary_endpoint',
                'power_coefficients_ascending': [_json_fraction(x) for x in coefficients]})
        tasks = [('patch_boundary', bv[i], bt[i], b0[i], b1[i], 1)
                 for i in range(4) if i not in (edge_index, following)]
        tasks.append(('patch_center', None, center, c0, c1, 1))
        for i, vid in enumerate(tv):
            if vid in shared:
                if any(plane(pt[i])):
                    raise ValueError('Shared retained endpoint is not on the same exact relative plane.')
            else:
                tasks.append(('retained_nonshared', vid, pt[i], p0[i], p1[i], -1))
        bmax, margins = Fraction(0), []
        for role, vid, vertex, start, end, sign in tasks:
            coefficients = plane(vertex)
            signed = tuple(sign*x for x in coefficients)
            minimum_at, minimum, _ = _minimum_quadratic(signed, max_rational_bits)
            margins.append(minimum)
            report['constraints'].append({'role': role, 'source_vid': vid, 'required_sign': sign,
                'plane_power_coefficients_ascending': [_json_fraction(x) for x in coefficients],
                'minimum_signed_value': _json_fraction(minimum),
                'minimum_at_normalized_time': _json_fraction(minimum_at), 'strict': minimum > 0})
            if minimum <= 0 and report['witness'] is None:
                report['witness'] = {'kind': 'relative_plane_sign_not_strict', 'source_vid': vid,
                    'role': role, 'normalized_time': _json_fraction(minimum_at),
                    'signed_value': _json_fraction(minimum), 'meaning': 'No sufficient plane proof, not a collision diagnosis.'}
            for positions, point in ((b0, start), (b1, end)):
                e = tuple(x-y for x, y in zip(positions[following], positions[edge_index]))
                d = tuple(x-y for x, y in zip(point, positions[edge_index]))
                i, j = axes
                bound = (abs(e[0])+abs(d[1])+abs(e[1])+abs(d[0])
                         +abs(weight)*(abs(e[i])+abs(d[j])+abs(e[j])+abs(d[i])))
                bmax = max(bmax, _check_budget(bound, max_rational_bits))
        margin = min(margins)
        report['minimum_signed_margin'] = _json_fraction(margin)
        report['Bmax'] = _json_fraction(bmax)
        report['Bmax_proof'] = 'Sum of absolute affine functions is convex; its maximum on each closed branch occurs at an endpoint.'
        report['perturbation_error_formula'] = '2*epsilon*Bmax + 8*(1+abs(tilt))*epsilon^2'
        report['epsilon_conditions'] = epsilon_conditions(margin, bmax, weight)
        report['allowed_contact'] = 'shared_edge' if len(shared) == 2 else 'shared_vertex' if shared else 'empty'
        report['status'] = 'PASS' if margin > 0 else 'UNKNOWN'
        report['reason'] = ('All nonzero features have exact whole-branch strict relative-plane signs; perturbation conclusions remain conditional.'
                            if margin > 0 else 'The chosen relative plane is not certified on this branch.')
    except _BudgetExceeded as error:
        report.update(status='UNKNOWN', reason=str(error), witness=None)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError) as error:
        report.update(status='REJECT', reason='Invalid diagnostic input: '+str(error),
                      witness={'kind': 'invalid_input', 'diagnostic': str(error)})
    return report


def main():
    # Heavy cache/numpy imports are unnecessary for the pure synthetic tests.
    from window_source import read_inventory, fr, file_signatures, input_files
    from runtime_retained import build_retained_unit
    from run_window_audit import units_for, center_at

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=90.)
    args = parser.parse_args()
    source_path, cache, output = args.source_report.resolve(), args.cache_root.resolve(), args.output.resolve()
    if (output.exists() or output.suffix != '.json' or cache == output or cache in output.parents
            or source_path.parent == output or source_path.parent in output.parents):
        raise ValueError('Output must be a fresh JSON outside original source/cache inputs.')
    if not 0 < args.max_seconds <= 600:
        raise ValueError('Expected an explicit bounded wall time <=600 seconds.')
    started = time.monotonic()
    original = source_path.read_bytes()
    source = json.loads(original)
    if source.get('status') != 'PASS_SOURCE_BRANCH_CONTRACT' or tuple(source['boundary_cycle']) != BOUNDARY_CYCLE:
        raise ValueError('Expected the frozen E2 source branch report.')
    if [fr(source['levels'][key]) for key in ('lower', 'root', 'upper')] != [Fraction(102, 5), Fraction(104, 5), Fraction(106, 5)]:
        raise ValueError('Unexpected E2 fixed window.')
    specs = list(units_for(source))
    if Counter(spec['kind'] for spec in specs) != Counter({'affine_branch': 3, 'actual_singleton': 4}):
        raise ValueError('Expected exactly three branches and four singleton states.')
    inventory = read_inventory(cache)
    if inventory.digest != source['cache_input_sha256']:
        raise ValueError('Frozen source and original cache hashes differ.')
    elements = {owner[0] for spec in specs for owner in spec['owners']}
    if len(elements) != 1:
        raise ValueError('Ambiguous event element.')
    element = next(iter(elements))
    result = {'schema': 'c1-lite-e2-relative-plane-feasibility-v1', 'event_key': EVENT_KEY,
        'status': 'INCOMPLETE', 'runtime_admitted': False, 'production_modified': False,
        'coordinate_model': source['coordinate_model'],
        'scope': 'Fixed seven candidate types only, not all retained triangles or a runtime certificate.',
        'source_report_path': str(source_path), 'source_report_sha256': hashlib.sha256(original).hexdigest(),
        'cache_root': str(cache), 'cache_input_sha256': inventory.digest,
        'raw_retained_scope': 'Original-time filtered canonical SourceVID geometric quotient, not actual C++ arrays.',
        'fixed_plane': {'edge_index': 1, 'projection': 'yz', 'tilt': _json_fraction(Fraction(1))},
        'units': [], 'actual_coordinate_error_bound_proven': False,
        'actual_source_vid_mapping_proven': False, 'actual_double_time_selector_proven': False}
    certificates = []
    for spec in specs:
        if time.monotonic()-started > args.max_seconds:
            raise TimeoutError('Bounded diagnostic time exhausted; no report published.')
        lo, hi = fr(spec['t0']), fr(spec['t1'])
        retained = build_retained_unit(inventory, lo, hi, spec['owners'])
        lookup = defaultdict(list)
        for triangle in retained.triangles:
            lookup[triangle['element'], frozenset(triangle['source_vertices'])].append(triangle)
        row = {key: spec[key] for key in ('kind', 'index', 't0', 't1')}
        row['candidates'] = []
        for index, target in enumerate(CANDIDATES):
            matches = lookup[element, frozenset(target)]
            if len(matches) != 1:
                raise ValueError('Candidate missing or ambiguous: '+repr(target))
            triangle = matches[0]
            certificate = audit_relative_plane(
                spec['boundary_start'], spec['boundary_end'], BOUNDARY_CYCLE,
                center_at(source, lo), center_at(source, hi),
                triangle['positions_t0'], triangle['positions_t1'], triangle['source_vertices'],
                edge_index=1, tilt=Fraction(1), projection='yz')
            certificates.append(certificate)
            row['candidates'].append({'candidate_index': index,
                'source_vertices': triangle['source_vertices'], 'owners': triangle['owners'],
                'certificate': certificate})
        result['units'].append(row)
    if source_path.read_bytes() != original or file_signatures(input_files(cache)) != inventory.signatures:
        raise ValueError('Frozen inputs changed; no report published.')
    passed = all(c['status'] == 'PASS' for c in certificates)
    result.update(status='COMPLETE_CONDITIONAL_GEOMETRY_FEASIBILITY' if passed else 'COMPLETE_WITH_UNCERTIFIED_PLANES',
                  unit_count=len(specs), candidate_type_count=len(CANDIDATES),
                  unit_candidate_count=len(certificates), certificate_status_counts=dict(Counter(c['status'] for c in certificates)),
                  frozen_inputs_unchanged=True)
    if passed:
        margin = min(fr(c['minimum_signed_margin']) for c in certificates)
        bmax = max(fr(c['Bmax']) for c in certificates)
        result['global_minimum_signed_margin'] = _json_fraction(margin)
        result['global_Bmax'] = _json_fraction(bmax)
        result['global_epsilon_conditions'] = epsilon_conditions(margin, bmax, Fraction(1))
    scripts = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ('window_geometry.py', 'window_exterior.py', 'window_contact_v2.py', 'window_source.py',
         'runtime_retained.py', 'run_window_audit.py')],
        Path(__file__).parent.parent/'source_splice'/'processed_mesh.py']
    result['script_sha256'] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
    result['python_version'] = sys.version
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError('Diagnostic output exceeds the 512 KiB bound.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(payload)
    print(json.dumps({'output': str(output), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                      'status': result['status'], 'counts': result['certificate_status_counts'], 'runtime_admitted': False}))


if __name__ == '__main__':
    main()
