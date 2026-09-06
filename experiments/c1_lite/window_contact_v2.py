#!/usr/bin/env python3
"""Sufficient exact source-incidence contact certificates for convex XY fans.

The caller must certify the same affine branch as a strictly convex XY graph
fan. This module does not certify runtime rounding, source ownership, manifold
links, or the retained mesh. UNKNOWN is not an intersection diagnosis.
"""

from fractions import Fraction
from math import gcd

from window_exterior import (
    _ids, _maximum_polynomial, _polynomial_record,
    certify_exterior_separation,
)
from window_geometry import (
    DEFAULT_MAX_RATIONAL_BITS, _BudgetExceeded, _boundary, _check_budget,
    _cross_xy, _evaluate, _json_fraction, _minimum_quadratic, _multiply,
    _subtract, _trajectory, _vector,
)


def _cross_component(first, second, axes):
    i, j = axes
    a, b = _multiply(first[i], second[j]), _multiply(first[j], second[i])
    return tuple(x-y for x, y in zip(a, b))


def _strict_record(power, limit):
    time, maximum = _maximum_polynomial(power, limit)
    return _polynomial_record(power, time, maximum)


def _negative(record):
    return record['strictly_negative']


def _normal_certificate(triangle, limit):
    """Conservative regularity test; exact rational zeros may supply witnesses.

One fixed normal component bounded away from zero suffices. If the normal
rotates so that no one component suffices, return UNKNOWN, not degeneracy.
"""
    first = _subtract(triangle[1], triangle[0])
    second = _subtract(triangle[2], triangle[0])
    powers = [_cross_component(first, second, axes)
              for axes in ((1, 2), (2, 0), (0, 1))]
    candidates = {Fraction(0), Fraction(1)}
    checks = []
    for axis, power in enumerate(powers):
        for value in power:
            _check_budget(value, limit)
        time_min, minimum, extrema = _minimum_quadratic(power, limit)
        time_max, maximum = _maximum_polynomial(power, limit)
        candidates.update(time for time, _ in extrema)
        if not power[2] and power[1]:
            zero = _check_budget(-power[0]/power[1], limit)
            if 0 <= zero <= 1:
                candidates.add(zero)
        checks.append({
            'axis': 'xyz'[axis],
            'power_coefficients_ascending': [_json_fraction(x) for x in power],
            'exact_minimum': _json_fraction(minimum),
            'exact_maximum': _json_fraction(maximum),
            'minimum_at_normalized_time': _json_fraction(time_min),
            'maximum_at_normalized_time': _json_fraction(time_max),
        })
        if minimum > 0 or maximum < 0:
            return {'status': 'PASS', 'kind': 'STRICT_NORMAL_COMPONENT',
                    'check': checks[-1]}
    for time in sorted(candidates):
        values = [_check_budget(_evaluate(power, time), limit) for power in powers]
        if all(value == 0 for value in values):
            return {'status': 'REJECT', 'kind': 'retained_triangle_degenerate',
                    'normalized_time': _json_fraction(time),
                    'normal_components': [_json_fraction(value) for value in values]}
    return {'status': 'UNKNOWN', 'kind': 'RETAINED_REGULARITY_NOT_CERTIFIED',
            'checks': checks,
            'reason': 'No single normal component is nonzero throughout the branch; this does not prove degeneracy.'}


def _weights(limit):
    # Integer pairs enumerate positive rational ratios without duplicate scales.
    # Small/simple ratios first; ordering is independent of event IDs.
    return sorted(((a, b) for a in range(1, limit+1)
                   for b in range(1, limit+1) if gcd(a, b) == 1),
                  key=lambda pair: (max(pair), sum(pair), pair))


def _pass(report, kind, shared, reason, **evidence):
    report.update(status='PASS', reason=reason,
                  patch_triangle_interior_disjoint_certified=True)
    report['separation_certificate'] = {
        'kind': kind,
        'allowed_contact': ('shared_edge' if len(shared) == 2 else
                            'shared_vertex' if shared else 'empty'),
        'allowed_contact_source_vids': sorted(shared),
        'retained_regularity_certificate': report.get('retained_regularity_certificate'),
        **evidence,
    }
    return report


def certify_contact(boundary_start, boundary_end, boundary_vids,
                    triangle_start, triangle_end, triangle_vids, *,
                    center_start=None, center_end=None, orientation_xy=None,
                    max_rational_bits=DEFAULT_MAX_RATIONAL_BITS, max_weight=32):
    """Return a JSON-compatible conditional PASS / REJECT / UNKNOWN report.

First reuse the old strict exterior/axis certificate. For declared contacts,
also require a sufficient exact nondegeneracy certificate for the retained
triangle. Repeated IDs or contradictory shared trajectories are rejected.

New sufficient tests are:
* At a shared vertex, a positive rational combination of its two incident
  inward half-planes is strictly negative at both other retained vertices.
* If all retained vertices project identically onto one supporting line, the
  fan can meet them only on that boundary edge. Two shared endpoints certify
  exactly the shared edge. One shared endpoint additionally requires strict
  separation from the spatial edge line, or strict projection beyond that
  endpoint, at both other vertices.

Every polynomial inequality is checked on the entire closed affine interval,
including interior extrema. An identically zero supporting-line polynomial
is used only with explicit allowed-incidence proofs, never as a blanket <=0
replacement for strict separation. Search/bit-budget failures are UNKNOWN.
"""
    report = certify_exterior_separation(
        boundary_start, boundary_end, boundary_vids,
        triangle_start, triangle_end, triangle_vids,
        center_start=center_start, center_end=center_end,
        orientation_xy=orientation_xy, max_rational_bits=max_rational_bits)
    first_status = report['status']
    first_certificate = report['separation_certificate']
    report['schema'] = 'c1-lite-affine-source-contact-v2'
    report['first_tier_status'] = first_status
    report['limits']['max_weight'] = max_weight
    report['positive_combination_candidates_tested'] = 0
    report['retained_regularity_certificate'] = None
    report['not_certified'].append(
        'A whole-window runtime admission, binary32/smooth realization, or any manifold-link theorem.')
    try:
        if (isinstance(max_weight, bool) or not isinstance(max_weight, int)
                or not 1 <= max_weight <= 32):
            raise ValueError('max_weight must be an integer in [1, 32].')
        if first_status == 'REJECT':
            return report
        if report.get('declared_shared_vids') == []:
            return report
        b0, b1 = _boundary(boundary_start, max_rational_bits), _boundary(boundary_end, max_rational_bits)
        t0 = tuple(_vector(value, max_rational_bits) for value in triangle_start)
        t1 = tuple(_vector(value, max_rational_bits) for value in triangle_end)
        bv, tv = _ids(boundary_vids, 4, 'boundary_vids'), _ids(triangle_vids, 3, 'triangle_vids')
        shared = set(bv) & set(tv)
        # No new rule admits nonshared coincident vertices. A disjoint old-tier
        # PASS may include degenerate retained geometry: it proves only empty
        # intersection, not regularity of unrelated baseline triangles.
        if not shared:
            return report
        if first_status == 'PASS' and report['separation_certificate']['allowed_contact'] == 'empty':
            return report
        report.update(status='UNKNOWN', patch_triangle_interior_disjoint_certified=False,
                      separation_certificate=None)
        boundary = [_trajectory(a, b) for a, b in zip(b0, b1)]
        triangle = [_trajectory(a, b) for a, b in zip(t0, t1)]
        regularity = _normal_certificate(triangle, max_rational_bits)
        report['retained_regularity_certificate'] = regularity
        if regularity['status'] != 'PASS':
            report['status'] = regularity['status']
            report['reason'] = ('A declared-contact retained triangle is exactly degenerate.'
                                if regularity['status'] == 'REJECT' else regularity['reason'])
            if regularity['status'] == 'REJECT':
                report['witness'] = regularity
            return report
        if first_status == 'PASS':
            report.update(status='PASS',
                          patch_triangle_interior_disjoint_certified=True,
                          separation_certificate=first_certificate)
            report['separation_certificate']['retained_regularity_certificate'] = regularity
            return report
        if 'orientation_xy' not in report:
            return report
        orientation = report['orientation_xy']
        powers = []
        for edge in range(4):
            direction = _subtract(boundary[(edge+1) % 4], boundary[edge])
            powers.append([tuple(orientation*x for x in _cross_xy(
                direction, _subtract(vertex, boundary[edge]))) for vertex in triangle])
        for edge in powers:
            for power in edge:
                for value in power:
                    _check_budget(value, max_rational_bits)

        if len(shared) == 1:
            shared_vid = next(iter(shared))
            corner = bv.index(shared_vid)
            incident = ((corner-1) % 4, corner)
            nonshared = [i for i in range(3) if tv[i] != shared_vid]
            for weight_a, weight_b in _weights(max_weight):
                report['positive_combination_candidates_tested'] += 1
                checks = []
                for vertex in nonshared:
                    power = tuple(weight_a*a+weight_b*b for a, b in
                                  zip(powers[incident[0]][vertex], powers[incident[1]][vertex]))
                    record = _strict_record(power, max_rational_bits)
                    record.update(triangle_vertex=vertex, source_vid=tv[vertex])
                    checks.append(record)
                if all(_negative(check) for check in checks):
                    return _pass(
                        report, 'POSITIVE_INCIDENT_HALFPLANE_COMBINATION', shared,
                        'A positive combination of the incident source half-planes separates both nonshared vertices throughout the branch.',
                        incident_edges=list(incident), positive_integer_weights=[weight_a, weight_b],
                        checks=checks)

        for edge in range(4):
            following = (edge+1) % 4
            edge_vids = {bv[edge], bv[following]}
            if not shared or not shared <= edge_vids:
                continue
            if not all(all(value == 0 for value in power) for power in powers[edge]):
                continue
            common = {
                'candidate_edge_index': edge,
                'edge_source_vids': [bv[edge], bv[following]],
                'supporting_line_polynomials_ascending': [
                    [_json_fraction(value) for value in power] for power in powers[edge]],
            }
            if len(shared) == 2:
                return _pass(
                    report, 'SUPPORTING_LINE_SOURCE_EDGE', shared,
                    'The retained projection lies identically on the source supporting line; the graph fan meets that line only on the exact shared boundary edge.',
                    **common)
            shared_vid = next(iter(shared))
            nonshared = [i for i in range(3) if tv[i] != shared_vid]
            origin = boundary[bv.index(shared_vid)]
            other = boundary[following if bv[edge] == shared_vid else edge]
            outward_edge = _subtract(other, origin)
            differences = [_subtract(vertex, origin) for vertex in triangle]
            # A dot product <0 means strictly beyond the shared endpoint,
            # away from the other endpoint, in the supporting XY line.
            beyond = [tuple(a+b for a, b in zip(
                _multiply(outward_edge[0], differences[i][0]),
                _multiply(outward_edge[1], differences[i][1]))) for i in nonshared]
            checks = [_strict_record(power, max_rational_bits) for power in beyond]
            if all(_negative(check) for check in checks):
                return _pass(report, 'SUPPORTING_LINE_BEYOND_SHARED_VERTEX', shared,
                             'Every nonshared projected vertex stays strictly beyond the declared endpoint of the supporting edge.',
                             **common, checks=checks)
            # In this vertical supporting plane the following linear
            # functionals vanish on the full spatial source edge line.
            for axes in ((0, 2), (1, 2)):
                raw = [_cross_component(outward_edge, differences[i], axes) for i in nonshared]
                for sign in (1, -1):
                    checks = [_strict_record(tuple(sign*x for x in power), max_rational_bits)
                              for power in raw]
                    if all(_negative(check) for check in checks):
                        return _pass(
                            report, 'SUPPORTING_LINE_SPATIAL_SHARED_VERTEX', shared,
                            'Both nonshared vertices remain strictly on the same side of the spatial boundary-edge line within its supporting vertical plane.',
                            **common, projection_axes=''.join('xyz'[axis] for axis in axes),
                            signed_functional=sign, checks=checks)
        report['reason'] = ('Neither the old strict separator nor the bounded exact source-contact tests certify this branch. '
                            'UNKNOWN does not diagnose an intersection.')
    except _BudgetExceeded as error:
        report.update(status='UNKNOWN', reason=str(error), witness=None,
                      patch_triangle_interior_disjoint_certified=False, separation_certificate=None)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError) as error:
        report.update(status='REJECT', reason='Invalid certificate input: '+str(error),
                      witness={'kind': 'invalid_input', 'diagnostic': str(error)},
                      patch_triangle_interior_disjoint_certified=False, separation_certificate=None)
    return report
