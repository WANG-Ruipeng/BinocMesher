#!/usr/bin/env python3
"""Exact local certificates for affine four-boundary-vertex graph fans.

This intentionally narrow kernel certifies a strictly convex XY footprint and
an interior center throughout ONE affine source branch.  It does not certify
source ownership, retained-exterior intersections, or rounded C++ evaluation.
No sampling, numerical tolerance, or floating-point polynomial solver is used.
"""

from fractions import Fraction
from numbers import Integral, Real


DEFAULT_MAX_RATIONAL_BITS = 4096


class _BudgetExceeded(Exception):
    pass


def _json_fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def _check_budget(value, limit):
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > limit:
        raise _BudgetExceeded('Exact rational coordinate/coefficient exceeds the bit budget.')
    return value


def _fraction(value, limit):
    if isinstance(value, bool):
        raise ValueError('Boolean coordinates are not supported.')
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, dict):
        numerator, denominator = value['numerator'], value['denominator']
        if (isinstance(numerator, bool) or isinstance(denominator, bool)
                or not isinstance(numerator, Integral)
                or not isinstance(denominator, Integral)):
            raise ValueError('Fraction JSON must contain integer numerator and denominator.')
        result = Fraction(int(numerator), int(denominator))
    elif isinstance(value, Integral):
        result = Fraction(int(value))
    elif isinstance(value, Real):
        # Promotion of a binary32 value to binary64 is exact as well.
        result = Fraction.from_float(float(value))
    else:
        raise ValueError('Coordinates must be rational JSON, Fraction, integer, or finite real.')
    return _check_budget(result, limit)


def _vector(values, limit):
    if len(values) != 3:
        raise ValueError('Each position must have exactly three coordinates.')
    return tuple(_fraction(value, limit) for value in values)


def _boundary(values, limit):
    if len(values) != 4:
        raise ValueError('Exactly four boundary vertices are required.')
    return tuple(_vector(value, limit) for value in values)


def _trajectory(first, second):
    return tuple((a, b-a) for a, b in zip(first, second))


def _subtract(first, second):
    return tuple((a[0]-b[0], a[1]-b[1]) for a, b in zip(first, second))


def _multiply(first, second):
    return (first[0]*second[0],
            first[0]*second[1]+first[1]*second[0],
            first[1]*second[1])


def _cross_xy(first, second):
    left = _multiply(first[0], second[1])
    right = _multiply(first[1], second[0])
    return tuple(a-b for a, b in zip(left, right))


def _evaluate(coefficients, time):
    constant, linear, quadratic = coefficients
    return constant + time*(linear + time*quadratic)


def _minimum_quadratic(coefficients, limit):
    """All extrema on [0,1]: endpoints plus the rational derivative root.

The polynomial degree is at most two.  Thus this is a complete exact minimum
calculation, including tangential zeros; no subdivision/root-isolation budget
or heuristic decision on mixed Bernstein signs is necessary.
"""
    constant, linear, quadratic = coefficients
    candidates = [Fraction(0), Fraction(1)]
    if quadratic:
        stationary = _check_budget(-linear/(2*quadratic), limit)
        if 0 < stationary < 1:
            candidates.append(stationary)
    values = [(time, _check_budget(_evaluate(coefficients, time), limit))
              for time in sorted(candidates)]
    minimum_time, minimum = min(values, key=lambda pair: (pair[1], pair[0]))
    return minimum_time, minimum, values


def _base_report(limit):
    return {
        'schema': 'c1-lite-local-affine-graph-certificate-v1',
        'status': 'UNKNOWN',
        'scope': 'ONE_AFFINE_BRANCH_STRICTLY_CONVEX_XY_GRAPH_FAN_ONLY',
        'coordinate_model': 'Exact rational affine trajectories; input floats denote their exact binary64 values.',
        'normalized_interval': [_json_fraction(Fraction(0)), _json_fraction(Fraction(1))],
        'limits': {'max_rational_bits': limit},
        'proof_method': 'Exact quadratic extrema on the closed interval; Bernstein coefficients are also recorded.',
        'local_segment_geometry_certified': False,
        'orientation_xy': None,
        'constraints': [],
        'witness': None,
        'not_certified': [
            'Whether the supplied boundary equals the ordinary source throughout this branch.',
            'Source branch/owner stability or exact-once suppression.',
            'Intersections or incidence with retained exterior or other event patches.',
            'Binary32 quantization, C++ runtime interpolation, or emitted mesh arrays.',
            'General non-graph fans, temporal C1 smoothness, or a topology-changing saddle.',
        ],
    }


def certify_segment(boundary_start, boundary_end, center_start, center_end,
                    *, max_rational_bits=DEFAULT_MAX_RATIONAL_BITS):
    """Return a JSON-compatible PASS / REJECT / UNKNOWN local certificate.

Boundary inputs are 4x3 cyclically ordered positions and centers are length 3.
Every vertex moves affinely from its start to end position as normalized time
goes from 0 to 1.  Scalars may be Fraction, integer, finite real, or fraction
JSON.  A float means the exact binary64 value, never an intended decimal.

PASS proves all eight other-vertex half-plane inequalities and all four center
half-plane inequalities strictly positive in one fixed orientation.  Hence the
footprint is a convex quadrilateral and the fan is its non-overlapping graph
triangulation at every time.  Z values are unrestricted affine trajectories.

REJECT gives an exact nonpositive-inequality witness (or an input diagnostic).
It rejects this sufficient graph class, not every possible 3D realization.
UNKNOWN denotes an exhausted arithmetic-size budget, not invalid geometry.
"""
    report = _base_report(max_rational_bits)
    try:
        if (isinstance(max_rational_bits, bool)
                or not isinstance(max_rational_bits, Integral)
                or max_rational_bits < 1):
            raise ValueError('max_rational_bits must be a positive integer.')
        b0, b1 = _boundary(boundary_start, max_rational_bits), _boundary(boundary_end, max_rational_bits)
        c0, c1 = _vector(center_start, max_rational_bits), _vector(center_end, max_rational_bits)
        boundary = [_trajectory(a, b) for a, b in zip(b0, b1)]
        center = _trajectory(c0, c1)
        reference = _cross_xy(_subtract(boundary[1], boundary[0]),
                              _subtract(boundary[2], boundary[0]))[0]
        orientation = 1 if reference >= 0 else -1
        report['orientation_xy'] = orientation
        raw_constraints = []
        for edge in range(4):
            following = (edge+1) % 4
            direction = _subtract(boundary[following], boundary[edge])
            for other in range(4):
                if other in (edge, following):
                    continue
                coefficients = _cross_xy(direction, _subtract(boundary[other], boundary[edge]))
                raw_constraints.append(('boundary_strict_halfplane', edge, other, coefficients))
            coefficients = _cross_xy(direction, _subtract(center, boundary[edge]))
            raw_constraints.append(('fan_center_strict_halfplane', edge, None, coefficients))

        for kind, edge, other, raw in raw_constraints:
            power = tuple(_check_budget(orientation*value, max_rational_bits) for value in raw)
            bernstein = (power[0], power[0]+power[1]/2, sum(power, Fraction(0)))
            for value in bernstein:
                _check_budget(value, max_rational_bits)
            time, minimum, candidates = _minimum_quadratic(power, max_rational_bits)
            item = {
                'kind': kind, 'edge': [edge, (edge+1) % 4], 'other_vertex': other,
                'power_coefficients_ascending': [_json_fraction(value) for value in power],
                'bernstein_degree_2': [_json_fraction(value) for value in bernstein],
                'bernstein_all_strictly_positive': all(value > 0 for value in bernstein),
                'exact_minimum': _json_fraction(minimum),
                'minimum_at_normalized_time': _json_fraction(time),
                'extremum_candidates': [{'time': _json_fraction(t), 'value': _json_fraction(v)}
                                       for t, v in candidates],
                'strictly_positive': minimum > 0,
            }
            report['constraints'].append(item)
            if minimum <= 0 and report['witness'] is None:
                report['witness'] = {
                    'kind': kind, 'edge': item['edge'], 'other_vertex': other,
                    'normalized_time': _json_fraction(time), 'exact_signed_value': _json_fraction(minimum),
                    'interpretation': 'Exact violation of the chosen strict convex-graph condition; not a claim about every general 3D fan.',
                }
        if report['witness'] is not None:
            report['status'] = 'REJECT'
            report['reason'] = 'A strict convex-graph inequality is nonpositive inside the closed branch.'
            return report
        fan_minimum = min(Fraction(item['exact_minimum']['numerator'], item['exact_minimum']['denominator'])
                          for item in report['constraints'] if item['kind'] == 'fan_center_strict_halfplane')
        report['status'] = 'PASS'
        report['reason'] = 'Every footprint and center half-plane inequality is strictly positive throughout the branch.'
        report['local_segment_geometry_certified'] = True
        report['minimum_signed_projected_double_fan_area'] = _json_fraction(fan_minimum)
        report['minimum_3d_fan_area_lower_bound'] = _json_fraction(fan_minimum/2)
        report['certified_properties'] = [
            'Strictly convex simple XY boundary cycle in the recorded fixed orientation.',
            'Center strictly inside that footprint at every normalized query time.',
            'All four fan triangles nondegenerate with consistent projected orientation.',
            'Fan sectors intersect only in their declared abstract edges or center.',
            'Continuous local embedded disk family for these exact affine trajectories.',
        ]
    except _BudgetExceeded as error:
        report['status'] = 'UNKNOWN'
        report['reason'] = str(error)
        report['local_segment_geometry_certified'] = False
        report['witness'] = None
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError) as error:
        report['status'] = 'REJECT'
        report['reason'] = 'Invalid input: '+str(error)
        report['witness'] = {'kind': 'invalid_input', 'diagnostic': str(error)}
    return report
