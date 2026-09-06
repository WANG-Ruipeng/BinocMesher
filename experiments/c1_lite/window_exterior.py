#!/usr/bin/env python3
"""Conservative exact separation of a graph fan and one retained triangle.

The caller must already have certified the patch's convex XY graph geometry
on the same affine branch. Failure to find a separating half-plane is UNKNOWN,
never a collision diagnosis. This module does not inspect real caches.
"""

from fractions import Fraction

from window_geometry import (
    DEFAULT_MAX_RATIONAL_BITS, _BudgetExceeded, _boundary, _check_budget,
    _json_fraction, _minimum_quadratic, _subtract, _trajectory, _vector,
    _cross_xy,
)


def _maximum_polynomial(power, limit):
    for value in power:
        _check_budget(value, limit)
    negated = tuple(-value for value in power)
    time, minimum, _ = _minimum_quadratic(negated, limit)
    return time, -minimum


def _polynomial_record(power, time, maximum):
    bernstein = (power[0], power[0]+power[1]/2, sum(power, Fraction(0)))
    return {
        'power_coefficients_ascending': [_json_fraction(value) for value in power],
        'bernstein_degree_2': [_json_fraction(value) for value in bernstein],
        'exact_maximum': _json_fraction(maximum),
        'maximum_at_normalized_time': _json_fraction(time),
        'strictly_negative': maximum < 0,
    }


def _ids(values, length, name):
    if len(values) != length or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(name+' must contain the required number of nonempty SourceVID strings.')
    if len(set(values)) != length:
        raise ValueError(name+' contains repeated SourceVIDs; this certificate requires distinct vertices.')
    return tuple(values)


def _report(limit):
    return {
        'schema': 'c1-lite-affine-exterior-separation-v1',
        'status': 'UNKNOWN',
        'scope': 'CONDITIONAL_LOCAL_FAN_VS_ONE_AFFINE_RETAINED_TRIANGLE',
        'coordinate_model': 'Exact rational affine trajectories; no runtime rounding.',
        'required_precondition': 'Caller has certified the same boundary as a strictly convex XY graph fan with its center strictly inside throughout this branch.',
        'patch_geometry_checked_here': False,
        'patch_triangle_interior_disjoint_certified': False,
        'limits': {'max_rational_bits': limit},
        'separation_certificate': None,
        'candidate_edges': [],
        'witness': None,
        'not_certified': [
            'Any existing defect or self-intersection of the retained mesh.',
            'Boundary incidence, orientation, owner suppression, or whole-mesh manifold links.',
            'Source trajectories matching their runtime records or binary32 rounding.',
            'A collision when this conservative separator returns UNKNOWN.',
        ],
    }


def certify_exterior_separation(boundary_start, boundary_end, boundary_vids,
                                triangle_start, triangle_end, triangle_vids,
                                *, center_start=None, center_end=None,
                                orientation_xy=None,
                                max_rational_bits=DEFAULT_MAX_RATIONAL_BITS):
    """Prove separation by an exterior XY half-plane or coordinate axis.

An edge proves separation when every triangle vertex that is not a declared
shared edge endpoint stays STRICTLY outside that edge throughout [0,1]. A
shared endpoint must have the same SourceVID and identical exact trajectory.
This allows only the corresponding shared vertex/edge contact. Distinct IDs
at coincident coordinates do not qualify as declared shared endpoints.

Optional center trajectories enable a second sufficient test: strict axis
separation of the triangle and all five fan vertices. Without the center this
test is disabled: boundary Z bounds need not contain the fan's interior.

REJECT is reserved for invalid input or contradicted shared identity.
No separating certificate, including a real collision, returns UNKNOWN.
"""
    report = _report(max_rational_bits)
    try:
        if (isinstance(max_rational_bits, bool) or not isinstance(max_rational_bits, int)
                or max_rational_bits < 1):
            raise ValueError('max_rational_bits must be a positive integer.')
        b0, b1 = _boundary(boundary_start, max_rational_bits), _boundary(boundary_end, max_rational_bits)
        if len(triangle_start) != 3 or len(triangle_end) != 3:
            raise ValueError('The retained triangle must have three start and end vertices.')
        t0 = tuple(_vector(value, max_rational_bits) for value in triangle_start)
        t1 = tuple(_vector(value, max_rational_bits) for value in triangle_end)
        bv = _ids(boundary_vids, 4, 'boundary_vids')
        tv = _ids(triangle_vids, 3, 'triangle_vids')
        if (center_start is None) != (center_end is None):
            raise ValueError('Both center endpoints are required for the optional axis certificate.')
        c0 = _vector(center_start, max_rational_bits) if center_start is not None else None
        c1 = _vector(center_end, max_rational_bits) if center_end is not None else None
        boundary = [_trajectory(a, b) for a, b in zip(b0, b1)]
        triangle = [_trajectory(a, b) for a, b in zip(t0, t1)]
        initial_turn = _cross_xy(_subtract(boundary[1], boundary[0]),
                                 _subtract(boundary[2], boundary[0]))[0]
        if not initial_turn:
            report['reason'] = 'The required strict convex XY precondition fails at the branch start.'
            return report
        orientation = 1 if initial_turn > 0 else -1
        if orientation_xy is not None and orientation_xy != orientation:
            raise ValueError('Supplied orientation disagrees with the exact initial source cycle.')
        report['orientation_xy'] = orientation
        report['declared_shared_vids'] = sorted(set(bv) & set(tv))
        for index, source_vid in enumerate(tv):
            if source_vid not in bv:
                continue
            boundary_index = bv.index(source_vid)
            for name, actual, expected in (('start', t0[index], b0[boundary_index]),
                                           ('end', t1[index], b1[boundary_index])):
                if actual != expected:
                    report['status'] = 'REJECT'
                    report['reason'] = 'The same SourceVID has incompatible exact affine trajectories.'
                    report['witness'] = {
                        'kind': 'shared_identity_mismatch', 'source_vid': source_vid,
                        'endpoint': name,
                        'triangle_position': [_json_fraction(value) for value in actual],
                        'boundary_position': [_json_fraction(value) for value in expected],
                    }
                    return report

        for edge in range(4):
            following = (edge+1) % 4
            direction = _subtract(boundary[following], boundary[edge])
            edge_vids = (bv[edge], bv[following])
            checks, shared, separated = [], [], True
            for index, source_vid in enumerate(tv):
                if source_vid in edge_vids:
                    shared.append(source_vid)
                    checks.append({'triangle_vertex': index, 'source_vid': source_vid,
                                   'role': 'EXACT_SHARED_ENDPOINT_TRAJECTORY'})
                    continue
                raw = _cross_xy(direction, _subtract(triangle[index], boundary[edge]))
                power = tuple(orientation*value for value in raw)
                time, maximum = _maximum_polynomial(power, max_rational_bits)
                check = _polynomial_record(power, time, maximum)
                check.update({'triangle_vertex': index, 'source_vid': source_vid,
                              'role': 'NONSHARED_STRICT_EXTERIOR'})
                checks.append(check)
                separated = separated and maximum < 0
            candidate = {'edge': [edge, following], 'edge_source_vids': list(edge_vids),
                         'checks': checks, 'certified': separated}
            report['candidate_edges'].append(candidate)
            if separated:
                contact = ('shared_edge' if len(shared) == 2 else
                           'shared_vertex' if shared else 'empty')
                report['status'] = 'PASS'
                report['reason'] = 'A source-boundary half-plane separates every nonshared retained vertex throughout the branch.'
                report['patch_triangle_interior_disjoint_certified'] = True
                report['separation_certificate'] = {
                    'kind': 'STRICT_SOURCE_EDGE_EXTERIOR', 'candidate_edge_index': edge,
                    'allowed_contact': contact, 'allowed_contact_source_vids': sorted(shared),
                }
                return report

        if c0 is not None:
            patch_start, patch_end = (*b0, c0), (*b1, c1)
            for axis in range(3):
                for sign in (1, -1):
                    # sign*(patch - retained) < 0, for all pairs and times.
                    endpoint_values = [
                        sign*(patch[pi][axis]-retained[ti][axis])
                        for patch, retained in ((patch_start, t0), (patch_end, t1))
                        for pi in range(5) for ti in range(3)
                    ]
                    for value in endpoint_values:
                        _check_budget(value, max_rational_bits)
                    maximum = max(endpoint_values)
                    if maximum < 0:
                        report['status'] = 'PASS'
                        report['reason'] = 'Every fan vertex and retained vertex pair has a strict coordinate-axis gap at both endpoints; affine interpolation preserves it.'
                        report['patch_triangle_interior_disjoint_certified'] = True
                        report['separation_certificate'] = {
                            'kind': 'STRICT_COORDINATE_AXIS', 'axis': 'xyz'[axis],
                            'sign_of_patch_minus_retained': sign,
                            'exact_minimum_gap_lower_bound': _json_fraction(-maximum),
                            'endpoint_pair_count': len(endpoint_values),
                            'includes_patch_center': True, 'allowed_contact': 'empty',
                            'allowed_contact_source_vids': [],
                        }
                        return report
        report['reason'] = ('No strict source-edge or enabled full-fan coordinate-axis separator was certified. '
                            'This is not evidence of a collision.')
        report['axis_certificate_enabled'] = c0 is not None
    except _BudgetExceeded as error:
        report['status'] = 'UNKNOWN'
        report['reason'] = str(error)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError) as error:
        report['status'] = 'REJECT'
        report['reason'] = 'Invalid certificate input: '+str(error)
        report['witness'] = {'kind': 'invalid_input', 'diagnostic': str(error)}
    return report
