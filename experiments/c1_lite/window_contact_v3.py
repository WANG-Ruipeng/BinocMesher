#!/usr/bin/env python3
"""Bounded exact oblique and zero-feature certificates for graph fans.

This extends v2 without changing it. The same certified convex XY graph-fan
precondition is required. All trajectories here are ideal rational affine
trajectories, not binary32 evaluation or extra_smooth runtime coordinates.
"""

from fractions import Fraction
from itertools import combinations

from window_contact_v2 import (
    _cross_component, _negative, _pass, _strict_record,
    certify_contact as certify_contact_v2,
)
from window_exterior import _ids
from window_geometry import (
    DEFAULT_MAX_RATIONAL_BITS, _BudgetExceeded, _boundary, _check_budget,
    _cross_xy, _json_fraction, _subtract, _trajectory, _vector,
)


def oblique_directions(max_weight):
    """Fixed finite primitive normals, with the first nonzero entry positive.

For each coordinate pair use ratios 1:k and k:1, relative signs +/- and
1<=k<=max_weight. The caller tests both orientations of each normal.
"""
    directions = set()
    for i, j in combinations(range(3), 2):
        for k in range(1, max_weight+1):
            for a, b in ((1, k), (k, 1)):
                for sign in (1, -1):
                    normal = [0, 0, 0]
                    normal[i], normal[j] = a, sign*b
                    directions.add(tuple(normal))
    return tuple(sorted(directions, key=lambda n: (max(map(abs, n)), sum(map(abs, n)), n)))


def _dot(normal, point, limit):
    value = Fraction(0)
    for coefficient, coordinate in zip(normal, point):
        value = _check_budget(value+coefficient*coordinate, limit)
    return value


def certify_contact(boundary_start, boundary_end, boundary_vids,
                    triangle_start, triangle_end, triangle_vids, *,
                    center_start=None, center_end=None, orientation_xy=None,
                    max_rational_bits=DEFAULT_MAX_RATIONAL_BITS, max_weight=32,
                    max_direction_weight=8):
    """Sufficient exact PASS / UNKNOWN / REJECT contact certificate.

Reuse every v2 PASS/REJECT. New oblique strict separation includes all five
fan vertices and all three retained vertices at both endpoints. Every pair
projection difference is affine, so those exact bounds cover the full branch.

For a source supporting half-plane h>=0, require EACH retained-vertex h to be
either identically zero or strictly negative throughout the closed branch.
Only the convex hull of the identically-zero vertices can touch the patch.
In that feature, a spatial XZ/YZ edge-line functional must be identically zero
at declared shared endpoints and strictly of one sign at every other vertex.
Thus its intersection with the patch is exactly the declared point/edge, or
empty. An isolated h=0 is unsupported, never admitted by replacing < with <=.

The shared-face regularity and exact identity gates from v2 remain mandatory.
UNKNOWN means no sufficient proof, not an intersection diagnosis. Disjoint
retained degeneracies need not be repaired or certified regular by this API.
"""
    report = certify_contact_v2(
        boundary_start, boundary_end, boundary_vids,
        triangle_start, triangle_end, triangle_vids,
        center_start=center_start, center_end=center_end,
        orientation_xy=orientation_xy, max_rational_bits=max_rational_bits,
        max_weight=max_weight)
    report['schema'] = 'c1-lite-affine-source-contact-v3'
    report['v2_status'] = report['status']
    report['limits']['max_direction_weight'] = max_direction_weight
    report['oblique_directions_tested'] = 0
    report['zero_feature_candidates'] = []
    report['oblique_certificate_enabled'] = False
    report['not_certified'].append(
        'Stability of exact supporting-line identities under runtime rounding; ideal margins are not binary32 certificates.')
    try:
        if (isinstance(max_direction_weight, bool) or not isinstance(max_direction_weight, int)
                or not 1 <= max_direction_weight <= 8):
            raise ValueError('max_direction_weight must be an integer in [1, 8].')
        if report['status'] in ('PASS', 'REJECT'):
            return report
        if 'orientation_xy' not in report:
            return report
        b0, b1 = _boundary(boundary_start, max_rational_bits), _boundary(boundary_end, max_rational_bits)
        p0 = tuple(_vector(point, max_rational_bits) for point in triangle_start)
        p1 = tuple(_vector(point, max_rational_bits) for point in triangle_end)
        bv, tv = _ids(boundary_vids, 4, 'boundary_vids'), _ids(triangle_vids, 3, 'triangle_vids')
        shared = set(bv) & set(tv)
        for index, source_vid in enumerate(tv):
            if source_vid in shared:
                j = bv.index(source_vid)
                if p0[index] != b0[j] or p1[index] != b1[j]:
                    raise ValueError('Shared SourceVID has inconsistent exact endpoint trajectories.')
        if shared and (report.get('retained_regularity_certificate') or {}).get('status') != 'PASS':
            return report
        c0 = _vector(center_start, max_rational_bits) if center_start is not None else None
        c1 = _vector(center_end, max_rational_bits) if center_end is not None else None

        # A declared exact shared point makes strict full-hull separation
        # impossible. Skip those directions, rather than consuming the budget.
        if c0 is not None and not shared:
            report['oblique_certificate_enabled'] = True
            directions = oblique_directions(max_direction_weight)
            report['limits']['oblique_direction_count'] = len(directions)
            for normal in directions:
                report['oblique_directions_tested'] += 1
                endpoints = []
                for patch, retained in (((*b0, c0), p0), ((*b1, c1), p1)):
                    patch_values = [_dot(normal, vertex, max_rational_bits) for vertex in patch]
                    retained_values = [_dot(normal, vertex, max_rational_bits) for vertex in retained]
                    endpoints.append((patch_values, retained_values))
                for sign in (1, -1):
                    gaps = [_check_budget(sign*(q-p), max_rational_bits)
                            for patch, retained in endpoints for p in patch for q in retained]
                    margin = min(gaps)
                    if margin > 0:
                        oriented_normal = [sign*x for x in normal]
                        return _pass(
                            report, 'STRICT_BOUNDED_OBLIQUE_CONVEX_HULL', set(),
                            'A fixed bounded integer normal strictly separates all fan/retained vertex pairs at both endpoints; affine differences preserve the gap.',
                            normal=oriented_normal,
                            normal_l1=sum(map(abs, normal)),
                            minimum_signed_projection_gap=_json_fraction(margin),
                            endpoint_pair_count=len(gaps), includes_patch_center=True,
                            direction_family='coordinate pairs, ratios 1:k or k:1, both relative signs and both orientations')

        boundary = [_trajectory(a, b) for a, b in zip(b0, b1)]
        triangle = [_trajectory(a, b) for a, b in zip(p0, p1)]
        orientation = report['orientation_xy']
        for edge in range(4):
            following = (edge+1) % 4
            edge_vids = {bv[edge], bv[following]}
            if not shared <= edge_vids:
                continue
            direction = _subtract(boundary[following], boundary[edge])
            differences = [_subtract(vertex, boundary[edge]) for vertex in triangle]
            checks, zero_vertices, eligible = [], [], True
            for index, difference in enumerate(differences):
                power = tuple(_check_budget(orientation*x, max_rational_bits)
                              for x in _cross_xy(direction, difference))
                if all(value == 0 for value in power):
                    zero_vertices.append(index)
                    check = {'role': 'IDENTICALLY_ZERO_SUPPORT_FEATURE',
                             'power_coefficients_ascending': [_json_fraction(x) for x in power]}
                else:
                    check = _strict_record(power, max_rational_bits)
                    check['role'] = 'STRICT_EXTERIOR_REQUIRED'
                    eligible = eligible and _negative(check)
                check.update(triangle_vertex=index, source_vid=tv[index])
                checks.append(check)
            candidate = {'edge': [edge, following], 'halfplane_checks': checks,
                         'zero_feature_vertex_indices': zero_vertices,
                         'supported_halfplane_pattern': eligible}
            report['zero_feature_candidates'].append(candidate)
            if not eligible or not zero_vertices:
                continue
            if any(tv.index(source_vid) not in zero_vertices for source_vid in shared):
                raise ValueError('A declared shared source endpoint is not in the exact supporting-line zero feature.')
            nonshared_zero = [index for index in zero_vertices if tv[index] not in shared]
            common = {
                'candidate_edge_index': edge,
                'edge_source_vids': [bv[edge], bv[following]],
                'halfplane_checks': checks,
                'zero_feature_vertex_indices': zero_vertices,
                'zero_feature_source_vids': [tv[index] for index in zero_vertices],
            }
            if not nonshared_zero:
                return _pass(
                    report, 'SOURCE_HALFPLANE_ZERO_FEATURE_IDENTITY', shared,
                    'The entire supporting-line zero feature consists of exact declared shared endpoints; every other vertex is strictly outside.',
                    **common)
            for axes in ((0, 2), (1, 2)):
                powers = {index: _cross_component(direction, differences[index], axes)
                          for index in zero_vertices}
                for power in powers.values():
                    for value in power:
                        _check_budget(value, max_rational_bits)
                if any(any(x != 0 for x in powers[tv.index(source_vid)]) for source_vid in shared):
                    raise ValueError('A shared endpoint contradicts the exact spatial source-edge line.')
                for sign in (1, -1):
                    spatial_checks = []
                    for index in zero_vertices:
                        if tv[index] in shared:
                            check = {'role': 'EXACT_SHARED_ENDPOINT_ZERO',
                                     'power_coefficients_ascending': [_json_fraction(x) for x in powers[index]]}
                        else:
                            check = _strict_record(tuple(sign*x for x in powers[index]), max_rational_bits)
                            check['role'] = 'NONSHARED_ZERO_FEATURE_STRICT_SPATIAL_SIDE'
                        check.update(triangle_vertex=index, source_vid=tv[index])
                        spatial_checks.append(check)
                    if all(_negative(check) for check in spatial_checks if check['source_vid'] not in shared):
                        return _pass(
                            report, 'SOURCE_HALFPLANE_ZERO_FEATURE_SPATIAL', shared,
                            'Only the identically-zero projected feature can reach the fan; a spatial source-edge-line functional restricts that feature to exactly the declared contact.',
                            **common, projection_axes=''.join('xyz'[axis] for axis in axes),
                            signed_functional=sign, spatial_feature_checks=spatial_checks)
        report['reason'] = ('No v2 proof, bounded strict oblique separator, or exact zero-feature spatial-contact certificate was found. '
                            'Isolated or unresolved half-plane zeros are not admitted; UNKNOWN does not diagnose a collision.')
    except _BudgetExceeded as error:
        report.update(status='UNKNOWN', reason=str(error), witness=None,
                      patch_triangle_interior_disjoint_certified=False, separation_certificate=None)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError) as error:
        report.update(status='REJECT', reason='Invalid certificate input: '+str(error),
                      witness={'kind': 'invalid_input', 'diagnostic': str(error)},
                      patch_triangle_interior_disjoint_certified=False, separation_certificate=None)
    return report
