"""Conditional all-perturbations certificates; NEVER runtime admission.

All actual coordinates must be within the supplied uniform epsilon of their
corresponding ideal trajectories for every query time. Shared identities must
reuse the same actual coordinate. Those assumptions are NOT established here.
"""
from fractions import Fraction as F

from audit_relative_plane_feasibility import audit_relative_plane
from window_contact_v2 import _cross_component, _weights, _normal_certificate
from window_contact_v3 import oblique_directions
from window_geometry import (_boundary, _vector, _fraction, _trajectory,
    _subtract, _cross_xy, _minimum_quadratic, _json_fraction, _BudgetExceeded,
    certify_segment)
from window_exterior import _ids

BITS = 4096
TILTS = tuple(x for q in (F(1), F(1, 2), F(2), F(1, 4), F(4), F(1, 8),
                         F(8), F(1, 16), F(1, 32), F(1, 64)) for x in (q, -q))


def _fr(value):
    return _fraction(value, BITS)


def _base(epsilon, scope):
    return {'schema': 'c1-lite-conditional-perturbation-v1', 'status': 'UNKNOWN',
        'scope': scope, 'epsilon': _json_fraction(epsilon), 'runtime_admitted': False,
        'actual_error_bound_proven': False, 'actual_identity_mapping_proven': False,
        'actual_time_selector_proven': False, 'proof': None, 'reason': '',
        'required_condition': 'Every actual raw occurrence and merged representative lies within epsilon per coordinate of its corresponding ideal trajectory; shared interface entities reuse one actual coordinate. This includes all times, not samples.'}


def _error_record(margin, bmax, quadratic, epsilon):
    error = 2*epsilon*bmax+quadratic*epsilon*epsilon
    return {'ideal_strict_margin': _json_fraction(margin), 'Bmax': _json_fraction(bmax),
        'quadratic_coefficient': _json_fraction(F(quadratic)),
        'error_upper_bound': _json_fraction(error),
        'remaining_margin': _json_fraction(margin-error), 'sufficient': margin > error}


def _bilinear_bmax(first, second, third, axes):
    i, j = axes
    values = []
    for a, b, p in zip(first, second, third):
        e, d = tuple(x-y for x, y in zip(b, a)), tuple(x-y for x, y in zip(p, a))
        values.append(abs(e[i])+abs(d[j])+abs(e[j])+abs(d[i]))
    return max(values)


def _parse(boundary_start, boundary_end, center_start, center_end):
    return (_boundary(boundary_start, BITS), _boundary(boundary_end, BITS),
            _vector(center_start, BITS), _vector(center_end, BITS))


def certify_local_perturbation(boundary_start, boundary_end, center_start, center_end, epsilon):
    epsilon = _fr(epsilon)
    if epsilon < 0:
        raise ValueError('epsilon must be nonnegative.')
    report = _base(epsilon, 'CONDITIONAL_ALL_TIMES_CONVEX_XY_GRAPH_FAN')
    try:
        b0, b1, c0, c1 = _parse(boundary_start, boundary_end, center_start, center_end)
        exact = certify_segment(b0, b1, c0, c1)
        if exact['status'] != 'PASS':
            report.update(status=exact['status'], reason='Ideal local graph is not certified.', ideal=exact)
            return report
        checks = []
        for constraint in exact['constraints']:
            edge, following = constraint['edge']
            other = constraint['other_vertex']
            point = (c0, c1) if other is None else (b0[other], b1[other])
            bmax = _bilinear_bmax((b0[edge], b1[edge]), (b0[following], b1[following]), point, (0, 1))
            check = _error_record(_fr(constraint['exact_minimum']), bmax, 8, epsilon)
            check.update(edge=constraint['edge'], other_vertex=other,
                         kind=constraint['kind'])
            checks.append(check)
        report['proof'] = {'kind': 'ROBUST_TWELVE_STRICT_XY_HALFPLANES',
                           'orientation_xy': exact['orientation_xy'], 'checks': checks}
        report['status'] = 'PASS_CONDITIONAL' if all(c['sufficient'] for c in checks) else 'UNKNOWN'
        report['reason'] = ('Every graph/fan signed projected area remains strictly positive under the stated coordinate envelope.'
                            if report['status'] == 'PASS_CONDITIONAL' else 'The chosen epsilon exceeds at least one certified local graph margin.')
    except _BudgetExceeded as error:
        report['reason'] = str(error)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError) as error:
        report.update(status='REJECT', reason='Invalid conditional input: '+str(error))
    return report


def _shared_regularity(p0, p1, epsilon):
    pt = [_trajectory(a, b) for a, b in zip(p0, p1)]
    e, d = _subtract(pt[1], pt[0]), _subtract(pt[2], pt[0])
    for axes in ((1, 2), (2, 0), (0, 1)):
        power = _cross_component(e, d, axes)
        bmax = _bilinear_bmax((p0[0], p1[0]), (p0[1], p1[1]), (p0[2], p1[2]), axes)
        for sign in (1, -1):
            _, margin, _ = _minimum_quadratic(tuple(sign*x for x in power), BITS)
            check = _error_record(margin, bmax, 8, epsilon)
            if check['sufficient']:
                return {'status': 'PASS_CONDITIONAL', 'kind': 'ROBUST_SHARED_TRIANGLE_NORMAL_COMPONENT',
                        'axes': list(axes), 'sign': sign, **check}
    ideal = _normal_certificate(pt, BITS)
    return {'status': 'REJECT' if ideal['status'] == 'REJECT' else 'UNKNOWN',
            'kind': 'SHARED_TRIANGLE_REGULARITY_UNRESOLVED', 'ideal': ideal}


def _fixed_direction(b0, b1, c0, c1, p0, p1, normal, epsilon):
    gaps = []
    for patch, retained in (((*b0, c0), p0), ((*b1, c1), p1)):
        a = [sum((n*x for n, x in zip(normal, p)), F(0)) for p in patch]
        b = [sum((n*x for n, x in zip(normal, p)), F(0)) for p in retained]
        gaps.extend(q-p for p in a for q in b)
    error = 2*epsilon*sum(map(abs, normal))
    for sign in (1, -1):
        margin = min(sign*x for x in gaps)
        if margin > error:
            return {'kind': 'ROBUST_FIXED_DIRECTION_CONVEX_HULL',
                'normal': [sign*x for x in normal], 'includes_center': True,
                'endpoint_pairs': 30, 'ideal_strict_margin': _json_fraction(margin),
                'error_upper_bound': _json_fraction(error),
                'remaining_margin': _json_fraction(margin-error)}
    return None


def _relative_edge(b0, b1, bv, c0, c1, p0, p1, tv, edge, tilt, projection, epsilon):
    ideal = audit_relative_plane(b0, b1, bv, c0, c1, p0, p1, tv,
                                 edge_index=edge, tilt=tilt, projection=projection)
    if ideal['status'] != 'PASS':
        return None
    error = _error_record(_fr(ideal['minimum_signed_margin']), _fr(ideal['Bmax']),
                          8*(1+abs(tilt)), epsilon)
    if not error['sufficient']:
        return None
    return {'kind': 'ROBUST_SOURCE_EDGE_RELATIVE_PLANE', 'edge': [edge, (edge+1) % 4],
        'edge_source_vids': ideal['edge_source_vids'], 'tilt': _json_fraction(tilt),
        'projection': projection, 'orientation_xy': ideal['orientation_xy'],
        'allowed_contact': ideal['allowed_contact'], 'structural_zero': 'Common actual plane-defining endpoints.', **error}


def _corner(b0, b1, bv, c0, c1, p0, p1, tv, corner, weights, epsilon):
    wa, wb = weights
    total = wa+wb
    previous, following = (corner-1) % 4, (corner+1) % 4
    bt = [_trajectory(a, b) for a, b in zip(b0, b1)]
    left, right = _subtract(bt[corner], bt[previous]), _subtract(bt[following], bt[corner])
    direction = tuple(tuple(wa*a+wb*b for a, b in zip(x, y)) for x, y in zip(left, right))
    orientation = 1 if _cross_xy(_subtract(bt[1], bt[0]), _subtract(bt[2], bt[0]))[0] > 0 else -1
    tasks = [(b0[i], b1[i], 1) for i in range(4) if i != corner]
    tasks.append((c0, c1, 1))
    tasks.extend((a, b, -1) for a, b, vid in zip(p0, p1, tv) if vid != bv[corner])
    margins, bounds = [], []
    for start, end, sign in tasks:
        difference = _subtract(_trajectory(start, end), bt[corner])
        raw = _cross_xy(direction, difference)
        _, margin, _ = _minimum_quadratic(tuple(sign*orientation*x for x in raw), BITS)
        if margin <= 0:
            return None
        margins.append(margin)
        for positions, point in ((b0, start), (b1, end)):
            a = positions[corner]
            e = tuple(wa*(a[k]-positions[previous][k])+wb*(positions[following][k]-a[k]) for k in range(3))
            d = tuple(point[k]-a[k] for k in range(3))
            bounds.append(abs(e[0])+abs(e[1])+total*(abs(d[0])+abs(d[1])))
    check = _error_record(min(margins), max(bounds), 8*total, epsilon)
    if not check['sufficient']:
        return None
    return {'kind': 'ROBUST_SHARED_CORNER_RELATIVE_PLANE', 'corner': corner,
            'shared_source_vid': bv[corner], 'weights': list(weights),
            'orientation_xy': orientation, 'allowed_contact': 'shared_vertex',
            'structural_zero': 'Both incident edge functionals use one common actual corner as origin.', **check}


def certify_pair_perturbation(boundary_start, boundary_end, boundary_vids,
                              center_start, center_end,
                              triangle_start, triangle_end, triangle_vids, epsilon):
    epsilon = _fr(epsilon)
    if epsilon < 0:
        raise ValueError('epsilon must be nonnegative.')
    report = _base(epsilon, 'CONDITIONAL_ALL_TIMES_ONE_FAN_VS_ONE_RETAINED_OCCURRENCE_CLASS')
    report['requires_robust_local_graph_certificate'] = True
    report['search_limits'] = {'corner_max_weight': 32, 'tilt_candidates': len(TILTS),
                               'oblique_direction_max_weight': 8}
    try:
        b0, b1, c0, c1 = _parse(boundary_start, boundary_end, center_start, center_end)
        bv = _ids(boundary_vids, 4, 'boundary_vids')
        tv = tuple(triangle_vids)
        if len(tv) != 3 or any(not isinstance(x, str) or not x for x in tv):
            raise ValueError('Expected three nonempty triangle identity strings.')
        if len(triangle_start) != 3 or len(triangle_end) != 3:
            raise ValueError('Expected three retained positions at each endpoint.')
        p0, p1 = tuple(_vector(x, BITS) for x in triangle_start), tuple(_vector(x, BITS) for x in triangle_end)
        shared = set(tv) & set(bv)
        seen = {}
        for i, vid in enumerate(tv):
            trajectory = (p0[i], p1[i])
            if vid in seen and seen[vid] != trajectory:
                raise ValueError('Repeated identity has contradictory exact trajectories.')
            seen[vid] = trajectory
            if vid in shared:
                j = bv.index(vid)
                if trajectory != (b0[j], b1[j]):
                    raise ValueError('Shared source identity has contradictory exact trajectories.')
        report['shared_source_vids'] = sorted(shared)
        report['regularity'] = None
        if shared:
            regularity = _shared_regularity(p0, p1, epsilon)
            report['regularity'] = regularity
            if regularity['status'] != 'PASS_CONDITIONAL':
                report.update(status=regularity['status'], reason='Shared retained triangle regularity is not robustly certified at this epsilon.')
                return report

        def finish(proof):
            report.update(status='PASS_CONDITIONAL', proof=proof,
                          reason='All stated perturbations preserve this relative contact certificate, conditional on occurrence coverage and common actual identities.')
            return report

        if not shared:
            for normal in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                proof = _fixed_direction(b0, b1, c0, c1, p0, p1, normal, epsilon)
                if proof:
                    return finish(proof)
        eligible_edges = [i for i in range(4) if shared <= {bv[i], bv[(i+1) % 4]}]
        if len(set(tv)) == 3:
            for edge in eligible_edges:
                proof = _relative_edge(b0, b1, bv, c0, c1, p0, p1, tv, edge, F(0), 'xz', epsilon)
                if proof:
                    return finish(proof)
            if len(shared) == 1:
                corner = bv.index(next(iter(shared)))
                for weights in _weights(32):
                    proof = _corner(b0, b1, bv, c0, c1, p0, p1, tv, corner, weights, epsilon)
                    if proof:
                        return finish(proof)
            for tilt in TILTS:
                for edge in eligible_edges:
                    for projection in ('xz', 'yz'):
                        proof = _relative_edge(b0, b1, bv, c0, c1, p0, p1, tv, edge, tilt, projection, epsilon)
                        if proof:
                            return finish(proof)
        if not shared:
            for normal in oblique_directions(8):
                proof = _fixed_direction(b0, b1, c0, c1, p0, p1, normal, epsilon)
                if proof:
                    return finish(proof)
        report['reason'] = 'No bounded robust separating/relative plane certifies the stated epsilon; this is not a collision or impossibility diagnosis.'
    except _BudgetExceeded as error:
        report['reason'] = str(error)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError) as error:
        report.update(status='REJECT', reason='Invalid conditional input: '+str(error))
    return report
