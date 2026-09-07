"""Exact fallback for a single retained triangle versus the four actual fan faces.

This completes a finite-query fixed shared-feature contact check. It does not
classify a contact as newly introduced relative to the ordinary baseline.
"""
from forest_exact_contact import check_triangle_contact


def certify_fan_contact(boundary, center, cycle, triangle, ids, center_id,
                        *, max_rational_bits=4096, max_work=100000):
    result = {'status': 'UNKNOWN', 'scope': 'ONE_RETAINED_TRIANGLE_FOUR_FAN_FACES',
              'triangles_checked': 0, 'arithmetic_work': 0,
              'maximum_observed_rational_bits': 0, 'baseline_novelty_checked': False}
    try:
        boundary, cycle = tuple(boundary), tuple(map(tuple, cycle))
        ids, center_id = tuple(map(tuple, ids)), tuple(center_id)
        if len(boundary) != 4 or len(cycle) != 4 or len(set(cycle)) != 4:
            raise ValueError('Expected a four-vertex actual source boundary.')
        if center_id in (*cycle, *ids):
            raise ValueError('Fresh fan center identity collides with an existing vertex.')
        for i in range(4):
            fan = (boundary[i], boundary[(i+1) % 4], center)
            fan_ids = (cycle[i], cycle[(i+1) % 4], center_id)
            proof = check_triangle_contact(fan, triangle, fan_ids, ids,
                max_rational_bits=max_rational_bits, max_work=max_work)
            result['triangles_checked'] += 1
            result['arithmetic_work'] += proof['arithmetic_work']
            result['maximum_observed_rational_bits'] = max(
                result['maximum_observed_rational_bits'], proof['maximum_observed_rational_bits'])
            if proof['status'] != 'PASS':
                result.update(status='REJECT_POLICY_CONTACT' if proof['status'] == 'REJECT_POLICY_CONTACT' else 'UNKNOWN',
                    fan_triangle_index=i, exact_contact=proof,
                    fan_triangle_coordinates=[[float(x) for x in p] for p in fan],
                    fan_triangle_ids=[list(x) for x in fan_ids],
                    retained_triangle_coordinates=[[float(x) for x in p] for p in triangle],
                    retained_triangle_ids=[list(x) for x in ids])
                return result
        result.update(status='PASS', certificate='ALL_FOUR_EXACT_TRIANGLE_CONTACT_TESTS_PASS')
        return result
    except (ValueError, TypeError, ArithmeticError, KeyError, IndexError) as error:
        result.update(status='UNKNOWN', exception_type=type(error).__name__, reason=str(error))
        return result
