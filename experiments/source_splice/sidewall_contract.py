"""Exact necessary boundary contract for the legacy linear 4D side wall.

Affine endpoint trajectories do not imply a planar swept edge. On a slab,
let e0=b0-a0 and e1=b1-a1. A diagonal point at time s is a(s)+s*e1,
whereas the ordinary edge has direction (1-s)*e0+s*e1. Their cross product
is s*(1-s)*(e1 x e0). Thus rotating edges cannot be matched by this
two-triangle side wall at every time. More coplanar triangles do not help.

Predicates below are exact over the serialized binary64 endpoints. They do
not certify the entire replacement, source trajectories between probes,
embedding, non-intersection, or runtime gluing.
"""
from fractions import Fraction
import math


def vector(value):
    if len(value) != 3 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError('sidewall position must be a finite 3-vector')
    return tuple(Fraction.from_float(float(x)) for x in value)


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def audit_sidewall(vertices4):
    if len(vertices4) != 15 or any(len(row) != 4 for row in vertices4):
        raise ValueError('expected five vertices at each of three time levels')
    for level in range(3):
        ts = [float(row[3]) for row in vertices4[level*5:level*5+5]]
        if not all(math.isfinite(t) and t == ts[0] for t in ts):
            raise ValueError('invalid time-level layout')
    if not float(vertices4[0][3]) < float(vertices4[5][3]) < float(vertices4[10][3]):
        raise ValueError('time levels must be strictly increasing')
    positions = [vector(row[:3]) for row in vertices4]
    records = []
    for slab in (0, 1):
        for edge in range(4):
            following = (edge+1) % 4
            a0, b0 = positions[5*slab+edge], positions[5*slab+following]
            a1, b1 = positions[5*(slab+1)+edge], positions[5*(slab+1)+following]
            e0, e1 = sub(b0, a0), sub(b1, a1)
            normal = cross(e0, e1)
            dot = sum(x*y for x, y in zip(e0, e1))
            parallel = all(x == 0 for x in normal)
            compatible = parallel and dot > 0
            records.append({'slab': slab, 'boundary_edge_indices': [edge, following],
                'edge_cross_product_exact': [str(x) for x in normal],
                'edge_dot_product_exact': str(dot),
                'parallel_exact': parallel, 'noncollapsing_same_direction': dot > 0 if parallel else False,
                'compatible': compatible})
    passed = all(row['compatible'] for row in records)
    return {'schema': 'binoc-linear-sidewall-contract-v1',
        'pass': passed,
        'verdict': 'PASS_LINEAR_SIDEWALL_NECESSARY_CONTRACT' if passed else 'REJECT_LINEAR_SIDEWALL_WINDOW',
        'arithmetic': 'Exact rational predicates on serialized binary64 positions; no epsilon.',
        'scope': 'Boundary of the legacy two-triangle-per-edge linear 4D wall only; not whole-window admission.',
        'edges': records, 'incompatible_edges': sum(not r['compatible'] for r in records)}


def require_window_boundary_contract(cylinder):
    """Recompute instead of trusting legacy or caller-supplied PASS metadata."""
    audit = audit_sidewall(cylinder['vertices4'])
    if not audit['pass']:
        raise ValueError('REJECT_LINEAR_SIDEWALL_WINDOW: rotating or collapsing edge; root-only admission cannot authorize a time window')
    return audit
