"""Exact, bounded triangle contact for a finite native mesh query.

The intersection is the linear image of a product-of-simplices feasibility
polytope. Enumerating its basic feasible solutions handles noncoplanar,
coplanar, line and point cases without numerical tolerances. This checks a
fixed shared-feature policy, NOT whether a contact is new relative to baseline.
No native code or campaign imports this module automatically.
"""
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
import math
from numbers import Integral, Real


class _Limit(Exception):
    pass


class _Budget:
    def __init__(self, bits, work):
        if any(not isinstance(x, Integral) or isinstance(x, bool) or x < 1
               for x in (bits, work)):
            raise ValueError('Positive integer bit and work budgets required.')
        self.bits, self.limit = int(bits), int(work)
        self.work = self.peak_bits = 0

    def check(self, value):
        self.work += 1
        if self.work > self.limit:
            raise _Limit('EXACT_ARITHMETIC_WORK_BUDGET')
        n = max(value.numerator.bit_length(), value.denominator.bit_length())
        self.peak_bits = max(self.peak_bits, n)
        if n > self.bits:
            raise _Limit('RATIONAL_BIT_BUDGET')
        return value

    def add(self, a, b): return self.check(a+b)
    def sub(self, a, b): return self.check(a-b)
    def mul(self, a, b): return self.check(a*b)
    def div(self, a, b): return self.check(a/b)


def _number(value, budget):
    if isinstance(value, bool):
        raise ValueError('Boolean coordinates are unsupported.')
    if isinstance(value, F):
        result = value
    elif isinstance(value, Integral):
        result = F(int(value))
    elif isinstance(value, Real) and hasattr(value, 'as_integer_ratio'):
        try:
            numerator, denominator = value.as_integer_ratio()
            result = F(int(numerator), int(denominator))
        except (OverflowError, ValueError) as error:
            raise ValueError('Coordinates must be finite.') from error
    else:
        raise ValueError('Coordinates must be integers, binary reals or Fractions.')
    return budget.check(result)


def _triangle(values, budget):
    try:
        rows = tuple(tuple(row) for row in values)
    except TypeError as error:
        raise ValueError('Expected a (3,3) triangle.') from error
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError('Expected a (3,3) triangle.')
    return tuple(tuple(_number(x, budget) for x in row) for row in rows)


def _ids(values):
    try:
        result = tuple(tuple(row) for row in values)
    except TypeError as error:
        raise ValueError('Use three actual (element,index) identities.') from error
    if (len(result) != 3 or any(len(row) != 2 for row in result)
            or any(not isinstance(x, Integral) or isinstance(x, bool) or x < 0
                   for row in result for x in row)
            or len({row[0] for row in result}) != 1):
        raise ValueError('Use three nonnegative IDs from one element per triangle.')
    return tuple(tuple(int(x) for x in row) for row in result)


def _reduce(rows, variables, budget):
    """Exact Gauss-Jordan reduction of a small augmented matrix."""
    rows = [list(row) for row in rows]
    rank = 0
    for col in range(variables):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][col]
        rows[rank] = [budget.div(x, divisor) for x in rows[rank]]
        for i in range(len(rows)):
            if i == rank or not rows[i][col]:
                continue
            multiplier = rows[i][col]
            rows[i] = [budget.sub(x, budget.mul(multiplier, y))
                       for x, y in zip(rows[i], rows[rank])]
        rank += 1
        if rank == len(rows):
            break
    inconsistent = any(not any(row[:variables]) and row[-1] for row in rows)
    return rows[:rank], rank, bool(inconsistent)


def _sum_products(a, b, budget):
    value = F(0)
    for x, y in zip(a, b):
        value = budget.add(value, budget.mul(x, y))
    return value


def _allowed(point, shared_points, budget):
    if not shared_points:
        return False
    if len(shared_points) == 1 or shared_points[0] == shared_points[1]:
        return point == shared_points[0]
    a, b = shared_points
    delta = tuple(budget.sub(y, x) for x, y in zip(a, b))
    offset = tuple(budget.sub(x, y) for x, y in zip(point, a))
    axis = next(i for i, x in enumerate(delta) if x)
    parameter = budget.div(offset[axis], delta[axis])
    return (0 <= parameter <= 1 and
            all(x == budget.mul(parameter, y) for x, y in zip(offset, delta)))


def _strings(values):
    return [str(x) for x in values]


def check_triangle_contact(triangle_a, triangle_b, ids_a, ids_b, *,
                           max_rational_bits=4096, max_work=100000):
    """Return PASS, REJECT_POLICY_CONTACT or UNKNOWN_WITH_BUDGET.

    Each triangle has three 3D points. Finite binary floating values are
    interpreted exactly; Fractions and integers are also accepted. Every ID
    is (element, actual_vertex_index), including the proposed center's new ID.
    Permitted intersection is empty, the one shared vertex, or the segment
    between two shared vertices. Three shared IDs are outside this profile.
    Repeated IDs/degenerate triangles are allowed only with consistent points.

    Input errors raise ValueError for the caller to classify as UNKNOWN.
    Work counts bounded rational arithmetic/results, not wall-clock seconds.
    A rejection is a forbidden contact, not necessarily a *new* collision.
    """
    budget = _Budget(max_rational_bits, max_work)
    report = {
        'schema': 'forest-exact-triangle-contact-v1', 'status': 'UNKNOWN_WITH_BUDGET',
        'scope': 'ONE_ACTUAL_QUERY_FIXED_SHARED_FEATURE_POLICY',
        'new_contact_relative_to_baseline_proven': False,
        'max_rational_bits': budget.bits, 'max_work': budget.limit,
        'equality_rank': None, 'basis_count': 0, 'checked_bases': 0,
        'feasible_bases': 0, 'exhaustive_basis_search': False,
        'intersection_vertices_exact': [], 'witness': None,
    }
    try:
        a, b = _triangle(triangle_a, budget), _triangle(triangle_b, budget)
        ia, ib = _ids(ids_a), _ids(ids_b)
        positions = {}
        for identity, point in zip(ia+ib, a+b):
            if identity in positions and positions[identity] != point:
                raise ValueError('The same actual identity has inconsistent coordinates.')
            positions[identity] = point
        shared = tuple(sorted(set(ia) & set(ib)))
        if len(shared) > 2:
            raise ValueError('Three shared IDs require a different duplicate-face policy.')
        shared_points = tuple(positions[x] for x in shared)
        input_record = {'a': [_strings(p) for p in a], 'b': [_strings(p) for p in b],
                        'ids_a': [list(x) for x in ia], 'ids_b': [list(x) for x in ib]}
        report.update(input_exact=input_record,
            input_sha256=hashlib.sha256(json.dumps(input_record, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(),
            shared_ids=[list(x) for x in shared],
            allowed_feature=('empty', 'vertex', 'edge')[len(shared)])
        if any(max(p[k] for p in a) < min(p[k] for p in b) or
               max(p[k] for p in b) < min(p[k] for p in a) for k in range(3)):
            report.update(status='PASS', relation='DISJOINT', proof='STRICT_EXACT_AABB')
            return report

        # x=(lambda0..2,mu0..2): A lambda - B mu = 0, sum lambda=sum mu=1.
        equations = [list(p[k] for p in a)+[budget.sub(F(0), p[k]) for p in b]+[F(0)]
                     for k in range(3)]
        equations += [[F(1)]*3+[F(0)]*3+[F(1)], [F(0)]*3+[F(1)]*3+[F(1)]]
        reduced, rank, inconsistent = _reduce(equations, 6, budget)
        report['equality_rank'] = rank
        if inconsistent:
            report.update(status='PASS', relation='DISJOINT', proof='INCONSISTENT_EXACT_AFFINE_EQUATIONS')
            return report
        report['basis_count'] = math.comb(6, rank)
        vertices = set()
        for basis in combinations(range(6), rank):
            square = [[row[j] for j in basis]+[row[-1]] for row in reduced]
            solved, basis_rank, bad = _reduce(square, rank, budget)
            report['checked_bases'] += 1
            if bad or basis_rank != rank:
                continue
            values = [F(0)]*6
            for j, row in zip(basis, solved):
                values[j] = row[-1]
            if any(x < 0 for x in values):
                continue
            # Verify the unreduced five equations independently of elimination.
            if any(_sum_products(row[:6], values, budget) != row[-1] for row in equations):
                raise ArithmeticError('Exact basis solution failed original equations.')
            report['feasible_bases'] += 1
            point = tuple(_sum_products(values[:3], [p[k] for p in a], budget) for k in range(3))
            vertices.add(point)
            if not _allowed(point, shared_points, budget):
                report.update(status='REJECT_POLICY_CONTACT', relation='FORBIDDEN_CONTACT',
                    proof='EXACT_FEASIBLE_INTERSECTION_POINT_OUTSIDE_ALLOWED_FEATURE',
                    witness={'point_exact': _strings(point),
                        'barycentric_a_exact': _strings(values[:3]),
                        'barycentric_b_exact': _strings(values[3:]),
                        'basis_columns': list(basis),
                        'interpretation': 'Contact violates the fixed shared-feature policy; baseline comparison was not performed.'})
                report['intersection_vertices_exact'] = [_strings(p) for p in sorted(vertices)]
                return report
        report['exhaustive_basis_search'] = True
        report['intersection_vertices_exact'] = [_strings(p) for p in sorted(vertices)]
        if shared and not vertices:
            raise ArithmeticError('A consistent shared vertex has no feasible intersection basis.')
        report.update(status='PASS', relation='INTERSECTION_SUBSET_ALLOWED_FEATURE' if vertices else 'DISJOINT',
            proof='ALL_BASIC_FEASIBLE_INTERSECTION_IMAGES_LIE_IN_ALLOWED_CONVEX_FEATURE' if vertices
                  else 'NO_NONNEGATIVE_BASIC_FEASIBLE_SOLUTION')
        return report
    except _Limit as error:
        report.update(status='UNKNOWN_WITH_BUDGET', reason=str(error), proof=None)
        return report
    finally:
        report['arithmetic_work'] = budget.work
        report['maximum_observed_rational_bits'] = budget.peak_bits
