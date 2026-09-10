"""New, explicitly defined model-sampling reconstruction, NOT historical GL.

Only the 2D slice is required to be a graph. XY=(u,v), so rank-deficient
spatial source maps are allowed. No production mesher/checker is imported.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np


SEEDS = (1103, 2207, 3301)
FAMILIES = ('affine', 'bilinear', 'rational', 'strong_rational')
DOMAINS = {
    'whole_square': np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
    'cropped_asymmetric_hexagon': np.array([
        [.08, .16], [.71, .04], [.97, .36], [.83, .92], [.30, .98], [.03, .65]]),
}
CORNERS = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bilinear(values, q):
    q = np.atleast_2d(np.asarray(q, dtype=float))
    u, v = q.T
    a, b, c, d = np.asarray(values, dtype=float)
    result = a*(1-u)*(1-v) + b*u*(1-v) + c*u*v + d*(1-u)*v
    du = (b-a)*(1-v) + (c-d)*v
    dv = (d-a)*(1-u) + (c-b)*u
    return result, du, dv


@dataclass(frozen=True)
class GraphModel:
    lower: np.ndarray
    upper: np.ndarray
    zlower: np.ndarray
    zupper: np.ndarray
    tau: float = 0.

    def validate(self):
        arrays = (self.lower, self.upper, self.zlower, self.zupper)
        require(all(np.asarray(a).shape == (4,) for a in arrays), 'four corners required')
        require(all(np.isfinite(a).all() for a in arrays), 'nonfinite source')
        require(np.isfinite(self.tau), 'nonfinite tau')
        require(np.all(self.lower < self.tau) and np.all(self.tau < self.upper),
                'unsupported full-square active domain')

    def evaluate(self, q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        require(np.isfinite(q).all() and np.all(q >= -1e-12) and
                np.all(q <= 1+1e-12), 'parameter outside containing square')
        lo, lu, lv = bilinear(self.lower, q)
        hi, hu, hv = bilinear(self.upper, q)
        zl, zlu, zlv = bilinear(self.zlower, q)
        zu, zuu, zuv = bilinear(self.zupper, q)
        delta = hi-lo
        require(np.all(delta > 0), 'nonpositive temporal denominator')
        s = (self.tau-lo)/delta
        require(np.all(s >= -1e-12) and np.all(s <= 1+1e-12), 'invalid active lift')
        su = -(lu+s*(hu-lu))/delta
        sv = -(lv+s*(hv-lv))/delta
        z = zl+s*(zu-zl)
        dzdu = zlu+s*(zuu-zlu)+(zu-zl)*su
        dzdv = zlv+s*(zuv-zlv)+(zu-zl)*sv
        xyz = np.column_stack((q, z))
        normal = np.column_stack((-dzdu, -dzdv, np.ones(len(q))))
        jac = np.linalg.norm(normal, axis=1)
        require(np.isfinite(xyz).all() and np.isfinite(jac).all(), 'nonfinite graph')
        return xyz, normal/jac[:, None], jac

    def receipt(self):
        return {name: getattr(self, name).tolist()
                for name in ('lower', 'upper', 'zlower', 'zupper')} | {'tau': self.tau}


def polygon_area(q):
    q = np.asarray(q)
    return .5*float(np.sum(q[:, 0]*np.roll(q[:, 1], -1) -
                            np.roll(q[:, 0], -1)*q[:, 1]))


def area_centroid(q):
    q = np.asarray(q)
    after = np.roll(q, -1, axis=0)
    cross = q[:, 0]*after[:, 1] - after[:, 0]*q[:, 1]
    require(float(cross.sum()) > 0, 'polygon must be CCW')
    return np.sum((q+after)*cross[:, None], axis=0)/(3*cross.sum())


def base_faces(m, root=0):
    order = list(range(root, m))+list(range(root))
    return np.array([[order[0], order[i], order[i+1]] for i in range(1, m-1)])


def validate_mesh(vertices, faces, boundary):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    m = len(boundary)
    require(np.isfinite(vertices).all(), 'nonfinite mesh')
    require(np.array_equal(vertices[:m], boundary), 'mesh boundary changed')
    require(faces.ndim == 2 and faces.shape[1] == 3 and np.all(faces >= 0)
            and np.all(faces < len(vertices)), 'invalid face indices')
    require(set(faces.ravel()) == set(range(len(vertices))), 'unused vertex')
    xy = vertices[faces, :2]
    cross = np.cross(xy[:, 1]-xy[:, 0], xy[:, 2]-xy[:, 0])
    require(np.all(cross > 1e-13), 'degenerate or reversed parameter face')
    expected_area = polygon_area(boundary[:, :2])
    require(abs(cross.sum()/2-expected_area) < 1e-11, 'parameter area changed')
    counts = {}
    orientation = {}
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            counts[key] = counts.get(key, 0)+1
            orientation[key] = orientation.get(key, 0)+(1 if a < b else -1)
    expected_edges = {tuple(sorted((i, (i+1) % m))) for i in range(m)}
    require({edge for edge, count in counts.items() if count == 1} == expected_edges,
            'boundary edge mismatch')
    require(all(count in (1, 2) for count in counts.values()), 'nonmanifold edge')
    require(all(orientation[e] == 0 for e, c in counts.items() if c == 2),
            'inconsistent interior edge orientation')
    k = len(vertices)-m
    require(len(faces) == m+2*k-2, 'disk face budget mismatch')
    require(len(vertices)-len(counts)+len(faces) == 1, 'disk Euler mismatch')
    return {'vertices': len(vertices), 'faces': len(faces), 'internal_vertices': k,
            'parameter_area': expected_area, 'minimum_parameter_double_area': float(cross.min())}


def make_cases():
    cases = []
    for domain_name, polygon in DOMAINS.items():
        edge = np.roll(polygon, -1, axis=0)-polygon
        require(np.all(np.cross(edge, np.roll(edge, -1, axis=0)) > 0), 'nonconvex domain')
        for family in FAMILIES:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                lo = -rng.uniform(.25, 1.25, 4)
                hi = rng.uniform(.25, 1.25, 4)
                a, b, c = rng.uniform(-.25, .25, 3)
                plane = a*CORNERS[:, 0]+b*CORNERS[:, 1]+c
                if family == 'affine':
                    # Exactly a plane at tau=0 despite nonconstant time fields.
                    zl, zu = plane+.4*lo, plane+.4*hi
                elif family == 'bilinear':
                    height = plane+rng.uniform(.5, 1.2)*CORNERS[:, 0]*CORNERS[:, 1]
                    zl, zu = height.copy(), height.copy()  # spatial rank two is valid
                elif family == 'rational':
                    zl, zu = rng.uniform(-.7, .7, (2, 4))
                else:
                    lo = -rng.uniform(.08, 1.6, 4)
                    hi = rng.uniform(.08, 1.6, 4)
                    zl, zu = rng.uniform(-1.5, 1.5, (2, 4))
                model = GraphModel(lo, hi, zl, zu)
                model.validate()
                cases.append({'id': f'{domain_name}__{family}__{seed}',
                              'domain': domain_name, 'family': family,
                              'seed': seed, 'polygon': polygon.copy(), 'model': model})
    return cases


def construct(model, polygon, kind, k=0):
    """Construction has no access to evaluator/reference samples."""
    start = perf_counter()
    boundary = model.evaluate(polygon)[0]
    boundary_seconds = perf_counter()-start
    start = perf_counter()
    vertices = list(boundary.copy())
    faces = list(base_faces(len(polygon), 1 if kind == 'boundary_root1' else 0))
    queries = candidates = 0
    selection = []
    if kind in ('xyz_mean_fan', 'rebuilt_lifted_vertex_mean', 'rebuilt_lifted_area_centroid'):
        if kind == 'xyz_mean_fan':
            point = np.mean(boundary, axis=0)
        else:
            q = np.mean(polygon, axis=0) if kind.endswith('vertex_mean') else area_centroid(polygon)
            point = model.evaluate(q)[0][0]
            queries += 1
        vertices.append(point)
        faces = [[i, (i+1) % len(polygon), len(polygon)] for i in range(len(polygon))]
        selection.append(point[:2].tolist())
    elif kind in ('pure_pl', 'uniform_model', 'greedy_model'):
        score_cache = {}
        for _ in range(k):
            scores = []
            for face in faces:
                tri = np.asarray([vertices[i] for i in face])
                area = abs(np.cross(tri[1, :2]-tri[0, :2], tri[2, :2]-tri[0, :2]))/2
                if kind != 'greedy_model':
                    scores.append(area)
                else:
                    key = tuple(face)
                    if key not in score_cache:
                        q = tri[:, :2].mean(axis=0)
                        point = model.evaluate(q)[0][0]
                        queries += 1
                        candidates += 1
                        score_cache[key] = (abs(point[2]-tri[:, 2].mean())*area, point)
                    scores.append(score_cache[key][0])
            chosen = int(np.argmax(scores))
            old = faces.pop(chosen)
            tri = np.asarray([vertices[i] for i in old])
            if kind == 'pure_pl':
                point = tri.mean(axis=0)
            elif kind == 'uniform_model':
                point = model.evaluate(tri[:, :2].mean(axis=0))[0][0]
                queries += 1
            else:
                point = score_cache[tuple(old)][1]
            selection.append(point[:2].tolist())
            index = len(vertices)
            vertices.append(point)
            faces.extend([[int(old[i]), int(old[(i+1) % 3]), index] for i in range(3)])
    else:
        require(kind in ('boundary_root0', 'boundary_root1'), 'unknown method')
    elapsed = perf_counter()-start
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=int)
    checks = validate_mesh(vertices, faces, boundary)
    name = f'{kind}_k{k}' if kind in ('pure_pl', 'uniform_model', 'greedy_model') else kind
    return {'name': name, 'vertices_array': vertices, 'faces_array': faces,
            'triangles': vertices[faces], 'selection_uv': selection, 'checks': checks,
            'budget': {'boundary_model_point_evaluations': len(polygon),
                       'interior_model_point_evaluations': queries,
                       'candidate_model_point_evaluations': candidates,
                       'candidate_queries_are_subset_of_interior_queries': True,
                       'original_target_queries': 0,
                       'reference_queries_used_for_construction': 0,
                       'boundary_setup_seconds': boundary_seconds,
                       'method_construction_seconds': elapsed,
                       'timing_note': 'single Python construction, not production benchmark'}}


def all_methods(model, polygon):
    methods = [construct(model, polygon, kind) for kind in (
        'boundary_root0', 'boundary_root1', 'xyz_mean_fan',
        'rebuilt_lifted_vertex_mean', 'rebuilt_lifted_area_centroid')]
    methods.extend(construct(model, polygon, kind, k)
                   for k in (1, 2, 4) for kind in ('pure_pl', 'uniform_model', 'greedy_model'))
    return methods


def parameter_triangles(polygon, resolution):
    """Conforming uniform subdivision of the root-0 parameter triangulation."""
    require(resolution >= 1, 'positive subdivision resolution required')
    result = []
    for a, b, c in polygon[base_faces(len(polygon))]:
        def point(i, j):
            return a+(b-a)*(i/resolution)+(c-a)*(j/resolution)
        for i in range(resolution):
            for j in range(resolution-i):
                result.append([point(i, j), point(i+1, j), point(i, j+1)])
                if i+j < resolution-1:
                    result.append([point(i+1, j), point(i+1, j+1), point(i, j+1)])
    return np.asarray(result)


def quadrature(polygon, resolution):
    triangles = parameter_triangles(polygon, resolution)
    weights = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])/2
    require(np.all(weights > 0), 'invalid quadrature')
    return triangles.mean(axis=1), weights


def reference_triangles(model, polygon, resolution):
    uv = parameter_triangles(polygon, resolution)
    return model.evaluate(uv.reshape(-1, 2))[0].reshape(-1, 3, 3)
