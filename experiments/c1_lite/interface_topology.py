"""Restricted oriented-disk/interface tests in the source-identity quotient.

PASS needs an independent runtime-identity equivalence gate before admission.
This module does not infer geometric intersection or floating-point safety.
"""
from collections import Counter, defaultdict


def _edges(face):
    return [(face[i], face[(i+1) % len(face)]) for i in range(len(face))]


def original_disk(cycle, source_faces):
    cycle = tuple(cycle)
    faces = [tuple(face) for face in source_faces]
    if len(cycle) != 4 or len(set(cycle)) != 4 or len(faces) != 2:
        return {'status': 'REJECT', 'reason': 'Expected one four-vertex boundary and two original triangles.'}
    if any(len(face) != 3 or len(set(face)) != 3 or not set(face) <= set(cycle) for face in faces):
        return {'status': 'REJECT', 'reason': 'Original source face is malformed or has repeated/extra identities.'}
    directed = Counter(edge for face in faces for edge in _edges(face))
    boundary = Counter(_edges(cycle))
    if not all(directed[edge] == 1 for edge in boundary):
        return {'status': 'REJECT', 'reason': 'Original oriented boundary does not equal the full declared cycle.'}
    leftover = directed-boundary
    if sum(leftover.values()) != 2 or len(leftover) != 2 or not all(leftover[(b, a)] == n == 1 for (a, b), n in leftover.items()):
        return {'status': 'REJECT', 'reason': 'Original patch lacks one oppositely oriented internal diagonal.'}
    return {'status': 'PASS', 'internal_diagonal': sorted(next(iter(leftover))),
            'reason': 'Two oriented triangles form a disk with exactly the declared boundary.'}


def audit_interface(cycle, source_faces, retained, *, element=0, max_witnesses=8):
    """Require a valid retained collar: one edge attachment and a link path.

retained is the canonical-source oriented-face set after raw filtering and
suppression. Multiplicity/identity equivalence to actual arrays is NOT assumed.
"""
    disk = original_disk(cycle, source_faces)
    report = {'schema': 'c1-lite-source-interface-topology-v1', 'status': disk['status'],
              'original_disk': disk, 'coordinate_model': 'source-identity quotient',
              'runtime_identity_equivalence': 'UNKNOWN', 'witnesses': [],
              'witness_count': 0, 'boundary_edge_checks': [], 'vertex_link_checks': []}
    if disk['status'] != 'PASS':
        return report
    cycle = tuple(cycle)
    boundary = set(cycle)
    edge_counts = Counter()
    links = {v: [] for v in cycle}

    def fail(kind, detail):
        report['status'] = 'REJECT'
        report['witness_count'] += 1
        if len(report['witnesses']) < max_witnesses:
            report['witnesses'].append({'kind': kind, **detail})

    seen = set()
    for triangle in retained:
        if triangle['element'] != element:
            continue
        face = tuple(triangle['source_vertices'])
        if not set(face) & boundary:
            continue
        if len(face) != 3 or len(set(face)) != 3:
            fail('RETAINED_INTERFACE_REPEATED_ID', {'face': list(face), 'owners': triangle.get('owners', [])})
            continue
        if set(disk['internal_diagonal']) <= set(face):
            fail('RETAINED_USES_ORIGINAL_INTERNAL_DIAGONAL', {'face': list(face), 'owners': triangle.get('owners', [])})
            continue
        key = min(face[i:]+face[:i] for i in range(3))
        if key in seen:
            fail('DUPLICATE_RETAINED_IDENTITY_FACE', {'face': list(face)})
            continue
        seen.add(key)
        edge_counts.update(_edges(face))
        for v in boundary & set(face):
            j = face.index(v)
            links[v].append((face[(j+1) % 3], face[(j+2) % 3]))

    for a, b in _edges(cycle):
        item = {'edge': [a, b], 'same_direction': edge_counts[(a, b)], 'opposite_direction': edge_counts[(b, a)]}
        item['status'] = 'PASS' if item['same_direction'] == 0 and item['opposite_direction'] == 1 else 'REJECT'
        report['boundary_edge_checks'].append(item)
        if item['status'] != 'PASS':
            fail('BOUNDARY_EDGE_ATTACHMENT', item)
    for i, v in enumerate(cycle):
        pairs = links[v]
        degree = Counter(x for pair in pairs for x in pair)
        adjacency = defaultdict(set)
        for a, b in pairs:
            adjacency[a].add(b)
            adjacency[b].add(a)
        reached = set()
        pending = [next(iter(degree))] if degree else []
        while pending:
            x = pending.pop()
            if x not in reached:
                reached.add(x)
                pending.extend(adjacency[x]-reached)
        ends = {cycle[i-1], cycle[(i+1) % 4]}
        incoming = Counter(b for a, b in pairs)
        outgoing = Counter(a for a, b in pairs)
        oriented = all(
            incoming[x] == (0 if x == cycle[i-1] else 1)
            and outgoing[x] == (0 if x == cycle[(i+1) % 4] else 1)
            for x in degree)
        valid = (set(x for x, d in degree.items() if d == 1) == ends
                 and all(d == (1 if x in ends else 2) for x, d in degree.items())
                 and reached == set(degree) and oriented)
        item = {'source_vid': v, 'status': 'PASS' if valid else 'REJECT',
                'link_edges': len(pairs), 'link_vertices': len(degree),
                'directed_link_path_compatible': oriented,
                'expected_path_endpoints': sorted(ends)}
        report['vertex_link_checks'].append(item)
        if not valid:
            fail('RETAINED_LINK_NOT_ONE_PATH', item)
    report['reason'] = ('Original disk and retained collar are valid in the source quotient; a compatible fan preserves this interface topology.'
                        if report['status'] == 'PASS' else 'The restrictive source-quotient interface contract is not satisfied; no runtime collision is inferred.')
    return report
