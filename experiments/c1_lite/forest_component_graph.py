"""Conservative finite-schedule support graph; never an all-real-time claim."""
from collections import Counter
from itertools import combinations
import hashlib
import json

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def actual_footprints(snapshot, requests, tau):
    """requests: event_id, element and spec OR complete candidate_actual_owners.

    Candidate-only requests are permanently baseline-blocked reservations,
    not invented source contracts. Missing resolution is explicitly unknown.
    One full face scan per element constructs every selected vertex star.
    """
    from forest_native_patch import _snapshot_cache, _element_index, _resolve, _lookup
    cache = _snapshot_cache(snapshot)
    rows = []
    for request in requests:
        element = int(request['element'])
        row = {'event_id': request['event_id'], 'element': element, 'status': 'UNKNOWN'}
        try:
            index = _element_index(snapshot, element, cache)
            v, f, _ = snapshot['meshes'][element]
            owners = snapshot['owner_ledgers'][element]
            spec = request.get('spec')
            if spec is not None:
                cycle, removed, consumed = _resolve(snapshot, spec, tau, index)
                center = np.asarray(spec.center(tau), dtype=v.dtype)
                if center.shape != (3,) or not np.all(np.isfinite(center)):
                    raise ValueError('Nonfinite or malformed actual rounded center.')
                coords = np.concatenate((v[list(cycle)], center[None]))
                boundary = list(cycle)
                source_rows = list(removed)
                row.update(kind='FIXED_SOURCE_AND_REPLACEMENT', center=center.tolist(),
                           boundary_actual_ids=boundary, source_face_rows=source_rows,
                           consumed_owners=[list(x) for x in consumed])
            else:
                requested = request['candidate_actual_owners']
                if not requested or len(set(map(tuple, requested))) != len(requested):
                    raise ValueError('Empty or duplicate blocked candidate domain.')
                found = []
                for owner in requested:
                    hits = _lookup(index['owner_records'], index['owner_order'], owner)
                    if len(hits) != 1:
                        raise ValueError('Blocked candidate owner missing or ambiguous.')
                    found.append(int(owners[int(hits[0]), 7]))
                source_rows = sorted(set(found))
                boundary = sorted(set(map(int, f[source_rows].reshape(-1))))
                coords = v[boundary]
                row.update(kind='BASELINE_BLOCKED_CURRENT_CANDIDATE_DOMAIN',
                           boundary_actual_ids=boundary, source_face_rows=source_rows,
                           consumed_owners=[list(x) for x in requested],
                           future_replacement_bound=False)
            row.update(status='COMPLETE_ACTUAL_REQUESTED_SUPPORT',
                       bounds=[coords.min(axis=0).tolist(), coords.max(axis=0).tolist()],
                       boundary_coordinates=v[boundary].tolist(),
                       source_triangles=f[source_rows].tolist(),
                       retained_star_face_rows=[])
        except ValueError as error:
            row['reason'] = str(error)
        rows.append(row)
    # A complete boundary vertex star, not a nearest-neighbour/ROI approximation.
    for element in sorted({r['element'] for r in rows if r['status'].startswith('COMPLETE')}):
        selected = [r for r in rows if r['element'] == element and r['status'].startswith('COMPLETE')]
        _, faces, _ = snapshot['meshes'][element]
        vertices = np.asarray(sorted({x for r in selected for x in r['boundary_actual_ids']}))
        blocks = []
        for start in range(0, len(faces), 65536):
            hits = np.flatnonzero(np.any(np.isin(faces[start:start+65536], vertices), axis=1))
            blocks.append(hits+start)
        star = np.concatenate(blocks) if blocks else np.asarray([], np.int64)
        for row in selected:
            face_rows = star[np.any(np.isin(faces[star], row['boundary_actual_ids']), axis=1)]
            row['retained_star_face_rows'] = sorted(set(map(int, face_rows))-set(row['source_face_rows']))
            row['full_interface_star_enumerated'] = True
    return rows


def build_graph(events, query_footprints, *, symbolic_edges=()):
    """Missing finite-query evidence joins the whole root, not an empty node."""
    population = {e['event_id']: e for e in events}
    if len(population) != len(events) or not population:
        raise ValueError('Empty or duplicate event population.')
    edges = {}
    def edge(a, b, reason, query):
        if a == b or a not in population or b not in population:
            raise ValueError('Invalid graph endpoints.')
        if population[a]['root'] != population[b]['root']:
            raise ValueError('Disjoint root windows must not be silently combined.')
        key = tuple(sorted((a, b)))
        entry = edges.setdefault(key, {'events': list(key), 'reasons': {}})
        # Keep first witness per reason; full input footprints remain archived.
        entry['reasons'].setdefault(reason, query)
    for item in symbolic_edges:
        edge(*item['events'], item['reason'], item.get('cell', 'SOURCE_CELLS'))
    coverage = {eid: set() for eid in population}
    for query in query_footprints:
        root, key = query['root'], query['query']['key']
        expected = {eid for eid, e in population.items() if e['root'] == root}
        rows = {r['event_id']: r for r in query['events']}
        if set(rows) != expected or len(rows) != len(query['events']):
            raise ValueError('Footprint query changed complete root population.')
        for eid in rows:
            coverage[eid].add(key)
        for a, b in combinations(sorted(rows), 2):
            x, y = rows[a], rows[b]
            if any(r['status'] != 'COMPLETE_ACTUAL_REQUESTED_SUPPORT' for r in (x, y)):
                edge(a, b, 'UNKNOWN_ACTUAL_SUPPORT_CONSERVATIVE_ROOT_JOIN', key)
                continue
            if set(map(tuple, x['consumed_owners'])) & set(map(tuple, y['consumed_owners'])):
                edge(a, b, 'RAW_OWNER_OVERLAP', key)
            if x['element'] == y['element']:
                sx, sy = set(x['source_face_rows']), set(y['source_face_rows'])
                rx, ry = set(x['retained_star_face_rows']), set(y['retained_star_face_rows'])
                if sx & sy:
                    edge(a, b, 'SOURCE_FACE_OVERLAP', key)
                if set(x['boundary_actual_ids']) & set(y['boundary_actual_ids']):
                    edge(a, b, 'BOUNDARY_ID_OR_INTERFACE_LINK_OVERLAP', key)
                if rx & ry:
                    edge(a, b, 'RETAINED_INTERFACE_FACE_OVERLAP', key)
                if sx & ry or sy & rx:
                    edge(a, b, 'SOURCE_RETAINED_INTERFACE_DEPENDENCY', key)
            lowx, highx = np.asarray(x['bounds'])
            lowy, highy = np.asarray(y['bounds'])
            if np.all(lowx <= highy) and np.all(lowy <= highx):
                edge(a, b, 'ACTUAL_SUPPORT_AABB_POSSIBLE_INTERACTION', key)
    for eid, event in population.items():
        required = {q['key'] for q in event['schedule']['all_queries'] if q['active']}
        if coverage[eid] != required:
            raise ValueError('Incomplete actual active schedule support for '+eid)
    neighbors = {eid: set() for eid in population}
    for a, b in edges:
        neighbors[a].add(b); neighbors[b].add(a)
    components, visited = [], set()
    for eid in sorted(population):
        if eid in visited:
            continue
        pending, members = [eid], set()
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current); pending.extend(neighbors[current]-members)
        visited.update(members)
        members = sorted(members)
        rejected = [i for i in members if population[i]['decision'] != 'ADMITTED_REQUESTED_SCHEDULE']
        components.append({'component_id': 'component-'+digest(members)[:16],
            'root': population[eid]['root'], 'events': members,
            'independently_rejected_members': rejected,
            'decision': 'FAIL_CLOSED_REJECTED_MEMBER' if rejected else 'PENDING_JOINT_CERTIFICATION'})
    return {'schema': 'forest-support-component-graph-v1',
        'status': 'COMPLETE_CONSERVATIVE_REQUESTED_SUPPORT_GRAPH',
        'scope': 'Current fixed source/candidate domain, finite requested schedule; not all-real-time or minimal contact graph.',
        'event_count': len(events), 'events_sha256': digest(sorted(population)),
        'edges': list(edges.values()), 'components': components,
        'edge_reason_counts': dict(Counter(reason for e in edges.values() for reason in e['reasons'])),
        'unknown_actual_support_rows': sum(r['status'] != 'COMPLETE_ACTUAL_REQUESTED_SUPPORT'
                                        for q in query_footprints for r in q['events']),
        'root_counts': {root: {'events': sum(e['root'] == root for e in events),
            'components': sum(c['root'] == root for c in components),
            'pending_components': sum(c['root'] == root and not c['independently_rejected_members'] for c in components)}
            for root in sorted({e['root'] for e in events})}}
