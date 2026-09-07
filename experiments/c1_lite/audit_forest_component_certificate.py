"""Supplemental finite-schedule component audit; no native slices or mesh files.

The archived native receipts remain responsible for actual identity/coordinates
and complete interface stars. This audit checks their coverage and bindings,
recomputes compact local/pair geometry under explicitly bound dependencies,
and replays graph decisions. It does not retrospectively claim that previously
unlisted dependencies were hash-bound during the original certification run.
"""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from forest_component_graph import build_graph, digest
from forest_exact_contact import check_triangle_contact
from forest_native_campaign import file_sha
from interface_topology import original_disk
from window_geometry import certify_segment


HERE = Path(__file__).resolve().parent
DEPENDENCIES = (
    'audit_forest_component_certificate.py', 'forest_component_graph.py',
    'forest_component_union.py', 'forest_sequence_inputs.py',
    'run_forest_component_certification.py', 'forest_native_campaign.py',
    'forest_exact_contact.py', 'interface_topology.py', 'window_geometry.py',
)
PASS = 'PASS_SUPPLEMENTAL_COMPONENT_CERTIFICATE_AUDIT'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge_bindings(*maps):
    """Conflicting provenance cannot be hidden by dict-update order."""
    result = {}
    for mapping in maps:
        for key, value in mapping.items():
            path = str(Path(key).resolve())
            require(path not in result or result[path] == value,
                    'Conflicting source/input digest: '+path)
            result[path] = value
    return result


def fraction(value):
    return F(value['numerator'], value['denominator']) if isinstance(value, dict) else F(value)


def unique(rows, key, label):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), 'Duplicate '+label)
    return result


def validate_schedule(events, queries, *, event_count=131, root_count=2,
                      active_queries_per_root=9, natural_queries_per_root=8):
    population = unique(events, 'event_id', 'event ID')
    require(len(population) == event_count, 'Incomplete all-event graph denominator')
    by_root = {}
    for event in events:
        root = event['root']
        schedule = event['schedule']
        bounds = tuple(fraction(schedule['bounds'][k]) for k in ('lower', 'root', 'upper'))
        require(bounds[0] < bounds[1] < bounds[2] and bounds[1] == F(root), 'Invalid root window')
        source = event.get('compiler', {}).get('source')
        if source:
            require(bounds == tuple(fraction(source['levels'][k]) for k in ('lower', 'root', 'upper')),
                    'Source/schedule window mismatch')
        all_queries = unique(schedule['all_queries'], 'key', 'event schedule key')
        active = {k: q for k, q in all_queries.items() if q['active']}
        require(len(active) == active_queries_per_root, 'Incomplete active event schedule')
        require(sum(q['kind'] == 'natural' for q in active.values()) == natural_queries_per_root,
                'Incomplete natural active schedule')
        roots = [q for q in active.values() if q['kind'] == 'exact_root']
        require(len(roots) == 1 and F(roots[0]['evaluation_tau']) == F(root)
                and roots[0]['time_mode'] == 'exact', 'Missing exact-root query')
        for query in all_queries.values():
            tau = F(query['evaluation_tau'])
            require(bool(query['active']) == (bounds[0] < tau < bounds[2]), 'Activation/window mismatch')
        if root in by_root:
            require(by_root[root]['bounds'] == bounds and by_root[root]['active'] == active,
                    'Events at one root have different active schedules/windows')
        else:
            by_root[root] = {'bounds': bounds, 'active': active}
    require(len(by_root) == root_count, 'Root denominator mismatch')
    for a, b in combinations(by_root.values(), 2):
        require(a['bounds'][2] < b['bounds'][0] or b['bounds'][2] < a['bounds'][0],
                'Cross-root windows are not strictly disjoint')
        require(not set(a['active']) & set(b['active']), 'Cross-root active query token overlap')
    expected = {key: (root, query) for root, row in by_root.items() for key, query in row['active'].items()}
    actual = unique([{'key': q['query']['key'], 'row': q} for q in queries], 'key', 'actual query key')
    require(set(actual) == set(expected), 'Missing or extraneous actual query')
    for key, item in actual.items():
        root, query = expected[key]
        require(item['row']['root'] == root and item['row']['query'] == query,
                'Actual query differs from the frozen schedule')
    return population, by_root


def validate_footprint(row, event, baseline):
    require(row['element'] == event['element'], 'Footprint element namespace mismatch')
    require(row['status'] == 'COMPLETE_ACTUAL_REQUESTED_SUPPORT',
            'Supplemental audit currently requires all recorded actual supports complete')
    require(row.get('full_interface_star_enumerated') is True, 'Incomplete interface star declaration')
    ids = row['boundary_actual_ids']
    require(ids and len(set(ids)) == len(ids), 'Empty or duplicate actual support identity')
    receipt = baseline[row['element']]
    require(all(type(x) is int and 0 <= x < receipt['vertices'] for x in ids), 'Actual vertex ID out of range')
    coordinates = np.asarray(row['boundary_coordinates'], np.float64)
    require(coordinates.shape == (len(ids), 3) and np.all(np.isfinite(coordinates)), 'Malformed support coordinates')
    require(np.array_equal(coordinates, coordinates.astype(np.float32).astype(np.float64)), 'Support is not represented binary32')
    source = row['source_face_rows']
    retained = row['retained_star_face_rows']
    require(len(set(source)) == len(source) and len(set(retained)) == len(retained)
            and not set(source) & set(retained), 'Source/interface face-set inconsistency')
    require(all(type(x) is int and 0 <= x < receipt['faces'] for x in source+retained), 'Face ID out of range')
    triangles = row['source_triangles']
    require(len(triangles) == len(source) and all(len(t) == 3 and set(t) <= set(ids) for t in triangles),
            'Source faces escape the declared support bound')
    owners = row['consumed_owners']
    require(owners and len(set(map(tuple, owners))) == len(owners)
            and all(len(o) == 7 and o[0] == row['element'] and all(type(x) is int and x >= 0 for x in o) for o in owners),
            'Malformed complete raw owner list')
    points = coordinates
    if row['kind'] == 'FIXED_SOURCE_AND_REPLACEMENT':
        center = np.asarray(row['center'], np.float64)
        require(len(ids) == 4 and len(source) == 2 and center.shape == (3,)
                and np.all(np.isfinite(center)) and np.array_equal(center, center.astype(np.float32).astype(np.float64)),
                'Malformed represented source/replacement bound')
        points = np.vstack((points, center))
    else:
        require(row['kind'] == 'BASELINE_BLOCKED_CURRENT_CANDIDATE_DOMAIN'
                and row.get('future_replacement_bound') is False
                and not event.get('compiler', {}).get('source'), 'Unsupported candidate-only reservation claim')
    require(np.array_equal(row['bounds'], [points.min(axis=0), points.max(axis=0)]),
            'Support AABB is not the exact extrema of its represented points')


def validate_plan(record, footprint, event, query, baseline):
    cases = [c for c in event['runtime']['cases'] if c['query']['key'] == query['key']]
    require(len(cases) == 1 and cases[0]['query'] == query and cases[0]['audit']['status'] == 'PASS',
            'Missing original independent actual-query certificate')
    plan = record['plan']
    require(plan == cases[0]['plan'], 'Archived plan differs from the original independently certified plan')
    require(plan['element'] == footprint['element']
            and plan['boundary_actual_ids'] == footprint['boundary_actual_ids']
            and sorted(plan['removed_face_rows']) == sorted(footprint['source_face_rows'])
            and sorted(map(tuple, plan['consumed_owners'])) == sorted(map(tuple, footprint['consumed_owners']))
            and plan['center'] == footprint['center'], 'Plan/actual footprint binding mismatch')
    count = baseline[plan['element']]
    require(plan['baseline_vertex_count'] == count['vertices'] == plan['new_center_id']
            and plan['baseline_face_count'] == count['faces'], 'Plan baseline layout mismatch')
    cycle = footprint['boundary_actual_ids']
    require(plan['fan_faces'] == [[cycle[i], cycle[(i+1) % 4], count['vertices']] for i in range(4)],
            'Plan changed the oriented four-fan template')
    require(original_disk(cycle, footprint['source_triangles'])['status'] == 'PASS', 'Source is not the oriented two-face disk')
    local = certify_segment(footprint['boundary_coordinates'], footprint['boundary_coordinates'],
                            footprint['center'], footprint['center'])
    require(local['status'] == 'PASS', 'Compact actual local graph recheck failed')


def recheck_pair(a, b, baseline):
    """Recheck all 16 Q-Q and 16 Q-other-removed pairs without full meshes."""
    same = a['element'] == b['element']
    shared = (same and (set(a['boundary_actual_ids']) & set(b['boundary_actual_ids'])
                         or set(a['source_face_rows']) & set(b['source_face_rows'])))
    shared = shared or set(map(tuple, a['consumed_owners'])) & set(map(tuple, b['consumed_owners']))
    if shared:
        return {'status': 'UNSUPPORTED_SHARED_SUPPORT', 'triangle_pairs_excluded': 0, 'exact_calls': 0}
    alo, ahi = np.asarray(a['bounds']); blo, bhi = np.asarray(b['bounds'])
    if np.any(alo > bhi) or np.any(blo > ahi):
        return {'status': 'PASS_PAIR_INTERACTIONS', 'triangle_pairs_excluded': 32, 'exact_calls': 0}
    def geometry(row, ordinal):
        element = row['element']
        ids = [(element, x) for x in row['boundary_actual_ids']]
        center = (element, baseline[element]['vertices']+ordinal)
        positions = {x: p for x, p in zip(ids, row['boundary_coordinates'])}
        positions[center] = row['center']
        fans = [([ids[i], ids[(i+1) % 4], center]) for i in range(4)]
        removed = [[(element, x) for x in triangle] for triangle in row['source_triangles']]
        return positions, fans, removed
    pa, qa, sa = geometry(a, 0)
    pb, qb, sb = geometry(b, 1 if same else 0)
    jobs = [(pa, x, pb, y) for x in qa for y in qb]
    jobs += [(pa, x, pb, y) for x in qa for y in sb]
    jobs += [(pb, x, pa, y) for x in qb for y in sa]
    require(len(jobs) == 32, 'Compact pair scope incomplete')
    calls = 0
    for px, ix, py, iy in jobs:
        proof = check_triangle_contact([px[x] for x in ix], [py[y] for y in iy], ix, iy)
        calls += 1
        if proof['status'] != 'PASS':
            return {'status': proof['status'], 'triangle_pairs_excluded': 0, 'exact_calls': calls}
    return {'status': 'PASS_PAIR_INTERACTIONS', 'triangle_pairs_excluded': 0, 'exact_calls': calls}


def validate_payload(events, queries, graph, summary, inventory, frozen_queries, *,
                     event_count=131, root_count=2, active_queries_per_root=9, natural_queries_per_root=8):
    population, roots = validate_schedule(events, queries, event_count=event_count, root_count=root_count,
        active_queries_per_root=active_queries_per_root, natural_queries_per_root=natural_queries_per_root)
    require(summary['query_count'] == len(queries), 'Summary query denominator mismatch')
    counts = Counter()
    for query in queries:
        key, root = query['query']['key'], query['root']
        require(query['baseline'] == frozen_queries[key]['baseline'], 'Five-element baseline receipt mismatch')
        encoding = query['identity_encoding']
        require(encoding['version'] == 2 and encoding['encoding'] == 'ORIGINAL_EFFECTIVE_SOURCE_VID',
                'Actual SourceVID encoding is not verified original v2')
        expected_events = {eid for eid, event in population.items() if event['root'] == root}
        footprints = unique(query['events'], 'event_id', 'query footprint')
        require(set(footprints) == expected_events, 'Query omitted a rejected or admitted root event')
        for eid, footprint in footprints.items():
            validate_footprint(footprint, population[eid], query['baseline'])
            counts['complete_actual_support_rows'] += 1
        selected = {eid for eid in expected_events if population[eid]['decision'] == 'ADMITTED_REQUESTED_SCHEDULE'}
        records = unique(query['certified_independent_plans'], 'event_id', 'independent plan')
        require(set(records) == selected, 'Missing or extra independently admitted plan')
        for eid, record in records.items():
            validate_plan(record, footprints[eid], population[eid], query['query'], query['baseline'])
            counts['actual_local_rechecks'] += 1
        expected_pairs = set(combinations(sorted(selected), 2))
        seen = set()
        for pair in query['pair_proofs']:
            ids = pair['events']
            require(len(ids) == 2 and ids[0] != ids[1], 'Malformed pair identity')
            identity = tuple(sorted(ids))
            require(identity in expected_pairs and identity not in seen, 'Duplicate or extraneous pair proof')
            seen.add(identity)
            result = recheck_pair(footprints[identity[0]], footprints[identity[1]], query['baseline'])
            if pair['report']['status'] == 'PASS_PAIR_INTERACTIONS':
                require(result['status'] == 'PASS_PAIR_INTERACTIONS', 'Previously passing pair failed compact exact recheck')
                if pair['report'].get('proof') == 'STRICT_ACTUAL_BINARY32_SUPPORT_AABB':
                    require(result['triangle_pairs_excluded'] == pair['report']['triangle_pairs_excluded'] == 32,
                            'Claimed strict AABB does not exclude all 32 triangle pairs')
            counts['event_pair_rechecks'] += 1
            counts['triangle_pairs_excluded_by_exact_aabb'] += result['triangle_pairs_excluded']
            counts['exact_triangle_contact_calls'] += result['exact_calls']
        require(seen == expected_pairs, 'Incomplete pair proof matrix')
    symbolic = list(inventory.get('symbolic_edges', []))
    for query in queries:
        for pair in query['pair_proofs']:
            if pair['report']['status'] != 'PASS_PAIR_INTERACTIONS':
                symbolic.append({'events': pair['events'], 'reason': 'ACTUAL_JOINT_PAIR_NOT_CERTIFIED',
                                 'cell': query['query']['key']})
    replay = build_graph(events, queries, symbolic_edges=symbolic)
    from run_forest_component_certification import decide_components
    decision = decide_components(replay, queries)
    require(replay == graph, 'Full-node graph or component decision replay mismatch')
    require(all(summary.get(k) == v for k, v in decision.items()), 'Component admission summary differs from complete replay')
    return {'counts': dict(counts), 'event_count': len(population), 'query_count': len(queries),
        'roots': {root: {'lower': str(row['bounds'][0]), 'upper': str(row['bounds'][2]),
                        'query_keys': sorted(row['active'])} for root, row in roots.items()},
        'strict_cross_root_window_disjointness': True, 'pair_matrix_complete_and_unique': True,
        'rejected_nodes_preserved_and_propagated': True, 'compact_geometry_rechecked': True,
        'graph_and_component_decisions_replayed': True, **decision}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_component_certificate(component_root, *, inputs=None):
    """Read-only audit; raises ValueError on every incomplete/inconsistent gate."""
    start = time.monotonic()
    root = Path(component_root).resolve()
    summary_path = root/'summary.json'
    summary = _read(summary_path)
    require(summary['status'] == 'PASS_FOREST_COMPONENT_REQUESTED_CERTIFICATION'
            and summary.get('final_input_verification', {}).get('status') == 'PASS'
            and summary.get('private_cache_removed') is True, 'Component run did not finish cleanly')
    from forest_sequence_inputs import load_certified_inputs
    inputs = load_certified_inputs() if inputs is None else inputs
    new_dependencies = {str(HERE/name): file_sha(HERE/name) for name in DEPENDENCIES}
    bound = merge_bindings(inputs['bindings'], summary['input_bindings_sha256'],
        summary['executed_sources_sha256'], {str(summary_path): file_sha(summary_path)}, new_dependencies)
    artifacts = [summary['support_graph'], *summary['query_artifacts']]
    artifact_paths = []
    for item in artifacts:
        path = (root/item['path']).resolve()
        require(path.is_relative_to(root) and path not in artifact_paths, 'Escaping or duplicate component artifact')
        artifact_paths.append(path)
        bound = merge_bindings(bound, {str(path): item['sha256']})
    for path, expected in bound.items():
        require(file_sha(path) == expected, 'Bound artifact or source changed: '+path)
    inventory_candidates = [path for path, sha in summary['input_bindings_sha256'].items()
                            if sha == summary['source_support_inventory_sha256']]
    require(len(inventory_candidates) == 1, 'Inventory has no unique hash-bound input')
    inventory = _read(inventory_candidates[0])
    require(inventory['status'] == 'PASS_COMPLETE_FIXED_SOURCE_CANDIDATE_INVENTORY'
            and set(unique(inventory['events'], 'event_id', 'inventory event')) == {e['event_id'] for e in inputs['events']},
            'Incomplete all-event candidate inventory')
    queries = [_read(root/item['path']) for item in summary['query_artifacts']]
    result = validate_payload(inputs['events'], queries, _read(root/summary['support_graph']['path']),
                              summary, inventory, inputs['queries'])
    for path, expected in bound.items():
        require(file_sha(path) == expected, 'Input changed during supplemental audit: '+path)
    old_dependencies = merge_bindings(summary['input_bindings_sha256'], summary['executed_sources_sha256'])
    return {'schema': 'forest-component-supplemental-audit-v1', 'status': PASS,
        'component_summary_sha256': file_sha(summary_path), **result,
        'audited_files_sha256': bound, 'supplemental_executed_sources_sha256': new_dependencies,
        'dependencies_first_explicitly_bound_in_this_recheck': sorted(set(new_dependencies)-set(old_dependencies)),
        'historical_missing_hashes_retroactively_claimed': False, 'native_slice_calls': 0,
        'sequence_executed': False, 'wall_seconds': time.monotonic()-start,
        'scope': 'Finite complete-matrix/graph and compact-geometry supplemental recheck. Actual coordinate, owner and full-star provenance is inherited from hash-bound native receipts. Not a new native run, all-time certificate, or combined-sequence receipt.'}


def verify_component_certificate(component_root, *, inputs=None):
    return audit_component_certificate(component_root, inputs=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(args.component.resolve()),
            'Supplemental output must be new and outside the frozen component directory')
    result = verify_component_certificate(args.component)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'event_count', 'query_count', 'counts',
                                           'jointly_admitted_events', 'admitted_components')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
