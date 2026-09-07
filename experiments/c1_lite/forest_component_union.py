"""Canonical, all-or-none actual-query unions of disjoint source patches.

This low-level module does not admit a graph component or a whole schedule.
The caller must bind the complete support graph, accepted single-event
certificates and identical ordinary baseline. No overlap solver is included.
"""
from collections import Counter
from itertools import combinations

import numpy as np

from forest_exact_contact import check_triangle_contact
from forest_native_campaign import array_sha, mesh_receipt
from interface_topology import original_disk
from window_geometry import certify_segment


class _Unsupported(Exception):
    def __init__(self, reason, witness):
        super().__init__(reason)
        self.witness = witness


def _integer(value):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError('Expected an integer identity/index.')
    return int(value)


def _prepare(meshes, records, expected_components):
    if len(meshes) != 5:
        raise ValueError('Expected five immutable ordinary baseline elements.')
    for v, f, tags in meshes:
        if (not all(isinstance(a, np.ndarray) and not a.flags.writeable and a.flags.c_contiguous
                    for a in (v, f, tags)) or v.ndim != 2 or v.shape[1] != 3
                or not np.issubdtype(v.dtype, np.floating) or f.ndim != 2 or f.shape[1] != 3
                or not np.issubdtype(f.dtype, np.integer) or tags.shape != (len(v),)):
            raise ValueError('Malformed or mutable baseline arrays.')
    result = []
    events, components = set(), {}
    for record in sorted(records, key=lambda r: r['event_id']):
        eid, cid, plan = record['event_id'], record['component_id'], record['plan']
        if not isinstance(eid, str) or not eid or not isinstance(cid, str) or not cid or eid in events:
            raise ValueError('Expected unique event IDs and explicit string component IDs.')
        events.add(eid)
        components.setdefault(cid, []).append(eid)
        element = _integer(plan['element'])
        if not 0 <= element < 5:
            raise ValueError('Invalid element namespace.')
        v, f, _ = meshes[element]
        cycle = tuple(_integer(x) for x in plan['boundary_actual_ids'])
        removed = tuple(_integer(x) for x in plan['removed_face_rows'])
        owners = tuple(tuple(_integer(x) for x in owner) for owner in plan['consumed_owners'])
        if (len(cycle) != 4 or len(set(cycle)) != 4 or any(not 0 <= x < len(v) for x in cycle)
                or len(removed) != 2 or len(set(removed)) != 2 or any(not 0 <= x < len(f) for x in removed)
                or not owners or len(set(owners)) != len(owners)
                or any(len(o) != 7 or o[0] != element or min(o) < 0 for o in owners)):
            raise ValueError('Malformed complete source support or raw owner equivalence class.')
        if (plan['new_center_id'] != len(v) or plan['baseline_vertex_count'] != len(v)
                or plan['baseline_face_count'] != len(f)):
            raise ValueError('Single-event plan is not compiled against this baseline layout.')
        expected_fan = tuple((cycle[i], cycle[(i+1) % 4], len(v)) for i in range(4))
        if tuple(tuple(_integer(x) for x in face) for face in plan['fan_faces']) != expected_fan:
            raise ValueError('Single-event fan differs from its declared oriented source cycle.')
        center = np.asarray(plan['center'], np.float64)
        boundary = v[list(cycle)]
        if (center.shape != (3,) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(boundary))
                or not np.array_equal(center, center.astype(np.float32).astype(np.float64))
                or not np.array_equal(boundary, boundary.astype(np.float32).astype(boundary.dtype))):
            raise ValueError('Patch coordinates must be finite, exactly represented binary32.')
        disk = original_disk(cycle, [tuple(int(x) for x in f[i]) for i in removed])
        if disk['status'] != 'PASS':
            raise ValueError('Removed source triangles do not form the declared oriented disk: '+disk['reason'])
        local = certify_segment(boundary.tolist(), boundary.tolist(), center.tolist(), center.tolist())
        if local['status'] != 'PASS':
            raise ValueError('Proposed query patch lacks the strict local graph certificate.')
        result.append({'event_id': eid, 'component_id': cid, 'element': element, 'cycle': cycle,
                       'removed': removed, 'owners': owners, 'center': center, 'boundary': boundary,
                       'plan': plan})
    if expected_components is not None:
        expected = {}
        for cid, members in expected_components.items():
            if not isinstance(cid, str) or len(set(members)) != len(members):
                raise ValueError('Malformed expected component membership.')
            expected[cid] = sorted(members)
        if expected != components:
            raise ValueError('Partial, missing or extraneous graph component membership.')
    boundary_used, faces_used, owners_used = {}, {}, {}
    ordinals = Counter()
    for row in result:
        eid, element = row['event_id'], row['element']
        for name, items, seen in (
                ('boundary identity', ((element, v) for v in row['cycle']), boundary_used),
                ('source face', ((element, f) for f in row['removed']), faces_used),
                ('raw owner', row['owners'], owners_used)):
            for item in items:
                if item in seen:
                    raise _Unsupported('Shared '+name+' needs an unsupported joint construction.',
                                       {'first_event': seen[item], 'second_event': eid, 'shared': list(item)})
                seen[item] = eid
        row['center_id'] = len(meshes[element][0])+ordinals[element]
        ordinals[element] += 1
        row['scoped_cycle'] = tuple((element, i) for i in row['cycle'])
        row['scoped_center'] = (element, row['center_id'])
        row['fan_ids'] = tuple((row['scoped_cycle'][i], row['scoped_cycle'][(i+1) % 4], row['scoped_center'])
                               for i in range(4))
        row['fan_coordinates'] = tuple((row['boundary'][i], row['boundary'][(i+1) % 4], row['center'])
                                       for i in range(4))
        row['support_low'] = np.vstack((row['boundary'], row['center'])).min(axis=0)
        row['support_high'] = np.vstack((row['boundary'], row['center'])).max(axis=0)
    return result, components


def _base_report(membership):
    return {'schema': 'forest-component-query-union-v1', 'status': 'UNKNOWN',
            'scope': 'ONE_ACTUAL_QUERY_PROVIDED_PLAN_UNION_ONLY',
            'whole_schedule_admitted': False, 'support_graph_certified': False,
            'component_membership_verified': membership is not None,
            'partial_output_published': False, 'caller_single_event_certificates_required': True,
            'caller_baseline_hash_binding_required': True}


def _interactions(meshes, prepared, components, report):
    counts = Counter(exact_triangle_pair_calls=0, triangle_pairs_excluded_by_support_aabb=0,
                     event_pairs_strict_support_aabb=0)
    checked_events = []
    for a, b in combinations(prepared, 2):
        if np.any(a['support_low'] > b['support_high']) or np.any(b['support_low'] > a['support_high']):
            # Both removed disks use only the four boundary vertices included
            # in these bounds; one strict order proof excludes all 32 pairs.
            counts['event_pairs_strict_support_aabb'] += 1
            counts['triangle_pairs_excluded_by_support_aabb'] += 32
            continue
        jobs = [('Q_Q', i, j, x, y, a['fan_ids'][i], b['fan_ids'][j])
                for i, x in enumerate(a['fan_coordinates']) for j, y in enumerate(b['fan_coordinates'])]
        for first, second, direction in ((a, b, 'Q_A_OTHER_REMOVED_B'), (b, a, 'Q_B_OTHER_REMOVED_A')):
            v, f, _ = meshes[second['element']]
            for i, tri in enumerate(first['fan_coordinates']):
                for face_id in second['removed']:
                    ids = tuple((second['element'], int(k)) for k in f[face_id])
                    jobs.append((direction, i, face_id, tri, tuple(v[f[face_id]]), first['fan_ids'][i], ids))
        for relation, i, j, x, y, ix, iy in jobs:
            proof = check_triangle_contact(x, y, ix, iy)
            counts['exact_triangle_pair_calls'] += 1
            counts[relation+'_checked'] += 1
            if proof['status'] != 'PASS':
                report.update(status=('REJECT_POLICY_CONTACT' if proof['status'] == 'REJECT_POLICY_CONTACT'
                                      else 'UNKNOWN_PAIR_INTERACTION'),
                    reason='A cross-event interaction is forbidden or not certified.',
                    failing_pair={'event_a': a['event_id'], 'event_b': b['event_id'],
                        'component_a': a['component_id'], 'component_b': b['component_id'],
                        'relation': relation, 'first_triangle': i, 'second_triangle': j,
                        'exact_contact': proof}, counts=dict(counts),
                    new_contact_relative_to_baseline_proven=False)
                return False
        checked_events.append([a['event_id'], b['event_id']])
    expected_pairs = len(prepared)*(len(prepared)-1)//2
    if counts['triangle_pairs_excluded_by_support_aabb']+counts['exact_triangle_pair_calls'] != 32*expected_pairs:
        raise ValueError('Cross-event triangle-pair denominator is incomplete.')
    report.update(status='PASS_PAIR_INTERACTIONS', event_count=len(prepared), component_count=len(components),
                  expected_event_pairs=len(prepared)*(len(prepared)-1)//2, counts=dict(counts),
                  expected_triangle_pairs=32*expected_pairs,
                  exact_checked_event_pairs=checked_events, components=components,
                  reason='All supplied cross-event fan/fan and fan/other-removed-face pairs were certified.')
    return True


def check_pair_interactions(meshes, records, *, expected_components=None):
    report = _base_report(expected_components)
    try:
        prepared, components = _prepare(meshes, records, expected_components)
        _interactions(meshes, prepared, components, report)
    except _Unsupported as error:
        report.update(status='UNSUPPORTED_SHARED_SUPPORT', reason=str(error), witness=error.witness)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError, OverflowError) as error:
        report.update(status='UNKNOWN_INPUT_OR_PROOF', reason=type(error).__name__+': '+str(error))
    return report


def compile_union(meshes, records, *, expected_components=None):
    report = _base_report(expected_components)
    try:
        prepared, components = _prepare(meshes, records, expected_components)
        if not _interactions(meshes, prepared, components, report):
            return None, report
        baseline_receipts = mesh_receipt(meshes)
        outputs = list(meshes)
        mappings = {}
        for element in range(5):
            selected = [row for row in prepared if row['element'] == element]
            if not selected:
                continue
            v, f, tags = meshes[element]
            centers = np.asarray([row['center'] for row in selected], dtype=v.dtype)
            out_v = np.concatenate((v, centers))
            out_tags = np.concatenate((tags, np.ones(len(selected), dtype=tags.dtype)))
            fans = [np.asarray([[row['cycle'][i], row['cycle'][(i+1) % 4], row['center_id']]
                                for i in range(4)], dtype=f.dtype) for row in selected]
            out_f = np.concatenate((f, *(fan[2:] for fan in fans)))
            removed_all = []
            for ordinal, (row, fan) in enumerate(zip(selected, fans)):
                # Preserve the single-event correspondence between sector0/1
                # and the supplied removed-face order, not a new face sort.
                out_f[list(row['removed'])] = fan[:2]
                removed_all.extend(row['removed'])
                new_rows = [*row['removed'], len(f)+2*ordinal, len(f)+2*ordinal+1]
                mappings[row['event_id']] = {'event_id': row['event_id'], 'component_id': row['component_id'],
                    'element': element, 'new_center_id': row['center_id'],
                    'fan_face_rows_by_sector': new_rows, 'fan_actual_vertex_ids': fan.tolist(),
                    'source_face_rows': list(row['removed']), 'consumed_owners': [list(x) for x in row['owners']]}
                if not np.array_equal(out_f[new_rows], fan):
                    raise ValueError('Canonical union fan placement disagrees with its event mapping.')
            if array_sha(out_v[:len(v)]) != array_sha(v) or array_sha(out_tags[:len(tags)]) != array_sha(tags):
                raise ValueError('Union changed original vertices or tags.')
            left = 0
            for right in (*sorted(removed_all), len(f)):
                if array_sha(out_f[left:right]) != array_sha(f[left:right]):
                    raise ValueError('Union changed a retained original face row.')
                left = right+1
            for array in (out_v, out_f, out_tags):
                array.flags.writeable = False
            outputs[element] = (out_v, out_f, out_tags)
        if mesh_receipt(meshes) != baseline_receipts:
            raise ValueError('Union evaluation mutated the original immutable baseline.')
        report.update(status='PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED',
            provided_records_all_or_none=True, canonical_event_order=[row['event_id'] for row in prepared],
            source_faces_consumed_once=sum(len(row['removed']) for row in prepared),
            raw_owners_consumed_once=sum(len(row['owners']) for row in prepared),
            old_vertices_and_tags_byte_identical=True, retained_original_face_rows_byte_identical=True,
            original_baseline_unchanged=True, event_mapping=mappings,
            component_mapping={cid: {'events': members, 'event_mapping_keys': members} for cid, members in components.items()},
            baseline=baseline_receipts, output=mesh_receipt(outputs),
            reason='Canonical actual arrays constructed for all supplied records; caller still owns graph and schedule admission.')
        return tuple(outputs), report
    except _Unsupported as error:
        report.update(status='UNSUPPORTED_SHARED_SUPPORT', reason=str(error), witness=error.witness)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError, OverflowError) as error:
        report.update(status='UNKNOWN_INPUT_OR_PROOF', reason=type(error).__name__+': '+str(error))
    return None, report
