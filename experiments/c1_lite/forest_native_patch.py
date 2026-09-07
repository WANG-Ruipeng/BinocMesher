"""Actual-query multi-element source patches, without whole-mesh proposals.

Snapshots must be immutable arrays owned by the caller.  Complete ledger
validation is cached once per snapshot; source resolution then uses a lossless
two-face view with every raw owner, through the original runtime resolver.
PASS is one requested query only, never an all-time or same-root-group claim.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import importlib
import json
import time

import numpy as np


class _PolicyViolation(ValueError):
    pass


def _modules():
    # The driver selects the frozen native package before this first call.
    return (importlib.import_module('binocmesher.window_runtime'),
            importlib.import_module('binocmesher.window_geometry_runtime'))


@dataclass(frozen=True)
class NativeSpec:
    element: int
    lower: F
    root: F
    upper: F
    cycle: tuple
    anchors: tuple
    segments: tuple
    points: tuple
    source_digest: str
    cache_digest: str

    def cell(self, tau):
        tau = F(tau)
        for t, owners, faces in self.points:
            if tau == t:
                return owners, faces
        for a, b, owners, faces in self.segments:
            if a < tau < b:
                return owners, faces
        raise ValueError('Query outside declared source cells.')

    def center(self, tau):
        wr, _ = _modules()
        return wr.WindowSpec.center(self, F(tau))


def spec_from_source(document):
    wr, _ = _modules()
    lower, root, upper = (wr._fraction(document['levels'][k])
                          for k in ('lower', 'root', 'upper'))
    if not 0 <= lower < root < upper:
        raise ValueError('Invalid window bounds.')
    cycle = tuple(wr.source_key(x) for x in document['boundary_cycle'])
    if len(cycle) != 4 or len(set(cycle)) != 4 or any(min(x) < 0 for x in cycle):
        raise ValueError('Expected four canonical nonnegative boundary IDs.')
    anchors = tuple(tuple(wr._fraction(x) for x in document['anchors'][k]['position'])
                    for k in ('lower', 'root', 'upper'))
    if any(len(p) != 3 or any(max(x.numerator.bit_length(), x.denominator.bit_length()) > 4096
                             for x in p) for p in anchors):
        raise ValueError('Invalid bounded exact anchors.')
    elements = set()
    def cell(row):
        owners = tuple(tuple(int(x) for x in o) for o in row['owners'])
        faces = tuple(tuple(wr.source_key(x) for x in f) for f in row['source_faces'])
        if (not owners or len(owners) > 1024 or len(set(owners)) != len(owners)
                or any(len(o) != 7 or min(o) < 0 or o[0] > 4 for o in owners)
                or len(faces) != 2 or any(len(f) != 3 or len(set(f)) != 3
                    or not set(f) <= set(cycle) for f in faces)
                or tuple(wr.source_key(x) for x in row['boundary_cycle']) != cycle):
            raise ValueError('Malformed multi-element owner/face source contract.')
        elements.update(o[0] for o in owners)
        return owners, faces
    segments = tuple((wr._fraction(r['t0']), wr._fraction(r['t1']), *cell(r))
                     for r in document['segments'])
    points = tuple((wr._fraction(r['time']), *cell(r)) for r in document['breakpoint_points'])
    if (len(elements) != 1 or not 1 <= len(segments) <= 128
            or segments[0][0] != lower or segments[-1][1] != upper
            or any(a >= b or a < root < b for a, b, *_ in segments)
            or any(segments[i][1] != segments[i+1][0] for i in range(len(segments)-1))
            or len(points) != len(segments)+1
            or {t for t, *_ in points} != {t for a, b, *_ in segments for t in (a, b)}):
        raise ValueError('Source has inconsistent element or missing temporal cells.')
    element = next(iter(elements))
    if 'element' in document and int(document['element']) != element:
        raise ValueError('Declared element differs from original owner identity.')
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return NativeSpec(element, lower, root, upper, cycle, anchors, segments, points,
                      hashlib.sha256(encoded).hexdigest(), str(document['cache_input_sha256']))


def _records(rows):
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    dtype = np.dtype([(str(i), np.int64) for i in range(rows.shape[1])])
    return rows.view(dtype).reshape(-1)


def _canonical_vids(rows):
    rows = np.asarray(rows, np.int64).copy()
    swap = (rows[:, 0] > rows[:, 2]) | ((rows[:, 0] == rows[:, 2]) & (rows[:, 1] > rows[:, 3]))
    rows[swap] = rows[swap][:, [2, 3, 0, 1]]
    return rows


def _lookup(records, order, query):
    key = _records(np.asarray(query, np.int64).reshape(1, -1))[0]
    lo, hi = np.searchsorted(records, key, side='left'), np.searchsorted(records, key, side='right')
    return order[int(lo):int(hi)]


def _snapshot_cache(snapshot):
    if int(snapshot.get('identity_status', 0)) != 1:
        raise ValueError('Native identity ledger did not report complete success.')
    if (snapshot.get('source_vid_encoding_version') != 2
            or snapshot.get('source_vid_encoding') != 'ORIGINAL_EFFECTIVE_SOURCE_VID'):
        raise ValueError('Original effective SourceVID encoding is unverified; merger-normalized keys are not source identities.')
    shifts = snapshot.get('source_vid_shifts')
    if (not isinstance(shifts, np.ndarray) or shifts.shape != (5, 4)
            or shifts.dtype != np.dtype(np.int32) or shifts.flags.writeable or np.any(shifts < 0)):
        raise ValueError('Missing immutable int32 five-element SourceVID shift binding.')
    meshes = snapshot['meshes']
    if len(meshes) != 5 or len(snapshot['vertex_ledgers']) != 5 or len(snapshot['owner_ledgers']) != 5:
        raise ValueError('Expected all five actual elements and identity ledgers.')
    arrays = [a for mesh in meshes for a in mesh]
    arrays += list(snapshot['vertex_ledgers']) + list(snapshot['owner_ledgers'])
    arrays.append(shifts)
    if any(not isinstance(a, np.ndarray) or a.flags.writeable for a in arrays):
        raise ValueError('Snapshot arrays must be immutable before index caching.')
    binding = tuple((id(a), a.shape, a.dtype.str) for a in arrays)
    cached = snapshot.get('_forest_native_patch_cache')
    if cached is not None:
        if cached['binding'] != binding:
            raise ValueError('Snapshot array binding changed after index validation.')
        return cached
    for v, f, tags in meshes:
        if (v.ndim != 2 or v.shape[1] != 3 or not np.issubdtype(v.dtype, np.floating)
                or not np.all(np.isfinite(v)) or f.ndim != 2 or f.shape[1] != 3
                or not np.issubdtype(f.dtype, np.integer) or np.any(f < 0) or np.any(f >= len(v))
                or tags.shape != (len(v),)):
            raise ValueError('Malformed native mesh arrays.')
        if not np.array_equal(v, v.astype(np.float32).astype(v.dtype)):
            raise ValueError('Native baseline contains non-binary32 coordinates.')
    cached = {'binding': binding, 'elements': {}}
    snapshot['_forest_native_patch_cache'] = cached
    return cached


def _element_index(snapshot, element, cache):
    if element in cache['elements']:
        return cache['elements'][element]
    v, f, _ = snapshot['meshes'][element]
    vids, owners = snapshot['vertex_ledgers'][element], snapshot['owner_ledgers'][element]
    if (vids.shape != (len(v), 4) or owners.ndim != 2 or owners.shape[1] != 11
            or not np.issubdtype(vids.dtype, np.integer) or not np.issubdtype(owners.dtype, np.integer)
            or np.any(vids < 0) or np.any(owners < 0)
            or np.any(owners[:, 0] != element) or np.any(owners[:, 7] >= len(f))):
        raise ValueError('Malformed complete native element ledgers.')
    # Vectorized checks cover every row, not only selected source owners.
    if not np.array_equal(owners[:, 8:], f[owners[:, 7]]):
        raise ValueError('Native owner face labels disagree with actual faces.')
    owner_records = _records(owners[:, :7])
    owner_order = np.argsort(owner_records, kind='stable')
    owner_records = owner_records[owner_order]
    if len(owner_records) > 1 and np.any(owner_records[1:] == owner_records[:-1]):
        raise ValueError('Duplicate raw owner identity in complete native ledger.')
    face_order = np.argsort(owners[:, 7], kind='stable')
    face_labels = owners[face_order, 7]
    if not np.array_equal(np.unique(face_labels), np.arange(len(f))):
        raise ValueError('Some actual native faces lack complete owner evidence.')
    vertex_records = _records(_canonical_vids(vids))
    vertex_order = np.argsort(vertex_records, kind='stable')
    index = dict(owner_records=owner_records, owner_order=owner_order,
                 vertex_records=vertex_records[vertex_order], vertex_order=vertex_order,
                 face_order=face_order, face_labels=face_labels)
    cache['elements'][element] = index
    return index


def _resolve(snapshot, spec, tau, index):
    wr, _ = _modules()
    v, f, tags = snapshot['meshes'][spec.element]
    vids, owners = snapshot['vertex_ledgers'][spec.element], snapshot['owner_ledgers'][spec.element]
    requested, _ = spec.cell(tau)
    actual = []
    for key in spec.cycle:
        hits = _lookup(index['vertex_records'], index['vertex_order'], key)
        if len(hits) != 1:
            raise _PolicyViolation('Boundary SourceVID missing or ambiguous in complete native ledger.')
        actual.append(int(hits[0]))
    source_rows = []
    for owner in requested:
        hits = _lookup(index['owner_records'], index['owner_order'], owner)
        if len(hits) != 1:
            raise _PolicyViolation('Requested source owner was not emitted in actual native output.')
        source_rows.append(int(owners[hits[0], 7]))
    removed = tuple(sorted(set(source_rows)))
    if len(removed) != 2:
        raise _PolicyViolation('Actual source owners do not select exactly two source faces.')
    all_rows = []
    for face in removed:
        lo = np.searchsorted(index['face_labels'], face, side='left')
        hi = np.searchsorted(index['face_labels'], face, side='right')
        all_rows.extend(index['face_order'][lo:hi].tolist())
    # This reduction is lossless: global source identity uniqueness and full
    # ledger validity have been checked, and ALL owners of both faces remain.
    local_ids = {a: i for i, a in enumerate(actual)}
    if any(int(x) not in local_ids for row in f[list(removed)] for x in row):
        raise _PolicyViolation('Selected native source face has an undeclared vertex.')
    local_faces = np.asarray([[local_ids[int(x)] for x in f[i]] for i in removed], dtype=f.dtype)
    local_owners = owners[all_rows].copy()
    face_map = {a: i for i, a in enumerate(removed)}
    for row in local_owners:
        row[7] = face_map[int(row[7])]
        row[8:] = local_faces[int(row[7])]
    try:
        cycle, local_removed, consumed = wr.resolve_support(
            (v[actual], local_faces, tags[actual]), vids[actual], local_owners, spec, tau)
    except ValueError as error:
        # All complete ledger structure and source uniqueness were validated
        # above; this original resolver now checks only the finite owner/face
        # equivalence contract on the lossless source view.
        raise _PolicyViolation(str(error)) from error
    return tuple(actual[i] for i in cycle), tuple(removed[i] for i in local_removed), consumed


def _interface(v, f, removed, cycle, wg):
    # Complete vertex star, not only edge neighbours; disconnected link cycles
    # and original-diagonal reuse therefore remain visible to the old checker.
    star = np.flatnonzero(np.any(np.isin(f, np.asarray(cycle)), axis=1))
    lookup = {int(x): i for i, x in enumerate(star)}
    error = wg.interface_check(f[star], {lookup[i] for i in removed}, cycle)
    if error:
        return error, {'interface_star_faces': len(star)}
    for row_id in star:
        if int(row_id) in removed:
            continue
        tri = tuple(wg._p(x) for x in v[f[row_id]])
        exact_cross = wg._cross(wg._sub(tri[1], tri[0]), wg._sub(tri[2], tri[0]))
        if not any(exact_cross):
            return 'Degenerate actual retained triangle touches source interface.', {
                'interface_star_faces': len(star), 'failing_face': int(row_id),
                'failing_face_actual_ids': [int(x) for x in f[row_id]],
                'failing_face_binary32_coordinates': v[f[row_id]].tolist(),
                'failing_face_exact_cross': [str(x) for x in exact_cross],
                'failing_face_shared_boundary_ids': sorted(set(int(x) for x in f[row_id]) & set(cycle)),
                'failure_scope': 'Pre-existing retained interface degeneracy violates this fixed admission policy; not a claim of a newly introduced defect.'}
    return None, {'interface_star_faces': len(star)}


def audit_query(snapshot, spec, tau, full_exterior=True, *, exact_contact_fallback=False):
    started = time.perf_counter()
    report = {'status': 'UNKNOWN', 'scope': 'ONE_ACTUAL_QUERY_ONLY',
              'continuous_window_admitted': False, 'same_root_group_admitted': False,
              'element': spec.element, 'tau': str(F(tau)), 'source_digest': spec.source_digest,
              'native_owner_element_remapped': False, 'retained_faces_checked': 0,
              'certificate_counts': {}, 'reason': None,
              'exact_contact_fallback_enabled': bool(exact_contact_fallback),
              'exact_contact_arithmetic_work': 0}
    stage = 'snapshot_and_ledger_binding'
    try:
        wr, wg = _modules()
        tau = F(tau)
        if not spec.lower < tau < spec.upper:
            report.update(status='PASS_BASELINE_ONLY', reason='Declared inactive/outside-or-endpoint policy.')
            return None, report
        cache = _snapshot_cache(snapshot)
        report.update(source_vid_encoding_version=2,
                      source_vid_encoding='ORIGINAL_EFFECTIVE_SOURCE_VID',
                      source_vid_shifts=snapshot['source_vid_shifts'].tolist(),
                      source_vid_shifts_applied_by_python=False)
        index = _element_index(snapshot, spec.element, cache)
        stage = 'actual_source_ownership'
        cycle, removed, consumed = _resolve(snapshot, spec, tau, index)
        report.update(boundary_actual_ids=list(cycle), replaced_face_rows=list(removed),
                      consumed_owners=[list(x) for x in consumed])
        v, f, tags = snapshot['meshes'][spec.element]
        center = spec.center(tau)
        boundary = tuple(wg._p(v[i]) for i in cycle)
        c = wg._p(center)
        stage = 'actual_local_graph'
        turns = [wg._cross2(wg._sub(boundary[(i+1) % 4], boundary[i]),
                           wg._sub(boundary[(i+2) % 4], boundary[(i+1) % 4])) for i in range(4)]
        orientation = 1 if turns[0] > 0 else -1
        inward = [orientation*wg._cross2(wg._sub(boundary[(i+1) % 4], boundary[i]),
                                        wg._sub(c, boundary[i])) for i in range(4)]
        report['local_exact'] = {'turns': [str(x) for x in turns], 'inward': [str(x) for x in inward],
                                 'boundary': v[list(cycle)].tolist(), 'center': center.tolist()}
        if not all(orientation*x > 0 for x in turns):
            raise _PolicyViolation('Actual source boundary is not a strict convex XY graph.')
        if not all(x > 0 for x in inward):
            raise _PolicyViolation('Actual rounded center is not strictly inside source boundary.')
        stage = 'actual_interface'
        error, interface = _interface(v, f, removed, cycle, wg)
        report.update(interface)
        if error:
            raise _PolicyViolation(error)
        if not full_exterior:
            report.update(status='PASS_LOCAL_ONLY', reason='Full five-element retained geometry was not requested.')
            return None, report
        stage = 'actual_five_element_exterior'
        patch = np.asarray([v[i] for i in cycle]+[center])
        low, high = patch.min(axis=0), patch.max(axis=0)
        scoped_cycle = tuple((spec.element, i) for i in cycle)
        counts = Counter()
        for element, (ev, ef, _) in enumerate(snapshot['meshes']):
            checked = 0
            for start in range(0, len(ef), 65536):
                stop = min(start+65536, len(ef))
                coords = ev[ef[start:stop]]
                separated = np.any(coords.min(axis=1) > high, axis=1) | np.any(coords.max(axis=1) < low, axis=1)
                retained = np.ones(stop-start, dtype=bool)
                if element == spec.element:
                    for face_id in removed:
                        if start <= face_id < stop:
                            retained[face_id-start] = False
                broad = int(np.count_nonzero(separated & retained))
                counts['STRICT_ACTUAL_AABB'] += broad
                checked += broad
                for offset in np.flatnonzero(~separated & retained):
                    face_id = start+int(offset)
                    ids = tuple((element, int(x)) for x in ef[face_id])
                    kind, error = wg._contact(boundary, c, scoped_cycle,
                        tuple(wg._p(p) for p in coords[offset]), ids, orientation)
                    if kind is None:
                        if exact_contact_fallback:
                            from forest_fan_contact import certify_fan_contact
                            fallback = certify_fan_contact(boundary, c, scoped_cycle,
                                tuple(wg._p(p) for p in coords[offset]), ids, (spec.element, len(v)))
                            report['exact_contact_arithmetic_work'] += fallback['arithmetic_work']
                            if fallback['status'] == 'PASS':
                                kind = 'EXACT_FOUR_FAN_TRIANGLE_CONTACT'
                            else:
                                report.update(exact_contact_fallback=fallback,
                                    failing_element=element, failing_face=face_id,
                                    failing_ids=[list(x) for x in ids], failing_coordinates=coords[offset].tolist(),
                                    certificate_counts=dict(counts), stage=stage,
                                    retained_faces_checked=report['retained_faces_checked']+checked,
                                    insufficient_separator_reason=error)
                                if fallback['status'] == 'REJECT_POLICY_CONTACT':
                                    report['reason_code'] = 'FORBIDDEN_ACTUAL_PATCH_RETAINED_CONTACT'
                                    raise _PolicyViolation('Actual fan-retained contact violates the fixed shared-feature policy; baseline novelty was not tested.')
                                report['reason'] = 'Exact contact remains unknown under its declared arithmetic budget or input contract.'
                                return None, report
                    if kind is None:
                        report.update(reason=error, failing_element=element, failing_face=face_id,
                            failing_ids=[list(x) for x in ids], failing_coordinates=coords[offset].tolist(),
                            certificate_counts=dict(counts), stage=stage,
                            retained_faces_checked=report['retained_faces_checked']+checked)
                        return None, report
                    counts[kind] += 1
                    checked += 1
            report['retained_faces_checked'] += checked
        expected = sum(len(mesh[1]) for mesh in snapshot['meshes'])-2
        if report['retained_faces_checked'] != expected:
            raise RuntimeError('Retained denominator accounting mismatch.')
        fan = tuple((cycle[i], cycle[(i+1) % 4], len(v)) for i in range(4))
        plan = {'element': spec.element, 'removed_face_rows': removed, 'boundary_actual_ids': cycle,
                'center': tuple(float(x) for x in center), 'new_center_id': len(v), 'fan_faces': fan,
                'consumed_owners': consumed, 'source_digest': spec.source_digest,
                'baseline_vertex_count': len(v), 'baseline_face_count': len(f),
                'outside_arrays_policy': 'Original vertices/tags unchanged; replace two face rows and append two.'}
        report.update(status='PASS', reason='Actual identity, local graph, interface and every retained face certified.',
                      stage=stage, certificate_counts=dict(counts), full_exterior=True,
                      expected_retained_faces=expected, proposal_only=True)
        return plan, report
    except _PolicyViolation as error:
        report.update(status='REJECT', reason=str(error), stage=stage,
                      certified_necessary_policy_failure=True)
        return None, report
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RuntimeError) as error:
        # Internal/transport exceptions do not constitute geometric witnesses.
        report.update(status='UNKNOWN', reason=str(error), stage=stage,
                      exception_type=type(error).__name__)
        return None, report
    finally:
        report['wall_seconds'] = time.perf_counter()-started
