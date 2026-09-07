"""Dynamic opaque-element form of the frozen finite-query geometry wiring.

This is an explicit static derivative, not monkeypatching or padding. The
unchanged source/identity/local/contact algorithms come from forest_native_patch,
forest_component_graph.actual_footprints and forest_component_union. Only element
bounds/shapes are parameterized; raw-only input and index-capacity guards are
explicit. Five-element reports/arrays remain parity-testable. No new CCD,
projection choice, source selector, quantization rule or overlap solver is used.

Dependencies: the caller must select the frozen binocmesher Python package
before the first spec/geometry call and hash it with the pure certificate
modules. PASS still covers one actual query, never an unbound whole window.
"""
from pathlib import Path
import sys

C1_LITE = Path(__file__).resolve().parents[1]/'c1_lite'
if str(C1_LITE) not in sys.path:
    sys.path.insert(0, str(C1_LITE))

from itertools import combinations
from forest_exact_contact import check_triangle_contact
from forest_native_campaign import array_sha, mesh_receipt
from interface_topology import original_disk
from window_geometry import certify_segment

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


def spec_from_source(document, n_elements):
    if type(n_elements) is not int or not 1 <= n_elements <= 64:
        raise ValueError('Expected the actual bounded opaque-element count.')
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
                or any(len(o) != 7 or min(o) < 0 or o[0] >= n_elements for o in owners)
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
    n_elements = len(snapshot['meshes'])
    if not 1 <= n_elements <= 64 or snapshot.get('n_elements', n_elements) != n_elements:
        raise ValueError('Actual opaque-element count differs from the snapshot arrays.')
    if snapshot.get('extra_smooth', False):
        raise ValueError('The frozen geometry admission profile is raw-only, not extra_smooth.')
    if int(snapshot.get('identity_status', 0)) != 1:
        raise ValueError('Native identity ledger did not report complete success.')
    if (snapshot.get('source_vid_encoding_version') != 2
            or snapshot.get('source_vid_encoding') != 'ORIGINAL_EFFECTIVE_SOURCE_VID'):
        raise ValueError('Original effective SourceVID encoding is unverified; merger-normalized keys are not source identities.')
    shifts = snapshot.get('source_vid_shifts')
    if (not isinstance(shifts, np.ndarray) or shifts.shape != (n_elements, 4)
            or shifts.dtype != np.dtype(np.int32) or shifts.flags.writeable or np.any(shifts < 0)):
        raise ValueError('Missing immutable int32 five-element SourceVID shift binding.')
    meshes = snapshot['meshes']
    if len(meshes) != n_elements or len(snapshot['vertex_ledgers']) != n_elements or len(snapshot['owner_ledgers']) != n_elements:
        raise ValueError('Expected all five actual elements and identity ledgers.')
    arrays = [a for mesh in meshes for a in mesh]
    arrays += list(snapshot['vertex_ledgers']) + list(snapshot['owner_ledgers'])
    arrays.append(shifts)
    if any(not isinstance(a, np.ndarray) or a.flags.writeable for a in arrays):
        raise ValueError('Snapshot arrays must be immutable before index caching.')
    binding = tuple((id(a), a.shape, a.dtype.str) for a in arrays)
    cached = snapshot.get('_scene_native_patch_cache')
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
    snapshot['_scene_native_patch_cache'] = cached
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
            report.update(status='PASS_LOCAL_ONLY', reason=('Full five-element retained geometry was not requested.' if len(snapshot['meshes']) == 5 else 'Full all-element retained geometry was not requested.'))
            return None, report
        stage = 'actual_five_element_exterior' if len(snapshot['meshes']) == 5 else 'actual_all_element_exterior'
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



def actual_footprints(snapshot, requests, tau):
    """requests: event_id, element and spec OR complete candidate_actual_owners.

    Candidate-only requests are permanently baseline-blocked reservations,
    not invented source contracts. Missing resolution is explicitly unknown.
    One full face scan per element constructs every selected vertex star.
    """
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

def original_identity_receipt(snapshot):
    n_elements = len(snapshot['meshes'])
    if not 1 <= n_elements <= 64 or snapshot.get('n_elements', n_elements) != n_elements:
        raise ValueError('Actual opaque-element count differs from identity snapshot.')
    if snapshot.get('identity_status') != 1:
        raise ValueError('Original SourceVID observation is not ready.')
    if (snapshot.get('source_vid_encoding_version') != 2
            or snapshot.get('source_vid_encoding') != 'ORIGINAL_EFFECTIVE_SOURCE_VID'):
        raise ValueError('Original SourceVID encoding is unverified; normalized merger keys cannot establish admission.')
    shifts = snapshot.get('source_vid_shifts')
    if (not isinstance(shifts, np.ndarray) or shifts.shape != (n_elements, 4)
            or shifts.dtype != np.int32 or shifts.flags.writeable):
        raise ValueError('Original SourceVID requires an immutable int32 five-element shift receipt.')
    return {'version': 2, 'encoding': 'ORIGINAL_EFFECTIVE_SOURCE_VID',
            'shifts_order': ['node0', 'group0', 'node1', 'group1'],
            'per_element_shifts': shifts.tolist()}



class _Unsupported(Exception):
    def __init__(self, reason, witness):
        super().__init__(reason)
        self.witness = witness


def _integer(value):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError('Expected an integer identity/index.')
    return int(value)


def _prepare(meshes, records, expected_components):
    if not 1 <= len(meshes) <= 64:
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
        if not 0 <= element < len(meshes):
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
        if row['center_id'] > np.iinfo(meshes[element][1].dtype).max:
            raise ValueError('Fresh center index exceeds the actual face dtype capacity.')
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
        for element in range(len(meshes)):
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


