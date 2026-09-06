"""Pure, independent E2 source-identity and array-contract validator.

No mesher, cache, plan compiler, coordinate welding, or filesystem I/O is used.
The supplied production trace must be complete: an array validator cannot
discover raw owner records that its trace producer omitted. PASS does not
certify exterior geometry, temporal continuity, or runtime admission.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json

import numpy as np


class ContractError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def canonical_vid(value):
    """Accept ordered ABI (node, group, node, group), or SourceVID token."""
    if isinstance(value, str):
        try:
            value = tuple(int(x) for part in value.split('|') for x in part.split(':'))
        except ValueError as error:
            raise ContractError('Invalid SourceVID token.') from error
    row = tuple(value)
    _require(len(row) == 4 and all(isinstance(x, (int, np.integer)) and
             not isinstance(x, (bool, np.bool_)) for x in row),
             'SourceVID requires four integers.')
    row = tuple(int(x) for x in row)
    _require(all(x >= 0 for x in row), 'SourceVID components must be nonnegative.')
    a, b = row[:2], row[2:]
    return (*a, *b) if a <= b else (*b, *a)


def oriented_face(row):
    row = tuple(int(x) for x in row)
    _require(len(row) == 3, 'A face requires three indices.')
    return min(row, row[1:] + row[:1], row[2:] + row[:2])


def _integer_matrix(value, columns, name):
    array = np.asarray(value)
    _require(array.ndim == 2 and array.shape[1] == columns,
             f'{name} requires shape (n, {columns}).')
    _require(array.dtype.kind in 'iu', f'{name} requires integer dtype.')
    return array


def _mesh(value, name):
    _require(len(value) == 3, f'{name} requires (vertices, faces, tags).')
    vertices, faces, tags = (np.asarray(x) for x in value)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3 and
             vertices.dtype.kind == 'f' and vertices.dtype.itemsize in (4, 8),
             f'{name} vertices require binary32/64 shape (n, 3).')
    _require(bool(np.all(np.isfinite(vertices))), f'{name} has nonfinite vertices.')
    faces = _integer_matrix(faces, 3, name + ' faces')
    _require(tags.ndim == 1 and len(tags) == len(vertices) and
             tags.dtype.kind in 'iub', f'{name} tags have invalid shape or dtype.')
    _require(not faces.size or (int(faces.min()) >= 0 and
             int(faces.max()) < len(vertices)), f'{name} face index out of range.')
    return vertices, faces, tags


def _same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _owners(values, name):
    result = []
    for value in values:
        row = tuple(value)
        _require(len(row) == 7 and all(isinstance(x, (int, np.integer)) and
                 not isinstance(x, (bool, np.bool_)) for x in row),
                 f'{name} requires seven-integer owner rows.')
        row = tuple(int(x) for x in row)
        _require(all(x >= 0 for x in row), f'{name} has negative owner component.')
        result.append(row)
    return result


def _nonzero_triangle(points):
    p = [tuple(Fraction.from_float(float(x)) for x in row) for row in points]
    u = tuple(b-a for a, b in zip(p[0], p[1]))
    v = tuple(b-a for a, b in zip(p[0], p[2]))
    return any(x != 0 for x in (u[1]*v[2]-u[2]*v[1],
                               u[2]*v[0]-u[0]*v[2],
                               u[0]*v[1]-u[1]*v[0]))


def _digest_rows(rows):
    payload = json.dumps(sorted(rows), separators=(',', ':')).encode('ascii')
    return hashlib.sha256(payload).hexdigest()


def validate_e2_runtime(baseline, result, vertex_ledger, owner_ledger,
                        boundary_cycle, suppression_owners, *, consumed_owners,
                        expected_center=None):
    """Return JSON-safe PASS/STOP; never infer authorized removals from a diff.

    vertex_ledger: actual ordered VID array (n_baseline_vertices, 4).
    owner_ledger: complete raw owner array (n_raw, 11): owner7, final face row,
      and canonical oriented baseline face3. Multiple owners may share a row.
    consumed_owners: independently observed consumption sequence; duplicates
      must remain present, not be converted to a set before this call.
    Replacement contract: sorted authorized face rows receive fan 0 and 1;
      fan 2 and 3 are appended. Every other baseline face row is untouched.
    expected_center is optional; its omission is explicitly reported.
    """
    report = {
        'schema': 'c1-lite-e2-runtime-identity-array-validation-v1',
        'scope': 'IDENTITY_ARRAY_CONTRACT_ONLY',
        'pass': False, 'status': 'STOP_IDENTITY_ARRAY_CONTRACT',
        'checks': {}, 'errors': [],
        'geometry_admission': 'NOT_CERTIFIED',
        'temporal_continuity': 'NOT_CERTIFIED',
        'trace_assumption': 'Complete production owner/vertex ledgers supplied by caller.',
        'expected_center_checked': expected_center is not None,
    }
    checks = report['checks']
    try:
        bv, bf, bt = _mesh(baseline, 'Baseline')
        rv, rf, rt = _mesh(result, 'Result')
        vids = _integer_matrix(vertex_ledger, 4, 'Vertex ledger')
        owners = _integer_matrix(owner_ledger, 11, 'Owner ledger')
        _require(len(vids) == len(bv), 'Vertex ledger must cover every baseline vertex ID.')
        _require(rv.dtype == bv.dtype and rf.dtype == bf.dtype and rt.dtype == bt.dtype,
                 'Output dtypes changed.')
        _require(len(rv) == len(bv)+1 and len(rf) == len(bf)+2,
                 'Expected exactly one appended vertex and net two appended faces.')
        checks['shape_dtype_and_finite'] = True

        mapping = defaultdict(list)
        for global_id, row in enumerate(vids):
            mapping[canonical_vid(row)].append(global_id)
        _require(len(boundary_cycle) == 4, 'Frozen boundary cycle must contain four SourceVIDs.')
        cycle = [canonical_vid(row) for row in boundary_cycle]
        _require(len(set(cycle)) == 4, 'Frozen boundary contains canonical aliases.')
        _require(all(len(mapping[key]) == 1 for key in cycle),
                 'Boundary SourceVID missing or canonical alias maps to multiple global IDs.')
        boundary_ids = [mapping[key][0] for key in cycle]
        report['boundary_global_ids'] = boundary_ids
        checks['boundary_identity_unique_no_coordinate_lookup'] = True

        owner_to_face = {}
        face_to_owners = defaultdict(set)
        for row in owners:
            owner = _owners([row[:7]], 'Owner ledger')[0]
            face_id = int(row[7])
            _require(0 <= face_id < len(bf), 'Owner ledger final face ID out of range.')
            face = tuple(int(x) for x in row[8:])
            _require(face == oriented_face(face) == oriented_face(bf[face_id]),
                     'Owner ledger face indices disagree with canonical baseline face.')
            _require(owner not in owner_to_face, 'Owner ledger duplicates a raw owner.')
            owner_to_face[owner] = face_id
            face_to_owners[face_id].add(owner)
        _require(set(face_to_owners) == set(range(len(bf))),
                 'Owner ledger does not cover every baseline face row.')
        requested_list = _owners(suppression_owners, 'Suppression request')
        requested = set(requested_list)
        _require(len(requested) == len(requested_list), 'Suppression request repeats an owner.')
        _require(requested and requested <= set(owner_to_face),
                 'Suppression request has missing/untraced owners.')
        removed_rows = sorted({owner_to_face[owner] for owner in requested})
        _require(len(removed_rows) == 2, 'Frozen owner request must resolve exactly two baseline faces.')
        expected_owners = set().union(*(face_to_owners[i] for i in removed_rows))
        _require(requested == expected_owners, 'Suppression request omits raw owner replicas.')
        _require(not isinstance(consumed_owners, (set, frozenset)),
                 'Actual consumption must preserve multiplicity, not be a set.')
        consumed = _owners(consumed_owners, 'Actual consumption')
        _require(Counter(consumed) == Counter({owner: 1 for owner in expected_owners}),
                 'Actual owner consumption is not exact-once for the complete authorized set.')
        report.update(expected_removed_face_rows=removed_rows,
                      expected_raw_owner_count=len(expected_owners),
                      expected_owner_sha256=_digest_rows(expected_owners),
                      consumed_owner_sha256=_digest_rows(consumed))
        checks['owners_authorize_exactly_two_faces_and_all_replicas_consumed_once'] = True

        directed = Counter()
        undirected = Counter()
        for index in removed_rows:
            face = tuple(int(x) for x in bf[index])
            _require(len(set(face)) == 3, 'Authorized baseline face repeats a vertex.')
            for a, b in zip(face, face[1:] + face[:1]):
                directed[a, b] += 1
                undirected[tuple(sorted((a, b)))] += 1
        outer = {edge for edge, count in directed.items()
                 if undirected[tuple(sorted(edge))] == 1}
        expected_outer = set(zip(boundary_ids, boundary_ids[1:] + boundary_ids[:1]))
        internal = [edge for edge, count in undirected.items() if count == 2]
        _require(len(undirected) == 5 and len(internal) == 1 and outer == expected_outer,
                 'Authorized source faces do not have the frozen oriented quad boundary.')
        a, b = internal[0]
        _require(directed[a, b] == directed[b, a] == 1,
                 'Authorized source faces have inconsistent internal diagonal orientation.')
        checks['authorized_patch_matches_frozen_oriented_boundary'] = True

        _require(_same_bits(rv[:len(bv)], bv), 'Original vertex prefix changed bits or IDs.')
        _require(_same_bits(rt[:len(bt)], bt), 'Original tag prefix changed bits or IDs.')
        external = [i for i in range(len(bf)) if i not in removed_rows]
        _require(_same_bits(rf[external], bf[external]), 'External face row changed.')
        checks['all_old_vertex_tag_bits_and_external_face_rows_unchanged'] = True

        center_id = len(bv)
        fan_rows = removed_rows + [len(bf), len(bf)+1]
        expected_fan = [(center_id, boundary_ids[i], boundary_ids[(i+1) % 4])
                        for i in range(4)]
        _require(all(oriented_face(rf[row]) == oriented_face(face)
                     for row, face in zip(fan_rows, expected_fan)),
                 'Replacement rows do not contain the four prescribed oriented fan faces.')
        counts = Counter(tuple(sorted(int(x) for x in face)) for face in rf)
        _require(all(counts[tuple(sorted(face))] == 1 for face in expected_fan),
                 'Replacement fan contains a duplicate face.')
        _require(all(_nonzero_triangle(rv[rf[row]]) for row in fan_rows),
                 'Replacement fan has an exactly zero-area triangle.')
        checks['one_new_center_four_correct_unique_exact_nonzero_faces'] = True
        report['replacement_face_rows'] = fan_rows
        if expected_center is not None:
            expected = np.asarray(expected_center, dtype=rv.dtype)
            _require(expected.shape == (3,) and np.all(np.isfinite(expected)),
                     'Expected center must be a finite 3-vector.')
            _require(_same_bits(rv[center_id], expected), 'Actual center differs from expected center bits.')
            checks['expected_center_bits'] = True
        report['pass'] = True
        report['status'] = 'PASS_IDENTITY_ARRAY_CONTRACT_ONLY'
    except (ContractError, ValueError, TypeError, IndexError, OverflowError) as error:
        report['errors'].append(str(error))
    return report
