"""Independent ordinary source authorization and the shared array executor.

The resolver consumes the complete actual ledgers and an immutable WindowSpec.
It does not import a production resolver, accept pre-resolved support, use
coordinates to match identities, or cache across query snapshots. Geometry and
the source-derived center are deliberately supplied by the common contract in
the caller. A ValueError is a policy/input refusal, not an experiment verdict.

This is the bounded single-element/two-triangle/four-boundary-vertex reference,
not a replacement for the Forest batch path or a general mesh editing library.
"""

import numpy as np


WORK_COUNTER_DEFINITIONS = {
    "vertex_records_visited": "Actual vertex-ledger rows read while indexing.",
    "source_key_normalizations": "SourceVID values normalized, including requests.",
    "owner_records_visited": "Actual owner-ledger rows read while indexing.",
    "face_rows_visited": "Actual face-array rows read for owner integrity or support identity.",
    "owner_face_integrity_checks": "Exact owner-row triangle versus actual face comparisons.",
    "indices_built": "Source identity, owner-to-face, and face-to-owner indices constructed.",
    "indices_reused": "Previously constructed query indices reused; this cold path uses zero.",
    "source_identity_lookups": "Requested boundary/source-triangle identity lookups.",
    "owner_lookups": "Requested authorization-object lookups in the owner index.",
    "coverage_face_slots_checked": "Face slots checked for nonempty complete owner evidence.",
    "expected_faces_resolved": "Source triangles resolved to actual oriented vertex triples.",
    "removed_faces_checked": "Retired face rows compared with independently resolved triangles.",
    "owner_cover_checks": "Complete output-owner sets compared with the authorization request.",
    "schedule_points_visited": "Singleton schedule rows considered for this query.",
    "schedule_segments_visited": "Open schedule rows considered for this query.",
    "array_bytes_copied": "Explicit ndarray payload bytes copied/cast/assigned by assemble; excludes allocation and Python objects.",
}


def _count(counters, key, amount=1):
    if counters is not None:
        counters[key] = counters.get(key, 0) + amount


def _source_identity(value, counters):
    """Unoriented source edge with its two (element, source vertex) endpoints."""
    _count(counters, "source_key_normalizations")
    if isinstance(value, str):
        halves = value.split("|")
        if len(halves) != 2:
            raise ValueError("SourceVID must contain two endpoint pairs.")
        endpoints = [tuple(int(v) for v in half.split(":")) for half in halves]
        if any(len(endpoint) != 2 for endpoint in endpoints):
            raise ValueError("SourceVID endpoint must have two integers.")
    else:
        row = tuple(int(v) for v in value)
        if len(row) != 4:
            raise ValueError("SourceVID must have four integers.")
        endpoints = [row[0:2], row[2:4]]
    endpoints.sort()
    return endpoints[0] + endpoints[1]


def _oriented_triangle(face):
    """Choose the smallest cyclic order; a reversal remains a different face."""
    a, b, c = (int(v) for v in face)
    return min((a, b, c), (b, c, a), (c, a, b))


def _select_request(spec, tau, counters):
    # Independently implement the singleton-before-open-cell schedule contract.
    for time, owners, source_faces in spec.points:
        _count(counters, "schedule_points_visited")
        if tau == time:
            return owners, source_faces
    for start, end, owners, source_faces in spec.segments:
        _count(counters, "schedule_segments_visited")
        if start < tau < end:
            return owners, source_faces
    raise ValueError("Query has no declared source request.")


def plain_resolve(mesh, vertex_ledger, owner_ledger, spec, tau, counters=None):
    """Return (boundary IDs, sorted retired rows, sorted complete owners).

    Every raw owner denotes a separate authorization object. Copies of one
    logical source, if desired, must already have been expanded by the request
    contract. In particular, authorizing A never implicitly authorizes B merely
    because they share a final face. Ledger completeness is a shared premise;
    all supplied owner evidence is nevertheless checked against actual faces.
    """
    vertices, faces, _tags = mesh
    if (vertex_ledger.shape != (len(vertices), 4) or
            owner_ledger.ndim != 2 or owner_ledger.shape[1] != 11):
        raise ValueError("Actual identity ledgers have invalid dimensions.")
    if counters is not None:
        for key in WORK_COUNTER_DEFINITIONS:
            counters.setdefault(key, 0)

    # Mark duplicated SourceVIDs without choosing one actual vertex by geometry.
    actual_by_source = {}
    _count(counters, "indices_built")
    for actual_id, source_row in enumerate(vertex_ledger):
        _count(counters, "vertex_records_visited")
        key = _source_identity(source_row, counters)
        if key in actual_by_source:
            actual_by_source[key] = None
        else:
            actual_by_source[key] = actual_id

    def actual_id(source):
        key = _source_identity(source, counters)
        _count(counters, "source_identity_lookups")
        result = actual_by_source.get(key)
        if result is None:
            raise ValueError("Requested SourceVID is absent or ambiguous.")
        return result

    cycle = tuple(actual_id(source) for source in spec.cycle)
    requested, source_faces = _select_request(spec, tau, counters)
    expected = set()
    for source_face in source_faces:
        expected.add(_oriented_triangle(tuple(actual_id(source) for source in source_face)))
        _count(counters, "expected_faces_resolved")
    if len(expected) != 2:
        raise ValueError("Request must identify two distinct oriented source triangles.")

    # A complete face-indexed adjacency list is independently built from raw
    # owner records. No owner-to-face structure is borrowed from the other arm.
    output_by_owner = {}
    owners_by_output = [[] for _ in range(len(faces))]
    _count(counters, "indices_built", 2)
    for owner_row in owner_ledger:
        _count(counters, "owner_records_visited")
        owner = tuple(int(v) for v in owner_row[0:7])
        output_id = int(owner_row[7])
        if owner in output_by_owner:
            raise ValueError("An owner occurs more than once in the actual ledger.")
        if not 0 <= output_id < len(faces):
            raise ValueError("An owner references an unavailable actual face.")
        recorded_triangle = tuple(int(v) for v in owner_row[8:11])
        actual_triangle = tuple(int(v) for v in faces[output_id])
        _count(counters, "face_rows_visited")
        _count(counters, "owner_face_integrity_checks")
        if recorded_triangle != actual_triangle:
            raise ValueError("An owner's recorded triangle disagrees with its actual face.")
        output_by_owner[owner] = output_id
        owners_by_output[output_id].append(owner)

    for evidence in owners_by_output:
        _count(counters, "coverage_face_slots_checked")
        if not evidence:
            raise ValueError("An actual face has no raw owner evidence.")

    requested = tuple(tuple(int(v) for v in owner) for owner in requested)
    removed_ids = set()
    for owner in requested:
        _count(counters, "owner_lookups")
        if owner not in output_by_owner:
            raise ValueError("A requested owner was not emitted in this query.")
        removed_ids.add(output_by_owner[owner])
    removed = tuple(sorted(removed_ids))
    if len(removed) != 2:
        raise ValueError("Requested owners do not retire exactly two actual faces.")
    actual_support = set()
    for output_id in removed:
        actual_support.add(_oriented_triangle(faces[output_id]))
        _count(counters, "face_rows_visited")
        _count(counters, "removed_faces_checked")
    if actual_support != expected:
        raise ValueError("Owner support differs from independently resolved source triangles.")

    consumed = tuple(sorted(owner for output_id in removed
                            for owner in owners_by_output[output_id]))
    _count(counters, "owner_cover_checks")
    # Sorted lists preserve multiplicity, including repeated request owners.
    if consumed != tuple(sorted(requested)):
        raise ValueError("The request omits or duplicates an owner of a retired face.")
    return cycle, removed, consumed


def assemble(mesh, cycle, removed, center, counters=None):
    """Execute the common four-face fan after authorization and geometry PASS.

    This is the local array layout used by the frozen propose_actual function:
    its first two fan rows replace the sorted retired rows and its final two
    rows append. Inputs are not changed. The executor performs no resolution,
    geometry checking, center selection, or policy decision of its own.
    """
    vertices, faces, tags = mesh
    center = np.asarray(center)
    center_row = center[None, :].astype(vertices.dtype)
    new_vertices = np.concatenate((vertices, center_row))
    center_tag = np.asarray([1], dtype=tags.dtype)
    new_tags = np.concatenate((tags, center_tag))
    fan = np.asarray([(cycle[i], cycle[(i + 1) % 4], len(vertices))
                      for i in range(4)], dtype=faces.dtype)
    copied_faces = faces.copy()
    new_faces = np.concatenate((copied_faces, fan[2:]))
    new_faces[list(removed)] = fan[:2]
    copied = (center_row.nbytes + new_vertices.nbytes + new_tags.nbytes +
              copied_faces.nbytes + new_faces.nbytes + fan[:2].nbytes)
    _count(counters, "array_bytes_copied", int(copied))
    return new_vertices, new_faces, new_tags
