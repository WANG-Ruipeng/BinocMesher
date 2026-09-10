"""A: bounded source-authorization mechanisms, with trusted complete ledgers.

All owners in this fixture are independently authorized source objects. In
particular the extra owner of face zero is NOT a cache replica that selection
should expand automatically. Geometry, source coordinates, and the current
array replacement backend are fixed. These synthetic checks establish contract
behavior, not emitter completeness, native-scene usefulness, or novelty.

Importing this module does not execute a test. The ordered campaign calls run().
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json

import numpy as np

from binocmesher.window_geometry_runtime import check_actual_patch
from binocmesher.window_runtime import WindowSpec, propose_actual, resolve_support

from .plain_reference import assemble, plain_resolve


PROGRESS = {}
TAU = Fraction(1)
OWNER_SEMANTICS = "INDEPENDENT_AUTHORIZATION_OBJECTS_NOT_CACHE_REPLICAS"


def _require(condition, message):
    if not condition:
        raise RuntimeError("A mechanism mismatch: " + message)


def _oriented(row):
    row = tuple(row)
    return min(row[i:] + row[:i] for i in range(3))


def _fixture(*, multiple_owners=False, complete_request=True,
             coincident_distinct_vid=False, select_coincident_vid=False):
    """Adapt the frozen runtime square/collar fixture with explicit semantics."""
    vertices = np.asarray([
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
        [-1, -1, 0], [3, -1, 0], [3, 3, 0], [-1, 3, 0],
    ], np.float64)
    rows = [(0, 1, 2), (0, 2, 3)]
    for i in range(4):
        j = (i + 1) % 4
        rows.extend(((j, i, i + 4), (j, i + 4, j + 4)))
    faces = np.asarray([_oriented(row) for row in rows], np.int32)
    vids = np.asarray([(i, 0, i + 100, 0) for i in range(len(vertices))], np.int32)
    owner_rows = [(0, 0, 0, i, 0, 0, 0, i, *face)
                  for i, face in enumerate(faces)]
    if multiple_owners:
        owner_rows.append((0, 0, 0, 100, 0, 0, 0, 0, *faces[0]))
    owners = np.asarray(owner_rows, np.int32)
    cycle = [f"{i}:0|{i + 100}:0" for i in range(4)]
    if coincident_distinct_vid:
        # A legitimate unused baseline vertex remains separate at exactly the
        # coordinate of vertex zero; identity is never obtained from position.
        vertices = np.concatenate((vertices, vertices[0:1].copy()))
        vids = np.concatenate((vids, np.asarray([[8, 0, 108, 0]], np.int32)))
        if select_coincident_vid:
            cycle[0] = "8:0|108:0"
    tags = np.ones(len(vertices), np.int32)
    selected = [0, 1]
    if multiple_owners and complete_request:
        selected.append(len(owners) - 1)
    cell = {
        "owners": [owners[i, :7].tolist() for i in selected],
        "source_faces": [[cycle[i] for i in face] for face in faces[:2]],
        "boundary_cycle": cycle,
    }
    document = {
        "levels": {"lower": 0, "root": 1, "upper": 2},
        "boundary_cycle": cycle,
        "anchors": {key: {"position": [1, 1, 0]}
                    for key in ("lower", "root", "upper")},
        "segments": [{"t0": 0, "t1": 1, **deepcopy(cell)},
                     {"t0": 1, "t1": 2, **deepcopy(cell)}],
        "breakpoint_points": [{"time": time, **deepcopy(cell)}
                              for time in (0, 1, 2)],
        "cache_input_sha256": "synthetic-no-cache",
    }
    spec = WindowSpec.from_source_contract(document)
    spec.validate_layout()
    return (vertices, faces, tags), vids, owners, spec


def _snapshot(arrays):
    return tuple((array.shape, array.dtype.str, array.tobytes()) for array in arrays)


def _mesh_digest(mesh):
    digest = hashlib.sha256()
    for shape, dtype, contents in _snapshot(mesh):
        digest.update(repr((shape, dtype)).encode("ascii"))
        digest.update(contents)
    return digest.hexdigest()


def _semantic_mesh(mesh, vids):
    """Source-labelled, oriented semantics invariant to actual row numbering."""
    vertices, faces, tags = mesh
    keys = [tuple(int(value) for value in row) for row in vids]
    if len(vertices) == len(vids) + 1:
        keys.append(("derived_center",))
    _require(len(keys) == len(vertices), "unexpected output vertex count")
    # repr gives a common sortable representation for SourceVID and center keys.
    names = [repr(key) for key in keys]
    vertex_rows = sorted((names[i], tuple(float(x).hex() for x in vertex), int(tags[i]))
                         for i, vertex in enumerate(vertices))
    face_rows = sorted(_oriented(tuple(names[int(i)] for i in face)) for face in faces)
    return vertex_rows, face_rows


def _reindex(data, vertex_order, face_order):
    mesh, vids, owners, spec = data
    vertices, faces, tags = mesh
    vp = np.asarray(vertex_order, np.int32)
    fp = np.asarray(face_order, np.int32)
    _require(sorted(vp.tolist()) == list(range(len(vertices))), "vertex permutation is not bijective")
    _require(sorted(fp.tolist()) == list(range(len(faces))), "face permutation is not bijective")
    old_to_new_vertex = np.empty_like(vp)
    old_to_new_vertex[vp] = np.arange(len(vp), dtype=np.int32)
    old_to_new_face = np.empty_like(fp)
    old_to_new_face[fp] = np.arange(len(fp), dtype=np.int32)
    new_faces = old_to_new_vertex[faces[fp]]
    new_owners = owners[::-1].copy()
    new_owners[:, 7] = old_to_new_face[new_owners[:, 7]]
    new_owners[:, 8:] = old_to_new_vertex[new_owners[:, 8:]]
    return (vertices[vp].copy(), new_faces, tags[vp].copy()), vids[vp].copy(), new_owners, spec


def _evaluate(name, data, *, authorized, expected_current_reason=None):
    PROGRESS["current_case"] = name
    mesh, vids, owners, spec = data
    inputs = (*mesh, vids, owners)
    original = _snapshot(inputs)
    try:
        resolved = {}
        errors = {}
        for method, resolver in (("current", resolve_support), ("ordinary", plain_resolve)):
            try:
                resolved[method] = resolver(mesh, vids, owners, spec, TAU)
            except ValueError as exc:
                errors[method] = str(exc)
            _require(_snapshot(inputs) == original, name + ": resolver mutated input")
            _require((method in resolved) == authorized,
                     name + ": " + method + " authorization differs from expectation; " +
                     errors.get(method, "unexpected acceptance"))
        record = {
            "name": name,
            "expected_authorization": "ACCEPT" if authorized else "REJECT",
            "current_authorization": "ACCEPT" if "current" in resolved else "REJECT",
            "ordinary_authorization": "ACCEPT" if "ordinary" in resolved else "REJECT",
            "baseline_mesh_sha256": _mesh_digest(mesh),
            "vertices": len(mesh[0]), "faces": len(mesh[1]),
            "owner_records": len(owners),
            "requested_owners": [list(owner) for owner in spec.cell(TAU)[0]],
        }
        if not authorized:
            if expected_current_reason is not None:
                _require(expected_current_reason in errors["current"],
                         name + ": current rejected for an unexpected reason: " + errors["current"])
            try:
                propose_actual(mesh, vids, owners, spec, TAU)
            except ValueError as exc:
                record["public_proposal_rejection"] = str(exc)
            else:
                raise RuntimeError("A mechanism mismatch: " + name + ": public proposal accepted")
            # Fixed original square support is a geometric control only. It is
            # not provided to either source resolver and cannot authorize edits.
            control = check_actual_patch(mesh[0], mesh[1], (0, 1), (0, 1, 2, 3), spec.center(TAU))
            _require(control["status"] == "PASS", name + ": common geometry control did not pass")
            record.update(rejection_reasons=errors, geometry_control_only=control,
                          proposal_produced=False, policy_decisions_agree=True)
            output = None
        else:
            _require(resolved["current"] == resolved["ordinary"], name + ": source resolution differs")
            cycle, removed, consumed = resolved["ordinary"]
            ordinary_geometry = check_actual_patch(mesh[0], mesh[1], removed, cycle, spec.center(TAU))
            _require(ordinary_geometry["status"] == "PASS", name + ": ordinary geometry did not pass")
            ordinary_output = assemble(mesh, cycle, removed, spec.center(TAU))
            output, proposal = propose_actual(mesh, vids, owners, spec, TAU)
            _require(output is not None and proposal["status"] == "PROPOSED_NOT_PUBLISHED",
                     name + ": public proposal not produced")
            _require(proposal["geometry"]["status"] == "PASS", name + ": public geometry did not pass")
            _require(proposal["geometry"] == ordinary_geometry, name + ": geometry evidence differs")
            _require(_snapshot(output) == _snapshot(ordinary_output), name + ": actual array outputs differ")
            _require(len(output[0]) == len(mesh[0]) + 1 and len(output[1]) == len(mesh[1]) + 2,
                     name + ": replacement topology differs")
            _require(_snapshot((output[0][:-1], output[2][:-1])) == _snapshot((mesh[0], mesh[2])),
                     name + ": original vertices or tags changed")
            retained = sorted(set(range(len(mesh[1]))) - set(removed))
            _require(_snapshot((output[1][retained],)) == _snapshot((mesh[1][retained],)),
                     name + ": retained faces changed")
            record.update(
                geometry=ordinary_geometry, boundary_actual_ids=list(cycle),
                retired_face_rows=list(removed), consumed_owners=[list(owner) for owner in consumed],
                proposal_produced=True, published=False, policy_decisions_agree=True,
                actual_outputs_byte_equal=True, proposed_mesh_sha256=_mesh_digest(output),
            )
        _require(_snapshot(inputs) == original, name + ": proposal or checker mutated input")
        record["input_mesh_and_ledgers_byte_unchanged"] = True
        return record, output
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("A mechanism unexpected failure in " + name + ": " + str(exc)) from exc
    finally:
        _require(_snapshot(inputs) == original, name + ": input mutation detected on return or failure")


def run():
    """Run five fixed A mechanism families; fail immediately on any mismatch."""
    families = []
    PROGRESS["completed_families"] = families
    single_data = _fixture()
    single, single_output = _evaluate("single_owner_positive", single_data, authorized=True)
    families.append({"family": "single_owner", "cases": [single]})

    multiple_data = _fixture(multiple_owners=True)
    complete, complete_output = _evaluate("complete_independent_multi_owner_positive", multiple_data,
                                          authorized=True)
    _require(single["baseline_mesh_sha256"] == complete["baseline_mesh_sha256"],
             "single and multi-owner baselines differ geometrically")
    _require(_snapshot(single_output) == _snapshot(complete_output),
             "authorized single and complete multi-owner outputs differ")
    families.append({"family": "complete_independent_multi_owner", "cases": [complete]})

    partial, _ = _evaluate("partial_independent_owner_request_rejected",
                           _fixture(multiple_owners=True, complete_request=False), authorized=False,
                           expected_current_reason="Partial suppression")
    _require(partial["baseline_mesh_sha256"] == single["baseline_mesh_sha256"],
             "partial-owner control changed geometry")
    _require(partial["requested_owners"] == single["requested_owners"],
             "single versus partial-owner requests are not identical")
    families.append({"family": "partial_independent_owner_request", "cases": [partial]})

    coincident_data = _fixture(coincident_distinct_vid=True)
    coincident, coincident_output = _evaluate("same_coordinate_distinct_sourcevid_preserved",
                                             coincident_data, authorized=True)
    _require(np.array_equal(coincident_output[0][0], coincident_output[0][8]),
             "coincident coordinates no longer coincide")
    _require(not np.array_equal(coincident_data[1][0], coincident_data[1][8]),
             "coincident vertices lost distinct source identities")
    _require(8 not in coincident["boundary_actual_ids"], "coincident unused vertex substituted for boundary")
    spoofed, _ = _evaluate("coincident_sourcevid_cannot_impersonate_requested_source_face",
                           _fixture(coincident_distinct_vid=True, select_coincident_vid=True),
                           authorized=False, expected_current_reason="independently expected source faces")
    _require(coincident["baseline_mesh_sha256"] == spoofed["baseline_mesh_sha256"],
             "coincident-identity controls changed geometry")
    families.append({"family": "same_coordinate_distinct_sourcevid", "cases": [coincident, spoofed],
                     "scope": "unused coincident baseline vertex; no coordinate welding or identity substitution"})

    reindexed_cases = []
    semantic_baseline = _semantic_mesh(multiple_data[0], multiple_data[1])
    semantic_output = _semantic_mesh(complete_output, multiple_data[1])
    permutations = [
        ([7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        ([2, 5, 0, 7, 1, 4, 3, 6], [3, 8, 1, 6, 0, 9, 4, 2, 7, 5]),
    ]
    for index, (vp, fp) in enumerate(permutations):
        data = _reindex(multiple_data, vp, fp)
        _require(_semantic_mesh(data[0], data[1]) == semantic_baseline,
                 "reindex changed source-labelled input semantics")
        record, output = _evaluate(f"vertex_face_reindex_{index}", data, authorized=True)
        _require(_semantic_mesh(output, data[1]) == semantic_output,
                 "reindex changed source-labelled output semantics")
        _require(record["consumed_owners"] == complete["consumed_owners"],
                 "reindex changed consumed authorization objects")
        record.update(vertex_new_to_old=vp, face_new_to_old=fp,
                      owner_ledger_order_reversed=True, source_labelled_output_semantics_equal=True)
        reindexed_cases.append(record)
    families.append({"family": "vertex_face_reindex", "cases": reindexed_cases})

    report = {
        "schema": "source-contract-probe-A-v1", "status": "PASS",
        "experiment_kind": "SYNTHETIC_MECHANISM_NOT_NATIVE_SCENE_EVIDENCE",
        "owner_semantics": OWNER_SEMANTICS, "ledger_assumption": "SHARED_COMPLETE_CORRECT_TRUSTED",
        "emitter_completeness_tested": False,
        "shared_geometry_checker": "binocmesher.window_geometry_runtime.check_actual_patch",
        "current_public_path": "binocmesher.window_runtime.resolve_support/propose_actual",
        "ordinary_resolver": "experiments.source_contract_probe.plain_reference.plain_resolve",
        "families": families, "family_count": len(families),
        "case_count": sum(len(family["cases"]) for family in families),
        "all_policy_decisions_agree": True, "all_inputs_byte_unchanged": True,
        "conclusion_scope": "Complete source relations are needed for this authorization semantics; ordinary grouping can enforce the same contract. No novelty or backend superiority is implied.",
    }
    # Catch accidental NumPy/Fraction evidence before the campaign writes JSON.
    json.dumps(report, allow_nan=False)
    return report
