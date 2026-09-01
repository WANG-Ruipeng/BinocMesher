#!/usr/bin/env python3
"""Compile one production-derived exact-time replacement patch.

The selected patch is a connected two-triangle disk within the real canonical
saddle event star. All replicated raw owners are listed for exact-once
suppression. Two payloads share the same ownership/boundary contract:

* identity: re-emits the two original triangles;
* star: splits both triangles at the midpoint of their shared source edge.

The star changes triangulation but exactly preserves the original piecewise
linear surface, making it a strong boundary-gluing test without confounding the
experiment with a critical-time geometry change.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from processed_mesh import (
    HVID,
    RawTriangle,
    SourceVID,
    TriangleRef,
    canonical_face_groups,
    edge_incidence,
    selected_event_rows,
    trace_processed_triangles,
)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def oriented_reverse(
    vertices: tuple[SourceVID, SourceVID, SourceVID],
) -> tuple[SourceVID, SourceVID, SourceVID]:
    reversed_cycle = (vertices[0], vertices[2], vertices[1])
    candidates = tuple(
        reversed_cycle[offset:] + reversed_cycle[:offset]
        for offset in range(3)
    )
    return min(candidates)


def directed_boundary_cycle(
    faces: Sequence[tuple[SourceVID, SourceVID, SourceVID]],
) -> tuple[SourceVID, ...]:
    directed = Counter()
    for face in faces:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            directed[(face[first], face[second])] += 1
    for first, second in list(directed):
        if first >= second:
            continue
        cancel = min(
            directed[(first, second)], directed[(second, first)]
        )
        directed[(first, second)] -= cancel
        directed[(second, first)] -= cancel
    remaining = [edge for edge, count in directed.items() for _ in range(count)]
    if len(remaining) != 4:
        raise RuntimeError(
            f"selected two-triangle patch has {len(remaining)} boundary edges"
        )
    outgoing: dict[SourceVID, SourceVID] = {}
    incoming: Counter[SourceVID] = Counter()
    for first, second in remaining:
        if first in outgoing:
            raise RuntimeError("patch boundary branches")
        outgoing[first] = second
        incoming[second] += 1
    if set(outgoing) != set(incoming) or any(value != 1 for value in incoming.values()):
        raise RuntimeError("patch boundary is not one directed cycle")
    start = min(outgoing)
    cycle = [start]
    while len(cycle) < len(remaining):
        following = outgoing[cycle[-1]]
        if following in cycle:
            raise RuntimeError("patch boundary closes early")
        cycle.append(following)
    if outgoing[cycle[-1]] != start:
        raise RuntimeError("patch boundary does not close")
    return tuple(cycle)


def representative_positions(
    groups: dict[tuple[int, tuple[SourceVID, ...]], list[RawTriangle]],
    selected: Sequence[tuple[int, tuple[SourceVID, ...]]],
) -> tuple[dict[SourceVID, np.ndarray], dict[SourceVID, bool]]:
    positions: dict[SourceVID, list[np.ndarray]] = defaultdict(list)
    in_view: dict[SourceVID, list[bool]] = defaultdict(list)
    for key in selected:
        for raw in groups[key]:
            for source_vid, position, visible in zip(
                raw.source_vertices, raw.positions, raw.in_view
            ):
                positions[source_vid].append(np.asarray(position, dtype=np.float64))
                in_view[source_vid].append(bool(visible))
    position_result = {}
    view_result = {}
    for source_vid, values in positions.items():
        reference = values[0]
        if any(np.linalg.norm(value - reference) > 1e-7 for value in values[1:]):
            raise RuntimeError(
                f"source VID {source_vid.text()} has inconsistent coordinates"
            )
        position_result[source_vid] = reference
        view_result[source_vid] = all(in_view[source_vid])
    return position_result, view_result


def write_plan(
    path: Path,
    plan_id: str,
    exact_time: Fraction,
    suppressions: Sequence[TriangleRef],
    boundary_vertices: Sequence[SourceVID],
    internal_vertices: Sequence[tuple[np.ndarray, bool]],
    faces: Sequence[tuple[int, int, int]],
    element: int,
) -> None:
    rows = [
        "SSP1",
        f"PLAN {json.dumps(plan_id)}",
        f"TIME {exact_time.numerator} {exact_time.denominator}",
        f"EXPECT {len(suppressions)} {len(boundary_vertices)} "
        f"{len(internal_vertices)} {len(faces)}",
    ]
    for reference in sorted(suppressions):
        rows.append("SUPPRESS " + " ".join(map(str, reference.values())))
    for local_id, source_vid in enumerate(boundary_vertices):
        rows.append(
            "VERTEX_SOURCE "
            f"{local_id} {element} "
            f"{source_vid.first.node} {source_vid.first.group} "
            f"{source_vid.second.node} {source_vid.second.group}"
        )
    for internal_index, (position, visible) in enumerate(internal_vertices):
        local_id = len(boundary_vertices) + internal_index
        rows.append(
            "VERTEX_INTERNAL "
            f"{local_id} {element} "
            f"{position[0]:.17g} {position[1]:.17g} {position[2]:.17g} "
            f"{int(visible)}"
        )
    for a, b, c in faces:
        rows.append(f"FACE {element} {a} {b} {c}")
    rows.append("END")
    path.write_text("\n".join(rows) + "\n")


def compile_at_time(
    cache_root: Path,
    exact_time: Fraction,
) -> tuple[
    list[RawTriangle],
    dict[tuple[int, tuple[SourceVID, ...]], list[RawTriangle]],
    tuple[tuple[int, tuple[SourceVID, ...]], ...],
    dict[str, object],
]:
    raw, trace_summary = trace_processed_triangles(cache_root, exact_time)
    groups = canonical_face_groups(raw)
    face_keys = sorted(groups)
    incidence = edge_incidence(face_keys)
    event_keys = [
        key for key, values in groups.items()
        if any(value.event_record for value in values)
    ]

    candidates = []
    for index, first in enumerate(event_keys):
        first_element, first_vertices = first
        first_set = set(first_vertices)
        if len(first_set) != 3:
            continue
        # Reject if the opposite orientation is already a separate output face.
        if (first_element, oriented_reverse(first_vertices)) in groups:
            continue
        for second in event_keys[index + 1:]:
            second_element, second_vertices = second
            if second_element != first_element or len(set(second_vertices)) != 3:
                continue
            if (second_element, oriented_reverse(second_vertices)) in groups:
                continue
            shared = first_set & set(second_vertices)
            union = first_set | set(second_vertices)
            if len(shared) != 2 or len(union) != 4:
                continue
            shared_edge = tuple(sorted(shared))
            boundary_edges = []
            for vertices in (first_vertices, second_vertices):
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    edge = tuple(sorted((vertices[a], vertices[b])))
                    if edge != shared_edge:
                        boundary_edges.append(edge)
            if len(set(boundary_edges)) != 4:
                continue
            # A closed whole-mesh local neighborhood: internal diagonal has the
            # two patch incidences; each boundary edge has one patch incidence
            # and one external ordinary face incidence.
            if incidence[(first_element, *shared_edge)] != 2:
                continue
            if any(
                incidence[(first_element, *edge)] != 2
                for edge in boundary_edges
            ):
                continue
            # Exclude degenerate source patches. The runtime seam must be
            # validated on a genuine PL disk rather than on coincident or
            # collinear triangles that already exist in the baseline mesh.
            positions, _ = representative_positions(groups, (first, second))
            first_points = np.asarray([positions[value] for value in first_vertices])
            second_points = np.asarray([positions[value] for value in second_vertices])
            first_area2 = float(np.linalg.norm(np.cross(
                first_points[1] - first_points[0],
                first_points[2] - first_points[0])))
            second_area2 = float(np.linalg.norm(np.cross(
                second_points[1] - second_points[0],
                second_points[2] - second_points[0])))
            scale = max(
                1.0,
                max(float(np.linalg.norm(value)) for value in positions.values()),
            )
            tolerance = 1e-10 * scale * scale
            if min(first_area2, second_area2) <= tolerance:
                continue
            cycle = directed_boundary_cycle((first_vertices, second_vertices))
            center = 0.5 * (positions[shared_edge[0]] + positions[shared_edge[1]])
            cycle_points = [positions[value] for value in cycle]
            star_areas = [
                float(np.linalg.norm(np.cross(
                    cycle_points[(index + 1) % len(cycle)] - cycle_points[index],
                    center - cycle_points[index])))
                for index in range(len(cycle))
            ]
            if min(star_areas) <= tolerance:
                continue
            center_clearance = min(
                float(np.linalg.norm(center - positions[value]))
                for value in cycle
            )
            if center_clearance <= 1e-10 * scale:
                continue
            # Prefer the best-conditioned valid disk; raw-owner count is a
            # secondary stress signal rather than the primary selector.
            score = (
                -min(star_areas),
                -min(first_area2, second_area2),
                -center_clearance,
                -(len(groups[first]) + len(groups[second])),
                first,
                second,
            )
            candidates.append((score, (first, second), shared_edge))
    if not candidates:
        raise RuntimeError(
            f"no closed connected two-triangle event-star patch at {exact_time}"
        )
    _, selected, shared_edge = min(candidates)
    return raw, groups, selected, {
        **trace_summary,
        "exact_time": fraction_text(exact_time),
        "final_oriented_faces": len(groups),
        "event_final_oriented_faces": len(event_keys),
        "selected_shared_edge": [vertex.text() for vertex in shared_edge],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    event_id, event_rows = selected_event_rows(cache_root)
    root = Fraction(
        int(event_rows[0]["root_num"]), int(event_rows[0]["root_den"])
    )
    corner_times = [int(event_rows[0][f"t{index}"]) for index in range(4)]
    lower_values = [Fraction(value, 1) for value in corner_times
                    if Fraction(value, 1) < root]
    upper_values = [Fraction(value, 1) for value in corner_times
                    if Fraction(value, 1) > root]
    if not lower_values or not upper_values:
        raise RuntimeError("selected event is not bracketed by corner times")
    probe_candidates = [
        ("lower", (max(lower_values) + root) / 2),
        ("upper", (root + min(upper_values)) / 2),
    ]

    failure_messages = []
    selected_result = None
    for side, probe_time in probe_candidates:
        try:
            raw, groups, selected, trace_summary = compile_at_time(
                cache_root, probe_time
            )
            selected_result = (
                side, probe_time, raw, groups, selected, trace_summary
            )
            break
        except RuntimeError as error:
            failure_messages.append(f"{side}: {error}")
    if selected_result is None:
        raise RuntimeError("; ".join(failure_messages))

    side, probe_time, raw, groups, selected, trace_summary = selected_result
    first, second = selected
    element = first[0]
    face_vertices = (first[1], second[1])
    cycle = directed_boundary_cycle(face_vertices)
    if len(set(cycle)) != 4:
        raise RuntimeError("selected patch boundary is not a quadrilateral")

    positions, in_view = representative_positions(groups, selected)
    shared_vertices = tuple(sorted(set(first[1]) & set(second[1])))
    if len(shared_vertices) != 2:
        raise RuntimeError("selected faces do not have one shared diagonal")
    center = 0.5 * (
        positions[shared_vertices[0]] + positions[shared_vertices[1]]
    )
    center_visible = in_view[shared_vertices[0]] and in_view[shared_vertices[1]]

    local_ids = {vertex: index for index, vertex in enumerate(cycle)}
    identity_faces = [
        tuple(local_ids[vertex] for vertex in face)
        for face in face_vertices
    ]
    star_center = len(cycle)
    star_faces = [
        (
            local_ids[cycle[index]],
            local_ids[cycle[(index + 1) % len(cycle)]],
            star_center,
        )
        for index in range(len(cycle))
    ]

    suppressions = sorted({
        raw_triangle.reference
        for key in selected
        for raw_triangle in groups[key]
    })
    if not suppressions:
        raise RuntimeError("selected patch has no raw source owners")

    write_plan(
        output / "identity.ssp1",
        "production-derived-identity",
        probe_time,
        suppressions,
        cycle,
        [],
        identity_faces,
        element,
    )
    write_plan(
        output / "star.ssp1",
        "production-derived-shared-edge-star",
        probe_time,
        suppressions,
        cycle,
        [(center, center_visible)],
        star_faces,
        element,
    )

    # Negative fail-closed plans.
    wrong_time = (probe_time.numerator + probe_time.denominator,
                  probe_time.denominator)
    wrong_text = (output / "identity.ssp1").read_text().replace(
        f"TIME {probe_time.numerator} {probe_time.denominator}",
        f"TIME {wrong_time[0]} {wrong_time[1]}",
        1,
    )
    (output / "wrong_time.ssp1").write_text(wrong_text)

    missing_reference = TriangleRef(
        element,
        max(reference.t_group for reference in suppressions) + 100,
        0,
        0,
        0,
        0,
        0,
    )
    missing_plan = (output / "identity.ssp1").read_text()
    missing_plan = missing_plan.replace(
        f"EXPECT {len(suppressions)} {len(cycle)} 0 {len(identity_faces)}",
        f"EXPECT {len(suppressions) + 1} {len(cycle)} 0 {len(identity_faces)}",
        1,
    )
    marker = next(
        line for line in missing_plan.splitlines()
        if line.startswith("SUPPRESS ")
    )
    missing_plan = missing_plan.replace(
        marker,
        marker + "\nSUPPRESS " + " ".join(map(str, missing_reference.values())),
        1,
    )
    (output / "missing_ref.ssp1").write_text(missing_plan)

    bad_boundary = (output / "identity.ssp1").read_text().splitlines()
    for index, line in enumerate(bad_boundary):
        if line.startswith("VERTEX_SOURCE "):
            fields = line.split()
            fields[3] = "2147483000"
            fields[4] = "0"
            fields[5] = "2147483001"
            fields[6] = "0"
            bad_boundary[index] = " ".join(fields)
            break
    (output / "bad_boundary.ssp1").write_text(
        "\n".join(bad_boundary) + "\n"
    )

    boundary_interface = {
        "schema": "binoc-beb1-slice-interface-v1",
        "scope": (
            "A BEB1 tetrahedral block slicer must hand its post-slice boundary "
            "vertices to SSP1 as exact source-edge VIDs. SSP1 validates "
            "ownership and performs global-ID gluing; it does not coordinate-weld."
        ),
        "event_id": event_id,
        "event_root": fraction_text(root),
        "probe_side": side,
        "probe_time": fraction_text(probe_time),
        "source_boundary": [
            {
                "local_id": local_ids[vertex],
                "source_vid": vertex.text(),
                "position": positions[vertex].tolist(),
                "in_view": in_view[vertex],
            }
            for vertex in cycle
        ],
        "shared_diagonal": [value.text() for value in shared_vertices],
        "star_internal_vertex": {
            "position": center.tolist(),
            "in_view": center_visible,
        },
        "suppressed_source_owners": len(suppressions),
    }
    (output / "beb1_slice_interface.json").write_text(
        json.dumps(boundary_interface, indent=2, sort_keys=True) + "\n"
    )

    metadata = {
        "schema": "binoc-source-splice-plan-metadata-v1",
        "event_id": event_id,
        "event_root": fraction_text(root),
        "probe_side": side,
        "probe_time": fraction_text(probe_time),
        "element": element,
        "trace": trace_summary,
        "selected_oriented_faces": [
            [vertex.text() for vertex in key[1]] for key in selected
        ],
        "raw_suppressions": [list(reference.values())
                              for reference in suppressions],
        "suppression_count": len(suppressions),
        "selected_suppression_occurrences": len(suppressions),
        "suppression_refs": [list(reference.values()) for reference in suppressions],
        "boundary_cycle": [vertex.text() for vertex in cycle],
        "boundary_count": len(cycle),
        "identity_face_count": len(identity_faces),
        "star_face_count": len(star_faces),
        "star_internal_count": 1,
        "shared_diagonal": [vertex.text() for vertex in shared_vertices],
        "selected_raw_owner_counts": [len(groups[key]) for key in selected],
    }
    serialized_metadata = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    (output / "plan_metadata.json").write_text(serialized_metadata)
    (output / "plan_summary.json").write_text(serialized_metadata)
    print("PASS_COMPILE_PRODUCTION_SOURCE_SPLICE_PLANS")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
