#!/usr/bin/env python3
'''Compile the selected production saddle into a closed BEB1 Event IR.

TV3 supplies one relative two-tetrahedron half-handle.  This compiler completes
its branch-link disk to a sphere with the forced complementary two tetrahedra,
freezes the source-labelled side trace S_B, and compares all one-sided/root
boundaries with the ordinary whole-mesh patch.  It then constructs an explicit
double mapping cylinder over those three disks.  SSP1 is emitted only after
that relative 3-manifold has positive 4D Gram volumes, the critical side seams
are internal, and every external edge is a SourceVID.
'''
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

TV_DIR = Path(__file__).resolve().parents[1] / 'tv0_tv4'
if str(TV_DIR) not in sys.path:
    sys.path.insert(0, str(TV_DIR))
P1_REFERENCE_DIR = Path(__file__).resolve().parents[1] / 'p1_p4' / 'reference'
if str(P1_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(P1_REFERENCE_DIR))

from theory_audit import (  # type: ignore
    complete_production_event_star,
    face_segments,
    mesh_topology,
)
from processed_mesh import selected_event_rows
from compile_splice_plans import (
    compile_at_time,
    directed_boundary_cycle,
    representative_positions,
    write_plan,
)
from event_complex import audit_volume


TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def fraction_json(value: Fraction) -> dict[str, int]:
    return {'numerator': value.numerator, 'denominator': value.denominator}


def parse_fraction(value: str) -> Fraction:
    numerator, denominator = value.split('/', 1)
    return Fraction(int(numerator), int(denominator))


def fraction_from_json(value: dict[str, Any]) -> Fraction:
    return Fraction(int(value['numerator']), int(value['denominator']))


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def source_vid(first: str, second: str) -> str:
    def key(value: str) -> tuple[int, int]:
        node, group = value.split(':', 1)
        return int(node), int(group)
    return '|'.join(sorted((first, second), key=key))


def block_sources(capsule: dict[str, Any]) -> tuple[list[str | None], list[Fraction]]:
    geometry = capsule.get('geometry', {})
    sources = geometry.get('block_vertex_source_hvids')
    exact_times = geometry.get('block_vertex_exact_times')
    if sources is not None and exact_times is not None:
        return (
            [None if value is None else str(value) for value in sources],
            [fraction_from_json(value) for value in exact_times],
        )

    root = fraction_from_json(capsule['root'])
    ordered = list(zip(capsule['hvids'], map(int, capsule['times'])))
    lower = [(str(hvid), time) for hvid, time in ordered
             if Fraction(time, 1) < root]
    upper = [(str(hvid), time) for hvid, time in ordered
             if Fraction(time, 1) > root]
    if len(lower) != 2 or len(upper) != 2:
        raise RuntimeError('BEB1 block does not have a 2-lower/2-upper split')
    branches = lower + upper
    return [None] + [value[0] for value in branches], [
        root, *[Fraction(value[1], 1) for value in branches]]


def intersection_label(
    key: tuple[str, int, int],
    sources: list[str | None],
    event_id: str,
) -> dict[str, str]:
    kind, first, second = key
    if kind == 'v':
        if first == 0:
            return {'kind': 'critical', 'id': 'critical:' + event_id}
        source = sources[first]
        if source is None:
            raise RuntimeError('noncritical block vertex has no source HVID')
        return {'kind': 'source_hvid', 'id': source}
    first_source = sources[first]
    second_source = sources[second]
    if first_source is not None and second_source is not None:
        return {
            'kind': 'source_vid',
            'id': source_vid(first_source, second_source),
        }
    branch = second_source if first_source is None else first_source
    if branch is None:
        raise RuntimeError('critical block edge has no source branch')
    return {
        'kind': 'critical_seam',
        'id': 'critical:' + event_id + '|branch:' + branch,
    }


def canonical_segment(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def registry_face_trace(
    rows: list[dict[str, str]],
    tau: Fraction,
) -> dict[str, Any]:
    incidence_traces: dict[str, list[list[str]]] = {}
    canonical_traces = set()
    for row in rows:
        hvids = [row[f'h{index}'] for index in range(4)]
        times = [int(row[f't{index}']) for index in range(4)]
        segments = []
        for first_edge, second_edge in face_segments(
                (0, 1, 2, 3), times, tau):
            first = source_vid(
                hvids[first_edge[0]], hvids[first_edge[1]])
            second = source_vid(
                hvids[second_edge[0]], hvids[second_edge[1]])
            segments.append(canonical_segment(first, second))
        canonical = tuple(sorted(segments))
        canonical_traces.add(canonical)
        incidence_traces[row['logical_incidence_id']] = [
            list(segment) for segment in canonical]
    return {
        'time': fraction_json(tau),
        'agrees_across_registry_incidences': len(canonical_traces) == 1,
        'canonical_segments': (
            [list(segment) for segment in next(iter(canonical_traces))]
            if len(canonical_traces) == 1 else []
        ),
        'incidence_traces': incidence_traces,
    }


def block_source_boundary_segments(
    block_slice: dict[str, Any],
) -> list[list[str]]:
    segments = []
    for edge in block_slice['boundary_edges']:
        if edge['classification'] != 'ordinary_source_boundary':
            continue
        segments.append(list(canonical_segment(
            edge['labels'][0]['id'], edge['labels'][1]['id'])))
    return sorted(segments)


def patch_boundary_segments(cycle: Any) -> list[list[str]]:
    return sorted([
        list(canonical_segment(
            cycle[index].text(),
            cycle[(index + 1) % len(cycle)].text()))
        for index in range(len(cycle))
    ])


def compile_ordinary_patch(
    cache_root: Path,
    tau: Fraction,
    required_boundary: frozenset[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, groups, selected, trace = compile_at_time(
        cache_root, tau, required_boundary=required_boundary)
    faces = (selected[0][1], selected[1][1])
    cycle = directed_boundary_cycle(faces)
    positions, in_view = representative_positions(groups, selected)
    suppressions = sorted({
        triangle.reference
        for key in selected
        for triangle in groups[key]
    })
    payload = {
        'time': fraction_json(tau),
        'element': selected[0][0],
        'boundary_cycle': [vertex.text() for vertex in cycle],
        'boundary_segments': patch_boundary_segments(cycle),
        'boundary_positions': {
            vertex.text(): positions[vertex].tolist() for vertex in cycle
        },
        'boundary_in_view': {
            vertex.text(): bool(in_view[vertex]) for vertex in cycle
        },
        'source_faces': [
            [vertex.text() for vertex in face] for face in faces
        ],
        'raw_suppression_count': len(suppressions),
        'trace': trace,
    }
    runtime = {
        'cycle': cycle,
        'positions': positions,
        'in_view': in_view,
        'suppressions': suppressions,
        'faces': faces,
        'element': selected[0][0],
    }
    return payload, runtime


def source_position_map(block_slice: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for vertex in block_slice['vertices']:
        label = vertex['label']
        if label['kind'] != 'source_vid':
            continue
        position = np.asarray(vertex['position'], dtype=np.float64)
        previous = result.get(label['id'])
        if previous is not None and np.linalg.norm(previous - position) > 1e-10:
            raise RuntimeError(
                'event-star SourceVID has inconsistent slice positions')
        result[label['id']] = position
    return result


def patch_position_agreement(
    block_slice: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    block_positions = source_position_map(block_slice)
    patch_positions = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in patch['boundary_positions'].items()
    }
    common = sorted(set(block_positions) & set(patch_positions))
    errors = {
        key: float(np.linalg.norm(block_positions[key] - patch_positions[key]))
        for key in common
    }
    scale = max(
        [1.0] +
        [float(np.linalg.norm(value)) for value in block_positions.values()] +
        [float(np.linalg.norm(value)) for value in patch_positions.values()])
    tolerance = 1e-7 * scale
    return {
        'same_source_vids': set(block_positions) == set(patch_positions),
        'maximum_position_error': max(errors.values(), default=0.0),
        'tolerance': tolerance,
        'agrees': (
            set(block_positions) == set(patch_positions) and
            max(errors.values(), default=0.0) <= tolerance
        ),
    }


def build_whole_mesh_patch_slice(
    patch: dict[str, Any],
    runtime: dict[str, Any],
    event_id: str,
    critical_position: np.ndarray | None = None,
) -> dict[str, Any]:
    cycle = runtime['cycle']
    local = {vertex: index for index, vertex in enumerate(cycle)}
    positions = [
        np.asarray(runtime['positions'][vertex], dtype=np.float64)
        for vertex in cycle
    ]
    labels = [
        {'kind': 'source_vid', 'id': vertex.text()} for vertex in cycle
    ]
    if critical_position is None:
        faces = [
            tuple(local[vertex] for vertex in face)
            for face in runtime['faces']
        ]
    else:
        center = len(positions)
        positions.append(np.asarray(critical_position, dtype=np.float64))
        labels.append({'kind': 'critical', 'id': 'critical:' + event_id})
        faces = [
            (index, (index + 1) % len(cycle), center)
            for index in range(len(cycle))
        ]
    edge_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    for face in faces:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge_counts[tuple(sorted((face[first], face[second])))] += 1
    boundary = [
        edge for edge, count in sorted(edge_counts.items()) if count == 1
    ]
    minimum_area2 = min(
        float(np.linalg.norm(np.cross(
            positions[face[1]] - positions[face[0]],
            positions[face[2]] - positions[face[0]])))
        for face in faces
    )
    reference_faces = [
        tuple(local[vertex] for vertex in face)
        for face in runtime['faces']
    ]
    reference_normal = sum((
        np.cross(
            positions[face[1]] - positions[face[0]],
            positions[face[2]] - positions[face[0]])
        for face in reference_faces
    ), start=np.zeros(3, dtype=np.float64))
    orientation_dots = [
        float(np.dot(
            np.cross(
                positions[face[1]] - positions[face[0]],
                positions[face[2]] - positions[face[0]]),
            reference_normal))
        for face in faces
    ]
    orientation_tolerance = 1e-12 * max(
        1.0, float(np.dot(reference_normal, reference_normal)))
    position_array = np.asarray(positions, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    return {
        'time': patch['time'],
        'vertices': [
            {
                'index': index,
                'position': position.tolist(),
                'label': labels[index],
            }
            for index, position in enumerate(positions)
        ],
        'faces': [list(face) for face in faces],
        'topology': mesh_topology(position_array, face_array),
        'minimum_double_area': minimum_area2,
        'orientation_reference_normal': reference_normal.tolist(),
        'orientation_dots': orientation_dots,
        'orientation_coherent': (
            float(np.linalg.norm(reference_normal)) > 1e-12 and
            min(orientation_dots) > orientation_tolerance
        ),
        'boundary_edges': [
            {
                'vertices': list(edge),
                'labels': [labels[edge[0]], labels[edge[1]]],
                'classification': 'ordinary_source_boundary',
            }
            for edge in boundary
        ],
        'unresolved_boundary_edges': [],
        'source_boundary_complete': True,
    }


def side_trace_affine_audit(
    patches: dict[str, Any],
    probes: dict[str, Fraction],
) -> dict[str, Any]:
    if set(patches) != set(probes):
        return {
            'same_boundary_segments': False,
            'same_boundary_vertices': False,
            'affine_trajectory_error': None,
            'tolerance': None,
            'regular': False,
        }
    segments = {
        tuple(tuple(segment) for segment in patch['boundary_segments'])
        for patch in patches.values()
    }
    vertex_sets = {
        tuple(sorted(patch['boundary_positions']))
        for patch in patches.values()
    }
    same_segments = len(segments) == 1
    same_vertices = len(vertex_sets) == 1
    errors: dict[str, float] = {}
    tolerance: float | None = None
    if same_vertices:
        denominator = probes['upper'] - probes['lower']
        weight = float(
            (probes['critical'] - probes['lower']) / denominator)
        scale = 1.0
        for vertex in next(iter(vertex_sets)):
            lower = np.asarray(
                patches['lower']['boundary_positions'][vertex], dtype=float)
            critical = np.asarray(
                patches['critical']['boundary_positions'][vertex], dtype=float)
            upper = np.asarray(
                patches['upper']['boundary_positions'][vertex], dtype=float)
            expected = (1.0 - weight) * lower + weight * upper
            errors[vertex] = float(np.linalg.norm(critical - expected))
            scale = max(
                scale, float(np.linalg.norm(lower)),
                float(np.linalg.norm(critical)), float(np.linalg.norm(upper)))
        tolerance = 1e-7 * scale
    maximum = max(errors.values(), default=0.0)
    return {
        'same_boundary_segments': same_segments,
        'same_boundary_vertices': same_vertices,
        'affine_trajectory_errors': errors,
        'affine_trajectory_error': maximum,
        'tolerance': tolerance,
        'regular': (
            same_segments and same_vertices and tolerance is not None and
            maximum <= tolerance
        ),
    }


def build_event_star_mapping_cylinder(
    patches: dict[str, Any],
    runtimes: dict[str, Any],
    probes: dict[str, Fraction],
    event_id: str,
    critical_position: np.ndarray,
) -> dict[str, Any]:
    '''Build an explicit disk x interval whose middle disk is the BEB1 fan.'''
    if set(patches) != set(probes) or set(runtimes) != set(probes):
        raise RuntimeError('mapping cylinder requires lower/root/upper patches')
    reference_cycle = tuple(runtimes['critical']['cycle'])
    if len(reference_cycle) != 4:
        raise RuntimeError('mapping cylinder requires a quadrilateral S_B')
    if any(
            set(runtimes[name]['cycle']) != set(reference_cycle)
            for name in probes):
        raise RuntimeError('mapping cylinder boundary identities changed')

    level_names = ('lower', 'critical', 'upper')
    fan_faces = (
        (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))
    level_positions: list[list[np.ndarray]] = []
    center_positions: dict[str, list[float]] = {}
    for name in level_names:
        runtime = runtimes[name]
        boundary = [
            np.asarray(runtime['positions'][vertex], dtype=np.float64)
            for vertex in reference_cycle
        ]
        if name == 'critical':
            center = np.asarray(critical_position, dtype=np.float64)
        else:
            shared = set(runtime['faces'][0]) & set(runtime['faces'][1])
            if len(shared) != 2:
                raise RuntimeError(
                    'ordinary endpoint patch has no unique shared diagonal')
            center = 0.5 * sum((
                np.asarray(runtime['positions'][vertex], dtype=np.float64)
                for vertex in shared
            ), start=np.zeros(3, dtype=np.float64))
        center_positions[name] = center.tolist()
        level_positions.append([*boundary, center])

    vertices4 = np.asarray([
        [*position.tolist(), float(probes[name])]
        for name, positions in zip(level_names, level_positions)
        for position in positions
    ], dtype=np.float64)
    tetrahedra: list[tuple[int, int, int, int]] = []
    for slab in (0, 1):
        low_offset = 5 * slab
        high_offset = 5 * (slab + 1)
        for raw_face in fan_faces:
            a, b, c = sorted(raw_face)
            a0, b0, c0 = (
                low_offset + a, low_offset + b, low_offset + c)
            a1, b1, c1 = (
                high_offset + a, high_offset + b, high_offset + c)
            tetrahedra.extend((
                (a0, b0, c0, c1),
                (a0, b0, b1, c1),
                (a0, a1, b1, c1),
            ))
    tets = np.asarray(tetrahedra, dtype=np.int64)
    volumes = []
    for tet in tets:
        edges = (
            vertices4[np.asarray(tet[1:], dtype=np.int64)] -
            vertices4[int(tet[0])])
        determinant = float(np.linalg.det(edges @ edges.T))
        volumes.append(float(np.sqrt(max(determinant, 0.0)) / 6.0))

    facet_counts: collections.Counter[tuple[int, int, int]] = (
        collections.Counter(
            tuple(sorted(face))
            for tet in tets
            for face in (
                (int(tet[0]), int(tet[1]), int(tet[2])),
                (int(tet[0]), int(tet[1]), int(tet[3])),
                (int(tet[0]), int(tet[2]), int(tet[3])),
                (int(tet[1]), int(tet[2]), int(tet[3])),
            )
        )
    )
    boundary_faces = sorted(
        face for face, count in facet_counts.items() if count == 1)
    lower_cap = [face for face in boundary_faces if all(v < 5 for v in face)]
    upper_cap = [face for face in boundary_faces if all(v >= 10 for v in face)]
    side_faces = [
        face for face in boundary_faces
        if face not in lower_cap and face not in upper_cap
    ]
    boundary_edges = sorted({
        edge
        for face in boundary_faces
        for edge in (
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[0], face[2]))),
            tuple(sorted((face[1], face[2]))),
        )
    })
    critical_side_edges = [
        edge for edge in boundary_edges if 9 in edge
    ]
    center_indices = {4, 9, 14}
    side_center_incidence = sum(
        bool(set(face) & center_indices) for face in side_faces)
    volume_audit = audit_volume(
        [tuple(map(int, tet)) for tet in tets]).to_dict()
    checks = {
        'fifteen_vertices': vertices4.shape == (15, 4),
        'twenty_four_tetrahedra': tets.shape == (24, 4),
        'positive_gram_volumes': min(volumes) > 1e-12,
        'valid_relative_3_manifold':
            volume_audit['valid_relative_3_manifold'] is True,
        'lower_cap_four_faces': len(lower_cap) == 4,
        'upper_cap_four_faces': len(upper_cap) == 4,
        'side_trace_sixteen_faces': len(side_faces) == 16,
        'side_trace_avoids_all_centers': side_center_incidence == 0,
        'middle_critical_vertex_not_on_boundary': all(
            9 not in face for face in boundary_faces),
        'critical_side_edges_zero': not critical_side_edges,
    }
    return {
        'schema': 'binoc-beb1-double-mapping-cylinder-v1',
        'pass': all(checks.values()),
        'verdict': (
            'PASS_BEB1_DOUBLE_MAPPING_CYLINDER'
            if all(checks.values()) else
            'STOP_BEB1_DOUBLE_MAPPING_CYLINDER'
        ),
        'event_id': event_id,
        'level_vertex_layout': (
            'five vertices per level: four S_B vertices followed by center'),
        'levels': {
            name: fraction_json(probes[name]) for name in level_names
        },
        'boundary_cycle': [vertex.text() for vertex in reference_cycle],
        'center_positions': center_positions,
        'vertices4': vertices4.tolist(),
        'tetrahedra': tets.tolist(),
        'tetrahedron_gram_volumes': volumes,
        'minimum_gram_volume': min(volumes),
        'boundary_faces': [list(face) for face in boundary_faces],
        'lower_cap_faces': [list(face) for face in lower_cap],
        'upper_cap_faces': [list(face) for face in upper_cap],
        'side_trace_faces': [list(face) for face in side_faces],
        'critical_side_edges': [list(edge) for edge in critical_side_edges],
        'critical_side_edges_remaining': len(critical_side_edges),
        'volume_audit': volume_audit,
        'checks': checks,
    }


def slice_block(
    vertices4: np.ndarray,
    tets: np.ndarray,
    exact_times: list[Fraction],
    sources: list[str | None],
    tau: Fraction,
    event_id: str,
) -> dict[str, Any]:
    vertex_keys: list[tuple[str, int, int]] = []
    positions: list[np.ndarray] = []
    key_to_index: dict[tuple[str, int, int], int] = {}
    faces: list[tuple[int, int, int]] = []

    def add_vertex(key: tuple[str, int, int]) -> int:
        if key in key_to_index:
            return key_to_index[key]
        kind, first, second = key
        if kind == 'v':
            position = vertices4[first, :3]
        else:
            first_time = exact_times[first]
            second_time = exact_times[second]
            weight = float(
                (tau - first_time) / (second_time - first_time))
            position = (
                (1.0 - weight) * vertices4[first, :3] +
                weight * vertices4[second, :3]
            )
        index = len(positions)
        key_to_index[key] = index
        vertex_keys.append(key)
        positions.append(np.asarray(position, dtype=np.float64))
        return index

    for tet_row in np.asarray(tets, dtype=np.int64):
        tet = [int(value) for value in tet_row]
        crossing: list[int] = []
        for vertex in tet:
            if exact_times[vertex] == tau:
                crossing.append(add_vertex(('v', vertex, -1)))
        for local_first, local_second in TET_EDGES:
            first, second = tet[local_first], tet[local_second]
            first_time, second_time = exact_times[first], exact_times[second]
            if ((first_time < tau < second_time) or
                    (second_time < tau < first_time)):
                crossing.append(add_vertex((
                    'e', min(first, second), max(first, second))))
        crossing = list(dict.fromkeys(crossing))
        if len(crossing) == 3:
            faces.append(tuple(crossing))
        elif len(crossing) == 4:
            points = np.asarray([positions[index] for index in crossing])
            centered = points - points.mean(axis=0)
            _, _, basis = np.linalg.svd(centered)
            angles = np.arctan2(centered @ basis[1], centered @ basis[0])
            ordered = [crossing[index] for index in np.argsort(angles)]
            faces.extend((
                (ordered[0], ordered[1], ordered[2]),
                (ordered[0], ordered[2], ordered[3]),
            ))
        elif crossing:
            raise RuntimeError(
                f'unsupported BEB1 slice polygon with {len(crossing)} vertices')

    vertex_array = np.asarray(positions, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    labels = [
        intersection_label(key, sources, event_id) for key in vertex_keys]
    edge_incidence: collections.Counter[tuple[int, int]] = collections.Counter()
    for face in faces:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge_incidence[tuple(sorted(
                (face[first], face[second])))] += 1
    boundary_edges = [
        edge for edge, count in sorted(edge_incidence.items()) if count == 1]
    classified_boundary = []
    unresolved = []
    for first, second in boundary_edges:
        first_label, second_label = labels[first], labels[second]
        resolved = (
            first_label['kind'] == 'source_vid' and
            second_label['kind'] == 'source_vid'
        )
        row = {
            'vertices': [first, second],
            'labels': [first_label, second_label],
            'classification': (
                'ordinary_source_boundary' if resolved
                else 'unresolved_critical_side_trace'
            ),
        }
        classified_boundary.append(row)
        if not resolved:
            unresolved.append(row)
    topology = mesh_topology(vertex_array, face_array)
    return {
        'time': fraction_json(tau),
        'vertices': [
            {
                'index': index,
                'position': vertex_array[index].tolist(),
                'block_intersection': list(vertex_keys[index]),
                'label': labels[index],
            }
            for index in range(len(vertex_array))
        ],
        'faces': [list(map(int, face)) for face in faces],
        'topology': topology,
        'boundary_edges': classified_boundary,
        'unresolved_boundary_edges': unresolved,
        'source_boundary_complete': not unresolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--theory-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-output', type=Path)
    parser.add_argument('--expected-root', default='104/5')
    parser.add_argument('--require-whole-mesh-ready', action='store_true')
    args = parser.parse_args()

    cache_root = args.cache_root.resolve()
    theory_root = args.theory_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    event_id, rows = selected_event_rows(cache_root)
    roots = {
        Fraction(int(row['root_num']), int(row['root_den']))
        for row in rows
    }
    if len(roots) != 1:
        raise RuntimeError('selected registry event has inconsistent roots')
    root = next(iter(roots))
    expected_root = parse_fraction(args.expected_root)

    capsule = json.loads(
        (theory_root / 'tv3' / 'event_capsule.json').read_text())
    tv3 = json.loads(
        (theory_root / 'tv3' / 'tv3_summary.json').read_text())
    tv4 = json.loads(
        (theory_root / 'tv4' / 'tv4_summary.json').read_text())
    with np.load(
            theory_root / 'tv3' / 'event_block.npz',
            allow_pickle=False) as block:
        vertices4 = np.asarray(block['vertices4'], dtype=np.float64)
        tets = np.asarray(block['tets'], dtype=np.int64)

    sources, exact_times = block_sources(capsule)
    if (len(vertices4) != len(sources) or
            len(vertices4) != len(exact_times)):
        raise RuntimeError('BEB1 block source labels have the wrong arity')
    if any(
            abs(float(exact_times[index]) - vertices4[index, 3]) > 1e-12
            for index in range(len(vertices4))):
        raise RuntimeError('BEB1 block exact times disagree with geometry')

    corner_times = [Fraction(int(value), 1) for value in capsule['times']]
    clearance = min(abs(value - root) for value in corner_times)
    if clearance <= 0:
        raise RuntimeError('selected saddle has no regular one-sided window')
    epsilon = clearance / 2
    probes = {
        'lower': root - epsilon,
        'critical': root,
        'upper': root + epsilon,
    }
    half_handle_slices = {
        name: slice_block(
            vertices4, tets, exact_times, sources, tau, event_id)
        for name, tau in probes.items()
    }
    event_star_tets, completion = complete_production_event_star(tets)
    complement_tets = np.asarray(
        completion['added_tetrahedra'], dtype=np.int64)
    complement_slices = {
        name: slice_block(
            vertices4, complement_tets, exact_times, sources, tau, event_id)
        for name, tau in probes.items()
    }
    slices = {
        name: slice_block(
            vertices4, event_star_tets, exact_times, sources, tau, event_id)
        for name, tau in probes.items()
    }
    registry_traces = {
        name: registry_face_trace(rows, probes[name])
        for name in ('lower', 'upper')
    }
    registry_half_agreement = {
        name: (
            registry_traces[name]['agrees_across_registry_incidences'] and
            block_source_boundary_segments(half_handle_slices[name]) ==
            registry_traces[name]['canonical_segments']
        )
        for name in ('lower', 'upper')
    }
    registry_complement_agreement = {
        name: (
            registry_traces[name]['agrees_across_registry_incidences'] and
            block_source_boundary_segments(complement_slices[name]) ==
            registry_traces[name]['canonical_segments']
        )
        for name in ('lower', 'upper')
    }
    registry_interface_coverage = {
        name: (
            registry_half_agreement[name] or
            registry_complement_agreement[name]
        )
        for name in ('lower', 'upper')
    }

    ordinary_patches: dict[str, Any] = {}
    ordinary_patch_runtime: dict[str, Any] = {}
    ordinary_patch_errors: dict[str, str] = {}
    required_boundary = None
    for name, tau in probes.items():
        try:
            patch, runtime = compile_ordinary_patch(
                cache_root, tau, required_boundary=required_boundary)
            ordinary_patches[name] = patch
            ordinary_patch_runtime[name] = runtime
            if required_boundary is None:
                required_boundary = frozenset(runtime['cycle'])
        except Exception as error:
            ordinary_patch_errors[name] = repr(error)
    core_boundary_agreement = {
        name: (
            name in ordinary_patches and
            block_source_boundary_segments(slices[name]) ==
            ordinary_patches[name]['boundary_segments']
        )
        for name in probes
    }
    core_position_agreement = {
        name: (
            patch_position_agreement(slices[name], ordinary_patches[name])
            if name in ordinary_patches else {
                'same_source_vids': False,
                'maximum_position_error': None,
                'tolerance': None,
                'agrees': False,
            }
        )
        for name in probes
    }
    side_trace_audit = side_trace_affine_audit(ordinary_patches, probes)
    reference_segments = (
        ordinary_patches['critical']['boundary_segments']
        if 'critical' in ordinary_patches else []
    )
    whole_mesh_boundary_agreement = {
        name: (
            name in ordinary_patches and
            ordinary_patches[name]['boundary_segments'] == reference_segments
        )
        for name in probes
    }
    critical_position = np.asarray(
        capsule['geometry']['critical_position'], dtype=np.float64)
    whole_mesh_slices: dict[str, Any] = {}
    if not ordinary_patch_errors:
        whole_mesh_slices = {
            'lower': build_whole_mesh_patch_slice(
                ordinary_patches['lower'], ordinary_patch_runtime['lower'],
                event_id),
            'critical': build_whole_mesh_patch_slice(
                ordinary_patches['critical'],
                ordinary_patch_runtime['critical'], event_id,
                critical_position),
            'upper': build_whole_mesh_patch_slice(
                ordinary_patches['upper'], ordinary_patch_runtime['upper'],
                event_id),
        }
    mapping_cylinder: dict[str, Any] | None = None
    mapping_cylinder_error: str | None = None
    if not ordinary_patch_errors and len(whole_mesh_slices) == 3:
        try:
            mapping_cylinder = build_event_star_mapping_cylinder(
                ordinary_patches,
                ordinary_patch_runtime,
                probes,
                event_id,
                critical_position,
            )
        except Exception as error:
            mapping_cylinder_error = repr(error)

    raw_ids = [row['raw_id'] for row in rows]
    logical_ids = sorted({row['logical_incidence_id'] for row in rows})
    block_payload = {
        'vertices4': vertices4.tolist(),
        'tets': tets.tolist(),
        'sources': sources,
        'exact_times': [fraction_json(value) for value in exact_times],
    }
    topology_keys = (
        'V', 'E', 'F', 'chi', 'components', 'boundary_edges',
        'boundary_loops', 'nonmanifold_edges', 'duplicate_faces',
    )

    def topology_matches(name: str) -> bool:
        expected = capsule[name + '_slice']
        actual = half_handle_slices[name]['topology']
        return all(
            int(actual.get(key, -1)) == int(expected.get(key, -2))
            for key in topology_keys
        )

    resolved_cells = tv4.get('resolved_logical_cells', {})
    completed_tv4 = tv4.get('completed_event_core', {})
    relative_unresolved_counts = {
        name: len(value['unresolved_boundary_edges'])
        for name, value in half_handle_slices.items()
    }
    completed_unresolved_counts = {
        name: len(value['unresolved_boundary_edges'])
        for name, value in slices.items()
    }
    checks = {
        'production_event_selected': event_id.startswith('element='),
        'target_event_is_exact_104_over_5': root == expected_root,
        'capsule_event_identity_exact': (
            capsule.get('event_id') == event_id and
            fraction_from_json(capsule['root']) == root
        ),
        'registry_raw_observations_unique': (
            bool(raw_ids) and len(raw_ids) == len(set(raw_ids))
        ),
        'registry_logical_batch_nontrivial': len(logical_ids) > 1,
        'tv3_half_handle_certified': (
            tv3.get('verdict') ==
            'PASS_TV3_OFFLINE_LOCAL_EVENT_BLOCK_COMPILATION'
        ),
        'tv4_event_star_closure_certified': (
            tv4.get('verdict') ==
            'PASS_TV4_PRODUCTION_DERIVED_OFFLINE_GLOBAL_SPLICE' and
            int(completed_tv4.get('tetrahedra', -1)) == 4 and
            completed_tv4.get(
                'critical_link', {}).get('is_sphere') is True and
            int(completed_tv4.get(
                'critical_side_faces_remaining', -1)) == 0
        ),
        'tv4_consumes_same_registry_batch': (
            int(tv4.get('raw_observations', -1)) == len(raw_ids) and
            int(tv4.get('logical_incidences', -1)) == len(logical_ids) and
            set(resolved_cells) == set(logical_ids)
        ),
        'registry_one_sided_interfaces_agree': all(
            registry_traces[name]['agrees_across_registry_incidences']
            for name in ('lower', 'upper')
        ),
        'event_star_outside_unchanged': (
            int(tv4.get('outside_changed_cells', -1)) == 0
        ),
        'two_tetrahedron_beb1_kernel': (
            tets.shape == (2, 4) and
            capsule.get('critical_link', {}).get('is_disk') is True
        ),
        'block_vertices_have_source_roles': (
            len(sources) == 5 and sources[0] is None and
            all(value is not None for value in sources[1:]) and
            len(set(sources[1:])) == 4
        ),
        'lower_slice_matches_tv3': topology_matches('lower'),
        'upper_slice_matches_tv3': topology_matches('upper'),
        'relative_half_handle_component_transition_2_to_1': (
            half_handle_slices['lower']['topology']['components'] == 2 and
            half_handle_slices['upper']['topology']['components'] == 1
        ),
        'relative_half_handle_has_four_side_seams': all(
            relative_unresolved_counts[name] == 4 for name in probes
        ),
        'completed_event_star_four_tetrahedra': (
            event_star_tets.shape == (4, 4)
        ),
        'completed_critical_link_is_sphere': (
            completion['critical_link']['is_sphere'] is True
        ),
        'completed_critical_side_faces_zero': (
            completion['critical_side_faces_remaining'] == 0
        ),
        'completed_slices_are_regular_disks': all(
            slices[name]['topology']['components'] == 1 and
            slices[name]['topology']['chi'] == 1 and
            slices[name]['topology']['boundary_loops'] == 1 and
            slices[name]['topology']['boundary_edges'] == 4 and
            slices[name]['topology']['nonmanifold_edges'] == 0 and
            slices[name]['topology']['duplicate_faces'] == 0
            for name in probes
        ),
        'completed_side_seams_eliminated': all(
            completed_unresolved_counts[name] == 0 for name in probes
        ),
        'critical_slice_contains_critical_vertex': any(
            vertex['label']['kind'] == 'critical'
            for vertex in slices['critical']['vertices']
        ),
        'one_sided_window_avoids_source_vertex_times': all(
            probes[name] not in set(corner_times)
            for name in ('lower', 'upper')
        ),
    }
    event_ir_pass = all(checks.values())
    source_boundary_ready = all(
        slices[name]['source_boundary_complete']
        for name in ('lower', 'critical', 'upper')
    )
    interface_ready = all(registry_interface_coverage.values())
    mapping_cylinder_ready = (
        mapping_cylinder is not None and
        mapping_cylinder['pass'] is True
    )
    ordinary_boundary_ready = (
        not ordinary_patch_errors and
        all(whole_mesh_boundary_agreement.values()) and
        side_trace_audit['regular'] and
        all(
            value['topology']['components'] == 1 and
            value['topology']['chi'] == 1 and
            value['topology']['boundary_loops'] == 1 and
            value['topology']['boundary_edges'] == 4 and
            value['minimum_double_area'] > 1e-12 and
            value['orientation_coherent']
            for value in whole_mesh_slices.values()
        ) and
        len(whole_mesh_slices) == 3
    )
    root_runtime = ordinary_patch_runtime.get('critical')
    critical_plan_ready = (
        root_runtime is not None and
        len(root_runtime['cycle']) == 4 and
        len(root_runtime['suppressions']) > 0 and
        whole_mesh_boundary_agreement['critical'] and
        ordinary_boundary_ready and
        mapping_cylinder_ready
    )
    whole_mesh_ready = (
        event_ir_pass and interface_ready and source_boundary_ready and
        ordinary_boundary_ready and critical_plan_ready)
    disposition = (
        'READY_FOR_WHOLE_MESH_SPLICE'
        if whole_mesh_ready else
        'EVENT_STAR_GEOMETRY_UNRESOLVED'
    )
    plan_metadata: dict[str, Any] | None = None
    if critical_plan_ready and root_runtime is not None:
        cycle = root_runtime['cycle']
        center_id = len(cycle)
        replacement_faces = [
            (index, (index + 1) % len(cycle), center_id)
            for index in range(len(cycle))
        ]
        plan_metadata = {
            'schema': 'binoc-critical-beb1-whole-mesh-plan-v1',
            'plan_id': 'production-critical-beb1-event-star',
            'event_id': event_id,
            'exact_time': fraction_json(root),
            'element': int(root_runtime['element']),
            'boundary_cycle': [vertex.text() for vertex in cycle],
            'critical_position': critical_position.tolist(),
            'raw_suppressions': [
                list(reference.values())
                for reference in root_runtime['suppressions']
            ],
            'raw_suppression_count': len(root_runtime['suppressions']),
            'replacement_faces': [list(face) for face in replacement_faces],
            'replacement_face_count': len(replacement_faces),
            'critical_vertex_is_internal': True,
        }
        if whole_mesh_ready and args.plan_output is not None:
            plan_output = args.plan_output.resolve()
            if plan_output.exists():
                raise FileExistsError(plan_output)
            plan_output.parent.mkdir(parents=True, exist_ok=True)
            write_plan(
                plan_output,
                plan_metadata['plan_id'],
                root,
                root_runtime['suppressions'],
                cycle,
                [(
                    critical_position,
                    all(root_runtime['in_view'][vertex] for vertex in cycle),
                )],
                replacement_faces,
                int(root_runtime['element']),
            )
            plan_metadata['path'] = str(plan_output)
            plan_metadata['sha256'] = hashlib.sha256(
                plan_output.read_bytes()).hexdigest()
    payload = {
        'schema': 'binoc-critical-beb1-event-ir-v2',
        'pass': event_ir_pass,
        'verdict': (
            'PASS_CRITICAL_BEB1_EVENT_IR'
            if event_ir_pass else
            'STOP_CRITICAL_BEB1_EVENT_IR'
        ),
        'runtime_disposition': disposition,
        'whole_mesh_splice_ready': whole_mesh_ready,
        'event': {
            'event_id': event_id,
            'root': fraction_json(root),
            'u': capsule['u'],
            'v': capsule['v'],
            'raw_observations': len(raw_ids),
            'logical_incidences': len(logical_ids),
            'logical_incidence_ids': logical_ids,
            'resolved_logical_cells': resolved_cells,
        },
        'one_sided_window': {
            'epsilon': fraction_json(epsilon),
            'lower': fraction_json(probes['lower']),
            'critical': fraction_json(root),
            'upper': fraction_json(probes['upper']),
            'source_corner_times': [fraction_json(value)
                                    for value in corner_times],
        },
        'kernel': {
            'kind': 'BEB1_COMPLETED_FOUR_TETRAHEDRON_CORE',
            'block_sha256': stable_hash(block_payload),
            'vertices4': block_payload['vertices4'],
            'relative_half_handle_tets': block_payload['tets'],
            'event_star_tets': event_star_tets.tolist(),
            'block_vertex_source_hvids': sources,
            'block_vertex_exact_times': block_payload['exact_times'],
            'relative_critical_link': capsule['critical_link'],
            'completed_critical_link': completion['critical_link'],
            'completion': completion,
        },
        'relative_half_handle_slices': half_handle_slices,
        'complement_slices': complement_slices,
        'core_event_star_slices': slices,
        'slices': whole_mesh_slices,
        'registry_interface_traces': registry_traces,
        'ordinary_whole_mesh_patches': ordinary_patches,
        'ordinary_whole_mesh_patch_errors': ordinary_patch_errors,
        'side_trace': {
            'symbol': 'S_B',
            'kind': 'TRIANGULATED_SOURCEVID_AFFINE_BOUNDARY_CYLINDER',
            'boundary_segments': reference_segments,
            'trajectories': [
                {
                    'source_vid': source_id,
                    'lower_position':
                        ordinary_patches['lower']['boundary_positions'][
                            source_id],
                    'critical_position':
                        ordinary_patches['critical']['boundary_positions'][
                            source_id],
                    'upper_position':
                        ordinary_patches['upper']['boundary_positions'][
                            source_id],
                }
                for source_id in sorted(
                    ordinary_patches.get(
                        'critical', {}).get('boundary_positions', {}))
            ] if not ordinary_patch_errors else [],
            'affine_audit': side_trace_audit,
            'triangulated_faces': (
                mapping_cylinder['side_trace_faces']
                if mapping_cylinder is not None else []
            ),
            'critical_vertex_on_side_trace': False,
            'regular_on_one_sided_window': side_trace_audit['regular'],
        },
        'event_star_geometry': {
            'kind': 'EXPLICIT_DOUBLE_MAPPING_CYLINDER',
            'lower_patch': whole_mesh_slices.get('lower'),
            'critical_patch': whole_mesh_slices.get('critical'),
            'upper_patch': whole_mesh_slices.get('upper'),
            'mapping_cylinder': mapping_cylinder,
            'mapping_cylinder_error': mapping_cylinder_error,
            'critical_vertex_internal_at_root': True,
            'core_boundary_differs_from_external_s_b': any(
                not value for value in core_boundary_agreement.values()),
        },
        'whole_mesh_replacement_plan': plan_metadata,
        'checks': checks,
        'admission': {
            'event_ir_certified': event_ir_pass,
            'all_boundary_edges_source_resolved': source_boundary_ready,
            'registry_half_agreement': registry_half_agreement,
            'registry_complement_agreement': registry_complement_agreement,
            'registry_interface_coverage': registry_interface_coverage,
            'ordinary_whole_mesh_boundary_agreement':
                whole_mesh_boundary_agreement,
            'core_boundary_agreement': core_boundary_agreement,
            'core_position_agreement': core_position_agreement,
            'side_trace_affine_audit': side_trace_audit,
            'mapping_cylinder_ready': mapping_cylinder_ready,
            'mapping_cylinder_error': mapping_cylinder_error,
            'relative_unresolved_boundary_edge_counts':
                relative_unresolved_counts,
            'completed_unresolved_boundary_edge_counts':
                completed_unresolved_counts,
            'critical_plan_ready': critical_plan_ready,
            'policy': (
                'Emit SSP1 only after the complementary half closes the '
                'critical link, all former critical seams are internal, and '
                'lower/root/upper external boundaries agree in SourceVID and '
                'production geometry with the ordinary whole-mesh patch, and '
                'the explicit double mapping cylinder passes its relative '
                '3-manifold and positive 4D Gram-volume audits.'
            ),
        },
        'scope': {
            'completed': (
                'production-selected 104/5 BEB1 event identity, registry batch '
                'closure, four-tetrahedron event-star, source-prescribed S_B, '
                'explicit 15-vertex/24-tetrahedron mapping cylinder, and '
                'fail-closed whole-mesh SSP1 plan admission'
            ),
            'not_claimed': (
                'runtime whole-mesh success until the emitted plan passes the '
                'OMP, topology, intersection, and exact-once cloud campaign'
            ),
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(payload['verdict'])
    print(disposition)
    if not event_ir_pass:
        for name, passed in checks.items():
            if not passed:
                print('FAILED:', name)
        return 2
    if args.require_whole_mesh_ready and not whole_mesh_ready:
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
