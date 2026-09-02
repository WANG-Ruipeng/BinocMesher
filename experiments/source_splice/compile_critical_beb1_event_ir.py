#!/usr/bin/env python3
'''Compile the selected production saddle into a source-labelled BEB1 Event IR.

This is deliberately an admission compiler, not a coordinate-welding shim.
It labels every one-sided slice vertex of the TV3 tetrahedral half-handle as
either an ordinary source-edge trajectory or an unresolved critical side
seam.  A whole-mesh SSP1 plan is admissible only when every boundary edge is
source-resolved after event-star closure.
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

from theory_audit import face_segments, mesh_topology  # type: ignore
from processed_mesh import selected_event_rows


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
    slices = {
        name: slice_block(
            vertices4, tets, exact_times, sources, tau, event_id)
        for name, tau in probes.items()
    }
    registry_traces = {
        name: registry_face_trace(rows, probes[name])
        for name in ('lower', 'upper')
    }
    interface_agreement = {
        name: (
            registry_traces[name]['agrees_across_registry_incidences'] and
            block_source_boundary_segments(slices[name]) ==
            registry_traces[name]['canonical_segments']
        )
        for name in ('lower', 'upper')
    }

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
        actual = slices[name]['topology']
        return all(
            int(actual.get(key, -1)) == int(expected.get(key, -2))
            for key in topology_keys
        )

    resolved_cells = tv4.get('resolved_logical_cells', {})
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
            'PASS_TV4_PRODUCTION_DERIVED_OFFLINE_GLOBAL_SPLICE'
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
        'beb1_component_transition_2_to_1': (
            slices['lower']['topology']['components'] == 2 and
            slices['upper']['topology']['components'] == 1
        ),
        'one_sided_slices_are_regular': all(
            slices[name]['topology']['nonmanifold_edges'] == 0 and
            slices[name]['topology']['duplicate_faces'] == 0
            for name in ('lower', 'upper')
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
    interface_ready = all(interface_agreement.values())
    source_boundary_ready = all(
        slices[name]['source_boundary_complete']
        for name in ('lower', 'critical', 'upper')
    )
    whole_mesh_ready = (
        event_ir_pass and interface_ready and source_boundary_ready)
    disposition = (
        'READY_FOR_WHOLE_MESH_SPLICE'
        if whole_mesh_ready else
        'SINGULAR_UNRESOLVED_SIDE_TRACE'
    )
    payload = {
        'schema': 'binoc-critical-beb1-event-ir-v1',
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
            'kind': 'BEB1_TWO_TETRAHEDRON_RELATIVE_HALF_HANDLE',
            'block_sha256': stable_hash(block_payload),
            'vertices4': block_payload['vertices4'],
            'tets': block_payload['tets'],
            'block_vertex_source_hvids': sources,
            'block_vertex_exact_times': block_payload['exact_times'],
            'critical_link': capsule['critical_link'],
        },
        'slices': slices,
        'registry_interface_traces': registry_traces,
        'checks': checks,
        'admission': {
            'event_ir_certified': event_ir_pass,
            'all_boundary_edges_source_resolved': source_boundary_ready,
            'block_registry_interface_agreement': interface_agreement,
            'unresolved_boundary_edge_counts': {
                name: len(value['unresolved_boundary_edges'])
                for name, value in slices.items()
            },
            'policy': (
                'Do not emit an SSP1 whole-mesh plan until event-star closure '
                'cancels every critical side seam and every remaining patch '
                'boundary edge is carried by ordinary SourceVID trajectories.'
            ),
        },
        'scope': {
            'completed': (
                'production-selected 104/5 BEB1 event identity, registry batch '
                'closure, source-labelled exact one-sided block slices, and '
                'explicit relative-boundary admission status'
            ),
            'not_claimed': (
                'whole-mesh critical block insertion while any side-trace '
                'boundary remains unresolved'
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
