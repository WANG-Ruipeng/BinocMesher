#!/usr/bin/env python3
"""Read-only time-window admission preflight; never emit a runtime plan."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from compile_critical_beb1_event_ir import compile_ordinary_patch
from run_geometry_gate import require
from sidewall_contract import require_window_boundary_contract


def rational(value):
    return Fraction(value['numerator'], value['denominator'])


def slice_triangle(vertices4, indices, times, tau):
    points = []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        a, b = indices[i], indices[j]
        ta, tb = times[a], times[b]
        if ta == tau:
            points.append(vertices4[a, :3])
        if min(ta, tb) < tau < max(ta, tb):
            weight = float((tau-ta)/(tb-ta))
            points.append((1-weight)*vertices4[a, :3] + weight*vertices4[b, :3])
    return points


def segment_distance(point, a, b):
    direction = b-a
    norm2 = float(direction @ direction)
    require(norm2 > 0, 'Degenerate ordinary boundary edge.')
    weight = np.clip((point-a) @ direction / norm2, 0, 1)
    return float(np.linalg.norm(point-(a+weight*direction)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-ir', required=True, type=Path)
    parser.add_argument('--cache-root', required=True, type=Path)
    parser.add_argument('--geometry-result', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'schema': 'binoc-window-boundary-preflight-v1', 'stage': 2,
        'status': 'RUNNING', 'measurements': [], 'stage_3_started': False,
        'runtime_plan_emitted': False,
        'purpose': 'Slice the actual admitted triangulated side wall at slab midpoints and compare it to source-labeled ordinary patch boundary edges.',
        'limitations': 'Passing finite probes would not certify the entire window. No whole-mesh changes are made.'}
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    save()
    try:
        gate = json.loads(args.geometry_result.read_text())
        require(gate['status'] == 'PASS_STAGE_1', 'Stage 1 has not passed.')
        ir = json.loads(args.event_ir.read_text())
        source_ir = json.loads((Path(gate['event_root']) / 'critical_beb1_event_ir.json').read_text())
        require(ir['event']['event_id'] == source_ir['event']['event_id'] and
                ir['event']['root'] == source_ir['event']['root'] and
                ir['event_star_geometry']['critical_patch'] == source_ir['event_star_geometry']['critical_patch'],
                'Event identity or measured root geometry changed after stage 1.')
        require(ir['whole_mesh_splice_ready'] is True, 'Root event not admitted.')
        cylinder = ir['event_star_geometry']['mapping_cylinder']
        require(cylinder['pass'] is True, 'Mapping cylinder failed existing audits.')
        report['event_id'] = ir['event']['event_id']
        report['event_ir_sha256'] = hashlib.sha256(args.event_ir.read_bytes()).hexdigest()
        report['geometry_result_sha256'] = hashlib.sha256(args.geometry_result.read_bytes()).hexdigest()
        report['exact_sidewall_boundary_contract'] = require_window_boundary_contract(cylinder)
        levels = [rational(cylinder['levels'][name]) for name in ['lower', 'critical', 'upper']]
        times = [t for t in levels for _ in range(5)]
        vertices4 = np.asarray(cylinder['vertices4'], dtype=float)
        cycle = cylinder['boundary_cycle']
        require(vertices4.shape == (15, 4) and len(cycle) == 4, 'Unexpected cylinder layout.')
        # Roundoff-screening tolerance only; never use it to weld or move vertices.
        tolerance = 1e-7 * max(1., float(np.linalg.norm(np.ptp(vertices4[:, :3], axis=0))))
        report['numeric_screening_tolerance'] = tolerance
        for slab in [0, 1]:
            tau = (levels[slab]+levels[slab+1])/2
            patch, _ = compile_ordinary_patch(args.cache_root.resolve(), tau, event_id=ir['event']['event_id'])
            require(set(patch['boundary_cycle']) == set(cycle), 'SourceVID boundary changes inside the window.')
            positions = {key: np.asarray(value, dtype=float) for key, value in patch['boundary_positions'].items()}
            records = []
            for face in cylinder['side_trace_faces']:
                points = slice_triangle(vertices4, face, times, tau)
                if not points:
                    continue
                pair = sorted({index % 5 for index in face})
                require(len(pair) == 2 and max(pair) < 4, 'Side-wall triangle is not assigned to one source boundary edge.')
                a, b = positions[cycle[pair[0]]], positions[cycle[pair[1]]]
                for point in points:
                    records.append({'side_face': face, 'point': point.tolist(),
                        'boundary_source_vids': [cycle[pair[0]], cycle[pair[1]]],
                        'baseline_edge_endpoints': [a.tolist(), b.tolist()],
                        'distance_to_baseline_edge': segment_distance(point, a, b)})
            require(bool(records), 'No side-wall slice found.')
            worst = max(records, key=lambda item:item['distance_to_baseline_edge'])
            maximum = worst['distance_to_baseline_edge']
            report['measurements'].append({'slab': slab,
                'time': {'numerator':tau.numerator, 'denominator':tau.denominator},
                'slice_endpoint_checks': len(records),
                'maximum_boundary_distance': maximum,
                'worst_witness': worst,
                'gate_pass': maximum <= tolerance})
            save()
            require(maximum <= tolerance,
                    'Sliced mapping-cylinder side wall does not match ordinary boundary geometry at an intermediate time. Stop before SSP1 emission and rendering.')
        report['status'] = 'PASS_PREFLIGHT_ONLY'
        save()
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        report['status'] = 'STOP_STAGE_2'
        report['stop_reason'] = str(error)
        save()
        print(json.dumps(report, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
