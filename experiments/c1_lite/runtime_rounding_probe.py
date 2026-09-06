#!/usr/bin/env python3
"""Prospective E2 binary32 fan at the original 33 times; never emits a plan.

The first boundary coordinate mismatch/ambiguity stops this numerical path.
All mesh arrays remain in RAM and the mutable cache is temporary.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import window_source as source
from anchor_ablation import fixed_times
from run_window_audit import center_at
from runtime_baseline_audit import exact_call, manifest
from runtime_common import initialize_mesher, canonical_mesh_hash, canonical_face
from runtime_breakpoint_crosscheck import round_fraction_binary32, sha, write_fresh

EVENT = 'event-02-88ade47aa4fa'
MAX_CACHE_BYTES = 10*1024*1024


class ProbeStop(RuntimeError):
    pass


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def exact_point(values):
    return tuple(Fraction.from_float(float(x)) for x in values)


def boundary_lookup(vertices, expected):
    lookup = defaultdict(list)
    for i, point in enumerate(vertices):
        lookup[tuple(float(x) for x in point)].append(i)
    matches = [lookup[tuple(float(x) for x in point)] for point in expected]
    counts = [len(row) for row in matches]
    return {'status': 'UNIQUE_COORDINATE_MATCH_ONLY' if all(n == 1 for n in counts)
            else 'STOP_BOUNDARY_COORDINATE_MISMATCH_OR_AMBIGUITY',
            'coordinate_match_counts': counts, 'coordinate_match_ids': matches,
            'source_identity_proof': False}


def endpoint_contract(center, a, b):
    center, a, b = map(exact_point, (center, a, b))
    direction, offset = sub(b, a), sub(center, a)
    collinear = all(x == 0 for x in cross(direction, offset))
    norm2 = sum(x*x for x in direction)
    dot = sum(x*y for x, y in zip(direction, offset))
    interior = norm2 > 0 and 0 < dot < norm2
    return {'collinear_exact': collinear, 'strictly_inside_diagonal': collinear and interior,
        'cross_exact': [source.fj(x) for x in cross(direction, offset)],
        'arithmetic': 'Exact Fractions of prospective binary32 center and actual baseline endpoint coordinates.'}


def fan_metrics(boundary, center, ideal_boundary, ideal_center):
    actual = [exact_point(p) for p in boundary]
    c = exact_point(center)
    twice_areas, area_squared, signed_xy, errors = [], [], [], []
    for i in range(4):
        normal = cross(sub(actual[i], c), sub(actual[(i+1)%4], c))
        ideal = cross(sub(ideal_boundary[i], ideal_center), sub(ideal_boundary[(i+1)%4], ideal_center))
        q = sum(x*x for x in normal)
        area_squared.append(source.fj(q/4))
        twice_areas.append(float(q)**0.5)
        signed_xy.append(source.fj(normal[2]))
        errors.append(max(abs(x-y) for x, y in zip(normal, ideal)))
    center_error = max(abs(x-y) for x, y in zip(c, ideal_center))
    boundary_error = max(abs(x-y) for p, q in zip(actual, ideal_boundary) for x,y in zip(p,q))
    return {'fan_triangle_area_approx': [a/2 for a in twice_areas],
        'fan_triangle_area_squared_exact': area_squared,
        'fan_signed_twice_xy_area_exact': signed_xy,
        'all_fan_triangles_nonzero_exact': all(source.fr(q) > 0 for q in area_squared),
        'center_max_coordinate_quantization_error': source.fj(center_error),
        'boundary_max_coordinate_quantization_error': source.fj(boundary_error),
        'max_fan_normal_component_error': source.fj(max(errors)),
        'meaning': 'Finite-time prospective fan numerical diagnostics; not continuous certification or rendered/terrain quality.'}


def source_faces_at(report, tau):
    for point in report['breakpoint_points']:
        if source.fr(point['time']) == tau:
            return point['source_faces']
    for segment in report['segments']:
        if source.fr(segment['t0']) < tau < source.fr(segment['t1']):
            return segment['source_faces']
    raise ProbeStop('Time has no frozen source branch or singleton.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo','cache-root','source-report','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    cache, output = args.cache_root.resolve(), args.output.resolve()
    if output.exists() or output == cache or cache in output.parents:
        raise ValueError('Output must be fresh and outside original cache.')
    protocol = json.loads(args.source_report.read_text())
    if args.source_report.parent.name != EVENT:
        raise ValueError('This probe only accepts fixed E2.')
    levels = tuple(source.fr(protocol['levels'][name]) for name in ('lower','root','upper'))
    if levels != (Fraction(102,5),Fraction(104,5),Fraction(106,5)):
        raise ValueError('Refuse changed E2 window.')
    times = fixed_times(levels)
    if any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError('Cache symlinks unsupported.')
    size = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
    if size > MAX_CACHE_BYTES:
        raise ValueError('Cache exceeds ten MiB.')
    before = manifest(cache)
    inventory = source.read_inventory(cache)
    if inventory.digest != protocol['cache_input_sha256']:
        raise ValueError('Frozen source/cache mismatch.')
    output.mkdir(parents=True)
    report = {'schema':'c1-lite-runtime-rounding-probe-v1','event':EVENT,'status':'RUNNING',
        'omp_threads':1,'mode':'raw','planned_times':[source.fj(t) for t in times],
        'center_rule':'Original A exact endpoint/root anchors, piecewise exact Fraction interpolation, exact RNE32.',
        'source_report_sha256':sha(args.source_report),'script_sha256':sha(__file__),
        'core_so_sha256':sha(args.repo/'binocmesher/lib/core.so'),
        'prospective_only':True,'runtime_plan_emitted':False,'runtime_admission':False,
        'render_started':False,'cache_copy_bytes':size,'cases':[]}
    started = time.monotonic(); temporary = None
    try:
        os.environ['OMP_NUM_THREADS']='1'
        for name in ('BINOC_SOURCE_SPLICE_PLAN','BINOC_SOURCE_SPLICE_AUDIT','BINOC_SOURCE_SPLICE_TRACE'):
            os.environ.pop(name,None)
        with tempfile.TemporaryDirectory(prefix='c1-rounding-e2-') as temporary:
            copied=Path(temporary)/'cache';shutil.copytree(cache,copied)
            mesher=initialize_mesher(args.repo.resolve(),copied)
            cycle=protocol['boundary_cycle']
            for tau in times:
                if time.monotonic()-started>180:
                    raise ProbeStop('Bounded probe time budget reached.')
                (v,f,tags),elapsed=exact_call(mesher,tau,False)
                ideal_boundary=[]
                for label in cycle:
                    vid=source.vid(label)
                    effective,coeff,_,_=source.source_formula(vid,tau,inventory.hypervertices)
                    if effective!=vid:
                        raise ProbeStop('Frozen boundary changes effective identity.')
                    ideal_boundary.append(source.evaluate(coeff,tau))
                expected=[[round_fraction_binary32(x) for x in point] for point in ideal_boundary]
                lookup=boundary_lookup(v,expected)
                case={'time':source.fj(tau),'baseline_vertices':len(v),'baseline_faces':len(f),
                      'baseline_mesh_hash':canonical_mesh_hash(v,f,tags),'slice_seconds':elapsed,
                      'expected_boundary_binary32':expected,'boundary_lookup':lookup}
                report['cases'].append(case)
                if lookup['status']!='UNIQUE_COORDINATE_MATCH_ONLY':
                    raise ProbeStop('Boundary ideal-RNE32 lookup is missing or ambiguous; no nearest/epsilon fallback.')
                indices=[row[0] for row in lookup['coordinate_match_ids']]
                boundary=v[indices]
                face_counts=Counter(canonical_face(face) for face in f)
                by_id=dict(zip(cycle,indices))
                case['source_patch_face_coordinate_match_counts']=[face_counts[canonical_face([by_id[x] for x in face])]
                                                                  for face in source_faces_at(protocol,tau)]
                if case['source_patch_face_coordinate_match_counts']!=[1,1]:
                    raise ProbeStop('Expected baseline source triangles are not uniquely present at matched coordinates.')
                ideal_center=tuple(source.fr(x) for x in center_at(protocol,tau))
                center=[round_fraction_binary32(x) for x in ideal_center]
                case['prospective_center_binary32']=center
                case['fan']=fan_metrics(boundary,center,ideal_boundary,ideal_center)
                if tau in (levels[0],levels[2]):
                    endpoint='lower' if tau==levels[0] else 'upper'
                    a,b=protocol['anchors'][endpoint]['source_diagonal']
                    case['endpoint_contract']=endpoint_contract(center,v[by_id[a]],v[by_id[b]])
                    if not case['endpoint_contract']['strictly_inside_diagonal']:
                        raise ProbeStop('Prospective center is not exactly inside actual baseline endpoint diagonal.')
                if not case['fan']['all_fan_triangles_nonzero_exact']:
                    raise ProbeStop('Prospective rounded fan has an exactly zero-area triangle.')
                print(json.dumps({'time':str(tau),'status':'PASS_FINITE_NUMERICAL_PROBE'}),flush=True)
        report['status']='COMPLETE_33_FINITE_NUMERICAL_PROBES_ONLY'
    except Exception as error:
        report['status']='STOP_NUMERICAL_MODEL_PATH';report['reason']=str(error)
    finally:
        report['original_cache_unchanged']=manifest(cache)==before
        report['temporary_cache_removed']=temporary is not None and not Path(temporary).exists()
        report['elapsed_seconds']=time.monotonic()-started
        if not report['original_cache_unchanged'] or not report['temporary_cache_removed']:
            report['status']='STOP_ISOLATION_CHECK'
        payload=json.dumps(report,indent=2,sort_keys=True)
        if len(payload.encode())>1024*1024:
            raise RuntimeError('Probe JSON exceeds one MiB.')
        write_fresh(output/'result.json',report)
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),'reason':report.get('reason'),
                      'output':str(output/'result.json')}),flush=True)
    return 0 if report['status']=='COMPLETE_33_FINITE_NUMERICAL_PROBES_ONLY' else 2


if __name__=='__main__':
    raise SystemExit(main())
