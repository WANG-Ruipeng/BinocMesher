#!/usr/bin/env python3
"""Bounded, fresh-cache Forest event census. No scene generation or rendering.

The fixed pilot reuses seed 0's saved scene and takes frames 1--24. It keeps
the Forest96 pilot's mesher settings, not the paper's finer 3px setting.
Only the worker imports Blender. The supervisor can stop its process group.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

SCHEMA = 'c1-lite-forest-census-pilot-v1'
FIRST_FRAME, LAST_FRAME, FPS = 1, 24, 24
WALL_SECONDS = 1200
DISK_LIMIT = 2_000_000_000
DISK_STOP = 1_500_000_000
FILE_LIMIT = 128 * 1024 * 1024
DEFAULT_REPO = Path('/home/warpwang/src/BinocMesher')
DEFAULT_SOURCE = Path('/home/warpwang/runs/forest96-four-baseline-pilot-20260905/binoc/0')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path, data):
    path = Path(path)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def regular_files(root):
    """Never traverse symlinks: source assets must be a self-contained tree."""
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders + files:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f'Symlink not permitted in monitored tree: {Path(directory)/name}')
        for name in files:
            yield Path(directory) / name


def disk_bytes(root):
    total = 0
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [name for name in folders if not (Path(directory)/name).is_symlink()]
        for name in files:
            path = Path(directory)/name
            try:
                total += path.lstat().st_size
            except FileNotFoundError:
                pass  # mesher cleanup may finish between listing and stat
    return total


def require_fresh_output(output, source, repo):
    output, source, repo = map(lambda p: Path(p).resolve(), (output, source, repo))
    if output.exists():
        raise FileExistsError(output)
    for protected in (source, repo):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f'Output overlaps protected input: {protected}')
    return output


def source_fingerprints(source):
    paths = [source/'coarse/scene.blend', source/'coarse/MaskTag.json',
             source/'logs/operative_gin_fineterrain.txt', source/'run_pipeline.sh']
    paths.extend(regular_files(source/'coarse/assets'))
    paths.extend(regular_files(source/'fine/HyperMesh'))
    return {str(path.relative_to(source)): {'bytes': path.stat().st_size,
             'sha256': sha256(path)} for path in sorted(paths)}


def fraction_json(value):
    value = Fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def summarize_registry(path):
    """Only census registry identities; this does not classify/certify BEB1."""
    path = Path(path)
    if not path.is_file():
        return {'status': 'MISSING_REGISTRY', 'raw_observations': None}
    if path.stat().st_size > FILE_LIMIT:
        raise ValueError('Registry exceeds predeclared read budget.')
    groups = defaultdict(lambda: {'rows': 0, 'roots': set(), 'logical': set()})
    raw_ids, roots, logical_ids = set(), set(), set()
    count = 0
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'raw_id', 'canonical_event_id', 'root_num', 'root_den', 'logical_incidence_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Unexpected registry schema.')
        for row in reader:
            root = Fraction(int(row['root_num']), int(row['root_den']))
            count += 1
            roots.add(root)
            logical_ids.add(row['logical_incidence_id'])
            raw_ids.add(row['raw_id'])
            group = groups[row['canonical_event_id']]
            group['rows'] += 1
            group['roots'].add(root)
            group['logical'].add(row['logical_incidence_id'])
    if any(len(group['roots']) != 1 for group in groups.values()):
        raise ValueError('A canonical registry ID has inconsistent exact roots.')
    events = [{'canonical_event_id': event_id,
               'root_internal': fraction_json(next(iter(group['roots']))),
               'raw_observations': group['rows'],
               'logical_incidences': len(group['logical']),
               'beb1_classification': 'NOT_COMPILED',
               'window_admission': 'NOT_ATTEMPTED'}
              for event_id, group in sorted(groups.items(),
                  key=lambda item: (next(iter(item[1]['roots'])), item[0]))]
    return {'status': 'REGISTRY_CENSUS_COMPLETE', 'path': str(path),
            'sha256': sha256(path), 'bytes': path.stat().st_size,
            'raw_observations': count, 'unique_raw_observation_ids': len(raw_ids),
            'logical_incidences': len(logical_ids), 'exact_roots': len(roots),
            'canonical_events': len(groups), 'events': events,
            'distinct_roots_internal': [fraction_json(root) for root in sorted(roots)],
            'scope': 'Per cache/element namespace; not a six-scene or BEB1 coverage denominator.'}


def forest_time_mapping(camera_times, fading_time=1.0):
    """Use actual get_caminfo output + core.py/coarse_step.cpp formulas.

    This records the executed Python input values, not serialized C++ params.
    No four-demo timing constants or profile are used.
    """
    origin = min(camera_times)
    shifted = [float(t-origin) for t in camera_times]
    duration = max(shifted)+1e-5
    steps = [b-a for a, b in zip(sorted(set(shifted)), sorted(set(shifted))[1:])]
    effective_fading = max(fading_time, min(steps))
    level = 0
    while effective_fading*(1 << (level+1)) < duration:
        level += 1
    maximum = 2 << level
    return {'status': 'RECONSTRUCTED_FROM_EXECUTED_FOREST_CAMERA_INPUT_NOT_CPP_SERIALIZED',
            'origin_seconds': fraction_json(Fraction.from_float(origin)),
            'duration_seconds': fraction_json(Fraction.from_float(duration)),
            'delta_seconds': fraction_json(Fraction.from_float(duration/maximum)),
            'maximum_discrete_time': maximum, 'temporal_group_count': 1 << level,
            'natural_frame_formula': '(frame_number - 0.5) / 24',
            'min_t_offset': 0, 'use_alignment': False,
            'effective_fading_seconds': effective_fading,
            'limitation': 'Input-value reconstruction; no runtime exact-root or continuous-window admission.'}


def worker(args):
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_LIMIT, FILE_LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    repo, source, output = args.repo.resolve(), args.source.resolve(), args.output.resolve()
    infinigen = repo/'infinigen_binocmesher'
    os.chdir(infinigen)
    sys.path.insert(0, str(infinigen))
    # A small private asset working copy prevents lazy Terrain writes reaching
    # the source scene. The much larger scene.blend is loaded in place, not copied.
    assets = source/'coarse/assets'
    asset_size = sum(p.stat().st_size for p in regular_files(assets))
    if asset_size > 100_000_000:
        raise ValueError('Private asset copy exceeds fixed 100 MB allowance.')
    shutil.copytree(assets, output/'assets')
    print(f'Private assets ready: {asset_size} bytes', flush=True)

    import bpy
    import gin
    import numpy as np
    # Registers exactly the original nature/terrain gin selectors, but does not
    # invoke main(), execute_tasks(), scene generation, or rendering.
    from infinigen_examples import generate_nature  # noqa: F401
    from infinigen.core import init, surface
    from infinigen.core.placement import camera as cam_util
    from infinigen.core.tagging import tag_system
    from infinigen.terrain.core import Terrain, UntexturedBinocMesher
    from infinigen.terrain.utils import get_caminfo
    scene_seed = init.apply_scene_seed('0')
    mandatory = [Path('infinigen_examples/configs_nature/scene_types')]
    overrides = [
        'scene.upsidedown_mountains_chance=0', 'scene.sdf_trees_chance=1',
        'nishita_lighting.sun_elevation=20', 'nishita_lighting.sun_rotation=0',
        'shader_atmosphere.density=0.0005', 'shader_atmosphere.anisotropy=0.8',
        'fine_terrain.mesher_backend="BinocMesher"',
        'compose_nature.load_cameras="../infinigen_example_scenes/forest.txt"',
        'BinocMesher.pixels_per_cube=6', 'Terrain.device="cpu"',
        'execute_tasks.generate_resolution=[960,540]',
        f'execute_tasks.frame_range=[1,{LAST_FRAME}]', 'execute_tasks.camera_id=[0,0]',
    ]
    init.apply_gin_configs(configs=['base_nature.gin', 'mountain', 'monocular', 'simple', 'no_assets'],
        overrides=overrides, config_folders='infinigen_examples/configs_nature',
        mandatory_folders=mandatory, mutually_exclusive_folders=mandatory)
    bpy.ops.wm.open_mainfile(filepath=str(source/'coarse/scene.blend'))
    tag_system.load_tag(path=str(source/'coarse/MaskTag.json'))
    bpy.context.scene.frame_start = FIRST_FRAME
    bpy.context.scene.frame_end = LAST_FRAME
    bpy.context.scene.frame_set(FIRST_FRAME)
    bpy.context.scene.render.fps = FPS
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 540
    bpy.context.view_layer.update()
    surface.registry.initialize_from_gin()
    init.configure_blender()
    cam_util.set_active_camera(0, 0)
    cameras = [cam_util.get_camera(i, j) for i, j in cam_util.get_cameras_ids()]
    caminfo = get_caminfo(cameras, fs=FIRST_FRAME, fe=LAST_FRAME)[0]
    if len(cameras) != 1 or len(caminfo[4]) != LAST_FRAME:
        raise ValueError('Expected the original one-camera Forest trajectory.')
    camera_payload = {'camera_ids': [list(v) for v in cam_util.get_cameras_ids()],
        'poses': np.asarray(caminfo[0]).tolist(), 'intrinsics': np.asarray(caminfo[1]).tolist(),
        'heights': caminfo[2], 'widths': caminfo[3], 'times_seconds': caminfo[4],
        'time_mapping': forest_time_mapping(caminfo[4]), 'get_caminfo_relax': 1.05}
    write_json_new(output/'camera_inputs.json', camera_payload)
    if 'atmosphere' not in bpy.data.objects:
        raise ValueError('Saved scene does not identify the expected Forest terrain.')
    import pickle
    with (output/'assets/info.pickle').open('rb') as handle:
        info = pickle.load(handle)
    terrain = Terrain(scene_seed, surface.registry, task=['fine_terrain'],
        on_the_fly_asset_folder=output/'assets', height_offset=info['height_offset'],
        whole_bbox=info['whole_bbox'])
    terrain.sample_surface_templates()
    terrain.surfaces_into_sdf()
    write_json_new(output/'effective_inputs.json', {
        'mesher_python': str(Path(sys.modules[UntexturedBinocMesher.__module__].__file__).resolve()),
        'core_so': str(repo/'binocmesher/lib/core.so'),
        'core_so_sha256': sha256(repo/'binocmesher/lib/core.so'),
        'terrain_elements': [element.__class__.name for element in terrain.elements_list],
        'bounds': list(terrain.bounds), 'overrides': overrides,
        'note': 'Existing mesh_extraction; write_attribute_enabled=False skips only material/tag evaluation, not SDF geometry.'})
    print('Starting unchanged Forest mesh_extraction; cache only, no scene export/render.', flush=True)
    started = time.monotonic()
    meshes, _ = terrain.mesh_extraction(mesher_backend='BinocMesher', cameras=cameras,
        hypermesh_path=output/'HyperMesh', fs=FIRST_FRAME, fe=LAST_FRAME,
        write_attribute_enabled=False)
    (output/'operative_gin.txt').write_text(gin.operative_config_str(), encoding='utf-8')
    names = sorted(meshes)
    del meshes
    write_json_new(output/'worker_complete.json', {'status': 'GEOMETRY_BUILD_COMPLETE',
        'mesh_extraction_seconds': time.monotonic()-started, 'mesh_names': names,
        'rendering_started': False, 'scene_exported': False,
        'ordinary_initial_slice': 'Existing core.__call__ emits first slice in RAM; no intervention plan loaded.'})
    return 0


def stop_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def supervise(args):
    repo, source = args.repo.resolve(), args.source.resolve()
    output = require_fresh_output(args.output, source, repo)
    report_dir = args.report_dir.resolve()
    previous_stage = None
    previous_gate = None
    if args.stage == 'forest64':
        if args.previous_stage is None:
            raise ValueError('forest64 requires the completed first-stage summary.')
        previous_stage = json.loads(args.previous_stage.read_text(encoding='utf-8'))
        previous_protocol = json.loads((args.previous_stage.parent/'protocol.json').read_text())
        prior_cap_path = Path(previous_protocol['output'])/'HyperMesh/OpaqueTerrain/hyperpoly_meta/0.bin'
        confirmed_cap_stop = (
            args.allow_confirmed_file_limit_stop and
            previous_stage.get('status') == 'STOP_BUILD_OR_CONTRACT_FAILURE' and
            previous_stage.get('worker_exit_code') == 1 and
            previous_protocol.get('per_file_limit_bytes') == 128*1024**2 and
            prior_cap_path.stat().st_size == 128*1024**2 and
            'failed to write hyperpoly provenance record' in
                (args.previous_stage.parent/'worker_tail.log').read_text(errors='replace'))
        if previous_stage.get('source_inputs_sha256_unchanged') is not True:
            raise ValueError('Prior source integrity check failed.')
        if previous_stage.get('status') != 'COMPLETE_REGISTRY_CENSUS_ONLY' and not confirmed_cap_stop:
            raise ValueError('Prior failure is not the explicitly authorized file-limit-only stop.')
        previous_gate = 'CONFIRMED_SELF_IMPOSED_FILE_LIMIT_STOP' if confirmed_cap_stop else 'COMPLETE'
        if previous_protocol.get('frames_inclusive') != [1, 24]:
            raise ValueError('Expected the fixed first-stage 24-frame protocol.')
    if report_dir.exists():
        raise FileExistsError(report_dir)
    before = source_fingerprints(source)
    output.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    driver_snapshot = output/'driver_snapshot.py'
    shutil.copyfile(Path(__file__).resolve(), driver_snapshot)
    protocol = {'schema': SCHEMA, 'source_scene_root': str(source), 'output': str(output),
        'seed': 0, 'frames_inclusive': [FIRST_FRAME, LAST_FRAME], 'fps': FPS,
        'previous_stage_gate': previous_gate,
        'stage': args.stage,
        'previous_stage_summary_sha256': sha256(args.previous_stage) if previous_stage else None,
        'resolution': [960, 540], 'opaque_pixels_per_cube': 6,
        'coarse_pixels_per_cube': 30, 'outview_pixels_per_cube': 120,
        'atmosphere_pixels_per_cube': 100, 'fading_time': 1, 'omp_threads': 8,
        'maximum_worker_wall_seconds': WALL_SECONDS, 'disk_budget_bytes': DISK_LIMIT,
        'early_disk_stop_bytes': DISK_STOP, 'per_file_limit_bytes': FILE_LIMIT,
        'registry_enabled': True, 'provenance_enabled': True,
        'build_script_sha256': sha256(driver_snapshot), 'rendering_permitted': False,
        'selection': f'Earliest fixed {LAST_FRAME} Forest frames; no outcome-dependent reselection.',
        'limitations': ['Shortened camera window changes temporal subdivision/cache and is not the old 96-frame cache.',
            '6px matches the existing pilot, not the paper 3px final setting.',
            f'Disk supervised every 0.2 seconds with {DISK_LIMIT-DISK_STOP} byte reserve and {FILE_LIMIT} byte per-file cap; not an OS directory quota.',
            'No production C1 transition, eligibility certificate, geometry quality comparison, or rendering.']}
    write_json_new(output/'protocol.json', protocol)
    write_json_new(report_dir/'protocol.json', protocol)
    write_json_new(report_dir/'source_fingerprints_before.json', before)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('BINOC_SOURCE_SPLICE') or key.startswith('BINOC_EVENT_') or key.startswith('BINOC_PROVENANCE'):
            del env[key]
    env.update({'BINOC_EVENT_MODE': '1', 'BINOC_PROVENANCE_V2': '1',
        'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'BINOC_MAX_SERIALIZED_CACHE_RECORDS': '100000000',
        'BINOC_MAX_SERIALIZED_CACHE_PAYLOAD_BYTES': str(FILE_LIMIT),
        'LD_LIBRARY_PATH': '/usr/lib/wsl/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')})
    command = [sys.executable, '-B', str(driver_snapshot), '--worker',
        '--repo', str(repo), '--source', str(source), '--output', str(output), '--stage', args.stage]
    started = time.monotonic()
    reason, peak = None, disk_bytes(output)
    process = None
    try:
        with (output/'worker.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=repo/'infinigen_binocmesher',
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'status': 'STARTED', 'pid': process.pid,
                'output': str(output), 'fixed_frames': [1, LAST_FRAME]}), flush=True)
            next_notice = started
            while process.poll() is None:
                current = disk_bytes(output)
                peak = max(peak, current)
                elapsed = time.monotonic()-started
                if current >= DISK_STOP:
                    reason = 'STOP_DISK_RESERVE_THRESHOLD'
                elif elapsed >= WALL_SECONDS:
                    reason = 'STOP_WALL_BUDGET'
                if reason:
                    stop_group(process)
                    break
                if time.monotonic() >= next_notice:
                    print(json.dumps({'status': 'RUNNING', 'elapsed_seconds': round(elapsed, 1),
                        'bytes': current}), flush=True)
                    next_notice = time.monotonic()+30
                time.sleep(0.2)
    except BaseException:
        if process is not None:
            stop_group(process)
        raise
    elapsed = time.monotonic()-started
    after = source_fingerprints(source)
    unchanged = after == before
    caches = []
    for cache in sorted((output/'HyperMesh').glob('*')):
        if not cache.is_dir():
            continue
        manifest = cache/'slicing_preprocess.manifest.json'
        completion = cache/'slicing_preprocess.finish'
        metadata = json.loads(manifest.read_text()) if manifest.is_file() else None
        expected = metadata is not None and all(metadata.get(k) is True for k in
            ('event_registry_enabled', 'provenance_enabled', 'provenance_requested'))
        try:
            census = summarize_registry(cache/'event_registry_p1.csv')
        except Exception as error:
            census = {'status': 'STOP_REGISTRY_PARSE_FAILURE',
                      'error': type(error).__name__+': '+str(error)}
        caches.append({'cache_name': cache.name, 'cache_path': str(cache),
            'completed': completion.is_file() and expected,
            'manifest': metadata, 'registry': census,
            'processed_owner_sidecars': len(list((cache/'processed_hyperpolys').glob('*_hpmeta.bin')))})
    complete = (output/'worker_complete.json').is_file() and process.returncode == 0
    status = reason or ('COMPLETE_REGISTRY_CENSUS_ONLY' if complete and caches and
        all(c['completed'] and c['registry']['status'] == 'REGISTRY_CENSUS_COMPLETE' for c in caches)
        else 'STOP_BUILD_OR_CONTRACT_FAILURE')
    if not unchanged:
        status = 'STOP_SOURCE_INPUT_CHANGED'
    final_bytes = disk_bytes(output)
    peak = max(peak, final_bytes)
    if peak >= DISK_LIMIT:
        status = 'STOP_DISK_BUDGET_EXCEEDED'
    report = {'schema': SCHEMA, 'status': status, 'worker_exit_code': process.returncode,
        'worker_wall_seconds': elapsed, 'peak_observed_output_bytes': peak,
        'final_output_bytes': final_bytes, 'source_inputs_sha256_unchanged': unchanged,
        'build_complete': complete, 'caches': caches,
        'canonical_count_across_cache_namespaces': sum(c['registry'].get('canonical_events', 0) for c in caches),
        'rendering_started': False, 'full_method_admission': 'NOT_ATTEMPTED',
        'beb1_coverage': 'NOT_ESTIMATED',
        'limitation': 'Only completed registries are a census; missing/partial caches are not zero events. No six-scene inference or full-window success claimed.'}
    write_json_new(output/'summary.json', report)
    write_json_new(report_dir/'summary.json', report)
    for name in ('camera_inputs.json', 'effective_inputs.json', 'worker_complete.json', 'operative_gin.txt'):
        path = output/name
        if path.is_file():
            shutil.copyfile(path, report_dir/name)
    with (output/'worker.log').open('rb') as handle:
        handle.seek(max(0, (output/'worker.log').stat().st_size-128*1024))
        (report_dir/'worker_tail.log').write_bytes(handle.read())
    print(json.dumps({key: report[key] for key in ('status', 'worker_wall_seconds',
        'final_output_bytes', 'canonical_count_across_cache_namespaces', 'source_inputs_sha256_unchanged')}), flush=True)
    return 0 if status == 'COMPLETE_REGISTRY_CENSUS_ONLY' else 2


def main():
    global LAST_FRAME, WALL_SECONDS, DISK_LIMIT, DISK_STOP, FILE_LIMIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['forest24', 'forest64'], default='forest24')
    parser.add_argument('--previous-stage', type=Path)
    parser.add_argument('--allow-confirmed-file-limit-stop', action='store_true')
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == 'forest64':
        LAST_FRAME = 64
        WALL_SECONDS = 45*60
        DISK_LIMIT = 3*1024**3
        DISK_STOP = 5*1024**3//2
        FILE_LIMIT = 512*1024**2
    if not args.worker and args.report_dir is None:
        parser.error('--report-dir is required for a supervised run')
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
