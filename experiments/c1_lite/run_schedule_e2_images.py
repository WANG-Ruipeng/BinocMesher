#!/usr/bin/env python3
"""Bounded actual E2 schedule/coherence plus geometry-image worker; no RGB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from e2_image_metrics import (terrain_reference, terrain_normals, angles, stats,
                             image_metrics, image_change, reference_check, gain_check)
from e2_render_views import (diagnostic_camera, camera_at, json_camera, window_points,
                             project, roi_rectangle, world_hits, save_geometry_previews,
                             save_difference)
from e2_raster import rasterize
from e2_schedule_coherence import prepare_schedule
from run_e2_runtime import manifest, mesh_hash


MAX_OUTPUT = 80*1024*1024
METHODS = ('raw', 'centroid_window', 'beb1_window')
SOURCE_SHA = '6b7f08fb7689ef8e62002c787e301a321fb343cda16d1ce37c54021b58c92dc2'
CORE_SHA = 'f4263a2f47ba5283175a921e49b8867998bdd8124aac810793b34242ec43a3c9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, record):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    payload = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False, default=convert)+'\n').encode()
    if len(payload) > 12*1024*1024:
        raise ValueError('Compact record exceeded 12 MiB.')
    with Path(path).open('xb') as handle:
        handle.write(payload)


def raster(mesh, camera):
    return rasterize(mesh[0], mesh[1], camera['c2w'], camera['K'],
                     width=camera['width'], height=camera['height'], near=1e-6)
def preview_changes(first, second):
    result = {}
    for name in ('depth', 'normal', 'silhouette'):
        with Image.open(first/(name+'.png')) as a, Image.open(second/(name+'.png')) as b:
            aa, bb = np.asarray(a), np.asarray(b)
            changed = aa != bb
            if changed.ndim == 3:
                changed = np.any(changed, axis=2)
            result[name+'_changed_pixels'] = int(changed.sum())
    result['meaning'] = 'Quantized PNG code-value changes, not a perceptual visibility test.'
    return result




def geometry_sheet(path, folders, labels, title):
    from PIL import ImageDraw
    width, height, label_height = 320, 180, 22
    sheet = Image.new('RGB', (width*len(folders), 40+3*(height+label_height)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet); draw.text((6, 5), title, fill='white')
    draw.text((6, 22), 'Half-size overview. Metrics use full-precision 640x360 buffers.', fill='white')
    for row, name in enumerate(('depth', 'normal', 'silhouette')):
        y = 40+row*(height+label_height)
        for col, (folder, label) in enumerate(zip(folders, labels)):
            with Image.open(Path(folder)/(name+'.png')) as source:
                panel = source.convert('RGB').resize((width, height), Image.Resampling.NEAREST)
            sheet.paste(panel, (col*width, y+label_height))
            draw.text((col*width+5, y+4), label+' / '+name, fill='white')
    sheet.save(path)


def run(args):
    output, cache, repo = args.output.resolve(), args.cache_root.resolve(), args.repo.resolve()
    source_path = args.source_report.resolve()
    if output.exists() or cache == output or cache in output.parents:
        raise ValueError('Fresh output outside the original cache required.')
    if sha(source_path) != SOURCE_SHA or sha(repo/'binocmesher/lib/core.so') != CORE_SHA:
        raise ValueError('Frozen source/library binding differs.')
    if any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError('Symlinked cache unsupported.')
    source = json.loads(source_path.read_bytes())
    before = manifest(cache)
    cache_bytes = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
    if cache_bytes > 10*1024*1024:
        raise ValueError('Only the fixed small E2 cache is supported.')
    output.mkdir(parents=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    report = {'schema': 'e2-schedule-image-validation-v1', 'status': 'PREPARING',
        'source_sha256': SOURCE_SHA, 'core_so_sha256': CORE_SHA,
        'frozen_commit': '000028e0fa5bc4922aeb51aa9fb9ee96b799301e',
        'continuous_window_admitted': False, 'rgb_rendered': False, 'cases': [],
        'natural_frame_indices_zero_based': [13, 14, 15, 16, 17],
        'exact_root_is_separate_diagnostic': True, 'methods': list(METHODS),
        'coordinate_buffers': 'camera-Z depth; world oriented flat normals; all faces without tag filtering',
        'reference': {'rounds': []}, 'timings': {}, 'input_cache_bytes': cache_bytes}
    temporary = None

    def checkpoint(stage):
        report['last_stage'] = stage
        elapsed = time.perf_counter()-started
        size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if elapsed > args.max_seconds or size > MAX_OUTPUT:
            raise RuntimeError('STOP_RESOURCE_BUDGET: '+stage)
        print(json.dumps({'stage': stage, 'elapsed_seconds': elapsed, 'output_bytes': size}), flush=True)

    try:
        os.environ['OMP_NUM_THREADS'] = '1'
        with tempfile.TemporaryDirectory(prefix='e2-schedule-images-') as temporary:
            copied = Path(temporary)/'cache'
            t = time.perf_counter(); shutil.copytree(cache, copied)
            report['timings']['cache_copy_seconds'] = time.perf_counter()-t
            checkpoint('prepare_actual_schedule')
            t = time.perf_counter(); prepared = prepare_schedule(repo, copied, source)
            report['timings']['schedule_prepare_seconds'] = time.perf_counter()-t
            dump(output/'structural_coherence.json', prepared['manifest'])
            cases = prepared['cases']
            if len(cases) != 6 or any(not set(METHODS) <= set(c['methods']) for c in cases):
                raise ValueError('Incomplete requested schedule/method set.')
            report['structural_coherence_status'] = prepared['manifest']['status']
            if report['structural_coherence_status'] != 'PASS_SCHEDULE_STRUCTURAL_COHERENCE_ONLY':
                raise ValueError('Requested structural manifest did not pass.')
            points, diag = window_points(source), diagnostic_camera(source)
            roi, roi_info = roi_rectangle(points, diag)
            z = project(points, diag)[1]
            depth_limits = [max(1e-6, float(z.min())-diag['span']), float(z.max())+diag['span']]
            report['diagnostic_camera'] = json_camera(diag)
            report['diagnostic_roi'] = roi_info
            report['depth_display_limits'] = {'diagnostic': depth_limits, 'original': [0., 30.]}
            Image.fromarray((roi*255).astype(np.uint8)).save(output/'diagnostic_roi.png')

            coarse = fine = None
            for resolution in (128, 256, 512):
                checkpoint('reference_grid_'+str(resolution))
                t = time.perf_counter(); vertices, faces = terrain_reference(diag['reference_bounds_xy'], resolution)
                build_seconds = time.perf_counter()-t
                t = time.perf_counter(); buffer = raster((vertices, faces, None), diag)
                raster_seconds = time.perf_counter()-t
                t = time.perf_counter(); selected, hits = world_hits(buffer, diag, buffer['mask'])
                gradient = terrain_normals(hits[:, :2], 1e-4)
                gradient_half = terrain_normals(hits[:, :2], 5e-5)
                gradient_check = stats(angles(gradient, gradient_half))
                buffer['normals'][selected] = gradient_half
                if gradient_check['mean'] is None or gradient_check['mean'] > .1:
                    raise RuntimeError('STOP_REFERENCE_GRADIENT_UNRESOLVED')
                row = {'resolution': resolution, 'vertices': len(vertices), 'faces': len(faces),
                    'height_queries': len(vertices), 'gradient_height_queries': 8*len(hits),
                    'build_seconds': build_seconds, 'raster_seconds': raster_seconds,
                    'gradient_seconds': time.perf_counter()-t, 'gradient_step_comparison': gradient_check,
                    'raster_stats': buffer['stats']}
                report['reference']['rounds'].append(row)
                save_geometry_previews(output/('reference_'+str(resolution)), buffer, depth_limits)
                if fine is not None:
                    coarse, fine = fine, buffer
                    refinement = reference_check(coarse, fine, roi)
                    report['reference']['refinement'] = refinement
                    if refinement['status'] == 'PASS_REFINEMENT_CHECK':
                        report['reference']['selected_resolution'] = resolution
                        break
                else:
                    fine = buffer
                del vertices, faces
            if coarse is None or report['reference'].get('refinement', {}).get('status') != 'PASS_REFINEMENT_CHECK':
                raise RuntimeError('STOP_REFERENCE_UNRESOLVED')

            delta = float.fromhex('0x1.eaabfa360338dp-6')
            root_physical = float(np.longdouble(104)*np.longdouble(delta)/np.longdouble(5))
            reference_folder = output/('reference_'+str(report['reference']['selected_resolution']))
            for case in cases:
                entry = {'key': case['key'], 'kind': case['kind'], 'frame': case['frame'],
                         'metadata': case.get('metadata', {}), 'views': {}, 'meshes': {}}
                report['cases'].append(entry)
                for name in METHODS:
                    v, f, tags = case['methods'][name]['mesh']
                    entry['meshes'][name] = {'vertices': len(v), 'faces': len(f), 'sha256': mesh_hash((v,f,tags)),
                                             'audit': case['methods'][name].get('audit', {})}
                entry['beb1_mesh_changed_vs_raw'] = entry['meshes']['beb1_window']['sha256'] != entry['meshes']['raw']['sha256']
                original = camera_at(case['frame'] if case['kind'] == 'natural' else 24*root_physical)
                for view_name, camera in [('original', original), ('diagnostic', diag)]:
                    checkpoint(case['key']+'_'+view_name)
                    current_roi, info = roi_rectangle(points, camera)
                    view = {'camera': json_camera(camera), 'roi': info, 'methods': {}, 'differences': {}}
                    entry['views'][view_name] = view
                    folder = output/view_name/case['key']; folder.mkdir(parents=True)
                    buffers = {}
                    for name in METHODS:
                        t = time.perf_counter(); actual = raster(case['methods'][name]['mesh'], camera)
                        seconds = time.perf_counter()-t; buffers[name] = actual
                        view['methods'][name] = {'raster_seconds': seconds, 'raster_stats': actual['stats']}
                        save_geometry_previews(folder/name, actual, depth_limits if view_name == 'diagnostic' else [0., 30.])
                        if view_name == 'diagnostic':
                            view['methods'][name]['reference_fine'] = image_metrics(actual, fine, current_roi)
                            view['methods'][name]['reference_coarse'] = image_metrics(actual, coarse, current_roi)
                    for first, second in [('beb1_window', 'raw'), ('centroid_window', 'raw'), ('beb1_window', 'centroid_window')]:
                        key = first+'_vs_'+second
                        view['differences'][key] = {'full_frame': image_change(buffers[first], buffers[second]),
                                                   'window_bbox_roi': image_change(buffers[first], buffers[second], current_roi)}
                        view['differences'][key]['quantized_preview'] = preview_changes(folder/first, folder/second)
                    source_rows = case.get('source_face_rows')
                    view['raw_visible_source_support_pixels'] = (int(np.isin(buffers['raw']['face_id'], source_rows).sum())
                                                                 if source_rows is not None else None)
                    if view_name == 'diagnostic':
                        for opponent in ('raw', 'centroid_window'):
                            view['differences']['beb1_window_vs_'+opponent]['reference_gain'] = gain_check(
                                view['methods'][opponent]['reference_fine'], view['methods']['beb1_window']['reference_fine'],
                                view['methods'][opponent]['reference_coarse'], view['methods']['beb1_window']['reference_coarse'])
                        source_rows = case.get('source_face_rows')
                        if source_rows is not None:
                            raw = case['methods']['raw']['mesh']
                            footprint = raster((raw[0], raw[1][list(source_rows)], None), camera)['mask']
                            Image.fromarray((footprint*255).astype(np.uint8)).save(folder/'source_footprint.png')
                            view['source_footprint_pixels'] = int(footprint.sum())
                            for name in METHODS:
                                view['methods'][name]['footprint_reference_fine'] = image_metrics(buffers[name], fine, footprint)
                                view['methods'][name]['footprint_reference_coarse'] = image_metrics(buffers[name], coarse, footprint)
                            for opponent in ('raw', 'centroid_window'):
                                view['differences']['beb1_window_vs_'+opponent]['footprint_reference_gain'] = gain_check(
                                    view['methods'][opponent]['footprint_reference_fine'], view['methods']['beb1_window']['footprint_reference_fine'],
                                    view['methods'][opponent]['footprint_reference_coarse'], view['methods']['beb1_window']['footprint_reference_coarse'])
                    if case['frame'] == 15 or case['kind'] == 'exact_root':
                        for opponent in ('raw', 'centroid_window'):
                            shared = current_roi & buffers['beb1_window']['mask'] & buffers[opponent]['mask']
                            save_difference(folder/('depth_beb1_minus_'+opponent+'_scale_0p01.png'),
                                buffers['beb1_window']['depth'], buffers[opponent]['depth'], shared, .01)
                        folders = [folder/name for name in METHODS]; labels = list(METHODS)
                        if view_name == 'diagnostic':
                            folders.append(reference_folder); labels.append('terrain_reference')
                        geometry_sheet(output/(view_name+'_'+case['key']+'_overview.png'), folders, labels,
                                       case['key']+' / '+view_name+' / geometry only')
                    if view_name == 'original' and info['roi_pixels'] == 0:
                        for key in ('beb1_window_vs_raw', 'centroid_window_vs_raw'):
                            change = view['differences'][key]['full_frame']
                            if change['depth_changed_pixels'] or change['normal_changed_pixels'] or change['silhouette_xor_pixels']:
                                raise RuntimeError('Unexpected visible change from an out-of-frustum patch: '+key)
                    del buffers
            natural = [c for c in report['cases'] if c['kind'] == 'natural']
            changed = [c for c in natural if c['beb1_mesh_changed_vs_raw']]
            visible_original = [c for c in natural if any(c['views']['original']['differences']['beb1_window_vs_raw']['full_frame'][k]
                                for k in ('depth_changed_pixels', 'normal_changed_pixels', 'silhouette_xor_pixels'))]
            lower = source['levels']['lower']; upper = source['levels']['upper']
            physical_bounds = [float(np.longdouble(t['numerator'])*np.longdouble(delta)/np.longdouble(t['denominator']))
                               for t in (lower, upper)]
            full_hits = [i for i in range(24)
                         if physical_bounds[0] < float((i+.5)/24)-float(.5/24) < physical_bounds[1]]
            if full_hits != [15]:
                raise RuntimeError('Actual full camera schedule temporal hits differ from the fixed protocol.')
            report['rates'] = {'selected_natural_frames': len(natural), 'full_original_camera_schedule_frames': 24,
                'temporally_hit_natural_frames_in_full_schedule': len(full_hits),
                'temporally_hit_frame_indices_full_schedule': full_hits,
                'temporal_hit_rate_full_schedule': len(full_hits)/24,
                'changed_natural_frames_selected': len(changed), 'changed_natural_frame_indices': [c['frame'] for c in changed],
                'changed_rate_selected': len(changed)/len(natural),
                'visible_original_natural_frames': len(visible_original),
                'visible_original_rate_among_modified': len(visible_original)/len(changed) if changed else None,
                'visible_original_definition': 'Above-threshold depth/normal or mask changes; not perceptual visibility.',
                'root_diagnostic_excluded': True}
            if [c['frame'] for c in changed] != [15]:
                raise RuntimeError('Natural modification pattern differs from the frozen event window.')
            report['status'] = 'COMPLETE_E2_SCHEDULE_AND_GEOMETRY_IMAGES'
            checkpoint('complete')
    except Exception as error:
        report.update(status='STOP', error=type(error).__name__+': '+str(error))
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['cpu_seconds'] = time.process_time()-cpu_started
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['original_cache_unchanged'] = manifest(cache) == before
        report['original_source_unchanged'] = sha(source_path) == SOURCE_SHA
        if not report['original_cache_unchanged'] or not report['original_source_unchanged']:
            report.update(status='STOP', error='Frozen inputs changed during the experiment.')
        report['temporary_cache_removed'] = temporary is not None and not Path(temporary).exists()
        report['output_bytes_before_summary'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        report['script_sha256'] = sha(__file__)
        report['module_sha256'] = {p.name: sha(p) for p in [Path(__file__).with_name(n) for n in
            ('e2_raster.py', 'e2_schedule_coherence.py', 'e2_image_metrics.py', 'e2_render_views.py')]}
        terrain_path = Path(__file__).resolve().parents[1]/'tv0_tv4/run_lightweight_profile.py'
        report['reference_terrain_source'] = {'path': str(terrain_path), 'sha256': sha(terrain_path)}
        report['scientific_limits'] = [
            'Reference grid and gradient refinement are empirical sensitivity checks, not a complete numerical error bound.',
            'Source footprint is separately projected raw support; raw_visible_source_support_pixels includes whole-mesh occlusion.',
            'Pixel silhouette/boundary differences do not certify mesh topology.',
            'Quantized preview differences do not establish human-perceptual visibility.']
        dump(output/'result.json', report)
    print(json.dumps({'status': report['status'], 'error': report.get('error'), 'wall_seconds': report['wall_seconds'],
                       'peak_rss_bytes': report['peak_rss_bytes'], 'output': str(output)}, sort_keys=True), flush=True)
    return 0 if report['status'].startswith('COMPLETE_') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'cache-root', 'source-report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1200)
    raise SystemExit(run(parser.parse_args()))
