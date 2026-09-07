"""Original-camera pixel-center triage of a verified combined Forest sequence.

Candidate denominators describe visible baseline source support, not visible
artifacts or counterfactual changes of rejected methods. No RGB/displacement.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import gc
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from PIL import Image

from forest_component_graph import digest
from forest_native_campaign import PrivateCache, file_sha, mesh_receipt, array_sha
from forest_native_reader import initialize_forest
from forest_sequence_inputs import HERE, CACHE, BUILD, LIBRARY_SHA, read, load_certified_inputs
from run_forest_atomic_sequence import load_component_certificate, reference_frames
from run_forest_formal import Reports, message


def face_counts(buffers):
    result = {}
    for element in range(5):
        ids, counts = np.unique(buffers['face_id'][buffers['element_id'] == element], return_counts=True)
        result[element] = dict(zip(map(int, ids), map(int, counts)))
    return result


def compact_source(mesh, rows):
    v, f, _ = mesh
    ids, remap = np.unique(f[rows], return_inverse=True)
    return v[ids], remap.reshape(-1, 3)


def compare_buffers(before, after, roi=None):
    valid = before['mask'] & after['mask']
    if roi is None:
        roi = before['mask'] | after['mask']
    valid &= roi
    depth = np.abs(after['depth'][valid]-before['depth'][valid])
    a, b = before['normals'][valid], after['normals'][valid]
    changed_normal = np.any(a != b, axis=1)
    angles = np.zeros(len(a), dtype=np.float64)
    angles[changed_normal] = np.degrees(np.arccos(np.clip(np.sum(a[changed_normal]*b[changed_normal], axis=1), -1, 1)))
    return {'roi_pixels': int(roi.sum()), 'common_foreground_pixels': int(valid.sum()),
        'depth_changed_pixels': int(np.count_nonzero(depth)),
        'normal_changed_pixels': int(np.count_nonzero(changed_normal)),
        'depth_over_1e_minus_6_scene_units_pixels': int(np.count_nonzero(depth > 1e-6)),
        'normal_over_1e_minus_3_degrees_pixels': int(np.count_nonzero(angles > 1e-3)),
        'silhouette_changed_pixels': int(np.count_nonzero((before['mask'] ^ after['mask']) & roi)),
        'depth_abs_p95': float(np.percentile(depth, 95)) if len(depth) else None,
        'depth_abs_max': float(depth.max()) if len(depth) else None,
        'normal_angle_degrees_p95': float(np.percentile(angles, 95)) if len(angles) else None,
        'normal_angle_degrees_max': float(angles.max()) if len(angles) else None}


def write_images(reports, key, baseline, combined):
    """Display-only PNGs; numeric metrics and raw buffer hashes stay separate."""
    path = reports.root/'images'; path.mkdir(exist_ok=True)
    mask = baseline['mask'] | combined['mask']
    depths = np.concatenate((baseline['depth'][baseline['mask']], combined['depth'][combined['mask']]))
    low, high = np.percentile(depths, [1, 99]) if len(depths) else (0., 1.)
    high = max(float(high), float(low)+1e-12)
    def depth_image(buffers):
        out = np.zeros(buffers['mask'].shape, np.uint8)
        out[buffers['mask']] = (255*(1-np.clip((buffers['depth'][buffers['mask']]-low)/(high-low), 0, 1))).astype(np.uint8)
        return np.repeat(out[..., None], 3, axis=2)
    def normal_image(buffers):
        out = np.zeros((*buffers['mask'].shape, 3), np.uint8)
        out[buffers['mask']] = (255*np.clip((buffers['normals'][buffers['mask']]+1)/2, 0, 1)).astype(np.uint8)
        return out
    delta = np.zeros((*mask.shape, 3), np.uint8)
    common = baseline['mask'] & combined['mask']
    difference = np.abs(baseline['depth'][common]-combined['depth'][common])
    delta[common, 0] = (255*np.clip(difference/0.02, 0, 1)).astype(np.uint8)
    delta[baseline['mask'] ^ combined['mask']] = [0, 255, 255]
    labels = combined['label_id']
    component = np.zeros((*mask.shape, 3), np.uint8)
    for cid in np.unique(labels[labels >= 0]):
        component[labels == cid] = [(int(cid)*73+41) % 256, (int(cid)*127+89) % 256, (int(cid)*191+137) % 256]
    images = {'depth_pair': np.concatenate((depth_image(baseline), depth_image(combined)), axis=1),
        'normal_pair': np.concatenate((normal_image(baseline), normal_image(combined)), axis=1),
        'depth_delta': delta, 'component_ids': component}
    refs = []
    for kind, pixels in images.items():
        target = path/(key+'-'+kind+'.png')
        if target.exists():
            raise ValueError('Refusing to overwrite a visibility image.')
        Image.fromarray(pixels).save(target)
        reports.bytes += target.stat().st_size
        if reports.bytes > 128*1024**2:
            raise MemoryError('Visibility image/report budget exceeded 128 MiB.')
        refs.append({'kind': kind, 'path': str(target.relative_to(reports.root)), 'sha256': file_sha(target)})
    return {'images': refs, 'depth_display_shared_percentiles': [float(low), high],
            'depth_difference_display_saturation_scene_units': 0.02,
            'component_color_is_hash_not_ordered_metric': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certification', type=Path, required=True)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--omp-comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=16)
    args = parser.parse_args()
    reports = Reports(args.output)
    started, cpu = time.monotonic(), time.process_time()
    private, reader, calls = None, None, 0
    summary = {'status': 'STOP_VISIBILITY_NOT_COMPLETE'}
    try:
        inputs = load_certified_inputs()
        certificate, graph, cert_queries, bindings = load_component_certificate(args.certification)
        sequence, seq_bindings = reference_frames(args.sequence, file_sha(args.certification/'summary.json'))
        bindings.update(seq_bindings)
        comparison = read(args.omp_comparison)
        if (comparison.get('status') != 'PASS_IDENTICAL_ATOMIC_FOREST_SEQUENCES'
                or comparison.get('audited_files_sha256', {}).get(str((args.sequence/'summary.json').resolve()))
                != file_sha(args.sequence/'summary.json')):
            raise ValueError('OMP sequence comparison did not pass.')
        for path, expected in comparison['audited_files_sha256'].items():
            if file_sha(path) != expected:
                raise ValueError('OMP compared sequence evidence changed: '+path)
            bindings[path] = expected
        bindings[str(args.omp_comparison.resolve())] = file_sha(args.omp_comparison)
        from forest_raster import forest_image_camera_document, render_scene
        from forest_component_union import compile_union
        camera = forest_image_camera_document(inputs['camera'])
        camera_ref = reports.save('original_render_camera.json', camera)
        code = {str(HERE/name): file_sha(HERE/name) for name in (
            'run_forest_visibility_triage.py', 'forest_raster.py', 'forest_component_union.py',
            'run_forest_atomic_sequence.py', 'forest_sequence_inputs.py')}
        components = sorted(graph['components'], key=lambda c: c['component_id'])
        cid = {c['component_id']: i for i, c in enumerate(components)}
        event_component = {eid: c for c in components for eid in c['events']}
        private = PrivateCache(CACHE)
        reader = initialize_forest(BUILD, private.path, inputs['camera'], inputs['effective'], expected_so_sha256=LIBRARY_SHA)
        selected_keys = sorted(k for k in cert_queries if k.startswith('frame_'))[:args.max_frames]
        if not selected_keys or not 1 <= args.max_frames <= 16:
            raise ValueError('Expected a bounded nonempty natural-frame triage schedule.')
        frame_refs, event_rows = [], []
        for key in selected_keys:
            before = time.monotonic(); frame = sequence[key]; query = frame['query']; cq = cert_queries[key]
            snapshot = reader.slice_query(float.fromhex(query['physical_time_hex']), mode='physical', ledger=False); calls += 1
            if mesh_receipt(snapshot['meshes']) != frame['baseline']:
                raise ValueError('Visibility baseline differs from verified sequence.')
            expected = frame['union']['components']
            outputs, union = compile_union(snapshot['meshes'], frame['records'], expected_components=expected)
            if outputs is None or union['output'] != frame['union']['output'] or union['event_mapping'] != frame['union']['event_mapping']:
                raise ValueError('Visibility consumer did not receive the certified combined scene.')
            baseline_labels = {e: np.full(len(m[1]), -1, np.int64) for e, m in enumerate(snapshot['meshes'])}
            for event in cq['events']:
                if event['status'] != 'COMPLETE_ACTUAL_REQUESTED_SUPPORT':
                    raise ValueError('All-candidate visibility denominator has unresolved support.')
                number = cid[event_component[event['event_id']]['component_id']]
                array = baseline_labels[event['element']]; old = array[event['source_face_rows']]
                if np.any((old != -1) & (old != number)):
                    raise ValueError('One baseline candidate face belongs to distinct graph components.')
                array[event['source_face_rows']] = number
            output_labels = {e: np.concatenate((baseline_labels[e], np.full(len(m[1])-len(baseline_labels[e]), -1, np.int64)))
                             for e, m in enumerate(outputs)}
            for mapping in union['event_mapping'].values():
                output_labels[mapping['element']][mapping['fan_face_rows_by_sector']] = cid[mapping['component_id']]
            index = query['frame_index_zero_based']
            raw = render_scene(snapshot['meshes'], camera, index, baseline_labels)
            changed = render_scene(outputs, camera, index, output_labels)
            counts, out_counts = face_counts(raw), face_counts(changed)
            local_rows = []
            for event in cq['events']:
                eid, element = event['event_id'], event['element']
                pixels = sum(counts[element].get(i, 0) for i in event['source_face_rows'])
                component = event_component[eid]
                is_admitted = component['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE'
                patch = render_scene([compact_source(snapshot['meshes'][element], event['source_face_rows'])], camera, index)
                unoccluded = int(patch['mask'].sum())
                if pixels:
                    classification = 'VISIBLE_SOURCE_SUPPORT'
                elif unoccluded:
                    classification = 'OCCLUDED_OR_DEPTH_TIE_NOT_SELECTED'
                else:
                    classification = 'NO_UNOCCLUDED_PIXEL_CENTER_COVERAGE'
                mapping = union['event_mapping'].get(eid)
                row = {'event_id': eid, 'component_id': component['component_id'], 'query': key,
                    'jointly_admitted': is_admitted, 'domain_kind': event['kind'],
                    'visible_baseline_source_pixels': pixels, 'unoccluded_source_pixels': unoccluded,
                    'visibility': classification,
                    'visible_replacement_pixels': sum(out_counts[element].get(i, 0) for i in mapping['fan_face_rows_by_sector']) if mapping else 0,
                    'projected_support_raster_stats': patch['stats']}
                local_rows.append(row); event_rows.append(row)
                del patch
            component_metrics = []
            for component in components:
                if component['root'] != cq['root']:
                    continue
                number = cid[component['component_id']]
                roi = (raw['label_id'] == number) | (changed['label_id'] == number)
                component_metrics.append({'component_id': component['component_id'],
                    'decision': component['decision'], 'metrics': compare_buffers(raw, changed, roi)})
            image_refs = write_images(reports, key, raw, changed)
            result = {'query': query, 'sequence_output_sha256': frame['union']['output'],
                'events': local_rows, 'components': component_metrics,
                'global_metrics': compare_buffers(raw, changed),
                'buffer_sha256': {name: {k: array_sha(buffers[k]) for k in
                    ('depth', 'normals', 'mask', 'element_id', 'face_id', 'label_id')}
                    for name, buffers in (('baseline', raw), ('combined', changed))},
                'raster_cost': {'baseline': raw['stats'], 'combined': changed['stats']},
                **image_refs, 'wall_seconds': time.monotonic()-before}
            frame_refs.append(reports.save('frames/'+key+'.json', result))
            message('forest_visibility_frame', query=key, metrics=result['global_metrics'],
                visible_candidates=sum(e['visible_baseline_source_pixels'] > 0 for e in local_rows),
                visible_admitted=sum(e['visible_replacement_pixels'] > 0 for e in local_rows),
                wall_seconds=result['wall_seconds'])
            del raw, changed, outputs, snapshot
            gc.collect()
        reader.close(); reader = None
        for path, expected in {**bindings, **code}.items():
            if file_sha(path) != expected:
                raise ValueError('Visibility input/code changed: '+path)
        all_candidates = {e['event_id'] for e in event_rows}
        visible_candidates = {e['event_id'] for e in event_rows if e['visible_baseline_source_pixels'] > 0}
        visible_admitted = {e['event_id'] for e in event_rows if e['jointly_admitted'] and e['visible_baseline_source_pixels'] > 0}
        visible_replacement = {e['event_id'] for e in event_rows if e['visible_replacement_pixels'] > 0}
        compiled = {e['event_id'] for e in event_rows if e['domain_kind'] == 'FIXED_SOURCE_AND_REPLACEMENT'}
        summary = {'schema': 'forest-original-camera-visibility-v1',
            'status': 'PASS_COMPLETE_NATURAL_SUPPORT_VISIBILITY_TRIAGE' if len(selected_keys) == 16 else 'PASS_PARTIAL_VISIBILITY_SMOKE_ONLY',
            'natural_queries_rendered': len(selected_keys), 'event_frame_queries': len(event_rows),
            'candidate_events_evaluated': len(all_candidates), 'source_contract_candidates': len(compiled),
            'visible_candidate_support_events': len(visible_candidates),
            'jointly_admitted_visible_source_events': len(visible_admitted),
            'visible_joint_replacement_events': len(visible_replacement),
            'visible_support_admission_rate': len(visible_admitted)/len(visible_candidates) if visible_candidates else None,
            'source_contract_only_visible_events': len(visible_candidates & compiled),
            'source_contract_only_visible_admitted_events': len(visible_admitted & compiled),
            'source_contract_only_visible_admission_rate': len(visible_admitted & compiled)/len(visible_candidates & compiled) if visible_candidates & compiled else None,
            'numeric_delta_threshold_scope': 'Predeclared depth 1e-6 scene units and normal 1e-3 degrees separate numerical-scale differences; neither is a perceptual visibility threshold.',
            'candidate_frame_visibility_classes': dict(Counter(e['visibility'] for e in event_rows)),
            'denominator_scope': 'Visible baseline source supports for 130 compiled events plus one broader baseline-blocked candidate Vhalo. Not visible artifacts, nor unavailable counterfactual changes for rejected events.',
            'visible_pixel_impact_coverage': None,
            'visible_pixel_impact_coverage_reason': 'Rejected replacement images are undefined; support coverage cannot stand in for artifact/impact-weighted coverage.',
            'original_render_camera': camera_ref, 'frame_artifacts': frame_refs,
            'component_label_dictionary': [{'label': cid[c['component_id']], 'component_id': c['component_id'],
                'decision': c['decision'], 'events': c['events']} for c in components],
            'component_image_scope': 'All candidate and fallback support components are labeled, not only modified faces. Component ROI deltas are descriptive regions, not isolated causal interventions.',
            'difference_scope': 'Visible replacement pixels and floating-buffer differences do not establish perceptual quality improvement.',
            'certification_sha256': file_sha(args.certification/'summary.json'),
            'sequence_sha256': file_sha(args.sequence/'summary.json'),
            'input_bindings_sha256': bindings, 'executed_sources_sha256': code,
            'scope': 'Original-camera pre-displacement pixel-center visibility and geometric image differences only, not visual quality improvement or post-displacement safety.'}
    except Exception as error:
        summary.update(status='STOP_VISIBILITY_NOT_COMPLETE', reason=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        if reader is not None:
            reader.close()
        if private is not None:
            try:
                summary['final_input_verification'] = private.verify(calls)
            except Exception as error:
                summary.update(status='INPUT_VERIFICATION_FAILED', verification_error=str(error))
            private.remove(); summary['private_cache_removed'] = True
        summary['cost'] = {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        reports.save('summary.json', summary)
    message('forest_visibility_summary', **{k: summary.get(k) for k in
        ('status', 'visible_candidate_support_events', 'visible_joint_replacement_events', 'reason')})
    return 0 if summary['status'].startswith('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
