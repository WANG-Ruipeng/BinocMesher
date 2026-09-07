"""Final budget-only native retry using the byte-identical registered worker.

No coarse generation, production edit, native recompilation, displacement or
rendering. Only the outer supervisor ceilings differ from the first attempts.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback

from scene_build import (DEFAULT_REPO, load_selection, require, sha, save,
                         inventory, fresh_path, monitor, environment)
from final_gate_amendment import load_amendment


def remaining_seconds(amendment, now=None):
    deadline = datetime.fromisoformat(amendment['campaign_deadline_utc'])
    require(deadline.tzinfo is not None, 'CAMPAIGN_DEADLINE_REQUIRES_TIMEZONE')
    current = now or datetime.now(timezone.utc)
    return max(0.0, (deadline-current).total_seconds())


def verify_hashes(bindings):
    for path, expected in bindings.items():
        require(sha(path) == expected, 'FINAL_GATE_INPUT_CHANGED:'+path)


def supervise(args):
    started_at_utc = datetime.now(timezone.utc).isoformat()
    amendment = load_amendment(args.budget_amendment, args.protocol, args.seal, args.segment)
    protocol, segment, overrides = load_selection(args.protocol, args.segment)
    require(args.segment in ('cave', 'forest_b'), 'ONLY_TWO_FIXED_RETRIES_AUTHORIZED')
    source_binding = amendment['retry_sources'][args.segment]
    source = Path(source_binding['source_coarse']).resolve()
    previous = Path(source_binding['previous_build_report']).resolve()
    expected_inventory = json.loads(Path(source_binding['source_inventory_path']).read_text())
    require(sha(source_binding['source_inventory_path']) == source_binding['source_inventory_sha256'],
            'RETRY_SOURCE_INVENTORY_CHANGED')
    require(inventory(source) == expected_inventory, 'SAVED_COARSE_INPUT_CHANGED_BEFORE_RETRY')
    previous_contract = json.loads((previous/'build_protocol.json').read_text())
    require(previous_contract['segment'] == segment and previous_contract['overrides'] == overrides,
            'FROZEN_SEGMENT_OR_GIN_CHANGED')
    repo = Path(previous_contract['repo']).resolve()
    require(repo == DEFAULT_REPO.resolve(), 'REGISTERED_BUILD_REPOSITORY_CHANGED')
    original_driver = Path(source_binding['driver_snapshot']).resolve()
    require(sha(original_driver) == source_binding['driver_sha256'] == previous_contract['driver_snapshot_sha256'],
            'FROZEN_NATIVE_WORKER_CHANGED')
    require(remaining_seconds(amendment) > 0, 'ORIGINAL_CAMPAIGN_DEADLINE_REACHED')
    output = fresh_path(args.output, [source, previous, repo, args.protocol.parent])
    reports = fresh_path(args.report_dir, [source, previous, repo])
    require(not output.is_relative_to(reports) and not reports.is_relative_to(output), 'DATA_REPORT_OVERLAP')
    require(output.name == args.segment+'_attempt02', 'ONLY_FRESH_SECOND_NATIVE_ATTEMPT_AUTHORIZED')
    camera = Path(previous_contract['camera_path'])
    require(sha(camera) == previous_contract['camera_path_sha256'], 'ORIGINAL_CAMERA_PATH_CHANGED')
    old_native = Path(previous_contract['output'])/'native'
    old_effective = json.loads((old_native/'effective_inputs.json').read_text())
    require(sha(old_effective['core_so']) == old_effective['core_so_sha256'] and
            sha(old_effective['mesher_python']) == old_effective['mesher_python_sha256'],
            'ACTUAL_BUILD_LIBRARY_OR_WRAPPER_CHANGED')
    output.mkdir(parents=True); reports.mkdir(parents=True)
    driver = output/'driver_snapshot.py'; shutil.copyfile(original_driver, driver)
    shutil.copyfile(args.protocol, reports/'preregistration.json')
    limits = amendment['effective_budgets']
    am_ref = {'path': str(args.budget_amendment.resolve()), 'sha256': sha(args.budget_amendment)}
    bindings = dict(amendment['input_sha256'])
    bindings.update({str(args.budget_amendment.resolve()): am_ref['sha256'],
                     str(Path(__file__).resolve()): sha(__file__),
                     str(original_driver): sha(original_driver), str(driver): sha(driver)})
    contract = {**previous_contract, 'output': str(output), 'schema': 'registered-scene-build-attempt-v1',
        'driver_snapshot_sha256': sha(driver), 'supervisor_sha256': sha(__file__),
        'budget_amendment': am_ref, 'effective_budgets': limits,
        'previous_attempt_contract': {'path': str(previous/'build_protocol.json'), 'sha256': sha(previous/'build_protocol.json')},
        'source_reuse': source_binding, 'coarse_regenerated': False,
        'worker_uses_original_protocol_unchanged': True,
        'only_supervisor_native_wall_and_output_ceiling_amended': True,
        'campaign_deadline_utc': amendment['campaign_deadline_utc']}
    save(reports/'build_protocol.json', contract)
    summary = {'status': 'STOP_INFRASTRUCTURE_BUILD_INCOMPLETE'}
    stages = {}; before = None
    try:
        verify_hashes(bindings)
        directory = output/'native'; directory.mkdir()
        command = [sys.executable, '-B', str(driver), '--worker', '--phase', 'native',
                   '--protocol', str(args.protocol.resolve()), '--segment', args.segment,
                   '--repo', str(repo), '--output', str(directory), '--source', str(source)]
        before = inventory(source); require(before == expected_inventory, 'SOURCE_CHANGED_BEFORE_NATIVE_LAUNCH')
        save(reports/'native_source_before.json', before)
        save(reports/'native_command.json', {'argv': command, 'coarse_exec_argv': None})
        wall = min(limits['cache_per_scene_wall_seconds'], remaining_seconds(amendment))
        require(wall > 0, 'ORIGINAL_CAMPAIGN_DEADLINE_REACHED')
        result = monitor(command, repo/'infinigen_binocmesher', environment(repo, True), directory,
                         wall, limits['cache_per_scene_output_bytes'], limits['build_peak_rss_bytes'])
        result.update(registered_native_wall_limit_seconds=limits['cache_per_scene_wall_seconds'],
                      original_campaign_deadline_utc=amendment['campaign_deadline_utc'],
                      wall_shortened_by_original_campaign_deadline=wall < limits['cache_per_scene_wall_seconds'])
        stages['native'] = result; save(reports/'native_process.json', result)
        after = inventory(source); save(reports/'native_source_after.json', after)
        require(after == before, 'ORIGINAL_COARSE_CHANGED_DURING_RETRY')
        require(result['exit_code'] == 0 and result['infrastructure_stop'] is None,
                'FINAL_BOUNDED_NATIVE_DID_NOT_COMPLETE:'+str(result))
        complete = json.loads((directory/'worker_complete.json').read_text())
        require(complete['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE', 'NATIVE_COMPLETION_REQUIRED')
        # These path-independent documents existed before either build's cache
        # results. Reuse must reproduce their complete values, not merely seed.
        for name in ('camera_inputs.json', 'original_render_camera.json', 'effective_inputs.json'):
            require(json.loads((directory/name).read_text()) == json.loads((old_native/name).read_text()),
                    'RETRY_CAMERA_KERNEL_OR_GIN_INPUTS_CHANGED:'+name)
        for name in ('worker_complete.json','camera_inputs.json','original_render_camera.json','effective_inputs.json',
                     'loaded_geometry_libraries.json','loaded_python_sources.json','source_inputs_before.json',
                     'source_inputs_after.json','cache_inventory.json'):
            shutil.copyfile(directory/name, reports/name)
        verify_hashes(bindings)
        summary.update(status='COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE', native_complete=complete,
                       exact_camera_effective_inputs_match_previous_attempt=True)
    except Exception as error:
        summary.update(status='STOP_INFRASTRUCTURE_BUILD_INCOMPLETE',
                       reason=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        if before is not None:
            try:
                final_inventory = inventory(source)
                summary['original_source_inputs_unchanged'] = before == final_inventory
                if before != final_inventory:
                    summary.update(status='STOP_INPUT_INTEGRITY_FAILED', reason='Original source input changed.')
            except Exception as verification_error:
                summary.update(status='STOP_INPUT_INTEGRITY_FAILED',
                               original_source_inputs_unchanged=None,
                               reason='Source verification failed: '+str(verification_error))
        try:
            verify_hashes(bindings)
            summary['final_bound_inputs_unchanged'] = True
        except Exception as verification_error:
            summary.update(status='STOP_INPUT_INTEGRITY_FAILED', final_bound_inputs_unchanged=False,
                           reason='Bound input verification failed: '+str(verification_error))
        summary.update(segment_id=args.segment, stages=stages, output=str(output),
            started_at_utc=started_at_utc, finished_at_utc=datetime.now(timezone.utc).isoformat(),
            build_protocol_sha256=sha(reports/'build_protocol.json'), budget_amendment=am_ref,
            input_sha256=bindings, geometry_stage='PRE_SURFACE_DISPLACEMENT_OPAQUE',
            full_infinigen_or_post_displacement_complete=False, no_rendering=True,
            no_method_admission_attempted=True, failed_build_is_not_zero_events=True,
            data_left_for_readonly_diagnosis=True, cleanup_performed=False,
            coarse_regenerated=False, additional_budget_increase_authorized=False)
        stopped = stages.get('native', {}).get('infrastructure_stop')
        summary['final_gate_resource_outcome'] = ('RESOURCE_UNRESOLVED' if stopped or
            stages.get('native', {}).get('exit_code') == -25 else
            ('BUILD_COMPLETE' if summary['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE' else
             'UNEXPECTED_BUILD_OR_INPUT_FAILURE_NOT_A_ZERO_RESULT'))
        save(reports/'summary.json', summary)
    print(json.dumps({k: summary.get(k) for k in
          ('status','segment_id','final_gate_resource_outcome','reason','output')}, sort_keys=True), flush=True)
    return 0 if summary['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','seal','budget-amendment','output','report-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--segment', choices=('cave','forest_b'), required=True)
    return supervise(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
