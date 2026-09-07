"""One final resource-only amendment; the original protocol/code remain sealed.

This module registers or validates evidence. It never builds, slices, renders,
deletes a cache, or authorizes a different scene/method. Runtime supervisors must
also enforce campaign_deadline_utc; loading historical evidence does not reset it.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE/'artifacts/screen_20260906'
AUTHORIZATION_SHA256 = 'a6090f7c9fb18a4fd3c4b85cf207db0f1e435488be81c68601435dcd7ac002a5'
DEADLINE = '2026-09-06T19:28:39+00:00'
ORDER = ['cave', 'forest_b']
CHANGES = {
    'cache_per_scene_output_bytes': {'old': 4*1024**3, 'new': 8*1024**3},
    'cache_per_scene_wall_seconds': {'old': 2700, 'new': 5400},
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 16*1024**2, 'MISSING_OR_OVERSIZED_AMENDMENT_INPUT:'+str(path))
    def invalid(value):
        raise ValueError('NONFINITE_AMENDMENT_JSON:'+value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid)


def bind(path, bindings, expected=None):
    path = Path(path).resolve()
    actual = file_sha(path)
    require(expected is None or expected == actual, 'AMENDMENT_INPUT_CHANGED:'+str(path))
    require(str(path) not in bindings or bindings[str(path)] == actual, 'CONFLICTING_INPUT_BINDING')
    bindings[str(path)] = actual
    return actual


def effective_budgets(protocol):
    budgets = deepcopy(protocol['budgets'])
    for key, values in CHANGES.items():
        require(budgets[key] == values['old'], 'BASE_RESOURCE_PROFILE_CHANGED:'+key)
        budgets[key] = values['new']
    require(protocol['budgets']['maximum_campaign_wall_seconds'] == 32400, 'CAMPAIGN_DURATION_CHANGED')
    return budgets


def original_deadline(protocol):
    start = datetime.strptime(protocol['registered_at_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    deadline = (start+timedelta(seconds=protocol['budgets']['maximum_campaign_wall_seconds'])).isoformat()
    require(deadline == DEADLINE, 'ORIGINAL_CAMPAIGN_DEADLINE_CHANGED')
    return deadline


def validate_contract(receipt, protocol, protocol_sha, seal_sha, segment_id=None):
    """Pure deep-equality policy check; useful independently of filesystem I/O."""
    require(receipt['schema'] == 'final-visibility-gate-resource-amendment-v1'
            and receipt['status'] == 'APPROVED_FINAL_RESOURCE_CEILING_AMENDMENT', 'INVALID_AMENDMENT_SCHEMA_OR_STATUS')
    require(receipt['amendment_number'] == 1 and receipt['supersedes_amendment'] is None
            and receipt['final_resource_increase'] is True, 'SECOND_RESOURCE_AMENDMENT_FORBIDDEN')
    require(receipt['protocol_sha256'] == protocol_sha and receipt['preregistration_seal_sha256'] == seal_sha,
            'AMENDMENT_BASE_PROTOCOL_OR_SEAL_MISMATCH')
    require(receipt['authorization']['sha256'] == AUTHORIZATION_SHA256, 'AUTHORIZATION_DOCUMENT_CHANGED')
    require(receipt['changes'] == CHANGES, 'ONLY_TWO_EXACT_RESOURCE_CHANGES_ALLOWED')
    budgets = effective_budgets(protocol)
    expected = deepcopy(protocol); expected['budgets'] = budgets
    require(receipt['effective_budgets'] == budgets and receipt['effective_protocol'] == expected,
            'NONRESOURCE_PROTOCOL_CHANGE_FORBIDDEN')
    require(receipt['campaign_deadline_utc'] == original_deadline(protocol), 'CAMPAIGN_CLOCK_RESET_FORBIDDEN')
    registered = datetime.fromisoformat(receipt['registered_at_utc'])
    require(registered.tzinfo is not None and registered < datetime.fromisoformat(DEADLINE), 'AMENDMENT_REGISTERED_AFTER_CAMPAIGN_DEADLINE')
    require(receipt['run_order'] == ORDER and receipt['attempt_forest_b_regardless_of_cave_outcome'] is True,
            'FINAL_SEGMENT_ORDER_OR_STOPPING_POLICY_CHANGED')
    require(set(receipt['retry_sources']) == set(ORDER) and receipt['completed_segment_reused'] == 'mountain',
            'FIXED_SEGMENT_POPULATION_CHANGED')
    require(receipt['on_resource_limit'] == 'RESOURCE_UNRESOLVED'
            and receipt['old_stop_artifacts_preserved'] is True
            and receipt['new_attempt_directories_required'] is True,
            'FINAL_RESOURCE_OR_EVIDENCE_POLICY_CHANGED')
    require(receipt['prohibited_until_both_retry_outcomes'] == ['RGB', 'SSIM', 'post-displacement', 'five-method-quality'],
            'PREMATURE_QUALITY_EXPERIMENT_NOT_AUTHORIZED')
    require(segment_id is None or segment_id in ORDER, 'SEGMENT_NOT_AUTHORIZED_FOR_FINAL_RETRY')
    return receipt


def _collect(protocol_path, seal_path, authorization_path, artifact_root):
    """Rebind the exact existing evidence and saved coarse inputs, without writes."""
    artifact_root = Path(artifact_root).resolve()
    bindings = {}
    protocol, seal = read(protocol_path), read(seal_path)
    psha, ssha = bind(protocol_path, bindings), bind(seal_path, bindings)
    require(seal['status'] == 'SEALED_BEFORE_NEW_SEGMENT_EXPERIMENTS' and seal['protocol_sha256'] == psha
            and seal['input_and_method_sha256'].get(str(Path(protocol_path).resolve())) == psha, 'ORIGINAL_PROTOCOL_SEAL_INVALID')
    for path, expected in seal['input_and_method_sha256'].items():
        bind(path, bindings, expected)
    effective_budgets(protocol); original_deadline(protocol)
    bind(authorization_path, bindings, AUTHORIZATION_SHA256)
    # The old 18 Python files are frozen as well as the original method seal.
    regression_path = artifact_root/'regression_tests02.json'
    regression = read(regression_path); bind(regression_path, bindings)
    for path, expected in regression['current_visibility_screen_python_sha256'].items():
        bind(path, bindings, expected)
    segments = {row['segment_id']: row for row in protocol['segments']}
    require(set(segments) == {'cave', 'forest_b', 'mountain'}, 'PREREGISTERED_SEGMENTS_CHANGED')
    retry, stops = {}, {}
    for segment_id in ORDER:
        report = artifact_root/segment_id/'build01'
        previous = read(report/'summary.json'); bind(report/'summary.json', bindings)
        contract = read(report/'build_protocol.json')
        bind(report/'build_protocol.json', bindings, previous['build_protocol_sha256'])
        require(previous['status'] == 'STOP_INFRASTRUCTURE_BUILD_INCOMPLETE'
                and previous['segment_id'] == segment_id and contract['segment'] == segments[segment_id]
                and contract['protocol_sha256'] == psha, 'ORIGINAL_RESOURCE_STOP_OR_SEGMENT_CHANGED')
        native = previous['stages']['native']
        require(native['wall_limit_seconds'] == 2700 and native['output_limit_bytes'] == 4*1024**3
                and native['infrastructure_stop'] in ('REGISTERED_WALL_LIMIT', 'REGISTERED_DISK_SAFETY_RESERVE'),
                'PRIOR_STOP_IS_NOT_ORIGINAL_RESOURCE_CEILING')
        command = read(report/'native_command.json'); bind(report/'native_command.json', bindings)
        argv = command['argv']
        require(argv.count('--source') == argv.count('--segment') == 1 and argv[argv.index('--segment')+1] == segment_id,
                'ORIGINAL_NATIVE_COMMAND_CHANGED')
        coarse = Path(argv[argv.index('--source')+1]).resolve()
        driver = Path(previous['output']).resolve()/'driver_snapshot.py'
        require(str(driver) in argv, 'ORIGINAL_DRIVER_COMMAND_CHANGED')
        driver_sha = bind(driver, bindings, contract['driver_snapshot_sha256'])
        inventory_path = report/'native_source_before.json'
        inventory = read(inventory_path); inventory_sha = bind(inventory_path, bindings)
        after_path = report/'native_source_after.json'
        require(read(after_path) == inventory, 'OLD_COARSE_BEFORE_AFTER_MISMATCH'); bind(after_path, bindings)
        paths = list(coarse.rglob('*'))
        require(not any(path.is_symlink() for path in paths), 'COARSE_SYMLINK_UNSUPPORTED')
        actual_names = {str(path.relative_to(coarse)) for path in paths if path.is_file()}
        require(actual_names == set(inventory), 'SAVED_COARSE_FILE_SET_CHANGED')
        for name, item in inventory.items():
            path = (coarse/name).resolve()
            require(coarse in path.parents and path.stat().st_size == item['bytes'], 'SAVED_COARSE_SIZE_OR_PATH_CHANGED')
            bind(path, bindings, item['sha256'])
        retry[segment_id] = {'source_coarse': str(coarse), 'previous_build_report': str(report),
            'source_inventory_sha256': inventory_sha, 'source_inventory_path': str(inventory_path),
            'driver_snapshot': str(driver), 'driver_sha256': driver_sha}
        stops[segment_id] = {'path': str(report/'summary.json'), 'sha256': bindings[str(report/'summary.json')]}
    # Reuse, do not mutate/re-run, the fully confirmed Mountain summary.
    from coverage_report import verify_confirmed_summaries
    mountain_path = artifact_root/'mountain/screen03_omp8/combined_summary.json'
    mountain = read(mountain_path); bind(mountain_path, bindings)
    require(mountain['status'] == 'PASS_PREREGISTERED_SEGMENT_SCREEN' and mountain['segment_id'] == 'mountain'
            and mountain['protocol_sha256'] == psha and mountain['preregistration_seal_sha256'] == ssha,
            'MOUNTAIN_COMPLETED_EVIDENCE_CHANGED')
    cref = mountain['omp_confirmation']['artifact']; cp = Path(cref['path'])
    if not cp.is_absolute(): cp = mountain_path.parent/cp
    confirmation = read(cp); bind(cp, bindings, cref['sha256'])
    verify_confirmed_summaries(mountain, confirmation, cp, bindings)
    return protocol, psha, ssha, bindings, retry, stops, {'path': str(mountain_path), 'sha256': bindings[str(mountain_path)]}


def register_amendment(protocol_path, seal_path, authorization_path, output_path, *, now=None):
    output = Path(output_path).resolve()
    require(output.name == 'budget_amendment01.json' and not output.exists()
            and not list(output.parent.glob('budget_amendment*.json')), 'SECOND_REGISTRATION_OR_OVERWRITE_FORBIDDEN')
    require(output.parent.is_dir(), 'EXISTING_ARTIFACT_ROOT_REQUIRED')
    now = datetime.now(timezone.utc) if now is None else now
    require(now.tzinfo is not None and now < datetime.fromisoformat(DEADLINE), 'CAMPAIGN_DEADLINE_EXPIRED')
    protocol, psha, ssha, bindings, retry, stops, mountain = _collect(
        protocol_path, seal_path, authorization_path, output.parent)
    budgets = effective_budgets(protocol); effective = deepcopy(protocol); effective['budgets'] = budgets
    receipt = {'schema': 'final-visibility-gate-resource-amendment-v1', 'status': 'APPROVED_FINAL_RESOURCE_CEILING_AMENDMENT',
        'amendment_number': 1, 'supersedes_amendment': None, 'final_resource_increase': True,
        'registered_at_utc': now.isoformat(), 'protocol_sha256': psha, 'preregistration_seal_sha256': ssha,
        'authorization': {'path': str(Path(authorization_path).resolve()), 'sha256': AUTHORIZATION_SHA256},
        'changes': deepcopy(CHANGES), 'effective_budgets': budgets, 'effective_protocol': effective,
        'campaign_deadline_utc': original_deadline(protocol), 'run_order': list(ORDER),
        'attempt_forest_b_regardless_of_cave_outcome': True, 'retry_sources': retry,
        'completed_segment_reused': 'mountain', 'mountain_completed_evidence': mountain,
        'previous_resource_stops': stops, 'old_stop_artifacts_preserved': True,
        'new_attempt_directories_required': True, 'on_resource_limit': 'RESOURCE_UNRESOLVED',
        'prohibited_until_both_retry_outcomes': ['RGB', 'SSIM', 'post-displacement', 'five-method-quality'],
        'input_sha256': bindings, 'registration_code_sha256': file_sha(__file__),
        'scope': 'Only two resource ceilings change after explicit user approval. No new sample, camera, window, Method, contact policy, or quality gate. Historical STOP is not a scientific negative.'}
    validate_contract(receipt, protocol, psha, ssha)
    for path, expected in bindings.items(): bind(path, {}, expected)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    return receipt


def load_amendment(amendment_path, protocol_path, seal_path, segment_id=None):
    path = Path(amendment_path).resolve()
    require(path.name == 'budget_amendment01.json', 'ONLY_FINAL_AMENDMENT01_IS_SUPPORTED')
    receipt = read(path)
    protocol, psha, ssha, bindings, retry, stops, mountain = _collect(
        protocol_path, seal_path, receipt['authorization']['path'], path.parent)
    validate_contract(receipt, protocol, psha, ssha, segment_id)
    require(receipt['input_sha256'] == bindings and receipt['retry_sources'] == retry
            and receipt['previous_resource_stops'] == stops and receipt['mountain_completed_evidence'] == mountain,
            'AMENDMENT_EVIDENCE_OR_RETRY_MAPPING_CHANGED')
    require(receipt['registration_code_sha256'] == file_sha(__file__), 'REGISTERED_AMENDMENT_CODE_CHANGED')
    result = deepcopy(receipt)
    result['input_sha256'][str(path)] = file_sha(path)
    result['amendment_path'] = str(path)
    result['amendment_sha256'] = file_sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'seal', 'authorization', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    receipt = register_amendment(args.protocol, args.seal, args.authorization, args.output)
    print(json.dumps({'status': receipt['status'], 'output': str(args.output.resolve()),
                      'sha256': file_sha(args.output), 'campaign_deadline_utc': receipt['campaign_deadline_utc']}))


if __name__ == '__main__':
    main()
