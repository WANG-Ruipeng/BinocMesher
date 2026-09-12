"""Bounded three-block NCU diagnosis after completed and independently audited timing.

Default mode only prepares a reviewable command plan. Actual collection requires
--execute-profile. One fresh process and one selected launch per block variant;
no target retry, extra calibration, clock locking, cache flushing or system change.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path
from ncu_csv import single_kernel_metrics, finite_metric, NcuCsvError

from workspace_paths import ROOT
from workspace_paths import tool, PYTHON
BLOCKS = (256, 128, 64)
FILTER = r'regex:local_group_kernel<\(int\)32>'
SECTIONS = ('LaunchStats', 'Occupancy', 'MemoryWorkloadAnalysis', 'SchedulerStats', 'WarpStateStats')
METRICS = {
    'achieved_occupancy': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'theoretical_occupancy': 'sm__maximum_warps_per_active_cycle_pct',
    'active_warps_per_SM': 'sm__warps_active.avg.per_cycle_active',
    'theoretical_active_warps': 'sm__maximum_warps_avg_per_active_cycle',
    'registers_per_thread': 'launch__registers_per_thread',
    'grid_blocks': 'launch__grid_size', 'block_threads': 'launch__block_size',
    'SM_count': 'launch__sm_count',
    'register_block_limit': 'launch__occupancy_limit_registers',
    'replay_passes': 'profiler__replayer_passes',
    'eligible_warps_per_scheduler': 'smsp__warps_eligible.avg.per_cycle_active',
    'active_warps_per_scheduler': 'smsp__warps_active.avg.per_cycle_active',
    'issue_active_percent': 'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'threads_executed_per_warp_instruction': 'smsp__thread_inst_executed_per_inst_executed.ratio',
    'pred_on_threads_per_warp_instruction': 'smsp__thread_inst_executed_pred_on_per_inst_executed.ratio',
    'spill_instructions': 'sass__inst_executed_register_spilling',
    'local_spill_requests': 'derived__local_spilling_requests',
    'shared_spill_requests': 'derived__shared_spilling_requests',
    'L1_hit_percent': 'l1tex__t_sector_hit_rate.pct',
    'L2_hit_percent': 'lts__t_sector_hit_rate.pct',
    'profiler_kernel_duration_ns': 'gpu__time_duration.sum',
    'stall_wait_ratio': 'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio',
    'stall_short_scoreboard_ratio': 'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
    'stall_long_scoreboard_ratio': 'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'user_static_shared_bytes': 'launch__shared_mem_per_block_static',
    'user_dynamic_shared_bytes': 'launch__shared_mem_per_block_dynamic',
    'driver_shared_bytes': 'launch__shared_mem_per_block_driver',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def within_results(path):
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT / 'results') and resolved != ROOT / 'results',
            'Use a child of current experiment results')
    return resolved


def gates(args):
    require(args.execute_profile and sys.platform.startswith('linux'), 'Actual profiling requires explicit WSL execution')
    require(read(ROOT / 'configs/PERFORMANCE_AUTHORIZATION.json').get('allowed') is True,
            'Performance authorization absent')
    require(read(args.performance / 'STATUS.json')['status'] == 'COMPLETED' and
            read(args.performance / 'SUMMARY.json')['status'] == 'COMPLETED',
            'Formal timing must finish successfully before separate profiles')
    audit = read(args.audit)
    require(audit.get('status') == 'PASS_INDEPENDENT_BLOCK_AUDIT' and audit.get('issues') == [],
            'Independent block performance audit must pass first')
    require(audit.get('suite') and Path(audit['suite']).resolve() == args.performance.resolve(),
            'Accepted audit belongs to a different performance suite')
    evidence = audit.get('evidence', {})
    for label, path in (
            ('summary_sha256', args.performance / 'SUMMARY.json'),
            ('selection_sha256', args.performance / 'SELECTION.json'),
            ('matrix_sha256', ROOT / 'configs/BLOCK_MATRIX.json')):
        require(evidence.get(label) == sha(path), 'Accepted audit does not match current ' + label)


def make_plan(args):
    ncu = Path(tool('ncu')).resolve()
    section_folder = Path(os.environ['BM_NCU_SECTIONS']).expanduser().resolve()
    paths = [ROOT / 'src/workspace_paths.py', ROOT / 'configs/BLOCK_MATRIX.json', ROOT / 'configs/PERFORMANCE_AUTHORIZATION.json',
             ROOT / 'src/profile_block_single.py', ROOT / 'src/run_block_profiles.py',
             ROOT / 'src/ncu_csv.py', ROOT / 'src/check_gpu_solver.py', ROOT / 'src/gpu_solver.cu',
             ROOT / 'src/run_stage.py', ROOT / 'build/libfield_bridge.so',
             ROOT / 'inputs/accepted_scene/field_parameters.npz', args.audit,
             args.performance / 'STATUS.json', args.performance / 'SUMMARY.json',
             args.performance / 'PLAN.json', args.performance / 'SELECTION.json']
    paths += [ROOT / f'build/libgpu_solver_b{block}.so' for block in BLOCKS]
    commands = []
    for block in BLOCKS:
        mode = 'b' + str(block)
        command = [str(ncu), '--config-file', '0', '--rename-kernels', '0',
                   '--target-processes', 'application-only', '--replay-mode', 'kernel',
                   '--graph-profiling', 'node', '--clock-control', 'none',
                   '--pipeline-boost-state', 'dynamic', '--cache-control', 'none',
                   '--kernel-name-base', 'demangled', '--kernel-name', FILTER,
                   '--launch-count', '1', '--kill', '0', '--section-folder', str(section_folder)]
        for section in SECTIONS:
            command += ['--section', section]
        command += ['--apply-rules', 'no', '--csv', '--page', 'raw',
                    '--log-file', str(args.out / ('ncu_' + mode + '.csv')),
                    '--export', str(args.out / ('ncu_' + mode)),
                    PYTHON, '-B', str(ROOT / 'src/profile_block_single.py'),
                    '--execute-profile', '--plan', str(args.out / 'PROFILE_PLAN.json'),
                    '--block', str(block), '--out', str(args.out / mode)]
        commands.append({'block': block, 'mode': mode, 'argv': command,
                         'timeout_seconds': 180, 'logical_complete_solves': 1,
                         'profiled_launches': 1})
    require(re.search(FILTER.removeprefix('regex:'), 'void local_group_kernel<(int)32>(SolverDevice, int, int)'),
            'Frozen corrected filter does not match the observed demangled kernel')
    parameters = ROOT / 'inputs/accepted_scene/field_parameters.npz'
    return {
        'status': 'READY_FOR_EXPLICIT_EXECUTION' if args.execute_profile else 'PREPARED_NOT_RUN',
        'case': 'g00000_b00000_SdfTrees', 'K': 3, 'N': 24, 'M': 136, 'Q': 568,
        'role': 'fixed calibration first batch; diagnostic only, not held-out selection',
        'blocks': list(BLOCKS), 'group_threads': 32, 'mode_id': 4,
        'warmups': 0, 'timing_api_calls': 0, 'automatic_retries': 0,
        'performance_suite': str(args.performance), 'audit_path': str(args.audit),
        'audit_sha256': sha(args.audit) if args.audit.is_file() else None,
        'parameters_sha256': sha(parameters),
        'fingerprints': [{'path': str(path.relative_to(ROOT)),
                          'sha256': sha(path) if path.is_file() else None} for path in paths],
        'ncu_path': str(ncu), 'ncu_sha256': sha(ncu) if ncu.is_file() else None,
        'section_folder': str(section_folder), 'sections': list(SECTIONS),
        'kernel_filter': FILTER, 'commands': commands,
        'clock_control': 'none', 'cache_control': 'none', 'pipeline_boost_state': 'dynamic',
        'profiler_replay': 'kernel; invasive tool replays of the selected launch',
        'isolation_environment': {'NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT': '1',
                                  'XDG_CONFIG_HOME': str(ROOT / 'cache/ncu_config')},
        'no_system_changes': True,
        'interpretation': 'Same complete fused node-solve kernel work, including node initialization, for each block; excludes field/solver preparation, uploads, readback and host caller. NCU duration is diagnostic; formal paired solve/common results determine speed.',
        'failure_policy': 'Preserve failed target/profile; no automatic retries or driver/security changes. Optional absent metrics stay unavailable.',
    }


def parse_report(out, block):
    mode = 'b' + str(block)
    raw_path = out / ('ncu_' + mode + '.csv')
    raw = raw_path.read_text(encoding='utf-8', errors='replace')
    require('No kernels were profiled' not in raw, 'NCU did not match any kernel')
    parsed = single_kernel_metrics(raw)
    require(re.search(FILTER.removeprefix('regex:'), parsed['identity']['Kernel Name']),
            'Profiled kernel is not the fixed GROUP=32 fused kernel')
    target = read(out / mode / 'STATUS.json')
    require(target['status'] == 'PASS' and target.get('output_bitwise') == 'PASS' and
            not target['mismatches'] and target['logical_solver_calls'] == 1 and
            target['timing_api_calls'] == 0 and target['warmups'] == 0 and
            target['info_after'][31:36] == [0, 0, 1, 1, 0],
            'Profile target did not establish one bitwise-correct normal solve')
    require(sha(out / mode / 'actual.npz') == target['actual_sha256'], 'Saved target output changed')
    report_path = out / ('ncu_' + mode + '.ncu-rep')
    require(report_path.is_file() and report_path.stat().st_size > 0, 'Missing NCU report artifact')
    metrics = parsed['metrics']
    selected, anomalies = {}, []
    essential = {'achieved_occupancy', 'theoretical_occupancy', 'registers_per_thread',
                 'grid_blocks', 'block_threads', 'SM_count'}
    for label, name in METRICS.items():
        try:
            value = finite_metric(metrics, name, '%' if 'occupancy' in label else None)
            selected[label] = {'status': 'AVAILABLE', 'value': value,
                               'unit': metrics[name]['unit'], 'metric': name}
        except NcuCsvError as error:
            if label in essential:
                raise
            selected[label] = {'status': 'UNAVAILABLE', 'metric': name, 'reason': str(error)}
    for label in ('achieved_occupancy', 'theoretical_occupancy'):
        require(0 <= selected[label]['value'] <= 100, 'Occupancy outside physical bounds')
    require(selected['block_threads']['value'] == block and
            selected['grid_blocks']['value'] == (24 * 32 + block - 1) // block,
            'Measured grid/block does not match the compiled variant')
    for label in ('L1_hit_percent', 'L2_hit_percent'):
        row = selected[label]
        if row['status'] == 'AVAILABLE' and not 0 <= row['value'] <= 100:
            row['status'] = 'OUT_OF_RANGE_NOT_USED'
            anomalies.append({'label': label, 'metric': row['metric'], 'value': row['value']})
    write(out / ('ncu_' + mode + '_parsed.json'), parsed)
    return {'status': 'PASS_ONE_PROFILE_AND_BITWISE_OUTPUT', 'block_threads': block,
            'group_threads': 32, 'identity': parsed['identity'], 'selected_metrics': selected,
            'metric_count': len(metrics), 'bounded_metric_anomalies': anomalies,
            'raw_sha256': sha(raw_path), 'report_sha256': sha(report_path),
            'report_bytes': report_path.stat().st_size, 'target_output_sha256': target['actual_sha256'],
            'logical_solver_calls': 1,
            'kernel_scope': 'complete fused node-solve kernel including node initialization; excludes field/solver preparation, uploads, readback and host caller',
            'metric_semantics': 'actual occupancy uses active-cycle normalization, not whole-device utilization; max normalized metrics need not be bounded by 100'}


def execute(args, plan):
    from run_stage import run
    gates(args)
    require(all(item['sha256'] for item in plan['fingerprints']), 'Missing frozen profile artifacts')
    require(plan['ncu_sha256'] and Path(plan['section_folder']).is_dir(), 'Existing NCU tool or sections unavailable')
    os.environ.update(plan['isolation_environment'])
    (ROOT / 'cache/ncu_config').mkdir(parents=True, exist_ok=True)
    state = {'status': 'RUNNING', 'completed_blocks': [], 'automatic_retries': 0,
             'formal_performance_affected': False, 'plan_sha256': sha(args.out / 'PROFILE_PLAN.json')}
    write(args.out / 'STATUS.json', state)
    results = {}
    try:
        for item in plan['commands']:
            block, mode = item['block'], item['mode']
            state['current_block'] = block
            write(args.out / 'STATUS.json', state)
            for row in plan['fingerprints']:
                require(sha(ROOT / row['path']) == row['sha256'], 'Frozen artifact changed: ' + row['path'])
            require(sha(Path(plan['ncu_path'])) == plan['ncu_sha256'], 'NCU executable changed')
            phase = 'profile_' + args.out.name + '_' + mode
            code = run(phase, item['argv'], item['timeout_seconds'])
            require(code == 0, f'NCU/target failed for {mode}, exit {code}; preserve this phase')
            results[mode] = parse_report(args.out, block)
            state['completed_blocks'].append(block)
            write(args.out / 'STATUS.json', state)
        for row in plan['fingerprints']:
            require(sha(ROOT / row['path']) == row['sha256'], 'Frozen artifact changed after profiles: ' + row['path'])
        summary = {'status': 'COMPLETE_REPRESENTATIVE_BLOCK_PROFILES', 'case': plan['case'],
                   'K': 3, 'N': 24, 'M': 136, 'Q': 568, 'group_threads': 32,
                   'blocks': results, 'new_target_solves': 3, 'new_selected_launches': 3,
                   'automatic_retries': 0, 'formal_performance_affected': False,
                   'interpretation': [
                       'Single fixed real calibration batch; descriptive mechanism evidence only.',
                       'All three targets compare the same six output arrays bitwise with the native oracle.',
                       'NCU kernel replay is invasive; kernel durations are not formal complete-solve speedup data.',
                       'Cache and clock control are none; no warmup and no clock lock.',
                       'Active-cycle occupancy is not whole-device utilization; the natural grid is small.',
                       'Missing optional or out-of-range bounded metrics are marked unavailable and excluded.',
                   ]}
        write(args.out / 'PROFILE_COMPARISON.json', summary)
        with (args.out / 'PROFILE_COMPARISON.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['label', 'metric', 'unit', 'b256', 'b128', 'b64'])
            writer.writeheader()
            for label, name in METRICS.items():
                row = {'label': label, 'metric': name,
                       'unit': results['b256']['selected_metrics'][label].get('unit', '')}
                for mode in ('b256', 'b128', 'b64'):
                    metric = results[mode]['selected_metrics'][label]
                    row[mode] = metric['value'] if metric['status'] == 'AVAILABLE' else metric['status']
                writer.writerow(row)
        state.update(status='COMPLETED', metrics='AVAILABLE_WITH_EXPLICIT_OPTIONAL_LIMITATIONS')
    except BaseException as error:
        state.update(status='PROFILE_INCOMPLETE_PRESERVED', error=repr(error),
                     traceback=traceback.format_exc(), metrics='PARTIAL_OR_UNAVAILABLE',
                     completed_results=results)
        write(args.out / 'PROFILE_ERROR.json', state)
        raise
    finally:
        write(args.out / 'STATUS.json', state)
    print(json.dumps({'status': state['status'], 'completed_blocks': state['completed_blocks']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--execute-profile', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/block_profile_01')
    parser.add_argument('--performance', type=Path, default=ROOT / 'results/block_performance_01')
    parser.add_argument('--audit', type=Path, default=ROOT / 'results/BLOCK_AUDIT.json')
    args = parser.parse_args()
    args.out, args.performance, args.audit = (within_results(path) for path in
                                             (args.out, args.performance, args.audit))
    if args.execute_profile:
        gates(args)
    plan = make_plan(args)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'PROFILE_PLAN.json', plan)
    if not args.execute_profile:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    execute(args, plan)


if __name__ == '__main__':
    main()