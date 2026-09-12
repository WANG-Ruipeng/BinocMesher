"""Finite real-field GROUP=32 block-size experiment. No GPU work without opt-in.

Only L1_32 (mode 4) is executed. The frozen variants are 256, 128 and 64
threads per block. First native field batches calibrate repeat counts only;
all remaining native batches are held out from calibration. No candidate is
dropped and no old performance stop gate is inherited.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from performance_suite import (
    ResidentCase, append_json, bind_timing, case_key, digest, field_key,
    load_execution_inputs, orders, read_json, remove_private_checkpoint,
    require, summarize, telemetry, write_json,
)

from workspace_paths import ROOT
MATRIX = ROOT / 'configs/BLOCK_MATRIX.json'
AUTH = ROOT / 'configs/PERFORMANCE_AUTHORIZATION.json'
BLOCKS = (256, 128, 64)
MODES = tuple('b' + str(block) for block in BLOCKS)


def matrix_check(m):
    require(m['blocks'] == list(BLOCKS), 'Block variants changed')
    require(m['node_group'] == 32 and m['mode_id'] == 4,
            'Frozen GROUP=32 mode-4 declaration differs from execution')
    require(m['processes'] == 3 and m['warmups'] == 10 and m['pairs'] == 20,
            'Fixed process/warmup/pair protocol changed')
    require(m['pair_orders'] == {'AB': 10, 'BA': 10} and m['K'] == [3, 6],
            'Pair balance or K changed')
    require(m['repeat_calibration'] == {'target_event_ms': 10, 'max_repeats': 4096},
            'Repeat calibration bounds changed')
    rows = m['cases']
    require(len(rows) == 18 and len({case_key(r) for r in rows}) == 18,
            'Expected exactly 18 unique original real field/K cases')
    require(sum(r['role'] == 'calibration' for r in rows) == 4 and
            sum(r['role'] == 'held_out' for r in rows) == 14,
            'Expected 4 calibration and 14 held-out cases')
    for k in m['K']:
        for field in ('LandTiles', 'SdfTrees'):
            subset = [r for r in rows if r['K'] == k and r['field'] == field]
            calibration = [r for r in subset if r['role'] == 'calibration']
            require(len(calibration) == 1 and calibration[0]['batch'] == 0,
                    'Only the first native field batch calibrates')
            require(all(r['batch'] > 0 for r in subset if r['role'] == 'held_out'),
                    'Calibration batch entered the held-out table')
        require(sum(r['N'] for r in rows if r['K'] == k) == 192 and
                sum(r['M'] for r in rows if r['K'] == k) == 1053,
                'Original natural group totals changed')
    for r in rows:
        require(r['field'] in ('LandTiles', 'SdfTrees') and r['K'] in (3, 6),
                'Only captured real fields and fixed K are permitted')
        require(r['N'] > 0 and r['N'] == len(r['m_counts']) and
                r['M'] == sum(r['m_counts']) and
                r['Q'] == r['N'] + (r['K'] + 1) * r['M'],
                'Frozen input dimensions differ')


def library(mode):
    require(mode in MODES, 'Invalid block variant')
    return ROOT / ('build/libgpu_solver_' + mode + '.so')


def gate(explicit):
    require(explicit, 'Missing explicit performance switch')
    require(read_json(AUTH).get('allowed') is True, 'Performance authorization absent')
    from common_runtime import performance_allowed
    performance_allowed(True)


def fingerprint_plan():
    m = read_json(MATRIX)
    matrix_check(m)
    paths = [MATRIX, AUTH, ROOT / 'configs/scene.json',
             ROOT / 'inputs/accepted_scene/inputs.npz',
             ROOT / 'inputs/accepted_scene/field_parameters.npz',
             ROOT / 'build/libfield_bridge.so',
             ROOT / 'upstream/main/binocmesher/lib/core.so']
    paths += [library(mode) for mode in MODES]
    paths += [ROOT / 'src' / name for name in (
        'workspace_paths.py', 'block_benchmark.py', 'performance_suite.py', 'common_runner.py',
        'common_runtime.py', 'check_gpu_solver.py', 'gpu_solver.cu')]
    paths += sorted((ROOT / 'upstream/main').rglob('*.py'))
    require(len(paths) == len(set(paths)), 'Duplicate fingerprint path')
    seed = ROOT / 'inputs/checkpoint_seed'
    return {
        'status': 'PREPARED_NOT_RUN', 'GPU': 'NOT_RUN',
        'matrix_sha256': digest(MATRIX),
        'fingerprints': [{'path': str(p.relative_to(ROOT)),
                          'sha256': digest(p) if p.is_file() else None} for p in paths],
        'checkpoint_files': [{'path': str(p.relative_to(seed)), 'sha256': digest(p)}
                             for p in sorted(seed.rglob('*')) if p.is_file()],
        'calibration_cases': [case_key(r) for r in m['cases'] if r['role'] == 'calibration'],
        'resident_formal_cases': [case_key(r) for r in m['cases'] if r['role'] == 'held_out'],
        'group_threads': 32, 'mode_id': 4, 'blocks': list(BLOCKS),
        'worker_order': [0, 1, 2], 'calibration_worker': 0,
        'comparisons': ['b256_vs_b128', 'b256_vs_b64'],
        'resident_formal_arms_per_worker': 14 * 2 * 20 * 2,
        'common_formal_arms_per_worker': 2 * 2 * 20 * 2,
        'all_workers_formal_arms': 3840,
        'common_warmup_arms_per_worker': 2 * 3 * 10,
        'common_repeats': 1,
        'common_boundary': 'private checkpoint -> complete original-format host output; before write_final_hypermesh',
        'checkpoint_provisioning': 'identical private copy excluded; actual reads included',
        'resident_boundary': 'GPU event bracket of repeated complete mode-4 solves; readback/check outside',
        'output_checks': 'first, repeat, last warmup, every calibration and formal arm; bitwise',
        'automatic_retries': 0,
        'variant_selection': 'NONE: all fixed blocks retained, calibration chooses repeats only',
        'statistics': 'paired A/B ratios per process; median/p90; aggregate median of 3 process medians',
        'generalization': 'one captured real scene; 14 held-out field/K cases are 7 field-batches at 2 K',
    }


def verify_plan(plan):
    require(plan['matrix_sha256'] == digest(MATRIX), 'Frozen block matrix changed')
    for item in plan['fingerprints']:
        require(item['sha256'] and digest(ROOT / item['path']) == item['sha256'],
                'Prepared artifact changed: ' + item['path'])
    seed = ROOT / 'inputs/checkpoint_seed'
    actual = {str(p.relative_to(seed)): digest(p) for p in seed.rglob('*') if p.is_file()}
    require(actual == {r['path']: r['sha256'] for r in plan['checkpoint_files']},
            'Full checkpoint seed inventory changed')


class BlockResident(ResidentCase):
    """Reuse accepted output/measurement ABI, without preparing any CUDA Graph."""
    def __init__(self, api, case, params, destination, mode):
        import ctypes as C
        from check_gpu_solver import pointer, PF, PI, PD
        self.C, self.pointer, self.PF, self.PI, self.PD = C, pointer, PF, PI, PD
        self.api, self.case, self.destination = api, case, destination
        self.mode = mode
        self.preparation, self.checks = {}, []
        start = time.perf_counter_ns()
        self.field, view = api.create_field(case.kind, params)
        self.preparation['field_initialization_upload_ms'] = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        self.handle = api.create_solver(case, view)
        self.preparation['solver_allocation_owner_map_upload_ms'] = (time.perf_counter_ns() - start) / 1e6
        before = self.info()
        require(before[1] == 0 and before[38] == 1 and before[31] == 0 and before[32] == 0,
                'Fresh ordinary, untraced, untimed solver required')
        require(before[2] == len(case.centers) and before[3] == len(case.endpoints) and
                before[7] == len(case.centers) + (case.K + 1) * len(case.endpoints),
                'Solver N/M/Q differs')
        elapsed = C.c_double()
        api.check(api.solver.bm_solver_timing_prepare(self.handle, 1, C.byref(elapsed)),
                  'timing_prepare')
        self.preparation.update(mode=mode, block_threads=int(mode[1:]), group_threads=32,
            field_instances='independent immutable instance per variant',
            graph_capture_instantiate='NOT_USED', event_creation_ms=elapsed.value,
            event_cost_is_measurement_instrumentation=True, info_before_events=before,
            info_after_events=self.info(), first_solve_host_ms={})
        write_json(destination / 'preparation.json', self.preparation)

    def check_now(self, purpose, save=False):
        info = super().check_output(self.mode, purpose, save)
        append_json(self.destination / 'checks.jsonl', self.checks[-1])
        return info

    def warmup_block(self, count):
        for j in range(count):
            start = time.perf_counter_ns() if j == 0 else None
            self.api.check(self.api.solver.bm_solver_run(self.handle, 4, self.case.K, 0),
                           'mode4_warmup')
            if j == 0:
                self.preparation['first_solve_host_ms'][self.mode] = (time.perf_counter_ns() - start) / 1e6
            if j in (0, 1):
                self.check_now('first' if j == 0 else 'repeat', True)
        self.check_now('warmup_final', True)
        write_json(self.destination / 'preparation.json', self.preparation)

    def measure_block(self, repeats):
        result = super().measure('L1_32', repeats)
        return {'block_threads': int(self.mode[1:]), 'group_threads': 32, **result}


def create_sessions(apis, case, params, destination):
    sessions = {}
    for mode in MODES:
        directory = destination / mode
        directory.mkdir(parents=True, exist_ok=False)
        sessions[mode] = BlockResident(apis[mode], case, params, directory, mode)
    return sessions


def calibrate(m, cases, params, apis, worker_dir, suite):
    selection = {'worker': 0, 'matrix_sha256': digest(MATRIX), 'fields': {},
                 'variant_selection': 'NONE', 'frozen_blocks': list(BLOCKS)}
    for index, r in enumerate(row for row in m['cases'] if row['role'] == 'calibration'):
        case = cases[case_key(r)]
        directory = worker_dir / 'calibration' / case_key(r).replace(':', '_')
        sessions = create_sessions(apis, case, params[case.kind], directory)
        for mode in MODES[index % 3:] + MODES[:index % 3]:
            sessions[mode].warmup_block(m['warmups'])
        repeats, level = 1, 0
        while True:
            durations = {}
            mode_order = MODES[level % 3:] + MODES[:level % 3]
            for mode in mode_order:
                observed = sessions[mode].measure_block(repeats)
                sessions[mode].check_now('calibration_r' + str(repeats), True)
                durations[mode] = observed['event_ms']
                append_json(directory / 'raw.jsonl', {'phase': 'repeat_calibration',
                    'case': case_key(r), 'mode': mode, 'mode_order': list(mode_order),
                    'full_output_check': 'PASS_OUTSIDE_TIMING', **observed})
            reached = min(durations.values()) >= m['repeat_calibration']['target_event_ms']
            if reached or repeats >= m['repeat_calibration']['max_repeats']:
                break
            repeats *= 2
            level += 1
        selection['fields'][field_key(r['field'], r['K'])] = {
            'calibration_case': case_key(r), 'repeats': repeats,
            'target_ms': 10, 'repeat_cap': 4096, 'resolution_target_reached': reached,
            'final_calibration_event_ms': durations,
            'variant_selection': 'NONE; repeats shared by all three fixed variants'}
        for session in sessions.values():
            session.finish()
        print(json.dumps({'phase': 'calibration', 'case': case_key(r),
                          'repeats': repeats, 'target_reached': reached}), flush=True)
    require(not (suite / 'SELECTION.json').exists(), 'Never overwrite frozen calibration')
    write_json(suite / 'SELECTION.json', selection)
    return selection


def resident_table(m, cases, params, apis, selection, worker, worker_dir, raw):
    for index, r in enumerate(row for row in m['cases'] if row['role'] == 'held_out'):
        case = cases[case_key(r)]
        directory = worker_dir / 'resident' / case_key(r).replace(':', '_')
        sessions = create_sessions(apis, case, params[case.kind], directory)
        rotation = (index + worker) % 3
        for mode in MODES[rotation:] + MODES[:rotation]:
            sessions[mode].warmup_block(m['warmups'])
        choice = selection['fields'][field_key(r['field'], r['K'])]
        for candidate in ('b128', 'b64'):
            comparison = 'b256_vs_' + candidate
            for pair, order in enumerate(orders(m, worker, case_key(r), comparison)):
                for arm in order:
                    mode = 'b256' if arm == 'A' else candidate
                    observed = sessions[mode].measure_block(choice['repeats'])
                    purpose = comparison + '_pair' + str(pair).zfill(2) + '_' + arm
                    sessions[mode].check_now(purpose, pair in (0, 19))
                    row = {'table': 'resident', 'case': r['id'], 'K': r['K'],
                           'field': r['field'], 'role': 'held_out', 'worker': worker,
                           'comparison': comparison, 'primary': True,
                           'mode_A': 'b256', 'mode_B': candidate, 'mode': mode,
                           'pair': pair, 'order': order, 'arm': arm,
                           'full_output_check': 'PASS_OUTSIDE_TIMING', **observed}
                    raw.append(row)
                    append_json(worker_dir / 'trials.jsonl', row)
        for session in sessions.values():
            session.finish()
        print(json.dumps({'worker': worker, 'phase': 'resident',
                          'case': case_key(r), 'status': 'COMPLETED'}), flush=True)


def common_table(m, worker, worker_dir, raw):
    import shutil
    import numpy as np
    from check_gpu_solver import same
    from common_runner import run_common
    source = ROOT / 'inputs/checkpoint_seed'
    private = worker_dir / 'private_checkpoints'
    private.mkdir()
    records = worker_dir / 'common'
    records.mkdir()
    serial, expected = 0, {}
    for k in m['K']:
        rows = [r for r in m['cases'] if r['K'] == k]
        capture, last = Path(rows[0]['capture_dir']), max(r['batch'] for r in rows)
        with np.load(capture / f'g00000_b{last:05d}_native_outputs.npz', allow_pickle=False) as f:
            expected[k] = {key: f[key].copy() for key in f.files}

    def arm(mode, k, phase, label, timed, save=False):
        nonlocal serial
        serial += 1
        identity = f'a{serial:05d}_K{k}_{mode}'
        cp = private / identity
        shutil.copytree(source, cp)
        actual, record = run_common('L1_32', k, cp, timing=timed, validate=False,
                                    solver_path=library(mode))
        bad = sorted(set(actual) ^ set(expected[k]))
        bad += [key for key in set(actual) & set(expected[k]) if not same(actual[key], expected[k][key])]
        record.update(block_variant=mode, group_threads=32, block_threads=int(mode[1:]),
                      phase=phase, label=label, complete_output_check='FAIL' if bad else 'PASS_OUTSIDE_TIMING')
        write_json(records / (identity + '.json'), record)
        if bad or save:
            np.savez(records / (identity + '_actual.npz'), **actual)
        if bad:
            np.savez(records / (identity + '_first_failure_oracle.npz'), **expected[k])
            write_json(records / 'first_failure.json',
                       {'id': identity, 'mismatches': bad, 'checkpoint': str(cp)})
            raise RuntimeError('Complete common output mismatch: ' + str(bad))
        remove_private_checkpoint(cp, private)
        if not timed:
            return None
        roots = [r for r in record['stages'] if r['stage'] == 'checkpoint_to_output' and r['parent'] is None]
        require(len(roots) == 1 and roots[0]['wall_ns'] > 0, 'Missing common parent timing')
        return {'per_solve_ms': roots[0]['wall_ns'] / 1e6, 'repeats': 1,
                'stage_file': str((records / (identity + '.json')).relative_to(worker_dir)),
                'full_output_check': 'PASS_OUTSIDE_TIMING', 'group_threads': 32,
                'block_threads': int(mode[1:]), 'checkpoint_provisioning_timed': False,
                'checkpoint_read_prepare_solve_readback_cleanup_timed': True}

    for k in m['K']:
        rotation = worker % 3
        for mode in MODES[rotation:] + MODES[:rotation]:
            for i in range(m['warmups']):
                arm(mode, k, 'warmup', i, False, i in (0, 1, 9))
        for candidate in ('b128', 'b64'):
            comparison = 'b256_vs_' + candidate
            for pair, order in enumerate(orders(m, worker, 'whole_checkpoint_K' + str(k), comparison)):
                for side in order:
                    mode = 'b256' if side == 'A' else candidate
                    measured = arm(mode, k, 'formal', f'{comparison}:{pair}:{side}', True,
                                   pair in (0, 1, 19))
                    row = {'table': 'common_checkpoint_to_output', 'case': 'whole_checkpoint',
                           'K': k, 'field': 'LandTiles+SdfTrees', 'role': 'complete_original_group',
                           'worker': worker, 'comparison': comparison, 'primary': False,
                           'mode_A': 'b256', 'mode_B': candidate, 'mode': mode,
                           'pair': pair, 'order': order, 'arm': side, **measured}
                    raw.append(row)
                    append_json(worker_dir / 'trials.jsonl', row)
            print(json.dumps({'worker': worker, 'phase': 'common', 'K': k,
                              'comparison': comparison, 'status': 'COMPLETED'}), flush=True)


def worker(args):
    gate(args.execute_performance)
    m = read_json(MATRIX)
    matrix_check(m)
    suite = args.out.resolve()
    verify_plan(read_json(suite / 'PLAN.json'))
    directory = suite / ('worker_' + str(args.worker_index))
    directory.mkdir(exist_ok=False)
    report = {'status': 'RUNNING', 'worker': args.worker_index, 'pid': os.getpid(),
              'performance': 'RUNNING', 'automatic_retries': 0,
              'failure_policy': 'Preserve failing phase; no additional CUDA calls from failed worker',
              'hardware_profiling': 'NOT_RUN'}
    write_json(directory / 'STATUS.json', report)
    raw = []
    try:
        report['telemetry_before'] = telemetry()
        from check_gpu_solver import API
        cases, params = load_execution_inputs(m)
        apis = {mode: API(ROOT / 'build/libfield_bridge.so', library(mode)) for mode in MODES}
        for api in apis.values():
            bind_timing(api)
        if args.worker_index == 0:
            selection = calibrate(m, cases, params, apis, directory, suite)
        else:
            require(digest(suite / 'SELECTION.json') == args.selection_sha256,
                    'Frozen calibration changed before worker')
            selection = read_json(suite / 'SELECTION.json')
        require(selection['matrix_sha256'] == digest(MATRIX) and selection['worker'] == 0,
                'Invalid calibration provenance')
        report['selection_sha256'] = digest(suite / 'SELECTION.json')
        write_json(directory / 'STATUS.json', report)
        resident_table(m, cases, params, apis, selection, args.worker_index, directory, raw)
        common_table(m, args.worker_index, directory, raw)
        require(len(raw) == 1280, 'Unexpected formal timing arm count')
        verify_plan(read_json(suite / 'PLAN.json'))
        require(digest(suite / 'SELECTION.json') == report['selection_sha256'],
                'Calibration changed during worker')
        report.update(status='COMPLETED', performance='MEASURED', summary=summarize(raw),
                      raw_timed_arms=len(raw), telemetry_after=telemetry(),
                      independence_unit='process and original input; internal repeats are not independent')
    except BaseException as error:
        report.update(status='FAILED', performance='FAILED_NOT_ELIGIBLE_FOR_SPEEDUP_CLAIMS',
                      error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(directory / 'STATUS.json', report)


def aggregate(suite):
    workers = [read_json(suite / f'worker_{i}/STATUS.json') for i in range(3)]
    require(all(w['status'] == 'COMPLETED' and w['raw_timed_arms'] == 1280 for w in workers),
            'Incomplete workers cannot produce aggregate speedup claims')
    groups = {}
    for w in workers:
        for r in w['summary']:
            key = (r['table'], r['case'], r['K'], r['comparison'])
            groups.setdefault(key, []).append({'worker': w['worker'], **r})
    require(len(groups) == 32, 'Expected 28 resident and 4 common comparisons')
    rows, flat = [], []
    for (table, case, k, comparison), process_rows in groups.items():
        require(len(process_rows) == 3, 'Missing process-level comparison')
        ratios = [r['median_paired_ratio'] for r in process_rows]
        row = {'table': table, 'case': case, 'K': k, 'comparison': comparison,
               'mode_A': process_rows[0]['mode_A'], 'mode_B': process_rows[0]['mode_B'],
               'primary': process_rows[0]['primary'], 'process_summaries': process_rows,
               'median_of_process_paired_medians': statistics.median(ratios),
               'process_ratio_range': [min(ratios), max(ratios)],
               'median_of_process_median_A_ms': statistics.median(r['median_A_ms'] for r in process_rows),
               'median_of_process_median_B_ms': statistics.median(r['median_B_ms'] for r in process_rows),
               'inference': 'descriptive 3-process min/max, not a confidence interval or independent scene replication'}
        rows.append(row)
        for r in process_rows:
            flat.append({key: value for key, value in r.items()
                         if key != 'paired_ratios_A_over_B'})
    write_json(suite / 'SUMMARY.json', {
        'status': 'COMPLETED', 'performance': 'MEASURED', 'rows': rows,
        'formal_timed_arms': 3840, 'selection_sha256': digest(suite / 'SELECTION.json'),
        'hardware_profiling': 'NOT_RUN', 'complete_meshing_speedup': 'NOT_RUN',
        'variant_selection': 'NONE; all negative results retained',
        'scope': 'single captured real group; natural original batches, K=3/6',
    })
    columns = sorted(set().union(*(r.keys() for r in flat)))
    with (suite / 'PROCESS_TIMINGS.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)


def output_path(path):
    out = path.resolve()
    require(out != ROOT and (out.is_relative_to(ROOT / 'results') or out.is_relative_to(ROOT / 'runs')),
            'Use a fresh child of this experiment results/ or runs/')
    return out


def orchestrate(args):
    gate(args.execute_performance)
    require(sys.platform.startswith('linux'), 'GPU workers require the isolated WSL runtime')
    prepared = fingerprint_plan()
    require(all(i['sha256'] for i in prepared['fingerprints']) and prepared['checkpoint_files'],
            'Build, input and full checkpoint artifacts must exist')
    out = output_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'PLAN.json', prepared)
    status = {'status': 'RUNNING', 'performance': 'RUNNING', 'completed_workers': [],
              'authorization_sha256': digest(AUTH), 'automatic_retries': 0}
    write_json(out / 'STATUS.json', status)
    frozen = None
    try:
        for index in range(3):
            gate(True)
            verify_plan(prepared)
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--execute-performance',
                       '--worker-index', str(index), '--out', str(out)]
            if index:
                require(digest(out / 'SELECTION.json') == frozen, 'Frozen repeat calibration changed')
                command += ['--selection-sha256', frozen]
            env = os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
            env.setdefault('CUDA_VISIBLE_DEVICES', '0')
            write_json(out / f'worker_{index}_command.json',
                       {'argv': command, 'timeout_seconds': args.worker_timeout})
            print(json.dumps({'phase': 'worker_start', 'worker': index}), flush=True)
            with (out / f'worker_{index}_stdout.log').open('xb') as stdout, (out / f'worker_{index}_stderr.log').open('xb') as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=args.worker_timeout)
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    raise
            require(code == 0, f'Worker {index} failed with exit {code}; failed phase is preserved')
            report = read_json(out / f'worker_{index}/STATUS.json')
            require(report['status'] == 'COMPLETED', 'Worker lacks a completed report')
            if index == 0:
                frozen = digest(out / 'SELECTION.json')
            require(report['selection_sha256'] == frozen, 'Worker used different repeat calibration')
            status['completed_workers'].append(index)
            write_json(out / 'STATUS.json', status)
            print(json.dumps({'phase': 'worker_complete', 'worker': index}), flush=True)
        aggregate(out)
        status.update(status='COMPLETED', performance='MEASURED', formal_timed_arms=3840)
    except BaseException as error:
        status.update(status='FAILED', performance='INCOMPLETE_NOT_ELIGIBLE_FOR_SPEEDUP_CLAIMS',
                      error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(out / 'STATUS.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--execute-performance', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker-timeout', type=int, default=1800)
    parser.add_argument('--worker-index', type=int, choices=(0, 1, 2), help=argparse.SUPPRESS)
    parser.add_argument('--selection-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.worker_timeout > 0, 'Worker timeout must be positive')
    args.out = output_path(args.out)
    if not args.execute_performance:
        require(args.worker_index is None, 'Workers require explicit execution switch')
        prepared = fingerprint_plan()
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / 'PLAN.json', prepared)
        print(json.dumps(prepared, indent=2, sort_keys=True))
    elif args.worker_index is not None:
        worker(args)
    else:
        orchestrate(args)


if __name__ == '__main__':
    main()
