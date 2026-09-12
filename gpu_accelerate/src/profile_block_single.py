"""One real SdfTrees K=3 GROUP=32 solve, owned by the bounded NCU supervisor."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import sys
import traceback
from pathlib import Path

from workspace_paths import ROOT


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def execute(args):
    require(sys.platform.startswith('linux'), 'The actual profile target requires WSL Linux')
    plan = read(args.plan)
    require(args.block in plan['blocks'] and plan['group_threads'] == 32 and plan['mode_id'] == 4,
            'Requested profile differs from the frozen plan')
    require(read(ROOT / 'configs/PERFORMANCE_AUTHORIZATION.json').get('allowed') is True,
            'Performance authorization absent')
    performance = Path(plan['performance_suite'])
    require(read(performance / 'STATUS.json')['status'] == 'COMPLETED' and
            read(performance / 'SUMMARY.json')['status'] == 'COMPLETED',
            'Formal timing must have ended successfully; never overlap')
    require(sha(Path(plan['audit_path'])) == plan['audit_sha256'], 'Accepted audit changed')
    audit = read(Path(plan['audit_path']))
    require(audit.get('status') == 'PASS_INDEPENDENT_BLOCK_AUDIT' and audit.get('issues') == [],
            'Independent block performance audit must pass first')
    require(audit.get('suite') and Path(audit['suite']).resolve() == performance.resolve(),
            'Accepted audit belongs to a different performance suite')
    evidence = audit.get('evidence', {})
    for label, path in (
            ('summary_sha256', performance / 'SUMMARY.json'),
            ('selection_sha256', performance / 'SELECTION.json'),
            ('matrix_sha256', ROOT / 'configs/BLOCK_MATRIX.json')):
        require(evidence.get(label) == sha(path), 'Accepted audit does not match current ' + label)
    for item in plan['fingerprints']:
        require(sha(ROOT / item['path']) == item['sha256'], 'Frozen profile artifact changed: ' + item['path'])
    out = args.out.resolve()
    require(out.is_relative_to(ROOT / 'results') and out != ROOT / 'results' and
            out.parent == args.plan.resolve().parent and out.name == 'b' + str(args.block),
            'Use the planned distinct block output directory')
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'block_threads': args.block, 'group_threads': 32,
              'mode': 'L1_32', 'mode_id': 4, 'case': plan['case'], 'K': 3, 'N': 24, 'M': 136,
              'logical_solver_calls': 0, 'warmups': 0, 'timing_api_calls': 0, 'trace': 0,
              'performance_comparison': False, 'automatic_retries': 0,
              'profile_plan_sha256': sha(args.plan),
              'profiler_replay': 'kernel; tool may replay the selected single launch for counters'}
    write(out / 'STATUS.json', report)
    try:
        import numpy as np
        from check_gpu_solver import API, compare, load_parameters, natural_cases
        matrix = read(ROOT / 'configs/BLOCK_MATRIX.json')
        rows = [r for r in matrix['cases'] if r['id'] == plan['case'] and r['K'] == 3]
        require(len(rows) == 1 and rows[0]['role'] == 'calibration', 'Frozen first batch changed')
        row = rows[0]
        require(row['N'] == 24 and row['M'] == 136 and row['Q'] == 568,
                'Representative natural dimensions changed')
        cases = natural_cases(Path(row['capture_dir']), (('SdfTrees', 4, 1, 1),), 3)
        matches = [case for case in cases if case.name == row['id']]
        require(len(matches) == 1, 'Expected one frozen SdfTrees case')
        case = matches[0]
        require(case.evidence['capture_input_sha256'] == row['input_sha256'] and
                len(case.centers) == row['N'] and len(case.endpoints) == row['M'] and
                case.evidence['counts'] == row['m_counts'], 'Frozen natural input differs')
        params = load_parameters(ROOT / 'inputs/accepted_scene/field_parameters.npz',
                                 plan['parameters_sha256'])
        binary = ROOT / ('build/libgpu_solver_b' + str(args.block) + '.so')
        api = API(ROOT / 'build/libfield_bridge.so', binary)
        field, view = api.create_field(case.kind, params[case.kind])
        handle = api.create_solver(case, view)
        words = (C.c_uint64 * 40)()
        api.check(api.solver.bm_solver_get_info(handle, 3, words, 40), 'initial_info')
        before = list(map(int, words))
        require(before[1] == 0 and before[31:35] == [0, 0, 0, 0],
                'Expected fresh production solver without trace/timing activity')
        actual = api.solve_once(handle, case, 4, trace=0, prepare_graph=False)
        report['logical_solver_calls'] = 1
        np.savez(out / 'actual.npz', **actual)
        mismatches = compare(case.oracle, actual)
        report.update(info_before=before, mismatches=mismatches, input_evidence=case.evidence,
                      output_contract=list(actual), actual_sha256=sha(out / 'actual.npz'),
                      solver_sha256=sha(binary))
        write(out / 'STATUS.json', report)
        if mismatches:
            np.savez(out / 'oracle.npz', **{key: case.oracle[key] for key in actual})
            raise RuntimeError('Profile target output mismatch: ' + str(mismatches))
        api.check(api.solver.bm_solver_get_info(handle, 3, words, 40), 'final_info')
        after = list(map(int, words))
        report['info_after'] = after
        require(after[31:36] == [0, 0, 1, 1, 0],
                'Expected one ordinary mode-4 solve, zero events and no graph')
        api.check(api.solver.bm_solver_destroy(handle), 'solver_destroy')
        api.check(api.field.bm_field_destroy(field), 'field_destroy')
        report.update(status='PASS', output_bitwise='PASS')
        write(out / 'STATUS.json', report)
        print(json.dumps({'status': 'PASS', 'block_threads': args.block,
                          'case': case.name, 'logical_solver_calls': 1,
                          'output_bitwise': 'PASS'}), flush=True)
    except BaseException as error:
        report.update(status='FAILED', error=repr(error), traceback=traceback.format_exc(),
                      failure_policy='Preserve first failure; no further CUDA calls from this target')
        write(out / 'STATUS.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-profile', action='store_true')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--block', type=int, choices=(256, 128, 64))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    require(args.plan.is_relative_to(ROOT / 'results'), 'Profile plan must be in current results')
    if not args.execute_profile:
        print(json.dumps(read(args.plan), indent=2))
        return
    require(args.block is not None and args.out is not None, 'Execution needs --block and --out')
    execute(args)


if __name__ == '__main__':
    main()