#!/usr/bin/env python3
"""Bounded worker for the real E2 requested-schedule production transaction.

Run separate processes for old baseline and new OMP 1/8. No mesh arrays are
written to disk and no renderer is imported. Use a fresh output directory.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'source_splice'))
from runtime_common import initialize_mesher
from e2_runtime_validation import validate_e2_runtime


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def mesh_hash(mesh):
    h = hashlib.sha256()
    for value in mesh:
        h.update(str(value.shape).encode()); h.update(value.dtype.str.encode()); h.update(value.tobytes())
    return h.hexdigest()


def array_hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def ordinary(mesher, value, mode, smooth=False):
    from binocmesher.utils.interface import AsInt
    vc, fc = np.zeros(5, np.int32), np.zeros(5, np.int32)
    if mode == 'exact':
        t = F(value)
        status = mesher.run_slicing_rational(t.numerator, t.denominator, AsInt(vc), AsInt(fc), smooth)
    else:
        status = mesher.run_slicing(float(value), AsInt(vc), AsInt(fc), smooth)
    try:
        if status != 0:
            raise RuntimeError(str(mesher.slicing_last_error()))
        v, f, tags = np.empty((int(vc[0]), 3), np.float64), np.empty((int(fc[0]), 3), np.int32), np.empty(int(vc[0]), np.int32)
        mesher.slicing_output(0, mesher.AF(v), AsInt(f), AsInt(tags))
        return v, f, tags
    finally:
        mesher.slicing_clean_up()


def queries(source, delta, smoke):
    lower, root, upper = (F(source['levels'][k]['numerator'], source['levels'][k]['denominator'])
                          for k in ('lower', 'root', 'upper'))
    exact = ([lower, (lower+root)/2, root, (root+3*upper)/4, upper] if smoke else
             [lower+(root-lower)*F(i, 16) for i in range(17)]+
             [root+(upper-root)*F(i, 16) for i in range(1, 17)])
    def physical(t):
        return float(np.longdouble(t.numerator)*np.longdouble(delta)/np.longdouble(t.denominator))
    double = [physical(t) for t in exact]
    for t in (lower, root, F(21), upper):
        value = physical(t)
        double.extend((float(np.nextafter(value, -np.inf)), value, float(np.nextafter(value, np.inf))))
    double = sorted({x.hex(): x for x in double}.values())
    return {'exact': exact, 'physical': double}


def run(args):
    repo, cache, output = args.repo.resolve(), args.cache_root.resolve(), args.output.resolve()
    if output.exists() or output == cache or cache in output.parents:
        raise ValueError('Use a fresh output outside original inputs.')
    if any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError('Symlinked original cache unsupported.')
    cache_bytes = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
    if cache_bytes > 10*1024*1024:
        raise ValueError('This driver accepts only the small fixed demo cache.')
    source_bytes = args.source_report.read_bytes(); source = json.loads(source_bytes)
    if hashlib.sha256(source_bytes).hexdigest() != '6b7f08fb7689ef8e62002c787e301a321fb343cda16d1ce37c54021b58c92dc2':
        raise ValueError('Frozen E2 source report differs; no replacement experiment.')
    before = manifest(cache); started = time.monotonic(); output.mkdir(parents=True)
    report = {'schema': 'e2-production-runtime-worker-v1', 'status': 'RUNNING',
              'omp_threads': args.omp, 'reference_only': args.reference_only, 'smoke': args.smoke,
              'core_so_sha256': sha(repo/'binocmesher/lib/core.so'),
              'source_report_sha256': hashlib.sha256(source_bytes).hexdigest(),
              'driver_sha256': sha(__file__), 'cache_copy_bytes': cache_bytes,
              'render_started': False, 'continuous_window_admitted': False,
              'production_scope': 'ACTUAL_PRODUCTION_SLICER_PLUS_ATOMIC_REQUESTED_SCHEDULE',
              'schedules': {}, 'negative_tests': {}}
    temporary = None
    try:
        os.environ['OMP_NUM_THREADS'] = str(args.omp)
        for name in ('BINOC_SOURCE_SPLICE_PLAN', 'BINOC_SOURCE_SPLICE_AUDIT', 'BINOC_SOURCE_SPLICE_TRACE'):
            os.environ.pop(name, None)
        with tempfile.TemporaryDirectory(prefix='e2-production-') as temporary:
            copied = Path(temporary)/'cache'; shutil.copytree(cache, copied)
            mesher = initialize_mesher(repo, copied)
            # Old core lacks the new metadata, but its existing time mapping is
            # independently known from this immutable demo cache shape.
            delta = float(mesher.tsize)/(2*16)
            if not args.reference_only and mesher._window_delta_t != delta:
                raise ValueError('Actual initialized time scale differs from fixed demo shape.')
            report.update(actual_delta_t_hex=delta.hex(), tsize_hex=float(mesher.tsize).hex())
            schedules = queries(source, delta, args.smoke)
            if not args.reference_only:
                from binocmesher.window_runtime import WindowRuntime, WindowSpec
                runtime, spec = WindowRuntime(mesher), WindowSpec.from_source_contract(source)
                spec.validate_layout()
            baseline_by_mode = {}
            for mode, values in schedules.items():
                if time.monotonic()-started > args.max_seconds:
                    raise TimeoutError('Bounded worker time exhausted.')
                baseline = [ordinary(mesher, value, mode) for value in values]
                baseline_by_mode[mode] = baseline
                entry = {'queries': [str(t) if mode == 'exact' else t.hex() for t in values],
                         'baseline_hashes': [mesh_hash(x) for x in baseline],
                         'baseline_shapes': [[len(x[0]), len(x[1])] for x in baseline]}
                report['schedules'][mode] = entry
                if args.reference_only:
                    continue
                # Independently check instrumentation before enabling treatment.
                traces, trace_same = [], []
                for value, expected in zip(values, baseline):
                    actual, ids, owners, status = runtime._slice(value, mode, False, True)
                    trace_same.append(mesh_hash(actual) == mesh_hash(expected))
                    if status != 1 or not trace_same[-1]:
                        raise ValueError('Identity instrumentation changed baseline or failed.')
                    traces.append((ids, owners))
                entry['instrumented_baseline_identical'] = all(trace_same)
                entry['identity_ledger_hashes'] = [{'vertices': array_hash(ids), 'owners': array_hash(owners),
                                                   'owner_rows': len(owners)} for ids, owners in traces]
                result, transaction = runtime.run(values, spec, time_mode=mode)
                entry.update(transaction=transaction, result_hashes=[mesh_hash(x) for x in result], independent=[])
                if transaction['status'] != 'COMMITTED_REQUESTED_SCHEDULE':
                    entry['entire_schedule_baseline_equal'] = all(mesh_hash(a) == mesh_hash(b) for a, b in zip(result, baseline))
                    raise ValueError('E2 schedule refused: '+str(transaction.get('fallback_reason')))
                for case, expected, actual, (ids, owners) in zip(transaction['cases'], baseline, result, traces):
                    if case['status'] == 'APPLIED_ACTUAL_QUERY':
                        tau = F(case['evaluation_tau']); requested, _ = spec.cell(tau)
                        audit = validate_e2_runtime(expected, actual, ids, owners, spec.cycle, requested,
                            consumed_owners=case['consumed_owners'], expected_center=spec.center(tau))
                        if not audit['pass']:
                            raise ValueError('Independent array/source check failed: '+str(audit))
                        entry['independent'].append(audit)
                    else:
                        if mesh_hash(expected) != mesh_hash(actual):
                            raise ValueError('Endpoint/outside query changed ordinary bytes.')
                        entry['independent'].append({'status': 'PASS_BASELINE_BYTES', 'scope': case['status']})
                # Public production API must agree with the directly audited transaction.
                public, public_report = mesher.slice_window_batch(values, source, time_mode=mode)
                entry['public_api_equal'] = ([mesh_hash(x) for x in public] == entry['result_hashes'] and
                                            public_report['status'] == transaction['status'])
                if not entry['public_api_equal']:
                    raise ValueError('Public production entry differs from verified transaction.')
            if not args.reference_only:
                values = schedules['exact']; expected = baseline_by_mode['exact']
                for name, kwargs in (('disabled', {'enabled': False}), ('invalid_spec', {})):
                    target = spec if name == 'disabled' else None
                    result, audit = runtime.run(values, target, **kwargs)
                    passed = all(mesh_hash(a) == mesh_hash(b) for a, b in zip(expected, result))
                    report['negative_tests'][name] = {'pass': passed, 'status': audit['status'], 'reason': audit.get('fallback_reason')}
                    if not passed:
                        raise ValueError('Negative transaction changed baseline: '+name)
                smooth_base = [ordinary(mesher, t, 'exact', True) for t in values]
                smooth, audit = runtime.run(values, spec, extra_smooth=True)
                smooth_ok = all(mesh_hash(a) == mesh_hash(b) for a, b in zip(smooth_base, smooth))
                report['negative_tests']['unsupported_smooth'] = {'pass': smooth_ok, 'status': audit['status']}
                if not smooth_ok:
                    raise ValueError('Unsupported smooth did not return its smooth baseline.')
                # A single late source-state corruption must discard earlier proposals.
                from dataclasses import replace
                segments = list(spec.segments); last = segments[-1]
                bad_owners = list(last[2]); bad_owners[0] = (*bad_owners[0][:3], 9999999, *bad_owners[0][4:])
                segments[-1] = (last[0], last[1], tuple(bad_owners), last[3])
                bad = replace(spec, segments=tuple(segments))
                result, audit = runtime.run(values, bad)
                late_ok = all(mesh_hash(a) == mesh_hash(b) for a, b in zip(expected, result)) and audit['status'] == 'BASELINE_ENTIRE_SCHEDULE'
                report['negative_tests']['late_failure_atomic_baseline'] = {'pass': late_ok, 'report': audit}
                if not late_ok:
                    raise ValueError('Late failure published a partial modified schedule.')
                restored, restored_report = runtime.run(values, spec)
                restored_ok = [mesh_hash(x) for x in restored] == report['schedules']['exact']['result_hashes']
                report['negative_tests']['state_reset_valid_again'] = {'pass': restored_ok, 'status': restored_report['status']}
                if not restored_ok:
                    raise ValueError('Refused transaction leaked state into the next valid one.')
            report['status'] = 'COMPLETE_REFERENCE_BASELINES' if args.reference_only else 'PASS_PRODUCTION_REQUESTED_SCHEDULE_ONLY'
    except Exception as error:
        report.update(status='STOP', error=type(error).__name__+': '+str(error))
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      original_cache_unchanged=manifest(cache) == before,
                      temporary_cache_removed=temporary is not None and not Path(temporary).exists())
        payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        if len(payload) > 8*1024*1024:
            raise ValueError('Compact worker output exceeds 8 MiB.')
        with (output/'result.json').open('xb') as handle:
            handle.write(payload)
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'elapsed_seconds': report['elapsed_seconds'], 'output_bytes': len(payload),
                      'report': str(output/'result.json')}, sort_keys=True), flush=True)
    return 0 if report['status'].startswith(('PASS_', 'COMPLETE_')) else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'cache-root', 'source-report', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--omp', type=int, choices=(1, 8), required=True)
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-seconds', type=float, default=600)
    raise SystemExit(run(p.parse_args()))
