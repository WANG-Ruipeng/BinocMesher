"""One supervised connectivity experiment, without changing frozen evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRIOR = HERE.parent/'graph_lift_probe'
FROZEN = PRIOR/'artifacts'/'run_20260907_130141'
WALL_SECONDS, ADDRESS_BYTES, OUTPUT_BYTES = 1800, 4*1024**3, 100*1024**2
REPO_LIMIT = 400_000_000_000


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def repo_size():
    return int(subprocess.check_output(['du','-sb',str(REPO)],text=True).split()[0])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bindings():
    paths = list(p for p in HERE.iterdir() if p.suffix in ('.py','.md'))
    paths += [PRIOR/name for name in ('model.py','metrics.py','run_probe.py')]
    paths += [FROZEN/'inputs.json', FROZEN/'summary.json']+list(sorted(FROZEN.glob('case_*.json')))
    return {str(p.relative_to(REPO)):digest(p) for p in sorted(paths)}


def validate_prior():
    inputs = read(FROZEN/'inputs.json')
    for name, expected in inputs['source_bindings'].items():
        if digest(PRIOR/name) != expected:
            raise RuntimeError(f'prior frozen source changed: {name}')
    if len(inputs['input_cases']) != 24:
        raise RuntimeError('prior case denominator changed')
    if read(FROZEN/'summary.json')['status'] != 'STAGE_1_COMPLETE_STOP_BEFORE_STAGE_2':
        raise RuntimeError('prior campaign not complete')
    return inputs


def worker(out):
    resource.setrlimit(resource.RLIMIT_AS,(ADDRESS_BYTES,ADDRESS_BYTES))
    import numpy as np
    import scipy
    from experiment import LEVELS, measure_case, summarize
    initial = bindings()
    inputs = validate_prior()
    save(out/'inputs.json', {'created_utc':utc(), 'source_bindings':initial,
        'frozen_source_run':str(FROZEN), 'prior_model_count':12,
        'input_cases':inputs['input_cases'], 'levels':LEVELS,
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'scope':'LOCAL_CONNECTIVITY_ATTRIBUTION_ONLY',
        'historical_gl_identity':'UNRESOLVED_NO_ORIGINAL_GL_CLAIM',
        'oracle_envelope_is_not_algorithm':True})
    start = time.monotonic()
    cases = []
    for index, case_input in enumerate(inputs['input_cases']):
        if time.monotonic()-start > WALL_SECONDS or size(HERE)>OUTPUT_BYTES:
            raise RuntimeError('worker time/storage budget exhausted')
        old = read(FROZEN/f'case_{index:02d}.json')
        if old['id'] != case_input['id']:
            raise RuntimeError('prior per-case identity mismatch')
        case = measure_case(case_input,old)
        if bindings()!=initial:
            raise RuntimeError('experiment or frozen evidence changed during campaign')
        save(out/f'case_{index:02d}.json',case)
        cases.append(case)
        print(json.dumps({'case':index+1,'of':len(inputs['input_cases']), 'id':case['id'],
            'seconds':case['elapsed_seconds'],'states':sum(p['enumeration_stats']['states'] for p in case['pointsets'].values())}),flush=True)
    report = {'status':'CONNECTIVITY_PROBE_COMPLETE_STOP', 'created_utc':utc(),
        'case_count':len(cases), 'distinct_source_models':12,
        'all_invariants_passed':True, 'all_frozen_bridges_passed':True,
        'square_alias_cases':sum(case['square_equivalence_passed'] is True for case in cases),
        'maximum_orbit_states':max(p['enumeration_stats']['states'] for c in cases for p in c['pointsets'].values()),
        'evaluation_unique_surface_measurements':sum(c['evaluation_cost']['unique_surface_measurements'] for c in cases),
        'bridge_measurements':sum(c['evaluation_cost']['bridge_measurements'] for c in cases),
        'summaries':summarize(cases), 'elapsed_seconds':time.monotonic()-start,
        'worker_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'claims_not_tested':['original_GL_reproduction','new_algorithm_novelty','original_target_accuracy',
                            'temporal_consistency','render_quality','production_safety'],
        'source_bindings':initial}
    save(out/'summary.json',report)
    print(json.dumps({'status':report['status'],'seconds':report['elapsed_seconds']}),flush=True)


def supervise():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out = HERE/'artifacts'/f'run_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    before = repo_size()
    initial = bindings()
    save(out/'launch.json',{'created_utc':utc(), 'scope':'CONNECTIVITY_ONLY',
        'wall_limit_seconds':WALL_SECONDS,'address_space_limit_bytes':ADDRESS_BYTES,
        'new_directory_limit_bytes':OUTPUT_BYTES,'repo_limit_bytes':REPO_LIMIT,
        'repo_bytes_before':before,'source_bindings':initial,
        'command':[sys.executable,str(Path(__file__).resolve()),'--worker',str(out)]})
    if before>=REPO_LIMIT or size(HERE)>=OUTPUT_BYTES:
        save(out/'STOP.json',{'status':'STOP','reason':'preflight storage budget','created_utc':utc()})
        return 2
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    env['PYTHONDONTWRITEBYTECODE']='1'
    start=time.monotonic()
    reason=None
    print(f'RUN_DIRECTORY={out}',flush=True)
    with (out/'worker.log').open('x',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(out)],
            stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        while proc.poll() is None:
            if time.monotonic()-start>WALL_SECONDS:
                reason='wall limit'
            elif size(HERE)>OUTPUT_BYTES:
                reason='storage limit'
            if reason:
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
                break
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    after=repo_size()
    if after>=REPO_LIMIT:
        reason=reason or 'repository storage limit'
    if bindings()!=initial:
        reason=reason or 'source/input binding changed'
    complete=proc.returncode==0 and (out/'summary.json').exists() and reason is None
    receipt={'status':'COMPLETE' if complete else 'STOP','created_utc':utc(),
        'worker_exit_code':proc.returncode,'reason':reason,'automatic_restart':False,
        'elapsed_seconds':time.monotonic()-start, 'repo_bytes_after':after,
        'new_directory_bytes':size(HERE)}
    save(out/'supervisor.json',receipt)
    if not complete and not (out/'STOP.json').exists():
        save(out/'STOP.json',receipt)
    print(json.dumps(receipt),flush=True)
    return 0 if complete else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except Exception as exc:
            traceback.print_exc()
            save(args.worker/'STOP.json',{'status':'STOP','created_utc':utc(),
                'reason':str(exc),'type':type(exc).__name__,
                'completed_cases':len(list(args.worker.glob('case_*.json')))})
            sys.exit(2)
    else:
        sys.exit(supervise())
