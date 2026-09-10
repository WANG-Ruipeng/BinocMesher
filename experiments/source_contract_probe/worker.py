"""One-shot stage worker. Unexpected errors preserve progress and stop; never retry."""
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (128*1024**2, 128*1024**2))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'experiments/c1_lite'), str(ROOT)]


def bindings():
    selected = [*sorted(HERE.glob('*.py')), HERE/'PROTOCOL.md',
                *sorted((ROOT/'binocmesher').glob('*.py')),
                *sorted((ROOT/'experiments/c1_lite').glob('*.py')),
                ROOT/'experiments/source_splice/runtime_common.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}


def main():
    stage, output = sys.argv[1], Path(sys.argv[2])
    report = {'stage': stage, 'status': 'RUNNING',
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'native_experiment': stage == 'C0', 'wmtk_used': False,
              'rendering_used': False, 'continuous_time_claim': False}
    started = time.monotonic()
    module = None
    try:
        before = bindings()
        report['input_bindings'] = before
        manifest = output.parent/'campaign_bindings.json'
        if stage == 'A':
            with manifest.open('x') as stream:
                stream.write(json.dumps(before, indent=2)+'\n')
        elif json.loads(manifest.read_text()) != before:
            raise RuntimeError('Source/input bindings changed between campaign stages.')
        names = {'A': 'experiments.source_contract_probe.authorization',
                 'B': 'joint_mechanism', 'C0': 'c0_probe'}
        module = importlib.import_module(names[stage])
        report['result'] = module.run()
        if bindings() != before:
            raise RuntimeError('Source/input bindings changed during experimental execution.')
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='STOP_UNEXPECTED_EXCEPTION_OR_MISMATCH',
                      exception_type=type(error).__name__, reason=str(error),
                      traceback=traceback.format_exc(),
                      partial_diagnostics=getattr(module, 'PROGRESS', {}))
    report.update(wall_seconds=time.monotonic()-started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'stage': stage, 'status': report['status'],
                      'wall_seconds': report['wall_seconds'], 'reason': report.get('reason')}, indent=2), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
