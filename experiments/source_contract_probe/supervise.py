"""Run one declared stage, stopping the entire campaign on any unexpected result."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

DEADLINE = dt.datetime(2026, 9, 7, 11, 5, 10, tzinfo=dt.timezone.utc).timestamp()
CAPS = {'A': 120, 'B': 180, 'C0': 300}
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RSS_LIMIT = 2 * 1024**3
OUTPUT_LIMIT = 1024**3
REPO_LIMIT = 400_000_000_000


def tree_size(path):
    total = 0
    for entry in path.rglob('*'):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except FileNotFoundError:
            # The worker is allowed to remove its own disposable cache.
            continue
    return total


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def terminate_owned(child):
    if child is None:
        return
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=CAPS)
    args = parser.parse_args()
    artifacts = HERE / 'artifacts' / 'run_20260907'
    artifacts.mkdir(parents=True, exist_ok=True)
    stop = artifacts / 'STOP.json'
    receipt = artifacts / f'{args.stage}_supervisor.json'
    result = artifacts / f'{args.stage}.json'
    log_path = artifacts / f'{args.stage}.log'
    if stop.exists() or receipt.exists() or result.exists() or log_path.exists():
        print('Campaign stopped or stage already attempted; refusing execution.', flush=True)
        return 1
    report = {'stage': args.stage, 'status': 'RUNNING',
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'deadline_utc': '2026-09-07T11:05:10+00:00',
              'wall_cap_seconds': CAPS[args.stage],
              'process_rss_limit_bytes': RSS_LIMIT, 'new_files_limit_bytes': OUTPUT_LIMIT,
              'peak_rss_bytes': 0}
    started = time.monotonic()
    child = None
    try:
        if args.stage != 'A':
            previous = 'A' if args.stage == 'B' else 'B'
            for suffix in ('', '_supervisor'):
                prior = json.loads((artifacts / f'{previous}{suffix}.json').read_text())
                if prior.get('status') != 'PASS':
                    raise RuntimeError('Previous worker and supervisor must both pass.')
        report['repository_initial_bytes'] = tree_size(ROOT)
        if (time.time() >= DEADLINE or report['repository_initial_bytes'] >= REPO_LIMIT
                or tree_size(HERE) >= OUTPUT_LIMIT):
            raise RuntimeError('Preflight time or storage budget exceeded.')
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                           BINOC_SOURCE_SPLICE_PLAN='', BINOC_SOURCE_SPLICE_AUDIT='',
                           BINOC_SOURCE_SPLICE_TRACE='')
        with log_path.open('xb') as log:
            child = subprocess.Popen([sys.executable, '-B', str(HERE/'worker.py'), args.stage, str(result)],
                                     env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            while child.poll() is None:
                report['peak_rss_bytes'] = max(report['peak_rss_bytes'], rss(child.pid))
                if time.time() >= DEADLINE or time.monotonic()-started >= CAPS[args.stage]:
                    raise RuntimeError('Wall-clock budget exceeded.')
                if report['peak_rss_bytes'] >= RSS_LIMIT:
                    raise RuntimeError('Process RSS budget exceeded.')
                if tree_size(HERE) >= OUTPUT_LIMIT:
                    raise RuntimeError('New-file budget exceeded.')
                time.sleep(0.1)
            child.wait()
        report['exit_code'] = child.returncode
        worker = json.loads(result.read_text()) if result.exists() else {}
        report['peak_rss_bytes'] = max(report['peak_rss_bytes'], worker.get('peak_rss_bytes', 0))
        report['worker_status'] = worker.get('status', 'MISSING_RECEIPT')
        if child.returncode != 0 or worker.get('status') != 'PASS':
            raise RuntimeError('Worker failed: ' + str(worker.get('reason', report['worker_status'])))
        report['generated_directory_bytes'] = tree_size(HERE)
        report['repository_final_bytes'] = tree_size(ROOT)
        if time.time() >= DEADLINE or time.monotonic()-started >= CAPS[args.stage]:
            raise RuntimeError('Final wall-clock budget exceeded.')
        if report['peak_rss_bytes'] >= RSS_LIMIT:
            raise RuntimeError('Final process peak RSS budget exceeded.')
        if report['generated_directory_bytes'] >= OUTPUT_LIMIT:
            raise RuntimeError('Final new-file budget exceeded.')
        if report['repository_final_bytes'] >= REPO_LIMIT:
            raise RuntimeError('Final repository budget exceeded.')
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='STOP', reason=str(error), exception_type=type(error).__name__,
                      traceback=traceback.format_exc())
    finally:
        terminate_owned(child)
        if child is not None:
            report['exit_code'] = child.returncode
        report['wall_seconds'] = time.monotonic()-started
        encoded = json.dumps(report, indent=2, allow_nan=False)+'\n'
        if report['status'] != 'PASS':
            with stop.open('x') as stream:
                stream.write(encoded)
        with receipt.open('x') as stream:
            stream.write(encoded)
        print(encoded, flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
