"""One-shot stdlib-only audit of existing receipts; never imports project code."""
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT/'experiments/c1_lite/artifacts/forest_sequence_20260906'
DEADLINE = dt.datetime(2026, 9, 7, 11, 58, 55, tzinfo=dt.timezone.utc).timestamp()
START = time.monotonic()
BOUND = {}
PROGRESS = {'phase': 'STARTED'}
resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def guard():
    require(time.time() < DEADLINE and time.monotonic()-START < 120,
            'Audit time budget exceeded.')
    require(sum(p.stat().st_size for p in HERE.parent.rglob('*') if p.is_file()) < 10_000_000,
            'Combined new artifact budget exceeded.')


def data(path, expected=None):
    guard()
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(),
            'Invalid existing input path: '+str(path))
    require(path.stat().st_size < 32*1024**2, 'Unexpectedly large audit input.')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, 'Existing input binding differs: '+str(path))
    name = path.relative_to(ROOT).as_posix()
    require(name not in BOUND or BOUND[name] == digest, 'Input changed while auditing: '+name)
    BOUND[name] = digest
    return raw


def read(path, expected=None):
    return json.loads(data(path, expected).decode('utf-8'))


def local_path(recorded):
    prefix = '/mnt/e/BinocMesher/'
    require(recorded.startswith(prefix), 'Unrecognized recorded source path: '+recorded)
    return ROOT/recorded[len(prefix):]


def verify_sources(summary):
    rows = summary['executed_sources_sha256']
    for recorded, digest in rows.items():
        data(local_path(recorded), digest)
    return len(rows)


def add_report(counts, report):
    require(report.get('status') == 'PASS_PAIR_INTERACTIONS', 'Unexpected non-PASS in certified pair population.')
    counts['event_pairs'] += 1
    if report.get('proof') == 'STRICT_ACTUAL_BINARY32_SUPPORT_AABB':
        counts['support_AABB_excluded_event_pairs'] += 1
        counts['AABB_excluded_triangle_relations'] += report['triangle_pairs_excluded']
    else:
        measured = report['counts']
        exact = measured['exact_triangle_pair_calls']
        require(exact + measured['triangle_pairs_excluded_by_support_aabb'] == 32,
                'Component pair relation denominator differs.')
        require(sum(measured.get(k, 0) for k in ('Q_Q_checked', 'Q_A_OTHER_REMOVED_B_checked',
                    'Q_B_OTHER_REMOVED_A_checked')) == exact, 'Component role counts incomplete.')
        counts['support_AABB_excluded_event_pairs'] += measured.get('event_pairs_strict_support_aabb', 0)
        counts['AABB_excluded_triangle_relations'] += measured.get('triangle_pairs_excluded_by_support_aabb', 0)
        for key in ('exact_triangle_pair_calls', 'Q_Q_checked',
                    'Q_A_OTHER_REMOVED_B_checked', 'Q_B_OTHER_REMOVED_A_checked'):
            counts[key] += measured.get(key, 0)


def component():
    root = BASE/'component01'
    summary = read(root/'summary.json')
    require(summary['status'] == 'PASS_FOREST_COMPONENT_REQUESTED_CERTIFICATION', 'Component status differs.')
    sources = verify_sources(summary)
    queries, counts, statuses = {}, Counter(), Counter()
    for item in summary['query_artifacts']:
        row = read(root/item['path'], item['sha256'])
        key = row['query']['key']
        require(key not in queries, 'Duplicate component query.')
        queries[key] = row
        # Correct field is pair_proofs, not the earlier mistaken pair_checks.
        for pair in row['pair_proofs']:
            add_report(counts, pair['report'])
            statuses[pair['report']['status']] += 1
    require(len(queries) == summary['query_count'] == 18, 'Component query count differs.')
    counts['logical_triangle_relations'] = counts['event_pairs']*32
    require(counts['AABB_excluded_triangle_relations'] + counts['exact_triangle_pair_calls']
            == counts['logical_triangle_relations'], 'Component total accounting differs.')
    return queries, {'query_count': len(queries), 'counts': dict(counts),
                     'pair_statuses': dict(statuses), 'whole_stage_cost': summary['cost'],
                     'executed_sources_verified': sources, 'isolated_role_seconds': 'UNMEASURED'}


def playback(directory, certified):
    root = BASE/directory
    summary = read(root/'summary.json')
    require(summary['status'] == 'PASS_FOREST_ATOMIC_REQUESTED_SEQUENCE', 'Sequence status differs.')
    cert_sha = hashlib.sha256(data(BASE/'component01/summary.json')).hexdigest()
    require(summary['certification_sha256'] == cert_sha, 'Sequence certification binding differs.')
    sources = verify_sources(summary)
    counts, timing_sums = Counter(), Counter()
    rows = []
    matching_plans = 0
    query_keys = set()
    for item in summary['frame_artifacts']:
        frame = read(root/item['path'], item['sha256'])
        union = frame['union']
        key = frame['query']['key']
        require(key not in query_keys, 'Duplicate playback query.')
        query_keys.add(key)
        require(frame['certification_sha256'] == cert_sha, 'Frame certification binding differs.')
        require(union['status'] == 'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED', 'Union did not pass.')
        measured = union['counts']
        exact = measured['exact_triangle_pair_calls']
        n = len(frame['records'])
        require(union['expected_event_pairs'] == n*(n-1)//2 and union['expected_triangle_pairs'] == 16*n*(n-1),
                'Playback relation denominator differs.')
        require(exact + measured['triangle_pairs_excluded_by_support_aabb'] == union['expected_triangle_pairs'],
                'Playback relation accounting differs.')
        require(sum(measured.get(k, 0) for k in ('Q_Q_checked', 'Q_A_OTHER_REMOVED_B_checked',
                    'Q_B_OTHER_REMOVED_A_checked')) == exact, 'Playback role counts incomplete.')
        counts['event_pairs'] += union.get('expected_event_pairs', 0)
        counts['logical_triangle_relations'] += union.get('expected_triangle_pairs', 0)
        for name, field in (('support_AABB_excluded_event_pairs', 'event_pairs_strict_support_aabb'),
                            ('AABB_excluded_triangle_relations', 'triangle_pairs_excluded_by_support_aabb'),
                            ('exact_triangle_pair_calls', 'exact_triangle_pair_calls'),
                            ('Q_Q_checked', 'Q_Q_checked'),
                            ('Q_A_OTHER_REMOVED_B_checked', 'Q_A_OTHER_REMOVED_B_checked'),
                            ('Q_B_OTHER_REMOVED_A_checked', 'Q_B_OTHER_REMOVED_A_checked')):
            counts[name] += measured.get(field, 0)
        records = frame['records']
        counts['event_query_plans'] += len(records)
        counts['active_or_root_queries'] += bool(records)
        for name, seconds in frame['cost'].items():
            timing_sums[name] += seconds
        if records:
            cert = certified[key]
            require(frame['baseline'] == cert['baseline'], 'Certified/playback baseline differs: '+key)
            for name in ('time_mode', 'evaluation_tau', 'physical_time_hex'):
                require(frame['query'].get(name) == cert['query'].get(name), 'Query binding differs: '+key)
            original = {r['event_id']: r['plan'] for r in cert['certified_independent_plans']}
            pairs = {tuple(sorted(r['events'])): r['report'] for r in cert['pair_proofs']}
            for record in records:
                require(original[record['event_id']] == record['plan'], 'Certified/playback plan differs: '+key)
                matching_plans += 1
            ids = sorted(r['event_id'] for r in records)
            for i, first in enumerate(ids):
                for second in ids[i+1:]:
                    require(pairs[(first, second)]['status'] == 'PASS_PAIR_INTERACTIONS',
                            'Playback pair lacks prior successful component proof: '+key)
        rows.append({'query': key, 'kind': frame['query']['kind'], 'records': len(records),
                     'counts': measured, 'event_pairs': union.get('expected_event_pairs', 0),
                     'cost': frame['cost'], 'mesh_dump_count': frame.get('persistent_mesh_files', 'UNMEASURED')})
    require(len(rows) == summary['actual_scene_outputs'] == 66, 'Playback frame count differs.')
    for name, value in summary['cost_partition_totals_seconds'].items():
        require(abs(timing_sums[name]-value) < 1e-8, 'Saved timing subtotal differs from frame sum: '+name)
    return {'frames': len(rows), 'counts': dict(counts), 'frame_cost_sums_seconds': dict(timing_sums),
            'summary_cost': summary['cost'], 'executed_sources_verified': sources,
            'same_query_baseline_plan_matches': matching_plans,
            'all_playback_pairs_have_prior_component_PASS': True,
            'performance_scope': summary['performance_scope'], 'per_query': rows,
            'isolated_role_seconds': 'UNMEASURED',
            'union_seconds_active_or_root': sum(r['cost']['verified_union_seconds_including_exact_recheck_and_array_audit'] for r in rows if r['records']),
            'union_seconds_no_edit': sum(r['cost']['verified_union_seconds_including_exact_recheck_and_array_audit'] for r in rows if not r['records'])}


def main():
    output = HERE/'existing_receipt_inventory.json'
    require(not output.exists() and not (HERE/'STOP.json').exists(), 'This audit attempt is one-shot.')
    report = {'status': 'RUNNING', 'native_calls': 0, 'geometry_recomputed': False}
    try:
        data(__file__)
        data(HERE/'PROTOCOL.md')
        PROGRESS['phase'] = 'COMPONENT_RECEIPTS'
        queries, result = component()
        report['component_certification'] = result
        PROGRESS['phase'] = 'PLAYBACK_RECEIPTS'
        report['playback'] = {}
        for name in ('sequence02_omp1', 'sequence03_omp8'):
            report['playback'][name] = playback(name, queries)
        PROGRESS['phase'] = 'SUPPLEMENTAL_AUDIT'
        audit = read(BASE/'component01_independent_audit.json')
        report['independent_audit'] = {key: audit.get(key, 'UNMEASURED') for key in
                                     ('status', 'counts', 'wall_seconds', 'native_slice_calls', 'scope')}
        PROGRESS['phase'] = 'EXECUTED_SNAPSHOT'
        native = ROOT/'experiments/c1_lite/artifacts/forest_formal_20260906/native04'
        snapshot = native/'executed_source_snapshot'
        manifest = read(snapshot/'manifest.json')
        data(native/'summary.json', manifest['attempt_summary_sha256'])
        require(manifest['status'] == 'PASS_EXECUTED_SOURCE_SNAPSHOT' and len(manifest['files']) == 15,
                'Unexpected executed snapshot status/count.')
        archived_bytes = 0
        for row in manifest['files']:
            archived = data(snapshot/row['path'], row['sha256'])
            require(len(archived) == row['bytes'], 'Snapshot byte count differs.')
            archived_bytes += len(archived)
            require(archived == data(ROOT/row['path']), 'Current source differs from archived source.')
        require(archived_bytes == manifest['total_bytes'], 'Snapshot total size differs.')
        report['native04_snapshot'] = {'files_verified': len(manifest['files']),
                                      'bytes': manifest['total_bytes'], 'all_current_files_identical': True}
        PROGRESS['phase'] = 'FINAL_BINDINGS'
        for name, digest in list(BOUND.items()):
            data(ROOT/name, digest)
        report.update(status='PASS_EXISTING_RECEIPT_INVENTORY', input_sha256=BOUND,
                      opportunity_verdict='INSUFFICIENT_COST_EVIDENCE',
                      isolated_removable_cost_seconds='UNMEASURED', maximum_speedup='UNMEASURED')
    except BaseException as error:
        report.update(status='STOP_READ_ONLY_AUDIT_ERROR', reason=str(error),
                      phase=PROGRESS['phase'], traceback=traceback.format_exc())
    report.update(wall_seconds=time.monotonic()-START,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    encoded = json.dumps(report, indent=2, allow_nan=False)+'\n'
    require(len(encoded.encode()) < 2_000_000, 'Compact inventory exceeds declared bound.')
    with output.open('x', encoding='utf-8') as stream:
        stream.write(encoded)
    if report['status'].startswith('STOP'):
        with (HERE/'STOP.json').open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    print(json.dumps({key: report.get(key) for key in ('status', 'reason', 'wall_seconds', 'peak_rss_bytes')}))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
