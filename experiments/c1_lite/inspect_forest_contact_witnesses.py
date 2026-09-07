#!/usr/bin/env python3
"""Read-only exact diagnosis of already-recorded root separator failures.

Checks ONLY each recorded failing retained face against the four proposed fan
triangles. This is not a rerun of admission or a complete retained-mesh audit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from forest_exact_contact import check_triangle_contact


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect(queries, *, expected_unknowns=59):
    directory = Path(queries).resolve()
    sources = []
    entries = []
    started = time.monotonic()
    for path in sorted(directory.glob('*.json')):
        data = path.read_bytes()
        document = json.loads(data)
        if document.get('query', {}).get('kind') != 'exact_root':
            continue
        if document.get('query_token', [None])[0] != 'exact':
            raise ValueError('An exact-root record has inconsistent query mode.')
        if document.get('all_events_preserved_shared_baseline') is not True:
            raise ValueError('Root record lacks unchanged actual-baseline evidence.')
        baseline = document['baseline']
        if len(baseline) != 5:
            raise ValueError('Expected all five baseline element receipts.')
        sources.append({'path': str(path), 'sha256': digest(data),
                        'query_token': document['query_token']})
        for row in document['events']:
            if row['status'] != 'UNKNOWN':
                continue
            audit = row['case']['audit']
            if (audit.get('stage') != 'actual_five_element_exterior'
                    or audit.get('reason') != 'No sufficient exact separator for the actual retained face.'
                    or audit.get('source_vid_encoding_version') != 2
                    or audit.get('source_vid_encoding') != 'ORIGINAL_EFFECTIVE_SOURCE_VID'):
                raise ValueError('Unknown record is outside this predeclared recorded-contact diagnosis.')
            element = int(audit['element'])
            cycle = tuple((element, int(x)) for x in audit['boundary_actual_ids'])
            retained_ids = tuple(tuple(map(int, x)) for x in audit['failing_ids'])
            if (len(cycle) != 4 or len(set(cycle)) != 4 or len(retained_ids) != 3
                    or any(not 0 <= e < 5 or not 0 <= i < baseline[e]['vertices']
                           for e, i in (*cycle, *retained_ids))):
                raise ValueError('Recorded actual identities disagree with baseline counts.')
            # This is the actual append position prescribed by the compact
            # plan, not an arbitrary ID which might coincide with a retained ID.
            center_id = (element, int(baseline[element]['vertices']))
            boundary = audit['local_exact']['boundary']
            center = audit['local_exact']['center']
            retained = audit['failing_coordinates']
            if len(boundary) != 4:
                raise ValueError('Missing complete proposed fan boundary.')
            sectors = []
            for k in range(4):
                try:
                    result = check_triangle_contact(
                        (boundary[k], boundary[(k+1) % 4], center), retained,
                        (cycle[k], cycle[(k+1) % 4], center_id), retained_ids)
                    compact = {key: result[key] for key in (
                        'status', 'relation', 'proof', 'shared_ids', 'allowed_feature',
                        'input_sha256', 'equality_rank', 'basis_count', 'checked_bases',
                        'feasible_bases', 'exhaustive_basis_search', 'witness',
                        'arithmetic_work', 'maximum_observed_rational_bits') if key in result}
                    if 'reason' in result:
                        compact['reason'] = result['reason']
                except (ValueError, ArithmeticError, TypeError, IndexError, KeyError) as error:
                    compact = {'status': 'UNKNOWN_INPUT_OR_INTERNAL', 'reason': str(error),
                               'exception_type': type(error).__name__}
                sectors.append({'fan_sector': k, **compact})
            statuses = [r['status'] for r in sectors]
            if 'REJECT_POLICY_CONTACT' in statuses:
                decision = 'EXACT_FORBIDDEN_CONTACT_WITH_RECORDED_FACE'
            elif statuses == ['PASS']*4:
                decision = 'ALL_FOUR_FANS_PASS_FOR_RECORDED_FACE_ONLY'
            else:
                decision = 'UNKNOWN_RECORDED_FACE'
            entries.append({'event_id': row['event_id'], 'query_token': document['query_token'],
                'status': decision, 'source_query_sha256': digest(data), 'element': element,
                'retained_element': int(audit['failing_element']), 'retained_face': int(audit['failing_face']),
                'boundary_actual_ids': [list(x) for x in cycle], 'center_actual_id': list(center_id),
                'retained_actual_ids': [list(x) for x in retained_ids],
                'boundary_binary32': boundary, 'center_binary32': center, 'retained_binary32': retained,
                'sectors': sectors, 'complete_retained_mesh_checked': False,
                'new_contact_relative_to_baseline_proven': False, 'runtime_admission': False})
        if path.read_bytes() != data:
            raise ValueError('A completed root-query report changed while being read.')
    if len(sources) != 2 or len(entries) != expected_unknowns:
        raise ValueError('Expected two roots and exactly '+str(expected_unknowns)+' recorded UNKNOWN events; got '
                         +str(len(sources))+' roots / '+str(len(entries))+' events.')
    keys = [(row['event_id'], tuple(row['query_token'])) for row in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate event/root diagnosis.')
    by_root = {}
    for root in sorted({row['query_token'][1] for row in entries}):
        by_root[root] = dict(Counter(row['status'] for row in entries if row['query_token'][1] == root))
    kernel = Path(__file__).resolve().with_name('forest_exact_contact.py')
    return {'schema': 'forest-recorded-root-contact-diagnostic-v1',
            'status': 'COMPLETED_RECORDED_FACE_DIAGNOSIS_ONLY',
            'scope': 'Original recorded separator-failure face only; not full admission or a new-defect comparison.',
            'input_query_reports': sources, 'kernel_sha256': digest(kernel.read_bytes()),
            'events_diagnosed': len(entries), 'triangle_pairs_diagnosed': len(entries)*4,
            'event_status_counts': dict(Counter(row['status'] for row in entries)),
            'sector_status_counts': dict(Counter(s['status'] for row in entries for s in row['sectors'])),
            'by_root': by_root, 'runtime_admission_recomputed': False,
            'wall_seconds': time.monotonic()-started, 'events': entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-unknowns', type=int, default=59)
    args = parser.parse_args()
    target = args.output.resolve()
    if args.output.exists() or args.output.is_symlink() or not target.parent.is_dir():
        raise ValueError('Output must be a new file in an existing diagnostic directory.')
    if args.queries.resolve() == target.parent or args.queries.resolve() in target.parents:
        raise ValueError('Diagnostic output must not be written inside the frozen query-report directory.')
    result = inspect(args.queries, expected_unknowns=args.expected_unknowns)
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if len(encoded.encode()) > 2*1024*1024:
        raise ValueError('Compact contact diagnostic exceeded its 2 MiB output budget.')
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)
    print(json.dumps({key: result[key] for key in ('status', 'events_diagnosed', 'event_status_counts',
                       'sector_status_counts', 'by_root', 'wall_seconds')}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
