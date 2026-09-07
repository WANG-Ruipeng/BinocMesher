#!/usr/bin/env python3
"""Measure two actual Forest roots and observer non-mutation; never admission."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import resource
import time
import traceback

from forest_native_campaign import PrivateCache, mesh_receipt, file_sha
from run_forest_formal import Reports, message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--build-repo', type=Path, required=True)
    parser.add_argument('--camera-inputs', type=Path, required=True)
    parser.add_argument('--effective-inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-so-sha256', default='f4263a2f47ba5283175a921e49b8867998bdd8124aac810793b34242ec43a3c9')
    args = parser.parse_args()
    reports, private, reader, query_count = Reports(args.output), None, None, 0
    started = time.monotonic()
    result = {'status': 'UNKNOWN', 'admission_rate': None, 'queries': []}
    result.update(library_sha256=file_sha(args.build_repo/'binocmesher/lib/core.so'),
        camera_inputs_sha256=file_sha(args.camera_inputs),
        effective_inputs_sha256=file_sha(args.effective_inputs),
        registry_sha256=file_sha(args.cache/'event_registry_p1.csv'))
    try:
        from forest_native_reader import initialize_forest
        private = PrivateCache(args.cache)
        camera = json.loads(args.camera_inputs.read_text())
        effective = json.loads(args.effective_inputs.read_text())
        reader = initialize_forest(args.build_repo, private.path, camera, effective,
                                   expected_so_sha256=args.expected_so_sha256)
        result['initialization'] = reader.initialization
        for root in (F(3, 2), F(5, 2)):
            receipts = []
            for enabled in (False, True):
                message('native_probe_query', tau=str(root), identity_enabled=enabled)
                snapshot = reader.slice_query(root, mode='exact', ledger=enabled)
                query_count += 1
                item = {'root': str(root), 'identity_enabled': enabled,
                    'meshes': mesh_receipt(snapshot['meshes']), 'identity_status': snapshot['identity_status'],
                    'identity_error': snapshot.get('identity_error'), 'counts': snapshot['counts'], 'cost': snapshot['cost']}
                result['queries'].append(item); receipts.append(item['meshes'])
                message('native_probe_result', **{k: item[k] for k in ('root', 'identity_enabled', 'identity_status', 'identity_error', 'counts', 'cost')})
                del snapshot
            if receipts[0] != receipts[1]:
                raise ValueError('Identity observation changed actual ordinary baseline arrays.')
        reader.close(); reader = None
        result['verification'] = private.verify(query_count)
        result['status'] = 'PASS_ORDINARY_PARITY_AND_IDENTITY' if all(q['identity_status'] == 1 for q in result['queries'] if q['identity_enabled']) else 'OBSERVER_NOT_READY_BASELINE_PARITY_PASS'
    except Exception as error:
        result.update(status='HARNESS_OR_INPUT_STOP_NOT_ALGORITHM_FAILURE', reason=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        if reader is not None:
            reader.close()
        if private is not None:
            try:
                result['final_input_verification'] = private.verify(query_count)
            except Exception as error:
                result['status'] = 'INPUT_VERIFICATION_FAILED'; result['verification_error'] = str(error)
            private.remove()
            result['private_cache_removed'] = True
        result.update(wall_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        reports.save('summary.json', result)
    message('native_probe_complete', status=result['status'])
    return 0 if result['status'] == 'PASS_ORDINARY_PARITY_AND_IDENTITY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
