#!/usr/bin/env python3
"""One frozen-library ordinary query in an isolated child process.

The parent owns the private cache and serializes access. This helper neither
copies nor removes it; native log append is covered by the parent's audit.
Only a new compact JSON receipt outside the cache may be written.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import time

from forest_native_campaign import file_sha, mesh_receipt
from forest_native_reader import initialize_forest, FROZEN_SO_SHA256


def parse_query(time_mode, value):
    if time_mode == 'exact':
        parsed = Fraction(value)
        return parsed, ('exact', str(parsed))
    if time_mode == 'physical':
        text = str(value).lower()
        if not text.lstrip('+-').startswith('0x') or 'p' not in text:
            raise ValueError('Physical --value must be an explicit hexadecimal float.')
        parsed = float.fromhex(text)
        if not math.isfinite(parsed):
            raise ValueError('Physical time must be finite.')
        return parsed, ('physical', parsed.hex())
    raise ValueError('Unknown query time mode.')


def run_query(*, cache_copy, build_repo, camera_inputs, effective_inputs,
              time_mode, value, output):
    cache = Path(cache_copy).resolve()
    target = Path(output)
    if target.is_symlink() or target.exists():
        raise ValueError('Output must be a new, non-symlink file.')
    target = target.resolve()
    if target == cache or cache in target.parents:
        raise ValueError('Output must be outside the parent-owned private cache.')
    if not target.parent.is_dir():
        raise ValueError('Output parent directory must already exist.')
    if not cache.is_dir():
        raise ValueError('Parent-owned private cache directory is missing.')
    parsed, token = parse_query(time_mode, value)
    camera = Path(camera_inputs).resolve()
    effective = Path(effective_inputs).resolve()
    registry = cache/'event_registry_p1.csv'
    bindings = {'camera_inputs_sha256': file_sha(camera),
                'effective_inputs_sha256': file_sha(effective),
                'registry_sha256': file_sha(registry)}
    started = time.monotonic()
    reader = None
    try:
        # Deliberately no expected-hash override: this helper always uses the
        # original frozen E2 binary, never the enlarged observer build.
        reader = initialize_forest(build_repo, cache, camera, effective)
        if reader.library_sha256 != FROZEN_SO_SHA256:
            raise ValueError('Reader did not bind the original frozen library.')
        snapshot = reader.slice_query(parsed, mode=time_mode, ledger=False)
        if (snapshot['library_sha256'] != FROZEN_SO_SHA256
                or snapshot.get('ledger_requested') is not False
                or snapshot.get('baseline_only') is not True
                or snapshot.get('extra_smooth') is not False
                or len(snapshot['meshes']) != 5):
            raise ValueError('Frozen query returned the wrong library/mode/element contract.')
        result = {'schema': 'forest-frozen-ordinary-query-receipt-v1',
                  'status': 'FROZEN_ORDINARY_QUERY_RECEIPT',
                  'query_token': list(token), 'query': snapshot['query'],
                  'identity_enabled': False, 'baseline_only': True, 'extra_smooth': False,
                  'library_sha256': FROZEN_SO_SHA256,
                  'library_path': str(reader.library_path),
                  'actual_delta_t': reader.delta_t, 'actual_delta_t_hex': reader.delta_t.hex(),
                  'meshes': mesh_receipt(snapshot['meshes']),
                  'counts': snapshot['counts'], 'cost': snapshot['cost'],
                  'initialization': reader.initialization,
                  'input_documents_sha256': snapshot['input_documents_sha256'],
                  **bindings,
                  'cache_root': str(cache), 'cache_ownership': 'PARENT_OWNS_NO_COPY_NO_DELETE',
                  'nonlog_cache_integrity': 'PARENT_CAMPAIGN_OBLIGATION',
                  'continuous_window_admitted': False, 'mesh_files_written': 0}
        if (file_sha(camera) != bindings['camera_inputs_sha256']
                or file_sha(effective) != bindings['effective_inputs_sha256']
                or file_sha(registry) != bindings['registry_sha256']
                or file_sha(reader.library_path) != FROZEN_SO_SHA256):
            raise ValueError('A bound frozen query input changed during evaluation.')
    finally:
        if reader is not None:
            reader.close()
    result['reader_closed'] = True
    result['helper_wall_seconds'] = time.monotonic()-started
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if len(encoded.encode('utf-8')) > 64*1024:
        raise ValueError('Compact receipt exceeded 64 KiB; no mesh data may be written.')
    # Exclusive creation also protects against a file appearing during native
    # execution. No failed/partial receipt is silently overwritten on retry.
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-copy', type=Path, required=True)
    parser.add_argument('--build-repo', type=Path, required=True)
    parser.add_argument('--camera-inputs', type=Path, required=True)
    parser.add_argument('--effective-inputs', type=Path, required=True)
    parser.add_argument('--time-mode', choices=('exact', 'physical'), required=True)
    parser.add_argument('--value', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run_query(**vars(args))
    print(json.dumps({'status': result['status'], 'query_token': result['query_token'],
                      'output': str(args.output.resolve())}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
