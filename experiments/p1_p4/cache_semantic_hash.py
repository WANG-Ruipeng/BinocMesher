#!/usr/bin/env python3
"""Canonical hashes for the GCC/x86-64 HP/HV cache ABI used by the smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path


HV_RECORD_SIZE = 28
HP_RECORD_SIZE = 68


def _records(path: Path, record_size: int) -> list[memoryview]:
    payload = path.read_bytes()
    if len(payload) < 4:
        raise ValueError(f'{path}: missing int32 record-count header')
    count = struct.unpack_from('<i', payload, 0)[0]
    if count < 0:
        raise ValueError(f'{path}: negative record count')
    expected = 4 + count * record_size
    if len(payload) != expected:
        raise ValueError(
            f'{path}: expected {expected} bytes for {count} records, '
            f'found {len(payload)}'
        )
    view = memoryview(payload)
    return [
        view[4 + index * record_size:4 + (index + 1) * record_size]
        for index in range(count)
    ]


def canonical_hypervertex_records(path: Path) -> list[bytes]:
    canonical = []
    identities = set()
    for record in _records(path, HV_RECORD_SIZE):
        node = struct.unpack_from('<i', record, 0)[0]
        group = struct.unpack_from('<b', record, 4)[0]
        identity = (node, group)
        if identity in identities:
            raise ValueError(f'{path}: duplicate HVID {identity}')
        identities.add(identity)
        canonical.append(
            struct.pack('<ib', node, group)
            + bytes(record[8:20])
            + bytes((record[20], record[21], record[24]))
        )
    return sorted(canonical)


def canonical_hyperpoly_records(path: Path) -> list[bytes]:
    canonical = []
    for record in _records(path, HP_RECORD_SIZE):
        value = bytearray()
        for corner in range(8):
            offset = corner * 8
            node = struct.unpack_from('<i', record, offset)[0]
            group = struct.unpack_from('<b', record, offset + 4)[0]
            value.extend(struct.pack('<ib', node, group))
        value.extend(struct.pack('<b', struct.unpack_from('<b', record, 64)[0]))
        canonical.append(bytes(value))
    return sorted(canonical)


def semantic_file_hash(path: Path, kind: str) -> str:
    if kind == 'hypervertices':
        records = canonical_hypervertex_records(path)
    elif kind == 'hyperpolys':
        records = canonical_hyperpoly_records(path)
    else:
        raise ValueError(f'unknown cache kind: {kind}')
    digest = hashlib.sha256()
    digest.update(kind.encode('ascii') + b'\0')
    digest.update(struct.pack('<Q', len(records)))
    for record in records:
        digest.update(struct.pack('<Q', len(record)))
        digest.update(record)
    return digest.hexdigest()


def semantic_source_cache_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for kind in ('hypervertices', 'hyperpolys'):
        directory = root / kind
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('*.bin')):
            relative = path.relative_to(root).as_posix()
            manifest[relative] = semantic_file_hash(path, kind)
    return manifest


def _hv_record(
    node: int,
    group: int,
    xyz: tuple[float, float, float],
    times: tuple[int, int],
    in_view: int,
    padding: int,
) -> bytes:
    record = bytearray([padding] * HV_RECORD_SIZE)
    struct.pack_into('<i', record, 0, node)
    struct.pack_into('<b', record, 4, group)
    struct.pack_into('<3f', record, 8, *xyz)
    struct.pack_into('<2b', record, 20, *times)
    struct.pack_into('<b', record, 24, in_view)
    return bytes(record)


def _hp_record(
    hvids: list[tuple[int, int]],
    element: int,
    padding: int,
) -> bytes:
    if len(hvids) != 8:
        raise ValueError('HP fixture requires eight HVID slots')
    record = bytearray([padding] * HP_RECORD_SIZE)
    for corner, (node, group) in enumerate(hvids):
        struct.pack_into('<i', record, corner * 8, node)
        struct.pack_into('<b', record, corner * 8 + 4, group)
    struct.pack_into('<b', record, 64, element)
    return bytes(record)


def _write_vector(path: Path, records: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<i', len(records)) + b''.join(records))


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix='binoc-semantic-cache-') as temporary:
        root = Path(temporary)
        first = root / 'first'
        second = root / 'second'
        hvids = [(index + 10, index % 3) for index in range(8)]
        hv_a = _hv_record(10, 0, (1.0, 2.0, 3.0), (4, 5), 1, 0x00)
        hv_b = _hv_record(11, 1, (-1.0, 0.5, 8.0), (6, 7), 0, 0x00)
        hv_a_padding = _hv_record(
            10, 0, (1.0, 2.0, 3.0), (4, 5), 1, 0xA5)
        hv_b_padding = _hv_record(
            11, 1, (-1.0, 0.5, 8.0), (6, 7), 0, 0x5A)
        hp_a = _hp_record(hvids, 2, 0x00)
        hp_a_padding = _hp_record(hvids, 2, 0xC3)
        _write_vector(first / 'hypervertices/1.bin', [hv_a, hv_b])
        _write_vector(second / 'hypervertices/1.bin', [hv_b_padding, hv_a_padding])
        _write_vector(first / 'hyperpolys/1.bin', [hp_a])
        _write_vector(second / 'hyperpolys/1.bin', [hp_a_padding])
        assert semantic_source_cache_manifest(first) == semantic_source_cache_manifest(second)

        changed = root / 'changed'
        changed_hv = _hv_record(
            10, 0, (1.0, 2.0, 3.25), (4, 5), 1, 0xA5)
        _write_vector(changed / 'hypervertices/1.bin', [changed_hv, hv_b_padding])
        _write_vector(changed / 'hyperpolys/1.bin', [hp_a_padding])
        assert semantic_source_cache_manifest(first) != semantic_source_cache_manifest(changed)

        truncated = root / 'truncated.bin'
        truncated.write_bytes(struct.pack('<i', 1) + b'partial')
        try:
            canonical_hypervertex_records(truncated)
        except ValueError:
            pass
        else:
            raise AssertionError('truncated semantic cache was accepted')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs='*', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print('PASS_SEMANTIC_HP_HV_CACHE_HASH')
    if args.roots:
        result = {
            str(root): semantic_source_cache_manifest(root)
            for root in args.roots
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    if not args.self_test and not args.roots:
        parser.error('provide --self-test or at least one cache root')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
