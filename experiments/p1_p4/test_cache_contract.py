#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from binocmesher.core import (
    _SLICING_CACHE_FINISH,
    _SLICING_CACHE_MANIFEST,
    _slicing_cache_contract,
    _validate_slicing_cache_contract,
    _write_slicing_cache_completion,
)


@contextmanager
def observer_mode(provenance: int, event: int):
    names = ('BINOC_PROVENANCE_V2', 'BINOC_EVENT_MODE')
    previous = {name: os.environ.get(name) for name in names}
    os.environ[names[0]] = str(provenance)
    os.environ[names[1]] = str(event)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def require_raises(action, expected_fragment: str) -> None:
    try:
        action()
    except RuntimeError as error:
        if expected_fragment not in str(error):
            raise AssertionError(
                f'unexpected error {error!r}; expected {expected_fragment!r}'
            ) from error
    else:
        raise AssertionError(f'expected RuntimeError containing {expected_fragment!r}')


def main() -> int:
    contracts = {}
    for provenance in (0, 1):
        for event in (0, 1):
            with observer_mode(provenance, event):
                contracts[(provenance, event)] = _slicing_cache_contract()

    assert contracts[(0, 0)]['provenance_enabled'] is False
    assert contracts[(1, 0)]['provenance_enabled'] is True
    assert contracts[(0, 1)]['provenance_enabled'] is True
    assert contracts[(1, 1)]['provenance_enabled'] is True
    assert contracts[(0, 1)]['event_registry_enabled'] is True
    assert contracts[(1, 0)]['event_registry_enabled'] is False
    assert len({json.dumps(value, sort_keys=True) for value in contracts.values()}) == 4

    with tempfile.TemporaryDirectory(prefix='binoc-cache-contract-') as temporary:
        root = Path(temporary)
        contract = contracts[(0, 0)]
        _write_slicing_cache_completion(root, contract)
        assert (root / _SLICING_CACHE_FINISH).is_file()
        assert (root / _SLICING_CACHE_MANIFEST).is_file()
        assert not list(root.glob('*.tmp'))
        assert _validate_slicing_cache_contract(root, contract) == contract
        require_raises(
            lambda: _validate_slicing_cache_contract(root, contracts[(0, 1)]),
            'mode/schema mismatch',
        )

    with tempfile.TemporaryDirectory(prefix='binoc-legacy-cache-') as temporary:
        root = Path(temporary)
        (root / _SLICING_CACHE_FINISH).write_text('legacy\n')
        require_raises(
            lambda: _validate_slicing_cache_contract(root, contracts[(0, 0)]),
            'no mode/schema manifest',
        )

    with tempfile.TemporaryDirectory(prefix='binoc-bad-cache-') as temporary:
        root = Path(temporary)
        (root / _SLICING_CACHE_MANIFEST).write_text('{not-json\n')
        require_raises(
            lambda: _validate_slicing_cache_contract(root, contracts[(0, 0)]),
            'manifest is unreadable',
        )

    print('PASS_CACHE_MODE_SCHEMA_CONTRACT')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
