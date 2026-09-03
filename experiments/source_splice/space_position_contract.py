#!/usr/bin/env python3
'''Canonical numeric contract for SSP1 internal spatial vertices.

Production stores spatial coordinates in ``spaceT``, which is IEEE-754
binary32.  Theory artifacts use binary64.  This module makes the conversion
between those two representations explicit and independently auditable.
'''
from __future__ import annotations

import struct
from typing import Any

import numpy as np


EVENT_IR_SCHEMA = 'binoc-critical-beb1-event-ir-v3'
PLAN_SCHEMA = 'binoc-critical-beb1-whole-mesh-plan-v2'
POSITION_CONTRACT_SCHEMA = 'binoc-space-position-contract-v1'
SSP1_COORDINATE_FORMAT = 'SPACE_T_BINARY32_RNE'


def position_float64(value: Any) -> np.ndarray:
    '''Return one finite 3D position as a binary64 NumPy vector.'''
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError('space position must contain three finite coordinates')
    return result


def canonical_space_position(value: Any) -> np.ndarray:
    '''Round a theory position to production ``spaceT`` and promote to float64.'''
    theory = position_float64(value)
    with np.errstate(over='ignore', invalid='ignore'):
        encoded = theory.astype(np.float32)
    if not np.all(np.isfinite(encoded)):
        raise ValueError('space position is outside finite binary32 range')
    return encoded.astype(np.float64)


def binary32_hex(value: Any) -> list[str]:
    '''Return stable big-endian-style hexadecimal words for binary32 values.'''
    canonical = canonical_space_position(value).astype(np.float32)
    return [struct.pack('>f', float(coordinate)).hex()
            for coordinate in canonical]


def build_space_position_contract(theory_position: Any) -> dict[str, Any]:
    '''Describe the exact theory-to-runtime conversion for one 3D point.'''
    theory = position_float64(theory_position)
    canonical = canonical_space_position(theory)
    error = np.abs(canonical - theory)
    return {
        'schema': POSITION_CONTRACT_SCHEMA,
        'source_scalar': 'IEEE-754 binary64',
        'runtime_scalar': 'spaceT',
        'runtime_format': 'IEEE-754 binary32',
        'rounding_mode': 'roundTiesToEven',
        'ssp1_coordinate_format': SSP1_COORDINATE_FORMAT,
        'theory_position_float64': theory.tolist(),
        'canonical_position_float64': canonical.tolist(),
        'canonical_binary32_hex': binary32_hex(canonical),
        'maximum_absolute_quantization_error': float(np.max(error)),
        'exactly_representable': bool(np.array_equal(theory, canonical)),
    }


def space_position_contract_is_valid(value: Any) -> bool:
    '''Validate every redundant field of a serialized position contract.'''
    if not isinstance(value, dict):
        return False
    try:
        theory = position_float64(value['theory_position_float64'])
        canonical = position_float64(value['canonical_position_float64'])
        expected = build_space_position_contract(theory)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        value.get('schema') == POSITION_CONTRACT_SCHEMA and
        value.get('source_scalar') == expected['source_scalar'] and
        value.get('runtime_scalar') == expected['runtime_scalar'] and
        value.get('runtime_format') == expected['runtime_format'] and
        value.get('rounding_mode') == expected['rounding_mode'] and
        value.get('ssp1_coordinate_format') == SSP1_COORDINATE_FORMAT and
        np.array_equal(canonical, expected['canonical_position_float64']) and
        value.get('canonical_binary32_hex') ==
        expected['canonical_binary32_hex'] and
        value.get('maximum_absolute_quantization_error') ==
        expected['maximum_absolute_quantization_error'] and
        value.get('exactly_representable') is
        expected['exactly_representable']
    )


def plan_position_contract_is_valid(plan: Any) -> bool:
    '''Require a v2 plan whose runtime position matches its numeric contract.'''
    if not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA:
        return False
    contract = plan.get('critical_position_contract')
    if not space_position_contract_is_valid(contract):
        return False
    try:
        plan_position = position_float64(plan['critical_position'])
        canonical = position_float64(contract['canonical_position_float64'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return np.array_equal(plan_position, canonical)
