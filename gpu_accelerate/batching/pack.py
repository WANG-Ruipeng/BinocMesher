"""Prefix packing and bit-preserving round-major output dispatch, CPU only.

The CSR and trace indexing are migrated from the recorded mixed-job experiment;
experiment-specific paths, imports, fixed scene/shape and geometry rejection are
not part of this reusable module.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import struct

from .schema import (BudgetExceeded, DEFAULT_MAX_BYTES, FIVE, IncompatibleTasks,
                     OUTPUTS, OUTPUT_WIDTHS, PackedBatch, Segment, Task,
                     _integer, _require, _wire_bytes, array_bytes, checked_capacity)


def pack_tasks(tasks, *, max_bytes=DEFAULT_MAX_BYTES):
    """Return a PackedBatch, or None for no tasks; no waiting or task replication.

    Every input task is used once in supplied order. Equal geometry is permitted
    for independently supplied tasks, but task IDs must be unique. Incompatible
    BindingKeys raise IncompatibleTasks; malformed input is a plain ValueError.
    Caller must still ask the Field/backend to validate key liveness.
    """
    tasks = tuple(tasks)
    if not tasks:
        return None
    _require(all(isinstance(t, Task) for t in tasks), 'pack_tasks requires Task snapshots')
    _require(len({t.task_id for t in tasks}) == len(tasks), 'Duplicate task_id')
    if any(t.binding != tasks[0].binding for t in tasks[1:]):
        raise IncompatibleTasks('Tasks differ in Field instance/epoch, device/context, parameters, '
                                'numerical profile, backend, K, trace or audit')
    binding = tasks[0].binding
    n, m = sum(t.n for t in tasks), sum(t.m for t in tasks)
    checked_capacity(n, m, binding.k, binding.lattices, binding.trace, max_bytes=max_bytes)
    segments = []
    offsets = []
    owners = []
    node_start = endpoint_start = 0
    for task in tasks:
        segments.append(Segment(task.task_id, task.n, task.m, node_start, endpoint_start, task.k))
        offsets.extend(v[0]+endpoint_start for v in struct.iter_unpack('<i', task.offsets[:-4]))
        owners.extend(v[0]+node_start for v in struct.iter_unpack('<i', task.owners))
        node_start += task.n
        endpoint_start += task.m
    offsets.append(m)
    merged = Task('__offline_mixed__', binding, n, m,
                  b''.join(t.centers for t in tasks), b''.join(t.endpoints for t in tasks),
                  struct.pack('<'+'i'*len(offsets), *offsets),
                  struct.pack('<'+'i'*len(owners), *owners))
    return PackedBatch(merged, tuple(segments))


def _layout(segments):
    segments = tuple(segments)
    if not segments:
        return segments, 0, 0, 0
    _require(all(isinstance(s, Segment) for s in segments), 'Expected Segment records')
    n = m = 0
    k = _integer(segments[0].k, 'segment K', high=6)
    seen = set()
    for s in segments:
        _require(isinstance(s.task_id, str) and s.task_id and s.task_id not in seen,
                 'Duplicate/empty segment task_id')
        for name in ('n', 'm', 'node_start', 'endpoint_start'):
            _integer(getattr(s, name), 'segment ' + name)
        _require(s.n != 0 or s.m == 0, 'Segment endpoints require nodes')
        _require(type(s.k) is int and s.k == k and s.node_start == n and s.endpoint_start == m,
                 'Segments must be contiguous, ordered and have the same K')
        seen.add(s.task_id)
        n += s.n
        m += s.m
    return segments, n, m, k


def copy_outputs(outputs, n, m, k):
    """Validate requested wire extents/dtypes and return independent immutable bytes.

    Five final outputs are mandatory; optional aux and trace fields retain their
    actual presence. Backend/runner enforces requested trace configuration.
    """
    _require(set(FIVE).issubset(outputs), 'Missing required five final outputs')
    _require(set(outputs).issubset(OUTPUTS), 'Unknown output field')
    _integer(n, 'output n'); _integer(m, 'output m'); _integer(k, 'output k', high=6)
    result = {}
    for name in OUTPUTS:
        if name not in outputs:
            continue
        queries = n if name in FIVE else ((k+1)*n if name.startswith('trace_') else n+(k+1)*m)
        ctype = C.c_int32 if name in ('witness', 'valid', 'sign') else (
            C.c_double if name in ('left', 'right', 'trace_left', 'trace_right') else C.c_float)
        count = queries * OUTPUT_WIDTHS[name] // C.sizeof(ctype)
        result[name] = _wire_bytes(outputs[name], ctype, count, name)
    return result


def split_outputs(merged_outputs, segments):
    """Return {task_id: {output: owned bytes}} in original task order.

    Witness is a node-local rank and is copied verbatim. Query traces contain
    all centers followed by each endpoint round. Bounds traces are round-major
    node stripes. No returned object borrows a mutable native readback buffer.
    """
    segments, n, m, k = _layout(segments)
    if not segments:
        _require(not merged_outputs, 'No output buffers are valid for an empty task list')
        return {}
    source = copy_outputs(merged_outputs, n, m, k)
    result = {s.task_id: {} for s in segments}
    for name, data in source.items():
        w = OUTPUT_WIDTHS[name]
        for s in segments:
            if name in FIVE:
                part = data[s.node_start*w:(s.node_start+s.n)*w]
            elif name in ('trace_left', 'trace_right'):
                part = b''.join(data[(r*n+s.node_start)*w:(r*n+s.node_start+s.n)*w]
                                for r in range(k+1))
            else:
                part = data[s.node_start*w:(s.node_start+s.n)*w] + b''.join(
                    data[(n+r*m+s.endpoint_start)*w:(n+r*m+s.endpoint_start+s.m)*w]
                    for r in range(k+1))
            result[s.task_id][name] = part
    return result


def output_hashes(outputs):
    return {name: hashlib.sha256(array_bytes(value)).hexdigest() for name, value in outputs.items()}


def compare_outputs(actual, expected):
    """Exact CPU comparison only; return the first byte/key difference, never solve."""
    a = {name: array_bytes(value) for name, value in actual.items()}
    e = {name: array_bytes(value) for name, value in expected.items()}
    result = dict(status='PASS', first_difference=None,
                  actual_hashes=output_hashes(a), expected_hashes=output_hashes(e))
    if set(a) != set(e):
        result.update(status='FAIL', first_difference=dict(kind='keys',
                      missing=sorted(set(e)-set(a)), extra=sorted(set(a)-set(e))))
        return result
    for name in e:
        if a[name] == e[name]:
            continue
        av, ev = a[name], e[name]
        at = next((i for i, (x, y) in enumerate(zip(av, ev)) if x != y), min(len(av), len(ev)))
        result.update(status='FAIL', first_difference=dict(field=name, byte_offset=at,
                      kind='bytes' if at < min(len(av), len(ev)) else 'length',
                      actual_length=len(av), expected_length=len(ev)))
        break
    return result
