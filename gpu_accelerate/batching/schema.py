"""CPU task snapshots and compatibility keys for optional offline D1 batching.

Importing this module performs no file reads or CUDA operations. Binding keys
describe a live Field; only the owning backend can attest that it is still live.
All geometry is snapshotted as immutable, little-endian, owned wire bytes.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import math
import struct
import sys

INT32_MAX = (1 << 31) - 1
SIZE_MAX = (1 << (8 * C.sizeof(C.c_size_t))) - 1
DEFAULT_MAX_BYTES = 2 << 30
FIVE = ('position', 'witness', 'valid', 'left', 'right')
OUTPUT_WIDTHS = {'position': 12, 'witness': 4, 'valid': 4, 'left': 8,
                 'right': 8, 'aux': 12, 'sdf': 4, 'xyz': 12, 'sign': 4,
                 'trace_left': 8, 'trace_right': 8}
OUTPUTS = tuple(OUTPUT_WIDTHS)


class IncompatibleTasks(ValueError):
    """Valid tasks cannot share a solver; an explicit serial fallback is possible."""


class BudgetExceeded(MemoryError):
    """A known CPU buffer lower bound exceeds the caller's admission budget."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low=0, high=SIZE_MAX):
    _require(type(value) is int and low <= value <= high,
             f'{name} must be an integer in [{low}, {high}]')
    return value


def array_bytes(value):
    """Copy a contiguous buffer without numeric conversion; never follow a pointer."""
    if isinstance(value, bytes):
        return value  # Immutable bytes already have an independent lifetime.
    if isinstance(value, C.Array):
        return C.string_at(C.addressof(value), C.sizeof(value))
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise TypeError('Expected bytes or a contiguous buffer, not a list/pointer') from exc
    _require(view.c_contiguous, 'Non-contiguous input is rejected; make an explicit C-order copy')
    return view.tobytes()


def _wire_bytes(value, ctype, count, name):
    """Check exact dtype/extent before taking a snapshot; raw bytes declare wire dtype."""
    _require(sys.byteorder == 'little' and C.sizeof(C.c_double) == 8
             and C.sizeof(C.c_float) == 4 and C.sizeof(C.c_int32) == 4,
             'D1 wire buffers require little-endian float64/float32/int32')
    expected = count * C.sizeof(ctype)
    _require(expected <= SIZE_MAX, name + ' byte extent overflows size_t')
    if isinstance(value, C.Array):
        _require(value._type_ is ctype, name + ' ctypes dtype mismatch')
        _require(C.sizeof(value) == expected, name + ' byte extent mismatch')
    elif not isinstance(value, (bytes, bytearray)):
        try:
            view = memoryview(value)
        except TypeError as exc:
            raise TypeError(name + ' requires bytes or a typed contiguous buffer') from exc
        _require(view.c_contiguous, name + ' non-contiguous input is rejected')
        # B/b/c are an explicitly untyped byte view; typed views must match exactly.
        raw = view.format in ('B', 'b', 'c') and view.itemsize == 1
        code = {C.c_double: 'd', C.c_float: 'f', C.c_int32: 'i'}[ctype]
        _require(raw or (view.itemsize == C.sizeof(ctype)
                        and view.format in (code, '<' + code, '=' + code, '@' + code)),
                 name + ' typed buffer dtype/endianness mismatch')
        _require(view.nbytes == expected, name + ' byte extent mismatch')
    else:
        _require(len(value) == expected, name + ' byte extent mismatch')
    data = array_bytes(value)
    _require(len(data) == expected, name + ' byte extent mismatch')
    return data


def checked_capacity(n, m, k, lattices, trace=False, *,
                     max_bytes=DEFAULT_MAX_BYTES, max_k=6):
    """Check host integers before constructing C arrays or calling the native ABI.

    known_buffer_bytes is only geometry plus requested readback buffers, not a
    complete device plan. The native planner must also admit workspace, CUB,
    live Field/other handles and any Graph reservation under the shared cap.
    D1 allocates with max_k (default 6), even when this solve requests smaller K.
    """
    for name, value in (('n', n), ('m', m)):
        _integer(value, name, high=INT32_MAX)
    _integer(max_k, 'max_k', high=6)
    _integer(k, 'k', high=max_k)
    _integer(lattices, 'lattices', low=1, high=256)
    _require(type(trace) is bool, 'trace must be bool')
    _integer(max_bytes, 'max_bytes', high=DEFAULT_MAX_BYTES)
    _require(n != 0 or m == 0, 'Endpoints require an owner node')
    max_queries = max(n, m)
    descriptor_capacity = max_queries * lattices
    _require(descriptor_capacity <= INT32_MAX - 127,
             'q*lattices exceeds the D1 int32 request/grid bound')
    endpoint_round_queries = (k + 1) * m
    logical_queries = n + endpoint_round_queries
    bounds_samples = (k + 1) * n
    allocated_query_slots = n + (max_k + 1) * m
    allocated_bounds_slots = (max_k + 1) * n
    total_descriptor_slots = descriptor_capacity * (max_k + 2)
    # owners are retained on host for validation; native D1 derives them from CSR.
    geometry_bytes = 24 * (n + m) + 4 * (n + 1 + m)
    output_bytes = 36 * n
    if trace:
        output_bytes += 32 * logical_queries + 16 * bounds_samples
    sizes = dict(endpoint_round_queries=endpoint_round_queries,
                 logical_queries=logical_queries, bounds_samples=bounds_samples,
                 allocated_query_slots=allocated_query_slots,
                 allocated_bounds_slots=allocated_bounds_slots,
                 total_descriptor_slots=total_descriptor_slots,
                 geometry_bytes=geometry_bytes, output_bytes=output_bytes,
                 known_buffer_bytes=geometry_bytes + output_bytes)
    # These products cover wire extents and diagnostic index tripling before C ABI.
    for name, value in list(sizes.items()) + [
            ('allocated_query_bytes', 32 * allocated_query_slots),
            ('allocated_bounds_bytes', 16 * allocated_bounds_slots),
            ('descriptor_index_bytes', 8 * total_descriptor_slots),
            ('pipeline_query_bytes', 28 * max_queries * (max_k + 2))]:
        _require(value <= SIZE_MAX, name + ' overflows size_t')
    if sizes['known_buffer_bytes'] > max_bytes:
        raise BudgetExceeded('Known geometry/readback buffer bytes exceed admission budget: '
                             f"{sizes['known_buffer_bytes']} > {max_bytes}; native workspace is additional")
    return dict(n=n, m=m, k=k, lattices=lattices, max_k=max_k,
                max_queries=max_queries, descriptor_capacity=descriptor_capacity, **sizes)


@dataclass(frozen=True)
class BindingKey:
    """Backend-issued identity, never just a Field pointer or parameter equivalence.

    field_id/context_id are opaque instance tokens issued by the owning backend;
    epoch changes whenever the Field identity is invalidated. A copied key does
    not establish liveness: runner/backend must validate it again at submission.
    """
    field_id: str
    epoch: int
    device_id: str
    context_id: str
    field_parameters_sha256: str
    numeric_profile: str
    backend: str
    k: int
    lattices: int
    trace: bool = False
    audit: bool = False

    def __post_init__(self):
        for name in ('field_id', 'device_id', 'context_id', 'numeric_profile'):
            _require(isinstance(getattr(self, name), str) and bool(getattr(self, name)),
                     name + ' must be a nonempty backend-issued identity')
        _integer(self.epoch, 'epoch')
        digest = self.field_parameters_sha256
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(c in '0123456789abcdef' for c in digest),
                 'field_parameters_sha256 must be lowercase SHA256')
        _require(self.backend in ('published', 'graph'), 'Unknown backend')
        _integer(self.k, 'k', high=6)
        _integer(self.lattices, 'lattices', low=1, high=256)
        _require(type(self.trace) is bool and type(self.audit) is bool,
                 'trace and audit must be bool')


@dataclass(frozen=True)
class Task:
    """A ready, independent task; snapshots preserve input bits and endpoint order.

    centers/endpoints accept float64 C-contiguous typed buffers or little-endian
    wire bytes. offsets/owners use int32. Non-contiguous buffers and implicit
    numeric/list conversions are rejected. Repeated endpoint coordinates and
    distinct tasks with equal geometry are allowed. owners may be omitted and
    are then generated exactly from CSR. No data is uploaded during construction.
    """
    task_id: str
    binding: BindingKey
    n: int
    m: int
    centers: bytes
    endpoints: bytes
    offsets: bytes
    owners: bytes | None = None

    def __post_init__(self):
        _require(isinstance(self.task_id, str) and bool(self.task_id), 'task_id must be a nonempty string')
        _require(isinstance(self.binding, BindingKey), 'binding must be a BindingKey')
        checked_capacity(self.n, self.m, self.binding.k, self.binding.lattices, self.binding.trace)
        for name, ctype, count in (('centers', C.c_double, 3*self.n),
                                   ('endpoints', C.c_double, 3*self.m),
                                   ('offsets', C.c_int32, self.n+1)):
            object.__setattr__(self, name, _wire_bytes(getattr(self, name), ctype, count, name))
        offsets = tuple(v[0] for v in struct.iter_unpack('<i', self.offsets))
        _require(offsets[0] == 0 and offsets[-1] == self.m, 'CSR first/final offset mismatch')
        _require(all(0 <= v <= self.m for v in offsets), 'CSR offset out of bounds')
        _require(all(a <= b for a, b in zip(offsets, offsets[1:])), 'CSR offsets are not monotone')
        for name in ('centers', 'endpoints'):
            _require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<d', getattr(self, name))),
                     name + ' contains nonfinite coordinates')
        if self.owners is None:
            owners = b''.join(struct.pack('<i', node) * (stop-start)
                              for node, (start, stop) in enumerate(zip(offsets, offsets[1:])))
        else:
            owners = _wire_bytes(self.owners, C.c_int32, self.m, 'owners')
        for node, (start, stop) in enumerate(zip(offsets, offsets[1:])):
            _require(all(v[0] == node for v in struct.iter_unpack('<i', owners[start*4:stop*4])),
                     'owners disagree with CSR at node ' + str(node))
        object.__setattr__(self, 'owners', owners)

    @property
    def k(self):
        return self.binding.k

    @property
    def name(self):
        return self.task_id

    def ctypes_geometry(self):
        """Return newly owned C arrays for one native create; retain through that call."""
        return {name: (ctype * count).from_buffer_copy(getattr(self, name))
                for name, ctype, count in (('centers', C.c_double, 3*self.n),
                                           ('endpoints', C.c_double, 3*self.m),
                                           ('offsets', C.c_int32, self.n+1),
                                           ('owners', C.c_int32, self.m))}


@dataclass(frozen=True)
class Segment:
    task_id: str
    n: int
    m: int
    node_start: int
    endpoint_start: int
    k: int


@dataclass(frozen=True)
class PackedBatch:
    task: Task
    segments: tuple[Segment, ...]
