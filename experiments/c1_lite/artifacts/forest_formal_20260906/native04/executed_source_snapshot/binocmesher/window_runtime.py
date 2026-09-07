"""Opt-in, atomic requested-schedule runtime for source-labelled graph disks.

The ordinary C++ slicer remains the source of every baseline vertex, face and
identity. This module never welds by coordinates. It publishes a whole requested
schedule only after every replacement passes checks on its actual coordinates.
This API certifies the requested schedule, NOT every real time in the window.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from .window_geometry_runtime import check_actual_patch, _cross, _p, _sub


_LOCK = threading.RLock()
MAX_BATCH_BYTES = 512*1024*1024
MAX_QUERIES = 128


def _fraction(value):
    if isinstance(value, dict):
        return F(int(value['numerator']), int(value['denominator']))
    return F(value)


def source_key(value):
    if isinstance(value, str):
        a, b = (tuple(map(int, part.split(':'))) for part in value.split('|'))
    else:
        row = tuple(int(x) for x in value)
        if len(row) != 4:
            raise ValueError('Expected an ordered four-integer VID.')
        a, b = row[:2], row[2:]
    if len(a) != 2 or len(b) != 2:
        raise ValueError('Malformed SourceVID.')
    return tuple((*a, *b) if a <= b else (*b, *a))


def _canon(face):
    row = tuple(int(x) for x in face)
    return min(row[i:]+row[:i] for i in range(3))


def round_binary32(value):
    """Exact nearest/ties-even, avoiding rational -> double -> float ties."""
    value = F(value)
    seed = np.float32(float(value))
    if not np.isfinite(seed):
        raise ValueError('Center overflows binary32.')
    choices = [seed, np.nextafter(seed, np.float32(-np.inf)),
               np.nextafter(seed, np.float32(np.inf))]
    choices = [x for x in choices if np.isfinite(x)]
    return float(min(choices, key=lambda x: (abs(F.from_float(float(x))-value),
                       int(np.asarray(x, np.float32).view(np.uint32)) & 1)))


@dataclass(frozen=True)
class WindowSpec:
    lower: F
    root: F
    upper: F
    cycle: tuple
    anchors: tuple
    segments: tuple
    points: tuple
    source_digest: str
    cache_digest: str

    @classmethod
    def from_source_contract(cls, document):
        """Parse a proposal, not a trusted PASS/admission flag.

        All actual ownership, interface and geometry are rechecked. A historical
        ideal PASS field never authorizes a production replacement by itself.
        """
        levels = tuple(_fraction(document['levels'][k]) for k in ('lower', 'root', 'upper'))
        if not 0 <= levels[0] < levels[1] < levels[2]:
            raise ValueError('Invalid source window ordering.')
        cycle = tuple(source_key(x) for x in document['boundary_cycle'])
        if len(cycle) != 4 or len(set(cycle)) != 4:
            raise ValueError('Expected four distinct source boundary identities.')
        anchors = tuple(tuple(_fraction(x) for x in document['anchors'][k]['position'])
                        for k in ('lower', 'root', 'upper'))
        if any(len(p) != 3 for p in anchors):
            raise ValueError('Expected three 3D anchors.')
        def cell(row):
            owners = tuple(tuple(int(x) for x in owner) for owner in row['owners'])
            faces = tuple(tuple(source_key(x) for x in face) for face in row['source_faces'])
            if (not owners or len(set(owners)) != len(owners) or
                    any(len(owner) != 7 or owner[0] != 0 or min(owner) < 0 for owner in owners) or
                    len(faces) != 2 or any(len(f) != 3 for f in faces) or
                    tuple(source_key(x) for x in row['boundary_cycle']) != cycle):
                raise ValueError('Malformed single-element source owner/face contract.')
            return owners, faces
        segments = tuple((_fraction(row['t0']), _fraction(row['t1']), *cell(row))
                         for row in document['segments'])
        points = tuple((_fraction(row['time']), *cell(row)) for row in document['breakpoint_points'])
        if (not segments or segments[0][0] != levels[0] or segments[-1][1] != levels[2] or
                any(a >= b or a < levels[1] < b for a, b, *_ in segments) or
                any(segments[i][1] != segments[i+1][0] for i in range(len(segments)-1)) or
                len(set(t for t, *_ in points)) != len(points) or
                set(t for t, *_ in points) != {t for a, b, *_ in segments for t in (a, b)}):
            raise ValueError('Source schedule has missing/duplicate points or gaps.')
        encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return cls(*levels, cycle, anchors, segments, points,
                   hashlib.sha256(encoded).hexdigest(), str(document['cache_input_sha256']))

    def validate_layout(self):
        if not all(isinstance(x, F) for x in (self.lower, self.root, self.upper)) or not 0 <= self.lower < self.root < self.upper:
            raise ValueError('Invalid immutable window bounds.')
        if (not isinstance(self.cycle, tuple) or len(self.cycle) != 4 or
                any(not isinstance(x, tuple) or len(x) != 4 or
                    any(type(v) is not int or v < 0 for v in x) or source_key(x) != x for x in self.cycle) or
                len(set(self.cycle)) != 4):
            raise ValueError('Invalid immutable source cycle.')
        if (not isinstance(self.anchors, tuple) or len(self.anchors) != 3 or
                any(not isinstance(p, tuple) or len(p) != 3 or
                    any(not isinstance(x, F) or max(x.numerator.bit_length(), x.denominator.bit_length()) > 4096 for x in p)
                    for p in self.anchors)):
            raise ValueError('Invalid immutable anchor values.')
        if not isinstance(self.segments, tuple) or not 1 <= len(self.segments) <= 128 or not isinstance(self.points, tuple):
            raise ValueError('Invalid bounded source schedule.')
        previous = self.lower
        for a, b, owners, faces in self.segments:
            if a != previous or not a < b or a < self.root < b:
                raise ValueError('Source schedule has a gap, overlap or missing root.')
            previous = b
        if previous != self.upper or len(self.points) != len(self.segments)+1 or {p[0] for p in self.points} != {t for a,b,*_ in self.segments for t in (a,b)}:
            raise ValueError('Source singleton schedule differs from complete branch boundaries.')
        for owners, faces in [row[2:] for row in self.segments]+[row[1:] for row in self.points]:
            if (not isinstance(owners, tuple) or not 1 <= len(owners) <= 1024 or
                    any(not isinstance(o, tuple) or len(o) != 7 or o[0] != 0 or
                        any(type(x) is not int or x < 0 for x in o) for o in owners) or
                    len(set(owners)) != len(owners) or not isinstance(faces, tuple) or len(faces) != 2 or
                    any(not isinstance(face, tuple) or len(face) != 3 or len(set(face)) != 3 or
                        not set(face) <= set(self.cycle) for face in faces)):
                raise ValueError('Invalid immutable source owner/face contract.')

    def cell(self, tau):
        for t, owners, faces in self.points:
            if tau == t:
                return owners, faces
        for a, b, owners, faces in self.segments:
            if a < tau < b:
                return owners, faces
        raise ValueError('Query is outside the source schedule.')

    def center(self, tau):
        index = 0 if tau <= self.root else 1
        a, b = (self.lower, self.root) if index == 0 else (self.root, self.upper)
        u = (tau-a)/(b-a)
        return np.asarray([round_binary32((1-u)*x+u*y)
                           for x, y in zip(self.anchors[index], self.anchors[index+1])], np.float64)


def resolve_support(mesh, vertex_ledger, owner_ledger, spec, tau):
    """Resolve the expected source faces independently of the treatment output."""
    v, f, tags = mesh
    if vertex_ledger.shape != (len(v), 4) or owner_ledger.ndim != 2 or owner_ledger.shape[1] != 11:
        raise ValueError('Missing or malformed actual identity ledger.')
    mapping = defaultdict(list)
    for i, row in enumerate(vertex_ledger):
        mapping[source_key(row)].append(i)
    if any(len(mapping[key]) != 1 for key in spec.cycle):
        raise ValueError('Boundary SourceVID is missing or maps to multiple actual IDs.')
    cycle = tuple(mapping[key][0] for key in spec.cycle)
    requested, source_faces = spec.cell(tau)
    if any(len(mapping[key]) != 1 for face in source_faces for key in face):
        raise ValueError('Source triangle identity is missing or ambiguous.')
    expected = {_canon(tuple(mapping[key][0] for key in face)) for face in source_faces}
    if len(expected) != 2:
        raise ValueError('Original source faces are not two distinct triangles.')
    by_face, by_owner = defaultdict(list), {}
    for row in owner_ledger:
        owner = tuple(int(x) for x in row[:7]); face_id = int(row[7])
        if (owner in by_owner or not 0 <= face_id < len(f) or
                tuple(int(x) for x in row[8:]) != tuple(int(x) for x in f[face_id])):
            raise ValueError('Owner ledger is duplicated or disagrees with actual native faces.')
        by_owner[owner] = face_id; by_face[face_id].append(owner)
    if set(by_face) != set(range(len(f))):
        raise ValueError('Native face has no actual raw owner evidence.')
    if any(owner not in by_owner for owner in requested):
        raise ValueError('Requested source owner was not emitted.')
    removed = tuple(sorted({by_owner[owner] for owner in requested}))
    if len(removed) != 2 or {_canon(f[i]) for i in removed} != expected:
        raise ValueError('Owners do not resolve exactly to the independently expected source faces.')
    consumed = [owner for face in removed for owner in by_face[face]]
    if Counter(consumed) != Counter(requested):
        raise ValueError('Partial suppression or an unrelated owner shares a source face.')
    return cycle, removed, tuple(sorted(consumed))


def propose_actual(mesh, vertex_ledger, owner_ledger, spec, tau):
    cycle, removed, consumed = resolve_support(mesh, vertex_ledger, owner_ledger, spec, tau)
    v, f, tags = mesh; center = spec.center(tau)
    geometry = check_actual_patch(v, f, removed, cycle, center)
    if geometry['status'] != 'PASS':
        return None, {'status': 'REFUSED', 'geometry': geometry}
    result_v = np.concatenate((v, center[None, :].astype(v.dtype)))
    result_tags = np.concatenate((tags, np.asarray([1], dtype=tags.dtype)))
    fan = np.asarray([(cycle[i], cycle[(i+1) % 4], len(v)) for i in range(4)], dtype=f.dtype)
    result_f = np.concatenate((f.copy(), fan[2:]))
    result_f[list(removed)] = fan[:2]
    return (result_v, result_f, result_tags), {
        'status': 'PROPOSED_NOT_PUBLISHED', 'geometry': geometry,
        'boundary_actual_ids': list(cycle), 'replaced_face_rows': list(removed),
        'consumed_owners': [list(x) for x in consumed], 'new_center_id': len(v),
        'new_center': center.tolist(), 'source_identity_used': True,
        'source_identity_from_coordinate_matching': False,
    }


def _pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def source_cache_digest(cache):
    """The frozen source inventory digest, computed from actual input bytes."""
    cache = Path(cache).resolve()
    files = sorted((cache/'processed_hyperpolys').glob('*.bin'))+sorted((cache/'hypervertices').glob('*.bin'))+[cache/'event_registry_p1.csv']
    if len(files) < 3 or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('Missing or symlinked source-cache inputs.')
    def stamp():
        return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    before = stamp()
    if sum(row[1] for row in before) > 256*1024*1024:
        raise ValueError('Source cache exceeds the bounded runtime support.')
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(cache).as_posix().encode())
        digest.update(hashlib.sha256(p.read_bytes()).digest())
    current = sorted((cache/'processed_hyperpolys').glob('*.bin'))+sorted((cache/'hypervertices').glob('*.bin'))+[cache/'event_registry_p1.csv']
    if current != files or before != stamp():
        raise ValueError('Source cache changed while hashing.')
    return digest.hexdigest()


class WindowRuntime:
    """Single-element, raw-only schedule transaction on an initialized mesher.

    Concurrent ordinary slicer calls are unsupported by the underlying global
    C++ state. Use separate processes, not simultaneous mesher instances.
    """
    def __init__(self, mesher):
        if getattr(mesher, '_window_n_elements', None) != 1:
            raise ValueError('Initialize exactly one scene element before the opt-in window API.')
        self.mesher = mesher
        dll = mesher._runtime_library
        signatures = {
            'slicing_discard_output': ([], None),
            'slicing_identity_enable': ([ctypes.c_bool], None),
            'slicing_identity_status': ([], ctypes.c_int),
            'slicing_identity_last_error': ([], ctypes.c_char_p),
            'slicing_identity_vertex_count': ([ctypes.c_int], ctypes.c_int),
            'slicing_identity_owner_count': ([ctypes.c_int], ctypes.c_int),
            'slicing_identity_output_vertices': ([ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_int], ctypes.c_int),
            'slicing_identity_output_owners': ([ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_int], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            func = getattr(dll, name); func.argtypes = args; func.restype = result
            setattr(self, name, func)

    def _cache_digest(self):
        return source_cache_digest(self.mesher.path)

    def _slice(self, value, mode, smooth, ledger):
        m = self.mesher
        vc, fc = np.zeros(5, np.int32), np.zeros(5, np.int32)
        self.slicing_identity_enable(bool(ledger))
        try:
            if mode == 'exact':
                t = F(value)
                if not 0 <= t.numerator < 2**63 or not 0 < t.denominator < 2**63:
                    raise ValueError('Exact query is outside the rational C ABI.')
                status = m.run_slicing_rational(t.numerator, t.denominator, _pointer(vc), _pointer(fc), smooth)
            else:
                status = m.run_slicing(float(value), _pointer(vc), _pointer(fc), smooth)
            if status != 0:
                raise RuntimeError('Ordinary slicing failed: '+str(m.slicing_last_error()))
            if vc[0] < 0 or fc[0] < 0 or (int(vc[0])*44+int(fc[0])*12) > MAX_BATCH_BYTES:
                raise ValueError('Single query exceeds the runtime memory budget.')
            ids, owners = np.empty((0, 4), np.int32), np.empty((0, 11), np.int32)
            identity = int(self.slicing_identity_status())
            if ledger and identity == 1:
                nv, nr = int(self.slicing_identity_vertex_count(0)), int(self.slicing_identity_owner_count(0))
                if nv != vc[0] or nr < 0 or nr*44 > MAX_BATCH_BYTES:
                    raise ValueError('Invalid identity ledger counts.')
                ids, owners = np.empty((nv, 4), np.int32), np.empty((nr, 11), np.int32)
                if (self.slicing_identity_output_vertices(0, _pointer(ids), ids.size) != 0 or
                        self.slicing_identity_output_owners(0, _pointer(owners), owners.size) != 0):
                    raise ValueError('Identity output failed.')
            v, f, tags = np.empty((int(vc[0]), 3), np.float64), np.empty((int(fc[0]), 3), np.int32), np.empty(int(vc[0]), np.int32)
            m.slicing_output(0, m.AF(v), _pointer(f), _pointer(tags))
            return (v, f, tags), ids, owners, identity
        finally:
            try:
                self.slicing_discard_output()
            finally:
                m.slicing_clean_up(); self.slicing_identity_enable(False)

    def run(self, values, spec, *, time_mode='exact', extra_smooth=False, enabled=True):
        """Return (meshes, report). No partial results escape this transaction.

        Bad proposals/modes return their same-entry, same-mode ordinary arrays.
        Ordinary slicer failures still raise; they are not a valid baseline.
        Endpoints/outside remain byte-identical and do not add an unused center.
        """
        values = tuple(values)
        if not values or len(values) > MAX_QUERIES or time_mode not in ('exact', 'physical'):
            raise ValueError('Expected 1..128 valid exact or physical queries.')
        if os.environ.get('BINOC_SOURCE_SPLICE_PLAN'):
            raise ValueError('Do not combine an old SSP1 intervention with a window transaction.')
        report = {'schema': 'binoc-actual-window-schedule-v1', 'status': 'PREPARING',
                  'temporal_scope': 'ATOMIC_REQUESTED_SCHEDULE_ONLY',
                  'continuous_window_admitted': False, 'published_partial_results': False,
                  'time_mode': time_mode, 'extra_smooth': bool(extra_smooth), 'cases': []}
        with _LOCK:
            cache_reason = None
            if enabled and not extra_smooth and isinstance(spec, WindowSpec):
                try:
                    if self._cache_digest() != spec.cache_digest:
                        cache_reason = 'SOURCE_CACHE_MISMATCH'
                except (ValueError, OSError, AttributeError):
                    cache_reason = 'SOURCE_CACHE_UNAVAILABLE'
            baseline, traces, allocated = [], [], 0
            for value in values:
                mesh, ids, owners, identity = self._slice(value, time_mode, bool(extra_smooth), enabled and not extra_smooth)
                allocated += sum(x.nbytes for x in (*mesh, ids, owners))*2
                if allocated > MAX_BATCH_BYTES:
                    raise ValueError('Requested schedule exceeds the runtime memory budget; nothing published.')
                baseline.append(mesh); traces.append((ids, owners, identity))
            reason = ('DISABLED' if not enabled else 'UNSUPPORTED_EXTRA_SMOOTH' if extra_smooth else
                      'INVALID_SPEC' if not isinstance(spec, WindowSpec) else cache_reason)
            if reason is None:
                try:
                    spec.validate_layout()
                except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError):
                    reason = 'INVALID_SPEC'
            if reason:
                report.update(status='BASELINE_ENTIRE_SCHEDULE', fallback_reason=reason)
                return baseline, report
            proposals = []
            # The demo's deltaT is loaded by C++; callers must supply the actual
            # scale explicitly. fading_time is NOT assumed to equal deltaT.
            delta = float(getattr(self.mesher, '_window_delta_t', float('nan')))
            if time_mode == 'physical' and (not np.isfinite(delta) or delta <= 0):
                report.update(status='BASELINE_ENTIRE_SCHEDULE', fallback_reason='MISSING_ACTUAL_DELTA_T')
                return baseline, report
            physical_bounds = tuple(float(np.longdouble(t.numerator)*np.longdouble(delta)/np.longdouble(t.denominator))
                                    for t in (spec.lower, spec.upper)) if time_mode == 'physical' else None
            failed = None
            for index, (value, mesh, (ids, owners, identity)) in enumerate(zip(values, baseline, traces)):
                case = {'index': index, 'query': str(value)}; report['cases'].append(case)
                if time_mode == 'exact':
                    tau = F(value); inside = spec.lower < tau < spec.upper
                    endpoint = tau in (spec.lower, spec.upper)
                else:
                    physical = float(value); effective = float(physical/delta)
                    tau = F.from_float(effective)
                    inside = physical_bounds[0] < physical < physical_bounds[1]
                    endpoint = physical in physical_bounds
                    case.update(physical_time_hex=physical.hex(), effective_discrete_time_hex=effective.hex())
                    if inside or endpoint:
                        tau = max(spec.lower, min(spec.upper, tau))
                    if endpoint:
                        tau = spec.lower if physical == physical_bounds[0] else spec.upper
                case['evaluation_tau'] = str(tau)
                if not inside and not endpoint:
                    case['status'] = 'BASELINE_OUTSIDE'; proposals.append(mesh); continue
                try:
                    if identity != 1:
                        raise ValueError('Actual raw identity ledger is not ready.')
                    if endpoint:
                        cycle, removed, consumed = resolve_support(mesh, ids, owners, spec, tau)
                        common = set(mesh[1][removed[0]]) & set(mesh[1][removed[1]])
                        if len(common) != 2:
                            raise ValueError('Endpoint source diagonal is not unique.')
                        a, b = (_p(mesh[0][i]) for i in sorted(common)); c = _p(spec.center(tau))
                        d, offset = _sub(b, a), _sub(c, a)
                        if any(_cross(d, offset)) or not any(d):
                            raise ValueError('Rounded endpoint anchor is not exactly on the actual diagonal.')
                        axis = next(i for i in range(3) if d[i])
                        if not 0 < offset[axis]/d[axis] < 1:
                            raise ValueError('Endpoint anchor is not strictly internal.')
                        case.update(status='BASELINE_ENDPOINT', endpoint_contract='PASS')
                        proposals.append(mesh)
                    else:
                        proposal, audit = propose_actual(mesh, ids, owners, spec, tau)
                        case.update(audit)
                        if proposal is None:
                            failed = 'ACTUAL_GEOMETRY_REFUSED'; break
                        proposals.append(proposal)
                except (ValueError, TypeError, IndexError, OverflowError, KeyError, ZeroDivisionError) as error:
                    case.update(status='REFUSED', reason=str(error)); failed = str(error); break
            if failed is None:
                try:
                    if self._cache_digest() != spec.cache_digest:
                        failed = 'SOURCE_CACHE_CHANGED_BEFORE_PUBLICATION'
                except (ValueError, OSError, AttributeError):
                    failed = 'SOURCE_CACHE_UNAVAILABLE_BEFORE_PUBLICATION'
            if failed:
                report.update(status='BASELINE_ENTIRE_SCHEDULE', fallback_reason=failed,
                              discarded_proposals=len(proposals))
                return baseline, report
            for case in report['cases']:
                if case['status'] == 'PROPOSED_NOT_PUBLISHED':
                    case['status'] = 'APPLIED_ACTUAL_QUERY'
            report.update(status='COMMITTED_REQUESTED_SCHEDULE', source_proposal_digest=spec.source_digest)
            return proposals, report
