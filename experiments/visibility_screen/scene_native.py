"""Bounded scene-parametric ordinary native reader; no build or overlay.

Only the number/order of opaque elements and hash-bound scene inputs vary.
The registered profile remains 64 frames, 960x540, 6 px, two temporal groups,
maximum discrete time four. C++ state is process-global: the frozen reader's
lock/active lifecycle is shared, never its Forest configuration or allocator.
Original cache content verification and temporary-copy removal belong to the
caller. This reader never removes a cache or modifies a mesh/production source.
"""
from __future__ import annotations

import ctypes
from fractions import Fraction as F
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

C1_LITE = Path(__file__).resolve().parents[1]/'c1_lite'
if str(C1_LITE) not in sys.path:
    sys.path.insert(0, str(C1_LITE))
import forest_native_reader as frozen

NativeReadError = frozen.NativeReadError
require = frozen.require
MAX_QUERY_ARRAY_BYTES = 2*1024**3
MAX_OPAQUE_ELEMENTS = 64

# Fixed transport ABI of the pinned v2 observer, not a padded scene.
PINNED_V2_SHIFT_ROWS = 5


def prepare_parameters(camera, effective):
    """Pack original constructor inputs with explicitly built opaque order."""
    poses = np.asarray(camera['poses'], np.float64)
    intrinsics = np.asarray(camera['intrinsics'], np.float64)
    heights, widths = np.asarray(camera['heights']), np.asarray(camera['widths'])
    times = tuple(map(float, camera['times_seconds']))
    require(len(times) == 64 and poses.shape == (64, 4, 4) and intrinsics.shape == (64, 3, 3), 'SCREEN_CAMERA_SHAPE')
    require(heights.shape == widths.shape == (64,) and np.all(heights == 540) and np.all(widths == 960), 'SCREEN_CAMERA_RESOLUTION')
    require(np.isfinite(poses).all() and np.isfinite(intrinsics).all()
            and all(math.isfinite(t) for t in times) and all(a < b for a, b in zip(times, times[1:])), 'INVALID_CAMERA_OR_TIMES')
    require(np.all(poses[:, 3, :] == [0, 0, 0, 1]) and np.all(intrinsics[:, 2, :] == [0, 0, 1]), 'CAMERA_AFFINE_OR_PINHOLE_FORM')
    mapping, mesher = camera['time_mapping'], effective['mesher_parameters']
    require(mapping['use_alignment'] is False and mapping['min_t_offset'] == 0
            and mesher['use_alignment'] is False and mesher['min_t_offset'] == 0, 'UNSUPPORTED_TIME_ALIGNMENT')
    require(mapping['temporal_group_count'] == 2 and mapping['maximum_discrete_time'] == 4, 'SCREEN_TEMPORAL_PROFILE')
    expected = {'pixels_per_cube': 6., 'pixels_per_cube_coarse': 30.,
                'pixels_per_cube_outview': 120., 'min_dist': .1, 'fading_time': 1.}
    require(all(float(mesher[k]) == value for k, value in expected.items()), 'SCREEN_MESHER_PARAMETER_PROFILE')
    require(float(mapping['effective_fading_seconds']) == 1., 'SCREEN_FADING_TIME')
    elements, all_elements = effective['opaque_elements'], effective['terrain_elements']
    require(isinstance(elements, list) and 1 <= len(elements) <= MAX_OPAQUE_ELEMENTS
            and all(isinstance(e, str) and e for e in elements) and len(set(elements)) == len(elements), 'EXPLICIT_OPAQUE_ELEMENT_ORDER_REQUIRED')
    require(isinstance(all_elements, list) and all(isinstance(e, str) and e for e in all_elements)
            and len(set(all_elements)) == len(all_elements) and set(elements) <= set(all_elements)
            and [e for e in all_elements if e in set(elements)] == elements, 'OPAQUE_ORDER_NOT_BUILT_TERRAIN_SUBSEQUENCE')
    origin = min(times)
    relative = [t-origin for t in times]
    duration = max(relative)+1e-5
    require(origin.hex() == float(frozen._fraction(mapping['origin_seconds'])).hex(), 'ORIGIN_BITS_MISMATCH')
    require(duration.hex() == float(frozen._fraction(mapping['duration_seconds'])).hex(), 'DURATION_BITS_MISMATCH')
    delta = float(frozen._fraction(mapping['delta_seconds']))
    require(math.isfinite(delta) and delta > 0, 'INVALID_EXPECTED_DELTA')
    bounds = np.asarray(effective['bounds'], np.float64)
    require(bounds.shape == (6,) and np.isfinite(bounds).all() and np.all(bounds[1::2] > bounds[::2]), 'INVALID_SCENE_BOUNDS')
    center = np.asarray([(bounds[2*k]+bounds[2*k+1])/2 for k in range(3)], np.float64)
    size = float(max(bounds[1::2]-bounds[::2])*1.1)
    require(np.isfinite(center).all() and math.isfinite(size), 'SCENE_BOUNDS_OVERFLOW')
    packed = np.empty(64*27, np.float64)
    for i in range(64):
        try:
            inverse = np.linalg.inv(poses[i])
        except np.linalg.LinAlgError as error:
            raise NativeReadError('SINGULAR_CAMERA_POSE') from error
        require(np.isfinite(inverse).all(), 'NONFINITE_CAMERA_INVERSE')
        packed[i*27:(i+1)*27] = np.concatenate((inverse[:3, :4].reshape(-1), intrinsics[i].reshape(-1),
            [heights[i], widths[i], relative[i]], poses[i][:3, 3]))
    return {'center': center, 'size': size, 'duration': duration, 'n_cameras': 64, 'cameras': packed,
        **expected, 'n_elements': len(elements), 'opaque_elements': tuple(elements), 'origin_seconds': origin,
        'expected_delta': delta, 'expected_groups': 2, 'maximum_discrete_time': 4,
        'parameter_sources': 'Explicit hash-bound effective mesher parameters and actual build opaque-element order; original camera time mapping. No Forest element-count or transparent-element inference.'}


def validate_counts(vertex_counts, face_counts, n_elements, owner_counts=None):
    require(type(n_elements) is int and 1 <= n_elements <= MAX_OPAQUE_ELEMENTS, 'INVALID_OPAQUE_ELEMENT_COUNT')
    require(len(vertex_counts) == len(face_counts) == n_elements, 'NATIVE_COUNT_ARRAY_SHAPE')
    def integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and 0 <= int(value) < 2**31
    require(all(integer(x) for x in (*vertex_counts, *face_counts)), 'INVALID_NATIVE_MESH_COUNTS')
    total = sum(int(v)*28+int(f)*12 for v, f in zip(vertex_counts, face_counts))
    if owner_counts is not None:
        require(len(owner_counts) == n_elements and all(integer(x) for x in owner_counts), 'INVALID_NATIVE_OWNER_COUNTS')
        require(all(int(v)*4 < 2**31 for v in vertex_counts) and all(int(n)*11 < 2**31 for n in owner_counts), 'IDENTITY_INT_CAPACITY_OVERFLOW')
        total += sum(int(v)*16+int(o)*44 for v, o in zip(vertex_counts, owner_counts))
    require(total <= MAX_QUERY_ARRAY_BYTES, 'NATIVE_QUERY_ARRAYS_EXCEED_2_GIB')
    return total


def validate_cache_paths(cache_copy, original_cache, build_repo):
    """Explicit originals; reject symlink/hardlink aliases before native I/O."""
    cache, original, build = map(lambda p: Path(p).resolve(), (cache_copy, original_cache, build_repo))
    require(cache != original and original not in cache.parents and cache not in original.parents, 'PRIVATE_AND_ORIGINAL_CACHE_MUST_BE_DISJOINT')
    require(cache.is_dir() and original.is_dir() and (cache/'slicing_preprocess.finish').is_file()
            and (original/'slicing_preprocess.finish').is_file(), 'PRIVATE_AND_ORIGINAL_COMPLETED_CACHE_REQUIRED')
    require(not any(p.is_symlink() for p in cache.rglob('*')), 'PRIVATE_CACHE_SYMLINK_FORBIDDEN')
    for path in cache.rglob('*'):
        target = original/path.relative_to(cache)
        if path.is_file() and target.is_file():
            require(not os.path.samefile(path, target), 'PRIVATE_CACHE_INPUT_IS_ORIGINAL_HARDLINK')
    require(build != cache and build != original and cache not in build.parents and original not in build.parents,
            'BUILD_REPOSITORY_MUST_BE_OUTSIDE_CACHE_TREES')
    manifest = frozen._document(cache/'slicing_preprocess.manifest.json')[0]
    require(manifest.get('provenance_enabled') is True and manifest.get('bpm_version') == 2
            and manifest.get('bhp_version') == 2, 'BPM2_COMPLETED_CACHE_REQUIRED')
    return cache, original, build


class NativeSceneReader:
    def __init__(self, dll, parameters, cache, original_cache, library_path, library_hash,
                 document_bindings, *, extra_smooth=False):
        require(type(extra_smooth) is bool, 'EXTRA_SMOOTH_MUST_BE_EXPLICIT_BOOL')
        self.dll, self.parameters = dll, parameters
        self.cache, self.original_cache = Path(cache), Path(original_cache)
        self.library_path, self.library_sha256 = Path(library_path), library_hash
        self.input_documents_sha256 = document_bindings
        self.n_elements = parameters['n_elements']
        self.opaque_elements = parameters['opaque_elements']
        self.delta_t, self.maximum_discrete_time = parameters['expected_delta'], 4
        self.extra_smooth = extra_smooth
        self.source_vid_encoding_version, self.source_vid_encoding = frozen._encoding(dll)
        self._thread, self._closed = threading.get_ident(), False
        self._saved_environment, self.initialization = {}, {}

    def _discard(self):
        try:
            self.dll.slicing_discard_output()
        finally:
            try:
                self.dll.slicing_clean_up()
            finally:
                self.dll.slicing_identity_enable(False)

    def slice_query(self, value, mode='exact', ledger=True):
        require(not self._closed and threading.get_ident() == self._thread, 'NATIVE_READER_CLOSED_OR_WRONG_THREAD')
        require(mode in ('exact', 'physical') and type(ledger) is bool, 'UNSUPPORTED_QUERY_MODE_OR_LEDGER')
        if mode == 'exact':
            tau = F(value)
            require(0 <= tau <= 4 and 0 <= tau.numerator < 2**63 and 0 < tau.denominator < 2**63, 'EXACT_TIME_OUTSIDE_ABI_OR_PROFILE')
            physical = float(np.longdouble(tau.numerator)*np.longdouble(self.delta_t)/np.longdouble(tau.denominator))
            query = {'mode': mode, 'value': str(tau), 'exact_time': {'numerator': tau.numerator, 'denominator': tau.denominator},
                     'physical_time_hex': physical.hex()}
        else:
            physical = float(value)
            require(math.isfinite(physical) and 0 <= physical <= self.parameters['duration'], 'PHYSICAL_TIME_OUTSIDE_PROFILE')
            query = {'mode': mode, 'value': physical.hex(), 'exact_time': None, 'physical_time_hex': physical.hex(),
                     'effective_discrete_time_hex': float(physical/self.delta_t).hex()}
        frozen._check_rss()
        started, cpu = time.monotonic(), time.process_time()
        n = self.n_elements
        require(not ledger or n <= PINNED_V2_SHIFT_ROWS,
                'PINNED_OBSERVER_ELEMENT_CAPACITY_UNSUPPORTED')
        vc, fc = np.zeros(n, np.int32), np.zeros(n, np.int32)
        with frozen._LOCK:
            require(frozen._ACTIVE in (None, self), 'OTHER_NATIVE_READER_IS_ACTIVE')
            self.dll.slicing_identity_enable(ledger)
            try:
                if mode == 'exact':
                    status = int(self.dll.run_slicing_rational(tau.numerator, tau.denominator, frozen._int_pointer(vc), frozen._int_pointer(fc), self.extra_smooth))
                else:
                    status = int(self.dll.run_slicing(physical, frozen._int_pointer(vc), frozen._int_pointer(fc), self.extra_smooth))
                require(status == 0, 'ORDINARY_NATIVE_SLICING_FAILED: '+frozen._error(self.dll.slicing_last_error))
                frozen._check_rss()
                validate_counts(vc, fc, n)
                identity, error = int(self.dll.slicing_identity_status()), frozen._error(self.dll.slicing_identity_last_error)
                require(identity in (0, 1, 2, 3), 'INVALID_IDENTITY_STATUS')
                ids = [np.empty((0, 4), np.int32) for _ in range(n)]
                owners = [np.empty((0, 11), np.int32) for _ in range(n)]
                shifts = None
                if ledger and identity == 1:
                    ledger_v = [int(self.dll.slicing_identity_vertex_count(e)) for e in range(n)]
                    owner_counts = [int(self.dll.slicing_identity_owner_count(e)) for e in range(n)]
                    require(ledger_v == [int(v) for v in vc], 'IDENTITY_VERTEX_COUNT_NOT_ACTUAL_VERTEX_COUNT')
                    array_bytes = validate_counts(vc, fc, n, owner_counts)
                    # Every dynamic-element sidecar precedes ANY destructive mesh output.
                    for e in range(n):
                        ids[e], owners[e] = np.empty((ledger_v[e], 4), np.int32), np.empty((owner_counts[e], 11), np.int32)
                        require(self.dll.slicing_identity_output_vertices(e, frozen._int_pointer(ids[e]), ids[e].size) == 0, 'IDENTITY_VERTEX_OUTPUT_FAILED')
                        require(self.dll.slicing_identity_output_owners(e, frozen._int_pointer(owners[e]), owners[e].size) == 0, 'IDENTITY_OWNER_OUTPUT_FAILED')
                    if self.source_vid_encoding == 'ORIGINAL_EFFECTIVE_SOURCE_VID':
                        # The frozen exporter writes five rows, even for n=1.
                        # Only the first actual n rows are exposed as scene data.
                        array_bytes += n*4*np.dtype(np.int32).itemsize
                        transport_bytes = PINNED_V2_SHIFT_ROWS*4*np.dtype(np.int32).itemsize
                        require(array_bytes+transport_bytes <= MAX_QUERY_ARRAY_BYTES, 'NATIVE_QUERY_ARRAYS_EXCEED_2_GIB')
                        transport = np.empty((PINNED_V2_SHIFT_ROWS, 4), np.int32)
                        require(self.dll.slicing_identity_output_source_vid_shifts(frozen._int_pointer(transport), transport.size) == 0,
                                'ORIGINAL_SOURCE_VID_SHIFT_OUTPUT_FAILED: '+frozen._error(self.dll.slicing_identity_last_error))
                        require(not np.any(transport[n:]), 'UNUSED_SOURCE_VID_SHIFT_SLOTS_NONZERO')
                        shifts = transport[:n].copy()
                        del transport
                        shifts.setflags(write=False)
                        require(array_bytes <= MAX_QUERY_ARRAY_BYTES, 'NATIVE_QUERY_ARRAYS_EXCEED_2_GIB')
                else:
                    array_bytes = validate_counts(vc, fc, n)
                meshes = []
                for e in range(n):
                    v, f, tags = np.empty((int(vc[e]), 3), np.float64), np.empty((int(fc[e]), 3), np.int32), np.empty(int(vc[e]), np.int32)
                    self.dll.slicing_output(e, frozen._double_pointer(v), frozen._int_pointer(f), frozen._int_pointer(tags))
                    require(np.isfinite(v).all(), 'NONFINITE_NATIVE_VERTEX')
                    require(not len(f) or int(f.min()) >= 0 and int(f.max()) < len(v), 'NATIVE_FACE_INDEX_OUT_OF_RANGE')
                    require(np.all((tags == 0) | (tags == 1)), 'NATIVE_INVIEW_TAG_NOT_BOOLEAN')
                    for start in range(0, len(v), 65536):
                        block = v[start:start+65536]
                        require(np.array_equal(block.astype(np.float32).astype(np.float64), block), 'NATIVE_COORDINATES_NOT_EXACT_BINARY32_VALUES')
                    if ledger and identity == 1:
                        o = owners[e]
                        require(np.all(o[:, 0] == e), 'IDENTITY_OWNER_ELEMENT_DISAGREEMENT')
                        require(not len(o) or np.all(o[:, :8] >= 0) and int(o[:, 7].max()) < len(f), 'IDENTITY_OWNER_PROVENANCE_OR_FACE_RANGE')
                        for start in range(0, len(o), 65536):
                            block = o[start:start+65536]
                            require(np.array_equal(block[:, 8:11], f[block[:, 7]]), 'IDENTITY_OWNER_ORIENTED_FACE_DISAGREEMENT')
                    for array in (v, f, tags, ids[e], owners[e]):
                        array.setflags(write=False)
                    meshes.append((v, f, tags))
                frozen._check_rss()
                return {'meshes': tuple(meshes), 'vertex_ledgers': tuple(ids), 'owner_ledgers': tuple(owners),
                    'identity_status': identity, 'identity_error': error, 'ledger_requested': ledger,
                    'source_vid_encoding_version': self.source_vid_encoding_version,
                    'source_vid_encoding': self.source_vid_encoding, 'source_vid_shifts': shifts,
                    'source_vid_shift_transport_capacity_ints': PINNED_V2_SHIFT_ROWS*4 if shifts is not None else None,
                    'unused_source_vid_shift_slots_verified_zero': bool(shifts is not None),
                    'identity_contract_verified': bool(ledger and identity == 1 and shifts is not None),
                    'counts': [{'element': e, 'element_name': self.opaque_elements[e], 'vertices': int(vc[e]), 'faces': int(fc[e]),
                                'identity_vertices': len(ids[e]), 'raw_owners': len(owners[e])} for e in range(n)],
                    'cost': {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
                             'array_bytes': array_bytes, 'peak_rss_bytes': frozen._peak_rss()},
                    'query': query, 'delta_t': self.delta_t, 'delta_t_hex': self.delta_t.hex(),
                    'n_elements': n, 'opaque_element_names': list(self.opaque_elements), 'maximum_discrete_time': 4,
                    'cache_root': str(self.cache), 'original_cache_root': str(self.original_cache),
                    'library_sha256': self.library_sha256, 'input_documents_sha256': self.input_documents_sha256,
                    'baseline_only': True, 'extra_smooth': self.extra_smooth, 'arrays_immutable': True,
                    'raw_method_admission_claimed': False}
            finally:
                self._discard()

    def close(self):
        if self._closed:
            return
        require(threading.get_ident() == self._thread, 'CLOSE_NATIVE_READER_ON_CREATING_THREAD')
        with frozen._LOCK:
            try:
                self._discard()
            finally:
                self._closed = True
                for key, old in self._saved_environment.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old
                if frozen._ACTIVE is self:
                    frozen._ACTIVE = None

    cleanup = close


def initialize_scene(build_repo, cache_copy, camera_document, effective_document, *,
                     original_cache, expected_so_sha256, extra_smooth=False):
    """Load verified ordinary native entrypoints only; caller owns both caches."""
    with frozen._LOCK:
        require(frozen._ACTIVE is None, 'ONE_NATIVE_READER_PER_PROCESS')
        require(type(extra_smooth) is bool, 'EXTRA_SMOOTH_MUST_BE_EXPLICIT_BOOL')
        cache, original, build = validate_cache_paths(cache_copy, original_cache, build_repo)
        require(not any(os.environ.get(k) for k in os.environ if k.startswith('BINOC_SOURCE_SPLICE')), 'NO_SSP1_ENVIRONMENT_ALLOWED')
        camera, camera_binding = frozen._document(camera_document)
        effective, effective_binding = frozen._document(effective_document)
        parameters = prepare_parameters(camera, effective)
        library = build/'binocmesher/lib/core.so'
        require(isinstance(expected_so_sha256, str) and len(expected_so_sha256) == 64
                and all(c in '0123456789abcdef' for c in expected_so_sha256)
                and frozen._hash(library) == expected_so_sha256, 'NATIVE_LIBRARY_SHA256_MISMATCH')
        dll = ctypes.CDLL(str(library))
        frozen._bind(dll)
        result = NativeSceneReader(dll, parameters, cache, original, library, expected_so_sha256,
            {'camera': camera_binding, 'effective': effective_binding}, extra_smooth=extra_smooth)
        for key in ('BINOC_PROVENANCE_V2', 'BINOC_EVENT_MODE'):
            result._saved_environment[key] = os.environ.get(key)
            os.environ[key] = '1'
        frozen._ACTIVE = result
        try:
            result._discard(); frozen._check_rss()
            groups = int(dll.load_parameters(frozen._double_pointer(parameters['center']), parameters['size'], parameters['duration'],
                parameters['n_cameras'], frozen._double_pointer(parameters['cameras']), parameters['fading_time'],
                parameters['pixels_per_cube'], parameters['pixels_per_cube_coarse'], parameters['pixels_per_cube_outview'],
                parameters['min_dist'], parameters['n_elements'], str(cache).encode('utf-8')))
            delta = ctypes.c_double.in_dll(dll, '_ZN6params6deltaTE').value
            elements = ctypes.c_int32.in_dll(dll, '_ZN6params10n_elementsE').value
            level = ctypes.c_int32.in_dll(dll, '_ZN6params6max_tLE').value
            require(level == 1 and groups == parameters['expected_groups'] == 2, 'ACTUAL_NATIVE_TEMPORAL_GROUP_MISMATCH')
            require(elements == parameters['n_elements'], 'ACTUAL_NATIVE_ELEMENT_COUNT_MISMATCH')
            require(delta.hex() == parameters['expected_delta'].hex(), 'ACTUAL_NATIVE_DELTA_BITS_MISMATCH')
            require(frozen._hash(library) == expected_so_sha256, 'NATIVE_LIBRARY_CHANGED_DURING_INITIALIZATION')
            result.delta_t = delta
            result.initialization = {'actual_delta_t_hex': delta.hex(), 'actual_max_tL': level,
                'actual_elements': elements, 'opaque_element_names': list(parameters['opaque_elements']), 'time_groups': groups,
                'source_vid_encoding_version': result.source_vid_encoding_version, 'source_vid_encoding': result.source_vid_encoding,
                'library_sha256': expected_so_sha256, 'duration_hex': parameters['duration'].hex(),
                'origin_seconds_hex': parameters['origin_seconds'].hex(), 'private_cache_root': str(cache),
                'original_cache_root': str(original), 'build_repository': str(build), 'extra_smooth': extra_smooth,
                'frozen_helper_source_sha256': frozen._hash(Path(frozen.__file__)),
                'native_entrypoints_called': ['slicing_discard_output', 'slicing_clean_up', 'slicing_identity_enable', 'load_parameters'],
                'tree_or_preprocess_or_initial_slice_run': False, 'parameter_sources': parameters['parameter_sources']}
            return result
        except BaseException:
            result.close()
            raise
