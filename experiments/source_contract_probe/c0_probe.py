"""One-shot native E2 C0 probe; imported only by the supervised stage worker.

No binocmesher module is imported until the frozen isolated build is bound.
The production and ordinary frontends share the checker, center and executor.
Timing runs contain no tracing; logical work is measured in a separate pass.
"""

from fractions import Fraction as F
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = Path('/home/warpwang/binoc-runs/e2-runtime-loop-20260906/build')
CACHE = Path('/home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache')
SOURCE = ROOT/'experiments/c1_lite/artifacts/ab_20260906/window_audit/event-02-88ade47aa4fa/source.json'
SOURCE_SHA = '6b7f08fb7689ef8e62002c787e301a321fb343cda16d1ce37c54021b58c92dc2'
CORE_SHA = 'f4263a2f47ba5283175a921e49b8867998bdd8124aac810793b34242ec43a3c9'
CACHE_SHA = '329c7681d849938a86e1f8812a1d05e3018070a06fb4bd070b81fd6ffb6cd3f6'
DELTA_HEX = '0x1.eaabfa360338dp-6'
TSIZE_HEX = '0x1.eaabfa360338dp-1'
QUERIES = (F(103, 5), F(104, 5), F(102, 5))
PROGRESS = {'phase': 'IMPORTED_NOT_STARTED', 'queries': []}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest(path):
    _require(path.is_dir() and not path.is_symlink(), 'Missing or symlinked input directory: '+str(path))
    entries = sorted(path.rglob('*'))
    _require(not any(p.is_symlink() for p in entries), 'Symlink found in bounded cache.')
    return {p.relative_to(path).as_posix(): _sha(p) for p in entries if p.is_file()}


def _array_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mesh_hash(mesh):
    return hashlib.sha256(''.join(_array_hash(a) for a in mesh).encode('ascii')).hexdigest()


def _snapshot_hash(mesh, ids, owners):
    return [_mesh_hash(mesh), _array_hash(ids), _array_hash(owners)]


def _line_for(function, snippet):
    source, start = inspect.getsourcelines(function)
    matches = [start+i for i, line in enumerate(source) if snippet in line]
    _require(len(matches) == 1, 'Tracing source binding not unique: '+snippet)
    return matches[0]


def _traced(function, arguments, lines=(), calls=()):
    """Count only declared code/line events in an untimed execution."""
    _require(sys.gettrace() is None, 'An unrelated tracer is active.')
    line_map = {(owner.__code__, _line_for(owner, snippet)): (name, amount)
                for owner, snippet, name, amount in lines}
    call_map = {owner.__code__: name for owner, name in calls}
    counts = {name: 0 for _owner, _snippet, name, _amount in lines}
    counts.update({name: 0 for _owner, name in calls})
    tracked_codes = {code for code, _line in line_map} | set(call_map)

    def tracer(frame, event, arg):
        code = frame.f_code
        if event == 'call':
            if code in call_map:
                key = call_map[code]
                counts[key] += 1
            return tracer if code in tracked_codes else None
        if event == 'line':
            item = line_map.get((code, frame.f_lineno))
            if item is not None:
                key, amount = item
                counts[key] += amount
        return tracer

    sys.settrace(tracer)
    try:
        result = function(*arguments)
    finally:
        sys.settrace(None)
    return result, counts


def _production_counts(runtime_module, arguments):
    resolve = runtime_module.resolve_support
    result, measured = _traced(resolve, arguments, lines=(
        (resolve, 'mapping = defaultdict(list)', 'source_indices_built', 1),
        (resolve, 'mapping[source_key(row)].append(i)', 'vertex_records_visited', 1),
        (resolve, 'by_face, by_owner = defaultdict(list), {}', 'owner_indices_built', 2),
        (resolve, 'owner = tuple(int(x) for x in row[:7]); face_id = int(row[7])', 'owner_records_visited', 1),
        (resolve, 'by_owner[owner] = face_id; by_face[face_id].append(owner)', 'owner_face_integrity_checks_passed', 1),
    ), calls=((runtime_module.source_key, 'source_key_function_calls'),))
    mesh, _ids, owners, spec, tau = arguments
    requested, _faces = spec.cell(tau)
    _require(measured['vertex_records_visited'] == len(mesh[0]) and
             measured['owner_records_visited'] == len(owners) and
             measured['owner_face_integrity_checks_passed'] == len(owners),
             'Production resolver tracing did not count its frozen loop bodies exactly.')
    return result, {
        'measured_untimed_trace': measured,
        'derived_after_success': {
            'face_rows_visited': len(owners)+len(result[1]),
            'indices_built': measured['source_indices_built']+measured['owner_indices_built'],
            'indices_reused': 0,
            'requested_owner_count': len(requested),
            'complete_face_owner_coverage_size': len(mesh[1]),
        },
        'derivation': 'On successful frozen resolve_support, each owner reads one actual face for integrity and each retired row is read once for source-triangle matching. No timing sample uses this tracer.',
    }


def _geometry_counts(geometry_module, arguments):
    checker = geometry_module.check_actual_patch
    report, measured = _traced(checker, arguments, lines=(
        (geometry_module.interface_check, 'if index in removed:', 'interface_face_slots_examined', 1),
        (checker, 'coords = v[row]', 'retained_AABB_relations_tested', 1),
    ), calls=((geometry_module._contact, 'exact_fan_retained_contact_calls'),))
    if report['status'] == 'PASS':
        _require(measured['interface_face_slots_examined'] == len(arguments[1]),
                 'Shared interface trace differs from full actual face scan.')
        _require(measured['retained_AABB_relations_tested'] == report['retained_faces_checked'],
                 'Shared checker trace disagrees with retained certificate counts.')
        exact = report['retained_faces_checked']-report['certificate_counts'].get('STRICT_ACTUAL_AABB', 0)
        _require(measured['exact_fan_retained_contact_calls'] == exact,
                 'Measured exact-contact calls disagree with successful checker certificates.')
    measured.update(event_pairs_enumerated=0, certificate_relations_inherited=0)
    return report, {
        'measured_untimed_trace': measured,
        'measured_checker_report': {
            'retained_faces_checked': report['retained_faces_checked'],
            'certificate_counts': report['certificate_counts'],
        },
        'scope': '_contact counts exact fan-versus-retained calls, not individual determinant predicates or four separate triangle-pair calls. One edit per query; no event graph or certificate reuse.',
    }


def _endpoint(mesh, removed, center):
    """The frozen runtime diagonal/strict-interior anchor policy, shared by arms."""
    common = set(mesh[1][removed[0]]) & set(mesh[1][removed[1]])
    _require(len(common) == 2, 'Endpoint source diagonal is not unique.')
    points = [tuple(F.from_float(float(x)) for x in mesh[0][i]) for i in sorted(common)]
    anchor = tuple(F.from_float(float(x)) for x in center)
    direction = tuple(b-a for a, b in zip(*points))
    offset = tuple(c-a for a, c in zip(points[0], anchor))
    cross = (direction[1]*offset[2]-direction[2]*offset[1],
             direction[2]*offset[0]-direction[0]*offset[2],
             direction[0]*offset[1]-direction[1]*offset[0])
    _require(any(direction) and not any(cross), 'Endpoint anchor is not on the actual nonzero source diagonal.')
    axis = next(i for i in range(3) if direction[i])
    _require(0 < offset[axis]/direction[axis] < 1, 'Endpoint anchor is not strictly internal.')
    return {'status': 'PASS_ENDPOINT_DIAGONAL_ANCHOR', 'mesh_edit': False,
            'retained_faces_checked': 0, 'certificate_counts': {},
            'geometry_scope': 'Frozen endpoint source-diagonal contract; no replacement fan.'}


def _ordinary(mesher, runtime, tau, np, as_int):
    """Direct ordinary rational C ABI call with identity collection disabled."""
    runtime.slicing_identity_enable(False)
    vertices_count, faces_count = np.zeros(5, np.int32), np.zeros(5, np.int32)
    try:
        status = mesher.run_slicing_rational(tau.numerator, tau.denominator,
                                            as_int(vertices_count), as_int(faces_count), False)
        _require(status == 0, 'Ordinary rational native slicing failed: '+str(mesher.slicing_last_error()))
        nv, nf = int(vertices_count[0]), int(faces_count[0])
        _require(nv >= 0 and nf >= 0 and nv*28+nf*12 < 64*1024**2,
                 'Ordinary E2 output exceeds bounded array allocation.')
        vertices = np.empty((nv, 3), np.float64)
        faces, tags = np.empty((nf, 3), np.int32), np.empty(nv, np.int32)
        mesher.slicing_output(0, mesher.AF(vertices), as_int(faces), as_int(tags))
        return vertices, faces, tags
    finally:
        runtime.slicing_discard_output()
        mesher.slicing_clean_up()
        runtime.slicing_identity_enable(False)


def _execute_arm(name, mesh, ids, owners, spec, tau, runtime_module,
                 geometry_module, plain, *, count_work):
    arguments = (mesh, ids, owners, spec, tau)
    work = {}
    start = time.perf_counter_ns()
    if name == 'production':
        if count_work:
            resolved, work['resolver'] = _production_counts(runtime_module, arguments)
        else:
            resolved = runtime_module.resolve_support(*arguments)
    else:
        counters = {} if count_work else None
        resolved = plain.plain_resolve(*arguments, counters=counters)
        if count_work:
            work['resolver'] = {'measured_logical_counters': counters}
    resolved_at = time.perf_counter_ns()
    center = spec.center(tau)
    center_at = time.perf_counter_ns()
    cycle, removed, consumed = resolved
    endpoint = tau in (spec.lower, spec.upper)
    if endpoint:
        certificate = _endpoint(mesh, removed, center)
        if count_work:
            work['geometry'] = {'scope': 'Endpoint diagonal policy only; no fan/retained scan.',
                                'retained_faces_checked': 0, 'exact_fan_retained_contact_calls': 0}
    else:
        arguments = (*mesh[:2], removed, cycle, center)
        if count_work:
            certificate, work['geometry'] = _geometry_counts(geometry_module, arguments)
        else:
            certificate = geometry_module.check_actual_patch(*arguments)
        _require(certificate['status'] == 'PASS', name+' shared geometry refused at '+str(tau)+': '+str(certificate))
    geometry_at = time.perf_counter_ns()
    assembly_counts = {} if count_work else None
    output = mesh if endpoint else plain.assemble(mesh, cycle, removed, center, counters=assembly_counts)
    assembled_at = time.perf_counter_ns()
    if count_work:
        work['assembly'] = {'measured_logical_counters': assembly_counts or {'array_bytes_copied': 0},
                            'scope': plain.WORK_COUNTER_DEFINITIONS['array_bytes_copied']}
    elapsed = None if count_work else {
        'resolver_seconds': (resolved_at-start)/1e9,
        'center_seconds': (center_at-resolved_at)/1e9,
        'geometry_seconds': (geometry_at-center_at)/1e9,
        'assembly_seconds': (assembled_at-geometry_at)/1e9,
        'total_arm_seconds': (assembled_at-start)/1e9,
    }
    return resolved, output, certificate, work, elapsed


def run():
    PROGRESS.update(phase='BINDING_INPUTS', queries=[])
    sys.dont_write_bytecode = True
    _require(not any(name == 'binocmesher' or name.startswith('binocmesher.') for name in sys.modules),
             'binocmesher imported before selecting the frozen isolated build.')
    _require(_sha(SOURCE) == SOURCE_SHA and _sha(BUILD/'binocmesher/lib/core.so') == CORE_SHA,
             'Frozen E2 source or native core binding differs.')
    python_bindings = {}
    for relative in ('__init__.py', 'core.py', 'window_runtime.py', 'window_geometry_runtime.py',
                     'utils/__init__.py', 'utils/interface.py', 'utils/timer.py'):
        expected = ROOT/'binocmesher'/relative
        actual = BUILD/'binocmesher'/relative
        _require(expected.read_bytes() == actual.read_bytes(), 'Isolated Python runtime differs: '+relative)
        python_bindings[relative] = _sha(actual)
    _require(not any(os.environ.get(key) for key in
                     ('BINOC_SOURCE_SPLICE_PLAN', 'BINOC_SOURCE_SPLICE_AUDIT', 'BINOC_SOURCE_SPLICE_TRACE')),
             'Unrelated native intervention environment is active.')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        _require(os.environ.get(key) == '1', 'Supervised C0 requires '+key+'=1 before import.')
    before = _manifest(CACHE)
    cache_bytes = sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    _require(cache_bytes <= 10*1024**2, 'Frozen E2 cache exceeds 10 MiB.')
    source = json.loads(SOURCE.read_bytes())
    # Build first, then helper paths. Helpers do not import the package eagerly.
    sys.path[:0] = [str(BUILD), str(ROOT/'experiments/source_splice'), str(ROOT/'experiments/c1_lite')]
    import numpy as np
    from runtime_common import initialize_mesher
    from e2_runtime_validation import validate_e2_runtime
    from binocmesher import window_runtime as runtime_module
    from binocmesher import window_geometry_runtime as geometry_module
    from binocmesher.utils.interface import AsInt
    from experiments.source_contract_probe import plain_reference as plain

    _require(Path(runtime_module.__file__).resolve() == (BUILD/'binocmesher/window_runtime.py').resolve(),
             'The imported resolver is not the frozen isolated build.')
    spec = runtime_module.WindowSpec.from_source_contract(source)
    spec.validate_layout()
    _require(spec.cache_digest == CACHE_SHA and runtime_module.source_cache_digest(CACHE) == CACHE_SHA,
             'Frozen source-cache inventory binding differs.')
    _require(spec.lower == F(102, 5) and spec.root == F(104, 5), 'Frozen E2 query roles differ.')
    scratch = (HERE/'.scratch').resolve()
    _require(scratch.parent == HERE.resolve() and not scratch.is_symlink(), 'Invalid scratch directory.')
    scratch.mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix='c0-e2-', dir=scratch)
    temporary_path = Path(temporary.name).resolve()
    _require(temporary_path.parent == scratch and temporary_path.name.startswith('c0-e2-'),
             'Temporary cache target escaped the owned scratch parent.')
    copied = temporary_path/'cache'
    records = PROGRESS['queries']
    report = {
        'schema': 'source-contract-c0-native-e2-v1', 'status': 'RUNNING',
        'source_sha256': SOURCE_SHA, 'core_sha256': CORE_SHA, 'cache_inventory_sha256': CACHE_SHA,
        'python_byte_bindings': python_bindings, 'cache_copy_bytes': cache_bytes,
        'query_order': [str(tau) for tau in QUERIES], 'queries': records,
        'timing_repetitions_per_arm_per_query': 5,
        'timing_scope': 'Warm immutable per-query snapshots, fresh resolver indices each call; alternating arm order. Native slicing, hashes, validation, tracing and serialization excluded from stage timings.',
        'work_scope': 'One separate counted pass per arm per query. Production line/call tracing and ordinary explicit counters are separate from timing. Derived counts are named separately.',
        'shared_policy': 'Frozen source-derived center, exact actual interface/retained checker, same array fan executor; endpoint validates source diagonal and preserves baseline.',
        'not_claimed': ['Forest batch-path cost or index reuse', 'Scaling or performance superiority',
                        'Certificate reuse optimization', 'Continuous-time safety', 'New scene or WMTK comparison'],
    }
    try:
        shutil.copytree(CACHE, copied)
        _require(_manifest(copied) == before, 'Disposable cache copy differs before initialization.')
        PROGRESS.update(phase='INITIALIZING_NATIVE', temporary_cache=str(copied))
        started = time.perf_counter()
        mesher = initialize_mesher(BUILD, copied)
        report['initialization_seconds'] = time.perf_counter()-started
        _require(Path(mesher._runtime_library._name).resolve() == (BUILD/'binocmesher/lib/core.so').resolve(),
                 'Initialized mesher loaded a different core library.')
        delta = float(mesher._window_delta_t)
        _require(delta.hex() == DELTA_HEX and float(mesher.tsize).hex() == TSIZE_HEX and
                 delta == float(mesher.tsize)/(2*16), 'Initialized E2 delta/tsize binding differs.')
        report.update(actual_delta_t_hex=delta.hex(), actual_tsize_hex=float(mesher.tsize).hex())
        runtime = runtime_module.WindowRuntime(mesher)
        PROGRESS['phase'] = 'NATIVE_QUERIES'
        for query_index, tau in enumerate(QUERIES):
            row = {'query': str(tau), 'status': 'PREPARING', 'counted_arms': {}, 'timing_samples': []}
            records.append(row)
            native_started = time.perf_counter()
            baseline = _ordinary(mesher, runtime, tau, np, AsInt)
            row['ordinary_slice_seconds'] = time.perf_counter()-native_started
            native_started = time.perf_counter()
            observed, ids, owners, identity_status = runtime._slice(tau, 'exact', False, True)
            row['ledger_slice_seconds'] = time.perf_counter()-native_started
            _require(identity_status == 1 and _mesh_hash(observed) == _mesh_hash(baseline),
                     'Native identity ledger failed or changed ordinary baseline at '+str(tau))
            row.update(status='NATIVE_BASELINE_VALIDATED', baseline_hash=_mesh_hash(baseline),
                       baseline_vertices=len(baseline[0]), baseline_faces=len(baseline[1]),
                       owner_rows=len(owners), vertex_ledger_hash=_array_hash(ids),
                       owner_ledger_hash=_array_hash(owners), instrumented_baseline_identical=True)
            del observed
            for array in (*baseline, ids, owners):
                array.flags.writeable = False
            snapshot = _snapshot_hash(baseline, ids, owners)
            counted = {}
            for name in ('production', 'plain'):
                resolved, output, certificate, work, _elapsed = _execute_arm(
                    name, baseline, ids, owners, spec, tau, runtime_module, geometry_module, plain, count_work=True)
                output_hash = _mesh_hash(output)
                if tau in (spec.lower, spec.upper):
                    _require(output_hash == row['baseline_hash'], 'Endpoint output changed baseline bytes.')
                    validation = {'pass': True, 'status': 'PASS_ENDPOINT_BASELINE_BYTES'}
                else:
                    requested, _source_faces = spec.cell(tau)
                    validation = validate_e2_runtime(baseline, output, ids, owners, spec.cycle, requested,
                                                     consumed_owners=resolved[2], expected_center=spec.center(tau))
                    _require(validation['pass'], name+' independent E2 array validator refused: '+str(validation))
                counted[name] = (resolved, output_hash, certificate)
                row['counted_arms'][name] = {'resolution': {'cycle': list(resolved[0]),
                    'removed': list(resolved[1]), 'consumed': [list(owner) for owner in resolved[2]]},
                    'output_hash': output_hash, 'geometry': certificate, 'work': work,
                    'independent_array_validation': validation}
                row['status'] = 'COUNTED_'+name.upper()+'_VALIDATED'
            _require(counted['production'] == counted['plain'], 'C0 arm semantic/output/checker mismatch at '+str(tau))
            row.update(status='COUNTED_ARMS_EQUAL', semantic_equal=True)
            for repeat in range(5):
                order = ('production', 'plain') if (repeat+query_index) % 2 == 0 else ('plain', 'production')
                pair = {'repeat': repeat, 'order': list(order), 'arms': {}}
                row['timing_samples'].append(pair)
                for name in order:
                    resolved, output, certificate, _work, elapsed = _execute_arm(
                        name, baseline, ids, owners, spec, tau, runtime_module, geometry_module, plain, count_work=False)
                    signature = (resolved, _mesh_hash(output), certificate)
                    _require(signature == counted[name], 'Timed C0 arm changed its counted semantics at '+str(tau))
                    pair['arms'][name] = elapsed
                row['timing_pairs_validated'] = repeat+1
            _require(_snapshot_hash(baseline, ids, owners) == snapshot, 'C0 shared input snapshot mutated.')
            row['timing_summary_seconds'] = {}
            for name in ('production', 'plain'):
                keys = row['timing_samples'][0]['arms'][name]
                row['timing_summary_seconds'][name] = {
                    key: {'median': statistics.median(sample['arms'][name][key] for sample in row['timing_samples']),
                          'minimum': min(sample['arms'][name][key] for sample in row['timing_samples']),
                          'maximum': max(sample['arms'][name][key] for sample in row['timing_samples'])}
                    for key in keys}
            row.update(status='PASS_EQUAL_ENDPOINT' if tau in (spec.lower, spec.upper) else 'PASS_EQUAL_ACTIVE',
                       inputs_unchanged=True)
        report['table'] = [{'query': row['query'], 'status': row['status'],
                            'production_median_seconds': row['timing_summary_seconds']['production'],
                            'plain_median_seconds': row['timing_summary_seconds']['plain']} for row in records]
        report['status'] = 'PASS_THREE_NATIVE_QUERIES_SAME_BACKEND'
        PROGRESS['phase'] = 'VERIFYING_FINAL_BINDINGS'
    except Exception as error:
        PROGRESS.update(phase='STOP', reason=type(error).__name__+': '+str(error))
        raise RuntimeError('C0 stopped without retry: '+str(error)) from error
    finally:
        # Always verify originals and remove only this invocation's owned copy.
        binding_failure = None
        try:
            original_same = _manifest(CACHE) == before
            report['original_cache_unchanged'] = original_same
            if not original_same:
                binding_failure = 'Original cache changed during C0.'
            if copied.exists():
                final_copy = _manifest(copied)
                changes = sorted(key for key in set(before) | set(final_copy)
                                 if before.get(key) != final_copy.get(key))
                report['temporary_cache_changed_files'] = changes
                if set(changes)-{'log.txt'}:
                    binding_failure = 'Disposable cache changed outside expected log.txt: '+str(changes)
                if runtime_module.source_cache_digest(copied) != CACHE_SHA:
                    binding_failure = 'Disposable source inventory changed.'
            if _sha(SOURCE) != SOURCE_SHA or _sha(BUILD/'binocmesher/lib/core.so') != CORE_SHA:
                binding_failure = 'Frozen source/core binding changed during C0.'
            for relative, digest in python_bindings.items():
                if _sha(BUILD/'binocmesher'/relative) != digest or _sha(ROOT/'binocmesher'/relative) != digest:
                    binding_failure = 'Python runtime changed during C0: '+relative
        finally:
            _require(temporary_path.parent == scratch and temporary_path != scratch and
                     temporary_path.name.startswith('c0-e2-'), 'Refusing cleanup of unowned temporary path.')
            temporary.cleanup()
            report['temporary_cache_removed'] = not temporary_path.exists()
            PROGRESS.update(temporary_cache_removed=report['temporary_cache_removed'],
                            original_cache_unchanged=report.get('original_cache_unchanged'))
        _require(report['temporary_cache_removed'], 'Temporary E2 cache cleanup failed.')
        _require(binding_failure is None, binding_failure or '')
    PROGRESS['phase'] = 'COMPLETE'
    return report
