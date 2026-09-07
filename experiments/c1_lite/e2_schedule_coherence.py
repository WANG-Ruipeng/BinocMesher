"""Six-query E2 structural manifest and RAM-only mesh preparation.

No files are written. The caller supplies and owns a disposable cache copy.
The certificate is about this declared schedule, never all real window times.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN_SOURCE_SHA = '6b7f08fb7689ef8e62002c787e301a321fb343cda16d1ce37c54021b58c92dc2'
FROZEN_CORE_SHA = 'f4263a2f47ba5283175a921e49b8867998bdd8124aac810793b34242ec43a3c9'
FROZEN_DELTA_HEX = '0x1.eaabfa360338dp-6'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fraction(value):
    return F(int(value['numerator']), int(value['denominator'])) if isinstance(value, dict) else F(value)


def fj(value):
    value = F(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def array_hash(value):
    return hashlib.sha256(str(value.shape).encode()+value.dtype.str.encode()+value.tobytes()).hexdigest()


def mesh_hash(mesh):
    return digest([array_hash(value) for value in mesh])


def same_mesh(first, second):
    return all(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
               for a, b in zip(first, second))


def oriented(face):
    face = tuple(face)
    return min(face[i:]+face[:i] for i in range(3))


def audit_cache_side_effects(before, after, log_before, log_after):
    # Only slicing.cpp's top-level log.txt appends are mutable. A real frozen
    # native probe also observed MEASURE_TIME and RSS records. Their prose is
    # not a scientific input: immutable-path/hash protection is independent of
    # log syntax, so do not confuse a changed diagnostic format with corruption.
    # All other files and path sets, including manifests, remain immutable.
    fixed_before = {k: v for k, v in before.items() if k != 'log.txt'}
    fixed_after = {k: v for k, v in after.items() if k != 'log.txt'}
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if fixed_before != fixed_after:
        raise ValueError('Unexpected non-log cache change: '+repr([k for k in changed if k != 'log.txt']))
    if 'log.txt' in before and 'log.txt' not in after:
        raise ValueError('Runtime slicing log was deleted.')
    if not isinstance(log_before, bytes) or not isinstance(log_after, bytes) or not log_after.startswith(log_before):
        raise ValueError('Runtime slicing log rewrote existing bytes instead of appending.')
    appended = log_after[len(log_before):]
    if len(appended) > 128*1024:
        raise ValueError('Runtime slicing log exceeds the fixed 128 KiB append budget.')
    return {'all_cache_file_hashes_unchanged': before == after,
        'all_nonlog_file_hashes_and_path_set_unchanged': True,
        'allowed_runtime_log_mutation': bool(appended), 'changed_relative_paths': changed,
        'log_append_bytes': len(appended), 'log_append_sha256': hashlib.sha256(appended).hexdigest(),
        'log_original_bytes_preserved': True, 'log_append_record_count': len(appended.splitlines()),
        'allowlist': ['Top-level log.txt only: original byte prefix preserved, append <=128 KiB'],
        'log_syntax_is_not_an_input_integrity_gate': True,
        'observed_native_writer_records': ['processing', 'before/after merging face counts', 'MEASURE_TIME stage durations', 'RSS Memory Usage'],
        'manifest_changes_allowed': False}


def disk_boundary(faces, cycle):
    """Exact labelled two-triangle disk contract, independent of coordinates."""
    faces = [tuple(face) for face in faces]
    if len(faces) != 2 or any(len(f) != 3 or len(set(f)) != 3 for f in faces):
        raise ValueError('Source support is not two nonrepeated triangles.')
    directed, edges = Counter(), Counter()
    for face in faces:
        for a, b in zip(face, face[1:]+face[:1]):
            directed[a, b] += 1
            edges[tuple(sorted((a, b)))] += 1
    internal = [edge for edge, n in edges.items() if n == 2]
    outer = {edge for edge in directed if edges[tuple(sorted(edge))] == 1}
    if (len(edges) != 5 or len(internal) != 1 or
        outer != set(zip(cycle, cycle[1:]+cycle[:1])) or
        any(directed[a, b] != 1 or directed[b, a] != 1 for a, b in internal)):
        raise ValueError('Source support changes the oriented boundary/disk template.')
    return tuple(internal[0])


def natural_schedule(delta):
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Invalid actual deltaT.')
    minimum = float(0.5/24.0)
    result = []
    for frame in range(13, 18):
        global_time = float((frame+0.5)/24.0)
        local = global_time-minimum  # Preserve actual camera timestamp evaluation.
        tau = float(local/delta)
        result.append({'key': f'natural_{frame}', 'kind': 'natural', 'frame': frame,
            'value': local, 'metadata': {'global_camera_time_hex': global_time.hex(),
                'minimum_camera_time_hex': minimum.hex(), 'physical_time_hex': local.hex(),
                'effective_discrete_time_hex': tau.hex(), 'evaluation_tau': str(F.from_float(tau)),
                'delta_t_hex': delta.hex(), 'time_mode': 'physical',
                'phase_rule': 'float((i+0.5)/24)-float(0.5/24); no event-time alignment'}})
    root = F(104, 5)
    physical = float(np.longdouble(root.numerator)*np.longdouble(delta)/np.longdouble(root.denominator))
    result.append({'key': 'exact_root_104_5', 'kind': 'exact_root', 'frame': None, 'value': root,
        'metadata': {'physical_time_hex': physical.hex(), 'evaluation_tau': str(root),
            'interpolation_discrete_time_hex': float(physical/delta).hex(),
            'delta_t_hex': delta.hex(), 'time_mode': 'exact',
            'phase_rule': 'Separate exact event query, not a natural frame'}})
    return result


def controlled_centroid_document(source):
    result = deepcopy(source)
    root = fraction(source['levels']['root'])
    point = next(p for p in source['breakpoint_points'] if fraction(p['time']) == root)
    center = [sum((fraction(row['position'][i]) for row in point['boundary']), F(0))/4 for i in range(3)]
    result['anchors']['root'] = {'kind': 'controlled_exact_four_boundary_centroid', 'position': [fj(x) for x in center]}
    return result


def source_relations(source, inventory=None):
    """Relations are algebraic on every declared cell, not inferred from frames."""
    from run_window_audit import check_junctions, center_at
    from window_source import vid, source_formula, evaluate
    from runtime_retained import build_retained_unit
    junctions = check_junctions(source)
    cycle = list(source['boundary_cycle'])
    points = {fraction(p['time']): p for p in source['breakpoint_points']}
    lower, root, upper = (fraction(source['levels'][k]) for k in ('lower', 'root', 'upper'))
    cells = []
    for segment in source['segments']:
        a, b = fraction(segment['t0']), fraction(segment['t1'])
        for row in segment['boundary']:
            slope, intercept = [tuple(fraction(x) for x in row[k]) for k in ('slope', 'intercept')]
            for key, tau in (('position_t0', a), ('position_t1', b)):
                if tuple(fraction(x) for x in row[key]) != tuple(c+m*tau for c, m in zip(intercept, slope)):
                    raise ValueError('Declared affine boundary coefficients disagree with endpoints.')
            if inventory is not None:
                effective, coefficients, _, _ = source_formula(vid(row['source_vid']), (a+b)/2, inventory.hypervertices)
                if effective.text() != row['source_vid'] or coefficients != (intercept, slope):
                    raise ValueError('Declared boundary differs from original source interpolation.')
        cells.append(('affine_branch', a, b, segment))
    cells += [('actual_singleton', t, t, row) for t, row in sorted(points.items())]
    report = {'status': 'PASS_DECLARED_SOURCE_RELATIONS', 'junctions': junctions,
        'all_declared_cells_checked': len(cells), 'cells': [], 'cross_time_identity': 'Semantic SourceVID only; array indices are per-query.',
        'continuous_runtime_geometry_proven': False, 'raw_cache_owner_coverage_checked': inventory is not None}
    for kind, a, b, row in cells:
        diagonal = disk_boundary(row['source_faces'], cycle)
        owners = [tuple(x) for x in row['owners']]
        if not owners or len(set(owners)) != len(owners):
            raise ValueError('Empty or repeated declared raw owners.')
        if inventory is not None:
            # This call refuses every unpartitioned source/global-group/original
            # record-time threshold. The midpoint only labels a constant cell.
            raw = build_retained_unit(inventory, a, b)
            expected_faces = {oriented(face) for face in row['source_faces']}
            matches = [t for t in raw.triangles if t['element'] == 0 and oriented(t['source_vertices']) in expected_faces]
            if len(matches) != 2 or Counter(tuple(o) for t in matches for o in t['owners']) != Counter(owners):
                raise ValueError('Declared source does not include the complete two-face owner equivalence classes.')
            if a == b:
                for vertex in row['boundary']:
                    effective, coefficients, _, _ = source_formula(vid(vertex['source_vid']), a, inventory.hypervertices)
                    if effective.text() != vertex['source_vid'] or evaluate(coefficients, a) != tuple(fraction(x) for x in vertex['position']):
                        raise ValueError('Actual source singleton differs from frozen boundary.')
        report['cells'].append({'kind': kind, 't0': fj(a), 't1': fj(b), 'owners': [list(o) for o in owners],
            'owner_equivalence_class_sha256': digest(sorted(owners)), 'source_faces': row['source_faces'],
            'source_diagonal': list(diagonal), 'oriented_boundary_cycle': cycle,
            'source_label_support_sha256': digest(row['source_faces']),
            'center_at_t0': center_at(source, a), 'center_at_t1': center_at(source, b)})
    for name, tau in (('lower', lower), ('upper', upper)):
        diagonal = disk_boundary(points[tau]['source_faces'], cycle)
        by_id = {v['source_vid']: [fraction(x) for x in v['position']] for v in points[tau]['boundary']}
        midpoint = [(a+b)/2 for a, b in zip(by_id[diagonal[0]], by_id[diagonal[1]])]
        if midpoint != [fraction(x) for x in source['anchors'][name]['position']]:
            raise ValueError('Endpoint anchor is not the exact source diagonal midpoint.')
    if center_at(source, root) != source['anchors']['root']['position']:
        raise ValueError('Piecewise anchor rule disagrees at the root tie.')
    report.update(anchors_shared_across_every_partition=True, exact_endpoint_midpoints=True,
        root_tie_uses_one_exact_anchor=True,
        source_face_templates_may_change_only_as_declared=True)
    return report


@contextmanager
def timed_runtime(runtime, module, counters):
    originals = {}
    def wrap(name, function):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                row = counters.setdefault(name, {'calls': 0, 'seconds': 0.0})
                row['calls'] += 1; row['seconds'] += time.perf_counter()-start
        return timed
    originals['slice'] = runtime._slice
    originals['resolve_support'] = module.resolve_support
    originals['check_actual_patch'] = module.check_actual_patch
    runtime._slice = wrap('baseline_with_identity_in_transaction', originals['slice'])
    module.resolve_support = wrap('actual_support_plan_resolution', originals['resolve_support'])
    module.check_actual_patch = wrap('actual_candidate_geometry', originals['check_actual_patch'])
    try:
        yield
    finally:
        runtime._slice = originals['slice']
        module.resolve_support = originals['resolve_support']
        module.check_actual_patch = originals['check_actual_patch']


def forced_late_failure(runtime, values, spec, baseline):
    """Refuse at final input recheck, after all proposals, without changing files."""
    original = runtime._cache_digest
    calls = 0
    def rejection():
        nonlocal calls
        calls += 1
        return original() if calls == 1 else 'deliberate-test-rejection-before-publication'
    runtime._cache_digest = rejection
    try:
        output, report = runtime.run(values, spec, time_mode='physical')
    finally:
        runtime._cache_digest = original
    passed = (report['status'] == 'BASELINE_ENTIRE_SCHEDULE' and
        report.get('discarded_proposals') == len(values) and calls == 2 and
        any(row.get('status') == 'PROPOSED_NOT_PUBLISHED' for row in report.get('cases', [])) and
        all(same_mesh(a, b) for a, b in zip(output, baseline)))
    if not passed:
        raise ValueError('Late refusal did not restore every same-query baseline.')
    return {'pass': True, 'mechanism': 'Injected final cache-digest refusal; no cache file mutated.',
        'all_requested_queries_returned_baseline_bits': True, 'report': report}


def prepare_schedule(repo, cache_copy, source_document):
    """Return six RAM meshes/case plus a JSON-safe structural manifest."""
    repo, cache = Path(repo).resolve(), Path(cache_copy).resolve()
    if os.name != 'posix':
        raise ValueError('Run the frozen Linux build through WSL.')
    if any(os.environ.get(k) for k in ('BINOC_SOURCE_SPLICE_PLAN', 'BINOC_SOURCE_SPLICE_AUDIT', 'BINOC_SOURCE_SPLICE_TRACE')):
        raise ValueError('Existing SSP1 intervention environment is not part of this experiment.')
    import resource
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    started = time.perf_counter()
    frozen = HERE/'artifacts/ab_20260906/window_audit/event-02-88ade47aa4fa/source.json'
    original = frozen.read_bytes()
    if hashlib.sha256(original).hexdigest() != FROZEN_SOURCE_SHA:
        raise ValueError('Frozen source report bytes changed.')
    source = deepcopy(source_document) if isinstance(source_document, dict) else json.loads(Path(source_document).read_bytes())
    if digest(source) != digest(json.loads(original)):
        raise ValueError('This diagnostic only accepts the frozen original E2 source.')
    core = repo/'binocmesher/lib/core.so'
    if hashlib.sha256(core.read_bytes()).hexdigest() != FROZEN_CORE_SHA:
        raise ValueError('Expected the frozen independently tested E2 native build.')
    loaded = sys.modules.get('binocmesher')
    if loaded is not None and repo not in Path(loaded.__file__).resolve().parents:
        raise ValueError('A different binocmesher package is already loaded; use a fresh process.')
    sys.path.insert(0, str(repo))
    from binocmesher import window_runtime as wr
    from run_e2_runtime import ordinary, initialize_mesher, manifest as cache_manifest
    from e2_runtime_validation import validate_e2_runtime
    from window_source import read_inventory, file_signatures, input_files
    before = cache_manifest(cache)
    log_path = cache/'log.txt'
    if log_path.is_symlink():
        raise ValueError('Symlinked slicing log is not supported.')
    log_before = log_path.read_bytes() if log_path.is_file() else b''
    inventory = read_inventory(cache)
    if inventory.digest != source['cache_input_sha256']:
        raise ValueError('Supplied cache copy differs from frozen source inputs.')
    plan_start = time.perf_counter()
    relations = source_relations(source, inventory)
    control = controlled_centroid_document(source)
    control_relations = source_relations(control)  # Identical cache-backed source fields, only root anchor differs.
    for key in ('levels', 'boundary_cycle', 'segments', 'breakpoint_points'):
        if control[key] != source[key]:
            raise ValueError('Centroid control changed more than the root anchor.')
    specs = {'centroid_window': wr.WindowSpec.from_source_contract(control),
             'beb1_window': wr.WindowSpec.from_source_contract(source)}
    for spec in specs.values():
        spec.validate_layout()
    plan_seconds = time.perf_counter()-plan_start
    initialization_start = time.perf_counter()
    mesher = initialize_mesher(repo, cache)
    initialization_seconds = time.perf_counter()-initialization_start
    if Path(mesher._runtime_library._name).resolve() != core.resolve():
        raise ValueError('Actually loaded native library differs from the frozen path.')
    delta = float(mesher._window_delta_t)
    if delta.hex() != FROZEN_DELTA_HEX or float(mesher.tsize)/(2*16) != delta:
        raise ValueError('Initialized actual time scale differs from the frozen build.')
    runtime = wr.WindowRuntime(mesher)
    scheduled = natural_schedule(delta)
    cases, baselines, ledgers = [], [], []
    plain_seconds = trace_seconds = 0.0
    for item in scheduled:
        mode = item['metadata']['time_mode']
        begin = time.perf_counter(); base = ordinary(mesher, item['value'], mode); plain_seconds += time.perf_counter()-begin
        begin = time.perf_counter(); traced, ids, owners, status = runtime._slice(item['value'], mode, False, True); trace_seconds += time.perf_counter()-begin
        if status != 1 or not same_mesh(base, traced):
            raise ValueError('Actual identity instrumentation changes same-query baseline arrays.')
        baselines.append(base); ledgers.append((ids, owners))
        cases.append({k: deepcopy(v) for k, v in item.items() if k != 'value'})
        cases[-1]['methods'] = {'raw': {'mesh': base, 'audit': {
            'status': 'UNCHANGED_RAW_BASELINE', 'mesh_sha256': mesh_hash(base),
            'instrumented_same_query_baseline_bit_exact': True,
            'vertex_ledger_sha256': array_hash(ids), 'raw_owner_ledger_sha256': array_hash(owners),
            'vertex_ledger_rows': len(ids), 'raw_owner_ledger_rows': len(owners)}}}
        cases[-1].update(source_face_rows=None, source_boundary_actual_ids=None)
    transactions, profiles = {}, {}
    for method, spec in specs.items():
        method_transactions, measurements = [], {}
        for mode, indices in (('physical', list(range(5))), ('exact', [5])):
            values = [scheduled[i]['value'] for i in indices]
            begin = time.perf_counter()
            with timed_runtime(runtime, wr, measurements):
                output, report = runtime.run(values, spec, time_mode=mode)
            method_transactions.append({'mode': mode, 'wall_seconds': time.perf_counter()-begin, 'report': report})
            if report['status'] != 'COMMITTED_REQUESTED_SCHEDULE':
                raise ValueError(method+' refused: '+str(report.get('fallback_reason')))
            for index, actual, audit in zip(indices, output, report['cases']):
                expected, (ids, owners) = baselines[index], ledgers[index]
                tau = F(audit['evaluation_tau'])
                if audit['status'] == 'APPLIED_ACTUAL_QUERY':
                    validation_start = time.perf_counter()
                    requested, _ = spec.cell(tau)
                    cycle, removed, consumed = wr.resolve_support(expected, ids, owners, spec, tau)
                    independent = validate_e2_runtime(expected, actual, ids, owners, spec.cycle, requested,
                        consumed_owners=audit['consumed_owners'], expected_center=spec.center(tau))
                    if not independent['pass'] or tuple(sorted(map(tuple, audit['consumed_owners']))) != consumed:
                        raise ValueError('Independent same-frame identity/array/owner contract failed.')
                    timing = measurements.setdefault('independent_array_and_support_validation', {'calls': 0, 'seconds': 0.0})
                    timing['calls'] += 1; timing['seconds'] += time.perf_counter()-validation_start
                    if cases[index]['source_face_rows'] is not None and cases[index]['source_face_rows'] != list(removed):
                        raise ValueError('Controls disagree about original authorized support.')
                    cases[index].update(source_face_rows=list(removed), source_boundary_actual_ids=list(cycle))
                    audit = {**audit, 'independent': independent,
                        'semantic_boundary_cycle': source['boundary_cycle'],
                        'source_to_actual_id': [{'source_vid': key, 'actual_id': int(value)} for key, value in zip(source['boundary_cycle'], cycle)],
                        'internal_center_semantic_role': 'E2/window-center',
                        'cross_time_array_id_equality_claimed': False,
                        'oriented_disk_template': [[i, (i+1) % 4, 'center'] for i in range(4)]}
                else:
                    if audit['status'] not in ('BASELINE_OUTSIDE', 'BASELINE_ENDPOINT') or not same_mesh(expected, actual):
                        raise ValueError('Inactive/endpoint query changed baseline bytes.')
                    audit = {**audit, 'same_query_baseline_bit_exact': True}
                cases[index]['methods'][method] = {'mesh': actual, 'audit': audit}
        transactions[method] = method_transactions
        profiles[method] = measurements
    negative_begin = time.perf_counter()
    negative = forced_late_failure(runtime, [item['value'] for item in scheduled[:5]], specs['beb1_window'], baselines[:5])
    negative_seconds = time.perf_counter()-negative_begin
    # Refusal must not contaminate the next complete physical transaction.
    restored, restoration = runtime.run([item['value'] for item in scheduled[:5]], specs['beb1_window'], time_mode='physical')
    if restoration['status'] != 'COMMITTED_REQUESTED_SCHEDULE' or not all(same_mesh(mesh, cases[i]['methods']['beb1_window']['mesh']) for i, mesh in enumerate(restored)):
        raise ValueError('Late-refusal test leaked state into the next schedule.')
    if file_signatures(input_files(cache)) != inventory.signatures or wr.source_cache_digest(cache) != inventory.digest:
        raise ValueError('Protected source inputs changed in bytes, paths or timestamps during preparation.')
    if log_path.is_symlink():
        raise ValueError('Runtime slicing log became a symlink.')
    side_effects = audit_cache_side_effects(before, cache_manifest(cache), log_before,
        log_path.read_bytes() if log_path.is_file() else b'')
    manifest_cases = []
    for case in cases:
        entry = {k: v for k, v in case.items() if k != 'methods'}
        entry['methods'] = {name: {'audit': row['audit'], 'mesh_sha256': mesh_hash(row['mesh']),
            'vertices': len(row['mesh'][0]), 'faces': len(row['mesh'][1]),
            'array_bytes': sum(x.nbytes for x in row['mesh'])} for name, row in case['methods'].items()}
        manifest_cases.append(entry)
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result = {'schema': 'e2-fixed-schedule-structural-coherence-v1',
        'status': 'PASS_SCHEDULE_STRUCTURAL_COHERENCE_ONLY',
        'continuous_window_admitted': False, 'same_root_group_admitted': False,
        'target_scope': 'E2 only. No E2/E3 group isolation or arbitrary group admission is assumed.',
        'schedule': [{k: v for k, v in item.items() if k != 'value'} for item in scheduled],
        'schedule_sha256': digest([{k: v for k, v in item.items() if k != 'value'} for item in scheduled]),
        'input_binding': {'core_so_sha256': FROZEN_CORE_SHA, 'core_path': str(core),
            'actual_loaded_library_path': str(Path(mesher._runtime_library._name).resolve()),
            'runtime_python_module_path': str(Path(wr.__file__).resolve()),
            'runtime_python_sha256': hashlib.sha256(Path(wr.__file__).read_bytes()).hexdigest(),
            'coherence_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_file_sha256': FROZEN_SOURCE_SHA, 'source_document_sha256': digest(source),
            'cache_input_sha256': inventory.digest, 'delta_t_hex': delta.hex(), 'tsize_hex': float(mesher.tsize).hex(),
            'cache_copy_source_inputs_unchanged': True,
            'source_input_hashes_and_timestamps_unchanged': True,
            'cache_copy_all_file_bytes_unchanged': side_effects['all_cache_file_hashes_unchanged'],
            'allowed_runtime_log_mutation': side_effects['allowed_runtime_log_mutation'],
            'cache_side_effect_audit': side_effects, 'caller_owns_cache_cleanup': True},
        'source_relations': relations, 'centroid_anchor_relations': control_relations,
        'controlled_comparison': {'only_root_anchor_changed': True,
            'beb1_root': source['anchors']['root'], 'centroid_root': control['anchors']['root'],
            'common_endpoint_anchors': {k: source['anchors'][k] for k in ('lower', 'upper')},
            'boundary_owner_source_template_shared': True},
        'declared_policy': {'active': 'Strict window interior; actual physical input selects physical activation bounds.',
            'endpoints': 'Same-query raw baseline, with endpoint contract verified if explicitly requested.',
            'outside': 'Same-query raw baseline; no appended unused center.',
            'root': 'Separate exact root request uses its complete singleton owner state and one frozen method-specific root anchor.',
            'physical_batch': 'All five natural queries prepare then publish atomically.',
            'root_batch': 'Separate one-query exact transaction, explicitly not merged with physical timeline.',
            'cross_frame_comparison': 'Semantic SourceVID and declared oriented template, never array-index equality across times.'},
        'actual_active_natural_frames': [c['frame'] for c in cases[:5] if c['methods']['beb1_window']['audit']['status'] == 'APPLIED_ACTUAL_QUERY'],
        'cases': manifest_cases, 'transactions': transactions,
        'negative_late_failure': negative, 'state_reset_after_failure': True,
        'cost': {'initialization_and_cache_reuse_seconds': initialization_seconds,
            'source_relations_and_plan_construction_seconds': plan_seconds,
            'plain_baseline_6queries_seconds': plain_seconds, 'identity_baseline_6queries_seconds': trace_seconds,
            'identity_minus_plain_wall_estimate_seconds': trace_seconds-plain_seconds,
            'identity_cost_notice': 'Difference of separate wall timings, not isolated native ledger CPU cost; may include cache/noise effects.',
            'transaction_stages': profiles, 'negative_test_seconds': negative_seconds,
            'total_prepare_seconds': time.perf_counter()-started,
            'process_peak_rss_before_bytes': peak_before, 'process_peak_rss_after_bytes': peak_after,
            'process_peak_rss_increase_bytes': max(0, peak_after-peak_before),
            'peak_rss_notice': 'Linux process high-water RSS, not allocator-isolated method memory.',
            'plain_baseline_queries': 6, 'standalone_identity_queries': 6,
            'primary_transaction_queries': 12, 'negative_test_queries': 5, 'state_restore_queries': 5,
            'native_build_cost_included': False, 'render_cost_included': False},
        'scientific_limits': ['This proves fixed-schedule structural coherence and checks actual geometry only at those queries.',
            'No all-real-time continuity/embedding or visual superiority follows from this status.',
            'Natural schedule may contain only one active query; the explicit active-frame list is part of the result.',
            'Finite source relations are ideal-source algebra tied to cache; not automatic full-time floating realization.']}
    json.dumps(result, allow_nan=False)
    return {'cases': cases, 'manifest': result}
