"""Existing lower-first source selector over completeness-certified halos.

This does not derive ordinary S_B from the critical face's four edges. It
reuses the existing event-record disk selector unchanged, then validates its
chosen labelled support. A source PASS is neither runtime nor group admission.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
for folder in (HERE.parent/'source_splice', HERE.parent/'tv0_tv4'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import compile_splice_plans as legacy
import compile_critical_beb1_event_ir as ir
import window_source as ws
from processed_mesh import HVID, SourceVID, canonical_face_groups, edge_incidence

_LOCK = threading.RLock()


def fr(x):
    return F(x['numerator'], x['denominator']) if isinstance(x, dict) else F(x)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Incomplete(ValueError):
    pass


class FixedReject(ValueError):
    def __init__(self, message, **witness):
        super().__init__(message)
        self.witness = witness


def levels(root, corner_times):
    root = F(root)
    times = [F(t) for t in corner_times]
    if len(times) != 4 or not min(times) < root < max(times):
        raise Incomplete('Malformed four-corner event-time input.')
    epsilon = min(abs(t-root) for t in times)/2
    if epsilon <= 0:
        raise FixedReject('NO_POSITIVE_OLD_COMPILER_CLEARANCE_WINDOW')
    return root-epsilon, root, root+epsilon


def required_times(root, corner_times):
    a, root, b = levels(root, corner_times)
    points = sorted({a, root, b, *(F(i) for i in range(a.numerator//a.denominator, b.numerator//b.denominator+1) if a <= i <= b)})
    return sorted(set(points) | {(x+y)/2 for x, y in zip(points, points[1:])})


def _snapshot(snapshots, event_id, tau):
    snapshot = snapshots.get((event_id, F(tau)))
    if snapshot is None or not getattr(snapshot, 'halo_complete', False) or not getattr(snapshot, 'event_candidates_complete', False):
        raise Incomplete('Missing complete event-candidate/vertex-halo snapshot at '+str(tau))
    raw = tuple(snapshot.raw_triangles)
    refs = [r.reference.values() for r in raw]
    if len(set(refs)) != len(refs):
        raise Incomplete('Duplicated raw owner in snapshot.')
    vertices = {v for r in raw if r.event_record for v in r.source_vertices}
    if vertices != set(snapshot.halo_vertex_ids):
        raise Incomplete('Candidate-vertex halo does not match all event-record vertices.')
    # Reader verifies primary-file sorted-index monotonicity. This is the exact
    # legacy traversal order, including decreasing active Fenwick group order.
    raw = tuple(sorted(raw, key=lambda r: (-r.reference.t_group, r.reference.t_start,
        r.reference.sorted_record_index, r.reference.interval_index, r.reference.face_index, r.reference.fan_index)))
    return snapshot, raw


def select_existing(snapshots, event_id, tau, required_boundary=None):
    snapshot, raw = _snapshot(snapshots, event_id, tau)
    summary = {'event_id': event_id, 'raw_triangles': len(raw),
        'event_raw_triangles': sum(r.event_record for r in raw),
        'scope': 'Complete event-candidate vertex halo, not materialized whole mesh.',
        'event_candidates_complete': True, 'halo_complete': True}
    def supplied(cache, time, event_id=None):
        if F(time) != F(tau) or event_id != summary['event_id']:
            raise Incomplete('Legacy adapter query binding changed.')
        return list(raw), summary
    with _LOCK, patch.object(legacy, 'trace_processed_triangles', supplied):
        try:
            return ir.compile_ordinary_patch(Path('.'), F(tau), required_boundary, event_id)
        except RuntimeError as error:
            if str(error).startswith('no closed connected two-triangle event-star patch'):
                raise FixedReject('LEGACY_EXHAUSTIVE_SELECTOR_NO_CANDIDATE: '+str(error),
                    time=ws.fj(F(tau)), event_raw_candidates=sum(r.event_record for r in raw),
                    complete_halo_raw_triangles=len(raw),
                    required_boundary=None if required_boundary is None else sorted(v.text() for v in required_boundary)) from error
            raise Incomplete('Unexpected legacy selector error: '+str(error)) from error


def _hvid(x):
    return HVID(x.node, x.group) if hasattr(x, 'node') else HVID(*map(int, x))


def _original_source(pair):
    if isinstance(pair, SourceVID):
        return pair
    return SourceVID(_hvid(pair[0]), _hvid(pair[1]))


def exact_support(snapshot, raw, tau, cycle, element, hypervertices):
    groups = canonical_face_groups(raw)
    event_keys = {key for key, rows in groups.items() if any(r.event_record for r in rows)}
    try:
        selected = ws.unique_support(groups, event_keys, edge_incidence(groups), cycle, element)
    except ws.SourceStop as error:
        raise FixedReject('FIXED_UNIQUE_SAME_BOUNDARY_POLICY: '+str(error), time=ws.fj(F(tau)),
            required_cycle=[v.text() for v in cycle],
            complete_halo_oriented_faces=len(groups), complete_event_oriented_candidates=len(event_keys)) from error
    coeffs, owners, branches, views = {}, [], defaultdict(set), defaultdict(set)
    raw_counts = Counter()
    for key in selected:
        for row in groups[key]:
            owner = row.reference.values()
            detail = snapshot.details_by_owner.get(owner)
            if detail is None or type(detail.get('actual_raw_emitted')) is not bool:
                raise Incomplete('Selected owner lacks exact raw-emission evidence.')
            raw_counts['selected_superset_owners'] += 1
            raw_counts['selected_actual_raw_owners'] += detail['actual_raw_emitted']
            if not detail['actual_raw_emitted']:
                raise FixedReject('SELECTED_SUPERSET_OWNER_NOT_ACTUALLY_RAW_EMITTED: '+repr(owner),
                    owner=list(owner), time=ws.fj(F(tau)), raw_emission_counts=dict(raw_counts))
            if detail.get('native_legacy_identity_equal') is not True:
                raise Incomplete('SELECTED_NATIVE_LEGACY_IDENTITY_BINDING_UNRESOLVED: '+repr(owner))
            original = detail.get('ordered_raw_vids')
            if original is None or len(original) != 3:
                raise Incomplete('Selected owner lacks original ordered vertex identities.')
            owners.append(owner)
            for pair, target in zip(original, row.source_vertices):
                effective, coeff, view, branch = ws.source_formula(_original_source(pair), tau, hypervertices)
                if effective != target:
                    raise FixedReject('SELECTED_EXACT_SOURCE_IDENTITY_DISAGREEMENT')
                if target in coeffs and coeffs[target] != coeff:
                    raise FixedReject('SELECTED_REPLICA_AFFINE_TRAJECTORY_DISAGREEMENT')
                coeffs[target] = coeff
                branches[target].add(branch); views[target].add(view)
    if set(coeffs) != set(cycle) or len(set(owners)) != len(owners):
        raise FixedReject('SELECTED_SOURCE_LABEL_OR_OWNER_ARITY')
    return {'selected': selected, 'owners': sorted(owners), 'formulae': coeffs,
        'branches': branches, 'views': views, 'raw_counts': dict(raw_counts)}


def _payload(support, cycle):
    return {'source_faces': [[v.text() for v in key[1]] for key in support['selected']],
        'owners': [list(o) for o in support['owners']], 'boundary_cycle': [v.text() for v in cycle],
        'unique_combinatorial_support': True, 'numerical_selector_conditioning': 'Fixed legacy lower-first selector; unique same-boundary subsequent support.'}


def compile_kernel(rows, hypervertices):
    """Invoke actual TV3 half-handle and forced TV4 combinatorial completion."""
    from theory_audit import (HVID as THVID, SourceRecord, Cell, admissible_saddle,
        canonical_partition, build_production_half_handle, complete_production_event_star)
    report = {'status': 'UNKNOWN', 'runtime_admitted': False,
        'scope': 'Pure existing TV3 construction and forced combinatorial completion; not a fabricated TV3/4 campaign certificate.'}
    try:
        if not rows:
            raise Incomplete('No registry rows.')
        row = sorted(rows, key=lambda r: r['raw_id'])[0]
        parse = lambda text: THVID(*map(int, text.split(':')))
        hs = tuple(parse(row[f'h{i}']) for i in range(4))
        lookup = lambda h: hypervertices[HVID(h.node, h.group)]
        ts = tuple(lookup(h).time for h in hs)
        saddle = admissible_saddle(hs, ts, int(row['element']))
        if saddle is None or saddle.root != F(int(row['root_num']), int(row['root_den'])) or saddle.event_id() != row['canonical_event_id']:
            raise FixedReject('REGISTRY_NOT_THE_EXACT_RECOMPUTED_SADDLE')
        if any(r['canonical_event_id'] != saddle.event_id() or F(int(r['root_num']), int(r['root_den'])) != saddle.root for r in rows):
            raise FixedReject('REGISTRY_REPLICA_EVENT_DISAGREEMENT')
        ch = tuple(parse(row[f'source_h{i}']) for i in range(8))
        ct = tuple(lookup(h).time for h in ch)
        record = SourceRecord(int(row['source_t_group']), int(row['source_record_index']),
            tuple(int(row[k]) for k in ('edge_x', 'edge_y', 'edge_z')),
            *(int(row[k]) for k in ('edge_L', 'edge_tcoord', 'edge_tL', 'edge_dir', 'element')), ch)
        cell = Cell('compiler-selected-source-cell', record, len(rows), ch, ct,
            np.asarray([lookup(h).position for h in ch], np.float64), tuple(ct[i+4]-ct[i] for i in range(4)),
            'NOT_ROUTE_CERTIFIED', canonical_partition(ch), 'NOT_SAMPLED', 0., 0.)
        vertices, tets, metadata = build_production_half_handle(cell, saddle, int(row['face_side']))
        completed, completion = complete_production_event_star(tets)
        report.update(status='PASS_EXISTING_LOCAL_KERNEL', event_id=saddle.event_id(), root=ws.fj(saddle.root),
            corner_times=list(ts), critical_position=metadata['critical_position'],
            block_vertex_source_hvids=metadata['block_vertex_source_hvids'],
            block_vertex_exact_times=metadata['block_vertex_exact_times'],
            vertices4=vertices.tolist(), half_handle_tets=tets.tolist(), completed_tets=completed.tolist(),
            completion=completion, registry_logical_incidences=len({r['logical_incidence_id'] for r in rows}))
    except FixedReject as error:
        report.update(status='REJECT_FIXED_LOCAL_KERNEL', reason=str(error))
    except Exception as error:
        report.update(status='UNKNOWN', reason=type(error).__name__+': '+str(error))
    return report


def compile_event(event_id, root, corner_times, snapshots, hypervertices, critical_position,
                  *, cache_digest, group_count, maximum_discrete_time, element=None):
    report = {'schema': 'forest-existing-source-compiler-v1', 'status': 'UNKNOWN', 'event_id': event_id,
        'source': None, 'stages': {}, 'ordinary_patches': {}, 'selector_attempts': [], 'runtime_admitted': False,
        'same_root_group_admitted': False,
        'selector_scope': 'Unmodified compile_at_time, lower-first; complete event-record candidate-vertex halo.',
        'rejection_scope': 'Fixed existing compiler/policy only, not impossibility of every other source support or meshing method.',
        'halo_equivalence_argument': 'All candidate triangle vertices are in V. Every opposite face, incidence on a tested candidate edge, and candidate-face owner replica intersects V and is included by the complete V-star. Legacy traversal order is preserved.'}
    stage = 'input_contract'
    try:
        if not isinstance(group_count, int) or group_count <= 0 or maximum_discrete_time != 2*group_count:
            raise Incomplete('Integer source/group threshold model not verified.')
        lower, root, upper = levels(root, corner_times)
        if not 0 <= lower < upper <= maximum_discrete_time:
            raise FixedReject('FIXED_WINDOW_OUTSIDE_CACHE')
        probes = {'lower': lower, 'critical': root, 'upper': upper}
        runtimes, patches = {}, {}
        boundary = None
        stage = 'legacy_selector'
        for name, tau in probes.items():
            snapshot, candidate_raw = _snapshot(snapshots, event_id, tau)
            attempt = {'name': name, 'time': ws.fj(tau),
                'required_boundary': None if boundary is None else sorted(v.text() for v in boundary),
                'complete_event_raw_candidates': sum(r.event_record for r in candidate_raw),
                'complete_halo_raw_triangles': len(candidate_raw), 'status': 'ATTEMPTED'}
            report['selector_attempts'].append(attempt)
            payload, runtime = select_existing(snapshots, event_id, tau, boundary)
            patches[name], runtimes[name] = payload, runtime
            report['ordinary_patches'][name] = payload
            attempt.update(status='PASS', selected_source_faces=payload['source_faces'], selected_boundary=payload['boundary_cycle'])
            if boundary is None:
                boundary = frozenset(runtime['cycle'])
        report['ordinary_patches'] = patches
        cycle = runtimes['lower']['cycle']
        found_element = runtimes['lower']['element']
        if element is not None and int(element) != found_element:
            raise FixedReject('REQUESTED_ELEMENT_DIFFERS_FROM_SELECTED_SOURCE')
        element = found_element
        if any(r['element'] != element or not ws.same_cycle(r['cycle'], cycle) for r in runtimes.values()):
            raise FixedReject('LOWER_ROOT_UPPER_ORIENTED_SOURCE_BOUNDARY_CHANGED')
        report['stages'][stage] = {'status': 'PASS', 'order': list(probes),
            'fixed_boundary': [v.text() for v in cycle], 'element': element,
            'legacy_source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(legacy.__file__), Path(ir.__file__))}}
        stage = 'raw_owner_emission'
        points = sorted({lower, root, upper, *(F(i) for i in range(lower.numerator//lower.denominator, upper.numerator//upper.denominator+1) if lower <= i <= upper)})
        requests = sorted(set(points) | {(a+b)/2 for a, b in zip(points, points[1:])})
        support = {}
        raw_counts = Counter()
        for tau in requests:
            snapshot, raw = _snapshot(snapshots, event_id, tau)
            support[tau] = exact_support(snapshot, raw, tau, cycle, element, hypervertices)
            raw_counts.update(support[tau]['raw_counts'])
        report['stages'][stage] = {'status': 'PASS', **dict(raw_counts), 'all_selected_owner_replicas_actually_raw_emitted': True,
            'native_vs_legacy_source_identity_equal': True, 'scope': 'Exact source/owner model, not floating C++ realization.'}
        stage = 'exact_source_contract'
        contract = ir.build_space_position_contract(np.asarray(critical_position, np.float64))
        center = [F.from_float(float(x)) for x in contract['canonical_position_float64']]
        source = {'schema': 'forest-compiled-source-window-v1', 'status': 'PASS_SOURCE_BRANCH_CONTRACT',
            'element': element, 'event_id': event_id, 'coordinate_model': 'ideal rational interpolation of serialized HV binary32',
            'cache_input_sha256': cache_digest, 'levels': {k: ws.fj(t) for k, t in zip(('lower', 'root', 'upper'), (lower, root, upper))},
            'boundary_cycle': [v.text() for v in cycle], 'anchors': {'root': {'kind': 'canonical_tv3_binary32',
                'position': ws.vj(center), 'binary32_words': contract['canonical_binary32_hex']}},
            'segments': [], 'breakpoint_points': [], 'raw_owner_emission': report['stages']['raw_owner_emission'],
            'partition_proof': {'all_thresholds_integer': True, 'group_count': group_count,
                'maximum_discrete_time': maximum_discrete_time, 'points': [ws.fj(t) for t in points],
                'rule': 'Every original/expanded owner time, load/group threshold, and HV effective clamp threshold is an integer. All integers in the closed window plus root are split; a midpoint labels the resulting proven constant open cell.'},
            'junction_boundary_cycle_and_orientation_equal': True, 'boundary_continuity_exact': True,
            'runtime_admission': False, 'continuous_window_admitted': False}
        for tau in points:
            item = support[tau]
            source['breakpoint_points'].append({**_payload(item, cycle), 'time': ws.fj(tau),
                'boundary': [{'source_vid': v.text(), 'position': ws.vj(ws.evaluate(item['formulae'][v], tau))} for v in cycle]})
        for a, b in zip(points, points[1:]):
            item = support[(a+b)/2]
            rows = []
            for v in cycle:
                coeff = item['formulae'][v]
                pa, pb = ws.evaluate(coeff, a), ws.evaluate(coeff, b)
                if pa != ws.evaluate(support[a]['formulae'][v], a) or pb != ws.evaluate(support[b]['formulae'][v], b):
                    raise FixedReject('EXACT_SOURCE_TRAJECTORY_JUMPS_AT_A_DECLARED_BREAKPOINT')
                rows.append({'source_vid': v.text(), 'position_t0': ws.vj(pa), 'position_t1': ws.vj(pb),
                    'intercept': ws.vj(coeff[0]), 'slope': ws.vj(coeff[1]),
                    'source_branches': sorted(item['branches'][v]), 'in_view_values': sorted(item['views'][v])})
            source['segments'].append({**_payload(item, cycle), 't0': ws.fj(a), 't1': ws.fj(b),
                'slab': 'left' if b <= root else 'right', 'source_branch_certificate': True, 'boundary': rows})
        for name, tau in (('lower', lower), ('upper', upper)):
            selected = support[tau]['selected']
            diagonal = sorted(set(selected[0][1]) & set(selected[1][1]))
            positions = [ws.evaluate(support[tau]['formulae'][v], tau) for v in diagonal]
            if len(diagonal) != 2 or positions[0] == positions[1]:
                raise FixedReject('EXACT_SOURCE_ENDPOINT_DIAGONAL_DEGENERATE')
            source['anchors'][name] = {'kind': 'exact_source_diagonal_midpoint',
                'source_diagonal': [v.text() for v in diagonal],
                'position': ws.vj([(a+b)/2 for a, b in zip(*positions)])}
        report['source'] = source
        report['stages'][stage] = {'status': 'PASS', 'affine_branches': len(source['segments']),
            'actual_singletons': len(source['breakpoint_points']), 'source_contract_sha256': digest(source)}
        stage = 'legacy_mapping_cylinder'
        canonical = ir.position_float64(contract['canonical_position_float64'])
        cylinder = ir.build_event_star_mapping_cylinder(patches, runtimes, probes, event_id, canonical)
        report['stages'][stage] = {k: cylinder[k] for k in ('pass', 'verdict', 'pass_scope', 'checks',
            'minimum_gram_volume', 'critical_side_edges_remaining', 'sidewall_geometry_compatible')}
        report['stages'][stage]['status'] = 'PASS' if cylinder['pass'] else 'REJECT'
        stage = 'quantization'
        theory_slice = ir.build_whole_mesh_patch_slice(patches['critical'], runtimes['critical'], event_id, np.asarray(critical_position, np.float64))
        actual_slice = ir.build_whole_mesh_patch_slice(patches['critical'], runtimes['critical'], event_id, canonical)
        quantization = ir.audit_critical_position_quantization(theory_slice, actual_slice, contract)
        report['stages'][stage] = quantization
        report['stages'][stage]['status'] = 'PASS' if quantization['pass'] else 'REJECT'
        report['legacy_side_trace'] = ir.side_trace_affine_audit(patches, probes)
        # These are preserved legacy three-level gates, not a CCD/continuous
        # embedding proof. Actual interface/local/global gates remain separate.
        report.update(status='PASS_SOURCE_COMPILER', source_contract_ready=True,
            legacy_three_level_compatibility=bool(cylinder['pass'] and quantization['pass']),
            legacy_diagnostics_scope='Preserved old three-level construction diagnostics, not necessary all-time or nonquery-endpoint gates for this requested schedule.',
            remaining_gates=['Actual source interface/retained defects',
                'Actual binary32 requested-query local and exterior geometry', 'Requested-schedule atomic publication'])
    except FixedReject as error:
        report.update(status='REJECT_FIXED_SOURCE_COMPILER', reason=str(error), rejected_stage=stage, source_contract_ready=False)
        report['decisive_witness'] = error.witness
        report['stages'].setdefault(stage, {'status': 'REJECT', 'reason': str(error), 'witness': error.witness})
        if stage == 'legacy_selector' and report['selector_attempts']:
            report['selector_attempts'][-1].update(status='REJECT', reason=str(error), witness=error.witness)
    except Exception as error:
        report.update(status='UNKNOWN', reason=type(error).__name__+': '+str(error), unknown_stage=stage, source_contract_ready=False)
        report['stages'].setdefault(stage, {'status': 'UNKNOWN', 'reason': str(error)})
    return report
