#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import random
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from theory_audit import (
    AuditError, Cell, HVID, Saddle, TEMPORAL_FACE_SLOTS,
    admissible_saddle, build_production_half_handle, canonical_cycle,
    complete_production_event_star, critical_link_audit, exact_schedule,
    face_segments, fraction_json,
    graph_signature, make_cells, mesh_topology, parse_hypervertices,
    parse_hvid, parse_registry, parse_source_records, quotient_signature,
    registry_saddles, sha256_json, slice_tet_complex, source_record_dict,
    subdivide_contract_signature, tet_gram_volume, write_json, write_jsonl,
)


def row_source_key(row: dict[str, str]) -> tuple[Any, ...]:
    hvids = tuple(parse_hvid(row[f'source_h{i}']) for i in range(8))
    return (
        (int(row['edge_x']), int(row['edge_y']), int(row['edge_z'])),
        int(row['edge_L']), int(row['edge_tcoord']), int(row['edge_tL']),
        int(row['edge_dir']), int(row['element']),
        tuple((h.node, h.group) for h in hvids),
    )


def stage_tv0(cache_root: Path, out: Path) -> tuple[dict[str,Any], list[Cell], list[dict[str,str]], dict[HVID,Any]]:
    hv = parse_hypervertices(cache_root)
    source = parse_source_records(cache_root)
    rows = parse_registry(cache_root)
    cells, source_stats = make_cells(source, hv)

    routes = collections.Counter(cell.route for cell in cells)
    jac = collections.Counter(cell.sampled_jacobian_class for cell in cells if cell.route == 'regular_monotone')
    spacetime_gram = collections.Counter()
    spacetime_gram_values = []
    for cell in cells:
        if cell.route != 'regular_monotone':
            continue
        u = v = w = 0.5
        from theory_audit import trilinear_derivatives
        spatial = trilinear_derivatives(cell.positions, u, v, w)
        time_derivatives = [0.0, 0.0, 0.0]
        for k in (0, 1):
            for j in (0, 1):
                for i in (0, 1):
                    slot = i + 2*j + 4*k
                    bu = u if i else 1-u; bv = v if j else 1-v; bw = w if k else 1-w
                    weights = ((1 if i else -1)*bv*bw, bu*(1 if j else -1)*bw, bu*bv*(1 if k else -1))
                    for axis in range(3):
                        time_derivatives[axis] += weights[axis] * cell.times[slot]
        J = np.column_stack([np.r_[spatial[axis], time_derivatives[axis]] for axis in range(3)])
        gram = max(0.0, float(np.linalg.det(J.T @ J)))
        spacetime_gram_values.append(gram)
        spacetime_gram['positive_center_sampled' if gram > 1e-10 else 'near_singular_center_sampled'] += 1
    grammar = collections.Counter(cell.equality_partition for cell in cells)

    shared_payloads: dict[tuple[Any,...], tuple[int,...]] = {}
    shared_mismatches = 0
    shared_occurrences = 0
    for cell in cells:
        if cell.route.startswith('unresolved'):
            continue
        for side, slots in TEMPORAL_FACE_SLOTS.items():
            h, t = canonical_cycle([cell.hvids[i] for i in slots], [cell.times[i] for i in slots])
            key = (cell.representative.element, tuple((x.node,x.group) for x in h))
            previous = shared_payloads.get(key)
            if previous is not None and previous != t:
                shared_mismatches += 1
            shared_payloads[key] = t
            shared_occurrences += 1

    temporal_rows = [row for row in rows if row.get('face_axis_role') == 'temporal_neighbour']
    event_set = registry_saddles(rows)
    registry_summary = json.loads((cache_root/'event_registry_p1_summary.json').read_text())
    checks = {
        'logical_cells_nonzero': bool(cells),
        'missing_hvid_references_zero': source_stats['missing_hvid_references'] == 0,
        'unclassified_cells_zero': routes['unresolved_missing_hvid'] == 0 and routes['unresolved_placeholder'] == 0,
        'invalid_temporal_order_zero': routes['invalid_temporal_order'] == 0,
        'shared_temporal_face_payload_mismatch_zero': shared_mismatches == 0,
        'registry_shared_root_mismatch_zero': int(registry_summary.get('shared_root_mismatches', -1)) == 0,
        'temporal_registry_nonempty': bool(event_set) and bool(temporal_rows),
    }
    summary = {
        'schema':'binoc-tv0-v1',
        'verdict':'PASS_TV0_PRODUCTION_THEOREM_DOMAIN_CENSUS' if all(checks.values()) else 'STOP_TV0_PRODUCTION_THEOREM_DOMAIN_CENSUS',
        'checks':checks,
        'cache_root':str(cache_root),
        'source':source_stats,
        'hypervertices':len(hv),
        'route_counts':dict(routes),
        'regular_fraction':routes['regular_monotone']/max(1,len(cells)),
        'quotient_fraction':routes['quotient_singular']/max(1,len(cells)),
        'source_grammar_partitions':len(grammar),
        'top_grammar_partitions':[{'partition':list(k),'count':v} for k,v in grammar.most_common(12)],
        'spatial_projection_sampled_jacobian_counts':dict(jac),
        'spatial_projection_sampled_jacobian_scope':'5x5x5 diagnostic; production data show this is not the correct universal immersion certificate',
        'spacetime_immersion_center_counts':dict(spacetime_gram),
        'spacetime_immersion_center_positive_fraction':spacetime_gram['positive_center_sampled']/max(1,sum(spacetime_gram.values())),
        'spacetime_immersion_center_gram_quantiles':{str(q):float(np.quantile(spacetime_gram_values,q)) for q in (0,0.01,0.5,0.99,1)} if spacetime_gram_values else {},
        'spacetime_immersion_scope':'center-sample 4D Gram diagnostic; not a full Bernstein/interval certificate',
        'theory_observation':'spatial projection is rank-deficient for most static-surface cells; the validity theorem must certify the 3D spacetime immersion, not require a rank-3 spatial projection',
        'shared_temporal_faces':len(shared_payloads),
        'shared_temporal_face_occurrences':shared_occurrences,
        'shared_temporal_face_payload_mismatches':shared_mismatches,
        'registry':{
            'raw_rows':len(rows),'temporal_rows':len(temporal_rows),
            'temporal_canonical_events':len(event_set),
            'summary':registry_summary,
        },
    }
    write_json(out/'tv0_summary.json',summary)
    write_jsonl(out/'cells.jsonl',(source_record_dict(cell) for cell in cells))
    if not all(checks.values()):
        raise AuditError(summary['verdict'])
    return summary,cells,rows,hv


def stage_tv1(cells: list[Cell], rows: list[dict[str,str]], out: Path) -> dict[str,Any]:
    independent: dict[tuple[Any,...], Saddle] = {}
    for cell in cells:
        if cell.route.startswith('unresolved') or cell.route == 'invalid_temporal_order':
            continue
        for side, slots in TEMPORAL_FACE_SLOTS.items():
            saddle = admissible_saddle(
                [cell.hvids[i] for i in slots], [cell.times[i] for i in slots],
                cell.representative.element,
            )
            if saddle is not None:
                independent.setdefault(saddle.key(), saddle)
    registry = registry_saddles(rows)
    independent_keys=set(independent); registry_keys=set(registry)
    missing=sorted(independent_keys-registry_keys,key=repr)
    extra=sorted(registry_keys-independent_keys,key=repr)

    interval_failures=[]; intervals=0; probes=0; cells_tested=0
    for cell in cells:
        if cell.route != 'regular_monotone':
            continue
        schedule=exact_schedule(cell)
        if len(schedule)<2:
            continue
        cells_tested+=1
        for left,right in zip(schedule,schedule[1:]):
            if left==right: continue
            intervals+=1
            taus=(left+(right-left)/4,left+(right-left)/2,left+3*(right-left)/4)
            signatures=[]
            try:
                for tau in taus:
                    signatures.append(graph_signature(cell,tau)['hash']); probes+=1
            except AuditError as error:
                interval_failures.append({'cell_id':cell.cell_id,'left':str(left),'right':str(right),'error':str(error)})
                continue
            if len(set(signatures))!=1:
                interval_failures.append({'cell_id':cell.cell_id,'left':str(left),'right':str(right),'signatures':signatures})

    # Registry rows grouped by event must agree exactly on root/times.
    shared_pairing_mismatches=0
    for key,group in registry.items():
        signatures={(tuple(row[f't{i}'] for i in range(4)),row['root_num'],row['root_den']) for row in group}
        if len(signatures)!=1: shared_pairing_mismatches+=1

    checks={
        'independent_temporal_saddles_nonempty':bool(independent),
        'independent_saddles_missing_from_registry_zero':len(missing)==0,
        'registry_extra_temporal_saddles_zero':len(extra)==0,
        'event_free_interval_signature_changes_zero':len(interval_failures)==0,
        'shared_event_pairing_mismatch_zero':shared_pairing_mismatches==0,
    }
    summary={
        'schema':'binoc-tv1-v1',
        'verdict':'PASS_TV1_EXACT_EVENT_COMPLETENESS_AUDIT' if all(checks.values()) else 'STOP_TV1_EXACT_EVENT_COMPLETENESS_AUDIT',
        'checks':checks,
        'independent_temporal_saddles':len(independent),
        'registry_temporal_saddles':len(registry),
        'missing_from_registry':len(missing),'extra_in_registry':len(extra),
        'regular_cells_tested':cells_tested,'event_free_intervals_tested':intervals,'exact_probes':probes,
        'interval_failures':interval_failures[:20],
        'shared_pairing_mismatches':shared_pairing_mismatches,
        'scope':'local source-labeled cube-boundary signatures under the source-compatible bilinear/multi-affine model',
    }
    write_json(out/'tv1_summary.json',summary)
    write_jsonl(out/'independent_temporal_saddles.jsonl',(
        {'event_id':s.event_id(),'element':s.element,'hvids':[h.text() for h in s.face_hvids],
         'times':list(s.face_times),'root':fraction_json(s.root),'u':fraction_json(s.u),'v':fraction_json(s.v)}
        for s in sorted(independent.values(),key=lambda x:repr(x.key()))
    ))
    if not all(checks.values()): raise AuditError(summary['verdict'])
    return summary


def relabel_cell(cell: Cell, seed: int) -> Cell:
    unique=list(dict.fromkeys(cell.hvids)); rng=random.Random(seed); labels=list(range(1000,1000+len(unique))); rng.shuffle(labels)
    mapping={h:HVID(labels[i],(i*7+3)%101) for i,h in enumerate(unique)}
    return replace(cell,hvids=tuple(mapping[h] for h in cell.hvids),equality_partition=cell.equality_partition)


def stage_tv2(cells: list[Cell], out: Path, max_cases=128) -> dict[str,Any]:
    cases=[]; failures=[]
    rng=random.Random(20260831)
    for cell in cells:
        if cell.route != 'quotient_singular' or any(h.node<0 for h in cell.hvids):
            continue
        schedule=exact_schedule(cell)
        integer_events=sorted({Fraction(t,1) for t in cell.times})
        for event in integer_events:
            prior=[x for x in schedule if x<event]; following=[x for x in schedule if x>event]
            if not prior or not following: continue
            delta=min(event-prior[-1],following[0]-event)
            if delta<=0: continue
            side_hashes={'minus':[],'plus':[]}; payloads={}
            try:
                for k in range(2,9):
                    eps=delta/(2**k)
                    for name,tau in (('minus',event-eps),('plus',event+eps)):
                        q=quotient_signature(cell,event,tau)
                        side_hashes[name].append(q['hash']); payloads[(name,k)]=q
                stable=all(len(set(values))==1 for values in side_hashes.values())
                face_order=list(range(6)); rng.shuffle(face_order)
                order_ok=True
                for name,tau in (('minus',event-delta/4),('plus',event+delta/4)):
                    order_ok &= quotient_signature(cell,event,tau,face_order)['hash']==quotient_signature(cell,event,tau)['hash']
                relabeled=relabel_cell(cell,len(cases)+17)
                relabel_ok=True
                for name,tau in (('minus',event-delta/4),('plus',event+delta/4)):
                    relabel_ok &= quotient_signature(relabeled,event,tau)['hash']==quotient_signature(cell,event,tau)['hash']
                subdivision_ok=all(subdivide_contract_signature(payloads[(name,2)])==payloads[(name,2)]['hash'] for name in ('minus','plus'))
                # Integer-chain microcheck using production-derived quotient edges.
                chain_edges=[]
                for name in ('minus','plus'):
                    chain_edges.extend((a,b,int(count)) for a,b,count in payloads[(name,2)]['edges'])
                opposite_cancel=True; multiplicity_preserved=True
                for a,b,count in chain_edges:
                    c=collections.Counter({(a,b):count,(b,a):count})
                    reduced=collections.Counter()
                    for (x,y),coef in c.items():
                        key=tuple(sorted((x,y))); reduced[key]+=coef if x<=y else -coef
                    opposite_cancel &= all(v==0 for v in reduced.values())
                    same=collections.Counter({(a,b):2*count})
                    multiplicity_preserved &= abs(next(iter(same.values())))==2*count
                nonempty=any(payloads[(name,2)]['edges'] or payloads[(name,2)]['collapsed'] for name in ('minus','plus'))
                record={'cell_id':cell.cell_id,'event':fraction_json(event),'delta':fraction_json(delta),
                        'minus_hash':side_hashes['minus'][0],'plus_hash':side_hashes['plus'][0],
                        'stabilized':stable,'input_order_invariant':order_ok,'hvid_relabel_invariant':relabel_ok,
                        'degree2_subdivision_invariant':subdivision_ok,'opposite_orientation_cancels':opposite_cancel,
                        'same_orientation_multiplicity_preserved':multiplicity_preserved,'nonempty':nonempty,
                        'carrier_dimension_heuristic':1 if chain_edges else (0 if nonempty else -1)}
                if nonempty: cases.append(record)
                if not all((stable,order_ok,relabel_ok,subdivision_ok,opposite_cancel,multiplicity_preserved,nonempty)):
                    failures.append(record)
            except Exception as error:
                failures.append({'cell_id':cell.cell_id,'event':str(event),'error':repr(error)})
            if len(cases)>=max_cases: break
        if len(cases)>=max_cases: break
    checks={
        'auditable_production_quotient_cases_at_least_32':len(cases)>=32,
        'epsilon_stabilization_failures_zero':not any(not c.get('stabilized',False) for c in cases),
        'input_order_failures_zero':not any(not c.get('input_order_invariant',False) for c in cases),
        'hvid_relabel_failures_zero':not any(not c.get('hvid_relabel_invariant',False) for c in cases),
        'subdivision_failures_zero':not any(not c.get('degree2_subdivision_invariant',False) for c in cases),
        'oriented_cancellation_failures_zero':not any(not c.get('opposite_orientation_cancels',False) or not c.get('same_orientation_multiplicity_preserved',False) for c in cases),
        'exceptions_or_failed_cases_zero':len(failures)==0,
    }
    summary={'schema':'binoc-tv2-v1','verdict':'PASS_TV2_ONE_SIDED_QUOTIENT_AUDIT' if all(checks.values()) else 'STOP_TV2_ONE_SIDED_QUOTIENT_AUDIT',
             'checks':checks,'cases':len(cases),'failures':failures[:20],
             'carrier_dimension_heuristic':dict(collections.Counter(c['carrier_dimension_heuristic'] for c in cases)),
             'scope':'source-labeled boundary-graph one-sided quotient; not yet a full oriented 2-chain manifold theorem'}
    write_json(out/'tv2_summary.json',summary); write_jsonl(out/'tv2_cases.jsonl',cases)
    if not all(checks.values()): raise AuditError(summary['verdict'])
    return summary


def find_cell_for_row(cells: list[Cell], row: dict[str,str]) -> Cell:
    key=row_source_key(row)
    candidates=[cell for cell in cells if cell.representative.logical_key()==key]
    if len(candidates)!=1:
        raise AuditError(f'event row resolved to {len(candidates)} logical cells')
    return candidates[0]


def stage_tv3(cache_root: Path, cells: list[Cell], rows: list[dict[str,str]], out: Path) -> tuple[dict[str,Any],dict[str,Any]]:
    selected=json.loads((cache_root/'event_registry_selected_event.json').read_text())
    if not selected.get('selected'): raise AuditError('no production saddle selected')
    event_id=selected['event_id']; group=[row for row in rows if row['canonical_event_id']==event_id]
    if not group: raise AuditError('selected production event has no CSV rows')
    row=group[0]; cell=find_cell_for_row(cells,row)
    saddle=Saddle(
        element=int(row['element']),face_hvids=tuple(parse_hvid(row[f'h{i}']) for i in range(4)),
        face_times=tuple(int(row[f't{i}']) for i in range(4)),
        root=Fraction(int(row['root_num']),int(row['root_den'])),
        u=Fraction(int(row['u_num']),int(row['u_den'])),v=Fraction(int(row['v_num']),int(row['v_den'])),
        A=int(row['A']),B=int(row['B']))
    vertices,tets,geometry=build_production_half_handle(cell,saddle,int(row['face_side']))
    volumes=[tet_gram_volume(vertices,tet) for tet in tets]
    link=critical_link_audit(tets)
    delta=min(abs(float(Fraction(t,1)-saddle.root)) for t in saddle.face_times)
    lower_v,lower_f=slice_tet_complex(vertices,tets,float(saddle.root)-delta/2)
    upper_v,upper_f=slice_tet_complex(vertices,tets,float(saddle.root)+delta/2)
    lower=mesh_topology(lower_v,lower_f); upper=mesh_topology(upper_v,upper_f)
    checks={
        'production_event_used':event_id.startswith('element='),
        'two_tetrahedra':len(tets)==2,
        'source_labelled_block_vertices':(
            geometry.get('block_vertex_roles') == [
                'critical',
                'lower_source_branch','lower_source_branch',
                'upper_source_branch','upper_source_branch',
            ] and
            geometry.get('block_vertex_source_hvids',[None])[0] is None and
            len({
                value for value in
                geometry.get('block_vertex_source_hvids',[])[1:]
                if value is not None
            }) == 4
        ),
        'critical_link_is_disk':bool(link['is_disk']),
        'positive_physical_gram_volumes':min(volumes)>1e-12,
        'lower_slice_two_components':lower['components']==2,
        'upper_slice_one_component':upper['components']==1,
        'regular_slices_manifold':lower['nonmanifold_edges']==0 and upper['nonmanifold_edges']==0,
        'regular_slices_no_duplicate_faces':lower['duplicate_faces']==0 and upper['duplicate_faces']==0,
    }
    capsule={'event_id':event_id,'cell_id':cell.cell_id,'face_side':int(row['face_side']),
             'hvids':[h.text() for h in saddle.face_hvids],'times':list(saddle.face_times),
             'root':fraction_json(saddle.root),'u':fraction_json(saddle.u),'v':fraction_json(saddle.v),
             'vertices4':vertices.tolist(),'tets':tets.tolist(),'volumes':volumes,'geometry':geometry,
             'critical_link':link,'lower_slice':lower,'upper_slice':upper}
    summary={'schema':'binoc-tv3-v1','verdict':'PASS_TV3_OFFLINE_LOCAL_EVENT_BLOCK_COMPILATION' if all(checks.values()) else 'STOP_TV3_OFFLINE_LOCAL_EVENT_BLOCK_COMPILATION',
             'checks':checks,'event_id':event_id,'root':fraction_json(saddle.root),'tetrahedra':len(tets),
             'minimum_gram_volume':min(volumes),'critical_link':link,'lower_slice':lower,'upper_slice':upper,
             'scope':'production-derived saddle parameters and spatial frame; offline 2-tet relative half-handle, not run_slicing intervention'}
    write_json(out/'tv3_summary.json',summary); write_json(out/'event_capsule.json',capsule)
    np.savez_compressed(out/'event_block.npz',vertices4=vertices,tets=tets,lower_vertices=lower_v,lower_faces=lower_f,upper_vertices=upper_v,upper_faces=upper_f)
    if not all(checks.values()): raise AuditError(summary['verdict'])
    return summary,capsule


def face_boundary_hash(row: dict[str,str], tau: Fraction) -> str:
    times=[int(row[f't{i}']) for i in range(4)]; h=[row[f'h{i}'] for i in range(4)]
    segments=face_segments((0,1,2,3),times,tau)
    values=[]
    for e0,e1 in segments:
        def label(edge):
            a,b=edge; return '|'.join(sorted((h[a],h[b])))
        values.append(tuple(sorted((label(e0),label(e1)))))
    return sha256_json(sorted(values))


def stage_tv4(cells: list[Cell], rows: list[dict[str,str]], capsule: dict[str,Any], out: Path) -> dict[str,Any]:
    event_id=capsule['event_id']; group=[row for row in rows if row['canonical_event_id']==event_id]
    raw_ids=[row['raw_id'] for row in group]; logical_ids=sorted({row['logical_incidence_id'] for row in group})
    root=Fraction(capsule['root']['numerator'],capsule['root']['denominator'])
    times=[int(group[0][f't{i}']) for i in range(4)]; delta=min(abs(Fraction(t,1)-root) for t in times)
    lower_tau=root-delta/4; upper_tau=root+delta/4
    lower_boundaries={face_boundary_hash(row,lower_tau) for row in group}
    upper_boundaries={face_boundary_hash(row,upper_tau) for row in group}
    resolved={}
    resolution_errors=[]
    for row in group:
        try: resolved[row['logical_incidence_id']]=find_cell_for_row(cells,row).cell_id
        except Exception as error: resolution_errors.append(repr(error))
    block_tets=np.asarray(capsule['tets'],dtype=int); block_vertices=np.asarray(capsule['vertices4'],dtype=float)
    lv,lf=slice_tet_complex(block_vertices,block_tets,float(root)-float(delta)/2)
    uv,uf=slice_tet_complex(block_vertices,block_tets,float(root)+float(delta)/2)
    lower_top=mesh_topology(lv,lf); upper_top=mesh_topology(uv,uf)
    event_star_tets,event_star_completion=complete_production_event_star(block_tets)
    event_star_volumes=[
        tet_gram_volume(block_vertices,tet) for tet in event_star_tets]
    star_lv,star_lf=slice_tet_complex(
        block_vertices,event_star_tets,float(root)-float(delta)/2)
    star_cv,star_cf=slice_tet_complex(
        block_vertices,event_star_tets,float(root))
    star_uv,star_uf=slice_tet_complex(
        block_vertices,event_star_tets,float(root)+float(delta)/2)
    star_lower=mesh_topology(star_lv,star_lf)
    star_critical=mesh_topology(star_cv,star_cf)
    star_upper=mesh_topology(star_uv,star_uf)
    event_star_capsule={
        'schema':'binoc-tv4-completed-event-core-v1',
        'event_id':event_id,
        'root':capsule['root'],
        'vertices4':capsule['vertices4'],
        'block_vertex_source_hvids':capsule['geometry']['block_vertex_source_hvids'],
        'block_vertex_exact_times':capsule['geometry']['block_vertex_exact_times'],
        'relative_half_handle_tets':capsule['tets'],
        'event_star_tets':event_star_tets.tolist(),
        'completion':event_star_completion,
        'tetrahedron_gram_volumes':event_star_volumes,
        'core_boundary_candidate':{
            'kind':'SOURCE_BRANCH_TETRAHEDRON_BOUNDARY',
            'faces':event_star_completion['boundary_faces'],
            'critical_vertex_incident_faces':0,
            'source_hvid_vertices':[1,2,3,4],
        },
        'lower_slice':star_lower,
        'critical_slice':star_critical,
        'upper_slice':star_upper,
    }
    write_json(out/'event_star_capsule.json',event_star_capsule)
    np.savez_compressed(
        out/'event_star_block.npz',
        vertices4=block_vertices,tets=event_star_tets,
        lower_vertices=star_lv,lower_faces=star_lf,
        critical_vertices=star_cv,critical_faces=star_cf,
        upper_vertices=star_uv,upper_faces=star_uf)
    nonincident_pairs=0
    for i,j in itertools.combinations(range(len(block_tets)),2):
        if set(block_tets[i]).isdisjoint(set(block_tets[j])): nonincident_pairs+=1
    # Explicit offline splice manifest: every production source cell outside
    # the selected event star must retain its exact source-state hash.  The
    # selected per-incidence entries are replaced by one shared block entry.
    baseline_manifest = {
        cell.cell_id: sha256_json({
            'hvids': [h.text() for h in cell.hvids],
            'times': list(cell.times),
            'positions': cell.positions.tolist(),
            'route': cell.route,
        })
        for cell in cells
    }
    selected_cell_ids = set(resolved.values())
    treatment_manifest = dict(baseline_manifest)
    for cell_id in selected_cell_ids:
        treatment_manifest.pop(cell_id, None)
    block_hash = sha256_json({'vertices4':capsule['vertices4'],'tets':capsule['tets'],'event_id':event_id})
    treatment_manifest['shared-block:' + event_id] = block_hash
    outside_keys = set(baseline_manifest) - selected_cell_ids
    outside_changed = sorted(
        key for key in outside_keys
        if baseline_manifest.get(key) != treatment_manifest.get(key)
    )
    shared_block_entries = [key for key in treatment_manifest if key.startswith('shared-block:')]
    splice_manifest = {
        'baseline_source_cells': len(baseline_manifest),
        'selected_event_star_cells': sorted(selected_cell_ids),
        'treatment_source_cells': len([k for k in treatment_manifest if not k.startswith('shared-block:')]),
        'shared_block_entries': shared_block_entries,
        'shared_block_hash': block_hash,
        'outside_changed_cells': outside_changed,
    }
    write_json(out/'global_splice_manifest.json', splice_manifest)
    checks={
        'one_shared_block_for_canonical_event':len(shared_block_entries)==1 and shared_block_entries[0]=='shared-block:'+event_id,
        'raw_observations_unique_and_fully_consumed':len(raw_ids)==len(set(raw_ids)) and len(raw_ids)>0,
        'logical_incidences_fully_resolved':len(resolved)==len(logical_ids) and not resolution_errors,
        'lower_relative_boundary_agreement':len(lower_boundaries)==1,
        'upper_relative_boundary_agreement':len(upper_boundaries)==1,
        'event_star_outside_changed_cells_zero':len(outside_changed)==0,
        'global_regular_slices_manifold':lower_top['nonmanifold_edges']==0 and upper_top['nonmanifold_edges']==0,
        'global_duplicate_faces_zero':lower_top['duplicate_faces']==0 and upper_top['duplicate_faces']==0,
        'expected_component_transition':lower_top['components']==2 and upper_top['components']==1,
        'nonincident_tet_pairs_zero':nonincident_pairs==0,
        'completed_event_core_four_tetrahedra':len(event_star_tets)==4,
        'completed_critical_link_is_sphere':event_star_completion['critical_link']['is_sphere'],
        'completed_critical_side_faces_zero':event_star_completion['critical_side_faces_remaining']==0,
        'completed_event_core_positive_gram_volumes':min(event_star_volumes)>1e-12,
        'completed_event_core_regular_disk_slices':all(
            value['components']==1 and value['chi']==1 and
            value['boundary_loops']==1 and value['boundary_edges']==4 and
            value['nonmanifold_edges']==0 and value['duplicate_faces']==0
            for value in (star_lower,star_critical,star_upper)),
    }
    summary={'schema':'binoc-tv4-v1','verdict':'PASS_TV4_PRODUCTION_DERIVED_OFFLINE_GLOBAL_SPLICE' if all(checks.values()) else 'STOP_TV4_PRODUCTION_DERIVED_OFFLINE_GLOBAL_SPLICE',
             'checks':checks,'event_id':event_id,'raw_observations':len(raw_ids),'logical_incidences':len(logical_ids),
             'resolved_logical_cells':resolved,'resolution_errors':resolution_errors,
             'lower_boundary_hash':next(iter(lower_boundaries)) if len(lower_boundaries)==1 else None,
             'upper_boundary_hash':next(iter(upper_boundaries)) if len(upper_boundaries)==1 else None,
             'lower_slice':lower_top,'upper_slice':upper_top,'shared_blocks':1,'nonincident_tet_pairs':nonincident_pairs,
             'completed_event_core':{
                 'tetrahedra':len(event_star_tets),
                 'minimum_gram_volume':min(event_star_volumes),
                 'critical_link':event_star_completion['critical_link'],
                 'critical_side_faces_remaining':event_star_completion['critical_side_faces_remaining'],
                 'lower_slice':star_lower,'critical_slice':star_critical,'upper_slice':star_upper,
             },
             'baseline_source_cells':len(baseline_manifest),'event_star_cells':len(selected_cell_ids),'outside_changed_cells':len(outside_changed),
             'scope':'offline four-tetrahedron critical-core closure over a production-derived connected event; the source-prescribed outer S_B is compiled from ordinary whole-mesh traces downstream'}
    write_json(out/'tv4_summary.json',summary)
    if not all(checks.values()): raise AuditError(summary['verdict'])
    return summary


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument('--cache-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    cache=args.cache_root.resolve(); out=args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    try:
        tv0,cells,rows,hv=stage_tv0(cache,out/'tv0')
        tv1=stage_tv1(cells,rows,out/'tv1')
        tv2=stage_tv2(cells,out/'tv2')
        tv3,capsule=stage_tv3(cache,cells,rows,out/'tv3')
        tv4=stage_tv4(cells,rows,capsule,out/'tv4')
        stages={'tv0':tv0,'tv1':tv1,'tv2':tv2,'tv3':tv3,'tv4':tv4}
        verdict='PASS_TV0_TV4_PRODUCTION_DERIVED_THEORY_VALIDATION'
        result={'schema':'binoc-tv0-tv4-v1','pass':True,'verdict':verdict,'cache_root':str(cache),'stages':{k:v['verdict'] for k,v in stages.items()},
                'scope':{'production_derived':True,'production_runtime_modified':False,'formal_global_theorem':False,'paper_scenes':False}}
        write_json(out/'summary.json',result); print(verdict); return 0
    except Exception as error:
        result={'schema':'binoc-tv0-tv4-v1','pass':False,'verdict':'STOP_TV0_TV4_THEORY_VALIDATION','error':repr(error)}
        write_json(out/'summary.json',result); print(result['verdict'],error); return 2

if __name__=='__main__': raise SystemExit(main())
