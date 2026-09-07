"""Complete fixed-selector source/candidate domains for all Forest131 events.

Read-only source model extraction. No native library, runtime proposal, contact
solver or component admission is executed. Catalogue rows are shared between
the three singleton/two open cells, but every cell names its complete face
classes, owner replicas and vertex stars explicitly. Source AABBs are ideal
rational diagnostics and MUST NOT exclude actual binary32 interactions.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
from pathlib import Path
import resource
import time
import traceback

from forest_support_reader import read_event_candidates


def require(condition, reason):
    if not condition:raise ValueError(reason)


def frac(value):
    return F(value['numerator'],value['denominator']) if isinstance(value,dict) else F(value)


def fj(value):
    value=F(value);return {'numerator':value.numerator,'denominator':value.denominator}


def stable(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(stable(value)).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def scoped_vid(element,label):return str(element)+'/'+str(label)


def face_key(element,vertices):
    vertices=tuple(vertices)
    require(len(vertices)==3,'NONTRIANGULAR_ORDINARY_FACE')
    return element,min(vertices[i:]+vertices[:i] for i in range(3))


def cells_for_event(event):
    bounds=event['schedule']['bounds'];a,r,b=(F(bounds[k]) for k in ('lower','root','upper'))
    require(0<=a<r<b<=4,'UNSUPPORTED_FROZEN_FOREST_WINDOW')
    # This fixed cache has integer raw/expanded/load/group/HV-clamp predicates.
    # Refuse to use the three-point partition if any other integer is interior.
    require(not any(a<i<b for i in range(a.numerator//a.denominator,b.numerator//b.denominator+1)),
            'UNPARTITIONED_INTEGER_SOURCE_THRESHOLD')
    return [{'kind':'singleton','t0':fj(a),'t1':fj(a),'representative':fj(a)},
            {'kind':'open_interval','t0':fj(a),'t1':fj(r),'representative':fj((a+r)/2)},
            {'kind':'singleton','t0':fj(r),'t1':fj(r),'representative':fj(r)},
            {'kind':'open_interval','t0':fj(r),'t1':fj(b),'representative':fj((r+b)/2)},
            {'kind':'singleton','t0':fj(b),'t1':fj(b),'representative':fj(b)}]


def source_cell(source,cell):
    if source is None:return None
    if cell['kind']=='singleton':
        rows=[r for r in source['breakpoint_points'] if frac(r['time'])==frac(cell['t0'])]
    else:
        rows=[r for r in source['segments'] if frac(r['t0'])==frac(cell['t0']) and frac(r['t1'])==frac(cell['t1'])]
    require(len(rows)==1,'SOURCE_CELL_NOT_EXACTLY_FROZEN_PARTITION')
    return rows[0]


def ideal_bounds(source):
    if source is None:return None
    positions=[]
    for segment in source['segments']:
        for vertex in segment['boundary']:
            positions.extend(tuple(map(frac,vertex[k])) for k in ('position_t0','position_t1'))
    for point in source['breakpoint_points']:
        positions.extend(tuple(map(frac,v['position'])) for v in point['boundary'])
    positions.extend(tuple(map(frac,a['position'])) for a in source['anchors'].values())
    require(positions and all(len(p)==3 for p in positions),'INVALID_SOURCE_SPACE_SUPPORT')
    return {'coordinate_model':'ideal rational boundary and ideal interpolated center anchors',
        'minimum':[fj(min(p[k] for p in positions)) for k in range(3)],
        'maximum':[fj(max(p[k] for p in positions)) for k in range(3)],
        'proof':'Every declared affine segment reaches coordinate extrema at its endpoints; singleton positions and all center anchors are included.',
        'not_safe_for_actual32_exclusion':True,'actual_binary32_rounding_envelope':'NOT_PROVIDED'}


def stars(face_ids,catalogue,vertices):
    return {v:[i for i in face_ids if v in catalogue[i]['vids']] for v in sorted(vertices)}


def incidence_links(face_ids,catalogue,vertices):
    """All quotient corners, including repeated-ID degeneracies; never dropped."""
    links={v:[] for v in sorted(vertices)}
    for fid in face_ids:
        row=catalogue[fid]['vids']
        for corner,v in enumerate(row):
            if v in links:links[v].append([row[(corner+1)%3],row[(corner+2)%3],fid,corner])
    return links


def build_event(event,native,snapshots,dataset,bindings):
    eid=event['event_id'];element=event['element'];source=event['compiler'].get('source')
    owner_rows=[];owner_index={};record_rows=[];record_index={};face_rows=[];face_index={}
    def owner_id(owner,detail):
        record_key=tuple(detail['record_key'])
        if record_key not in record_index:
            record=dataset.records[record_key]
            require(tuple(record.provenance_values)==tuple(detail['provenance_values']),'OWNER_RECORD_PROVENANCE_DISAGREEMENT')
            record_index[record_key]=len(record_rows)
            record_rows.append({'key':list(record_key),'element':record.element,
                'original_times':list(record.times),'provenance_values':list(record.provenance_values),
                'primary':{'path':str(Path(record.primary_path).relative_to(dataset.cache)),
                    'offset':record.primary_offset,'size':record.primary_size,'sha256':record.primary_sha256},
                'metadata':{'path':str(Path(record.metadata_path).relative_to(dataset.cache)),
                    'offset':record.metadata_offset,'size':132,'sha256':record.metadata_sha256}})
        row={'reference':list(owner),'record_id':record_index[record_key],
             'ordered_raw_vids':[[*pair[0],*pair[1]] for pair in detail['ordered_raw_vids']]}
        if owner not in owner_index:
            owner_index[owner]=len(owner_rows);owner_rows.append(row)
        else:require(owner_rows[owner_index[owner]]==row,'ONE_RAW_OWNER_MULTIPLE_SERIALIZED_OCCURRENCES')
        return owner_index[owner]
    def classes(triangles,details):
        groups=defaultdict(list);seen=set()
        for triangle in triangles:
            owner=tuple(triangle.reference.values());require(owner not in seen,'DUPLICATE_SNAPSHOT_OWNER');seen.add(owner)
            key=face_key(triangle.reference.element,tuple(scoped_vid(triangle.reference.element,v.text()) for v in triangle.source_vertices))
            groups[key].append(owner_id(owner,details[owner]))
        ids=[]
        for (ele,vids),owners in sorted(groups.items()):
            signature=(ele,vids,tuple(sorted(owners)))
            if signature not in face_index:
                face_index[signature]=len(face_rows)
                face_rows.append({'element':ele,'vids':list(vids),'owner_ids':list(signature[2]),
                                  'has_repeated_vid':len(set(vids))!=3})
            ids.append(face_index[signature])
        return ids
    cells=[];candidate_union=set();source_owners=set();source_vids=set();retained_union=set()
    for cell in cells_for_event(event):
        representative=frac(cell['representative']);snapshot=snapshots[eid,representative]
        require(snapshot.event_id==eid and snapshot.tau==representative,'SNAPSHOT_ID_TIME_MISMATCH')
        require(snapshot.event_candidates_complete and snapshot.halo_complete,'INCOMPLETE_CANDIDATE_VHALO')
        details=snapshot.details_by_owner
        legacy=classes(snapshot.raw_triangles,details);actual=classes(snapshot.actual_raw_triangles,details)
        candidate_legacy=[];candidate_actual=[]
        actual_owners={tuple(t.reference.values()) for t in snapshot.actual_raw_triangles}
        for triangle in snapshot.raw_triangles:
            owner=tuple(triangle.reference.values())
            require(bool(details[owner]['actual_raw_emitted'])==(owner in actual_owners),'RAW_EMISSION_LAYER_DISAGREEMENT')
            if triangle.event_record:
                candidate_legacy.append(owner_index[owner])
                if owner in actual_owners:candidate_actual.append(owner_index[owner])
        candidate_vertices={scoped_vid(element,v.text()) for v in snapshot.halo_vertex_ids}
        require(candidate_legacy and candidate_vertices,'EMPTY_FIXED_CANDIDATE_DOMAIN')
        candidate_union.update(candidate_vertices)
        support=source_cell(source,cell);selected=None;selected_owner_ids=None;boundary=None
        if support is not None:
            require(source['partition_proof']['all_thresholds_integer'] is True,'SOURCE_PARTITION_PROOF_MISSING')
            boundary=[scoped_vid(element,v) for v in support['boundary_cycle']]
            require(set(boundary)<=candidate_vertices,'SOURCE_BOUNDARY_OUTSIDE_COMPLETE_CANDIDATE_VHALO')
            expected_owners={tuple(o) for o in support['owners']}
            require(expected_owners<=actual_owners,'SELECTED_OWNER_NOT_ACTUAL_RAW_EMITTED')
            selected_owner_ids={owner_index[o] for o in expected_owners}
            selected_keys={face_key(element,tuple(scoped_vid(element,v) for v in face)) for face in support['source_faces']}
            selected=[i for i in actual if face_key(face_rows[i]['element'],face_rows[i]['vids']) in selected_keys]
            require(len(selected)==2 and {o for i in selected for o in face_rows[i]['owner_ids']}==selected_owner_ids,
                    'SOURCE_FACE_FULL_OWNER_CLASS_DISAGREEMENT')
            source_owners.update(selected_owner_ids);source_vids.update(boundary)
        retained=[i for i in actual if selected is None or i not in selected]
        retained_union.update(v for i in retained for v in face_rows[i]['vids'])
        row={**cell,'event_candidates_complete':True,'candidate_vertex_halo_complete':True,
            'snapshot_counts':snapshot.summary(),'legacy_halo_face_ids':legacy,'actual_halo_face_ids':actual,
            'retained_face_ids':retained,'candidate_legacy_owner_ids':sorted(candidate_legacy),
            'candidate_actual_owner_ids':sorted(candidate_actual),'candidate_vids':sorted(candidate_vertices),
            'source_owner_ids':None if selected_owner_ids is None else sorted(selected_owner_ids),
            'source_face_ids':selected,'source_boundary_vids':boundary,
            'candidate_vertex_stars_actual':stars(actual,face_rows,candidate_vertices),
            'candidate_vertex_stars_legacy':stars(legacy,face_rows,candidate_vertices),
            'source_boundary_retained_stars':None if boundary is None else stars(retained,face_rows,boundary),
            'source_boundary_retained_link_corners':None if boundary is None else incidence_links(retained,face_rows,boundary),
            'native_legacy_identity_model_disagreements':sum(not d['native_legacy_identity_equal'] for d in details.values()),
            'actual_cpp_identity_coverage':'NOT_CLAIMED'}
        cells.append(row)
    if source is None:
        require(event['compiler']['reason'].startswith('LEGACY_EXHAUSTIVE_SELECTOR_NO_CANDIDATE:'),'UNEXPECTED_UNKNOWN_SOURCE_PROFILE')
    return {'event_id':eid,'element':element,'root':event['root'],'window':event['schedule']['bounds'],
        'bindings':bindings,'source_contract_sha256':None if source is None else digest(source),
        'source_status':'CANDIDATE_DOMAIN_ONLY' if source is None else 'COMPLETE_FIXED_COMPILED_SOURCE_MODEL',
        'future_general_joint_closure_coverage':'NOT_CLAIMED',
        'native_individual_decision':native['decision'],'native_individual_runtime_status':native.get('runtime',{}).get('status'),
        'record_catalog':record_rows,'owner_catalog':owner_rows,'face_class_catalog':face_rows,'cells':cells,
        'union':{'candidate_vids':sorted(candidate_union),'source_boundary_vids':None if source is None else sorted(source_vids),
                 'source_owner_ids':None if source is None else sorted(source_owners),'retained_halo_vids':sorted(retained_union)},
        'ideal_affine_patch_aabb':ideal_bounds(source),
        'critical_position_diagnostic':event['kernel']['critical_position'],
        'space_support_for_actual32_exclusion':'UNKNOWN_PENDING_ACTUAL_QUERY_BINDING',
        'temporal_partition_basis':{'all_predicate_thresholds_integer':True,'closed_window_internal_integer_thresholds':[],
            'meaning':'Open-cell source incidence follows the integer threshold predicate model, not sampled stability.',
            'actual_binary32_time_or_identity_equivalence':'NOT_CLAIMED'},
        'coverage_scope':'Complete fixed-selector event-record candidates and same-element candidate vertex stars; not all-scene geometry.'}


def inventory_cell(event,tau):
    """Return the certified source-predicate cell containing an actual query."""
    tau=F(tau)
    require(event['temporal_partition_basis']['all_predicate_thresholds_integer'] is True,'UNVERIFIED_SOURCE_PARTITION')
    for cell in event['cells']:
        a,b=frac(cell['t0']),frac(cell['t1'])
        if cell['kind']=='singleton' and tau==a==b:return cell
        if cell['kind']=='open_interval' and a<tau<b:return cell
    raise ValueError('QUERY_OUTSIDE_INVENTORIED_SOURCE_WINDOW')


def candidate_actual_owners(event,tau):
    """Complete event-record actual-emission owner7s; no native ID inference."""
    cell=inventory_cell(event,tau)
    require(cell['event_candidates_complete'] and cell['candidate_vertex_halo_complete'],'INCOMPLETE_CANDIDATE_CELL')
    rows=tuple(tuple(event['owner_catalog'][i]['reference']) for i in cell['candidate_actual_owner_ids'])
    require(rows and len(set(rows))==len(rows) and all(len(o)==7 and o[0]==event['element'] for o in rows),
            'EMPTY_DUPLICATE_OR_WRONG_ELEMENT_CANDIDATE_OWNERS')
    return rows


def _symbolic_cell_domain(event,cell):
    catalogue=event['face_class_catalog']
    if cell['source_face_ids'] is None:
        candidates=set(cell['candidate_legacy_owner_ids'])
        selected=[i for i in cell['legacy_halo_face_ids'] if candidates.intersection(catalogue[i]['owner_ids'])]
        vids=set(cell['candidate_vids'])
        star_ids={i for ids in cell['candidate_vertex_stars_actual'].values() for i in ids}
    else:
        selected=cell['source_face_ids'];vids=set(cell['source_boundary_vids'])
        star_ids={i for ids in cell['source_boundary_retained_stars'].values() for i in ids}
    require(selected and vids,'EMPTY_SYMBOLIC_SUPPORT_DOMAIN')
    faces=lambda ids:{tuple(catalogue[i]['vids']) for i in ids}
    return {'owners':{tuple(event['owner_catalog'][o]['reference']) for i in selected for o in catalogue[i]['owner_ids']},
            'faces':faces(selected),'vids':vids,'retained_interface_faces':faces(star_ids)}


def source_symbolic_edges(events):
    """Conservative identity dependencies over ALL five certified source cells.

    Candidate-only nodes reserve the complete legacy candidate face domain;
    this is not a source patch and not a future arbitrary-closure bound.
    Ideal AABBs are intentionally never read by this adapter.
    """
    groups=defaultdict(list);domains={}
    for event in events:
        groups[event['root']].append(event)
        domains[event['event_id']]=[_symbolic_cell_domain(event,c) for c in event['cells']]
    result=[]
    for root,group in sorted(groups.items()):
        for left,right in combinations(sorted(group,key=lambda e:e['event_id']),2):
            require([(c['kind'],c['t0'],c['t1']) for c in left['cells']]==
                    [(c['kind'],c['t0'],c['t1']) for c in right['cells']], 'ROOT_SOURCE_CELL_PARTITIONS_DIFFER')
            for index,cell in enumerate(left['cells']):
                a=domains[left['event_id']][index];b=domains[right['event_id']][index]
                overlaps={'SOURCE_CELL_RAW_OWNER_OVERLAP':a['owners']&b['owners'],
                    'SOURCE_CELL_SOURCE_OR_CANDIDATE_FACE_OVERLAP':a['faces']&b['faces'],
                    'SOURCE_CELL_BOUNDARY_OR_CANDIDATE_VID_OVERLAP':a['vids']&b['vids'],
                    'SOURCE_CELL_RETAINED_INTERFACE_FACE_OVERLAP':a['retained_interface_faces']&b['retained_interface_faces'],
                    'SOURCE_CELL_SOURCE_RETAINED_INTERFACE_DEPENDENCY':
                        (a['faces']&b['retained_interface_faces'])|(b['faces']&a['retained_interface_faces'])}
                for reason,shared in overlaps.items():
                    if shared:result.append({'events':[left['event_id'],right['event_id']],'reason':reason,
                        'cell':{'root':root,'kind':cell['kind'],'t0':str(frac(cell['t0'])),'t1':str(frac(cell['t1'])),
                            'shared_count':len(shared),'first_shared':sorted(shared)[:4],
                            'source_statuses':[left['source_status'],right['source_status']]}})
    return result


def load_inputs(attempt,native):
    old_verification=json.loads((attempt/'input_verification.json').read_text())
    require(old_verification['pass'],'OLD_SOURCE_INPUT_VERIFICATION_FAILED')
    index=json.loads((attempt/'events_index.json').read_text());ni=json.loads((native/'events_index.json').read_text())
    require(len(index)==len(ni)==131 and {r['event_id'] for r in index}=={r['event_id'] for r in ni},'FIXED_131_IDENTITY_SET_MISMATCH')
    native_index={r['event_id']:r for r in ni};events={};bindings={}
    paths=[attempt/x for x in ('summary.json','events_index.json','input_verification.json')]+[native/x for x in ('summary.json','events_index.json')]
    for row in index:
        eid=row['event_id'];nr=native_index[eid];sp=attempt/row['artifact']['path'];np=native/nr['artifact']['path']
        require(file_sha(sp)==row['artifact']['sha256'] and file_sha(np)==nr['artifact']['sha256'],'FROZEN_EVENT_HASH_MISMATCH')
        source_event=json.loads(sp.read_text());native_event=json.loads(np.read_text())
        require(source_event['event_id']==native_event['event_id']==eid,'FROZEN_EVENT_ID_MISMATCH')
        require(digest(source_event['compiler'])==digest(native_event['compiler']),'NATIVE_SOURCE_COMPILER_BINDING_DISAGREEMENT')
        events[eid]=(source_event,native_event);bindings[eid]={'source_artifact':row['artifact'],'native_artifact':nr['artifact']};paths.extend((sp,np))
    return events,bindings,old_verification,{str(p):file_sha(p) for p in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--source-attempt',type=Path,required=True)
    parser.add_argument('--native-attempt',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();cpu=time.process_time()
    require(not args.output.exists(),'OUTPUT_EXISTS');require(args.cache.resolve() not in args.output.resolve().parents,'OUTPUT_INSIDE_ORIGINAL_CACHE')
    events,bindings,prior,files=load_inputs(args.source_attempt,args.native_attempt)
    sources={str(Path(__file__).resolve()):file_sha(__file__),**prior['executed_source_sha256']}
    require(all(file_sha(p)==h for p,h in sources.items()),'FROZEN_SUPPORT_READER_SOURCE_CHANGED')
    report={'schema':'forest-root-support-inventory-v1','status':'STOP_NOT_A_METHOD_DECISION',
        'scope':'All131 fixed-selector source/candidate domains; no native/atomic graph or geometry admission.',
        'frozen_artifact_sha256':files,'executed_sources_sha256':sources,'events':[]}
    try:
        dataset=read_event_candidates(args.cache,seconds=600)
        require(set(dataset.event_ids)==set(events),'REGISTRY_POPULATION_CHANGED')
        require(dataset.groups==2 and dataset.maximum==4,'UNSUPPORTED_INTEGER_THRESHOLD_PROFILE')
        requests={eid:[frac(c['representative']) for c in cells_for_event(event)] for eid,(event,native) in events.items()}
        print(json.dumps({'stage':'all131_candidates_read','seconds':time.monotonic()-start}),flush=True)
        snapshots=dataset.load_halos(requests)
        for eid in dataset.event_ids:
            dataset.reader.check();event,native=events[eid]
            report['events'].append(build_event(event,native,snapshots,dataset,bindings[eid]))
        verification=dataset.verify_inputs()
        require(verification['input_full_sha256']==prior['input_full_sha256'],'FULL_INPUT_HASH_SET_DIFFERS_FROM_FROZEN_SOURCE_ATTEMPT')
        require(all(file_sha(p)==h for p,h in files.items()),'FROZEN_ARTIFACT_CHANGED_DURING_EXTRACTION')
        require(all(file_sha(p)==h for p,h in sources.items()),'EXECUTED_SOURCE_CHANGED_DURING_EXTRACTION')
        report.update(status='PASS_COMPLETE_FIXED_SOURCE_CANDIDATE_INVENTORY',event_count=len(events),
            symbolic_edges=source_symbolic_edges(report['events']),
            source_status_counts=dict(Counter(e['source_status'] for e in report['events'])),
            root_counts=dict(Counter(e['root'] for e in report['events'])),
            native_individual_decisions=dict(Counter(e['native_individual_decision'] for e in report['events'])),
            actual_binary32_complete_support_bound_proven=False,input_verification=verification,
            full_input_hash_set_equal_frozen=True,reader=dataset.summary())
    except Exception as error:
        report.update(reason=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    report['cost']={'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
                    'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    data=json.dumps(report,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n'
    require(len(data.encode())<=10*1024*1024,'SUPPORT_INVENTORY_EXCEEDS_10_MIB')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:stream.write(data)
    print(json.dumps({'status':report['status'],'bytes':len(data.encode()),'cost':report['cost'],'output':str(args.output)}),flush=True)
    return 0 if report['status']=='PASS_COMPLETE_FIXED_SOURCE_CANDIDATE_INVENTORY' else 2


if __name__=='__main__':raise SystemExit(main())
