"""Final approved resource-only source entry, copied from frozen scene_source.

No native calls. A source PASS is only readiness for the later actual-identity,
geometry, component and visibility gates. All registry identities are retained.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
C1 = HERE.parent/'c1_lite'
if str(C1) not in sys.path: sys.path.insert(0, str(C1))
import forest_support_reader as reader_api
import forest_source_compiler as compiler_api
import forest_source_support_inventory as inventory_api
from forest_requested_source_gates import audit_requested_source
from forest_schedule_policy import make_schedule
from run_forest_formal import classify_compiler, reader_binding
from screen_contracts import load_protocol, read
from final_gate_resources import (validated_budget, amendment_binding, amendment_inputs,
    validate_build_budget, verify_bindings, require_remaining)


def require(value, reason):
    if not value: raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def document(value):
    if isinstance(value,dict): return value,{'kind':'CANONICAL_JSON','sha256':digest(value)}
    path=Path(value).resolve();return json.loads(path.read_text()),{'path':str(path),'sha256':file_sha(path)}


def validate_profile(camera,spec):
    """Only the preregistered 64-frame/24FPS/6px source-time profile."""
    for key in ('segment_id','scene','first_frame','last_frame','fps','pixels_per_cube'):
        require(key in spec,'MISSING_SEGMENT_SPEC:'+key)
    first,last=spec['first_frame'],spec['last_frame']
    require(type(first) is int and type(last) is int and first>=1 and last-first+1==64,
            'UNSUPPORTED_FIXED_64_FRAME_PROFILE')
    require(spec['fps']==24 and spec['pixels_per_cube']==6,'UNSUPPORTED_FIXED_24FPS_6PX_PROFILE')
    times=camera['times_seconds']; require(len(times)==64,'CAMERA_FRAME_COUNT_MISMATCH')
    require(all(math.isfinite(float(t)) and float(t).hex()==float((first+i-0.5)/24).hex()
                for i,t in enumerate(times)),'ORIGINAL_ABSOLUTE_CAMERA_TIME_GRID_MISMATCH')
    require(all(len(camera[k])==64 for k in ('poses','intrinsics','widths','heights')),'CAMERA_ARRAY_LENGTH_MISMATCH')
    require(all(x==960 for x in camera['widths']) and all(x==540 for x in camera['heights']), 'CAMERA_RESOLUTION_MISMATCH')
    origin=float(times[0]);relative=[float(float(t)-origin) for t in times];duration=max(relative)+1e-5
    mapping=camera['time_mapping'];fr=compiler_api.fr
    require(mapping['use_alignment'] is False and mapping['min_t_offset']==0,'UNSUPPORTED_TIME_ALIGNMENT')
    require(float(fr(mapping['origin_seconds'])).hex()==origin.hex() and
            float(fr(mapping['duration_seconds'])).hex()==duration.hex(),'CAMERA_TIME_MAPPING_BITS_MISMATCH')
    fading=float(mapping['effective_fading_seconds']);level=0
    require(fading==1.0,'UNSUPPORTED_FIXED_FADING_TIME')
    while fading*(1<<(level+1))<duration:level+=1
    groups=1<<level;maximum=2*groups;delta=duration/maximum
    require((groups,maximum)==(2,4) and mapping['temporal_group_count']==groups and
            mapping['maximum_discrete_time']==maximum and
            float(fr(mapping['delta_seconds'])).hex()==delta.hex(),'CAMERA_GROUP_MAXIMUM_DELTA_MISMATCH')
    return {'group_count':groups,'maximum_discrete_time':maximum,'expected_delta_t_hex':delta.hex(),
        'origin_seconds_hex':origin.hex(),'duration_seconds_hex':duration.hex(),
        'actual_native_initialization_verified':False,
        'basis':'Executed camera binary64 inputs and frozen constructor arithmetic; native must independently bind actual delta/group count.'}


def cells_for_bounds(bounds,maximum=4):
    a,r,b=(F(bounds[k]) for k in ('lower','root','upper'))
    require(0<=a<r<b<=maximum,'FIXED_WINDOW_OUTSIDE_CACHE')
    points=sorted({a,r,b,*(F(i) for i in range(a.numerator//a.denominator,b.numerator//b.denominator+1) if a<=i<=b)})
    cells=[]
    for i,t in enumerate(points):
        cells.append({'kind':'singleton','t0':inventory_api.fj(t),'t1':inventory_api.fj(t),'representative':inventory_api.fj(t)})
        if i+1<len(points):
            u=points[i+1];cells.append({'kind':'open_interval','t0':inventory_api.fj(t),'t1':inventory_api.fj(u),
                                      'representative':inventory_api.fj((t+u)/2)})
    return cells


def kernel_domain_witness(rows,hypervertices,kernel):
    """Only recognize a independently checked frozen 2-lower/2-upper limit.

    Tangent degeneracy or any unrecognized compiler error is NOT silently
    converted to an unsupported-method case: it remains a harness STOP.
    """
    reason=kernel.get('reason','')
    if reason!='AuditError: saddle does not have two lower and two upper branches':return None
    row=rows[0];root=F(int(row['root_num']),int(row['root_den']))
    ids=[reader_api.HVID(*map(int,row[f'h{i}'].split(':'))) for i in range(4)]
    times=[F(hypervertices[h].time) for h in ids]
    counts={'lower':sum(t<root for t in times),'equal':sum(t==root for t in times),'upper':sum(t>root for t in times)}
    require((counts['lower'],counts['upper'])!=(2,2),'KERNEL_ERROR_NOT_REPRODUCED_BY_DOMAIN_WITNESS')
    return {'reason_code':'FROZEN_KERNEL_REQUIRES_TWO_LOWER_TWO_UPPER','corner_times':[str(t) for t in times],
        'root':str(root),'branch_counts':counts,'compiler_reason':reason,
        'scope':'Unsupported frozen construction family, not proof that the event is geometrically unrepairable.'}


def build_inventory_event(event,snapshots,dataset):
    """Same catalog semantics as frozen inventory; only partition wiring varies."""
    eid=event['event_id'];element=event['element'];source=event['compiler'].get('source')
    records=[];record_index={};owners=[];owner_index={};faces=[];face_index={}
    def own(owner,detail):
        key=tuple(detail['record_key'])
        if key not in record_index:
            record=dataset.records[key]
            require(tuple(record.provenance_values)==tuple(detail['provenance_values']),'OWNER_RECORD_PROVENANCE_DISAGREEMENT')
            record_index[key]=len(records)
            records.append({'key':list(key),'element':record.element,'original_times':list(record.times),
                'provenance_values':list(record.provenance_values),
                'primary':{'path':str(Path(record.primary_path).relative_to(dataset.cache)), 'offset':record.primary_offset,
                    'size':record.primary_size,'sha256':record.primary_sha256},
                'metadata':{'path':str(Path(record.metadata_path).relative_to(dataset.cache)), 'offset':record.metadata_offset,
                    'size':132,'sha256':record.metadata_sha256}})
        row={'reference':list(owner),'record_id':record_index[key],
             'ordered_raw_vids':[[*pair[0],*pair[1]] for pair in detail['ordered_raw_vids']]}
        if owner not in owner_index:owner_index[owner]=len(owners);owners.append(row)
        else:require(owners[owner_index[owner]]==row,'ONE_OWNER_MULTIPLE_RAW_OCCURRENCES')
        return owner_index[owner]
    def quotient(triangles,details):
        groups=defaultdict(list);seen=set()
        for triangle in triangles:
            owner=tuple(triangle.reference.values());require(owner not in seen,'DUPLICATED_SNAPSHOT_OWNER');seen.add(owner)
            key=inventory_api.face_key(triangle.reference.element,tuple(inventory_api.scoped_vid(triangle.reference.element,v.text()) for v in triangle.source_vertices))
            groups[key].append(own(owner,details[owner]))
        result=[]
        for (ele,vids),ids in sorted(groups.items()):
            signature=(ele,vids,tuple(sorted(ids)))
            if signature not in face_index:
                face_index[signature]=len(faces);faces.append({'element':ele,'vids':list(vids),
                    'owner_ids':list(signature[2]),'has_repeated_vid':len(set(vids))!=3})
            result.append(face_index[signature])
        return result
    cells=[]
    for cell in cells_for_bounds(event['schedule']['bounds'],dataset.maximum):
        tau=compiler_api.fr(cell['representative']);snapshot=snapshots[eid,tau]
        require(snapshot.event_id==eid and snapshot.tau==tau and snapshot.event_candidates_complete and snapshot.halo_complete,
                'INCOMPLETE_OR_MISBOUND_CANDIDATE_HALO')
        details=snapshot.details_by_owner;legacy=quotient(snapshot.raw_triangles,details);actual=quotient(snapshot.actual_raw_triangles,details)
        actual_owners={tuple(t.reference.values()) for t in snapshot.actual_raw_triangles}
        candidate_legacy=[];candidate_actual=[]
        for tri in snapshot.raw_triangles:
            owner=tuple(tri.reference.values())
            require(bool(details[owner]['actual_raw_emitted'])==(owner in actual_owners),'RAW_EMISSION_LAYER_DISAGREEMENT')
            if tri.event_record:
                candidate_legacy.append(owner_index[owner])
                if owner in actual_owners:candidate_actual.append(owner_index[owner])
        candidate_vids={inventory_api.scoped_vid(element,v.text()) for v in snapshot.halo_vertex_ids}
        selected=None;selected_owners=None;boundary=None
        if source is not None:
            support=inventory_api.source_cell(source,cell)
            boundary=[inventory_api.scoped_vid(element,v) for v in support['boundary_cycle']]
            expected={tuple(o) for o in support['owners']}
            require(set(boundary)<=candidate_vids and expected<=actual_owners,'SOURCE_OUTSIDE_COMPLETE_ACTUAL_CANDIDATES')
            selected_owners={owner_index[o] for o in expected}
            keys={inventory_api.face_key(element,tuple(inventory_api.scoped_vid(element,v) for v in face)) for face in support['source_faces']}
            selected=[i for i in actual if inventory_api.face_key(faces[i]['element'],faces[i]['vids']) in keys]
            require(len(selected)==2 and {o for i in selected for o in faces[i]['owner_ids']}==selected_owners,'PARTIAL_SOURCE_OWNER_CLASS')
        retained=[i for i in actual if selected is None or i not in selected]
        cells.append({**cell,'event_candidates_complete':True,'candidate_vertex_halo_complete':True,
            'snapshot_counts':snapshot.summary(),'legacy_halo_face_ids':legacy,'actual_halo_face_ids':actual,
            'retained_face_ids':retained,'candidate_legacy_owner_ids':sorted(candidate_legacy),
            'candidate_actual_owner_ids':sorted(candidate_actual),'candidate_vids':sorted(candidate_vids),
            'source_owner_ids':None if selected_owners is None else sorted(selected_owners),
            'source_face_ids':selected,'source_boundary_vids':boundary,
            'candidate_vertex_stars_actual':inventory_api.stars(actual,faces,candidate_vids),
            'candidate_vertex_stars_legacy':inventory_api.stars(legacy,faces,candidate_vids),
            'source_boundary_retained_stars':None if boundary is None else inventory_api.stars(retained,faces,boundary),
            'source_boundary_retained_link_corners':None if boundary is None else inventory_api.incidence_links(retained,faces,boundary),
            'empty_candidate_domain_verified':not candidate_legacy,
            'empty_domain_scope':'Exhaustive associated-record candidates at this cell only; not any future source closure.',
            'actual_cpp_identity_coverage':'NOT_CLAIMED',
            'native_legacy_identity_model_disagreements':sum(not d['native_legacy_identity_equal'] for d in details.values())})
    return {'event_id':eid,'namespace_id':event['namespace_id'],'element':element,'root':event['root'],
        'window':event['schedule']['bounds'],'source_contract_sha256':None if source is None else digest(source),
        'source_status':'CANDIDATE_DOMAIN_ONLY' if source is None else 'COMPLETE_FIXED_COMPILED_SOURCE_MODEL',
        'source_stage_status':event['source_status'],'native_individual_decision':event['decision'],
        'native_individual_runtime_status':'NOT_ATTEMPTED','record_catalog':records,'owner_catalog':owners,
        'face_class_catalog':faces,'cells':cells,'candidate_domain_complete':True,
        'ideal_affine_patch_aabb':inventory_api.ideal_bounds(source),'critical_position_diagnostic':event['kernel'].get('critical_position'),
        'temporal_partition_basis':{'all_predicate_thresholds_integer':True,
            'closed_window_internal_integer_thresholds':[str(compiler_api.fr(c['t0'])) for c in cells if c['kind']=='singleton' and
                compiler_api.fr(c['t0']).denominator==1],
            'actual_binary32_time_or_identity_equivalence':'NOT_CLAIMED'},
        'coverage_scope':'Complete fixed-selector associated-record candidate vertex halo, not entire scene or future arbitrary closure.',
        'space_support_for_actual32_exclusion':'UNKNOWN_PENDING_ACTUAL_QUERY_BINDING'}


def symbolic_dependencies(events):
    groups=defaultdict(list);result=[];heterogeneous=[]
    for event in events:groups[event['root']].append(event)
    for root,group in sorted(groups.items()):
        signatures=[[(c['kind'],c['t0'],c['t1']) for c in e['cells']] for e in group]
        empty=any(not c['candidate_legacy_owner_ids'] for e in group for c in e['cells'])
        if any(s!=signatures[0] for s in signatures) or empty:
            reason='HETEROGENEOUS_ROOT_SOURCE_PARTITION_CONSERVATIVE_JOIN' if not empty else 'EMPTY_CURRENT_CANDIDATE_DOMAIN_CONSERVATIVE_JOIN'
            heterogeneous.append({'root':root,'reason_code':reason,'event_ids':sorted(e['event_id'] for e in group),
                'runtime_component_coverage':'UNRESOLVED_DO_NOT_BORROW_FIRST_EVENT_SCHEDULE'})
            for i,a in enumerate(sorted(e['event_id'] for e in group)):
                for b in sorted(e['event_id'] for e in group)[i+1:]:result.append({'events':[a,b],'reason':reason,'cell':{'root':root}})
        else:result.extend(inventory_api.source_symbolic_edges(group))
    return result,heterogeneous


def base_event(eid,root,element,spec):
    return {'event_id':eid,'namespace_id':spec['segment_id']+'/OpaqueTerrain/'+eid,'root':str(root),'element':element,
        'decision':'UNKNOWN','source_status':'NOT_REACHED','runtime_attempted':False,'runtime':{'status':'NOT_ATTEMPTED'},
        'kernel':{'status':'NOT_REACHED'},'compiler':{'status':'NOT_REACHED','source':None}}


def write_new(path,value,maximum=32*1024*1024,*,budget=None):
    data=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    size=len(data.encode());require(size<=maximum,'SOURCE_ARTIFACT_BUDGET_EXCEEDED')
    if budget is not None:
        require(size<=budget['remaining'],'SOURCE_STAGE_TOTAL_OUTPUT_BUDGET_EXCEEDED')
        budget['remaining']-=size
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(data)
    return {'path':str(path),'sha256':file_sha(path),'bytes':len(data.encode())}


def run_source_stage(cache_root,camera_document,segment_spec,output_root,*,seconds=1200,
                     budget_amendment,protocol,seal,build_report):
    """Fresh report directory; returns summary + compatible events + inventory.

    Any unexplained helper UNKNOWN/exception stops the stage. All known IDs are
    still present with unprocessed rows NOT_REACHED, never dropped or admitted.
    """
    camera,cb=document(camera_document);spec,sb=document(segment_spec)
    receipt=validated_budget(budget_amendment,protocol,seal,spec['segment_id'])
    budget_binding=amendment_binding(budget_amendment)
    budget_inputs=amendment_inputs(budget_amendment,receipt,protocol,seal)
    budget_inputs.update(validate_build_budget(build_report,budget_amendment,receipt,protocol,spec['segment_id']))
    _,registered=load_protocol(protocol,spec['segment_id'])
    require(spec==registered,'SOURCE_SEGMENT_NOT_EXACT_REGISTERED_SPEC')
    complete=read(Path(build_report)/'worker_complete.json')
    require(complete['status']=='COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'
            and complete['source_inputs_unchanged'] is True,'AMENDED_WORKER_COMPLETION_REQUIRED')
    bound_camera=Path(build_report)/'camera_inputs.json'
    require(file_sha(bound_camera)==complete['camera_inputs_sha256']
            and camera==read(bound_camera),'AMENDED_SOURCE_CAMERA_BINDING_MISMATCH')
    require(Path(cache_root).resolve()==Path(complete['cache']).resolve(),'AMENDED_SOURCE_CACHE_BINDING_MISMATCH')
    for item in (Path(build_report)/'worker_complete.json',bound_camera):
        budget_inputs[str(item.resolve())]=file_sha(item)
    verify_bindings(budget_inputs)
    input_cache_limit=receipt['effective_budgets']['cache_per_scene_output_bytes']
    cache=Path(cache_root).resolve();output=Path(output_root).resolve()
    require(0<seconds<=1200,'SOURCE_STAGE_WALL_BUDGET_MUST_NOT_EXCEED_1200_SECONDS')
    requested_seconds=seconds
    seconds=min(seconds,require_remaining(receipt))
    require(not output.exists() and output!=cache and cache not in output.parents and output not in cache.parents,
            'OUTPUT_NOT_FRESH_OR_OVERLAPS_CACHE')
    output.mkdir(parents=True);started=time.monotonic();cpu=time.process_time()
    sources={str(p.resolve()):file_sha(p) for folder in (C1,HERE.parent/'source_splice',HERE.parent/'tv0_tv4') for p in folder.glob('*.py')}
    sources[str(Path(__file__).resolve())]=file_sha(__file__)
    for name in ('final_gate_resources.py','final_gate_amendment.py','screen_contracts.py',
                 'screen_resources.py','scene_source.py'):
        sources[str((HERE/name).resolve())]=file_sha(HERE/name)
    events=[];inventory_rows=[];dataset=None;verification=None;summary={'status':'STOP_SOURCE_STAGE_NOT_COMPLETE'}
    registry_complete=False;control_inputs={}
    registry_path=cache/'event_registry_p1.csv';registry_hash=None
    try:
        profile=validate_profile(camera,spec)
        cache_files=list(cache.rglob('*'))
        require(not any(p.is_symlink() for p in cache_files),'SYMLINK_CACHE_INPUT_UNSUPPORTED')
        require(sum(p.stat().st_size for p in cache_files if p.is_file())<=input_cache_limit,'SOURCE_INPUT_CACHE_EXCEEDS_FINAL_8_GIB')
        require((cache/'slicing_preprocess.finish').is_file(),'CACHE_PREPROCESS_NOT_COMPLETE')
        manifest_path=cache/'slicing_preprocess.manifest.json';manifest=json.loads(manifest_path.read_text())
        control_inputs={str(p):file_sha(p) for p in (manifest_path,cache/'slicing_preprocess.finish')}
        require(all(manifest.get(k) is True for k in ('event_registry_enabled','provenance_enabled','provenance_requested')) and
                manifest.get('bpm_version')==manifest.get('bhp_version')==2,'COMPLETED_EVENT_BPM2_BHP2_CACHE_REQUIRED')
        require(registry_path.stat().st_size<=16*1024*1024,'REGISTRY_READ_BUDGET')
        raw=registry_path.read_bytes();registry_hash=hashlib.sha256(raw).hexdigest()
        csv_reader=csv.DictReader(io.StringIO(raw.decode('utf-8')))
        require({'raw_id','canonical_event_id','logical_incidence_id','root_num','root_den','element'}<=set(csv_reader.fieldnames or []),
                'REGISTRY_SCHEMA_MISMATCH')
        rows=list(csv_reader);registry=defaultdict(list)
        for row in rows:registry[row['canonical_event_id']].append(row)
        for eid,replicas in registry.items():
            roots={F(int(r['root_num']),int(r['root_den'])) for r in replicas};elements={int(r['element']) for r in replicas}
            require(len(roots)==len(elements)==1,'REGISTRY_ROOT_OR_ELEMENT_DISAGREEMENT')
            events.append(base_event(eid,next(iter(roots)),next(iter(elements)),spec))
        events.sort(key=lambda e:(F(e['root']),e['event_id']))
        registry_complete=True
        if not events:
            groups,maximum=reader_api.infer_cache_shape(cache)
            reader_api._stream_inventory(cache,groups,maximum)
            require((groups,maximum)==(profile['group_count'],profile['maximum_discrete_time']),'CACHE_CAMERA_TIME_SHAPE_MISMATCH')
            verification={'pass':True,'scope':'Completed zero-row registry, manifest and complete primary/BPM2 file matrix; no event source support needed.',
                'input_full_sha256':{str(p):file_sha(p) for p in (registry_path,manifest_path,cache/'slicing_preprocess.finish')}}
        else:
            dataset=reader_api.read_event_candidates(cache,seconds=seconds)
            require(set(dataset.event_ids)==set(registry) and (dataset.groups,dataset.maximum)==(profile['group_count'],profile['maximum_discrete_time']),
                    'READER_POPULATION_OR_CAMERA_TIME_SHAPE_MISMATCH')
            requests={}
            for event in events:
                eid=event['event_id'];kernel=compiler_api.compile_kernel(dataset.registry[eid],dataset.hypervertices);event['kernel']=kernel
                witness=kernel_domain_witness(dataset.registry[eid],dataset.hypervertices,kernel)
                if witness is not None:
                    event.update(source_status='UNSUPPORTED',source_policy_witness=witness)
                    corner_times=list(map(F,witness['corner_times']))
                else:
                    require(kernel['status']=='PASS_EXISTING_LOCAL_KERNEL','UNEXPLAINED_KERNEL_FAILURE:'+str(kernel.get('reason',kernel['status'])))
                    corner_times=kernel['corner_times']
                bounds=compiler_api.levels(dataset.roots[eid],corner_times)
                event['schedule']=make_schedule(camera,*bounds);event['schedule_sha256']=digest(event['schedule'])
                event['original_frame_numbers']={q['key']:spec['first_frame']+q['frame_index_zero_based'] for q in event['schedule']['natural']}
                requests[eid]=compiler_api.required_times(dataset.roots[eid],corner_times)
            snapshots=dataset.load_halos(requests);binding=reader_binding(dataset)
            for event in events:
                dataset.reader.check();eid=event['event_id'];begin=time.monotonic()
                if event['source_status']!='UNSUPPORTED':
                    kernel=event['kernel'];compiled=compiler_api.compile_event(eid,F(event['root']),kernel['corner_times'],snapshots,dataset.hypervertices,
                        kernel['critical_position'],cache_digest=binding,group_count=dataset.groups,maximum_discrete_time=dataset.maximum,element=event['element'])
                    event['compiler']=compiled;rejection=classify_compiler(compiled)
                    if rejection is not None:event.update(source_status='REJECTED_FIXED_POLICY',decision='REJECTED_FIXED_POLICY',decisive_rejection=rejection)
                    else:
                        require(compiled['status']=='PASS_SOURCE_COMPILER','UNEXPLAINED_SOURCE_COMPILER_FAILURE:'+str(compiled.get('reason',compiled['status'])))
                        interface=audit_requested_source(compiled['source'],event['schedule'],snapshots,event_id=eid,element=event['element'])
                        event['requested_source_interface']=interface
                        if interface['status']=='REJECT':
                            event.update(source_status='REJECTED_FIXED_POLICY',decision='REJECTED_FIXED_POLICY',decisive_rejection={
                                **interface['decisive_rejection'],'gate':'requested_source_interface','compiler_sha256':digest(compiled)})
                        else:
                            require(interface['status']=='PASS_REQUESTED_SOURCE_INTERFACE','UNEXPLAINED_SOURCE_INTERFACE_UNKNOWN')
                            event['source_status']='SOURCE_READY'
                else:event['compiler']={'status':'UNSUPPORTED_FROZEN_KERNEL_DOMAIN','source':None,'reason':event['source_policy_witness']['reason_code']}
                inventory_rows.append(build_inventory_event(event,snapshots,dataset));event['cost']={'wall_seconds':time.monotonic()-begin}
            verification=dataset.verify_inputs()
        require(file_sha(registry_path)==registry_hash,'REGISTRY_CHANGED_DURING_SOURCE_STAGE')
        require(all(file_sha(p)==h for p,h in control_inputs.items()),'CACHE_COMPLETION_PROVENANCE_CHANGED')
        require(document(camera_document)[1]==cb and document(segment_spec)[1]==sb,'CAMERA_OR_SEGMENT_SPEC_CHANGED')
        require(all(file_sha(p)==h for p,h in sources.items()),'FROZEN_SOURCE_IMPLEMENTATION_CHANGED')
        verify_bindings(budget_inputs)
        require(time.monotonic()-started<=seconds,'SOURCE_STAGE_WALL_BUDGET_EXCEEDED')
        require_remaining(receipt)
        symbolic,unsupported_groups=symbolic_dependencies(inventory_rows)
        summary={'status':'COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION','profile':profile,
            'raw_registry_observations':len(rows),'logical_incidences':len({r['logical_incidence_id'] for r in rows}),
            'canonical_event_denominator':len(events),'root_population':dict(Counter(e['root'] for e in events)),
            'source_status_counts':dict(Counter(e['source_status'] for e in events)),
            'all_registry_identities_retained':True,'same_root_partition_warnings':unsupported_groups,
            'runtime_attempted':False,'admission_rate':None,'input_verification':verification,
            'reader':None if dataset is None else dataset.summary()}
    except Exception as error:
        symbolic=[];summary.update(reason=type(error).__name__+': '+str(error),canonical_event_denominator=len(events) if registry_complete else None,
            source_status_counts=dict(Counter(e['source_status'] for e in events)),runtime_attempted=False,admission_rate=None,
            input_verification=verification,completed_source_inventory_events=len(inventory_rows),
            source_status_not_reached_retained=True)
        completed={e['event_id'] for e in inventory_rows}
        for event in events:
            if event['event_id'] not in completed:
                inventory_rows.append({'event_id':event['event_id'],'namespace_id':event['namespace_id'],
                    'root':event['root'],'element':event['element'],'source_status':'NOT_REACHED',
                    'candidate_domain_complete':False,'cells':[],'record_catalog':[],
                    'face_class_catalog':[],'owner_catalog':[],
                    'scope':'Explicit incomplete placeholder after stage STOP; never an empty certified support domain.'})
    inventory={'schema':'scene-fixed-source-support-inventory-v1','status':'COMPLETE_FIXED_SOURCE_SUPPORT_INVENTORY' if
        summary['status']=='COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION' else 'INCOMPLETE_DO_NOT_USE_FOR_GRAPH',
        'events':inventory_rows,'event_count':len(inventory_rows),'registry_event_denominator':len(events) if registry_complete else None,
        'symbolic_edges':symbolic,'runtime_identity_or_geometry_proven':False,
        'coordinate_model':'Ideal rational serialized-HV interpolation; legacy parser binary64 separate; actual source VID requires observer encoding v2.'}
    index=[];output_budget={'remaining':100*1024*1024}
    for i,event in enumerate(events):
        artifact=write_new(output/'events'/f'{i:05d}-{hashlib.sha256(event["event_id"].encode()).hexdigest()[:12]}.json',event,budget=output_budget)
        index.append({k:event[k] for k in ('event_id','namespace_id','root','element','decision','source_status')}|{'artifact':artifact})
    summary.update(schema='scene-source-stage-v1',segment_id=spec.get('segment_id'),scene=spec.get('scene'),
        budget_amendment=budget_binding,amendment_input_sha256=budget_inputs,
        original_protocol_unchanged=True,budget_scope='Only input cache ceiling amended; source compute/output/reader limits unchanged.',
        cache_root=str(cache),registry_sha256=registry_hash,camera_binding=cb,segment_spec_binding=sb,
        cache_completion_provenance_sha256=control_inputs,
        executed_sources_sha256=sources,events_index=write_new(output/'events_index.json',index,budget=output_budget),
        source_support_inventory=write_new(output/'source_support_inventory.json',inventory,budget=output_budget),
        cost={'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
        scope='Fixed preregistered segment source stage only; no native, full exterior, component admission, or visibility claim.')
    summary['resource_limits']={'wall_seconds':seconds,'input_cache_bytes':input_cache_limit,'source_output_bytes':100*1024**2,
        'reader_selected_records':reader_api.MAX_SELECTED_RECORDS,'reader_snapshot_triangles':reader_api.MAX_SNAPSHOT_TRIANGLES,
        'reader_total_snapshot_triangles':reader_api.MAX_TOTAL_SNAPSHOT_TRIANGLES,'reader_rss_bytes':2*1024**3}
    summary['resource_limits'].update(requested_wall_seconds=requested_seconds,
        campaign_deadline_utc=receipt['campaign_deadline_utc'])
    try:
        require(time.monotonic()-started<=seconds,'SOURCE_STAGE_WALL_BUDGET_EXCEEDED')
        require_remaining(receipt)
        verify_bindings(budget_inputs)
    except Exception as error:
        summary.update(status='STOP_SOURCE_STAGE_NOT_COMPLETE',reason=type(error).__name__+': '+str(error),
                       admission_rate=None,runtime_attempted=False)
    write_new(output/'summary.json',summary,budget=output_budget)
    return {'summary':summary,'events':events,'inventory':inventory}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','seal','budget-amendment','build-report','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--segment',choices=('cave','forest_b'),required=True)
    parser.add_argument('--seconds',type=float,default=1200);args=parser.parse_args()
    _,spec=load_protocol(args.protocol,args.segment)
    complete=read(args.build_report/'worker_complete.json')
    result=run_source_stage(complete['cache'],args.build_report/'camera_inputs.json',spec,args.output,
        seconds=args.seconds,budget_amendment=args.budget_amendment,protocol=args.protocol,
        seal=args.seal,build_report=args.build_report)
    print(json.dumps({k:result['summary'].get(k) for k in ('status','canonical_event_denominator','source_status_counts','reason')}))
    return 0 if result['summary']['status']=='COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION' else 2


if __name__=='__main__':raise SystemExit(main())
