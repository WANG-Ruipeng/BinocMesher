"""Independent JSON-only OMP1/8 audit of a 64-natural + 2-root sequence.

No native code, arrays, source compiler or runtime helpers are imported.
Checks artifact hashes, exact query arithmetic, canonical records and union
placement before comparing reproducibility. Costs/OMP labels are not required
to match. Equal receipts are not a new all-real-time or image-quality proof.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time


class InvalidSequence(ValueError):pass


def require(value,reason):
    if not value:raise InvalidSequence(reason)


def stable(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(stable(value)).hexdigest()


def is_int(value):return isinstance(value,int) and not isinstance(value,bool)


def hash_value(value,name):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'INVALID_SHA256:'+name)
    return value


def receipts(value):
    require(isinstance(value,list) and len(value)==5,'EXPECTED_FIVE_MESH_RECEIPTS')
    for row in value:
        require(all(is_int(row.get(k)) and row[k]>=0 for k in ('vertices','faces')),'INVALID_MESH_COUNTS')
        require(set(row.get('sha256',{}))=={'vertices','faces','tags'},'INCOMPLETE_MESH_ARRAY_HASHES')
        for k,h in row['sha256'].items():hash_value(h,'mesh.'+k)
    return value


def input_verification(summary):
    first=summary['input_verification']
    result=None
    for row in (first,summary.get('final_input_verification',first)):
        require(row.get('status')=='PASS' and row.get('private_nonlog_inputs_unchanged') is True,
                'CACHE_INPUT_VERIFICATION_NOT_PASS')
        current={k:row[k] for k in ('original_input_content_sha256','original_files_unchanged','original_bytes')}
        hash_value(current['original_input_content_sha256'],'original_cache')
        require(all(is_int(current[k]) and current[k]>0 for k in ('original_files_unchanged','original_bytes')),
                'INVALID_ORIGINAL_CACHE_COUNTS')
        require(result is None or current==result,'CACHE_BINDING_CHANGED_WITHIN_SEQUENCE')
        result=current
    return result


def validate_query(query,delta):
    require(isinstance(query,dict),'MALFORMED_QUERY')
    key=query['key'];kind=query['kind'];tau=F(query['evaluation_tau'])
    physical=float.fromhex(query['physical_time_hex'])
    require(math.isfinite(physical) and physical>=0 and physical.hex()==query['physical_time_hex'],
            'NONCANONICAL_OR_NONFINITE_PHYSICAL_TIME')
    if kind=='natural':
        number=query['frame_number']
        require(is_int(number) and 1<=number<=64 and is_int(query['frame_index_zero_based']) and query['frame_index_zero_based']==number-1 and
                key==f'frame_{number:04d}' and query['time_mode']=='physical','INVALID_NATURAL_FRAME_IDENTITY')
        global_time=float.fromhex(query['global_camera_time_hex'])
        require(math.isfinite(global_time) and global_time.hex()==query['global_camera_time_hex'],
                'INVALID_GLOBAL_CAMERA_TIME')
        require(tau==F.from_float(float(physical/delta)),'PHYSICAL_QUERY_TO_INTERNAL_TIME_DISAGREEMENT')
    else:
        require(kind=='exact_root' and query['time_mode']=='exact' and tau in (F(3,2),F(5,2)),
                'UNEXPECTED_ROOT_QUERY')
        require(key=='root_'+str(tau.numerator)+'_'+str(tau.denominator) and query.get('frame_number') is None,
                'INVALID_ROOT_IDENTITY')
        require(float(tau*F.from_float(delta)).hex()==query['physical_time_hex'],'ROOT_PHYSICAL_TIME_DISAGREEMENT')
    return key


def validate_frame(frame,summary,delta):
    key=validate_query(frame['query'],delta)
    tau=F(frame['query']['evaluation_tau'])
    active_support=F(5,4)<tau<F(7,4) or F(9,4)<tau<F(11,4)
    require('root_support_replay_sha256' in frame,'MISSING_ROOT_SUPPORT_REPLAY_FIELD')
    if active_support:hash_value(frame['root_support_replay_sha256'],'root_support_replay')
    else:require(frame['root_support_replay_sha256'] is None,'OUTSIDE_QUERY_HAS_UNEXPECTED_ROOT_SUPPORT_REPLAY')
    require(frame.get('publication')=='STAGED_UNTIL_FULL_SEQUENCE_VERIFIED','FRAME_PARTIAL_PUBLICATION_POLICY')
    require(frame.get('single_scene_output_elements')==5 and frame.get('persistent_mesh_files')==0 and
            frame.get('fresh_baseline_for_each_timestamp') is True and frame.get('full_scene_arrays_actually_constructed') is True,
            'MISSING_COMPLETE_ACTUAL_SCENE_DECLARATION')
    require(is_int(frame.get('rejected_actual_domains_unchanged')) and frame['rejected_actual_domains_unchanged']>=0,
            'MISSING_REJECTED_DOMAIN_AUDIT_COUNT')
    require(frame['certification_sha256']==summary['certification_sha256'],'FRAME_CERTIFICATION_BINDING_MISMATCH')
    baseline=receipts(frame['baseline']);union=frame['union']
    require(union.get('schema')=='forest-component-query-union-v1' and
            union.get('status')=='PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED','UNION_NOT_COMPLETE_PASS')
    require(union.get('partial_output_published') is False,'UNION_PARTIAL_OUTPUT_PUBLISHED')
    for name in ('provided_records_all_or_none','component_membership_verified','old_vertices_and_tags_byte_identical',
                 'retained_original_face_rows_byte_identical','original_baseline_unchanged'):
        require(union.get(name) is True,'UNION_GATE_MISSING:'+name)
    require(union['baseline']==baseline,'FRAME_UNION_BASELINE_DISAGREEMENT')
    output=receipts(union['output']);records=frame['records']
    require(isinstance(records,list),'MISSING_CANONICAL_RECORDS')
    require(frame.get('changed') is bool(records) and (output!=baseline)==bool(records),'FRAME_CHANGE_ACCOUNTING_MISMATCH')
    require(hash_value(frame['records_sha256'],'records')==digest(records),'RECORDS_HASH_MISMATCH')
    ids=[r['event_id'] for r in records]
    require(all(isinstance(e,str) and e for e in ids) and ids==sorted(set(ids)),'NONCANONICAL_OR_DUPLICATE_RECORD_ORDER')
    require(union['canonical_event_order']==ids and set(union['event_mapping'])==set(ids),
            'UNION_EVENT_MAPPING_POPULATION_MISMATCH')
    expected_components=defaultdict(list);ordinals=Counter();removed=set();owners=set();boundary=set()
    for record in records:
        eid,cid,plan=record['event_id'],record['component_id'],record['plan']
        require(isinstance(cid,str) and cid,'INVALID_COMPONENT_ID')
        expected_components[cid].append(eid)
        element=plan['element'];require(is_int(element) and 0<=element<5,'INVALID_RECORD_ELEMENT')
        nv,nf=baseline[element]['vertices'],baseline[element]['faces']
        require(plan['baseline_vertex_count']==nv and plan['baseline_face_count']==nf and plan['new_center_id']==nv,
                'PLAN_NOT_SAME_BASELINE')
        cycle=plan['boundary_actual_ids'];source_faces=plan['removed_face_rows'];raw_owners=plan['consumed_owners']
        require(len(cycle)==4 and len(set(cycle))==4 and all(is_int(x) and 0<=x<nv for x in cycle),
                'INVALID_SOURCE_CYCLE')
        require(len(source_faces)==2 and len(set(source_faces))==2 and
                all(is_int(x) and 0<=x<nf for x in source_faces),'INVALID_REMOVED_SOURCE_FACES')
        require(raw_owners and all(len(o)==7 and o[0]==element and all(is_int(x) and x>=0 for x in o) for o in raw_owners),
                'INVALID_SOURCE_OWNER_REF')
        center=plan['center'];require(len(center)==3,'INVALID_CENTER_SHAPE')
        for x in center:
            require(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x),'INVALID_CENTER_NUMBER')
            try:rounded=struct.unpack('<f',struct.pack('<f',float(x)))[0]
            except (OverflowError,struct.error) as error:raise InvalidSequence('CENTER_BINARY32_OVERFLOW') from error
            require(float(x).hex()==float(rounded).hex(),'CENTER_NOT_EXACT_BINARY32')
        hash_value(plan['source_digest'],'source_contract')
        expected_plan_fan=[[cycle[i],cycle[(i+1)%4],nv] for i in range(4)]
        require(plan['fan_faces']==expected_plan_fan,'SINGLE_PLAN_FAN_ORIENTATION_MISMATCH')
        for name,values,seen in (
                ('boundary',[(element,x) for x in cycle],boundary),
                ('source_face',[(element,x) for x in source_faces],removed),
                ('raw_owner',[tuple(o) for o in raw_owners],owners)):
            for value in values:
                require(value not in seen,'SOURCE_CONSUMED_MULTIPLE_TIMES:'+name);seen.add(value)
        ordinal=ordinals[element];center_id=nv+ordinal;ordinals[element]+=1
        actual=union['event_mapping'][eid]
        expected={'event_id':eid,'component_id':cid,'element':element,'new_center_id':center_id,
            'fan_face_rows_by_sector':[*source_faces,nf+2*ordinal,nf+2*ordinal+1],
            'fan_actual_vertex_ids':[[cycle[i],cycle[(i+1)%4],center_id] for i in range(4)],
            'source_face_rows':source_faces,'consumed_owners':raw_owners}
        require(actual==expected,'CANONICAL_UNION_EVENT_PLACEMENT_MISMATCH:'+eid)
    expected_mapping={cid:{'events':members,'event_mapping_keys':members} for cid,members in expected_components.items()}
    require(union['component_mapping']==expected_mapping and union['components']==dict(expected_components),
            'PARTIAL_OR_WRONG_COMPONENT_MEMBERSHIP')
    require(union['event_count']==len(records) and union['component_count']==len(expected_components),
            'UNION_POPULATION_COUNTS_MISMATCH')
    require(union['source_faces_consumed_once']==len(removed)==2*len(records) and
            union['raw_owners_consumed_once']==len(owners),'UNION_CONSUMPTION_COUNTS_MISMATCH')
    expected_pairs=len(records)*(len(records)-1)//2
    require(union['expected_event_pairs']==expected_pairs and union['expected_triangle_pairs']==32*expected_pairs,
            'INCOMPLETE_PAIR_DENOMINATOR')
    counts=union['counts']
    require(counts['triangle_pairs_excluded_by_support_aabb']+counts['exact_triangle_pair_calls']==32*expected_pairs,
            'INCOMPLETE_TRIANGLE_PAIR_ACCOUNTING')
    for element in range(5):
        require(output[element]['vertices']==baseline[element]['vertices']+ordinals[element] and
                output[element]['faces']==baseline[element]['faces']+2*ordinals[element],
                'UNION_OUTPUT_COUNT_DELTA_MISMATCH')
        if not ordinals[element]:require(output[element]==baseline[element],'UNMODIFIED_ELEMENT_OUTPUT_CHANGED')
    return key


def validate_sequence(summary,frames):
    require(summary.get('schema')=='forest-atomic-sequence-v1' and
            summary.get('status')=='PASS_FOREST_ATOMIC_REQUESTED_SEQUENCE','SEQUENCE_NOT_COMPLETE_PASS')
    require(summary.get('private_cache_removed') is True,'PRIVATE_CACHE_CLEANUP_NOT_VERIFIED')
    require(summary.get('partial_output_published') is False,'SEQUENCE_PARTIAL_PUBLICATION_NOT_ZERO')
    require(str(summary.get('omp_num_threads')) in ('1','8'),'UNEXPECTED_OMP_LABEL')
    for name in ('support_graph_sha256','certification_sha256'):
        hash_value(summary[name],name)
    delta=float.fromhex(summary['actual_delta_t_hex'])
    require(math.isfinite(delta) and delta>0 and delta.hex()==summary['actual_delta_t_hex'],'INVALID_ACTUAL_DELTA')
    input_binding=input_verification(summary)
    require(len(frames)==66 and len(summary['frame_artifacts'])==66,'INCOMPLETE_64_PLUS_2_SEQUENCE')
    keyed={}
    for frame in frames:
        key=validate_frame(frame,summary,delta);require(key not in keyed,'DUPLICATE_QUERY_RECEIPT');keyed[key]=frame
    expected={f'frame_{i:04d}' for i in range(1,65)}|{'root_3_2','root_5_2'}
    require(set(keyed)==expected,'MISSING_OR_EXTRA_FRAME_OR_ROOT')
    require(summary.get('publication')=='COMMITTED_FULL_REQUESTED_SEQUENCE_MANIFEST' and
            summary.get('published_partial_outputs') is False,'MANIFEST_NOT_ATOMICALLY_COMMITTED')
    for name,value in (('natural_frame_count',64),('exact_root_diagnostic_count',2),('actual_scene_outputs',66),('root_denominator',2)):
        require(summary.get(name)==value,'SUMMARY_COMPLETE_SEQUENCE_COUNT:'+name)
    event_components={};component_members={}
    for frame in frames:
        for eid,mapping in frame['union']['event_mapping'].items():
            cid=mapping['component_id']
            require(eid not in event_components or event_components[eid]==cid,'EVENT_COMPONENT_CHANGED_ACROSS_SEQUENCE')
            event_components[eid]=cid
        for cid,mapping in frame['union']['component_mapping'].items():
            require(cid not in component_members or component_members[cid]==mapping['events'],'COMPONENT_MEMBERSHIP_CHANGED_ACROSS_SEQUENCE')
            component_members[cid]=mapping['events']
    modified=[i for i in range(1,65) if keyed[f'frame_{i:04d}']['changed']]
    replacements=sum(len(keyed[f'frame_{i:04d}']['records']) for i in range(1,65))
    require(summary.get('actual_modified_natural_frames')==modified and summary.get('natural_frame_change_rate')==len(modified)/64 and
            summary.get('actual_natural_event_frame_replacements')==replacements,'SUMMARY_NATURAL_REPLACEMENT_ACCOUNTING')
    require(summary.get('jointly_admitted_events')==len(event_components) and summary.get('admitted_components')==len(component_members),
            'SUMMARY_EVENT_COMPONENT_ACCOUNTING')
    require(summary.get('nonempty_treated_roots')==sum(bool(keyed[k]['records']) for k in ('root_3_2','root_5_2')),
            'SUMMARY_TREATED_ROOT_ACCOUNTING')
    natural=[keyed[f'frame_{i:04d}']['query'] for i in range(1,65)]
    origin=float.fromhex(natural[0]['global_camera_time_hex']);previous=-math.inf
    for query in natural:
        global_time=float.fromhex(query['global_camera_time_hex'])
        require(global_time>previous,'NATURAL_CAMERA_TIMES_NOT_STRICTLY_INCREASING');previous=global_time
        require(float(global_time-origin).hex()==query['physical_time_hex'],'CAMERA_ORIGIN_TO_PHYSICAL_TIME_DISAGREEMENT')
    return keyed,input_binding


def compare_sequences(left_summary,left_frames,right_summary,right_frames):
    """Validate both sequence contracts, then compare deterministic payloads."""
    left,li=validate_sequence(left_summary,left_frames);right,ri=validate_sequence(right_summary,right_frames)
    require({str(left_summary['omp_num_threads']),str(right_summary['omp_num_threads'])}=={'1','8'},'EXPECTED_OMP1_VS_OMP8')
    differences=[]
    def compare(name,a,b):
        if stable(a)!=stable(b):differences.append({'field':name,'left_sha256':digest(a),'right_sha256':digest(b)})
    for field in ('support_graph_sha256','certification_sha256','actual_delta_t_hex'):
        compare('summary.'+field,left_summary[field],right_summary[field])
    if 'native_library_sha256' in left_summary or 'native_library_sha256' in right_summary:
        require('native_library_sha256' in left_summary and 'native_library_sha256' in right_summary,'MISSING_LIBRARY_BINDING')
        for s in (left_summary,right_summary):hash_value(s['native_library_sha256'],'library')
        compare('summary.native_library_sha256',left_summary['native_library_sha256'],right_summary['native_library_sha256'])
    compare('summary.original_input_binding',li,ri)
    if 'executed_sources_sha256' in left_summary or 'executed_sources_sha256' in right_summary:
        require('executed_sources_sha256' in left_summary and 'executed_sources_sha256' in right_summary,'MISSING_EXECUTED_SOURCE_BINDING')
        for summary in (left_summary,right_summary):
            for path,value in summary['executed_sources_sha256'].items():hash_value(value,'executed_source:'+path)
        compare('summary.executed_sources_sha256',left_summary['executed_sources_sha256'],right_summary['executed_sources_sha256'])
    for key in sorted(left):
        for field in ('query','baseline','records','records_sha256','certification_sha256','publication','root_support_replay_sha256'):
            compare(key+'.'+field,left[key][field],right[key][field])
        for field in ('changed','rejected_actual_domains_unchanged','single_scene_output_elements',
                      'fresh_baseline_for_each_timestamp','full_scene_arrays_actually_constructed','persistent_mesh_files'):
            compare(key+'.'+field,left[key][field],right[key][field])
        # Union contains only deterministic proof/mapping/receipt fields; no costs.
        compare(key+'.union',left[key]['union'],right[key]['union'])
    return {'schema':'forest-sequence-comparison-v1',
        'status':'PASS_IDENTICAL_ATOMIC_FOREST_SEQUENCES' if not differences else 'FAIL_SEQUENCE_DIFFERENCE',
        'natural_frames_compared':64,'exact_roots_compared':2,'five_elements_per_query':5,
        'independent_record_hash_and_placement_validation':True,'zero_partial_publication_verified':True,
        'all_canonical_records_and_mesh_receipts_equal':not differences,
        'different_fields':len(differences),'first_differences':differences[:16],
        'costs_and_omp_labels_required_equal':False,
        'scope':'Finite complete requested sequence receipt reproducibility; not an all-time theorem or image-quality claim.'}


def read_json(path,maximum=4*1024*1024):
    require(path.is_file() and path.stat().st_size<=maximum,'MISSING_OR_OVERSIZED_JSON:'+str(path))
    data=path.read_bytes()
    def reject_constant(x):raise InvalidSequence('NONFINITE_JSON_CONSTANT:'+x)
    return json.loads(data,parse_constant=reject_constant),hashlib.sha256(data).hexdigest(),len(data)


def load_sequence(path):
    path=Path(path).resolve();path=path/'summary.json' if path.is_dir() else path
    summary,sha,total=read_json(path);bindings={str(path):sha};frames=[]
    rows=summary.get('frame_artifacts',[]);require(len(rows)==66,'INCOMPLETE_FRAME_ARTIFACT_LIST')
    for row in rows:
        relative=Path(row['path']);require(not relative.is_absolute(),'ABSOLUTE_FRAME_ARTIFACT_PATH')
        target=(path.parent/relative).resolve()
        require(path.parent in target.parents and target not in (path,),'FRAME_ARTIFACT_PATH_ESCAPES_SEQUENCE')
        require(str(target) not in bindings,'DUPLICATE_FRAME_ARTIFACT_PATH')
        value,actual,size=read_json(target);total+=size
        require(total<=64*1024*1024,'SEQUENCE_JSON_BUDGET_EXCEEDED')
        require(hash_value(row['sha256'],'frame_artifact')==actual,'FRAME_ARTIFACT_HASH_MISMATCH')
        bindings[str(target)]=actual;frames.append(value)
    return summary,frames,bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--omp1',type=Path,required=True);parser.add_argument('--omp8',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(not args.output.exists(),'REFUSE_TO_OVERWRITE_COMPARISON');start=time.monotonic()
    bindings={}
    try:
        a,af,ab=load_sequence(args.omp1);b,bf,bb=load_sequence(args.omp8)
        require(str(a['omp_num_threads'])=='1' and str(b['omp_num_threads'])=='8','CLI_OMP_LABEL_MISMATCH')
        bindings={**ab,**bb};report=compare_sequences(a,af,b,bf)
        require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in bindings.items()),'INPUT_CHANGED_DURING_COMPARISON')
        report['input_artifacts_unchanged']=True
    except (InvalidSequence,ValueError,TypeError,KeyError,IndexError,ArithmeticError,OSError) as error:
        report={'schema':'forest-sequence-comparison-v1','status':'STOP_INVALID_OR_UNVERIFIED_SEQUENCE',
                'reason':type(error).__name__+': '+str(error),'not_a_nondeterminism_proof':True}
    report.update(audited_files_sha256=bindings,executed_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  wall_seconds=time.monotonic()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({'status':report['status'],'output':str(args.output)}))
    return 0 if report['status']=='PASS_IDENTICAL_ATOMIC_FOREST_SEQUENCES' else 1 if report['status']=='FAIL_SEQUENCE_DIFFERENCE' else 2


if __name__=='__main__':raise SystemExit(main())
