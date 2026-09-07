"""JSON-only coverage ledger: old Forest negative + all preregistered segments.

Reads source/composition/visibility counts and preregistered qualification rows,
never quality deltas, images, caches, native libraries or alternative scenes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def require(value,reason):
    if not value:raise ValueError(reason)


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(path,bindings,expected=None):
    path=Path(path).resolve();require(path.is_file() and path.stat().st_size<=16*1024*1024,'MISSING_OR_OVERSIZED_COVERAGE_JSON:'+str(path))
    data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
    require(expected is None or actual==expected,'COVERAGE_INPUT_HASH_MISMATCH:'+str(path))
    def invalid(x):raise ValueError('NONFINITE_JSON:'+x)
    value=json.loads(data,parse_constant=invalid);bindings[str(path)]=actual
    return value,{'path':str(path),'sha256':actual}


def count(value,name):
    require(type(value) is int and value>=0,'INVALID_COUNT:'+name)
    return value


def rate(numerator,denominator):
    return numerator/denominator if denominator else None


def verify_confirmed_summaries(combined,confirmation,confirmation_path,bindings):
    """The combined counts must be exactly those in the confirmed OMP1 file."""
    originals=[];refs=[]
    for label,status,threads in (('omp1_summary','PASS_PREREGISTERED_SEGMENT_SCREEN_OMP1_ONLY','1'),
                                 ('omp8_summary','PASS_OMP8_SEQUENCE_REPLAY','8')):
        row=confirmation[label];path=Path(row['path'])
        if not path.is_absolute():path=Path(confirmation_path).resolve().parent/path
        document,ref=read_bound(path,bindings,row['sha256']);refs.append(ref);originals.append(document)
        require(document['status']==status and str(document['omp_num_threads'])==threads and
                document.get('private_cache_removed') is True and document.get('partial_output_published') is False and
                document.get('final_input_verification',{}).get('status')=='PASS', 'CONFIRMED_OMP_RUN_NOT_COMPLETE')
        require(all(document.get(k)==combined.get(k)==confirmation.get(k) for k in
                    ('segment_id','protocol_sha256','preregistration_seal_sha256')), 'CONFIRMED_OMP_RUN_INPUT_BINDING_MISMATCH')
    require(confirmation.get('all_arrays_records_component_membership_and_complete_root_supports_identical') is True and
            confirmation.get('queries_compared')==combined['schedule']['query_count'], 'INCOMPLETE_OMP_COMPARISON_RECEIPT')
    require(originals[1].get('primary_summary',{}).get('sha256')==refs[0]['sha256'], 'OMP8_REPLAY_REFERENCES_DIFFERENT_OMP1')
    fields=('source','composition','schedule','visibility','support_graph','certified_events',
            'component_queries','frame_artifacts','visibility_frame_artifacts','visibility_artifacts_base')
    require(all(k in combined and combined[k]==originals[0].get(k) for k in fields),
            'COMBINED_COUNTS_OR_VISIBILITY_NOT_EQUAL_CONFIRMED_OMP1')
    require(all(originals[0].get(k)==originals[1].get(k) for k in ('source','composition','schedule','visibility')),
            'CONFIRMED_OMP_SUMMARY_ACCOUNTING_DIFFERS')
    return refs


def _stage_stop(upstream_path,stage,segment,protocol_sha,seal_sha,bindings):
    require(stage in ('build','source'),'UNSUPPORTED_UPSTREAM_STOP_STAGE')
    upstream,ref=read_bound(upstream_path,bindings)
    require(upstream.get('status','').startswith('STOP') and upstream.get('segment_id')==segment['segment_id'],
            'UPSTREAM_NOT_A_STOP_FOR_THIS_SEGMENT')
    if stage=='build':
        contract,_=read_bound(Path(upstream_path).resolve().parent/'build_protocol.json',bindings,upstream['build_protocol_sha256'])
        require(contract['protocol_sha256']==protocol_sha and contract['segment']==segment,'BUILD_STOP_PREREGISTRATION_MISMATCH')
        source=None
    else:
        binding=upstream['segment_spec_binding']
        if binding.get('kind')=='CANONICAL_JSON':
            expected=hashlib.sha256(json.dumps(segment,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            require(binding['sha256']==expected,'SOURCE_STOP_SEGMENT_BINDING_MISMATCH')
        else:
            spec,_=read_bound(binding['path'],bindings,binding['sha256'])
            require(spec==segment,'SOURCE_STOP_SEGMENT_BINDING_MISMATCH')
        source={'canonical_events':upstream.get('canonical_event_denominator'),
            'raw_observations':upstream.get('raw_registry_observations'),'exact_roots':None,
            'source_decision_counts':upstream.get('source_status_counts')}
        if source['canonical_events'] is not None:
            item=upstream['events_index'];path=Path(item['path'])
            if not path.is_absolute():path=Path(upstream_path).resolve().parent/path
            index,_=read_bound(path,bindings,item['sha256'])
            require(len(index)==source['canonical_events'] and len({r['event_id'] for r in index})==len(index),
                    'STOP_KNOWN_SOURCE_POPULATION_MISMATCH')
            source['exact_roots']=len({r['root'] for r in index})
    return {'status':'STOP_UPSTREAM_'+stage.upper(),'segment_id':segment['segment_id'],
        'protocol_sha256':protocol_sha,'preregistration_seal_sha256':seal_sha,
        'reason':upstream.get('reason',upstream['status']),'source':source,
        'upstream_stage':stage,'upstream_status':upstream['status'],'upstream_failure_artifact':ref,
        'binding_scope':'Reporting adapter binds the supplied preregistration and actual stage failure; it does not claim that the failed upstream stage verified every input.',
        'omp_confirmation':{'status':'NOT_RUN'}},ref


def adapt_stage_stop(protocol_path,seal_path,segment_id,upstream_summary_path,stage):
    """Return a bound screen-shaped STOP without writing or promoting counts."""
    bindings={};protocol,pr=read_bound(protocol_path,bindings);seal,sr=read_bound(seal_path,bindings)
    require(seal['protocol_sha256']==pr['sha256'] and seal['status']=='SEALED_BEFORE_NEW_SEGMENT_EXPERIMENTS' and
            seal.get('input_and_method_sha256',{}).get(pr['path'])==pr['sha256'],
            'PROTOCOL_SEAL_BINDING_MISMATCH')
    segments=[s for s in protocol['segments'] if s['segment_id']==segment_id]
    require(len(segments)==1,'UNREGISTERED_UPSTREAM_STOP_SEGMENT')
    result,_=_stage_stop(upstream_summary_path,stage,segments[0],pr['sha256'],sr['sha256'],bindings)
    result['adapter_input_json_sha256']=bindings
    require(all(file_sha(p)==h for p,h in bindings.items()),'STAGE_STOP_INPUT_CHANGED')
    return result


def omitted_row(segment,status,reason):
    return {'segment_id':segment['segment_id'],'scene':segment['scene'],
        'absolute_frames':[segment['first_frame'],segment['last_frame']],
        'status':status,'screen_complete':False,'reason':reason,
        'raw_observations':None,'canonical_events':None,'exact_roots':None,
        'components':None,'admitted_components':None,'admitted_events':None,
        'fallback_components':None,'fallback_events':None,'event_admission_rate':None,
        'component_admission_rate':None,'natural_frames':None,'modified_natural_frames':None,
        'natural_event_frame_replacements':None,'query_count':None,
        'visible_candidate_events':None,'visible_admitted_source_events':None,
        'visible_replacement_events':None,'unknown_support_visibility_events':None,
        'visible_source_admission_rate':None,'compiled_source_visible_candidates':None,
        'compiled_source_visible_admitted':None,'qualified_components':[],
        'eligible_for_preregistered_pilot_selection':False}


def qualified_components(visibility,segment,rule):
    require(rule['select_first_per_segment'] is True and rule['selection_does_not_use_quality_delta'] is True and
        rule['rank']==['segment_id_lexicographic','earliest_qualifying_absolute_frame','component_id_lexicographic'],
        'QUALIFICATION_SELECTION_POLICY_CHANGED')
    rows=visibility['qualified_components'];seen=set()
    for row in rows:
        frames=row['qualifying_absolute_frames'];cid=row['component_id']
        require(row['segment_id']==segment['segment_id'] and isinstance(cid,str) and cid and cid not in seen,
                'DUPLICATE_OR_WRONG_SEGMENT_QUALIFICATION')
        seen.add(cid)
        require(frames==sorted(set(frames)) and len(frames)>=rule['minimum_natural_frames'] and
                all(type(i) is int and segment['first_frame']<=i<=segment['last_frame'] for i in frames),
                'INVALID_QUALIFYING_ORIGINAL_FRAMES')
        require(row['earliest_qualifying_absolute_frame']==frames[0] and
                isinstance(row['events'],list) and row['events'] and len(set(row['events']))==len(row['events']),
                'QUALIFICATION_FIRST_FRAME_OR_MEMBER_BINDING')
    ordered=sorted(rows,key=lambda r:(r['segment_id'],r['earliest_qualifying_absolute_frame'],r['component_id']))
    require(visibility.get('first_qualifying_component')==(ordered[0] if ordered else None),'QUALIFICATION_FIRST_RANK_DISAGREEMENT')
    # Only the preregistered rank/membership fields enter the coverage ledger.
    # Unrelated quality measurements, if present in a broader summary, are not
    # copied into either selected candidates or the report.
    keys=('segment_id','component_id','earliest_qualifying_absolute_frame','qualifying_absolute_frames','events')
    return [{k:row[k] for k in keys} for row in ordered]


def scene_row(summary,segment,protocol_sha,seal_sha,rule,confirmation):
    require(summary['segment_id']==segment['segment_id'] and summary.get('protocol_sha256')==protocol_sha and
            summary.get('preregistration_seal_sha256')==seal_sha,'SEGMENT_PROTOCOL_OR_SEAL_BINDING_MISMATCH')
    complete=(summary.get('status')=='PASS_PREREGISTERED_SEGMENT_SCREEN' and
              summary.get('omp_confirmation',{}).get('status')=='PASS_OMP1_OMP8_IDENTICAL')
    row=omitted_row(segment,summary.get('status','MISSING_STATUS'),summary.get('reason'))
    if 'upstream_failure_artifact' in summary:
        row.update(upstream_failure_artifact=summary['upstream_failure_artifact'],upstream_stage=summary['upstream_stage'],
                   upstream_status=summary['upstream_status'],stage_stop_binding_scope=summary['binding_scope'])
    source=summary.get('source')
    if source is not None:
        for name in ('raw_observations','canonical_events','exact_roots'):
            if source.get(name) is not None:row[name]=count(source[name],name)
        row['source_decision_counts']=source.get('source_decision_counts')
        row['source_counts_scope']='Reported measured source stage; not complete runtime or component admission.'
    if not complete:
        row['reason']=row['reason'] or 'OMP1_OMP8_COMPLETE_CONFIRMATION_NOT_AVAILABLE'
        row['terminal_infrastructure_stop']=summary.get('status','').startswith(('STOP','UNSUPPORTED','INPUT_VERIFICATION_FAILED'))
        row['partial_measurements_not_promoted_to_complete_admission']=True
        return row
    require(confirmation is not None and confirmation.get('status')=='PASS_OMP1_OMP8_IDENTICAL' and
            confirmation.get('segment_id')==segment['segment_id'] and confirmation.get('protocol_sha256')==protocol_sha and
            confirmation.get('preregistration_seal_sha256')==seal_sha,'OMP_CONFIRMATION_NOT_BOUND_TO_THIS_SEGMENT')
    require(all(row[k] is not None for k in ('raw_observations','canonical_events','exact_roots')),'COMPLETE_SCREEN_MISSING_SOURCE_COUNTS')
    n=row['canonical_events'];composition=summary['composition'];schedule=summary['schedule'];vis=summary['visibility']
    for name in ('components','admitted_components','admitted_events','fallback_components','fallback_events'):
        row[name]=count(composition[name],name)
    require(row['admitted_events']+row['fallback_events']==n and
            row['admitted_components']+row['fallback_components']==row['components'] and row['components']<=n,
            'COMPONENT_EVENT_DENOMINATOR_MISMATCH')
    require(row['admitted_components']<=row['admitted_events'] and row['fallback_components']<=row['fallback_events'],
            'COMPONENT_MEMBER_COUNT_MISMATCH')
    modified=schedule['modified_natural_frames']
    require(schedule['natural_frames']==64 and modified==sorted(set(modified)) and
            all(type(i) is int and segment['first_frame']<=i<=segment['last_frame'] for i in modified),
            'INVALID_FIXED_NATURAL_FRAME_SCHEDULE')
    require(count(schedule['query_count'],'query_count')==64+row['exact_roots'],'NATURAL_ROOT_QUERY_DENOMINATOR_MISMATCH')
    event_frames=count(schedule['natural_event_frame_replacements'],'natural_event_frame_replacements')
    require(len(modified)<=event_frames<=64*row['admitted_events'],'NATURAL_EVENT_FRAME_REPLACEMENT_BOUNDS')
    require(vis['canonical_events']==n,'VISIBILITY_CANONICAL_DENOMINATOR_MISMATCH')
    for name in ('visible_candidate_events','visible_admitted_source_events','visible_replacement_events',
                 'unknown_support_visibility_events','compiled_source_visible_candidates','compiled_source_visible_admitted'):
        row[name]=count(vis[name],name)
    require(row['visible_admitted_source_events']<=row['visible_candidate_events']<=n and
            row['visible_admitted_source_events']<=row['admitted_events'] and
            row['visible_replacement_events']<=row['admitted_events'] and row['unknown_support_visibility_events']<=n and
            row['compiled_source_visible_candidates']<=row['visible_candidate_events'] and
            row['compiled_source_visible_admitted']<=min(row['compiled_source_visible_candidates'],row['visible_admitted_source_events']),
            'VISIBILITY_COUNT_INCLUSION_MISMATCH')
    v_rate=rate(row['visible_admitted_source_events'],row['visible_candidate_events']) if not row['unknown_support_visibility_events'] else None
    require(vis['visible_source_admission_rate']==v_rate,'VISIBLE_RATE_MUST_BE_NULL_ON_ZERO_OR_UNKNOWN_DENOMINATOR')
    qualified=qualified_components(vis,segment,rule)
    require(len(qualified)<=row['admitted_components'] and
            all(set(x['qualifying_absolute_frames'])<=set(modified) for x in qualified),'QUALIFIED_COMPONENT_NOT_MODIFIED')
    row.update(screen_complete=True,reason=None,terminal_infrastructure_stop=False,
        event_admission_rate=rate(row['admitted_events'],n),component_admission_rate=rate(row['admitted_components'],row['components']),
        natural_frames=64,modified_natural_frames=modified,natural_event_frame_replacements=event_frames,
        query_count=schedule['query_count'],visible_source_admission_rate=v_rate,qualified_components=qualified,
        eligible_for_preregistered_pilot_selection=True,
        source_counts_scope='Completed fixed-policy requested-schedule screen with OMP1/8 confirmation.',
        qualification_evidence_scope='Hash-bound frozen screening summary; pixel/geometry evidence is inherited, not re-rasterized here.')
    return row


def prior_forest(protocol_path,protocol,seal,bindings):
    path=(Path(protocol_path).resolve().parent/protocol['previous_negative_result']).resolve()
    expected=seal['input_and_method_sha256'].get(str(path));require(expected is not None,'PREVIOUS_NEGATIVE_NOT_SEALED')
    old,old_ref=read_bound(path,bindings,expected)
    require(old['status']=='PASS_REQUESTED_SCHEDULE_STRUCTURAL_COHERENCE' and
            old['atomic_and_identity_checks']['omp1_omp8_all_66_receipts_identical'] is True,'PREVIOUS_FOREST_STRUCTURE_NOT_COMPLETE')
    vr=old['artifact_references']['visibility'];vp=Path(vr['path'])
    if not vp.is_absolute():vp=path.parent/vp
    vis,vis_ref=read_bound(vp,bindings,vr['sha256'])
    require(vis['status']=='PASS_COMPLETE_NATURAL_SUPPORT_VISIBILITY_TRIAGE' and
            vis['certification_sha256']==old['artifact_references']['component_certification']['sha256'] and
            vis['sequence_sha256'] in {old['artifact_references'][name]['sha256'] for name in ('sequence_omp1','sequence_omp8')},
            'PREVIOUS_VISIBILITY_STRUCTURE_BINDING_MISMATCH')
    p=old['population'];s=old['schedule'];n=p['canonical_events']
    require(vis['candidate_events_evaluated']==n and vis['visible_candidate_support_events']==0 and
            vis['jointly_admitted_visible_source_events']==0 and vis['visible_joint_replacement_events']==0 and
            vis['visible_support_admission_rate'] is None,'PREVIOUS_ZERO_VISIBLE_NEGATIVE_CHANGED')
    row=omitted_row({'segment_id':'forest_a_previous','scene':'Forest','first_frame':1,'last_frame':64},'PREVIOUS_COMPLETED_NEGATIVE',None)
    row.update(screen_complete=True,canonical_events=n,exact_roots=s['exact_root_diagnostics'],
        components=p['support_components'],admitted_components=p['jointly_admitted_components'],admitted_events=p['jointly_admitted_events'],
        fallback_components=p['fail_closed_components'],fallback_events=p['fail_closed_events'],event_admission_rate=p['event_admission_rate'],
        component_admission_rate=p['component_admission_rate'],natural_frames=s['natural_frames'],
        modified_natural_frames=s['modified_natural_frames'],natural_event_frame_replacements=s['natural_event_frame_replacements'],
        query_count=s['five_element_scene_outputs_per_omp'],visible_candidate_events=0,visible_admitted_source_events=0,
        visible_replacement_events=0,unknown_support_visibility_events=0,visible_source_admission_rate=None,
        compiled_source_visible_candidates=vis['source_contract_only_visible_events'],
        compiled_source_visible_admitted=vis['source_contract_only_visible_admitted_events'],
        prior_negative_preserved=True,eligible_for_preregistered_pilot_selection=False,
        artifact_references={'structural_coherence':old_ref,'visibility':vis_ref},
        note='Raw observation count is not in the prior compact structural manifest; null is unreported, not zero. Visible rate is null (0/0).')
    return row


def selection(rows,rule):
    completed=[r for r in rows if r['screen_complete']]
    selected=[r['qualified_components'][0] for r in completed if r['qualified_components']]
    selected.sort(key=lambda x:(x['segment_id'],x['earliest_qualifying_absolute_frame'],x['component_id']))
    selected=selected[:rule['maximum_selected_components']]
    terminal=all(r['screen_complete'] or r.get('terminal_infrastructure_stop') is True for r in rows)
    incomplete=[r['segment_id'] for r in rows if not r['screen_complete']]
    return {'all_three_segments_terminal':terminal,'selected_components':selected if terminal else [],
        'provisional_first_candidates_from_completed_segments':selected if not terminal else [],
        'pilot_selection_ready':terminal and bool(selected),
        'all_three_completed_zero_qualifying_components':all(r['screen_complete'] for r in rows) and not selected,
        'uncompleted_segments':incomplete,
        'ranking_scope':'First by the frozen rank among completed segments only; not a claimed first across stopped or unmeasured segments.' if incomplete else
                        'First per segment by the frozen rank across all three completed preregistered segments.',
        'uses_quality_delta':False,'post_displacement_gate_still_required':True,
        'visual_improvement_established':False}


def markdown(report):
    def value(x):return 'N/A' if x is None else str(x)
    lines=['# Preregistered coverage ledger','',
        'Source admission and visible source-support coverage are distinct. N/A means unmeasured, unresolved, or an empty denominator—not 0%.','',
        '| Segment | Status | Events | Joint events / components | Modified natural frames | Visible source / admitted | Visible rate |',
        '| --- | --- | ---: | --- | ---: | --- | --- |']
    for row in report['rows']:
        visible_rate='N/A' if row['visible_source_admission_rate'] is None else f"{100*row['visible_source_admission_rate']:.2f}%"
        lines.append('| '+ ' | '.join([row['segment_id'],row['status'],value(row['canonical_events']),
            value(row['admitted_events'])+' / '+value(row['admitted_components']),
            value(None if row['modified_natural_frames'] is None else len(row['modified_natural_frames'])),
            value(row['visible_candidate_events'])+' / '+value(row['visible_admitted_source_events']),visible_rate])+' |')
    lines+=['','The previous Forest negative is retained. A STOP or missing segment is never counted as zero events or zero visibility.','',
        '## Frozen selection','',report['selection']['ranking_scope'],'',
        'Pilot selection ready: '+str(report['selection']['pilot_selection_ready']).lower()+'. Post-displacement checks remain required.','']
    for row in report['rows']:
        if row.get('reason'):lines.append('- '+row['segment_id']+': '+str(row['reason']).replace('\n',' '))
    lines+=['','No quality delta was read for ranking. Coverage is not evidence of visual improvement, artifact-impact coverage, all-time admission, or post-displacement safety.','']
    return '\n'.join(lines)


def assemble_coverage(protocol_path,seal_path,scene_reports,output_dir):
    output=Path(output_dir).resolve();require(not output.exists(),'COVERAGE_OUTPUT_MUST_BE_FRESH')
    bindings={};protocol,pr=read_bound(protocol_path,bindings);seal,sr=read_bound(seal_path,bindings)
    require(protocol['schema']=='preregistered-paper-scene-visibility-screen-v1' and
            protocol['registration_state'].startswith('LOCKED_BEFORE_') and
            seal['schema']=='visibility-screen-preregistration-seal-v1' and seal['status']=='SEALED_BEFORE_NEW_SEGMENT_EXPERIMENTS' and
            seal['protocol_sha256']==pr['sha256'] and seal['input_and_method_sha256'].get(pr['path'])==pr['sha256'],
            'PROTOCOL_SEAL_BINDING_MISMATCH')
    segments=protocol['segments'];require(len(segments)==3 and {s['segment_id'] for s in segments}=={'forest_b','cave','mountain'},'PREREGISTERED_POPULATION_CHANGED')
    require(set(scene_reports)<={s['segment_id'] for s in segments},'UNREGISTERED_SCENE_REPORT_SUPPLIED')
    previous=prior_forest(protocol_path,protocol,seal,bindings);rows=[];rule=protocol['visibility']['qualifying_component_rule']
    for segment in segments:
        path=scene_reports.get(segment['segment_id'])
        if path is None:
            rows.append(omitted_row(segment,'NOT_RUN_OR_REPORT_MISSING','No completed or STOP summary supplied.'));continue
        if isinstance(path,dict):
            require(set(path)=={'stage','summary'},'UPSTREAM_STOP_SPEC_REQUIRES_STAGE_AND_SUMMARY')
            summary,ref=_stage_stop(path['summary'],path['stage'],segment,pr['sha256'],sr['sha256'],bindings)
            path=path['summary']
        else:summary,ref=read_bound(path,bindings)
        confirmation=None;cref=None;confirmed_refs=[]
        if summary.get('status')=='PASS_PREREGISTERED_SEGMENT_SCREEN':
            record=summary.get('omp_confirmation',{}).get('artifact');require(isinstance(record,dict),'COMPLETE_SCREEN_MISSING_OMP_ARTIFACT')
            target=Path(record['path']);target=target if target.is_absolute() else Path(path).resolve().parent/target
            confirmation,cref=read_bound(target,bindings,record['sha256'])
            confirmed_refs=verify_confirmed_summaries(summary,confirmation,target,bindings)
        row=scene_row(summary,segment,pr['sha256'],sr['sha256'],rule,confirmation)
        row['artifact_references']={'summary':ref}
        if cref is not None:row['artifact_references']['omp_confirmation']=cref
        if confirmed_refs:row['artifact_references']['confirmed_omp_summaries']=confirmed_refs
        rows.append(row)
    report={'schema':'preregistered-scene-coverage-ledger-v1','status':'PASS_HASH_BOUND_COVERAGE_LEDGER',
        'protocol':pr,'preregistration_seal':sr,'rows':[previous,*rows],'selection':selection(rows,rule),
        'new_segments_completed':sum(r['screen_complete'] for r in rows),'new_segments_denominator':3,
        'previous_negative_excluded_from_new_segment_selection':True,
        'rates_are_per_segment_not_pooled_across_changed_caches':True,
        'no_quality_metrics_read_or_used_for_selection':True,
        'scope':'Hash-bound summary aggregation; source/geometry/OMP/pixel evidence is inherited from campaign artifacts, not re-executed.',
        'not_claimed':['all-real-time admission','post-displacement safety','visual quality improvement','visible artifact-impact coverage'],
        'input_json_sha256':bindings,'executed_script_sha256':file_sha(__file__)}
    require(all(file_sha(p)==h for p,h in bindings.items()),'COVERAGE_INPUT_CHANGED_DURING_ASSEMBLY')
    output.mkdir(parents=True)
    with (output/'coverage.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    with (output/'coverage.md').open('x',encoding='utf-8') as stream:stream.write(markdown(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','seal','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scene-report',action='append',default=[],help='segment_id=summary.json (omit missing segments)')
    parser.add_argument('--stage-stop',action='append',default=[],help='segment_id:build|source=upstream_summary.json')
    args=parser.parse_args();inputs={}
    for item in args.scene_report:
        key,value=item.split('=',1);require(key not in inputs,'DUPLICATE_CLI_SEGMENT');inputs[key]=Path(value)
    for item in args.stage_stop:
        label,value=item.split('=',1);key,stage=label.split(':',1)
        require(key not in inputs,'DUPLICATE_CLI_SEGMENT');inputs[key]={'stage':stage,'summary':Path(value)}
    report=assemble_coverage(args.protocol,args.seal,inputs,args.output)
    print(json.dumps({'status':report['status'],'new_segments_completed':report['new_segments_completed'],
                      'pilot_selection_ready':report['selection']['pilot_selection_ready']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
