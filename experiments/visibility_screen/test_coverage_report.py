"""Small synthetic coverage and selection tests; no scene results are read."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import coverage_report as m


RULE={'select_first_per_segment':True,'selection_does_not_use_quality_delta':True,
    'rank':['segment_id_lexicographic','earliest_qualifying_absolute_frame','component_id_lexicographic'],
    'minimum_natural_frames':2,'minimum_baseline_support_pixels_per_frame':16,
    'minimum_actual_replacement_pixels_on_each_qualifying_frame':1,'maximum_selected_components':3}


def fixture(sid='forest_b',qualified=True):
    first=97 if sid=='forest_b' else 1
    segment={'segment_id':sid,'scene':sid,'first_frame':first,'last_frame':first+63}
    entry={'segment_id':sid,'component_id':'c1','earliest_qualifying_absolute_frame':first,
        'qualifying_absolute_frames':[first,first+1],'events':['e1','e2']}
    confirmation={'status':'PASS_OMP1_OMP8_IDENTICAL','segment_id':sid,'protocol_sha256':'p','preregistration_seal_sha256':'s'}
    vis={'canonical_events':3,'visible_candidate_events':2 if qualified else 0,'visible_admitted_source_events':2 if qualified else 0,
        'visible_replacement_events':2 if qualified else 0,'unknown_support_visibility_events':0,
        'visible_source_admission_rate':1 if qualified else None,'compiled_source_visible_candidates':2 if qualified else 0,
        'compiled_source_visible_admitted':2 if qualified else 0,'qualified_components':[entry] if qualified else [],
        'first_qualifying_component':entry if qualified else None}
    summary={'status':'PASS_PREREGISTERED_SEGMENT_SCREEN','segment_id':sid,'protocol_sha256':'p','preregistration_seal_sha256':'s',
        'source':{'raw_observations':6,'canonical_events':3,'exact_roots':1,'source_decision_counts':{'pass':3}},
        'composition':{'components':2,'admitted_components':1,'admitted_events':2,'fallback_components':1,'fallback_events':1},
        'schedule':{'natural_frames':64,'modified_natural_frames':[first,first+1],
            'natural_event_frame_replacements':4,'query_count':65},
        'visibility':vis,'omp_confirmation':{'status':'PASS_OMP1_OMP8_IDENTICAL'}}
    return summary,segment,confirmation


def synthetic_prior(directory):
    vis={'status':'PASS_COMPLETE_NATURAL_SUPPORT_VISIBILITY_TRIAGE','certification_sha256':'c'*64,'sequence_sha256':'a'*64,
        'candidate_events_evaluated':131,'visible_candidate_support_events':0,'jointly_admitted_visible_source_events':0,
        'visible_joint_replacement_events':0,'visible_support_admission_rate':None,
        'source_contract_only_visible_events':0,'source_contract_only_visible_admitted_events':0}
    vp=directory/'old_visibility.json';vp.write_text(json.dumps(vis))
    old={'status':'PASS_REQUESTED_SCHEDULE_STRUCTURAL_COHERENCE',
        'atomic_and_identity_checks':{'omp1_omp8_all_66_receipts_identical':True},
        'artifact_references':{'visibility':{'path':str(vp),'sha256':m.file_sha(vp)},
            'component_certification':{'sha256':'c'*64},'sequence_omp1':{'sha256':'a'*64},'sequence_omp8':{'sha256':'b'*64}},
        'population':{'canonical_events':131,'support_components':79,'jointly_admitted_components':20,
            'jointly_admitted_events':25,'fail_closed_components':59,'fail_closed_events':106,
            'event_admission_rate':25/131,'component_admission_rate':20/79},
        'schedule':{'exact_root_diagnostics':2,'natural_frames':64,'modified_natural_frames':[21,22],
            'natural_event_frame_replacements':200,'five_element_scene_outputs_per_omp':66}}
    op=directory/'old_structure.json';op.write_text(json.dumps(old));return op,vp


def write_json(path,value):
    path.write_text(json.dumps(value));return {'path':str(path),'sha256':m.file_sha(path)}


def confirmed_fixture(directory):
    combined,segment,confirmation=fixture()
    primary=deepcopy(combined)
    primary.update(status='PASS_PREREGISTERED_SEGMENT_SCREEN_OMP1_ONLY',omp_num_threads='1',
        private_cache_removed=True,partial_output_published=False,final_input_verification={'status':'PASS'},
        support_graph={'path':'graph.json','sha256':'g'},certified_events={'path':'events.json','sha256':'e'},
        component_queries={'path':'queries.json','sha256':'q'},frame_artifacts=[],
        visibility_frame_artifacts=[],visibility_artifacts_base=str(directory))
    first=write_json(directory/'omp1.json',primary)
    secondary=deepcopy(primary);secondary.update(status='PASS_OMP8_SEQUENCE_REPLAY',omp_num_threads='8',primary_summary=first)
    second=write_json(directory/'omp8.json',secondary)
    confirmation.update(omp1_summary=first,omp8_summary=second,queries_compared=65,
        all_arrays_records_component_membership_and_complete_root_supports_identical=True)
    combined={**primary,'status':'PASS_PREREGISTERED_SEGMENT_SCREEN'}
    cp=directory/'confirmation.json';write_json(cp,confirmation)
    return combined,confirmation,cp


def registration_fixture(directory):
    old,_=synthetic_prior(directory)
    protocol={'schema':'preregistered-paper-scene-visibility-screen-v1','registration_state':'LOCKED_BEFORE_TEST',
        'previous_negative_result':old.name,'segments':[fixture(s)[1] for s in ('forest_b','cave','mountain')],
        'visibility':{'qualifying_component_rule':RULE}}
    pp=directory/'protocol.json';write_json(pp,protocol)
    seal={'schema':'visibility-screen-preregistration-seal-v1','status':'SEALED_BEFORE_NEW_SEGMENT_EXPERIMENTS',
        'protocol_sha256':m.file_sha(pp),'input_and_method_sha256':{str(pp):m.file_sha(pp),str(old):m.file_sha(old)}}
    sp=directory/'seal.json';write_json(sp,seal)
    return pp,sp,protocol


class CoverageTests(unittest.TestCase):
    def row(self,summary,segment,confirmation):return m.scene_row(summary,segment,'p','s',RULE,confirmation)

    def test_complete_source_and_visible_rates_remain_distinct(self):
        row=self.row(*fixture())
        self.assertEqual(row['event_admission_rate'],2/3)
        self.assertEqual(row['visible_source_admission_rate'],1)
        self.assertTrue(row['screen_complete'])

    def test_zero_visible_denominator_is_null_not_zero_percent(self):
        data=fixture(qualified=False);row=self.row(*data)
        self.assertIsNone(row['visible_source_admission_rate'])
        data[0]['visibility']['visible_source_admission_rate']=0
        with self.assertRaisesRegex(ValueError,'MUST_BE_NULL'):self.row(*data)

    def test_zero_registry_is_valid_completed_zero_denominator(self):
        summary,segment,confirmation=fixture(qualified=False)
        summary['source'].update(raw_observations=0,canonical_events=0,exact_roots=0)
        summary['composition']={key:0 for key in summary['composition']}
        summary['schedule'].update(modified_natural_frames=[],natural_event_frame_replacements=0,query_count=64)
        summary['visibility']['canonical_events']=0
        row=self.row(summary,segment,confirmation)
        self.assertTrue(row['screen_complete']);self.assertIsNone(row['event_admission_rate'])

    def test_unknown_visibility_prevents_rate_even_with_visible_candidates(self):
        data=fixture();data[0]['visibility'].update(unknown_support_visibility_events=1,visible_source_admission_rate=None)
        self.assertIsNone(self.row(*data)['visible_source_admission_rate'])

    def test_omp1_only_never_promoted_to_complete_or_selected(self):
        data=fixture();data[0]['status']='PASS_PREREGISTERED_SEGMENT_SCREEN_OMP1_ONLY'
        data[0]['omp_confirmation']={'status':'NOT_RUN'}
        row=self.row(*data)
        self.assertEqual(row['canonical_events'],3)
        self.assertFalse(row['screen_complete']);self.assertIsNone(row['admitted_events'])
        self.assertFalse(row['eligible_for_preregistered_pilot_selection'])

    def test_stop_preserves_measured_source_and_null_unmeasured_admission(self):
        data=fixture();data[0].update(status='STOP_NATIVE_READER',reason='infrastructure limit')
        row=self.row(*data)
        self.assertEqual(row['canonical_events'],3);self.assertIsNone(row['visible_candidate_events'])
        self.assertTrue(row['terminal_infrastructure_stop'])

    def test_confirmation_cannot_be_from_another_segment(self):
        data=fixture();data[2]['segment_id']='cave'
        with self.assertRaisesRegex(ValueError,'CONFIRMATION_NOT_BOUND'):self.row(*data)

    def test_qualification_uses_rank_not_quality(self):
        data=fixture();vis=data[0]['visibility'];first=deepcopy(vis['qualified_components'][0])
        first.update(component_id='a0',quality_delta=-999)
        other=deepcopy(first);other.update(component_id='z0',quality_delta=999)
        vis.update(qualified_components=[other,first],first_qualifying_component=first)
        data[0]['composition'].update(components=3,admitted_components=2,fallback_components=1)
        row=self.row(*data)
        self.assertEqual(row['qualified_components'][0]['component_id'],'a0')
        self.assertNotIn('quality_delta',row['qualified_components'][0])

    def test_one_frame_is_not_a_qualifying_component(self):
        data=fixture();data[0]['visibility']['qualified_components'][0]['qualifying_absolute_frames']=[97]
        with self.assertRaisesRegex(ValueError,'INVALID_QUALIFYING'):self.row(*data)

    def test_stopped_earlier_segment_does_not_claim_global_first(self):
        rows=[self.row(*fixture('cave')),self.row(*fixture('mountain'))]
        segment=fixture('forest_b')[1]
        stopped=m.omitted_row(segment,'STOP_BUILD','limit');stopped['terminal_infrastructure_stop']=True
        result=m.selection([rows[0],stopped,rows[1]],RULE)
        self.assertTrue(result['pilot_selection_ready'])
        self.assertIn('among completed segments only',result['ranking_scope'])
        self.assertEqual([r['segment_id'] for r in result['selected_components']],['cave','mountain'])

    def test_missing_segment_keeps_pilot_gate_closed(self):
        rows=[self.row(*fixture('cave')),self.row(*fixture('mountain')),
              m.omitted_row(fixture('forest_b')[1],'NOT_RUN','missing')]
        result=m.selection(rows,RULE)
        self.assertFalse(result['pilot_selection_ready']);self.assertEqual(result['selected_components'],[])
        self.assertEqual(len(result['provisional_first_candidates_from_completed_segments']),2)

    def test_confirmed_runs_are_hash_bound_and_exactly_equal(self):
        with tempfile.TemporaryDirectory() as temp:
            data=confirmed_fixture(Path(temp));bindings={}
            refs=m.verify_confirmed_summaries(*data,bindings)
            self.assertEqual(len(refs),2);self.assertEqual(len(bindings),2)

    def test_changed_combined_visibility_cannot_borrow_pass_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            data=confirmed_fixture(Path(temp))
            data[0]['visibility']=deepcopy(data[0]['visibility']);data[0]['visibility']['visible_candidate_events']=3
            with self.assertRaisesRegex(ValueError,'NOT_EQUAL_CONFIRMED_OMP1'):
                m.verify_confirmed_summaries(*data,{})

    def test_tampered_original_omp_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);data=confirmed_fixture(directory)
            (directory/'omp1.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):
                m.verify_confirmed_summaries(*data,{})

    def test_replay_of_another_primary_cannot_confirm(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);data=confirmed_fixture(directory)
            doc=json.loads((directory/'omp8.json').read_text());doc['primary_summary']['sha256']='wrong'
            data[1]['omp8_summary']=write_json(directory/'omp8.json',doc)
            with self.assertRaisesRegex(ValueError,'REFERENCES_DIFFERENT_OMP1'):
                m.verify_confirmed_summaries(*data,{})

    def test_changed_visibility_artifact_base_cannot_borrow_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            data=confirmed_fixture(Path(temp));data[0]['visibility_artifacts_base']='another_attempt'
            with self.assertRaisesRegex(ValueError,'NOT_EQUAL_CONFIRMED_OMP1'):
                m.verify_confirmed_summaries(*data,{})

    def test_bound_build_stop_keeps_unmeasured_columns_null(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);pp,sp,protocol=registration_fixture(directory)
            contract=write_json(directory/'build_protocol.json',{'segment':protocol['segments'][0],'protocol_sha256':m.file_sha(pp)})
            path=directory/'failed_build.json';write_json(path,{'status':'STOP_INFRASTRUCTURE_BUILD_INCOMPLETE',
                'segment_id':'forest_b','reason':'wall budget','build_protocol_sha256':contract['sha256']})
            adapted=m.adapt_stage_stop(pp,sp,'forest_b',path,'build')
            self.assertIsNone(adapted['source']);self.assertEqual(adapted['upstream_failure_artifact']['sha256'],m.file_sha(path))
            result=m.assemble_coverage(pp,sp,{'forest_b':{'stage':'build','summary':path}},directory/'coverage')
            row=result['rows'][1]
            self.assertTrue(row['terminal_infrastructure_stop']);self.assertIsNone(row['canonical_events'])
            self.assertIsNone(row['admitted_events']);self.assertEqual(row['reason'],'wall budget')

    def test_source_stop_retains_bound_known_population_but_not_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);pp,sp,protocol=registration_fixture(directory)
            segment=protocol['segments'][0];spec_hash=m.hashlib.sha256(json.dumps(segment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            index=write_json(directory/'events_index.json',[{'event_id':'a','root':'3/2'},{'event_id':'b','root':'3/2'}])
            path=directory/'source_stop.json';write_json(path,{'status':'STOP_SOURCE_STAGE','segment_id':'forest_b',
                'segment_spec_binding':{'kind':'CANONICAL_JSON','sha256':spec_hash},'canonical_event_denominator':2,
                'events_index':index,'source_status_counts':{'NOT_REACHED':2}})
            adapted=m.adapt_stage_stop(pp,sp,'forest_b',path,'source')
            self.assertEqual(adapted['source']['canonical_events'],2);self.assertEqual(adapted['source']['exact_roots'],1)
            self.assertIsNone(adapted['source']['raw_observations'])
            row=m.scene_row(adapted,segment,m.file_sha(pp),m.file_sha(sp),RULE,None)
            self.assertEqual(row['canonical_events'],2);self.assertIsNone(row['event_admission_rate'])

    def test_source_stop_unknown_population_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);pp,sp,protocol=registration_fixture(directory)
            ref=write_json(directory/'segment.json',protocol['segments'][0])
            path=directory/'source_stop.json';write_json(path,{'status':'STOP_SOURCE_STAGE','segment_id':'forest_b',
                'segment_spec_binding':ref,'canonical_event_denominator':None,'source_status_counts':{}})
            result=m.adapt_stage_stop(pp,sp,'forest_b',path,'source')
            self.assertIsNone(result['source']['canonical_events']);self.assertIsNone(result['source']['exact_roots'])

    def test_build_stop_from_another_preregistration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);pp,sp,protocol=registration_fixture(directory)
            contract=write_json(directory/'build_protocol.json',{'segment':protocol['segments'][0],'protocol_sha256':'other'})
            path=directory/'stop.json';write_json(path,{'status':'STOP_BUILD','segment_id':'forest_b','build_protocol_sha256':contract['sha256']})
            with self.assertRaisesRegex(ValueError,'PREREGISTRATION_MISMATCH'):
                m.adapt_stage_stop(pp,sp,'forest_b',path,'build')

    def test_nonstop_cannot_be_adapted_as_terminal_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);pp,sp,_=registration_fixture(directory)
            path=directory/'complete.json';write_json(path,{'status':'COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION','segment_id':'forest_b'})
            with self.assertRaisesRegex(ValueError,'NOT_A_STOP'):
                m.adapt_stage_stop(pp,sp,'forest_b',path,'source')

    def test_complete_paper_ledger_always_keeps_old_negative_and_three_new_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);old,_=synthetic_prior(directory)
            protocol={'schema':'preregistered-paper-scene-visibility-screen-v1','registration_state':'LOCKED_BEFORE_TEST',
                'previous_negative_result':old.name,'segments':[fixture(s)[1] for s in ('forest_b','cave','mountain')],
                'visibility':{'qualifying_component_rule':RULE}}
            pp=directory/'protocol.json';pp.write_text(json.dumps(protocol))
            seal={'schema':'visibility-screen-preregistration-seal-v1','status':'SEALED_BEFORE_NEW_SEGMENT_EXPERIMENTS',
                'protocol_sha256':m.file_sha(pp),'input_and_method_sha256':{str(pp):m.file_sha(pp),str(old):m.file_sha(old)}}
            sp=directory/'seal.json';sp.write_text(json.dumps(seal))
            result=m.assemble_coverage(pp,sp,{},directory/'coverage')
            self.assertEqual(len(result['rows']),4)
            self.assertEqual(result['rows'][0]['visible_candidate_events'],0)
            self.assertTrue(all(r['canonical_events'] is None for r in result['rows'][1:]))
            self.assertIn('N/A',(directory/'coverage/coverage.md').read_text())
            self.assertFalse(result['selection']['pilot_selection_ready'])
            with self.assertRaisesRegex(ValueError,'MUST_BE_FRESH'):
                m.assemble_coverage(pp,sp,{},directory/'coverage')


if __name__=='__main__':unittest.main()
