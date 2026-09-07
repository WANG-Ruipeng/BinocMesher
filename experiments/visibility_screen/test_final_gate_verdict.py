import unittest
from final_gate_verdict import classify


def rows():
    return [{'segment_id': s, 'screen_complete': True, 'visible_candidate_events': 0,
             'visible_replacement_events': 0, 'unknown_support_visibility_events': 0,
             'qualified_components': []} for s in ('cave','forest_b','mountain')]


class FinalGateVerdictTests(unittest.TestCase):
    def test_complete_zeros_stop_only_visual_headline(self):
        result = classify(rows())
        self.assertEqual(result['decision'], 'STOP_ORIGINAL_CAMERA_VISUAL_HEADLINE')
        self.assertFalse(result['stop_method'])

    def test_resource_stop_is_not_scientific_zero(self):
        data = rows(); data[0].update(screen_complete=False, terminal_infrastructure_stop=True,
                                     visible_candidate_events=None,visible_replacement_events=None)
        result = classify(data)
        self.assertEqual(result['decision'], 'RESOURCE_OR_EVIDENCE_UNRESOLVED_STOP_VISIBILITY_CAMPAIGN')
        self.assertEqual(result['uncompleted_new_segments'], ['cave'])

    def test_pending_second_attempt_does_not_select_early(self):
        data = rows(); data[1]['screen_complete'] = False
        data[0]['qualified_components'] = [dict(segment_id='cave',earliest_qualifying_absolute_frame=21,component_id='a')]
        self.assertEqual(classify(data)['decision'], 'WAIT_FIXED_SECOND_ATTEMPTS_NOT_ALL_TERMINAL')
        self.assertIsNone(classify(data)['first_qualifying_component'])

    def test_visible_rejected_candidates_allow_only_diagnosis(self):
        data = rows(); data[0]['visible_candidate_events'] = 1
        result = classify(data)
        self.assertEqual(result['decision'], 'VISIBLE_CANDIDATES_WITHOUT_VISIBLE_REPLACEMENT_FIXED_DIAGNOSIS_ONLY')
        self.assertFalse(result['automatic_method_extension_authorized'])

    def test_small_visible_replacement_does_not_weaken_pixel_frame_gate(self):
        data = rows(); data[0].update(visible_candidate_events=1,visible_replacement_events=1)
        self.assertEqual(classify(data)['decision'], 'VISIBLE_REPLACEMENT_BELOW_PREREGISTERED_PILOT_QUALIFICATION')

    def test_qualified_selection_uses_fixed_rank_not_quality(self):
        data = rows()
        for row in data[:2]:
            row.update(visible_candidate_events=1,visible_replacement_events=1,
                       qualified_components=[dict(segment_id=row['segment_id'],earliest_qualifying_absolute_frame=25,
                                                  component_id='fixed-id')])
        result = classify(data)
        self.assertEqual(result['first_qualifying_component']['segment_id'], 'cave')
        self.assertFalse(result['visual_improvement_established'])

    def test_unknown_visibility_is_not_zero(self):
        data = rows(); data[0]['unknown_support_visibility_events'] = 1
        self.assertEqual(classify(data)['decision'], 'RESOURCE_OR_EVIDENCE_UNRESOLVED_STOP_VISIBILITY_CAMPAIGN')

    def test_changed_population_is_rejected(self):
        with self.assertRaises(ValueError): classify(rows()[:2])


from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import final_gate_verdict as m


def chain_fixture():
    amendment = {'amendment_path': '/e/budget_amendment01.json', 'amendment_sha256': 'a'*64,
        'protocol_sha256': 'p'*64, 'preregistration_seal_sha256': 's'*64,
        'effective_budgets': {'cache': 8}, 'retry_sources': {'cave': {'source_coarse': '/old/coarse'}},
        'effective_protocol': {'segments': [{'segment_id': 'cave'}],
                               'visibility': {'qualifying_component_rule': {}}}}
    binding = {'path': amendment['amendment_path'], 'sha256': amendment['amendment_sha256']}
    build_ref = {'path': '/e/cave/build02/summary.json', 'sha256': 'b'*64}
    contract_ref = {'path': '/e/cave/build02/build_protocol.json', 'sha256': 'c'*64}
    build = {'segment_id': 'cave', 'budget_amendment': binding,
        'status': 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE', 'output': '/data/cave_attempt02',
        'started_at_utc': '2026-09-06T14:00:00+00:00', 'finished_at_utc': '2026-09-06T15:00:00+00:00',
        'stages': {'native': {'exit_code': 0, 'infrastructure_stop': None}},
        'native_complete': {'status': 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'},
        'original_source_inputs_unchanged': True, 'final_bound_inputs_unchanged': True}
    contract = {'output': build['output'], 'segment': {'segment_id': 'cave'}, 'budget_amendment': binding,
        'protocol_sha256': 'p'*64, 'effective_budgets': {'cache': 8}, 'source_reuse': {'source_coarse': '/old/coarse'},
        'worker_uses_original_protocol_unchanged': True, 'only_supervisor_native_wall_and_output_ceiling_amended': True}
    row = {'segment_id': 'cave', 'screen_complete': True}
    outcome_ref = {'path': '/e/cave/screen02/combined_summary.json', 'sha256': 'd'*64}
    outcome = {'segment_id': 'cave', 'budget_amendment': binding,
        'status': 'PASS_PREREGISTERED_SEGMENT_SCREEN',
        'input_and_code_sha256': {build_ref['path']: build_ref['sha256'], contract_ref['path']: contract_ref['sha256']}}
    return amendment, build, contract, build_ref, contract_ref, row, outcome, outcome_ref


class FinalOutcomeChainTests(unittest.TestCase):
    def test_real_second_build_completion_and_link_are_accepted(self):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        self.assertTrue(m.validate_retry_build(b,c,'cave',a))
        self.assertTrue(m.validate_outcome_binding(row,out,oref,b,br,cr,a))

    def test_build_running_or_missing_finished_time_cannot_close_gate(self):
        for mutate in ('running','missing_finish'):
            a,b,c,*_ = chain_fixture()
            if mutate == 'running': b['status'] = 'RUNNING'
            else: del b['finished_at_utc']
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                m.validate_retry_build(b,c,'cave',a)

    def test_failed_native_cannot_claim_complete_build(self):
        a,b,c,*_ = chain_fixture(); b['stages']['native']['exit_code'] = -15
        with self.assertRaisesRegex(ValueError,'COMPLETION_NOT_VERIFIED'): m.validate_retry_build(b,c,'cave',a)

    def test_first_attempt_cannot_be_relabelled_second_by_amendment(self):
        a,b,c,*_ = chain_fixture(); b['output'] = c['output'] = '/data/cave_attempt01'
        with self.assertRaisesRegex(ValueError,'NOT_FIXED_SECOND'): m.validate_retry_build(b,c,'cave',a)

    def test_old_stop_plus_new_build_receipt_mixture_is_rejected(self):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        row.update(screen_complete=False,terminal_infrastructure_stop=True,upstream_stage='build',
                   upstream_failure_artifact={'path':'/e/cave/build01/summary.json','sha256':'old'})
        old = {'segment_id':'cave','status':'STOP_INFRASTRUCTURE_BUILD_INCOMPLETE','budget_amendment':out['budget_amendment']}
        with self.assertRaisesRegex(ValueError,'NOT_THIS_SECOND_BUILD'):
            m.validate_outcome_binding(row,old,row['upstream_failure_artifact'],b,br,cr,a)

    def test_complete_build_is_not_a_complete_screen_or_terminal_stop(self):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        row.update(screen_complete=False,terminal_infrastructure_stop=True,upstream_stage='build',upstream_failure_artifact=br)
        with self.assertRaisesRegex(ValueError,'NOT_THIS_SECOND_BUILD'):
            m.validate_outcome_binding(row,b,br,b,br,cr,a)

    def test_resource_stopped_second_build_can_remain_unmeasured_terminal(self):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        b['status']='STOP_INFRASTRUCTURE_BUILD_INCOMPLETE'
        row.update(screen_complete=False,terminal_infrastructure_stop=True,upstream_stage='build',upstream_failure_artifact=br)
        self.assertFalse(m.validate_retry_build(b,c,'cave',a))
        self.assertTrue(m.validate_outcome_binding(row,b,br,b,br,cr,a))

    def test_source_stop_is_accepted_only_with_same_second_build_and_contract_hashes(self):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        row.update(screen_complete=False,terminal_infrastructure_stop=True,upstream_stage='source')
        out['status']='STOP_SOURCE_STAGE_NOT_COMPLETE'
        out['amendment_input_sha256']=out.pop('input_and_code_sha256')
        self.assertTrue(m.validate_outcome_binding(row,out,oref,b,br,cr,a))
        out['amendment_input_sha256'][br['path']]='wrong'
        with self.assertRaisesRegex(ValueError,'NOT_BOUND_TO_THIS_SECOND_BUILD'):
            m.validate_outcome_binding(row,out,oref,b,br,cr,a)

    def test_unbound_screen_stop_requires_external_evidence_not_assumed_chain(self):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        row.update(screen_complete=False,terminal_infrastructure_stop=True)
        out['status']='STOP_SCREEN_ATTEMPT_NOT_COMPLETE'; del out['input_and_code_sha256']
        with self.assertRaisesRegex(ValueError,'NOT_BOUND_TO_THIS_SECOND_BUILD'):
            m.validate_outcome_binding(row,out,oref,b,br,cr,a)

    def test_source_complete_with_no_screen_is_pending_not_terminal(self):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        row.update(screen_complete=False,terminal_infrastructure_stop=True,upstream_stage='source')
        out['status']='COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION'
        out['amendment_input_sha256']=out['input_and_code_sha256']
        with self.assertRaisesRegex(ValueError,'SOURCE_NOT_A_TERMINAL_STOP'):
            m.validate_outcome_binding(row,out,oref,b,br,cr,a)

    def test_other_amendment_cannot_supply_completed_screen(self):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        out['budget_amendment']={'path':'/e/other.json','sha256':'a'*64}
        with self.assertRaisesRegex(ValueError,'NOT_FROM_FINAL_AMENDMENT'):
            m.validate_outcome_binding(row,out,oref,b,br,cr,a)

    def test_complete_row_missing_unknown_count_is_not_zero(self):
        data=rows(); del data[0]['unknown_support_visibility_events']
        with self.assertRaisesRegex(ValueError,'UNKNOWN_VISIBILITY_COUNT_REQUIRED'): classify(data)

    def test_corrupted_saved_mountain_reference_rejected_before_claim(self):
        a,build,contract,br,cr,row,out,oref=chain_fixture()
        a['mountain_completed_evidence']={'path':'/sealed/mountain.json','sha256':'f'*64}
        a['effective_protocol']['segments'] += [{'segment_id':'forest_b'},{'segment_id':'mountain'}]
        population=[{'segment_id':s,'screen_complete':True,'artifact_references':{'summary':
            {'path':'/e/'+s+'.json','sha256':'x'*64}}} for s in ('cave','forest_b','mountain')]
        retries={s:{'summary':dict(build,segment_id=s,build_protocol_sha256='c'*64),'ref':br} for s in ('cave','forest_b')}
        def fake_read(path,*args):
            if str(path).endswith('build_protocol.json'): return contract,cr
            return {'segment_id':'cave','status':'PASS_PREREGISTERED_SEGMENT_SCREEN','omp_confirmation':{'artifact':{'path':'/omp','sha256':'o'}}}, {'path':str(path),'sha256':'x'*64}
        # Isolate the final Mountain equality gate; other chain primitives have dedicated tests.
        def passthrough_row(outcome,segment,*args):
            return {k:v for k,v in next(r for r in population if r['segment_id']==segment['segment_id']).items() if k!='artifact_references'}
        with patch.object(m,'read_bound',side_effect=fake_read), patch.object(m,'validate_retry_build'), \
             patch.object(m,'validate_outcome_binding'), patch.object(m,'verify_confirmed_summaries',return_value=[]), \
             patch.object(m,'scene_row',side_effect=passthrough_row):
            # Using source-style STOP bypasses screen confirmation in this focused fixture.
            for r in population[:2]: r.update(upstream_stage='source',screen_complete=False)
            with patch.object(m,'_stage_stop',return_value=({'status':'STOP'},None)):
                with self.assertRaisesRegex(ValueError,'MOUNTAIN_MUST_USE_ORIGINAL'):
                    m.verify_final_outcomes({'rows':population},'/e/coverage.json',retries,a,{})


    def test_complete_filesystem_chain_keeps_two_resource_stops_and_original_mountain(self):
        import json
        import tempfile
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def save(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value,sort_keys=True),encoding='utf-8')
                return {'path':str(path.resolve()),'sha256':m.file_sha(path)}
            a,_,_,_,_,_,_,_=chain_fixture()
            specs=[{'segment_id':s,'scene':s,'first_frame':1,'last_frame':64} for s in ('cave','forest_b','mountain')]
            rule={'minimum_natural_frames':2,'minimum_baseline_support_pixels_per_frame':16,
                'select_first_per_segment':True,'selection_does_not_use_quality_delta':True,
                'rank':['segment_id_lexicographic','earliest_qualifying_absolute_frame','component_id_lexicographic']}
            a['effective_protocol']={'segments':specs,'visibility':{'qualifying_component_rule':rule}}
            a['retry_sources']={s:{'source_coarse':'/old/'+s} for s in ('cave','forest_b')}
            binding={'path':a['amendment_path'],'sha256':a['amendment_sha256']}
            retries={}; population=[]
            for s in ('cave','forest_b'):
                _,b,c,_,_,_,_,_=chain_fixture()
                b.update(segment_id=s,status='STOP_INFRASTRUCTURE_BUILD_INCOMPLETE',
                         output='/data/'+s+'_attempt02',reason='quota')
                c.update(output=b['output'],segment=next(x for x in specs if x['segment_id']==s),
                         source_reuse=a['retry_sources'][s])
                cref=save(root/s/'build02/build_protocol.json',c)
                b['build_protocol_sha256']=cref['sha256']
                bref=save(root/s/'build02/summary.json',b)
                retries[s]={'summary':b,'ref':bref}
                shaped,_=m._stage_stop(bref['path'],'build',c['segment'],a['protocol_sha256'],a['preregistration_seal_sha256'],{})
                r=m.scene_row(shaped,c['segment'],a['protocol_sha256'],a['preregistration_seal_sha256'],rule,None)
                r['artifact_references']={'summary':bref}; population.append(r)
            vis={k:0 for k in ('canonical_events','visible_candidate_events','visible_admitted_source_events',
                'visible_replacement_events','unknown_support_visibility_events','compiled_source_visible_candidates','compiled_source_visible_admitted')}
            vis.update(visible_source_admission_rate=None,qualified_components=[],first_qualifying_component=None)
            shared={'segment_id':'mountain','protocol_sha256':a['protocol_sha256'],
                'preregistration_seal_sha256':a['preregistration_seal_sha256'],
                'source':{'raw_observations':0,'canonical_events':0,'exact_roots':0,'source_decision_counts':{}},
                'composition':{k:0 for k in ('components','admitted_components','admitted_events','fallback_components','fallback_events')},
                'schedule':{'natural_frames':64,'modified_natural_frames':[],'query_count':64,'natural_event_frame_replacements':0},
                'visibility':vis,'support_graph':{},'certified_events':{},'component_queries':{},
                'frame_artifacts':[],'visibility_frame_artifacts':[],'visibility_artifacts_base':'/old/mountain',
                'private_cache_removed':True,'partial_output_published':False,'final_input_verification':{'status':'PASS'}}
            primary={**deepcopy(shared),'status':'PASS_PREREGISTERED_SEGMENT_SCREEN_OMP1_ONLY','omp_num_threads':'1'}
            pref=save(root/'mountain/omp1/summary.json',primary)
            replay={**deepcopy(shared),'status':'PASS_OMP8_SEQUENCE_REPLAY','omp_num_threads':'8','primary_summary':pref}
            rref=save(root/'mountain/omp8/summary.json',replay)
            confirmation={k:shared[k] for k in ('segment_id','protocol_sha256','preregistration_seal_sha256')}
            confirmation.update(status='PASS_OMP1_OMP8_IDENTICAL',omp1_summary=pref,omp8_summary=rref,queries_compared=64,
                all_arrays_records_component_membership_and_complete_root_supports_identical=True)
            oref=save(root/'mountain/omp8/omp_confirmation.json',confirmation)
            combined={**primary,'status':'PASS_PREREGISTERED_SEGMENT_SCREEN',
                      'omp_confirmation':{'status':'PASS_OMP1_OMP8_IDENTICAL','artifact':oref}}
            mref=save(root/'mountain/omp8/combined_summary.json',combined)
            a['mountain_completed_evidence']=mref
            r=m.scene_row(combined,specs[2],a['protocol_sha256'],a['preregistration_seal_sha256'],rule,confirmation)
            r['artifact_references']={'summary':mref}; population.append(r)
            bindings={}
            result=m.verify_final_outcomes({'rows':population},root/'coverage.json',retries,a,bindings)
            self.assertEqual(set(result),{'cave','forest_b','mountain'})
            self.assertEqual(len(result['mountain']['confirmed_omp_summaries']),2)
            self.assertEqual(m.classify(population)['decision'],'RESOURCE_OR_EVIDENCE_UNRESOLVED_STOP_VISIBILITY_CAMPAIGN')
            self.assertIsNone(population[0]['visible_candidate_events'])
            self.assertIn(pref['path'],bindings)


class ExternalStopBranchTests(unittest.TestCase):
    def exercise(self, complete=False, upstream=False):
        a,b,c,br,cr,row,out,oref = chain_fixture()
        b['build_protocol_sha256'] = cr['sha256']
        row.update(screen_complete=complete, terminal_infrastructure_stop=not complete,
                   artifact_references={'summary':oref})
        if upstream: row['upstream_stage'] = 'source'
        out['status'] = 'PASS_PREREGISTERED_SEGMENT_SCREEN' if complete else 'STOP_SCREEN_ATTEMPT_NOT_COMPLETE'
        out.pop('input_and_code_sha256', None)
        population = [row, {'segment_id':'forest_b'}, {'segment_id':'mountain'}]
        retries = {s:{'summary':b, 'ref':br} for s in ('cave','forest_b')}
        external = {'cave':{'summary':{'proof':'external'},'ref':{'path':'/external','sha256':'e'*64}}}
        def bound(path,*args):
            return (c,cr) if str(path).endswith('build_protocol.json') else (out,oref)
        with patch.object(m,'read_bound',side_effect=bound), patch.object(m,'validate_retry_build'), \
             patch('final_stop_binding.verify_external_stop',side_effect=RuntimeError('BOUND_VERIFIER_REACHED')) as check:
            if complete or upstream:
                with self.assertRaisesRegex(ValueError,'ONLY_FOR_ACTUAL_TERMINAL_SCREEN_STOP'):
                    m.verify_final_outcomes({'rows':population},'/coverage.json',retries,a,{},external)
                check.assert_not_called()
            else:
                with self.assertRaisesRegex(RuntimeError,'BOUND_VERIFIER_REACHED'):
                    m.verify_final_outcomes({'rows':population},'/coverage.json',retries,a,{},external)
                self.assertEqual(check.call_args.args[:4],(external['cave']['summary'],oref,br,cr))

    def test_external_stop_calls_independent_bound_verifier(self): self.exercise()
    def test_external_stop_cannot_bypass_complete_screen_confirmation(self): self.exercise(complete=True)
    def test_external_stop_cannot_relabel_source_stage(self): self.exercise(upstream=True)

    def test_zero_source_with_external_failure_keeps_runtime_unknown(self):
        data=rows(); data[1].update(screen_complete=False,terminal_infrastructure_stop=True,
            canonical_events=0, visible_candidate_events=None, visible_replacement_events=None)
        result=classify(data)
        self.assertEqual(result['decision'],'RESOURCE_OR_EVIDENCE_UNRESOLVED_STOP_VISIBILITY_CAMPAIGN')
        self.assertEqual(result['completed_new_segments'],['cave','mountain'])
        self.assertIsNone(result['first_qualifying_component'])

if __name__ == '__main__': unittest.main()
