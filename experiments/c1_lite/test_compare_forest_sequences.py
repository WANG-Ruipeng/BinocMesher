"""Synthetic JSON-only comparator tests; no native library or actual cache."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import compare_forest_sequences as m


def hashes(letter):return {name:letter*64 for name in ('vertices','faces','tags')}


def make_frame(query,active=False):
    tau=F(query['evaluation_tau']);support_active=F(5,4)<tau<F(7,4) or F(9,4)<tau<F(11,4)
    baseline=[{'vertices':4,'faces':2,'sha256':hashes('a')}]+[
        {'vertices':0,'faces':0,'sha256':hashes('b')} for _ in range(4)]
    output=deepcopy(baseline);records=[];mapping={};components={};component_mapping={}
    if active:
        owners=[[0,0,0,0,0,0,0],[0,0,0,0,0,1,0]]
        fan=[[i,(i+1)%4,4] for i in range(4)]
        plan={'element':0,'baseline_vertex_count':4,'baseline_face_count':2,'new_center_id':4,
            'boundary_actual_ids':[0,1,2,3],'removed_face_rows':[0,1],'consumed_owners':owners,
            'center':[1.0,1.0,0.0],'source_digest':'d'*64,'fan_faces':fan}
        records=[{'event_id':'e1','component_id':'c1','plan':plan}]
        mapping={'e1':{'event_id':'e1','component_id':'c1','element':0,'new_center_id':4,
            'fan_face_rows_by_sector':[0,1,2,3],'fan_actual_vertex_ids':fan,
            'source_face_rows':[0,1],'consumed_owners':owners}}
        components={'c1':['e1']};component_mapping={'c1':{'events':['e1'],'event_mapping_keys':['e1']}}
        output[0]={'vertices':5,'faces':4,'sha256':hashes('c')}
    union={'schema':'forest-component-query-union-v1','status':'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED',
        'partial_output_published':False,'provided_records_all_or_none':True,'component_membership_verified':True,
        'old_vertices_and_tags_byte_identical':True,'retained_original_face_rows_byte_identical':True,
        'original_baseline_unchanged':True,'baseline':deepcopy(baseline),'output':output,
        'canonical_event_order':['e1'] if active else [],'event_mapping':mapping,
        'component_mapping':component_mapping,'components':components,'event_count':int(active),
        'component_count':int(active),'source_faces_consumed_once':2*active,'raw_owners_consumed_once':2*active,
        'expected_event_pairs':0,'expected_triangle_pairs':0,
        'counts':{'triangle_pairs_excluded_by_support_aabb':0,'exact_triangle_pair_calls':0}}
    return {'query':query,'baseline':baseline,'union':union,'records':records,'records_sha256':m.digest(records),
        'root_support_replay_sha256':'6'*64 if support_active else None,
        'certification_sha256':'f'*64,'publication':'STAGED_UNTIL_FULL_SEQUENCE_VERIFIED',
        'single_scene_output_elements':5,'persistent_mesh_files':0,'fresh_baseline_for_each_timestamp':True,
        'full_scene_arrays_actually_constructed':True,'rejected_actual_domains_unchanged':10 if active else 0,
        'changed':active,'cost':{'wall_seconds':1.0}}


def fixture(omp='1'):
    delta=float.fromhex('0x1.500053e2d6239p-1');origin=0.5/24;frames=[]
    for index in range(64):
        global_time=(index+0.5)/24;physical=float(global_time-origin)
        q={'key':f'frame_{index+1:04d}','kind':'natural','frame_number':index+1,
            'frame_index_zero_based':index,'time_mode':'physical','physical_time_hex':physical.hex(),
            'global_camera_time_hex':global_time.hex(),'evaluation_tau':str(F.from_float(float(physical/delta)))}
        frames.append(make_frame(q,index+1==21))
    for root in (F(3,2),F(5,2)):
        q={'key':f'root_{root.numerator}_{root.denominator}','kind':'exact_root','frame_number':None,
            'time_mode':'exact','physical_time_hex':float(root*F.from_float(delta)).hex(),'evaluation_tau':str(root)}
        frames.append(make_frame(q,root==F(3,2)))
    verification={'status':'PASS','private_nonlog_inputs_unchanged':True,
        'original_input_content_sha256':'e'*64,'original_files_unchanged':35,'original_bytes':1096080433,
        'private_log_appended_bytes':100}
    summary={'schema':'forest-atomic-sequence-v1','status':'PASS_FOREST_ATOMIC_REQUESTED_SEQUENCE',
        'private_cache_removed':True,'partial_output_published':False,'published_partial_outputs':False,
        'publication':'COMMITTED_FULL_REQUESTED_SEQUENCE_MANIFEST','omp_num_threads':omp,
        'support_graph_sha256':'d'*64,'certification_sha256':'f'*64,'actual_delta_t_hex':delta.hex(),
        'input_verification':verification,'final_input_verification':deepcopy(verification),
        'frame_artifacts':[{'path':f'frames/{f["query"]["key"]}.json','sha256':'a'*64} for f in frames],
        'natural_frame_count':64,'exact_root_diagnostic_count':2,'actual_scene_outputs':66,'root_denominator':2,
        'actual_modified_natural_frames':[21],'natural_frame_change_rate':1/64,
        'actual_natural_event_frame_replacements':1,'jointly_admitted_events':1,'admitted_components':1,
        'nonempty_treated_roots':1,'cost':{'wall_seconds':100.0}}
    return summary,frames


class CompareTests(unittest.TestCase):
    def setUp(self):self.left,self.lf=fixture('1');self.right,self.rf=fixture('8')
    def compare(self):return m.compare_sequences(self.left,self.lf,self.right,self.rf)

    def test_identical_all_sixty_six_with_cost_and_omp_difference(self):
        self.right['cost']['wall_seconds']=20
        self.right['input_verification']['private_log_appended_bytes']=200
        self.rf[20]['cost']['wall_seconds']=0.1
        r=self.compare();self.assertEqual(r['status'],'PASS_IDENTICAL_ATOMIC_FOREST_SEQUENCES')
        self.assertEqual((r['natural_frames_compared'],r['exact_roots_compared']),(64,2))

    def test_output_array_hash_difference_is_nondeterministic_payload(self):
        self.rf[20]['union']['output'][0]['sha256']['vertices']='9'*64
        r=self.compare();self.assertEqual(r['status'],'FAIL_SEQUENCE_DIFFERENCE')
        self.assertTrue(any(x['field']=='frame_0021.union' for x in r['first_differences']))

    def test_valid_but_different_center_record_never_silently_equal(self):
        self.rf[20]['records'][0]['plan']['center'][2]=0.5
        self.rf[20]['records_sha256']=m.digest(self.rf[20]['records'])
        self.assertEqual(self.compare()['status'],'FAIL_SEQUENCE_DIFFERENCE')

    def test_record_hash_cannot_hide_center_change(self):
        self.rf[20]['records'][0]['plan']['center'][0]=0.5
        with self.assertRaisesRegex(m.InvalidSequence,'RECORDS_HASH'):self.compare()

    def test_missing_natural_or_root_is_invalid_input(self):
        self.rf.pop()
        with self.assertRaisesRegex(m.InvalidSequence,'64_PLUS_2'):self.compare()

    def test_duplicate_query_is_invalid(self):
        self.rf[-1]=deepcopy(self.rf[-2])
        with self.assertRaisesRegex(m.InvalidSequence,'DUPLICATE_QUERY'):self.compare()

    def test_internal_time_must_follow_actual_double_division(self):
        self.rf[20]['query']['evaluation_tau']='3/2'
        with self.assertRaisesRegex(m.InvalidSequence,'INTERNAL_TIME'):self.compare()

    def test_actual_center_id_or_fan_face_placement_must_be_canonical(self):
        self.rf[20]['union']['event_mapping']['e1']['new_center_id']=99
        with self.assertRaisesRegex(m.InvalidSequence,'PLACEMENT'):self.compare()
        self.right,self.rf=fixture('8');self.rf[20]['union']['event_mapping']['e1']['fan_face_rows_by_sector']=[1,0,2,3]
        with self.assertRaisesRegex(m.InvalidSequence,'PLACEMENT'):self.compare()

    def test_component_members_cannot_be_partial(self):
        self.rf[20]['union']['component_mapping']['c1']['events']=[]
        with self.assertRaisesRegex(m.InvalidSequence,'COMPONENT_MEMBERSHIP'):self.compare()

    def test_duplicate_owner_consumption_is_invalid(self):
        frame=self.rf[20];frame['records'][0]['plan']['consumed_owners'].append([0,0,0,0,0,0,0])
        frame['records_sha256']=m.digest(frame['records'])
        with self.assertRaisesRegex(m.InvalidSequence,'CONSUMED_MULTIPLE_TIMES'):self.compare()

    def test_partial_output_flag_or_missing_manifest_flag_is_invalid(self):
        self.rf[20]['union']['partial_output_published']=True
        with self.assertRaisesRegex(m.InvalidSequence,'PARTIAL_OUTPUT'):self.compare()
        self.right,self.rf=fixture('8');self.right.pop('partial_output_published')
        with self.assertRaisesRegex(m.InvalidSequence,'PARTIAL_PUBLICATION'):self.compare()

    def test_cache_cleanup_and_final_content_binding_are_mandatory(self):
        self.right['private_cache_removed']=False
        with self.assertRaisesRegex(m.InvalidSequence,'CLEANUP'):self.compare()
        self.right,self.rf=fixture('8');self.right['final_input_verification']['original_input_content_sha256']='8'*64
        with self.assertRaisesRegex(m.InvalidSequence,'CHANGED_WITHIN'):self.compare()

    def test_empty_frame_must_return_exact_baseline(self):
        self.rf[0]['union']['output'][1]['sha256']['tags']='8'*64
        with self.assertRaisesRegex(m.InvalidSequence,'CHANGE_ACCOUNTING'):self.compare()

    def test_graph_binding_difference_is_reported(self):
        self.right['support_graph_sha256']='7'*64
        self.assertEqual(self.compare()['status'],'FAIL_SEQUENCE_DIFFERENCE')

    def test_rejected_domain_audit_counts_are_deterministic_evidence(self):
        self.rf[20]['rejected_actual_domains_unchanged']=11
        self.assertEqual(self.compare()['status'],'FAIL_SEQUENCE_DIFFERENCE')

    def test_summary_replacement_counts_recomputed(self):
        self.right['actual_natural_event_frame_replacements']=2
        with self.assertRaisesRegex(m.InvalidSequence,'REPLACEMENT_ACCOUNTING'):self.compare()

    def test_same_omp_cannot_be_claimed_as_one_vs_eight(self):
        self.right['omp_num_threads']='1'
        with self.assertRaisesRegex(m.InvalidSequence,'OMP1_VS_OMP8'):self.compare()

    def test_same_element_two_event_union_assigns_distinct_fresh_centers(self):
        frame=deepcopy(self.lf[20]);baseline=frame['baseline']
        baseline[0].update(vertices=8,faces=4)
        union=frame['union'];union['baseline']=deepcopy(baseline)
        union['output'][0].update(vertices=10,faces=8)
        records=[];mapping={}
        for ordinal in range(2):
            eid='e'+str(ordinal+1);cycle=[4*ordinal+i for i in range(4)]
            source_faces=[2*ordinal,2*ordinal+1]
            owners=[[0,0,0,ordinal,0,i,0] for i in range(2)]
            plan={'element':0,'baseline_vertex_count':8,'baseline_face_count':4,'new_center_id':8,
                'boundary_actual_ids':cycle,'removed_face_rows':source_faces,'consumed_owners':owners,
                'center':[1.0+4*ordinal,1.0,0.0],'source_digest':'d'*64,
                'fan_faces':[[cycle[i],cycle[(i+1)%4],8] for i in range(4)]}
            records.append({'event_id':eid,'component_id':'c1','plan':plan})
            mapping[eid]={'event_id':eid,'component_id':'c1','element':0,'new_center_id':8+ordinal,
                'fan_face_rows_by_sector':[*source_faces,4+2*ordinal,5+2*ordinal],
                'fan_actual_vertex_ids':[[cycle[i],cycle[(i+1)%4],8+ordinal] for i in range(4)],
                'source_face_rows':source_faces,'consumed_owners':owners}
        frame.update(records=records,records_sha256=m.digest(records))
        union.update(canonical_event_order=['e1','e2'],event_mapping=mapping,
            component_mapping={'c1':{'events':['e1','e2'],'event_mapping_keys':['e1','e2']}},
            components={'c1':['e1','e2']},event_count=2,component_count=1,
            source_faces_consumed_once=4,raw_owners_consumed_once=4,
            expected_event_pairs=1,expected_triangle_pairs=32,
            counts={'triangle_pairs_excluded_by_support_aabb':32,'exact_triangle_pair_calls':0})
        self.assertEqual(m.validate_frame(frame,self.left,float.fromhex(self.left['actual_delta_t_hex'])),'frame_0021')
        union['event_mapping']['e2']['new_center_id']=8
        with self.assertRaisesRegex(m.InvalidSequence,'PLACEMENT'):
            m.validate_frame(frame,self.left,float.fromhex(self.left['actual_delta_t_hex']))

    def test_executed_producer_source_bindings_must_match_when_present(self):
        self.left['executed_sources_sha256']={'producer.py':'1'*64}
        self.right['executed_sources_sha256']={'producer.py':'2'*64}
        self.assertEqual(self.compare()['status'],'FAIL_SEQUENCE_DIFFERENCE')

    def test_complete_root_support_replay_hash_is_required_on_all_active_frames(self):
        self.rf[22]['root_support_replay_sha256']=None
        with self.assertRaisesRegex(m.InvalidSequence,'root_support_replay'):self.compare()
        self.right,self.rf=fixture('8');self.rf[0]['root_support_replay_sha256']='8'*64
        with self.assertRaisesRegex(m.InvalidSequence,'OUTSIDE_QUERY'):self.compare()

    def test_complete_root_support_replay_must_match_across_omp(self):
        self.rf[22]['root_support_replay_sha256']='8'*64
        r=self.compare();self.assertEqual(r['status'],'FAIL_SEQUENCE_DIFFERENCE')
        self.assertTrue(any(x['field']=='frame_0023.root_support_replay_sha256' for x in r['first_differences']))

    def write_artifacts(self,folder):
        summary,frames=fixture();folder=Path(folder)
        for binding,frame in zip(summary['frame_artifacts'],frames):
            path=folder/binding['path'];path.parent.mkdir(parents=True,exist_ok=True)
            data=m.stable(frame);path.write_bytes(data);binding['sha256']=hashlib.sha256(data).hexdigest()
        (folder/'summary.json').write_bytes(m.stable(summary))
        return summary

    def test_artifact_loader_checks_all_sixty_seven_hash_bindings(self):
        with tempfile.TemporaryDirectory(prefix='forest-sequence-unit-') as directory:
            self.write_artifacts(directory)
            summary,frames,bindings=m.load_sequence(directory)
            self.assertEqual(len(frames),66);self.assertEqual(len(bindings),67)
            m.validate_sequence(summary,frames)
            path=Path(directory)/'frames/frame_0001.json';path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(m.InvalidSequence,'HASH_MISMATCH'):m.load_sequence(directory)

    def test_artifact_path_escape_and_duplicate_path_rejected(self):
        with tempfile.TemporaryDirectory(prefix='forest-sequence-unit-') as directory:
            summary=self.write_artifacts(directory);summary['frame_artifacts'][0]['path']='../outside.json'
            (Path(directory)/'summary.json').write_bytes(m.stable(summary))
            with self.assertRaisesRegex(m.InvalidSequence,'ESCAPES'):m.load_sequence(directory)
            summary=self.write_artifacts(directory);summary['frame_artifacts'][1]=summary['frame_artifacts'][0]
            (Path(directory)/'summary.json').write_bytes(m.stable(summary))
            with self.assertRaisesRegex(m.InvalidSequence,'DUPLICATE_FRAME_ARTIFACT'):m.load_sequence(directory)


if __name__=='__main__':unittest.main()
