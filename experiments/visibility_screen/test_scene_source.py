"""Synthetic wiring tests only: no native code or real scene/cache inputs."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import scene_source as m
from test_forest_source_support_inventory import fixture


def profile(first=1):
    times=[float((first+i-0.5)/24) for i in range(64)];origin=times[0]
    duration=max(float(t-origin) for t in times)+1e-5
    fj=lambda x:{'numerator':F.from_float(x).numerator,'denominator':F.from_float(x).denominator}
    camera={'times_seconds':times,'poses':[[[1 if i==j else 0 for j in range(4)] for i in range(4)] for _ in times],
        'intrinsics':[[[1,0,0],[0,1,0],[0,0,1]] for _ in times], 'widths':[960]*64,'heights':[540]*64,
        'time_mapping':{'origin_seconds':fj(origin),'duration_seconds':fj(duration),'delta_seconds':fj(duration/4),
            'effective_fading_seconds':1.,'temporal_group_count':2,'maximum_discrete_time':4,
            'use_alignment':False,'min_t_offset':0}}
    return camera,{'segment_id':'synthetic','scene':'Synthetic','first_frame':first,'last_frame':first+63,'fps':24,'pixels_per_cube':6}


def inventory_fixture(compiled=True):
    event,native,snapshots,dataset=fixture(compiled)
    dataset.maximum=4
    event.update(namespace_id='segment/OpaqueTerrain/synthetic',source_status='SOURCE_READY' if compiled else 'REJECTED_FIXED_POLICY',
                 decision='UNKNOWN' if compiled else 'REJECTED_FIXED_POLICY')
    return event,snapshots,dataset


def empty_cache(root):
    root.mkdir();(root/'processed_hyperpolys').mkdir()
    for g in range(2):
        for s in range(4):
            for tail in ('','_hpmeta'):(root/'processed_hyperpolys'/f'{g}_{s}{tail}.bin').write_bytes(b'')
    (root/'slicing_preprocess.finish').write_text('complete')
    (root/'slicing_preprocess.manifest.json').write_text(json.dumps({'event_registry_enabled':True,
        'provenance_enabled':True,'provenance_requested':True,'bpm_version':2,'bhp_version':2}))
    (root/'event_registry_p1.csv').write_text('raw_id,canonical_event_id,logical_incidence_id,root_num,root_den,element\n')


class SourceStageTests(unittest.TestCase):
    def test_absolute_second_segment_keeps_true_origin_and_delta(self):
        camera,spec=profile(97);report=m.validate_profile(camera,spec)
        self.assertEqual(report['origin_seconds_hex'],camera['times_seconds'][0].hex())
        self.assertFalse(report['actual_native_initialization_verified'])
        self.assertEqual((report['group_count'],report['maximum_discrete_time']),(2,4))

    def test_scene_frame_offset_cannot_be_silently_rebased(self):
        camera,spec=profile();spec.update(first_frame=97,last_frame=160)
        with self.assertRaisesRegex(ValueError,'ABSOLUTE_CAMERA_TIME_GRID'):
            m.validate_profile(camera,spec)

    def test_stale_delta_binding_stops(self):
        camera,spec=profile();camera['time_mapping']['delta_seconds']={'numerator':1,'denominator':1}
        with self.assertRaisesRegex(ValueError,'DELTA_MISMATCH'):m.validate_profile(camera,spec)

    def test_all_integer_breakpoints_not_three_sample_inference(self):
        cells=m.cells_for_bounds({'lower':'3/4','root':'3/2','upper':'9/4'})
        single=[m.compiler_api.fr(c['t0']) for c in cells if c['kind']=='singleton']
        self.assertEqual(single,[F(3,4),F(1),F(3,2),F(2),F(9,4)])
        self.assertEqual(len(cells),9)

    def test_inventory_is_equivalent_to_frozen_catalog_on_old_profile(self):
        event,snapshots,dataset=inventory_fixture()
        old=m.inventory_api.build_event(event,{'decision':'UNKNOWN'},snapshots,dataset,{})
        new=m.build_inventory_event(event,snapshots,dataset)
        for key in ('record_catalog','owner_catalog','face_class_catalog'):
            self.assertEqual(new[key],old[key])
        for a,b in zip(new['cells'],old['cells']):
            for key in ('actual_halo_face_ids','legacy_halo_face_ids','retained_face_ids','source_owner_ids',
                        'source_face_ids','source_boundary_vids','candidate_actual_owner_ids','candidate_vertex_stars_actual'):
                self.assertEqual(a[key],b[key])

    def test_no_source_still_has_complete_candidate_halo_and_replicas(self):
        result=m.build_inventory_event(*inventory_fixture(False))
        self.assertEqual(result['source_status'],'CANDIDATE_DOMAIN_ONLY')
        self.assertTrue(result['candidate_domain_complete'])
        self.assertEqual(len(result['owner_catalog']),5)
        for cell in result['cells']:
            self.assertIsNone(cell['source_face_ids'])
            self.assertEqual(cell['retained_face_ids'],cell['actual_halo_face_ids'])
            self.assertTrue(cell['candidate_vertex_stars_legacy'])

    def test_partial_owner_class_is_rejected_by_inventory(self):
        event,snapshots,dataset=inventory_fixture()
        for row in event['compiler']['source']['segments']+event['compiler']['source']['breakpoint_points']:
            row['owners']=row['owners'][:2]
        with self.assertRaisesRegex(ValueError,'PARTIAL_SOURCE_OWNER_CLASS'):
            m.build_inventory_event(event,snapshots,dataset)

    def test_incomplete_halo_is_not_empty_proof(self):
        event,snapshots,dataset=inventory_fixture(False)
        next(iter(snapshots.values())).halo_complete=False
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):
            m.build_inventory_event(event,snapshots,dataset)

    def test_source_symbolic_dependencies_keep_rejected_nodes(self):
        left=m.build_inventory_event(*inventory_fixture());right=m.build_inventory_event(*inventory_fixture(False))
        right['event_id']='other'
        edges,warnings=m.symbolic_dependencies([left,right])
        self.assertTrue(edges);self.assertFalse(warnings)

    def test_heterogeneous_root_partition_is_explicit_conservative_not_first_event(self):
        left=m.build_inventory_event(*inventory_fixture());right=deepcopy(left);right['event_id']='other'
        right['cells'][0]['t0']=m.inventory_api.fj(F(1))
        edges,warnings=m.symbolic_dependencies([left,right])
        self.assertEqual(len(edges),1)
        self.assertEqual(warnings[0]['runtime_component_coverage'],'UNRESOLVED_DO_NOT_BORROW_FIRST_EVENT_SCHEDULE')

    def test_arbitrary_kernel_error_is_not_unsupported_policy(self):
        self.assertIsNone(m.kernel_domain_witness([],{}, {'reason':'IndexError: synthetic implementation bug'}))

    def test_two_two_domain_witness_is_independently_counted(self):
        row={'root_num':'3','root_den':'2',**{f'h{i}':f'{i}:0' for i in range(4)}}
        hv={m.reader_api.HVID(i,0):NS(time=t) for i,t in enumerate((0,1,1,2))}
        witness=m.kernel_domain_witness([row],hv,{'reason':'AuditError: saddle does not have two lower and two upper branches'})
        self.assertEqual(witness['branch_counts'],{'lower':3,'equal':0,'upper':1})

    def test_completed_zero_registry_is_zero_not_missing(self):
        camera,spec=profile()
        with tempfile.TemporaryDirectory() as temp:
            cache=Path(temp)/'cache';empty_cache(cache)
            with patch.object(m.reader_api,'read_event_candidates',side_effect=AssertionError('Empty registry must not invoke nonempty reader')):
                result=m.run_source_stage(cache,camera,spec,Path(temp)/'out')
            self.assertEqual(result['summary']['status'],'COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION')
            self.assertEqual(result['summary']['canonical_event_denominator'],0)
            self.assertIsNone(result['summary']['admission_rate'])
            self.assertEqual(result['inventory']['events'],[])

    def test_missing_cache_completion_has_unknown_not_zero_denominator(self):
        camera,spec=profile()
        with tempfile.TemporaryDirectory() as temp:
            cache=Path(temp)/'cache';empty_cache(cache);(cache/'slicing_preprocess.finish').unlink()
            result=m.run_source_stage(cache,camera,spec,Path(temp)/'out')
            self.assertEqual(result['summary']['status'],'STOP_SOURCE_STAGE_NOT_COMPLETE')
            self.assertIsNone(result['summary']['canonical_event_denominator'])

    def test_unknown_helper_failure_keeps_all_known_registry_rows(self):
        camera,spec=profile()
        with tempfile.TemporaryDirectory() as temp:
            cache=Path(temp)/'cache';empty_cache(cache)
            with (cache/'event_registry_p1.csv').open('a') as stream:
                stream.write('r0,e0,l0,3,2,0\nr1,e1,l1,5,2,0\n')
            with patch.object(m.reader_api,'read_event_candidates',side_effect=RuntimeError('reader harness failure')):
                result=m.run_source_stage(cache,camera,spec,Path(temp)/'out')
            self.assertEqual(result['summary']['status'],'STOP_SOURCE_STAGE_NOT_COMPLETE')
            self.assertEqual(result['summary']['canonical_event_denominator'],2)
            self.assertEqual({e['event_id'] for e in result['events']},{'e0','e1'})
            self.assertEqual(len(result['inventory']['events']),2)
            self.assertTrue(all(not e['candidate_domain_complete'] for e in result['inventory']['events']))


if __name__=='__main__':unittest.main()
