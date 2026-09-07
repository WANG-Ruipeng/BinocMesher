"""Pure/Fake-ABI tests only. Never loads a native library or touches caches."""
from fractions import Fraction as F
from pathlib import Path
import unittest

import numpy as np

import forest_native_reader as m


def fj(value):
    value=F(value);return {'numerator':value.numerator,'denominator':value.denominator}


def documents():
    times=[(i+0.5)/24 for i in range(64)];origin=min(times);duration=max(t-origin for t in times)+1e-5
    camera={'poses':[np.eye(4).tolist() for _ in times],'intrinsics':[np.eye(3).tolist() for _ in times],
        'heights':[540]*64,'widths':[960]*64,'times_seconds':times,
        'time_mapping':{'use_alignment':False,'min_t_offset':0,'origin_seconds':fj(origin),
            'duration_seconds':fj(duration),'delta_seconds':fj(duration/4),'temporal_group_count':2,
            'maximum_discrete_time':4,'effective_fading_seconds':1.0}}
    effective={'terrain_elements':['ground','landtiles','sdf_trees','warped_rocks','voronoi_rocks','atmosphere'],
        'overrides':['BinocMesher.pixels_per_cube=6'],'bounds':[-500,500,-500,500,-500,500]}
    return camera,effective


class FakeDll:
    def __init__(self,identity=1):
        self.requested=False;self.identity=identity;self.events=[];self.output_started=False
        self.failure_status=0;self.bad_count=False;self.bad_face=False;self.bad_owner=False;self.bad_ledger_count=False
    def slicing_identity_enable(self,value):self.requested=value;self.events.append(('enable',value))
    def _run(self,vc,fc,smooth):
        assert smooth is False
        self.output_started=False;self.events.append(('run',))
        np.ctypeslib.as_array(vc,shape=(5,))[:]=[-1 if self.bad_count else 3]*5
        np.ctypeslib.as_array(fc,shape=(5,))[:]=[1]*5
        return self.failure_status
    def run_slicing_rational(self,num,den,vc,fc,smooth):return self._run(vc,fc,smooth)
    def run_slicing(self,value,vc,fc,smooth):return self._run(vc,fc,smooth)
    def slicing_last_error(self):return b'synthetic ordinary failure'
    def slicing_identity_status(self):return self.identity if self.requested else 0
    def slicing_identity_last_error(self):return b'Identity owner ledger exceeds its bounded scope.' if self.identity==3 else b''
    def slicing_identity_vertex_count(self,e):return 2 if self.bad_ledger_count else 3
    def slicing_identity_owner_count(self,e):return 2
    def slicing_identity_output_vertices(self,e,ptr,capacity):
        assert not self.output_started;assert capacity==12
        self.events.append(('ids',e));array=np.ctypeslib.as_array(ptr,shape=(capacity,)).reshape(-1,4)
        array[:]=[[1,0,2,0],[3,0,4,0],[5,0,6,0]];return 0
    def slicing_identity_output_owners(self,e,ptr,capacity):
        assert not self.output_started;assert capacity==22
        self.events.append(('owners',e));array=np.ctypeslib.as_array(ptr,shape=(capacity,)).reshape(-1,11)
        array[:]=[[e,0,0,j,0,0,0,0,0,1,2] for j in range(2)]
        if self.bad_owner:array[0,10]=1
        return 0
    def slicing_output(self,e,v,f,tags):
        self.output_started=True;self.events.append(('mesh',e))
        np.ctypeslib.as_array(v,shape=(9,))[:]=[e,0,0,e+1,0,0,e,1,0]
        np.ctypeslib.as_array(f,shape=(3,))[:]=[99 if self.bad_face else 0,1,2]
        np.ctypeslib.as_array(tags,shape=(3,))[:]=[0,1,1]
    def slicing_discard_output(self):self.events.append(('discard',))
    def slicing_clean_up(self):self.events.append(('cleanup',))


class FakeOriginalSourceDll(FakeDll):
    def __init__(self,identity=1):
        super().__init__(identity);self.shift_status=0
        self.shifts=np.arange(20,dtype=np.int32).reshape(5,4)*7
    def slicing_identity_source_vid_encoding_version(self):return 2
    def slicing_identity_output_source_vid_shifts(self,ptr,capacity):
        assert not self.output_started;assert capacity==20
        self.events.append(('shifts',))
        if self.shift_status:return self.shift_status
        np.ctypeslib.as_array(ptr,shape=(capacity,))[:]=self.shifts.reshape(-1)
        return 0


def reader(dll):
    camera,effective=documents();parameters=m.prepare_parameters(camera,effective)
    return m.NativeForestReader(dll,parameters,Path('/tmp/fake_private_cache'),Path('/tmp/fake.so'),'0'*64,{})


class NativeReaderTests(unittest.TestCase):
    def test_camera_packing_and_exact_time_inputs(self):
        camera,effective=documents();p=m.prepare_parameters(camera,effective)
        self.assertEqual(p['center'].tolist(),[0,0,0]);self.assertEqual(p['size'],1100.0)
        self.assertEqual(p['cameras'].shape,(64*27,))
        self.assertEqual(p['cameras'][23],0.0)
        self.assertEqual(p['expected_delta'].hex(),float(F(**camera['time_mapping']['delta_seconds'])).hex())

    def test_wrong_duration_bits_stop_before_native(self):
        camera,effective=documents();camera['time_mapping']['duration_seconds']=fj(3)
        with self.assertRaisesRegex(m.NativeReadError,'DURATION_BITS'):m.prepare_parameters(camera,effective)

    def test_atmosphere_not_counted_as_sixth_opaque_element(self):
        camera,effective=documents();effective['terrain_elements'].reverse()
        with self.assertRaisesRegex(m.NativeReadError,'ELEMENT_ORDER'):m.prepare_parameters(camera,effective)

    def test_five_elements_all_ledgers_before_any_mesh(self):
        dll=FakeDll();r=reader(dll);snapshot=r.slice_query(F(3,2))
        self.assertEqual(snapshot['identity_status'],1);self.assertEqual(len(snapshot['meshes']),5)
        first_mesh=next(i for i,e in enumerate(dll.events) if e[0]=='mesh')
        self.assertEqual(sum(e[0] in ('ids','owners') for e in dll.events[:first_mesh]),10)
        self.assertEqual(snapshot['cost']['array_bytes'],1160)
        self.assertEqual(dll.events[-3:],[('discard',),('cleanup',),('enable',False)])
        for mesh,ids,owners in zip(snapshot['meshes'],snapshot['vertex_ledgers'],snapshot['owner_ledgers']):
            self.assertEqual(ids.shape,(3,4));self.assertEqual(owners.shape,(2,11))
            self.assertTrue(all(not a.flags.writeable for a in (*mesh,ids,owners)))
        r.close()

    def test_capacity_error_returns_baseline_not_fake_admission(self):
        dll=FakeDll(identity=3);r=reader(dll);snapshot=r.slice_query(F(5,2))
        self.assertEqual(snapshot['identity_status'],3)
        self.assertIn('exceeds',snapshot['identity_error'])
        self.assertEqual(len(snapshot['meshes']),5)
        self.assertTrue(all(x.shape==(0,11) for x in snapshot['owner_ledgers']))
        self.assertFalse(any(e[0]=='ids' for e in dll.events));r.close()

    def test_disabled_observer_still_returns_all_baselines(self):
        dll=FakeDll();r=reader(dll);snapshot=r.slice_query(0.5,mode='physical',ledger=False)
        self.assertEqual(snapshot['identity_status'],0)
        self.assertEqual(snapshot['query']['physical_time_hex'],float(0.5).hex())
        self.assertTrue(all(x.shape==(0,4) for x in snapshot['vertex_ledgers']));r.close()

    def test_ordinary_failure_cleans_pending_state(self):
        dll=FakeDll();dll.failure_status=-1;r=reader(dll)
        with self.assertRaisesRegex(m.NativeReadError,'ORDINARY_NATIVE_SLICING_FAILED'):r.slice_query(F(3,2))
        self.assertEqual(dll.events[-3:],[('discard',),('cleanup',),('enable',False)])
        dll.failure_status=0;self.assertEqual(r.slice_query(F(3,2))['identity_status'],1);r.close()

    def test_negative_count_rejected_before_allocation(self):
        dll=FakeDll();dll.bad_count=True;r=reader(dll)
        with self.assertRaisesRegex(m.NativeReadError,'INVALID_NATIVE_MESH_COUNTS'):r.slice_query(F(3,2))
        self.assertFalse(any(e[0]=='mesh' for e in dll.events));r.close()

    def test_mismatched_identity_count_rejected_and_cleaned(self):
        dll=FakeDll();dll.bad_ledger_count=True;r=reader(dll)
        with self.assertRaisesRegex(m.NativeReadError,'IDENTITY_VERTEX_COUNT'):r.slice_query(F(3,2))
        self.assertEqual(dll.events[-3:],[('discard',),('cleanup',),('enable',False)]);r.close()

    def test_actual_face_index_out_of_range_rejected(self):
        dll=FakeDll();dll.bad_face=True;r=reader(dll)
        with self.assertRaisesRegex(m.NativeReadError,'NATIVE_FACE_INDEX'):r.slice_query(F(3,2))
        self.assertEqual(dll.events[-3:],[('discard',),('cleanup',),('enable',False)]);r.close()

    def test_owner_oriented_face_mismatch_rejected(self):
        dll=FakeDll();dll.bad_owner=True;r=reader(dll)
        with self.assertRaisesRegex(m.NativeReadError,'OWNER_ORIENTED_FACE_DISAGREEMENT'):r.slice_query(F(3,2))
        self.assertEqual(dll.events[-3:],[('discard',),('cleanup',),('enable',False)]);r.close()

    def test_combined_five_element_array_budget(self):
        with self.assertRaisesRegex(m.NativeReadError,'QUERY_ARRAYS_EXCEED'):
            m.validate_counts([20_000_000]*5,[0]*5)
        with self.assertRaisesRegex(m.NativeReadError,'INT_CAPACITY_OVERFLOW'):
            m.validate_counts([0]*5,[0]*5,[300_000_000]*5)

    def test_closed_reader_cannot_run(self):
        dll=FakeDll();r=reader(dll);r.close();r.close()
        with self.assertRaisesRegex(m.NativeReadError,'CLOSED'):r.slice_query(F(3,2))

    def test_bad_time_does_not_start_native(self):
        dll=FakeDll();r=reader(dll)
        with self.assertRaises(m.NativeReadError):r.slice_query(F(5))
        with self.assertRaises(m.NativeReadError):r.slice_query(float('nan'),mode='physical')
        self.assertEqual(dll.events,[]);r.close()

    def test_old_abi_is_explicitly_normalized_unverified(self):
        dll=FakeDll();r=reader(dll);s=r.slice_query(F(3,2))
        self.assertEqual(s['source_vid_encoding_version'],0)
        self.assertEqual(s['source_vid_encoding'],'MERGER_NORMALIZED_UNVERIFIED')
        self.assertIsNone(s['source_vid_shifts']);r.close()

    def test_original_abi_nonzero_shifts_readonly_before_mesh_without_double_add(self):
        dll=FakeOriginalSourceDll();r=reader(dll);s=r.slice_query(F(3,2))
        self.assertEqual(s['source_vid_encoding_version'],2)
        self.assertEqual(s['source_vid_encoding'],'ORIGINAL_EFFECTIVE_SOURCE_VID')
        shifts=s['source_vid_shifts'];self.assertEqual(shifts.dtype,np.dtype(np.int32))
        self.assertEqual(shifts.shape,(5,4));self.assertFalse(shifts.flags.writeable)
        np.testing.assert_array_equal(shifts,dll.shifts)
        self.assertEqual(s['cost']['array_bytes'],1240)
        self.assertLess(dll.events.index(('shifts',)),dll.events.index(('mesh',0)))
        for ids in s['vertex_ledgers']:
            np.testing.assert_array_equal(ids,[[1,0,2,0],[3,0,4,0],[5,0,6,0]])
        dll.shifts[:]=0;second=r.slice_query(F(5,2))
        self.assertTrue(np.all(second['source_vid_shifts']==0))
        self.assertTrue(np.any(shifts!=0));r.close()

    def test_version_two_missing_shift_abi_is_not_original(self):
        dll=FakeDll();dll.slicing_identity_source_vid_encoding_version=lambda:2
        r=reader(dll);s=r.slice_query(F(3,2))
        self.assertEqual(s['source_vid_encoding'],'UNSUPPORTED_OBSERVER_SOURCE_VID_ENCODING')
        self.assertIsNone(s['source_vid_shifts']);r.close()

    def test_unknown_version_is_not_original(self):
        dll=FakeOriginalSourceDll();dll.slicing_identity_source_vid_encoding_version=lambda:7
        r=reader(dll);s=r.slice_query(F(3,2))
        self.assertEqual(s['source_vid_encoding_version'],7)
        self.assertEqual(s['source_vid_encoding'],'UNSUPPORTED_OBSERVER_SOURCE_VID_ENCODING')
        self.assertIsNone(s['source_vid_shifts']);self.assertNotIn(('shifts',),dll.events);r.close()

    def test_failed_shift_export_cleans_before_any_mesh(self):
        dll=FakeOriginalSourceDll();dll.shift_status=-1;r=reader(dll)
        with self.assertRaisesRegex(m.NativeReadError,'SOURCE_VID_SHIFT_OUTPUT_FAILED'):r.slice_query(F(3,2))
        self.assertFalse(any(e[0]=='mesh' for e in dll.events))
        self.assertEqual(dll.events[-3:],[('discard',),('cleanup',),('enable',False)])
        dll.shift_status=0;self.assertEqual(r.slice_query(F(3,2))['identity_status'],1);r.close()

    def test_unready_and_disabled_have_no_shift_snapshot(self):
        dll=FakeOriginalSourceDll(identity=3);r=reader(dll);s=r.slice_query(F(3,2))
        self.assertIsNone(s['source_vid_shifts']);self.assertNotIn(('shifts',),dll.events)
        dll.identity=1;s=r.slice_query(F(3,2),ledger=False)
        self.assertEqual(s['identity_status'],0);self.assertIsNone(s['source_vid_shifts'])
        self.assertNotIn(('shifts',),dll.events);r.close()


if __name__=='__main__':unittest.main()
