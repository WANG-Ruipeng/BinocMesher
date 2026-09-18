"""CPU-only contract tests: no numpy, assets, CUDA library or GPU required."""
from array import array
import ctypes as C
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
GPU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GPU_ROOT))
from batching import (BindingKey, BudgetExceeded, IncompatibleTasks, OUTPUTS,
                      OUTPUT_WIDTHS, Task, checked_capacity, compare_outputs,
                      copy_outputs, pack_tasks, split_outputs)
import batching.schema as schema


def binding(**changes):
    values = dict(field_id='field-instance-token', epoch=1, device_id='device-identity',
                  context_id='context-instance-token', field_parameters_sha256='1'*64,
                  numeric_profile='strict-d1-build-identity', backend='published',
                  k=3, lattices=60, trace=False, audit=False)
    values.update(changes)
    return BindingKey(**values)


def task(task_id, degrees, seed=1, key=None, explicit_owners=True):
    n, m = len(degrees), sum(degrees)
    offsets = [0]
    for degree in degrees:
        offsets.append(offsets[-1]+degree)
    owners = (C.c_int32*m)(*[i for i,d in enumerate(degrees) for _ in range(d)])
    return Task(task_id, key or binding(), n, m,
                (C.c_double*(3*n))(*[seed+i/8 for i in range(3*n)]),
                (C.c_double*(3*m))(*[seed+100+i/16 for i in range(3*m)]),
                (C.c_int32*(n+1))(*offsets), owners if explicit_owners else None)


def ints(data):
    return tuple(x[0] for x in struct.iter_unpack('<i', data))


def token(value, width):
    return struct.pack('<I', value)*(width//4)


class PackTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [task('A', [0,2,1], 10), task('B', [2], 20), task('C', [2,0], 30)]

    def test_plain_import_has_no_application_file_io_or_cuda(self):
        script = '''
import ctypes, pathlib, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
def forbidden(*a, **k): raise AssertionError('import performed application IO/CUDA')
with patch.object(ctypes,'CDLL',forbidden), patch.object(pathlib.Path,'read_bytes',forbidden), patch.object(pathlib.Path,'read_text',forbidden), patch.object(pathlib.Path,'open',forbidden):
    import batching
assert 'mixed_common' not in sys.modules
assert 'typed_api' not in sys.modules
'''
        p = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(GPU_ROOT)],
                           text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)

    def test_empty_input_is_no_batch_no_outputs(self):
        self.assertIsNone(pack_tasks([]))
        self.assertEqual(split_outputs({}, ()), {})
        with self.assertRaisesRegex(ValueError, 'empty task list'):
            split_outputs({'position': b''}, ())

    def test_unequal_shapes_prefix_not_i_times_N_or_M(self):
        packed = pack_tasks(self.tasks)
        self.assertEqual((packed.task.n, packed.task.m), (6,7))
        self.assertEqual([(s.node_start,s.endpoint_start) for s in packed.segments], [(0,0),(3,3),(4,5)])
        self.assertEqual(ints(packed.task.offsets), (0,0,2,3,5,7,7))
        self.assertEqual(ints(packed.task.owners), (1,1,2,3,3,4,4))
        self.assertEqual(len(packed.task.offsets), 4*(packed.task.n+1))
        self.assertNotEqual(packed.segments[2].node_start, 2*self.tasks[0].n)
        self.assertNotEqual(packed.segments[2].endpoint_start, 2*self.tasks[0].m)

    def test_geometry_bytes_original_order(self):
        packed = pack_tasks(self.tasks)
        self.assertEqual(packed.task.centers, b''.join(t.centers for t in self.tasks))
        self.assertEqual(packed.task.endpoints, b''.join(t.endpoints for t in self.tasks))
        self.assertEqual([s.task_id for s in packed.segments], ['A','B','C'])

    def test_reverse_order_prefix_and_identity(self):
        packed = pack_tasks(reversed(self.tasks))
        self.assertEqual([s.task_id for s in packed.segments], ['C','B','A'])
        self.assertEqual([(s.node_start,s.endpoint_start) for s in packed.segments], [(0,0),(2,2),(3,4)])
        self.assertEqual(ints(packed.task.offsets), (0,2,2,4,4,6,7))
        self.assertEqual(ints(packed.task.owners), (0,0,2,2,4,4,5))

    def test_same_geometry_independent_ids_allowed(self):
        second = replace(self.tasks[0], task_id='independent_equal_geometry')
        packed = pack_tasks([self.tasks[0],second])
        self.assertEqual(packed.task.n, 6)
        self.assertEqual(packed.task.centers, self.tasks[0].centers*2)
        self.assertEqual([s.task_id for s in packed.segments], ['A','independent_equal_geometry'])

    def test_duplicate_task_id_rejected_without_fallback_class(self):
        with self.assertRaises(ValueError) as caught:
            pack_tasks([self.tasks[0], self.tasks[0]])
        self.assertNotIsInstance(caught.exception, IncompatibleTasks)

    def test_empty_nodes_empty_task_and_zero_degree(self):
        empty = task('empty', [])
        zeros = task('zero-degree', [0,0], 7)
        packed = pack_tasks([empty,self.tasks[0],zeros])
        self.assertEqual((packed.task.n,packed.task.m), (5,3))
        self.assertEqual(ints(packed.task.offsets), (0,0,2,3,3,3))
        only_empty = pack_tasks([empty])
        self.assertEqual(ints(only_empty.task.offsets), (0,))
        self.assertEqual(split_outputs({n:b'' for n in OUTPUTS},only_empty.segments),
                         {'empty':{n:b'' for n in OUTPUTS}})

    def test_duplicate_endpoint_coordinates_not_removed(self):
        repeated = Task('repeated', binding(), 1, 3, (C.c_double*3)(1,2,3),
                        (C.c_double*9)(4,5,6,4,5,6,4,5,6), (C.c_int32*2)(0,3))
        packed = pack_tasks([repeated,self.tasks[0]])
        self.assertEqual(packed.task.endpoints[:72], repeated.endpoints)
        self.assertEqual(ints(repeated.owners), (0,0,0))

    def test_all_binding_dimensions_checked(self):
        differences = dict(field_id='other-field', epoch=2, device_id='other-device',
                           context_id='other-context', field_parameters_sha256='2'*64,
                           numeric_profile='other-profile', backend='graph', k=6,
                           lattices=61, trace=True, audit=True)
        for name,value in differences.items():
            with self.subTest(name=name):
                other = replace(self.tasks[1], binding=replace(self.tasks[1].binding, **{name:value}))
                with self.assertRaises(IncompatibleTasks):
                    pack_tasks([self.tasks[0],other])

    def test_schema_frozen_and_new_epoch_not_equal(self):
        with self.assertRaises(FrozenInstanceError):
            self.tasks[0].n = 9
        with self.assertRaises(FrozenInstanceError):
            self.tasks[0].binding.epoch = 2
        self.assertNotEqual(binding(),binding(epoch=2))

    def test_task_snapshots_mutable_inputs(self):
        c = (C.c_double*3)(-0.0,2,3)
        e = bytearray(struct.pack('<3d',4,5,6)); off = array('i',[0,1])
        created = Task('snap',binding(),1,1,c,e,off)
        before = (created.centers,created.endpoints,created.offsets,created.owners)
        c[0]=99; e[0]=44; off[1]=0
        self.assertEqual((created.centers,created.endpoints,created.offsets,created.owners), before)
        self.assertEqual(created.centers[:8],struct.pack('<d',-0.0))

    def test_ctypes_geometry_owned_independent(self):
        t = self.tasks[0]
        one, two = t.ctypes_geometry(), t.ctypes_geometry()
        one['centers'][0]=300; one['offsets'][1]=9
        self.assertNotEqual(one['centers'][0],two['centers'][0])
        self.assertEqual(ints(t.offsets), (0,0,2,3))
        self.assertEqual(two['owners'][:], [1,1,2])

    def test_accept_contiguous_typed_buffers_no_numpy(self):
        t = Task('typed',binding(),1,1,array('d',[1,2,3]),memoryview(array('d',[4,5,6])),array('i',[0,1]))
        self.assertEqual(t.centers,struct.pack('<3d',1,2,3))

    def test_noncontiguous_rejected_explicit_copy_allowed(self):
        view = memoryview(array('d',[1,9,2,9,3,9]))[::2]
        with self.assertRaisesRegex(ValueError,'non-contiguous'):
            Task('bad',binding(),1,0,view,b'',struct.pack('<2i',0,0))
        good = Task('copy',binding(),1,0,view.tobytes(),b'',struct.pack('<2i',0,0))
        self.assertEqual(good.centers,struct.pack('<3d',1,2,3))

    def test_wrong_dtype_and_implicit_list_rejected(self):
        with self.assertRaisesRegex(ValueError,'dtype'):
            Task('float32',binding(),1,0,(C.c_float*3)(1,2,3),b'',(C.c_int32*2)(0,0))
        with self.assertRaises(TypeError):
            Task('list',binding(),1,0,[1.,2.,3.],b'',(C.c_int32*2)(0,0))
        with self.assertRaisesRegex(ValueError,'dtype'):
            Task('int64',binding(),1,0,(C.c_double*3)(1,2,3),b'',(C.c_int64*2)(0,0))

    def test_nonfinite_source_rejected(self):
        for bad in (float('nan'),float('inf'),-float('inf')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError,'nonfinite'):
                Task('bad',binding(),1,0,(C.c_double*3)(bad,0,0),b'',(C.c_int32*2)(0,0))

    def test_csr_invalid_extents_monotonicity_owners(self):
        base = self.tasks[0]
        for offsets in ((0,0,2,2),(0,2,1,3),(0,-1,2,3),(0,0,4,3)):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError) as caught:
                replace(base, offsets=struct.pack('<4i',*offsets))
            self.assertNotIsInstance(caught.exception,IncompatibleTasks)
        with self.assertRaisesRegex(ValueError,'extent'):
            replace(base, offsets=struct.pack('<5i',0,0,2,3,3))
        with self.assertRaisesRegex(ValueError,'owners disagree'):
            replace(base, owners=struct.pack('<3i',1,0,2))

    def test_invalid_n_m_k_before_c_arrays(self):
        for kwargs in ({'n':-1},{'n':True},{'m':1<<31}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                replace(self.tasks[0],**kwargs)
        with self.assertRaises(ValueError):
            checked_capacity(0,1,3,60)
        with self.assertRaises(ValueError):
            binding(k=7)
        with self.assertRaises(ValueError):
            binding(trace=1)

    def test_capacity_qL_round_counts_and_budget_before_alloc(self):
        plan=checked_capacity(108,581,6,60,True)
        self.assertEqual(plan['descriptor_capacity'],581*60)
        self.assertEqual(plan['total_descriptor_slots'],581*60*8)
        self.assertEqual(plan['endpoint_round_queries'],581*7)
        self.assertEqual(plan['logical_queries'],4175)
        self.assertEqual(plan['bounds_samples'],756)
        self.assertEqual(plan['output_bytes'],36*108+32*4175+16*756)
        with self.assertRaises(BudgetExceeded):
            checked_capacity(108,581,6,60,True,max_bytes=plan['known_buffer_bytes']-1)
        self.assertEqual(checked_capacity(108,581,6,60,True,max_bytes=plan['known_buffer_bytes']),plan)
        with self.assertRaises(BudgetExceeded):
            pack_tasks(self.tasks,max_bytes=1)

    def test_capacity_integer_and_size_t_overflow(self):
        with self.assertRaisesRegex(ValueError,r'q\*lattices'):
            checked_capacity(1,(1<<31)-1,6,1)
        with self.assertRaisesRegex(ValueError,r'q\*lattices'):
            checked_capacity(1,10_000_000,6,256)
        with patch.object(schema,'SIZE_MAX',(1<<32)-1):
            with self.assertRaisesRegex(ValueError,'overflows size_t'):
                checked_capacity(1,100_000_000,0,1)
        with self.assertRaisesRegex(ValueError,'max_bytes'):
            checked_capacity(1,1,3,60,max_bytes=(2<<30)+1)

    def make_outputs(self, tasks, full=True):
        """Build marked observations directly in global query/round order."""
        fields = OUTPUTS if full else schema.FIVE
        n=sum(t.n for t in tasks);m=sum(t.m for t in tasks);k=tasks[0].k
        merged={};expected={t.task_id:{} for t in tasks}
        for fi,name in enumerate(fields):
            w=OUTPUT_WIDTHS[name];node=[];rounds=[[] for _ in range(k+1)]
            for ti,t in enumerate(tasks):
                seed=(ti+1)*1_000_000+fi*10_000
                center=[token(seed+i,w) for i in range(t.n)]
                if name in schema.FIVE:
                    own=b''.join(center);node.extend(center)
                elif name.startswith('trace_'):
                    local=[[token(seed+1000*r+i,w) for i in range(t.n)] for r in range(k+1)]
                    own=b''.join(v for stripe in local for v in stripe)
                    for r in range(k+1):rounds[r].extend(local[r])
                else:
                    local=[[token(seed+1000*(r+1)+i,w) for i in range(t.m)] for r in range(k+1)]
                    own=b''.join(center)+b''.join(v for stripe in local for v in stripe)
                    node.extend(center)
                    for r in range(k+1):rounds[r].extend(local[r])
                expected[t.task_id][name]=own
            merged[name]=b''.join(node)+b''.join(v for stripe in rounds for v in stripe)
        return merged,expected

    def test_five_output_dispatch_preserves_witness_local_rank(self):
        packed=pack_tasks(self.tasks)
        merged,expected=self.make_outputs(self.tasks,False)
        merged['witness']=struct.pack('<6i',-1,1,0,1,1,-1)
        for t,values in zip(self.tasks,((-1,1,0),(1,),(1,-1))):
            expected[t.task_id]['witness']=struct.pack('<'+'i'*len(values),*values)
        self.assertEqual(split_outputs(merged,packed.segments),expected)

    def test_full_round_major_trace_forward_reverse_and_tails(self):
        for k in (0,3,6):
            for reverse in (False,True):
                tasks=[replace(t,binding=binding(k=k,trace=True)) for t in self.tasks]
                if reverse:tasks.reverse()
                packed=pack_tasks(tasks);merged,expected=self.make_outputs(tasks)
                self.assertEqual(split_outputs(merged,packed.segments),expected)
                self.assertNotEqual(merged['sdf'],b''.join(expected[t.task_id]['sdf'] for t in tasks))
                self.assertEqual(list(split_outputs(merged,packed.segments)),[t.task_id for t in tasks])

    def test_returned_outputs_survive_mutated_deleted_backing(self):
        packed=pack_tasks(self.tasks)
        merged,expected=self.make_outputs(self.tasks)
        backing={k:bytearray(v) for k,v in merged.items()}
        result=split_outputs(backing,packed.segments)
        for a in backing.values():a[:]=b'\xff'*len(a)
        backing.clear();merged.clear()
        self.assertEqual(result,expected)
        self.assertTrue(all(isinstance(b,bytes) for x in result.values() for b in x.values()))

    def test_output_nan_signedzero_bits_are_not_reinterpreted(self):
        packed=pack_tasks([self.tasks[1]])
        outputs={n:b'\0'*(OUTPUT_WIDTHS[n]*self.tasks[1].n) for n in schema.FIVE}
        outputs['position']=struct.pack('<3I',0x80000000,0x7fc01234,0x7f800000)
        got=split_outputs(outputs,packed.segments)
        self.assertEqual(got['B']['position'],outputs['position'])

    def test_output_invalid_keys_extents_dtype_segments(self):
        packed=pack_tasks(self.tasks);merged,_=self.make_outputs(self.tasks)
        missing=dict(merged);del missing['valid']
        with self.assertRaisesRegex(ValueError,'five'):
            split_outputs(missing,packed.segments)
        extra=dict(merged,unknown=b'')
        with self.assertRaisesRegex(ValueError,'Unknown'):
            split_outputs(extra,packed.segments)
        short=dict(merged,sdf=merged['sdf'][:-4])
        with self.assertRaisesRegex(ValueError,'extent'):
            split_outputs(short,packed.segments)
        wrong=dict(merged,valid=(C.c_float*6)())
        with self.assertRaisesRegex(ValueError,'dtype'):
            split_outputs(wrong,packed.segments)
        with self.assertRaisesRegex(ValueError,'contiguous'):
            split_outputs(merged,packed.segments[::-1])

    def test_copy_outputs_and_comparator_exact_difference(self):
        merged,_=self.make_outputs(self.tasks,False)
        copied=copy_outputs(merged,6,7,3)
        self.assertEqual(copied,merged)
        wrong=dict(copied);data=bytearray(wrong['valid']);data[3]^=1;wrong['valid']=data
        result=compare_outputs(wrong,copied)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['first_difference']['byte_offset'],3)
        self.assertEqual(compare_outputs(copied,copied)['status'],'PASS')


if __name__ == '__main__':
    unittest.main()
