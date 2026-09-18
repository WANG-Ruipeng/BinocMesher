"""CPU-only runner contract tests; no fake result is claimed as a CUDA result."""
from dataclasses import replace
import struct, unittest
from unittest.mock import patch
from gpu_accelerate.batching.schema import BindingKey,Task
from gpu_accelerate.batching.runner import solve_many,BatchExecutionError
from gpu_accelerate.batching.backend import PreSubmitBudgetError

class FakeSolver:
    def __init__(self,field,task):self.field=field;self.task=task;self.closed=False;self.success=False
    def prepare(self):self.field.events.append('prepare')
    def run(self):
        self.field.events.append('run')
        if self.field.fail_run:raise RuntimeError('real execution failure stand-in')
        self.success=True
    def readback(self):
        if not self.success:raise RuntimeError('not ready')
        n=self.task.n
        return {'position':bytes(12*n),'witness':bytes(4*n),'valid':bytes(4*n),'left':bytes(8*n),'right':bytes(8*n)}
    def validation(self):return [0]*12
    def close(self):
        self.field.events.append('close');self.closed=True
        if self.field.fail_cleanup:raise RuntimeError('cleanup failure stand-in')

class FakeField:
    def __init__(self):
        self.key=BindingKey('field',1,'device','context','a'*64,'profile','published',6,60)
        self.created=[];self.events=[];self.fail_run=False;self.fail_cleanup=False;self.reject_large=False;self.active=True
    def binding(self,k,trace=False,audit=False):
        if not self.active:raise RuntimeError('stale field')
        return replace(self.key,k=k,trace=trace,audit=audit)
    def create_solver(self,task):
        self.events.append('create')
        if self.reject_large and task.n>1:raise PreSubmitBudgetError(2,'create fully cleaned')
        s=FakeSolver(self,task);self.created.append(s);return s

def task(field,name,k=6):
    return Task(name,field.binding(k),1,1,struct.pack('<3d',0,0,0),struct.pack('<3d',1,0,0),struct.pack('<2i',0,1),struct.pack('<i',0))

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.trap=patch('ctypes.CDLL',side_effect=AssertionError('CPU test loaded CUDA'));self.trap.start()
        self.field=FakeField();self.tasks=[task(self.field,'a'),task(self.field,'b')]
    def tearDown(self):self.trap.stop()
    def test_default_off_and_complete_cleanup(self):
        r=solve_many(self.field,self.tasks)
        self.assertEqual(r.selected_path,'off');self.assertEqual(len(self.field.created),2)
        self.assertEqual(r.task_ids,('a','b'));self.assertTrue(all(s.closed for s in self.field.created))
    def test_offline_one_solver_and_owned_dispatch(self):
        r=solve_many(self.field,self.tasks,batching='offline')
        self.assertEqual(len(self.field.created),1);self.assertEqual(tuple(r.outputs),('a','b'))
        self.assertTrue(all(isinstance(a,bytes) for o in r.outputs.values() for a in o.values()))
    def test_empty_no_field_calls(self):self.assertEqual(solve_many(None,[]).outputs,{})
    def test_graph_rejected_before_loading(self):
        with self.assertRaisesRegex(NotImplementedError,'OPTIONAL_BACKEND_NOT_PACKAGED'):solve_many(None,[],backend='graph')
    def test_mixed_k_explicit_fallback(self):
        tasks=[task(self.field,'a',3),task(self.field,'b',6)]
        r=solve_many(self.field,tasks,batching='offline',fallback='serial')
        self.assertEqual(r.selected_path,'serial_fallback');self.assertEqual(len(self.field.created),2)
    def test_mixed_k_default_reject_without_submit(self):
        with self.assertRaises(ValueError):solve_many(self.field,[task(self.field,'a',3),task(self.field,'b',6)],batching='offline')
        self.assertEqual(self.field.created,[])
    def test_pre_submit_native_budget_fallback(self):
        self.field.reject_large=True
        r=solve_many(self.field,self.tasks,batching='offline',fallback='serial')
        self.assertEqual(r.selected_path,'serial_fallback');self.assertEqual(len(self.field.created),2)
    def test_pack_budget_explicit_fallback(self):
        r=solve_many(self.field,self.tasks,batching='offline',fallback='serial',max_pack_bytes=1)
        self.assertEqual(r.selected_path,'serial_fallback');self.assertEqual(len(self.field.created),2)
    def test_run_failure_never_falls_back(self):
        self.field.fail_run=True
        with self.assertRaisesRegex(RuntimeError,'execution failure'):solve_many(self.field,self.tasks,batching='offline',fallback='serial')
        self.assertEqual(len(self.field.created),1);self.assertTrue(self.field.created[0].closed)
    def test_partial_serial_failure_does_not_return_partial_success(self):
        original=self.field.create_solver
        def create(t):
            if self.field.created:raise PreSubmitBudgetError(2,'second create cleaned')
            return original(t)
        self.field.create_solver=create
        with self.assertRaises(PreSubmitBudgetError):solve_many(self.field,self.tasks)
        self.assertTrue(self.field.created[0].closed)
    def test_cleanup_failure_preserves_first_error(self):
        self.field.fail_run=True;self.field.fail_cleanup=True
        with self.assertRaises(BatchExecutionError) as caught:solve_many(self.field,self.tasks,batching='offline')
        self.assertIn('execution failure',str(caught.exception.primary));self.assertIn('cleanup failure',str(caught.exception.cleanup))
    def test_stale_and_different_field_rejected(self):
        self.field.active=False
        with self.assertRaises(RuntimeError):solve_many(self.field,self.tasks,batching='offline')
        self.field.active=True;self.field.key=replace(self.field.key,epoch=2)
        with self.assertRaises(ValueError):solve_many(self.field,self.tasks,batching='offline',fallback='serial')
        self.assertEqual(self.field.created,[])
    def test_duplicate_ids_rejected_before_submit(self):
        with self.assertRaises(ValueError):solve_many(self.field,[self.tasks[0],self.tasks[0]])
        self.assertEqual(self.field.created,[])
if __name__=='__main__':unittest.main()