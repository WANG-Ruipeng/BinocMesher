"""CPU-only backend contracts; no native library is loaded."""
import ctypes as C
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gpu_accelerate.batching import backend as b

class BackendContracts(unittest.TestCase):
    def solver(self):
        calls=[]
        lib=SimpleNamespace(mb_solver_run=lambda token,k,t,a:calls.append(('run',k)) or (0 if 0<=k<=6 else 1),
            mb_solver_prepare=lambda token,k,t,a:calls.append(('prepare',k)) or (0 if 0<=k<=6 else 1),
            mb_solver_readback=lambda *a:0,mb_solver_trace=lambda *a:0,mb_solver_validation=lambda *a:0,
            mb_solver_counts=lambda *a:0,mb_solver_destroy=lambda *a:0)
        field=SimpleNamespace(_live=lambda:None,backend=SimpleNamespace(_lib=lib),_solvers={})
        task=SimpleNamespace(n=2,m=3,binding=SimpleNamespace(k=6,trace=False,audit=False))
        solver=b.Solver(field,101,task);field._solvers[101]=solver
        return solver,calls
    def test_graph_rejected_before_library_load(self):
        with patch.object(C,'CDLL',side_effect=AssertionError('CUDA load forbidden')):
            with self.assertRaisesRegex(NotImplementedError,'OPTIONAL_BACKEND_NOT_PACKAGED'):
                b.PublishedBackend('/missing',backend='graph')
    def test_create_cleanup_error_preserves_both_codes_and_token(self):
        with self.assertRaises(b.BackendError) as caught:b._create_failure(2,101,lambda token:700,'solver_create')
        error=caught.exception
        self.assertEqual((error.code,error.cleanup_code,error.partial_token),(2,700,101))
        self.assertNotIsInstance(error,b.PreSubmitBudgetError)
        with self.assertRaises(b.PreSubmitBudgetError):b._create_failure(2,101,lambda token:0,'solver_create')
        with self.assertRaises(b.BackendError) as fatal:b._create_failure(700,101,lambda token:0,'solver_create')
        self.assertNotIsInstance(fatal.exception,b.PreSubmitBudgetError)
    def test_huge_nonintegral_k_and_nonbool_flags_do_not_reach_native(self):
        for operation in ('run','prepare'):
            for kwargs in ({'k':2**32+3},{'k':-(2**31)-1},{'k':3.0},{'k':True},{'trace':1},{'audit':0.5}):
                solver,calls=self.solver();solver.run();before=list(calls)
                with self.assertRaises((TypeError,ValueError)):getattr(solver,operation)(**kwargs)
                self.assertEqual(calls,before);self.assertIsNone(solver._last)
                for old in (solver.readback,solver.counts,solver.validation):
                    with self.assertRaises(RuntimeError):old()
    def test_int32_invalid_k_reaches_native_and_invalidates(self):
        solver,calls=self.solver();solver.run()
        with self.assertRaises(b.BackendError):solver.run(k=7)
        self.assertEqual(calls,[('run',6),('run',7)])
        for old in (solver.readback,solver.counts,solver.validation):
            with self.assertRaises(RuntimeError):old()
    def test_outputs_own_bytes_after_destruction(self):
        solver,_=self.solver();solver.prepare();solver.run();outputs=solver.readback()
        self.assertEqual(set(outputs),{'position','witness','valid','left','right'})
        self.assertEqual(solver.validation(),[0]*12);solver.close()
        self.assertEqual([len(outputs[k]) for k in ('position','witness','valid','left','right')],[24,8,8,16,16])
        self.assertTrue(all(isinstance(value,bytes) for value in outputs.values()))
        self.assertFalse(solver.field._solvers)
    def test_full_trace_owned_extents(self):
        solver,_=self.solver();solver.run(trace=True,audit=True);outputs=solver.readback()
        self.assertEqual(len(outputs),11);self.assertEqual(len(outputs['aux']),23*12)
        self.assertEqual(len(outputs['trace_left']),14*8);self.assertEqual(len(solver.counts()),160)

if __name__=='__main__':unittest.main(verbosity=2)
