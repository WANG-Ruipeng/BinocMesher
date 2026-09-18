"""Explicit Published-only native backend. Importing does not load CUDA.

The native Field schema/coordinate domain remains the historical validated
subset. Inputs outside it are rejected, not silently treated as general fields.
Only one live backend is allowed per process; calls are serialized and all
Fields/solvers share one native allocation budget and CUDA context identity.
"""
from __future__ import annotations
import ctypes as C
import hashlib
from pathlib import Path
import struct
import threading
import uuid

I=C.c_int32; U=C.c_uint64; Z=C.c_size_t; F=C.c_float; D=C.c_double
PI=C.POINTER(I); PU=C.POINTER(U); PF=C.POINTER(F); PD=C.POINTER(D); PC=C.POINTER(C.c_char)
_SIGNATURES={
 'mb_abi':[PU,Z], 'mb_configure':[Z], 'mb_context':[PU,Z,PC,Z,PC,Z],
 'mb_field_create':[I,Z,PI,Z,PF,Z,PI,Z,PF,PU],
 'mb_field_invalidate':[U], 'mb_field_destroy':[U],
 'mb_solver_create':[U,I,I,PI,PD,PD,I,PU],
 'mb_solver_prepare':[U,I,I,I], 'mb_solver_run':[U,I,I,I],
 'mb_solver_readback':[U,PF,Z,PI,Z,PI,Z,PD,Z,PD,Z],
 'mb_solver_trace':[U,PF,PF,PF,PI,PD,PD,Z,Z],
 'mb_solver_counts':[U,PU,Z], 'mb_solver_validation':[U,PU,Z],
 'mb_solver_info':[U,PU,Z], 'mb_solver_destroy':[U],
}
_LIVE=None
_PROCESS_LOCK=threading.RLock()

class BackendError(RuntimeError):
    def __init__(self,code,operation):
        self.code=int(code);self.operation=operation
        suffix=': unsupported validated field/geometry domain' if self.code==801 else ''
        super().__init__(f'{operation}: CUDA/adapter status {self.code}{suffix}')

class PreSubmitBudgetError(BackendError):
    """Create-only OOM after partial allocation cleanup completed successfully."""

BudgetExceeded=PreSubmitBudgetError

def _check(code,operation):
    if code:raise BackendError(code,operation)

def _create_failure(code,token,destroy,operation):
    if token:
        cleanup=int(destroy(token))
        if cleanup:
            error=BackendError(code,operation+'; partial cleanup also failed with status '+str(cleanup))
            error.cleanup_code=cleanup;error.partial_token=token
            raise error
    if code==2:raise PreSubmitBudgetError(code,operation+' (partial state cleaned)')
    raise BackendError(code,operation)

def _bytes(value):
    if isinstance(value,bytes):return value
    if isinstance(value,(bytearray,memoryview)):return bytes(value)
    if isinstance(value,C.Array):return C.string_at(C.addressof(value),C.sizeof(value))
    raise TypeError('Native input must be a bytes snapshot or ctypes array')

def _array(typ,value,count=None):
    data=_bytes(value)
    if len(data)%C.sizeof(typ) or (count is not None and len(data)!=count*C.sizeof(typ)):
        raise ValueError('Native input byte extent mismatch')
    return (typ*(len(data)//C.sizeof(typ))).from_buffer_copy(data)

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''):h.update(data)
    return h.hexdigest()

class PublishedBackend:
    """Load an explicitly supplied freshly built libbatching_published.so.

    Construction binds the library but does not initialize a CUDA context.
    context()/create_field() are explicit CUDA operations. Graph is not packaged.
    Close all resources before leaving the creating CUDA context.
    """
    backend='published'
    def __init__(self,library,*,expected_device_uuid=None,memory_limit_bytes=2**31,backend='published'):
        global _LIVE
        if backend!='published':raise NotImplementedError('OPTIONAL_BACKEND_NOT_PACKAGED: '+str(backend))
        if type(memory_limit_bytes) is not int or not 0<memory_limit_bytes<=2**31:
            raise ValueError('memory_limit_bytes must be in [1, 2 GiB]')
        with _PROCESS_LOCK:
            if _LIVE is not None:raise RuntimeError('Only one live PublishedBackend per process is permitted')
            self.library=Path(library).expanduser().resolve(strict=True)
            self.library_sha256=_sha(self.library);self.numeric_profile='published-strict:'+self.library_sha256
            self.expected_device_uuid=expected_device_uuid;self.memory_limit_bytes=memory_limit_bytes
            self.instance_id=uuid.uuid4().hex;self._fields={};self._closed=False
            self._lib=C.CDLL(str(self.library),mode=C.RTLD_LOCAL)
            for name,args in _SIGNATURES.items():
                fn=getattr(self._lib,name);fn.argtypes=args;fn.restype=I
            words=(U*8)();_check(self._lib.mb_abi(words,8),'ABI')
            if list(words)!=[1,8,8,8,1,128,6,0]:raise RuntimeError('Published batching native ABI mismatch')
            _check(self._lib.mb_configure(memory_limit_bytes),'configure shared budget')
            _LIVE=self
    def _live(self):
        if self._closed:raise RuntimeError('Backend is closed')
    def context(self):
        with _PROCESS_LOCK:
            self._live();words=(U*16)();name=C.create_string_buffer(256);ident=C.create_string_buffer(41)
            _check(self._lib.mb_context(words,16,name,256,ident,41),'context')
            w=list(words);device_id=ident.value.decode('ascii')
            if self.expected_device_uuid is not None and device_id!=self.expected_device_uuid:
                raise RuntimeError('CUDA device UUID differs from explicitly selected device')
            return dict(words=w,device_id=device_id,context_id=str(w[12]),name=name.value.decode(),
                free_device_bytes=w[7],total_device_bytes=w[8],limit_bytes=w[9],tracked_allocation_bytes=w[10],
                initial_free_bytes=w[11],graph_reserved_bytes=0,live_fields=w[14],live_solvers=w[15],
                measurement='cudaMemGetInfo point observation; device peak unknown')
    def create_field(self,meta,tree_ip,tree_fp,land_ip,land_fp,*,parameters_sha256=None):
        with _PROCESS_LOCK:
            self._live();ctx=self.context()
            ip=_array(I,tree_ip);fp=_array(F,tree_fp);ip2=_array(I,land_ip);fp2=_array(F,land_fp)
            if type(meta) is not int or not -(2**31)<=meta<2**31:raise ValueError('meta must be int32')
            if parameters_sha256 is not None and (len(parameters_sha256)!=64 or any(c not in '0123456789abcdef' for c in parameters_sha256)):
                raise ValueError('parameters_sha256 must be lowercase SHA256')
            digest=hashlib.sha256(struct.pack('<i4Q',meta,len(ip),len(fp),len(ip2),len(fp2)))
            for a in (ip,fp,ip2,fp2):digest.update(_bytes(a))
            token=U();code=int(self._lib.mb_field_create(meta,len(ip),ip,len(fp),fp,len(ip2),ip2,len(fp2),fp2,C.byref(token)))
            if code:_create_failure(code,token.value,self._lib.mb_field_destroy,'field_create')
            field=Field(self,token.value,ctx,digest.hexdigest(),parameters_sha256,int(ip[1]))
            self._fields[token.value]=field;return field
    def close(self):
        global _LIVE
        with _PROCESS_LOCK:
            if self._closed:return
            for field in list(self._fields.values()):field.close()
            ctx=self.context()
            if ctx['tracked_allocation_bytes'] or ctx['live_fields'] or ctx['live_solvers']:
                raise RuntimeError('Native resources remain after backend cleanup')
            self._closed=True;_LIVE=None
    def __enter__(self):self._live();return self
    def __exit__(self,*_):self.close()

class Field:
    def __init__(self,backend,token,context,payload_sha,source_sha,lattices):
        self.backend=backend;self.token=token;self.field_id=backend.instance_id+':'+str(token)
        self.epoch=1;self.device_id=context['device_id'];self.context_id=context['context_id']
        self.field_parameters_sha256=payload_sha;self.source_parameters_sha256=source_sha
        self.numeric_profile=backend.numeric_profile;self.lattices=lattices
        self._active=True;self._closed=False;self._solvers={}
    def _live(self):
        self.backend._live()
        if not self._active or self._closed:raise RuntimeError('Field identity/epoch is no longer live')
    def binding(self,k,trace=False,audit=False):
        from .schema import BindingKey
        with _PROCESS_LOCK:
            self._live();context=self.backend.context()
            if (context['device_id'],context['context_id'])!=(self.device_id,self.context_id):
                raise RuntimeError('Field CUDA context/device changed')
            return BindingKey(field_id=self.field_id,epoch=self.epoch,device_id=self.device_id,context_id=self.context_id,
                field_parameters_sha256=self.field_parameters_sha256,numeric_profile=self.numeric_profile,
                backend='published',k=k,lattices=self.lattices,trace=trace,audit=audit)
    def create_solver(self,task):
        from .schema import Task
        with _PROCESS_LOCK:
            self._live()
            if not isinstance(task,Task):raise TypeError('Expected an immutable validated Task')
            if task.binding.backend!='published':raise NotImplementedError('OPTIONAL_BACKEND_NOT_PACKAGED: '+task.binding.backend)
            if task.binding!=self.binding(task.binding.k,task.binding.trace,task.binding.audit):
                raise ValueError('Task Field identity/epoch/device/context/profile does not match live Field')
            centers=_array(D,task.centers,3*task.n);endpoints=_array(D,task.endpoints,3*task.m)
            offsets=_array(I,task.offsets,task.n+1);token=U()
            code=int(self.backend._lib.mb_solver_create(self.token,task.n,task.m,offsets,centers,endpoints,6,C.byref(token)))
            if code:_create_failure(code,token.value,self.backend._lib.mb_solver_destroy,'solver_create')
            solver=Solver(self,token.value,task);self._solvers[token.value]=solver;return solver
    def invalidate(self):
        with _PROCESS_LOCK:
            if self._closed:raise RuntimeError('Field is closed')
            _check(self.backend._lib.mb_field_invalidate(self.token),'field_invalidate')
            self._active=False;self.epoch+=1
            for solver in self._solvers.values():solver._last=None
    def close(self):
        with _PROCESS_LOCK:
            if self._closed:return
            self._active=False;self.epoch+=1
            for solver in list(self._solvers.values()):solver.close()
            _check(self.backend._lib.mb_field_destroy(self.token),'field_destroy')
            self._closed=True;self.backend._fields.pop(self.token,None)
    def __enter__(self):self._live();return self
    def __exit__(self,*_):self.close()

class Solver:
    def __init__(self,field,token,task):
        self.field=field;self.token=token;self.task=task;self._closed=False;self._last=None
        self.k=task.binding.k;self.trace=task.binding.trace;self.audit=task.binding.audit
    @property
    def _lib(self):return self.field.backend._lib
    def _live(self):
        self.field._live()
        if self._closed:raise RuntimeError('Solver token is destroyed')
    def _config(self,k,trace,audit):
        k=self.k if k is None else k;trace=self.trace if trace is None else trace;audit=self.audit if audit is None else audit
        # Preserve in-range invalid K (e.g. 7) for the native invalidation path;
        # never let ctypes truncate a Python integer into a different solve.
        if type(k) is not int or not -(2**31)<=k<2**31:
            raise ValueError('k must be a signed int32 integer; no implicit conversion')
        if type(trace) is not bool or type(audit) is not bool:
            raise TypeError('trace and audit must be bool')
        return k,trace,audit
    def prepare(self,k=None,trace=None,audit=None):
        with _PROCESS_LOCK:
            self._last=None;self._live();k,trace,audit=self._config(k,trace,audit)
            _check(self._lib.mb_solver_prepare(self.token,k,int(trace),int(audit)),'solver_prepare')
            self.k,self.trace,self.audit=k,trace,audit
    def run(self,k=None,trace=None,audit=None):
        with _PROCESS_LOCK:
            self._last=None;self._live();k,trace,audit=self._config(k,trace,audit)
            _check(self._lib.mb_solver_run(self.token,k,int(trace),int(audit)),'solver_run')
            self.k,self.trace,self.audit=k,trace,audit;self._last=(k,trace,audit)
    def readback(self):
        with _PROCESS_LOCK:
            self._live()
            if self._last is None:raise RuntimeError('No successful current solve; old readback is invalid')
            n,m=self.task.n,self.task.m;k,trace,audit=self._last;q=n+(k+1)*m;b=(k+1)*n
            arrays={'position':(F*(3*n))(),'witness':(I*n)(),'valid':(I*n)(),'left':(D*n)(),'right':(D*n)()}
            try:
                _check(self._lib.mb_solver_readback(self.token,arrays['position'],3*n,arrays['witness'],n,arrays['valid'],n,arrays['left'],n,arrays['right'],n),'five readback')
                if trace:
                    arrays.update(aux=(F*(3*q))(),sdf=(F*q)(),xyz=(F*(3*q))(),sign=(I*q)(),trace_left=(D*b)(),trace_right=(D*b)())
                    _check(self._lib.mb_solver_trace(self.token,arrays['aux'],arrays['sdf'],arrays['xyz'],arrays['sign'],arrays['trace_left'],arrays['trace_right'],q,b),'full trace readback')
                return {key:_bytes(value) for key,value in arrays.items()}
            except BaseException:self._last=None;raise
    def counts(self):
        with _PROCESS_LOCK:
            self._live()
            if self._last is None:raise RuntimeError('No current successful solve for counts')
            words=(U*160)();_check(self._lib.mb_solver_counts(self.token,words,160),'solver_counts');return list(words)
    def validation(self):
        with _PROCESS_LOCK:
            self._live()
            if self._last is None:raise RuntimeError('No current successful solve for validation')
            words=(U*12)();_check(self._lib.mb_solver_validation(self.token,words,12),'solver_validation');return list(words)
    def info(self):
        with _PROCESS_LOCK:
            self._live();words=(U*24)();_check(self._lib.mb_solver_info(self.token,words,24),'solver_info');w=list(words)
            return dict(words=w,base_planned_bytes=w[15],pipeline_planned_bytes=w[16],global_tracked_bytes=w[17],global_limit_bytes=w[18],graph_reserved_bytes=0,successful_solves=w[20],attempt=w[21])
    def close(self):
        with _PROCESS_LOCK:
            if self._closed:return
            self._last=None;_check(self._lib.mb_solver_destroy(self.token),'solver_destroy')
            self._closed=True;self.field._solvers.pop(self.token,None)
    def __enter__(self):self._live();return self
    def __exit__(self,*_):self.close()

Backend=PublishedBackend
