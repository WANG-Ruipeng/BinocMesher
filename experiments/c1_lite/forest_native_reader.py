"""Isolated five-element native baseline/identity reader; no build or overlay.

Only load_parameters, ordinary slicing, identity sidecars and cleanup are
called. The caller owns a private cache COPY and must hash its original inputs
before/after, allowing only bounded log append. No source cache, production
library, tree, preprocessing or plan is written by this module.

The C++ state is process-global (observer additionally thread-local): use one
reader in one thread per worker, never concurrent native mesher instances.
"""
from __future__ import annotations

import ctypes
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import threading
import time

import numpy as np

FROZEN_SO_SHA256='f4263a2f47ba5283175a921e49b8867998bdd8124aac810793b34242ec43a3c9'
ORIGINAL_CACHE=Path('/home/warpwang/binoc-runs/forest-census64-repair-20260906/HyperMesh/OpaqueTerrain')
MAX_QUERY_ARRAY_BYTES=2*1024**3
MAX_RSS_BYTES=8*1024**3
N_ELEMENTS=5
_ACTIVE=None
_LOCK=threading.RLock()


class NativeReadError(RuntimeError):
    pass


def require(value,message):
    if not value:raise NativeReadError(message)


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def _document(value):
    if isinstance(value,dict):
        data=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        return value,{'sha256':hashlib.sha256(data).hexdigest(),'binding':'canonical JSON of caller-supplied document'}
    path=Path(value).resolve();data=path.read_bytes()
    return json.loads(data),{'sha256':hashlib.sha256(data).hexdigest(),'path':str(path),'binding':'exact document bytes'}


def _fraction(value):
    return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)


def _peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def _check_rss():
    require(_peak_rss()<=MAX_RSS_BYTES,'NATIVE_PROCESS_PEAK_RSS_EXCEEDS_8_GIB')


def _int_pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _double_pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _error(function):
    result=function()
    return result.decode('utf-8',errors='replace') if isinstance(result,bytes) else str(result or '')


def prepare_parameters(camera,effective):
    """Pure equivalent of frozen core.__init__ camera packing; no native calls."""
    poses=np.asarray(camera['poses'],np.float64);intrinsics=np.asarray(camera['intrinsics'],np.float64)
    heights=np.asarray(camera['heights']);widths=np.asarray(camera['widths'])
    times=[float(t) for t in camera['times_seconds']];n=len(times)
    require(n==64 and poses.shape==(n,4,4) and intrinsics.shape==(n,3,3),'FROZEN_FOREST_CAMERA_SHAPE')
    require(heights.shape==widths.shape==(n,) and np.all(heights==540) and np.all(widths==960),'FROZEN_FOREST_CAMERA_RESOLUTION')
    require(np.all(np.isfinite(poses)) and np.all(np.isfinite(intrinsics)) and all(math.isfinite(t) for t in times),'NONFINITE_FOREST_CAMERA')
    mapping=camera['time_mapping']
    require(mapping['use_alignment'] is False and mapping['min_t_offset']==0,'UNSUPPORTED_FOREST_TIME_ALIGNMENT')
    origin=min(times);relative=[t-origin for t in times];duration=max(relative)+1e-5
    require(origin.hex()==float(_fraction(mapping['origin_seconds'])).hex(),'FOREST_ORIGIN_BITS_MISMATCH')
    require(duration.hex()==float(_fraction(mapping['duration_seconds'])).hex(),'FOREST_DURATION_BITS_MISMATCH')
    require(mapping['temporal_group_count']==2 and mapping['maximum_discrete_time']==4,'FROZEN_FOREST_TIME_SHAPE')
    fading=float(mapping['effective_fading_seconds']);require(fading==1.0,'FROZEN_FOREST_FADING_TIME')
    elements=effective['terrain_elements']
    require(elements==['ground','landtiles','sdf_trees','warped_rocks','voronoi_rocks','atmosphere'],'FROZEN_FOREST_ELEMENT_ORDER')
    require('BinocMesher.pixels_per_cube=6' in effective['overrides'],'FROZEN_FOREST_OPAQUE_LOD')
    bounds=np.asarray(effective['bounds'],np.float64)
    require(bounds.shape==(6,) and np.all(np.isfinite(bounds)) and np.all(bounds[1::2]>bounds[::2]),'INVALID_FOREST_BOUNDS')
    center=np.asarray([(bounds[2*k]+bounds[2*k+1])/2 for k in range(3)],np.float64)
    size=float(max(bounds[1::2]-bounds[::2])*1.1)
    packed=np.empty(n*27,np.float64)
    for i in range(n):
        packed[i*27:(i+1)*27]=np.concatenate([np.linalg.inv(poses[i])[:3,:4].reshape(-1),
            intrinsics[i].reshape(-1),[heights[i]],[widths[i]],[relative[i]],poses[i][:3,3]])
    return {'center':center,'size':size,'duration':duration,'n_cameras':n,'cameras':packed,
        'fading_time':fading,'pixels_per_cube':6.0,'pixels_per_cube_coarse':30.0,
        'pixels_per_cube_outview':120.0,'min_dist':0.1,'origin_seconds':origin,
        'expected_delta':float(_fraction(mapping['delta_seconds'])),'expected_groups':2,
        'maximum_discrete_time':4,
        'parameter_sources':'Recorded camera/effective Forest documents; coarse=30, outview=120, min_dist=0.1 are unchanged frozen core constructor defaults, also recorded in the Forest build protocol.'}


def validate_counts(vertex_counts,face_counts,owner_counts=None):
    require(len(vertex_counts)==len(face_counts)==N_ELEMENTS,'NATIVE_COUNT_ARRAY_SHAPE')
    require(all(0<=int(n)<2**31 for n in (*vertex_counts,*face_counts)),'INVALID_NATIVE_MESH_COUNTS')
    total=sum(int(v)*28+int(f)*12 for v,f in zip(vertex_counts,face_counts))
    if owner_counts is not None:
        require(len(owner_counts)==N_ELEMENTS and all(0<=int(n)<2**31 for n in owner_counts),'INVALID_NATIVE_OWNER_COUNTS')
        require(all(int(v)*4<2**31 for v in vertex_counts) and all(int(n)*11<2**31 for n in owner_counts),'IDENTITY_INT_CAPACITY_OVERFLOW')
        total+=sum(int(v)*16+int(n)*44 for v,n in zip(vertex_counts,owner_counts))
    require(total<=MAX_QUERY_ARRAY_BYTES,'NATIVE_QUERY_ARRAYS_EXCEED_2_GIB')
    return total


def _bind(dll):
    integer=ctypes.c_int32;ip=ctypes.POINTER(integer);double=ctypes.c_double;dp=ctypes.POINTER(double)
    signatures={
        'load_parameters':([dp,double,double,integer,dp,double,double,double,double,double,integer,ctypes.c_char_p],integer),
        'run_slicing':([double,ip,ip,ctypes.c_bool],integer),
        'run_slicing_rational':([ctypes.c_int64,ctypes.c_int64,ip,ip,ctypes.c_bool],integer),
        'slicing_last_error':([],ctypes.c_char_p),'slicing_output':([integer,dp,ip,ip],None),
        'slicing_clean_up':([],None),'slicing_discard_output':([],None),
        'slicing_identity_enable':([ctypes.c_bool],None),'slicing_identity_status':([],integer),
        'slicing_identity_last_error':([],ctypes.c_char_p),
        'slicing_identity_vertex_count':([integer],integer),'slicing_identity_owner_count':([integer],integer),
        'slicing_identity_output_vertices':([integer,ip,integer],integer),
        'slicing_identity_output_owners':([integer,ip,integer],integer)}
    for name,(args,result) in signatures.items():
        function=getattr(dll,name);function.argtypes=args;function.restype=result
    optional={
        'slicing_identity_source_vid_encoding_version':([],integer),
        'slicing_identity_output_source_vid_shifts':([ip,integer],integer)}
    for name,(args,result) in optional.items():
        function=getattr(dll,name,None)
        if function is not None:function.argtypes=args;function.restype=result


def _encoding(dll):
    version_function=getattr(dll,'slicing_identity_source_vid_encoding_version',None)
    if version_function is None:return 0,'MERGER_NORMALIZED_UNVERIFIED'
    version=int(version_function())
    if version==2 and getattr(dll,'slicing_identity_output_source_vid_shifts',None) is not None:
        return version,'ORIGINAL_EFFECTIVE_SOURCE_VID'
    return version,'UNSUPPORTED_OBSERVER_SOURCE_VID_ENCODING'


class NativeForestReader:
    def __init__(self,dll,parameters,cache,library_path,library_hash,document_bindings):
        self.dll=dll;self.parameters=parameters;self.cache=Path(cache)
        self.library_path=Path(library_path);self.library_sha256=library_hash
        self.input_documents_sha256=document_bindings;self.n_elements=N_ELEMENTS
        self.source_vid_encoding_version,self.source_vid_encoding=_encoding(dll)
        self.delta_t=parameters['expected_delta'];self.maximum_discrete_time=4
        self._thread=threading.get_ident();self._closed=False;self._saved_environment={}
        self.initialization={}

    def _discard(self):
        try:self.dll.slicing_discard_output()
        finally:
            try:self.dll.slicing_clean_up()
            finally:self.dll.slicing_identity_enable(False)

    def slice_query(self,value,mode='exact',ledger=True):
        require(not self._closed and threading.get_ident()==self._thread,'NATIVE_READER_CLOSED_OR_WRONG_THREAD')
        require(mode in ('exact','physical'),'UNSUPPORTED_QUERY_MODE')
        if mode=='exact':
            tau=Fraction(value)
            require(0<=tau<=4 and 0<=tau.numerator<2**63 and 0<tau.denominator<2**63,'EXACT_TIME_OUTSIDE_ABI_OR_FOREST')
            physical=float(np.longdouble(tau.numerator)*np.longdouble(self.delta_t)/np.longdouble(tau.denominator))
            query={'mode':mode,'value':str(tau),'exact_time':{'numerator':tau.numerator,'denominator':tau.denominator},'physical_time_hex':physical.hex()}
        else:
            physical=float(value);require(math.isfinite(physical) and 0<=physical<=self.parameters['duration'],'PHYSICAL_TIME_OUTSIDE_FOREST')
            query={'mode':mode,'value':physical.hex(),'exact_time':None,'physical_time_hex':physical.hex(),
                   'effective_discrete_time_hex':float(physical/self.delta_t).hex()}
        _check_rss();started=time.monotonic();cpu=time.process_time()
        vc=np.zeros(N_ELEMENTS,np.int32);fc=np.zeros(N_ELEMENTS,np.int32)
        with _LOCK:
            self.dll.slicing_identity_enable(bool(ledger))
            try:
                if mode=='exact':status=int(self.dll.run_slicing_rational(tau.numerator,tau.denominator,_int_pointer(vc),_int_pointer(fc),False))
                else:status=int(self.dll.run_slicing(physical,_int_pointer(vc),_int_pointer(fc),False))
                require(status==0,'ORDINARY_NATIVE_SLICING_FAILED: '+_error(self.dll.slicing_last_error))
                _check_rss();validate_counts(vc,fc)
                identity=int(self.dll.slicing_identity_status());identity_error=_error(self.dll.slicing_identity_last_error)
                require(identity in (0,1,2,3),'INVALID_IDENTITY_STATUS')
                ids=[np.empty((0,4),np.int32) for _ in range(N_ELEMENTS)]
                owners=[np.empty((0,11),np.int32) for _ in range(N_ELEMENTS)]
                owner_counts=None
                source_vid_shifts=None
                if ledger and identity==1:
                    ledger_v=[int(self.dll.slicing_identity_vertex_count(e)) for e in range(N_ELEMENTS)]
                    owner_counts=[int(self.dll.slicing_identity_owner_count(e)) for e in range(N_ELEMENTS)]
                    require(ledger_v==[int(n) for n in vc],'IDENTITY_VERTEX_COUNT_NOT_ACTUAL_VERTEX_COUNT')
                    array_bytes=validate_counts(vc,fc,owner_counts)
                    # All sidecars are extracted BEFORE ANY slicing_output destroys mesh state.
                    for e in range(N_ELEMENTS):
                        ids[e]=np.empty((ledger_v[e],4),np.int32)
                        owners[e]=np.empty((owner_counts[e],11),np.int32)
                        require(self.dll.slicing_identity_output_vertices(e,_int_pointer(ids[e]),ids[e].size)==0,'IDENTITY_VERTEX_OUTPUT_FAILED')
                        require(self.dll.slicing_identity_output_owners(e,_int_pointer(owners[e]),owners[e].size)==0,'IDENTITY_OWNER_OUTPUT_FAILED')
                    if self.source_vid_encoding=='ORIGINAL_EFFECTIVE_SOURCE_VID':
                        source_vid_shifts=np.empty((N_ELEMENTS,4),np.int32)
                        require(self.dll.slicing_identity_output_source_vid_shifts(_int_pointer(source_vid_shifts),source_vid_shifts.size)==0,
                                'ORIGINAL_SOURCE_VID_SHIFT_OUTPUT_FAILED')
                        source_vid_shifts.setflags(write=False);array_bytes+=source_vid_shifts.nbytes
                        require(array_bytes<=MAX_QUERY_ARRAY_BYTES,'NATIVE_QUERY_ARRAYS_EXCEED_2_GIB')
                else:array_bytes=validate_counts(vc,fc)
                meshes=[]
                for e in range(N_ELEMENTS):
                    v=np.empty((int(vc[e]),3),np.float64);f=np.empty((int(fc[e]),3),np.int32);tags=np.empty(int(vc[e]),np.int32)
                    self.dll.slicing_output(e,_double_pointer(v),_int_pointer(f),_int_pointer(tags))
                    require(np.all(np.isfinite(v)),'NONFINITE_NATIVE_VERTEX')
                    require(not len(f) or int(f.min())>=0 and int(f.max())<len(v),'NATIVE_FACE_INDEX_OUT_OF_RANGE')
                    require(np.all((tags==0)|(tags==1)),'NATIVE_INVIEW_TAG_NOT_BOOLEAN')
                    for start in range(0,len(v),65536):
                        block=v[start:start+65536]
                        require(np.array_equal(block.astype(np.float32).astype(np.float64),block),'NATIVE_COORDINATES_NOT_EXACT_BINARY32_VALUES')
                    if ledger and identity==1:
                        o=owners[e]
                        require(np.all(o[:,0]==e),'IDENTITY_OWNER_ELEMENT_DISAGREEMENT')
                        require(not len(o) or np.all(o[:,:8]>=0) and int(o[:,7].max())<len(f),'IDENTITY_OWNER_PROVENANCE_OR_FACE_RANGE')
                        for start in range(0,len(o),65536):
                            block=o[start:start+65536]
                            require(np.array_equal(block[:,8:11],f[block[:,7]]),'IDENTITY_OWNER_ORIENTED_FACE_DISAGREEMENT')
                    for a in (v,f,tags,ids[e],owners[e]):a.setflags(write=False)
                    meshes.append((v,f,tags))
                _check_rss()
                return {'meshes':tuple(meshes),'vertex_ledgers':tuple(ids),'owner_ledgers':tuple(owners),
                    'identity_status':identity,'identity_error':identity_error,'ledger_requested':bool(ledger),
                    'source_vid_encoding_version':self.source_vid_encoding_version,
                    'source_vid_encoding':self.source_vid_encoding,'source_vid_shifts':source_vid_shifts,
                    'counts':[{'element':e,'vertices':int(vc[e]),'faces':int(fc[e]),'identity_vertices':len(ids[e]),'raw_owners':len(owners[e])} for e in range(N_ELEMENTS)],
                    'cost':{'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
                            'array_bytes':array_bytes,'peak_rss_bytes':_peak_rss()},
                    'query':query,'delta_t':self.delta_t,'delta_t_hex':self.delta_t.hex(),
                    'n_elements':N_ELEMENTS,'maximum_discrete_time':4,'cache_root':str(self.cache),
                    'library_sha256':self.library_sha256,'input_documents_sha256':self.input_documents_sha256,
                    'baseline_only':True,'extra_smooth':False,'arrays_immutable':True}
            finally:self._discard()

    def close(self):
        global _ACTIVE
        if self._closed:return
        require(threading.get_ident()==self._thread,'CLOSE_NATIVE_READER_ON_CREATING_THREAD')
        try:self._discard()
        finally:
            self._closed=True
            for key,old in self._saved_environment.items():
                if old is None:os.environ.pop(key,None)
                else:os.environ[key]=old
            if _ACTIVE is self:_ACTIVE=None

    cleanup=close


def initialize_forest(build_repo,cache_copy,camera_document,effective_document,*,expected_so_sha256=FROZEN_SO_SHA256):
    global _ACTIVE
    with _LOCK:
        require(_ACTIVE is None,'ONE_NATIVE_FOREST_READER_PER_PROCESS')
        cache=Path(cache_copy).resolve();original=ORIGINAL_CACHE.resolve()
        require(cache!=original and original not in cache.parents,'REFUSE_ORIGINAL_FOREST_CACHE')
        require(cache.is_dir() and (cache/'slicing_preprocess.finish').is_file(),'PRIVATE_COMPLETED_CACHE_REQUIRED')
        require(not any(p.is_symlink() for p in cache.rglob('*')),'PRIVATE_CACHE_SYMLINK_FORBIDDEN')
        if original.is_dir():
            for name in ('hypervertices/0.bin','processed_hyperpolys/0_0.bin'):
                require(not os.path.samefile(cache/name,original/name),'PRIVATE_CACHE_INPUT_IS_ORIGINAL_HARDLINK')
        manifest=json.loads((cache/'slicing_preprocess.manifest.json').read_text())
        require(manifest.get('provenance_enabled') is True and manifest.get('bpm_version')==2 and manifest.get('bhp_version')==2,'BPM2_COMPLETED_CACHE_REQUIRED')
        require(not any(os.environ.get(k) for k in os.environ if k.startswith('BINOC_SOURCE_SPLICE')),'NO_SSP1_ENVIRONMENT_ALLOWED')
        camera,camera_binding=_document(camera_document);effective,effective_binding=_document(effective_document)
        parameters=prepare_parameters(camera,effective)
        library=(Path(build_repo).resolve()/'binocmesher/lib/core.so')
        require(len(expected_so_sha256)==64 and _hash(library)==expected_so_sha256,'NATIVE_LIBRARY_SHA256_MISMATCH')
        dll=ctypes.CDLL(str(library));_bind(dll)
        result=NativeForestReader(dll,parameters,cache,library,expected_so_sha256,
            {'camera':camera_binding,'effective':effective_binding})
        for key in ('BINOC_PROVENANCE_V2','BINOC_EVENT_MODE'):
            result._saved_environment[key]=os.environ.get(key);os.environ[key]='1'
        _ACTIVE=result
        try:
            result._discard();_check_rss()
            groups=int(dll.load_parameters(_double_pointer(parameters['center']),parameters['size'],parameters['duration'],
                parameters['n_cameras'],_double_pointer(parameters['cameras']),parameters['fading_time'],
                parameters['pixels_per_cube'],parameters['pixels_per_cube_coarse'],parameters['pixels_per_cube_outview'],
                parameters['min_dist'],N_ELEMENTS,str(cache).encode('utf-8')))
            actual_delta=ctypes.c_double.in_dll(dll,'_ZN6params6deltaTE').value
            actual_elements=ctypes.c_int32.in_dll(dll,'_ZN6params10n_elementsE').value
            actual_level=ctypes.c_int32.in_dll(dll,'_ZN6params6max_tLE').value
            require(groups==parameters['expected_groups']==1<<actual_level,'ACTUAL_NATIVE_TEMPORAL_GROUP_MISMATCH')
            require(actual_elements==N_ELEMENTS,'ACTUAL_NATIVE_ELEMENT_COUNT_MISMATCH')
            require(actual_delta.hex()==parameters['expected_delta'].hex(),'ACTUAL_NATIVE_DELTA_BITS_MISMATCH')
            require(_hash(library)==expected_so_sha256,'NATIVE_LIBRARY_CHANGED_DURING_INITIALIZATION')
            result.delta_t=actual_delta
            result.initialization={'actual_delta_t_hex':actual_delta.hex(),'actual_max_tL':actual_level,
                'source_vid_encoding_version':result.source_vid_encoding_version,
                'source_vid_encoding':result.source_vid_encoding,
                'actual_elements':actual_elements,'time_groups':groups,'library_sha256':expected_so_sha256,
                'duration_hex':parameters['duration'].hex(),'origin_seconds_hex':parameters['origin_seconds'].hex(),
                'native_entrypoints_called':['slicing_discard_output','slicing_clean_up','slicing_identity_enable','load_parameters'],
                'tree_or_preprocess_or_initial_slice_run':False,'parameter_sources':parameters['parameter_sources']}
            return result
        except BaseException:
            result.close();raise
