"""Frozen original field/Python call path and shared input loading; no timers."""
import ast, ctypes as C, hashlib, importlib.machinery, json, os, sys, textwrap, types
from pathlib import Path
import numpy as np
from check_gpu_solver import API, Case, PI, PF, PD, pointer, require, same
from workspace_paths import ROOT
from workspace_paths import INFINIGEN_ROOT, REFERENCE_ROOT
MAIN=ROOT/'upstream/main'
INF=INFINIGEN_ROOT
CHECKPOINT=ROOT/'inputs/checkpoint_seed'
P8=C.POINTER(C.c_int8)
_IMPORTED=None

def setup_imports():
    global _IMPORTED
    if _IMPORTED is not None:return _IMPORTED
    sys.dont_write_bytecode=True
    os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
    sys.path.insert(0,str(INF))
    import infinigen
    name='infinigen.BinocMesher'
    if name not in sys.modules:
        p=types.ModuleType(name);p.__package__=name;p.__path__=[str(MAIN)]
        p.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
        p.__spec__.submodule_search_locations=p.__path__
        sys.modules[name]=p;infinigen.BinocMesher=p
    from infinigen.BinocMesher.binocmesher import BinocMesher
    from infinigen.terrain.elements.core import Element
    from infinigen.terrain.utils import Vars
    from infinigen.core.util.organization import Materials,Transparency
    require('binocmesher' not in sys.modules,'dual BinocMesher namespace imported')
    _IMPORTED=BinocMesher,Element,Vars,Materials,Transparency
    return _IMPORTED

def load_arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def field_tuples(parameters):
    p=parameters
    def arr(key,dtype):
        a=np.ascontiguousarray(p[key]);require(a.dtype==dtype,'frozen parameter dtype changed: '+key);return a
    li,lf=arr('land_int_params',np.int32),arr('land_float_params',np.float32)
    ti,tf=arr('trees_int_params',np.int32),arr('trees_float_params',np.float32)
    require(np.all(p['land_meta_params']==0),'unsupported caves closure')
    return [(0,li,lf,np.empty(0,np.int32),np.empty(0,np.float32)),(0,ti,tf,li,lf)]

class NativeFields:
    """Original Element.__call__, original init/call/cleanup; no asset regeneration."""
    active=False
    def __init__(self,parameters_path=None,parameters=None):
        require(not NativeFields.active,'only one original field set may be live')
        _,Element,Vars,Materials,Transparency=setup_imports()
        self.parameters=parameters if parameters is not None else load_arrays(parameters_path or ROOT/'inputs/accepted_scene/field_parameters.npz')
        self.elements=[];self.libraries=[];self.kernels=[];self.closed=False
        tuples=field_tuples(self.parameters)
        for index,(name,material,aux_names) in enumerate((('landtiles',Materials.MountainCollection,[None,None,None]),('sdf_trees',Materials.TreeCollection,['tree_id']))):
            obj=Element.__new__(Element)
            obj.device='cuda';obj.material=material;obj.transparency=Transparency.Opaque
            obj.aux_names=aux_names;obj.attributes=[material]
            obj.displacement=[];obj.height_offset=0;obj.whole_bbox=None
            meta,ip,fp,ip2,fp2=tuples[index]
            obj.int_params=ip;obj.float_params=fp;obj.int_params2=ip2;obj.float_params2=fp2
            obj.int_params3=np.empty(0,np.int32);obj.float_params3=np.empty(0,np.float32);obj.meta_params=[meta]
            lib=C.CDLL(str(INF/f'infinigen/terrain/lib/cuda/elements/{name}.so'))
            lib.init.argtypes=[C.c_int,C.c_int]+[C.c_size_t,PI,C.c_size_t,PF]*3;lib.init.restype=None
            lib.call.argtypes=[C.c_size_t,PF,PF,PF];lib.call.restype=None
            lib.cleanup.argtypes=[];lib.cleanup.restype=None
            obj.call=lib.call;obj.init=lib.init;obj.cleanup=lib.cleanup
            obj.init(meta,0,ip.size,pointer(ip,PI),fp.size,pointer(fp,PF),ip2.size,pointer(ip2,PI),fp2.size,pointer(fp2,PF),0,PI(),0,PF())
            self.libraries.append(lib);self.elements.append(obj)
            self.kernels.append(lambda xyz,element=obj:element(xyz)[Vars.SDF])
        NativeFields.active=True
    def close(self):
        if not self.closed:
            for obj in reversed(self.elements):obj.cleanup()
            self.closed=True;NativeFields.active=False
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def new_mesher(path,K,input_arrays=None,config=None):
    BinocMesher,*_=setup_imports()
    data=input_arrays if input_arrays is not None else load_arrays(ROOT/'inputs/accepted_scene/inputs.npz')
    cfg=config if config is not None else json.loads((ROOT/'configs/scene.json').read_text())
    opts=dict(cfg['mesher']);opts['bisection_iters']=K
    require(not opts['enclosed'] and not opts['use_alignment'],'unsupported native outer closure')
    mesher=BinocMesher(tuple(data[k] for k in ('cam_poses','Ks','Hs','Ws','Ts')),bounds=data['bounds'].tolist(),slicing_time=float(data['Ts'][0]),path=Path(path),**opts)
    return mesher

def original_group_loop():
    """Compile the original contiguous while-loop AST without changing its body."""
    setup_imports()
    import infinigen.BinocMesher.binocmesher.core as core
    source=Path(core.__file__).read_text();tree=ast.parse(source)
    candidates=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.While) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='bisection_hypermesh_verts' for n in ast.walk(node)):
            candidates.append(node)
    require(len(candidates)==1,'original bisection while-loop is not unique')
    node=candidates[0];fragment=ast.get_source_segment(source,node)
    body='def run(self,kernels,n_elements,t):\n    path=self.path\n    cnts=np.zeros(n_elements,dtype=np.int32)\n    center_cnts=np.zeros(n_elements,dtype=np.int32)\n'+textwrap.indent(textwrap.dedent(fragment),'    ')+'\n'
    env=dict(core.__dict__);exec(compile(body,str(ROOT/'src/ORIGINAL_GROUP_LOOP.generated.py'),'exec'),env)
    return env['run'],{'source':str(Path(core.__file__)),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'while_lines':[node.lineno,node.end_lineno],'while_sha256':hashlib.sha256(fragment.encode()).hexdigest(),'while_ast_unchanged':True}

class NativeAPI:
    def __init__(self):
        self.lib=C.CDLL(str(MAIN/'binocmesher/lib/core.so'))
        def bind(name,args,ret=C.c_int):API._bind(self.lib,name,args,ret)
        bind('bm_common_last_error',[],C.c_char_p)
        bind('bm_common_reset',[])
        bind('bm_checkpoint_read',[C.c_char_p])
        bind('bm_output_sizes',[C.POINTER(C.c_int64)])
        bind('bm_copy_outputs',[PF,PI,P8,PI])
    def check(self,rc,label):
        if rc:
            message=self.lib.bm_common_last_error()
            raise RuntimeError(label+': '+str(rc)+' '+(message.decode() if message else ''))
    def output(self):
        sizes=(C.c_int64*2)();self.check(self.lib.bm_output_sizes(sizes),'output_sizes')
        nv,nm=map(int,sizes)
        out={'xyz':np.empty((nv,3),np.float32),'times':np.empty((nv,2),np.int32),'tags':np.empty(nv,np.int8),'vertex_map':np.empty(nm,np.int32)}
        self.check(self.lib.bm_copy_outputs(pointer(out['xyz'],PF),pointer(out['times'],PI),pointer(out['tags'],P8),pointer(out['vertex_map'],PI)),'copy_outputs')
        return out

def init_checkpoint(native,mesher):
    native.check(native.lib.bm_common_reset(),'reset_before_checkpoint')
    count=mesher.load_parameters(mesher.AF(mesher.center),mesher.size,mesher.tsize,mesher.n_cameras,mesher.AF(mesher.cameras),mesher.fading_time,mesher.pixels_per_cube,mesher.pixels_per_cube_coarse,mesher.pixels_per_cube_outview,mesher.min_dist,2,str(mesher.path).encode())
    require(count==1,'only captured single-group closure supported')
    native.check(native.lib.bm_checkpoint_read(str(mesher.path/'native_state.bin').encode()),'checkpoint_read')
    mesher.load_tree_size();mesher.bisection_init(mesher.bisection_group);mesher.bisection_init_t(0)

def performance_allowed(explicit):
    gate=json.loads((ROOT/'configs/PERFORMANCE_AUTHORIZATION.json').read_text())
    require(explicit and gate.get('allowed') is True,'Performance not authorized: no warmups/calibration/timing permitted')
class NoTimer:
    def __init__(self,*args,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):return False

@__import__('contextlib').contextmanager
def python_timer_policy(enabled):
    setup_imports()
    import infinigen.BinocMesher.binocmesher.core as core
    original=core.Timer
    if not enabled:core.Timer=NoTimer
    try:yield
    finally:core.Timer=original