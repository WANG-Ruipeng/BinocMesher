"""Executable common checkpoint-to-output path. Timers default off."""
import argparse,contextlib,ctypes as C,json,shutil,time,traceback
from pathlib import Path
import numpy as np
from common_runtime import *

class Ledger:
    def __init__(self,enabled=False):self.enabled=enabled;self.rows=[];self.stack=[]
    @contextlib.contextmanager
    def stage(self,name):
        if not self.enabled:yield;return
        parent=self.stack[-1] if self.stack else None
        self.stack.append(name);start=time.perf_counter_ns()
        try:yield
        finally:
            elapsed=time.perf_counter_ns()-start;self.stack.pop()
            self.rows.append({'stage':name,'parent':parent,'inclusive':True,'wall_ns':elapsed})

def bind_common(native):
    bind=lambda name,args:API._bind(native.lib,name,args)
    bind('bm_common_plan_batch',[])
    bind('bm_common_field_sizes',[C.c_int,C.POINTER(C.c_int64)])
    bind('bm_common_copy_field',[C.c_int,C.c_int64,C.c_int64,PI,PD,PD,C.POINTER(C.c_int64),PI])
    bind('bm_common_commit',[C.c_int64,PF,PI,PI,PD,PD])

def current_plan(native,K,batch):
    native.check(native.lib.bm_common_plan_batch(),'plan_batch')
    plans=[]
    for e,name in enumerate(('LandTiles','SdfTrees')):
        sizes=(C.c_int64*4)();native.check(native.lib.bm_common_field_sizes(e,sizes),'field_sizes')
        n,m,first,query=map(int,sizes)
        offsets=np.empty(n+1,np.int32);centers=np.empty((n,3),np.float64);ends=np.empty((m,3),np.float64)
        heads=np.empty(n,np.int64);meta=np.empty((n,6),np.int32)
        native.check(native.lib.bm_common_copy_field(e,n,m,pointer(offsets,PI),pointer(centers,PD),pointer(ends,PD),pointer(heads,C.POINTER(C.c_int64)),pointer(meta,PI)),'copy_field')
        case=Case(f'g00000_b{batch:05d}_{name}',name,3+e,K,offsets,centers,ends,{}, {})
        plans.append({'case':case,'heads':heads,'metadata':meta,'first':first,'query':query})
    return plans

def compare_plan(plans,capture,batch,output_dir=None):
    source=capture/f'g00000_b{batch:05d}_input.jsonl'
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    for e,p in enumerate(plans):
        if output_dir is not None:
            c=p['case'];np.savez(output_dir/f'plan_{batch:02d}_{e}.npz',offsets=c.offsets,centers=c.centers,endpoints=c.endpoints,heads=p['heads'],metadata=p['metadata'])
        nodes=sorted([r for r in rows if r.get('type')=='node' and r['element']==e],key=lambda r:r['center_index'])
        case=p['case'];centers=np.asarray([r['center'] for r in nodes],np.float64).reshape(-1,3)
        ends=np.asarray([ep['xyz'] for r in nodes for ep in r['endpoints']],np.float64).reshape(-1,3)
        offsets=np.asarray([0]+list(np.cumsum([len(r['endpoints']) for r in nodes])),np.int32)
        require(same(case.centers,centers) and same(case.endpoints,ends) and same(case.offsets,offsets),'live original graph preparation differs from captured inputs: '+case.name)
        require(p['heads'].tolist()==[r['key'][1] for r in nodes],'live node order differs')

def compare_output(actual,capture,batch):
    path=capture/f'g00000_b{batch:05d}_native_outputs.npz'
    with np.load(path,allow_pickle=False) as target:
        for key in actual:require(same(actual[key],target[key]),f'full native output mismatch: batch {batch} {key}')

def solver_info(api,handle,K):
    API._bind(api.solver,'bm_solver_get_info',[C.c_void_p,C.c_int,C.POINTER(C.c_uint64),C.c_size_t])
    info=(C.c_uint64*40)();api.check(api.solver.bm_solver_get_info(handle,K,info,40),'solver_info')
    return list(map(int,info))

def debug_check(api,handle,info):
    if not info[1]:return None
    API._bind(api.solver,'bm_solver_debug_check',[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t])
    result=(C.c_uint64*8)();rc=api.solver.bm_solver_debug_check(handle,result,8)
    values=list(map(int,result));require(rc==0 and not any(values[2:]),'device guards/write counts failed: '+str(values))
    return values

def _run_common(mode,K,checkpoint,*,validate=False,output_dir=None,timing=False,solver_path=None):
    require(not (validate and timing),'Correctness tracing must not enter performance trials')
    if timing:performance_allowed(True)
    require(mode in ('C0','R0','R1','L0','L1_8','L1_32'),'invalid mode')
    ledger=Ledger(timing);record={'mode':mode,'K':K,'performance_test':timing,'GPU_executed':True,'batches':[],'solver_infos':[],'debug_checks':[]}
    capture=REFERENCE_ROOT/'runs'/('g0_observed_01' if K==3 else 'g0_observed_K6_01')/'capture'
    field_handles=[];fields=None;result=None
    with ledger.stage('checkpoint_to_output'):
        with ledger.stage('entry_binding_and_original_loop_prepare'):
            setup_imports();native=NativeAPI();bind_common(native)
            loop,loop_provenance=original_group_loop();record['original_python_loop']=loop_provenance
            api=None if mode=='C0' else API(ROOT/'build/libfield_bridge.so',solver_path or ROOT/'build/libgpu_solver.so')
        with ledger.stage('input_and_instance_prepare'):
            mesher=new_mesher(checkpoint,K)
            params=load_arrays(ROOT/'inputs/accepted_scene/field_parameters.npz')
        with ledger.stage('field_initialization_upload'):
            if mode=='C0':fields=NativeFields(parameters=params)
            else:
                for kind,p in zip((3,4),field_tuples(params)):field_handles.append(api.create_field(kind,p))
        with ledger.stage('reset_checkpoint_read_native_graph_load'):
            init_checkpoint(native,mesher)
        batch=0
        if mode=='C0':
            original_next=mesher.bisection_hypermesh_verts;original_finish=mesher.bisection_hypermesh_verts_finishing
            if validate:
                def checked_next(t,c,n):
                    nonlocal batch
                    rc=original_next(t,c,n)
                    if rc:
                        plans=current_plan(native,K,batch);compare_plan(plans,capture,batch,output_dir)
                        record['batches'].append({'batch':batch,'N':sum(len(p['case'].centers) for p in plans),'M':sum(len(p['case'].endpoints) for p in plans)})
                    return rc
                def checked_finish(t,s,c):
                    nonlocal batch
                    original_finish(t,s,c);out=native.output()
                    if output_dir is not None:np.savez(output_dir/f'batch_{batch:02d}.npz',**out)
                    compare_output(out,capture,batch)
                    batch+=1
                mesher.bisection_hypermesh_verts=checked_next;mesher.bisection_hypermesh_verts_finishing=checked_finish
            with ledger.stage('original_python_bisection_loop'):
                loop(mesher,fields.kernels,2,0)
        else:
            counts=np.zeros(2,np.int32);nodes=np.zeros(2,np.int32);mode_id=('R0','R1','L0','L1_8','L1_32').index(mode)
            while True:
                with ledger.stage('native_batch_boundary_and_ordered_list_prepare'):
                    active=mesher.bisection_hypermesh_verts(0,pointer(counts,PI),pointer(nodes,PI))
                    if not active:break
                    plans=current_plan(native,K,batch)
                    if validate:compare_plan(plans,capture,batch,output_dir)
                    n=int(nodes.sum());full={'position':np.empty((n,3),np.float32),'valid':np.empty(n,np.int32),'witness':np.empty(n,np.int32),'left':np.empty(n,np.float64),'right':np.empty(n,np.float64)}
                for element,p in enumerate(plans):
                    case=p['case']
                    if not len(case.centers):continue
                    with ledger.stage('candidate_allocate_owner_map_upload'):
                        handle=api.create_solver(case,field_handles[element][1])
                    if mode=='R1':
                        with ledger.stage('graph_capture_instantiate'):
                            api.check(api.solver.bm_solver_prepare_graph(handle,K,0),'prepare_graph')
                    with ledger.stage('gpu_reset_and_solve'):
                        api.check(api.solver.bm_solver_run(handle,mode_id,K,0),'solver_run')
                    with ledger.stage('readback_and_output_restore'):
                        count=len(case.centers)
                        outputs={'position':np.empty((count,3),np.float32),'valid':np.empty(count,np.int32),'witness':np.empty(count,np.int32),'left':np.empty(count,np.float64),'right':np.empty(count,np.float64)}
                        aux=np.empty((count+(K+1)*len(case.endpoints),3),np.float32) if validate else None
                        api.check(api.solver.bm_solver_readback(handle,pointer(outputs['position'],PF),pointer(outputs['witness'],PI),pointer(outputs['valid'],PI),pointer(outputs['left'],PD),pointer(outputs['right'],PD),pointer(aux,PF) if aux is not None else PF(),PF(),PF(),PI(),PD(),PD()),'readback')
                        sl=slice(p['first'],p['first']+count)
                        for key in full:full[key][sl]=outputs[key]
                    if validate:
                        if output_dir is not None:np.savez(output_dir/f'field_{batch:02d}_{element}.npz',**outputs,aux=aux)
                        oracle_dir=REFERENCE_ROOT/'results/correctness'/('gpu_natural_01' if K==3 else 'gpu_natural_K6_01')/case.name
                        with np.load(oracle_dir/'oracle.npz',allow_pickle=False) as oracle:
                            for key in ('position','valid','witness','left','right'):
                                require(same(outputs[key],oracle[key]),'common solver oracle mismatch: '+case.name+' '+key)
                            require(same(aux,oracle['aux']),'common aux oracle mismatch: '+case.name)
                        require(np.all(outputs['valid']==1),'GPU output invalid')
                        info=solver_info(api,handle,K);require(info[32]==0,'timing events executed during correctness')
                        record['solver_infos'].append({'case':case.name,'values':info});record['debug_checks'].append(debug_check(api,handle,info))
                    with ledger.stage('solver_cleanup'):
                        api.check(api.solver.bm_solver_destroy(handle),'solver_destroy')
                with ledger.stage('native_output_assembly'):
                    native.check(native.lib.bm_common_commit(n,pointer(full['position'],PF),pointer(full['valid'],PI),pointer(full['witness'],PI),pointer(full['left'],PD),pointer(full['right'],PD)),'candidate_commit')
                if validate:
                    out=native.output()
                    if output_dir is not None:np.savez(output_dir/f'batch_{batch:02d}.npz',**out)
                    compare_output(out,capture,batch)
                    record['batches'].append({'batch':batch,'N':int(nodes.sum()),'M':int(counts.sum())})
                batch+=1
        with ledger.stage('complete_native_format_host_output'):
            result=native.output()
        with ledger.stage('lifecycle_cleanup'):
            if fields is not None:fields.close()
            for handle,_ in reversed(field_handles):api.check(api.field.bm_field_destroy(handle),'field_destroy')
            native.check(native.lib.bm_common_reset(),'final_native_reset')
    if validate:
        require(len(record['batches'])==5,'unexpected natural batch count')
        require([x['N'] for x in record['batches']]==[47,44,43,56,2] and [x['M'] for x in record['batches']]==[265,259,265,257,7],'batch counts changed')
        record['status']='PASS_FULL_COMMON_OUTPUT'
    else:record['status']='EXECUTED'
    record['stages']=ledger.rows;record['outputs']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in result.items()}
    return result,record

def run_common(mode,K,checkpoint,**kwargs):
    # On failure callers preserve the first error and exit this worker process;
    # no additional CUDA call is issued to obscure a failing CUDA context.
    # OS process teardown releases partially constructed native/device resources.
    with python_timer_policy(kwargs.get('timing',False)):
        return _run_common(mode,K,checkpoint,**kwargs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--K',type=int,choices=(3,6),required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--debug',action='store_true');p.add_argument('--mode',choices=('all','C0','R0','R1','L0','L1_8','L1_32'),default='all');a=p.parse_args()
    out=a.out.resolve();require(out.is_relative_to(ROOT) and not out.exists(),'new output path required');out.mkdir(parents=True)
    report={'status':'RUNNING','performance_test':False,'timing_events_recorded':0,'rows':[],'purposes':['first_complete_solve','full_checkpoint_restore_repeat','final_complete_solve']}
    def save():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for mode in (('C0','R0','R1','L0','L1_8','L1_32') if a.mode=='all' else (a.mode,)):
            previous=None
            for repeat in range(3):
                dest=out/f'{mode}_{repeat}';dest.mkdir();cp=dest/'checkpoint';shutil.copytree(ROOT/'inputs/checkpoint_seed',cp)
                actual,row=run_common(mode,a.K,cp,validate=True,output_dir=dest,solver_path=ROOT/'build'/('libgpu_solver_debug.so' if a.debug else 'libgpu_solver.so'))
                if previous is not None:
                    for key in actual:require(same(previous[key],actual[key]),'repeated complete output mismatch: '+key)
                previous=actual;row['repeat']=repeat;report['rows'].append(row);save()
                print(json.dumps({'mode':mode,'K':a.K,'repeat':repeat,'status':row['status'],'batches':len(row['batches'])}),flush=True)
        report['status']='PASS_ON_EXECUTED_COMMON_ENTRY_CASES'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());raise
    finally:save()
if __name__=='__main__':main()