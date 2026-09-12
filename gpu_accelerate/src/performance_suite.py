"""Dormant finite suite. Default --prepare only; no application imports or clocks.
Actual execution needs --execute-performance, user gate and ready VERDICT.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, signal, statistics
import subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from workspace_paths import ROOT
MATRIX=ROOT/'configs/FROZEN_MATRIX.json'
AUTH=ROOT/'configs/PERFORMANCE_AUTHORIZATION.json'
VERDICT=ROOT/'VERDICT.json'
PARAMETERS=ROOT/'inputs/accepted_scene/field_parameters.npz'
MODES=('R0','R1','L0','L1_8','L1_32')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def append_json(path,row):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
def field_key(field,k):return f'{field}:K{k}'
def case_key(row):return f"K{row['K']}:{row['id']}"

def matrix_check(m):
    require(m['processes']==3 and m['warmups']==10 and m['pairs']==20,'Frozen timing protocol changed')
    require(m['pair_orders']=={'AB':10,'BA':10} and m['K']==[3,6],'Frozen K/pair balance changed')
    require(m['modes']==['C0',*MODES] and m['no_held_out_selection'] is True,'Mode/selection policy changed')
    for k in m['K']:
        for field in ('LandTiles','SdfTrees'):
            rows=[r for r in m['cases'] if r['K']==k and r['field']==field and r['role']=='calibration']
            require(len(rows)==1 and rows[0]['batch']==0,'Only first native batch calibrates')
    for r in m['cases']:
        require(r['N']>0 and r['N']==len(r['m_counts']) and r['M']==sum(r['m_counts']) and
                r['Q']==r['N']+(r['K']+1)*r['M'],'Frozen N/M/Q differs')
    require(m['repeat_calibration']['target_event_ms']==10 and
            m['repeat_calibration']['max_repeats']==4096,'Repeat bounds changed')

def held_out(m):
    result=[]
    for k in m['K']:
        for field in ('LandTiles','SdfTrees'):
            rows=[r for r in m['cases'] if r['K']==k and r['field']==field and r['role']=='held_out']
            sizes=sorted({r['N'] for r in rows})
            if sizes:
                wanted={sizes[0],sizes[len(sizes)//2],sizes[-1]}
                result.extend(r for r in rows if r['N'] in wanted)
    return result

def plan():
    m=read_json(MATRIX);matrix_check(m)
    paths=[ROOT/'src/workspace_paths.py',MATRIX,PARAMETERS,ROOT/'inputs/accepted_scene/inputs.npz',ROOT/'configs/scene.json',
           ROOT/'src/common_runner.py',ROOT/'src/common_runtime.py',ROOT/'src/gpu_solver.cu',
           ROOT/'src/performance_suite.py',ROOT/'src/check_gpu_solver.py',ROOT/'build/libgpu_solver.so',
           ROOT/'build/libfield_bridge.so',ROOT/'upstream/main/binocmesher/lib/core.so']
    seed=ROOT/'inputs/checkpoint_seed'
    return {'status':'PREPARED_NOT_RUN','performance':'NOT_RUN','GPU':'NOT_RUN',
      'matrix_sha256':digest(MATRIX),
      'fingerprints':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p) if p.is_file() else None} for p in paths],
      'checkpoint_files':[{'path':str(p.relative_to(seed)),'sha256':digest(p)}
                          for p in sorted(seed.rglob('*')) if p.is_file()] if seed.is_dir() else [],
      'calibration_cases':[case_key(r) for r in m['cases'] if r['role']=='calibration'],
      'resident_formal_cases':[case_key(r) for r in held_out(m)],
      'workers':[0,1,2],'worker_order':'sequential','calibration_worker':0,
      'resident_comparisons':'R1 versus each other fixed mode; selected resident/local pair also retained',
      'resident_C0':'N/A','common_comparisons':[f'C0 vs {mode}' for mode in MODES],
      'common_timed_arms_per_worker':len(m['K'])*len(MODES)*m['pairs']*2,
      'common_warmup_arms_per_worker':len(m['K'])*(1+len(MODES))*m['warmups'],
      'common_repeats':1,'single_use_graph_reuse':1,'fresh_checkpoint_per_common_arm':True,
      'checkpoint_provisioning':'identical private copy excluded; actual file reads included in run_common',
      'analytic_plane_diagnostic':{'status':'NOT_RUN','cases':['diagnostic_kind0_K3','diagnostic_kind0_K6'],
        'provider':'unmodified check_gpu_solver.diagnostic_cases','N':33,'M':608,
        'scope':'legal synthetic interface fixture, not a real graph','resident':'R1','local':'L1_8',
        'selection':'predeclared only; no held-out generalization','timed_arms_per_worker':320},
      'hardware_profiling':'NOT_RUN','automatic_retries':0,
      'gates':['--execute-performance','PERFORMANCE_AUTHORIZATION.allowed=true','VERDICT.ready_for_performance=true'],
      'statistics':'per process/case paired ratios, median/p90; internal repeats are not independent inputs'}

def gates(explicit,verdict=VERDICT):
    require(explicit,'Performance switch absent')
    require(read_json(AUTH).get('allowed') is True,'User performance authorization remains false')
    require(read_json(verdict).get('ready_for_performance') is True,'Preperformance verdict not ready')
    from common_runtime import performance_allowed
    performance_allowed(explicit)

def orders(m,worker,scope,comparison):
    seed=int.from_bytes(hashlib.sha256(f"{m['seed']}|{worker}|{scope}|{comparison}".encode()).digest()[:8],'little')
    result=['AB']*10+['BA']*10;random.Random(seed).shuffle(result);return result

def percentile(values,p):
    v=sorted(values);require(bool(v),'No samples');i=(len(v)-1)*p
    a,b=math.floor(i),math.ceil(i);return v[a]+(v[b]-v[a])*(i-a)

def pair_summary(rows):
    pairs={}
    for r in rows:pairs.setdefault(r['pair'],{})[r['arm']]=r
    require(len(pairs)==20 and all(set(p)=={'A','B'} for p in pairs.values()),'Incomplete pair set')
    a=[p['A']['per_solve_ms'] for p in pairs.values()];b=[p['B']['per_solve_ms'] for p in pairs.values()]
    ratios=[x/y for x,y in zip(a,b)]
    return {'pairs':20,'median_A_ms':statistics.median(a),'p90_A_ms':percentile(a,.9),
      'median_B_ms':statistics.median(b),'p90_B_ms':percentile(b,.9),
      'paired_ratios_A_over_B':ratios,'median_paired_ratio':statistics.median(ratios),
      'p90_paired_ratio':percentile(ratios,.9),'interpretation':'A/B>1 means B faster; retain every slow/negative trial'}

def summarize(raw):
    groups={}
    for r in raw:groups.setdefault((r['table'],r['case'],r['K'],r['comparison']),[]).append(r)
    return [{'table':t,'case':c,'K':k,'comparison':v,'primary':rows[0]['primary'],
             'mode_A':rows[0]['mode_A'],'mode_B':rows[0]['mode_B'],**pair_summary(rows)}
            for (t,c,k,v),rows in groups.items()]

def telemetry():
    command=['nvidia-smi','--query-gpu=index,uuid,name,temperature.gpu,clocks.sm,power.draw,memory.free',
             '--format=csv,noheader,nounits']
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=15,check=False)
        return {'status':'AVAILABLE' if p.returncode==0 else 'UNAVAILABLE','command':command,
                'stdout':p.stdout,'stderr':p.stderr,'returncode':p.returncode,
                'scope':'read-only snapshots, not profiler counters or controlled clocks'}
    except (OSError,subprocess.TimeoutExpired) as e:return {'status':'UNAVAILABLE','error':repr(e)}

def load_execution_inputs(m):
    from check_gpu_solver import natural_cases,load_parameters
    cases={}
    for k in m['K']:
        dirs={r['capture_dir'] for r in m['cases'] if r['K']==k}
        require(len(dirs)==1,'Expected one capture directory per K')
        for c in natural_cases(Path(next(iter(dirs))),(('LandTiles',3,0,3),('SdfTrees',4,1,1)),k):
            cases[f'K{k}:{c.name}']=c
    require(set(cases)=={case_key(r) for r in m['cases']},'Frozen case set changed')
    for r in m['cases']:
        c=cases[case_key(r)]
        require(c.evidence['capture_input_sha256']==r['input_sha256'] and len(c.centers)==r['N'] and
                len(c.endpoints)==r['M'] and c.evidence['counts']==r['m_counts'],'Frozen input mismatch')
    return cases,load_parameters(PARAMETERS,digest(PARAMETERS))

def bind_timing(api):
    import ctypes as C
    api._bind(api.solver,'bm_solver_get_info',[C.c_void_p,C.c_int,C.POINTER(C.c_uint64),C.c_size_t])
    api._bind(api.solver,'bm_solver_timing_prepare',[C.c_void_p,C.c_int,C.POINTER(C.c_double)])
    api._bind(api.solver,'bm_solver_measure_resident',
              [C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_double)])

class ResidentCase:
    # One immutable handle shared by all five modes. No field or input update.
    def __init__(self,api,case,params,destination):
        import ctypes as C
        from check_gpu_solver import pointer,PF,PI,PD
        self.C,self.pointer,self.PF,self.PI,self.PD=C,pointer,PF,PI,PD
        self.api,self.case,self.destination=api,case,destination
        self.preparation={};self.checks=[]
        start=time.perf_counter_ns();self.field,view=api.create_field(case.kind,params)
        self.preparation['field_initialization_upload_ms']=(time.perf_counter_ns()-start)/1e6
        start=time.perf_counter_ns();self.handle=api.create_solver(case,view)
        self.preparation['solver_allocation_owner_map_upload_ms']=(time.perf_counter_ns()-start)/1e6
        before=self.info()
        require(before[1]==0 and before[38]==1 and before[31]==0 and before[32]==0,'Fresh ordinary solver required')
        require(before[2]==len(case.centers) and before[3]==len(case.endpoints) and
                before[7]==len(case.centers)+(case.K+1)*len(case.endpoints),'Solver N/M/Q mismatch')
        start=time.perf_counter_ns();api.check(api.solver.bm_solver_prepare_graph(self.handle,case.K,0),'graph_prepare')
        self.preparation['graph_capture_instantiate_ms']=(time.perf_counter_ns()-start)/1e6
        elapsed=C.c_double()
        api.check(api.solver.bm_solver_timing_prepare(self.handle,1,C.byref(elapsed)),'timing_prepare')
        self.preparation.update(event_creation_ms=elapsed.value,info_before_graph_events=before,
            info_after_graph_events=self.info(),graph_update='NOT_USED: new handle per case',
            graph_key='immutable N/M/K/trace=0 and field/geometry buffer identity',first_solve_host_ms={},
            graph_cost_applies_to='R1',event_cost_is_measurement_instrumentation=True)
        write_json(destination/'preparation.json',self.preparation)
    def info(self):
        words=(self.C.c_uint64*40)()
        self.api.check(self.api.solver.bm_solver_get_info(self.handle,self.case.K,words,40),'solver_info')
        return list(map(int,words))
    def check_output(self,mode,purpose,save=False):
        import numpy as np
        from check_gpu_solver import same
        c=self.case;n,m=len(c.centers),len(c.endpoints)
        out={'position':np.empty((n,3),np.float32),'witness':np.empty(n,np.int32),'valid':np.empty(n,np.int32),
             'left':np.empty(n,np.float64),'right':np.empty(n,np.float64),'aux':np.empty((n+(c.K+1)*m,3),np.float32)}
        p,pf,pi,pd=self.pointer,self.PF,self.PI,self.PD
        self.api.check(self.api.solver.bm_solver_readback(self.handle,p(out['position'],pf),p(out['witness'],pi),
            p(out['valid'],pi),p(out['left'],pd),p(out['right'],pd),p(out['aux'],pf),pf(),pf(),pi(),pd(),pd()),'outside_timing_readback')
        bad=[k for k,v in out.items() if not same(v,c.oracle[k])]
        if save or bad:np.savez(self.destination/(mode+'_'+purpose+'_actual.npz'),**out)
        if bad:
            np.savez(self.destination/'first_failure_oracle.npz',**{k:c.oracle[k] for k in out})
            write_json(self.destination/'first_failure.json',{'case':c.name,'K':c.K,'mode':mode,
                'purpose':purpose,'mismatches':bad,'input_evidence':c.evidence})
            raise RuntimeError('Resident mismatch; actual/oracle saved: '+str(bad))
        self.checks.append({'mode':mode,'purpose':purpose,'status':'PASS','outside_timing':True,'trace':0})
        return self.info()
    def warmup(self,count):
        for i,mode in enumerate(MODES):
            for j in range(count):
                start=time.perf_counter_ns() if j==0 else None
                self.api.check(self.api.solver.bm_solver_run(self.handle,i,self.case.K,0),'warmup_solve')
                if j==0:self.preparation['first_solve_host_ms'][mode]=(time.perf_counter_ns()-start)/1e6
                if j in (0,1):self.check_output(mode,'first' if j==0 else 'repeat',True)
            self.check_output(mode,'warmup_final')
    def measure(self,mode,repeats):
        C=self.C;before=self.info();event,wall=C.c_double(),C.c_double()
        self.api.check(self.api.solver.bm_solver_measure_resident(self.handle,MODES.index(mode),self.case.K,
            repeats,1,C.byref(event),C.byref(wall)),'measure_resident')
        after=self.info()
        require(after[34]-before[34]==repeats and after[32]-before[32]==2,'Actual solve/event count mismatch')
        require(math.isfinite(event.value) and event.value>0 and math.isfinite(wall.value) and wall.value>0,'Invalid timing')
        return {'event_ms':event.value,'host_submission_and_sync_ms':wall.value,'repeats':repeats,
            'per_solve_ms':event.value/repeats,'host_ms_per_solve':wall.value/repeats,
            'logical_queries_per_solve':after[7],'logical_queries_in_block':repeats*after[7],
            'completed_solves_delta':after[34]-before[34],'event_records_delta':after[32]-before[32],
            'trace':0,'internal_repeats_are_independent_inputs':False}
    def finish(self):
        # Normal completion only. Failure exits worker without another CUDA call.
        self.preparation['final_info']=self.info()
        self.api.check(self.api.solver.bm_solver_destroy(self.handle),'solver_destroy')
        self.api.check(self.api.field.bm_field_destroy(self.field),'field_destroy')
        write_json(self.destination/'preparation.json',self.preparation)
        write_json(self.destination/'output_checks.json',self.checks)


def load_plane_cases():
    # Select exact already-validated synthetic fixtures. Never relabel a real graph.
    from check_gpu_solver import diagnostic_cases
    selected={f'K{c.K}:{c.name}':c for c in diagnostic_cases()
              if c.name in ('diagnostic_kind0_K3','diagnostic_kind0_K6')}
    require(set(selected)=={'K3:diagnostic_kind0_K3','K6:diagnostic_kind0_K6'},'Plane fixture missing')
    for c in selected.values():
        require(c.kind==0 and c.params is not None and len(c.centers)==33 and len(c.endpoints)==608,
                'Original plane fixture dimensions/field changed')
        require(c.evidence.get('not_real_field_coverage') is True,'Plane must stay diagnostic only')
    return selected

def calibrate_plane(m,cases,api,worker_dir):
    result={};target=m['repeat_calibration']['target_event_ms'];cap=m['repeat_calibration']['max_repeats']
    for name,case in cases.items():
        directory=worker_dir/'calibration_plane'/name.replace(':','_');directory.mkdir(parents=True)
        session=ResidentCase(api,case,case.params,directory);session.warmup(m['warmups'])
        repeats,reached=1,False
        while True:
            elapsed=[]
            for mode in MODES:
                measured=session.measure(mode,repeats);elapsed.append(measured['event_ms'])
                append_json(directory/'raw.jsonl',{'phase':'plane_repeat_resolution','mode':mode,**measured})
                session.check_output(mode,f'plane_repeat{repeats}')
            reached=min(elapsed)>=target
            if reached or repeats==cap:break
            repeats*=2
        result[field_key('analytic_plane',case.K)]={'calibration_case':name,'resident':'R1','local':'L1_8',
            'repeats':repeats,'target_ms':target,'repeat_cap':cap,'resolution_target_reached':reached,
            'selection':'PREDECLARED_R1_AND_L1_8_NOT_OPTIMIZED','has_held_out':False,
            'not_real_field_coverage':True,'does_not_select_real_field_configuration':True}
        session.finish()
        print(json.dumps({'worker':0,'phase':'plane_repeat_calibration','case':name,'repeats':repeats}),flush=True)
    return result

def calibrate(m,cases,params,api,worker_dir,suite_dir,plane_cases):
    selection={'status':'FROZEN_AFTER_WORKER0_CALIBRATION','worker':0,'matrix_sha256':digest(MATRIX),
               'fields':{},'held_out_used':False,'measurement_repeats_do_not_amortize_single_use_graph_cost':True}
    target=m['repeat_calibration']['target_event_ms'];maximum=m['repeat_calibration']['max_repeats']
    for r in (r for r in m['cases'] if r['role']=='calibration'):
        case=cases[case_key(r)];directory=worker_dir/'calibration'/case_key(r).replace(':','_')
        directory.mkdir(parents=True);s=ResidentCase(api,case,params[case.kind],directory);s.warmup(m['warmups'])
        repeats,reached=1,False
        while True:
            elapsed=[]
            for mode in MODES:
                observed=s.measure(mode,repeats);elapsed.append(observed['event_ms'])
                append_json(directory/'raw.jsonl',{'phase':'repeat_resolution','mode':mode,**observed})
                s.check_output(mode,f'calibration_repeat{repeats}')
            reached=min(elapsed)>=target
            if reached or repeats==maximum:break
            repeats*=2
        scores={mode:[] for mode in MODES}
        for trial in range(3):
            for mode in MODES[trial:]+MODES[:trial]:
                observed=s.measure(mode,repeats);scores[mode].append(observed['per_solve_ms'])
                append_json(directory/'raw.jsonl',{'phase':'configuration_selection','trial':trial,'mode':mode,**observed})
                s.check_output(mode,f'selection_{trial}',trial==2)
        medians={mode:statistics.median(values) for mode,values in scores.items()}
        resident=min(('R0','R1'),key=lambda mode:(medians[mode],0 if mode=='R1' else 1))
        local=min(('L0','L1_8','L1_32'),key=lambda mode:(medians[mode],{'L0':1,'L1_8':8,'L1_32':32}[mode]))
        selection['fields'][field_key(r['field'],r['K'])]={
            'calibration_case':case_key(r),'resident':resident,'local':local,'repeats':repeats,
            'target_ms':target,'resolution_target_reached':reached,'repeat_cap':maximum,
            'selection_median_ms':medians,'selection_samples_per_mode':3}
        s.finish()
        print(json.dumps({'worker':0,'phase':'calibration','case':case_key(r),'repeats':repeats,
                          'resident':resident,'local':local}),flush=True)
    selection['analytic_plane']=calibrate_plane(m,plane_cases,api,worker_dir)
    require(not (suite_dir/'SELECTION.json').exists(),'Never overwrite configuration selection')
    write_json(suite_dir/'SELECTION.json',selection)
    return selection

def comparisons(choice):
    result=[('R1',mode) for mode in MODES if mode!='R1']
    primary=(choice['resident'],choice['local'])
    if primary not in result:result.append(primary)
    return [(a,b,(a,b)==primary) for a,b in result]

def run_resident(m,cases,params,api,selection,worker,worker_dir,raw,*,rows=None,table='resident',role='held_out'):
    for r in (held_out(m) if rows is None else rows):
        case=cases[case_key(r)];directory=worker_dir/table/case_key(r).replace(':','_')
        directory.mkdir(parents=True)
        field_params=case.params if params is None else params[case.kind]
        s=ResidentCase(api,case,field_params,directory);s.warmup(m['warmups'])
        choice=selection['fields'][field_key(r['field'],r['K'])]
        for a,b,primary in comparisons(choice):
            comparison=a+'_vs_'+b
            for pair,order in enumerate(orders(m,worker,case_key(r),comparison)):
                for arm in order:
                    mode=a if arm=='A' else b;observed=s.measure(mode,choice['repeats'])
                    row={'table':table,'case':r['id'],'K':r['K'],'field':r['field'],'role':role,
                         'worker':worker,'comparison':comparison,'primary':primary,'mode_A':a,'mode_B':b,
                         'mode':mode,'pair':pair,'order':order,'arm':arm,**observed}
                    raw.append(row);append_json(worker_dir/'trials.jsonl',row)
                    if pair in (0,1,m['pairs']-1):
                        s.check_output(mode,comparison+f'_pair{pair:02d}',pair==m['pairs']-1)
        s.finish()
        print(json.dumps({'worker':worker,'phase':table,'case':case_key(r),'status':'COMPLETED'}),flush=True)

def remove_private_checkpoint(path,root):
    resolved,base=path.resolve(),root.resolve()
    require(resolved!=base and resolved.is_relative_to(base) and not path.is_symlink(),
            'Refuse to delete outside this worker private checkpoint root')
    shutil.rmtree(resolved)

def run_common_table(m,worker,worker_dir,raw):
    import numpy as np
    from check_gpu_solver import same
    from common_runner import run_common
    source=ROOT/'inputs/checkpoint_seed';private=worker_dir/'private_checkpoints';private.mkdir()
    records=worker_dir/'common';records.mkdir();serial=0;expected={}
    for k in m['K']:
        rows=[r for r in m['cases'] if r['K']==k];capture=Path(rows[0]['capture_dir'])
        last=max(r['batch'] for r in rows)
        with np.load(capture/f'g00000_b{last:05d}_native_outputs.npz',allow_pickle=False) as f:
            expected[k]={key:f[key].copy() for key in f.files}
    def arm(mode,k,phase,label,timed,save_output=False):
        nonlocal serial
        serial+=1;identity=f'a{serial:05d}_K{k}_{mode}';cp=private/identity
        shutil.copytree(source,cp) # same private input provisioning is outside the comparison boundary
        actual,record=run_common(mode,k,cp,timing=timed,validate=False,solver_path=ROOT/'build/libgpu_solver.so')
        write_json(records/(identity+'.json'),{'phase':phase,'label':label,'mode':mode,'K':k,**record})
        bad=[key for key,value in actual.items() if not same(value,expected[k][key])]
        if bad or save_output:np.savez(records/(identity+'_actual.npz'),**actual)
        if bad:
            np.savez(records/(identity+'_first_failure_oracle.npz'),**expected[k])
            write_json(records/'first_failure.json',{'id':identity,'mismatches':bad,'checkpoint':str(cp)})
            raise RuntimeError('Common output mismatch; first actual/oracle saved: '+str(bad))
        # Every complete output check above is outside run_common's timed parent.
        remove_private_checkpoint(cp,private)
        if not timed:return None
        roots=[r for r in record['stages'] if r['stage']=='checkpoint_to_output' and r['parent'] is None]
        require(len(roots)==1 and roots[0]['wall_ns']>0,'Missing complete common-boundary wall')
        return {'per_solve_ms':roots[0]['wall_ns']/1e6,'repeats':1,
          'stage_file':str((records/(identity+'.json')).relative_to(worker_dir)),
          'full_output_check':'PASS_OUTSIDE_TIMING','graph_reuse':1,
          'checkpoint_provisioning_timed':False,
          'checkpoint_read_prepare_solve_readback_cleanup_timed':True}
    for k in m['K']:
        for mode in ('C0',*MODES):
            for i in range(m['warmups']):
                arm(mode,k,'warmup',i,False,i in (0,1,m['warmups']-1))
        for candidate in MODES:
            comparison='C0_vs_'+candidate
            for pair,order in enumerate(orders(m,worker,'whole_checkpoint_K'+str(k),comparison)):
                for side in order:
                    mode='C0' if side=='A' else candidate
                    measured=arm(mode,k,'formal',f'{comparison}:{pair}:{side}',True,
                                 pair in (0,1,m['pairs']-1))
                    row={'table':'common_checkpoint_to_output','case':'whole_checkpoint','K':k,
                      'field':'LandTiles+SdfTrees','role':'complete_original_group','worker':worker,
                      'comparison':comparison,'primary':False,'mode_A':'C0','mode_B':candidate,
                      'mode':mode,'pair':pair,'order':order,'arm':side,**measured}
                    raw.append(row);append_json(worker_dir/'trials.jsonl',row)
            print(json.dumps({'worker':worker,'phase':'common','K':k,'comparison':comparison,'status':'COMPLETED'}),flush=True)

def worker(args):
    gates(args.execute_performance,args.verdict);m=read_json(MATRIX);matrix_check(m)
    suite=args.out.resolve();prepared=read_json(suite/'PLAN.json')
    require(prepared['matrix_sha256']==digest(MATRIX),'Frozen matrix changed')
    for item in prepared['fingerprints']:
        require(item['sha256'] and digest(ROOT/item['path'])==item['sha256'],
                'Prepared source/input/build changed: '+item['path'])
    for item in prepared['checkpoint_files']:
        require(digest(ROOT/'inputs/checkpoint_seed'/item['path'])==item['sha256'],'Checkpoint seed changed')
    directory=suite/f'worker_{args.worker_index}';directory.mkdir(exist_ok=False)
    report={'status':'RUNNING','worker':args.worker_index,'pid':os.getpid(),'performance':'RUNNING',
      'automatic_retries':0,'failure_policy':'save first error and exit; no additional CUDA call',
      'hardware_profiling':'NOT_RUN','analytic_plane_diagnostic':'NOT_RUN',
      'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','default_device_0')}
    write_json(directory/'STATUS.json',report);raw=[]
    try:
        report['telemetry_before']=telemetry()
        start=time.perf_counter_ns()
        from check_gpu_solver import API
        cases,params=load_execution_inputs(m);plane_cases=load_plane_cases()
        api=API(ROOT/'build/libfield_bridge.so',ROOT/'build/libgpu_solver.so');bind_timing(api)
        report['process_input_oracle_library_prepare_ms']=(time.perf_counter_ns()-start)/1e6
        if args.worker_index==0:
            report['analytic_plane_diagnostic']='CALIBRATION_PENDING'
            write_json(directory/'STATUS.json',report)
            selection=calibrate(m,cases,params,api,directory,suite,plane_cases)
            report['analytic_plane_diagnostic']='CALIBRATED_NOT_FORMALLY_MEASURED'
        else:
            require(digest(suite/'SELECTION.json')==args.selection_sha256,'Worker selection fingerprint changed')
            selection=read_json(suite/'SELECTION.json')
        require(selection['matrix_sha256']==digest(MATRIX) and selection['worker']==0,'Invalid selection owner/matrix')
        report['selection_sha256']=digest(suite/'SELECTION.json');write_json(directory/'STATUS.json',report)
        run_resident(m,cases,params,api,selection,args.worker_index,directory,raw)
        report['analytic_plane_diagnostic']='RUNNING';write_json(directory/'STATUS.json',report)
        plane_rows=[{'id':c.name,'K':c.K,'field':'analytic_plane','role':'diagnostic_no_held_out'}
                    for c in plane_cases.values()]
        run_resident(m,plane_cases,None,api,{'fields':selection['analytic_plane']},
                     args.worker_index,directory,raw,rows=plane_rows,
                     table='analytic_plane_diagnostic',role='diagnostic_no_held_out')
        report['analytic_plane_diagnostic']='MEASURED'
        run_common_table(m,args.worker_index,directory,raw)
        report.update(summary=summarize(raw),raw_timed_arms=len(raw),telemetry_after=telemetry(),
                      status='COMPLETED',performance='MEASURED',
                      independence_unit='process and original input; internal repeats are not independent')
    except BaseException as e:
        if report.get('analytic_plane_diagnostic') in ('RUNNING','CALIBRATION_PENDING'):
            report['analytic_plane_diagnostic']='INCOMPLETE_SEE_DIAGNOSTIC_LOGS'
        report.update(status='FAILED',performance='FAILED_NOT_ELIGIBLE_FOR_SPEEDUP_CLAIMS',
                      error=repr(e),traceback=traceback.format_exc());raise
    finally:write_json(directory/'STATUS.json',report)

def aggregate(suite):
    workers=[read_json(suite/f'worker_{i}/STATUS.json') for i in range(3)]
    require(all(w['status']=='COMPLETED' for w in workers),'Missing completed worker')
    groups={}
    for w in workers:
        for r in w['summary']:
            groups.setdefault((r['table'],r['case'],r['K'],r['comparison']),[]).append({'worker':w['worker'],**r})
    result=[]
    for (table,case,k,comparison),rows in groups.items():
        require(len(rows)==3,'Missing process-level comparison')
        ratios=[r['median_paired_ratio'] for r in rows]
        result.append({'table':table,'case':case,'K':k,'comparison':comparison,'primary':rows[0]['primary'],
            'process_summaries':rows,'median_of_process_paired_medians':statistics.median(ratios),
            'process_ratio_range':[min(ratios),max(ratios)],
            'inference':'descriptive three-process variation; no independent-geometry bootstrap claim'})
    write_json(suite/'SUMMARY.json',{'status':'COMPLETED','performance':'MEASURED','rows':result,
        'selection_sha256':digest(suite/'SELECTION.json'),'hardware_profiling':'NOT_RUN',
        'analytic_plane_diagnostic':'MEASURED_DIAGNOSTIC_ONLY_NO_HELD_OUT',
        'complete_meshing_speedup':'NOT_RUN',
        'resident_table_C0':'N/A','all_slow_and_negative_trials_retained':True})


def orchestrate(args):
    gates(args.execute_performance,args.verdict)
    require(sys.platform.startswith('linux'),'Actual workers require isolated WSL runtime')
    prepared=plan()
    require(all(i['sha256'] for i in prepared['fingerprints']) and prepared['checkpoint_files'],
            'Actual build artifacts and full checkpoint must exist before timing')
    out=args.out.resolve()
    require(out!=ROOT and (out.is_relative_to(ROOT/'results') or out.is_relative_to(ROOT/'runs')),
            'Use a fresh results/ or runs/ directory in this experiment')
    out.mkdir(parents=True,exist_ok=False);write_json(out/'PLAN.json',prepared)
    status={'status':'RUNNING','performance':'RUNNING','completed_workers':[],
            'authorization_sha256':digest(AUTH),'verdict_sha256':digest(args.verdict),'automatic_retries':0}
    write_json(out/'STATUS.json',status);frozen_selection=None
    try:
        for index in range(3):
            gates(True,args.verdict)
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--execute-performance',
                     '--worker-index',str(index),'--out',str(out),'--verdict',str(args.verdict)]
            if index:
                require(digest(out/'SELECTION.json')==frozen_selection,'Frozen worker 0 selection changed')
                command+=['--selection-sha256',frozen_selection]
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            env.setdefault('CUDA_VISIBLE_DEVICES','0')
            write_json(out/f'worker_{index}_command.json',{'argv':command,'timeout_seconds':args.worker_timeout})
            print(json.dumps({'phase':'worker_start','worker':index}),flush=True)
            with (out/f'worker_{index}_stdout.log').open('xb') as stdout,(out/f'worker_{index}_stderr.log').open('xb') as stderr:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                try:code=proc.wait(timeout=args.worker_timeout)
                except BaseException:
                    # Only stop this suite's newly launched worker process group.
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
            require(code==0,f'Worker {index} failed with exit {code}; stop without retry')
            report=read_json(out/f'worker_{index}/STATUS.json')
            require(report['status']=='COMPLETED','Worker lacks completed report')
            if index==0:
                frozen_selection=digest(out/'SELECTION.json')
                require(report['selection_sha256']==frozen_selection,'Worker 0 selection changed during formal trials')
            else:require(report['selection_sha256']==frozen_selection,'Workers used different selections')
            status['completed_workers'].append(index);write_json(out/'STATUS.json',status)
            print(json.dumps({'phase':'worker_complete','worker':index}),flush=True)
        aggregate(out);status.update(status='COMPLETED',performance='MEASURED')
    except BaseException as e:
        status.update(status='FAILED',performance='INCOMPLETE_NOT_ELIGIBLE_FOR_SPEEDUP_CLAIMS',
                      error=repr(e),traceback=traceback.format_exc());raise
    finally:write_json(out/'STATUS.json',status)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--describe',action='store_true')
    action.add_argument('--execute-performance',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--verdict',type=Path,default=VERDICT)
    parser.add_argument('--worker-timeout',type=int,default=3600)
    parser.add_argument('--worker-index',type=int,choices=(0,1,2),help=argparse.SUPPRESS)
    parser.add_argument('--selection-sha256',help=argparse.SUPPRESS)
    args=parser.parse_args();args.verdict=args.verdict.resolve()
    require(args.verdict.is_relative_to(ROOT),'Use this new experiment readiness verdict')
    require(args.worker_timeout>0,'Worker timeout must be positive')
    if not args.execute_performance:
        require(args.worker_index is None,'Workers require explicit execution switch')
        prepared=plan()
        if args.out is not None:
            out=args.out.resolve()
            require(out!=ROOT and (out.is_relative_to(ROOT/'results') or out.is_relative_to(ROOT/'runs')),
                    'Preparation output must remain inside new results/ or runs/')
            out.mkdir(parents=True,exist_ok=False);write_json(out/'PLAN.json',prepared)
        print(json.dumps(prepared,indent=2,sort_keys=True));return
    if args.out is None:args.out=ROOT/'results'/('performance_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    if args.worker_index is not None:
        require(args.out.resolve()!=ROOT and (args.out.resolve().is_relative_to(ROOT/'results') or
                args.out.resolve().is_relative_to(ROOT/'runs')),'Worker output outside experiment results/runs')
        worker(args)
    else:orchestrate(args)

if __name__=='__main__':main()
