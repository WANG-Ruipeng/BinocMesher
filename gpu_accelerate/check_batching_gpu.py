#!/usr/bin/env python3
"""Finite real-field integration checks; never runs CUDA unless --execute is set."""
from pathlib import Path
import argparse, hashlib, json, sys, traceback
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gpu_accelerate.batching.inputs import load_inputs
from gpu_accelerate.batching.runner import solve_many
from gpu_accelerate.batching.pack import pack_tasks, split_outputs, output_hashes
from gpu_accelerate.batching.backend import PublishedBackend

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(value,message):
    if not value: raise RuntimeError(message)
def compare(actual,expected):
    require(set(actual)==set(expected),'Task identity/order mismatch')
    for task,arrays in actual.items():
        require(set(arrays)==set(expected[task]),'Output contract mismatch '+task)
        for name,data in arrays.items():
            wanted=expected[task][name]
            if data!=wanted:
                at=next((i for i,(a,b) in enumerate(zip(data,wanted)) if a!=b),min(len(data),len(wanted)))
                raise RuntimeError(f'OUTPUT_MISMATCH task={task} field={name} first_byte={at}')
    return {name:output_hashes(arrays) for name,arrays in actual.items()}
def projected(refs,tasks,trace):
    keys=('position','witness','valid','left','right') if not trace else ('position','witness','valid','left','right','aux','sdf','xyz','sign','trace_left','trace_right')
    return {t.task_id:{key:refs[t.task_id][key] for key in keys} for t in tasks}
def historical_check(outputs,manifest):
    for task,arrays in outputs.items():
        for key,value in arrays.items():
            want=manifest[task][key]
            require(len(value)==want['bytes'] and hashlib.sha256(value).hexdigest()==want['sha256'],f'HISTORICAL_NUMERIC_BRIDGE {task}/{key}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--fields',type=Path,required=True);parser.add_argument('--jobs',type=Path,required=True)
    parser.add_argument('--fields-sha256',required=True);parser.add_argument('--jobs-sha256',required=True)
    parser.add_argument('--historical-hashes',type=Path,required=True,help='Published reference array sizes/hashes; comparison only')
    parser.add_argument('--expected-device-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('regression','smoke','memcheck'),default='regression')
    parser.add_argument('--audit',type=int,choices=(0,1),default=0)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(GPU_execution=False,phase=args.phase,requires_explicit_execute=True)));return 0
    root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    require(not out.is_relative_to(root) and not out.exists(),'Output must be new and outside the source checkout')
    out.mkdir(parents=True)
    report=dict(status='RUNNING',phase=args.phase,GPU_execution=True,audit=args.audit,records=[],memory=[],errors=[],cleanup_errors=[],
        historical_performance_retested=False,Graph='OPTIONAL_BACKEND_NOT_PACKAGED',actual_application_solves=0,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    field=None;backend=None;refs={}
    try:
        parameters,geometries,identity=load_inputs(args.fields,args.jobs,fields_sha256=args.fields_sha256,jobs_sha256=args.jobs_sha256)
        report['inputs']=identity
        require(len(geometries)==10 and {g.k for g in geometries}=={3,6},'Expected real ten-job fixture')
        for k in (3,6):require(len([g for g in geometries if g.k==k])==5,'Expected five real tasks for each K')
        history=json.loads(args.historical_hashes.read_text())
        backend=PublishedBackend(args.library,expected_device_uuid=args.expected_device_uuid)
        report['memory'].append(dict(stage='initial',context=backend.context()))
        field=backend.create_field(parameters.meta,parameters.tree_ip,parameters.tree_fp,parameters.land_ip,parameters.land_fp,parameters_sha256=parameters.sha256)
        report['memory'].append(dict(stage='field_ready',context=backend.context()))
        if args.phase!='memcheck':
            for geometry in geometries:
                task=geometry.task(field,trace=True,audit=True)
                result=solve_many(field,[task])
                report['actual_application_solves']+=1
                historical_check(result.outputs,history)
                refs.update(result.outputs)
                report['records'].append(dict(kind='single_Published_reference',task=task.task_id,trace=True,audit=True,hashes=output_hashes(refs[task.task_id]),status='PASS'))
        if args.phase=='memcheck':
            for k in (3,6):
                tasks=[g.task(field,trace=bool(args.audit),audit=bool(args.audit)) for g in geometries if g.k==k]
                for repeat in range(2):
                    result=solve_many(field,tasks,batching='offline')
                    report['actual_application_solves']+=1
                    historical_check(result.outputs,history)
                    report['records'].append(dict(kind='memcheck_mixed',K=k,audit=args.audit,trace=args.audit,repeat=repeat,status='PASS',hashes={t:output_hashes(a) for t,a in result.outputs.items()}))
        else:
            for k in (3,6):
                for audit in (False,True):
                    tasks=[g.task(field,trace=audit,audit=audit) for g in geometries if g.k==k]
                    expected=projected(refs,tasks,audit)
                    if args.phase=='regression':
                        serial=solve_many(field,tasks,batching='off')
                        report['actual_application_solves']+=5
                        report['records'].append(dict(kind='serial_default',K=k,audit=audit,hashes=compare(serial.outputs,expected),status='PASS'))
                    for reverse in ((False,True) if args.phase=='regression' else (False,)):
                        ordered=list(reversed(tasks)) if reverse else tasks
                        for repeat in range(2 if args.phase=='regression' else 1):
                            result=solve_many(field,ordered,batching='offline')
                            report['actual_application_solves']+=1
                            require(result.task_ids==tuple(t.task_id for t in ordered),'Task dispatch order')
                            report['records'].append(dict(kind='offline_mixed',K=k,audit=audit,trace=audit,reverse=reverse,repeat=repeat,
                                hashes=compare(result.outputs,expected),status='PASS'))
            if args.phase=='regression':
                tasks=[g.task(field,trace=True,audit=True) for g in geometries if g.k==6]
                packed=pack_tasks(tasks);solver=field.create_solver(packed.task)
                try:
                    solver.prepare()
                    report['memory'].append(dict(stage='live_K6_mixed_handle',context=backend.context(),solver=solver.info()))
                    prior=None
                    for repeat in range(2):
                        solver.run();report['actual_application_solves']+=1
                        outputs=solver.readback();distributed=split_outputs(outputs,packed.segments)
                        hashes=compare(distributed,projected(refs,tasks,True))
                        if prior is not None:require(prior==outputs,'Repeated run mutated a retained output copy')
                        prior={key:bytes(value) for key,value in outputs.items()}
                        report['records'].append(dict(kind='same_handle_cold_reset',repeat=repeat,hashes=hashes,validation=solver.validation(),info=solver.info(),status='PASS'))
                    # Exercise the C ABI itself while Python still has its last success.
                    code=int(backend._lib.mb_solver_run(solver.token,7,1,1))
                    require(code!=0,'Native invalid-K run accepted')
                    report['native_invalid_run_code']=code
                    try:solver.readback()
                    except Exception as error:report['native_failed_run_readback_rejected']=repr(error)
                    else:raise RuntimeError('Native failed run exposed stale readback')
                    rejected=False
                    try:solver.run(k=7)
                    except Exception as error:rejected=True;report['invalid_run_error']=repr(error)
                    require(rejected,'Invalid K run accepted')
                    try:solver.readback()
                    except Exception as error:report['failed_run_readback_rejected']=repr(error)
                    else:raise RuntimeError('Stale readback survived rejected run')
                finally:solver.close()
                try:solver.readback()
                except Exception as error:report['destroyed_readback_rejected']=repr(error)
                else:raise RuntimeError('Destroyed solver returned output')
        report['memory'].append(dict(stage='after_solver_cleanup',context=backend.context()))
        report['status']='PASS_'+args.phase.upper()
    except BaseException as error:
        report['status']='FAIL';report['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:
        if field is not None:
            try:field.close()
            except BaseException as error:report['cleanup_errors'].append(repr(error))
        if backend is not None:
            try:
                context=backend.context()
                report['memory'].append(dict(stage='after_field_cleanup',context=context))
                require(context['tracked_allocation_bytes']==0 and context['live_fields']==0 and context['live_solvers']==0,'Native resource accounting did not return to zero')
                backend.close()
            except BaseException as error:report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']:report['status']='FAIL_CLEANUP'
        dump(out/'RESULT.json',report)
    print(json.dumps(dict(status=report['status'],phase=args.phase,solves=report['actual_application_solves'])))
    return 0 if report['status'].startswith('PASS_') else 1
if __name__=='__main__':raise SystemExit(main())