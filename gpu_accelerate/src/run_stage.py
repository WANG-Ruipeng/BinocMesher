"""Run one logged stage; stop on the first tool/runtime/correctness error."""
import argparse, datetime, json, os, re, signal, subprocess, time
from pathlib import Path
from workspace_paths import ROOT
from workspace_paths import tool
# Caller activates CUDA/Python and LD_LIBRARY_PATH. Only generated caches are local.
NATIVE_TMP=Path(os.environ.get('BM_GPU_TMPDIR', str(ROOT/'tmp'))).expanduser().resolve()

def environment():
    env=os.environ.copy()
    env.update({'OMP_NUM_THREADS':env.get('OMP_NUM_THREADS','8'),'OPENBLAS_NUM_THREADS':env.get('OPENBLAS_NUM_THREADS','1'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OPENCV_IO_ENABLE_OPENEXR':'1','TMPDIR':str(NATIVE_TMP),'TMP':str(NATIVE_TMP),'TEMP':str(NATIVE_TMP)})
    for key,name in [('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('PIP_CACHE_DIR','pip')]:
        p=ROOT/'cache'/name;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    for k in ('PYTHONPATH','PYTHONHOME','NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN'):env.pop(k,None)
    return env

def run(phase,argv,timeout=3600,memcheck=False):
    state_path=ROOT/'STATUS.json'
    previous=json.loads(state_path.read_text()) if state_path.exists() else {'status':'PREPARED'}
    if previous['status']=='STOPPED_ON_ERROR':print('Prior failed phase preserved; explicit next phase authorized by user recoverable-error policy',flush=True)
    if not phase or Path(phase).name != phase or phase in ('.','..'):raise ValueError('phase must be one directory name')
    log=ROOT/'logs'/phase;log.mkdir(parents=True,exist_ok=False)
    assert not NATIVE_TMP.is_symlink()
    NATIVE_TMP.mkdir(mode=0o700,parents=True,exist_ok=True)
    sanitizer_log=log/'sanitizer.log'
    if memcheck:
        argv=[tool('compute-sanitizer'),'--tool','memcheck','--target-processes','application-only','--report-api-errors','all','--require-cuda-init','yes','--error-exitcode','97','--destroy-on-device-error','context','--print-session-details','--show-backtrace','yes','--launch-timeout','60','--kill','--log-file',str(sanitizer_log),*argv]
    entry={'phase':phase,'argv':argv,'status':'RUNNING','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':timeout,'memcheck':memcheck,'automatic_retries':0}
    def save():
        (log/'command.json').write_text(json.dumps(entry,indent=2)+'\n');state_path.write_text(json.dumps(entry,indent=2)+'\n')
    def stop(p):
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
    save();start=time.monotonic();code=1;proc=None
    pattern=re.compile(r'^=+\s+(?:Program hit\b|Invalid\b|Misaligned\b|Heap corruption\b|Double free\b|Hardware exception\b|Error\s*:|Fatal\s*:|Target application returned an error\b|ERROR SUMMARY:\s*[1-9])',re.M|re.I)
    try:
        with (log/'stdout.log').open('xb') as out,(log/'stderr.log').open('xb') as err:
            proc=subprocess.Popen(argv,cwd=ROOT,env=environment(),stdout=out,stderr=err,start_new_session=True);entry['pid']=proc.pid;save()
            while True:
                if memcheck and sanitizer_log.exists():
                    s=sanitizer_log.read_text(errors='replace')
                    match=pattern.search(s)
                    if match:entry['first_error']=s[match.start():].splitlines()[0];stop(proc);code=97;break
                if proc.poll() is not None:code=proc.returncode;break
                if time.monotonic()-start>timeout:entry['first_error']='timeout';stop(proc);code=124;break
                time.sleep(0.5)
        if code==0 and memcheck:
            s=sanitizer_log.read_text(errors='replace') if sanitizer_log.exists() else ''
            if pattern.search(s) or 'ERROR SUMMARY: 0 errors' not in s:entry['first_error']='memcheck did not establish zero errors';code=97
    except BaseException as e:
        if proc and proc.poll() is None:stop(proc)
        entry['exception']=repr(e);code=1
    entry.update(status='SUCCEEDED' if code==0 else 'STOPPED_ON_ERROR',returncode=code,elapsed_seconds=time.monotonic()-start)
    save()
    with (ROOT/'COMMANDS.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
    print(json.dumps(entry,indent=2),flush=True)
    for name in ('stdout.log','stderr.log','sanitizer.log'):
        p=log/name
        if p.exists():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16000));print(name+':\n'+f.read().decode(errors='replace'),flush=True)
    return code
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--timeout',type=int,default=3600);ap.add_argument('--memcheck',action='store_true');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command;raise SystemExit(run(a.phase,cmd,a.timeout,a.memcheck))
