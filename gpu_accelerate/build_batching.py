#!/usr/bin/env python3
"""Build the Published-only batching adapter from this checkout; never run a GPU.

Explicit external CUDA/GCC/Infinigen dependencies. Linux/WSL only for builds;
--help and --dry-run perform no compiler calls or filesystem writes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parent
REPOSITORY=PACKAGE.parent
REMOVED=('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS')
STRICT=['-O3','-std=c++17','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true',
 '-Xcompiler=-fPIC,-fno-fast-math,-ffp-contract=off,-fvisibility=hidden,-ffunction-sections,-fdata-sections','-Xptxas=-v']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''):h.update(data)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def executable(value,dry_run):
    found=shutil.which(str(value))
    if found:return str(Path(found).resolve())
    path=Path(value).expanduser()
    if path.is_file():return str(path.resolve())
    if dry_run:return str(path)
    raise ValueError('Missing existing compiler: '+str(value))


def build_plan(args):
    src=PACKAGE/'src';native=PACKAGE/'batching/native'
    sources=[native/'published_adapter.cu',native/'field_bridge_guarded.cu']
    nvcc=executable(args.nvcc,args.dry_run);host=executable(args.host_compiler,args.dry_run)
    common=[nvcc,'-ccbin',host,*STRICT,'-arch='+args.arch,'-DBM_L132_BLOCK_THREADS=128','-DBM_SOLVER_DEBUG_GUARDS=0',
        '-I',str(native),'-I',str(src),'-I',str(args.infinigen_include)]
    artifact=args.output/'libbatching_published.so'
    command=common+['-shared',*map(str,sources),'-L',str(args.driver_library_dir),'-lcuda',
        '-Xlinker=--gc-sections','-Xlinker=--version-script='+str(native/'exports.map'),'-o',str(artifact)]
    inputs=list(native.glob('*'))+[Path(__file__),PACKAGE/'build.py',PACKAGE/'D1_SOURCE_MANIFEST.json']
    inputs += [src/name for name in ('d4_solver.cu','d4_pipeline.cuh','d4_field_eval.cuh','gpu_solver.cu','field_bridge.cu','field_eval.cuh','field_types.h')]
    dependencies=[dict(name=source.stem,argv=common+['-M',str(source),'-o',str(args.output/(source.stem+'.d'))]) for source in sources]
    return dict(schema=1,backend='published',graph='OPTIONAL_BACKEND_NOT_PACKAGED',GPU_execution=False,
        output=str(args.output),artifact=str(artifact),command=command,dependency_commands=dependencies,
        input_sha256={str(p):digest(p) if p.is_file() else None for p in inputs},
        missing_inputs=[str(p) for p in inputs if not p.is_file()],infinigen_include=str(args.infinigen_include),
        driver_library_dir=str(args.driver_library_dir),strict_flags=STRICT,timeout_per_command_seconds=1800,
        removed_environment_overrides=list(REMOVED),compiler_versions_commands=[[nvcc,'--version'],[host,'--version']],
        build_scope='One new DSO; original numerical sources included unchanged. No old binary reuse; no Graph exports.')


def execute(args,plan):
    if not sys.platform.startswith('linux'):raise ValueError('Actual compilation requires existing Linux/WSL tools')
    if plan['missing_inputs']:raise ValueError('Missing source inputs: '+repr(plan['missing_inputs']))
    if not (args.infinigen_include/'terrain/source/common/elements/sdf_trees.h').is_file():
        raise ValueError('Expected inner Infinigen directory with terrain/source/common/elements/sdf_trees.h')
    if not args.driver_library_dir.is_dir():raise ValueError('Missing explicit CUDA driver library directory')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):raise ValueError('Build output must be new or empty')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'tmp').mkdir()
    environment=os.environ.copy()
    for name in REMOVED:environment.pop(name,None)
    environment.update(TMPDIR=str(args.output/'tmp'),TMP=str(args.output/'tmp'),TEMP=str(args.output/'tmp'))
    # Preserve the caller's runtime search path: all compiler selection is explicit.
    status=dict(status='BUILDING',GPU_execution=False,automatic_retries=0,commands=[])
    dump(args.output/'BUILD_PLAN.json',plan)
    def command(name,argv,timeout=1800):
        started=time.monotonic();record=dict(name=name,argv=argv,timeout_seconds=timeout)
        with (args.output/(name+'.stdout.log')).open('xb') as stdout,(args.output/(name+'.stderr.log')).open('xb') as stderr:
            process=subprocess.Popen(argv,cwd=args.output,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            try:code=process.wait(timeout=timeout)
            except BaseException:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise
        record.update(returncode=code,elapsed_seconds=time.monotonic()-started);status['commands'].append(record)
        with (args.output/'COMMANDS.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        if code:raise RuntimeError(name+' failed; original logs retained')
        return record
    try:
        for index,argv in enumerate(plan['compiler_versions_commands']):command('version_'+str(index),argv,30)
        for row in plan['dependency_commands']:command('dependencies_'+row['name'],row['argv'])
        closure=set()
        for row in plan['dependency_commands']:
            dependency=(args.output/(row['name']+'.d')).read_text().replace('\\\n',' ')
            for word in shlex.split(dependency.split(':',1)[1]):closure.add(Path(word).resolve())
        dependencies={str(path):digest(path) for path in sorted(closure)}
        dump(args.output/'INCLUDE_CLOSURE.json',dict(schema=1,files=dependencies,count=len(dependencies)))
        command('published',plan['command'])
        for path,expected in {**plan['input_sha256'],**dependencies}.items():
            if digest(path)!=expected:raise RuntimeError('Source/header changed during build: '+path)
        artifact=Path(plan['artifact'])
        status.update(status='COMPILED_NOT_GPU_VALIDATED',library=dict(path=str(artifact),bytes=artifact.stat().st_size,sha256=digest(artifact)),include_closure_count=len(dependencies))
        return 0
    except BaseException as error:status.update(status='FAILED_OR_INTERRUPTED',error=repr(error));raise
    finally:dump(args.output/'BUILD_STATUS.json',status)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc',required=True,help='existing nvcc executable')
    parser.add_argument('--host-compiler',required=True,help='existing GCC-compatible executable')
    parser.add_argument('--infinigen-include',required=True,type=Path,help='explicit inner upstream Infinigen directory')
    parser.add_argument('--driver-library-dir',required=True,type=Path,help='existing directory containing libcuda.so')
    parser.add_argument('--output',required=True,type=Path,help='new or empty directory outside checkout')
    parser.add_argument('--arch',default='sm_120')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    try:
        for name in ('infinigen_include','driver_library_dir','output'):setattr(args,name,getattr(args,name).expanduser().resolve())
        if not re.fullmatch(r'sm_[0-9]+[af]?',args.arch):raise ValueError('Invalid CUDA sm_ architecture')
        if args.output==Path(args.output.anchor) or args.output.is_relative_to(REPOSITORY):raise ValueError('--output must be outside checkout, not a filesystem root')
        plan=build_plan(args)
        if args.dry_run:print(json.dumps(plan,indent=2));return 0
        return execute(args,plan)
    except (ValueError,OSError,RuntimeError) as error:
        print('Batching build error: '+str(error),file=sys.stderr);return 1
    except KeyboardInterrupt:return 130


if __name__=='__main__':raise SystemExit(main())
