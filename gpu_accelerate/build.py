#!/usr/bin/env python3
"""Build the CUDA field bridge and GROUP=32 block variants outside the checkout.

Python standard library only. This builds sources; it does not run CUDA kernels,
install dependencies, change drivers, or establish correctness/performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BLOCKS = (256, 128, 64)
LOCAL_SOURCES = ('field_bridge.cu', 'gpu_solver.cu', 'field_eval.cuh', 'field_types.h')
UPSTREAM_HEADERS = (
    'terrain/source/common/utils/vectors.h',
    'infinigen_gpl/bnodes/utils/nodes_util.h',
    'infinigen_gpl/bnodes/utils/blender_noise.h',
    'terrain/source/common/utils/elements_util.h',
    'terrain/source/common/utils/FastNoiseLite.h',
    'terrain/source/common/utils/smooth_bool_ops.h',
    'terrain/source/common/elements/caves.h',
    'terrain/source/common/elements/landtiles.h',
    'terrain/source/common/elements/sdf_trees.h',
)
STRICT_FLAGS = (
    '-O3', '-std=c++17', '--fmad=false', '--ftz=false',
    '--prec-div=true', '--prec-sqrt=true',
    '-Xcompiler=-fPIC,-fno-fast-math,-ffp-contract=off', '-Xptxas=-v',
)


class BuildError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BuildError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def compiler_argument(value, dry_run):
    """Accept either a PATH command or a caller-supplied executable path."""
    located = shutil.which(value)
    if located:
        return str(Path(located).resolve())
    expanded = Path(value).expanduser()
    if expanded.is_file():
        return str(expanded.resolve())
    require(dry_run, 'Compiler not found: ' + value)
    return str(expanded) if expanded.parent != Path('.') else value


def build_plan(args):
    source_dir = PACKAGE / 'src'
    nvcc = compiler_argument(args.nvcc, args.dry_run)
    host = compiler_argument(args.host_compiler, args.dry_run) if args.host_compiler else None
    common = [nvcc]
    if host:
        common += ['-ccbin', host]
    # Match the accepted source compilation without enabling fast math or -G.
    common += [STRICT_FLAGS[0], STRICT_FLAGS[1], '-arch=' + args.arch, *STRICT_FLAGS[2:]]
    common += ['-I', str(args.infinigen_include), '-I', str(source_dir), '-shared']
    commands = [{
        'name': 'field_bridge', 'artifact': 'libfield_bridge.so',
        'argv': [*common, str(source_dir / 'field_bridge.cu'),
                 '-o', str(args.output / 'libfield_bridge.so')],
    }]
    for block in args.blocks:
        name = 'gpu_solver_b' + str(block) + ('_debug' if args.debug else '')
        defines = ['-DBM_L132_BLOCK_THREADS=' + str(block)]
        if args.debug:
            defines.append('-DBM_SOLVER_DEBUG_GUARDS=1')
        commands.append({
            'name': name, 'artifact': 'lib' + name + '.so',
            'block_threads': block, 'group_threads': 32,
            'argv': [common[0], *defines, *common[1:], str(source_dir / 'gpu_solver.cu'),
                     '-o', str(args.output / ('lib' + name + '.so'))],
        })
    inputs = [source_dir / name for name in LOCAL_SOURCES]
    inputs += [args.infinigen_include / name for name in UPSTREAM_HEADERS]
    return {
        'schema': 1, 'status': 'DRY_RUN' if args.dry_run else 'PREPARED',
        'source_directory': str(source_dir), 'infinigen_include': str(args.infinigen_include),
        'output_directory': str(args.output), 'arch': args.arch,
        'blocks': args.blocks, 'group_threads': 32, 'debug_guards': args.debug,
        'nvcc': nvcc, 'host_compiler': host or 'nvcc default from PATH',
        'commands': commands,
        'inputs': [{'path': str(path), 'sha256': file_sha256(path) if path.is_file() else None}
                   for path in inputs],
        'missing_inputs': [str(path) for path in inputs if not path.is_file()],
        'upstream_manifest_scope': 'Direct field_eval.cuh dependencies only; retain the full upstream revision separately.',
        'subprocess_environment': {
            'compiler_temporary_files': str(args.output / 'tmp'),
            'removed_override_variables': ['NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'NVCC_CCBIN'],
        },
        'gpu_execution': 'NOT_RUN', 'correctness_validation': 'NOT_RUN',
        'performance_validation': 'NOT_RUN',
    }


def execute(args, plan):
    require(sys.platform.startswith('linux'),
            'Actual builds require Linux or WSL and a GCC-compatible CUDA host compiler; use --dry-run elsewhere.')
    require(not plan['missing_inputs'],
            'Required sources/headers are missing:\n' + '\n'.join(plan['missing_inputs']))
    require(not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir())),
            'Use a new or empty output directory; previous build evidence is not overwritten.')
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'logs'
    temporary = args.output / 'tmp'
    logs.mkdir()
    temporary.mkdir()
    write_json(args.output / 'BUILD_PLAN.json', plan)
    status = {'status': 'RUNNING', 'started_utc': utc_now(), 'completed_artifacts': [],
              'gpu_execution': 'NOT_RUN', 'correctness_validation': 'NOT_RUN',
              'performance_validation': 'NOT_RUN', 'automatic_retries': 0}
    write_json(args.output / 'STATUS.json', status)
    environment = os.environ.copy()
    for key in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'NVCC_CCBIN'):
        environment.pop(key, None)
    environment.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    try:
        for item in plan['commands']:
            name = item['name']
            record = {'name': name, 'argv': item['argv'], 'cwd': str(args.output),
                      'started_utc': utc_now(), 'status': 'RUNNING'}
            command_path = logs / (name + '.command.json')
            write_json(command_path, record)
            print('Building ' + item['artifact'], flush=True)
            started = time.monotonic()
            code = None
            try:
                with (logs / (name + '.stdout.log')).open('xb') as stdout, (logs / (name + '.stderr.log')).open('xb') as stderr:
                    process = subprocess.run(item['argv'], cwd=args.output, env=environment,
                                             stdout=stdout, stderr=stderr, check=False)
                code = process.returncode
            finally:
                record.update(returncode=code, elapsed_seconds=time.monotonic() - started,
                              status='SUCCEEDED' if code == 0 else 'FAILED_OR_INTERRUPTED')
                write_json(command_path, record)
                with (args.output / 'COMMANDS.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
            require(code == 0, 'Compilation failed for ' + name + '; see ' + str(logs / (name + '.stderr.log')))
            artifact = args.output / item['artifact']
            require(artifact.is_file() and artifact.stat().st_size > 0,
                    'Compiler exited successfully without the expected library: ' + str(artifact))
            status['completed_artifacts'].append({
                'name': item['artifact'], 'sha256': file_sha256(artifact),
                'bytes': artifact.stat().st_size,
            })
            write_json(args.output / 'STATUS.json', status)
        for item in plan['inputs']:
            require(file_sha256(Path(item['path'])) == item['sha256'],
                    'Source/header changed during compilation: ' + item['path'])
        status['status'] = 'COMPLETED_SOURCE_BUILD_ONLY'
    except BaseException as error:
        status.update(status='FAILED_OR_INTERRUPTED', error=repr(error))
        raise
    finally:
        status['finished_utc'] = utc_now()
        write_json(args.output / 'STATUS.json', status)
    print('Build complete: ' + str(args.output), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc', help='nvcc executable path or PATH command (default: nvcc)')
    parser.add_argument('--host-compiler', help='optional GCC-compatible host compiler executable passed to nvcc -ccbin')
    parser.add_argument('--infinigen-include', type=Path, required=True,
                        help='inner infinigen directory containing terrain/ and infinigen_gpl/')
    parser.add_argument('--output', type=Path, required=True,
                        help='new or empty build directory outside this repository')
    parser.add_argument('--arch', default='sm_120', help='CUDA architecture (default: sm_120)')
    parser.add_argument('--blocks', nargs='+', type=int, choices=BLOCKS, default=list(BLOCKS),
                        help='L1_32 block sizes to build (default: 256 128 64)')
    parser.add_argument('--debug', action='store_true',
                        help='build solver *_debug.so with device guards; retains -O3 and does not add -G')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the JSON command plan without compiler execution or filesystem writes')
    args = parser.parse_args(argv)
    args.infinigen_include = args.infinigen_include.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    try:
        require(re.fullmatch(r'sm_[0-9]+[af]?', args.arch) is not None,
                '--arch must be an sm_ architecture name, for example sm_120 or sm_90a')
        require(len(args.blocks) == len(set(args.blocks)), '--blocks contains duplicate variants')
        require(args.output != REPOSITORY and not args.output.is_relative_to(REPOSITORY),
                '--output must be outside the source repository')
        require(args.output != Path(args.output.anchor), '--output cannot be a filesystem root')
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        return execute(args, plan)
    except (BuildError, OSError) as error:
        print('Build error: ' + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Build interrupted; completed logs and artifacts remain in the output directory.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())