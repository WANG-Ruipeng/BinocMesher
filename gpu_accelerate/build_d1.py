#!/usr/bin/env python3
"""Build the D1 split solver, original field bridge and B128 reference solver.

D1 is variant=1 in the preserved shared D4 library. This builder uses the same
strict floating-point flags as build.py and writes only outside the checkout.
It builds source; it does not run GPU validation or performance measurements.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

import build as reference_build


def build_plan(args):
    args.blocks = [128]
    plan = reference_build.build_plan(args)
    src = reference_build.PACKAGE / 'src'
    suffix = '_debug' if args.debug else ''
    artifact = 'libd4_solver' + suffix + '.so'
    reference = plan['commands'][1]
    argv = [str(src / 'd4_solver.cu') if arg == str(src / 'gpu_solver.cu') else
            str(args.output / artifact) if arg == str(args.output / reference['artifact']) else arg
            for arg in reference['argv']]
    plan['commands'].append({'name': 'd1_split' + suffix, 'artifact': artifact,
                             'variant': 1, 'field_block_threads': 128, 'argv': argv})
    extra = [src / name for name in ('d4_field_eval.cuh', 'd4_pipeline.cuh', 'd4_solver.cu')]
    extra += [reference_build.PACKAGE / 'include' / name
              for name in ('solver_api.h', 'd4_solver_api.h')]
    plan['inputs'] += [{'path': str(p), 'sha256': reference_build.file_sha256(p) if p.is_file() else None}
                       for p in extra]
    plan['missing_inputs'] += [str(p) for p in extra if not p.is_file()]
    plan['published_entry'] = 'D1 split: d4_solver_run(handle, 1, K, trace, diagnostic)'
    plan['shared_library'] = 'Unmodified D4 implementation also contains variants 2 and 3.'
    plan['cub_dependency'] = 'CUB headers from the selected existing CUDA toolkit; validated source used CUB 3.1.4.'
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--host-compiler')
    parser.add_argument('--infinigen-include', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new or empty directory outside checkout')
    parser.add_argument('--arch', default='sm_120')
    parser.add_argument('--debug', action='store_true', help='allocation guards, strict -O3, no CUDA -G')
    parser.add_argument('--dry-run', action='store_true', help='print commands; no compiler calls or writes')
    args = parser.parse_args(argv)
    args.infinigen_include = args.infinigen_include.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    try:
        reference_build.require(re.fullmatch(r'sm_[0-9]+[af]?', args.arch) is not None,
                                '--arch must name a CUDA sm_ architecture')
        reference_build.require(args.output != reference_build.REPOSITORY and
                                not args.output.is_relative_to(reference_build.REPOSITORY),
                                '--output must be outside the source repository')
        reference_build.require(args.output != Path(args.output.anchor), '--output cannot be a filesystem root')
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        return reference_build.execute(args, plan)
    except (reference_build.BuildError, OSError) as error:
        print('D1 build error: ' + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
