"""Run only the source stage of one already-built registered segment."""
import argparse
from pathlib import Path
import json
from screen_contracts import load_protocol, read, require, file_sha
from screen_resources import verify_bindings
from scene_source import run_source_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'seal', 'build-report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--segment', required=True)
    args = parser.parse_args()
    protocol, segment = load_protocol(args.protocol, args.segment)
    seal = read(args.seal)
    require(seal['protocol_sha256'] == file_sha(args.protocol), 'PREREGISTRATION_SEAL_CHANGED')
    verify_bindings(seal['input_and_method_sha256'])
    build = read(args.build_report/'summary.json')
    require(build['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'
            and build['segment_id'] == args.segment, 'BUILD_NOT_COMPLETE_OR_WRONG_SEGMENT')
    complete = read(args.build_report/'worker_complete.json')
    camera = args.build_report/'camera_inputs.json'
    require(file_sha(camera) == complete['camera_inputs_sha256'], 'BUILD_CAMERA_CHANGED')
    result = run_source_stage(complete['cache'], camera, segment, args.output,
                             seconds=protocol['budgets']['source_per_scene_wall_seconds'])
    verify_bindings(seal['input_and_method_sha256'])
    summary = result['summary']
    print(json.dumps({k: summary.get(k) for k in ('status', 'segment_id', 'canonical_event_denominator',
                                                'root_population', 'source_status_counts', 'reason')}), flush=True)
    return 0 if summary['status'] == 'COMPLETE_SOURCE_STAGE_NOT_RUNTIME_ADMISSION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
