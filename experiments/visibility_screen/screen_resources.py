"""Run-owned bounded storage and source seals; never deletes user inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import tempfile
import time

from screen_contracts import ROOT, C1, HERE, read, file_sha, write_new, require, load_protocol
from forest_native_campaign import PrivateCache, cache_inventory, content_inventory


class PrivateSceneCache(PrivateCache):
    """Same validated-copy/append-only verification as the frozen campaign.

    This experiment explicitly budgets a 4 GiB source cache. Only the constructor
    allowance differs; inherited verification and exact-target cleanup remain.
    """
    def __init__(self, original, *, parent='/home/warpwang/binoc-runs'):
        self.original = Path(original).resolve()
        self.before = cache_inventory(self.original)
        require(sum(row['bytes'] for row in self.before.values()) <= 4*1024**3,
                'PREREGISTERED_CACHE_COPY_EXCEEDS_4_GIB')
        self.parent = Path(parent).resolve()
        require(self.parent.is_dir(), 'PRIVATE_CACHE_PARENT_MISSING')
        self.temporary = Path(tempfile.mkdtemp(prefix='forest-native-20260906-', dir=self.parent)).resolve()
        self.path = self.temporary/'cache'
        self.closed = False
        require(self.temporary.parent == self.parent, 'PRIVATE_TARGET_OUTSIDE_DECLARED_PARENT')
        try:
            shutil.copytree(self.original, self.path)
            self.copy_before = cache_inventory(self.path)
            require(content_inventory(self.before) == content_inventory(self.copy_before),
                    'PRIVATE_CACHE_CONTENT_DIFFERS')
        except BaseException:
            self.remove()
            raise


def source_seal():
    folders = [C1, HERE.parent/'source_splice', HERE.parent/'tv0_tv4', ROOT/'binocmesher']
    paths = {p.resolve() for folder in folders for p in folder.glob('*.py')}
    paths.update(p.resolve() for p in (ROOT/'binocmesher/source').rglob('*') if p.is_file())
    return {str(p): file_sha(p) for p in sorted(paths)}


def verify_bindings(bindings):
    for path, expected in bindings.items():
        require(file_sha(path) == expected, 'FROZEN_INPUT_OR_IMPLEMENTATION_CHANGED:'+path)


def seal(protocol, output):
    document = load_protocol(protocol)
    bindings = source_seal()
    previous = (HERE/document['previous_negative_result']).resolve()
    bindings[str(previous)] = file_sha(previous)
    for name in ('observer', 'ordinary'):
        path = Path(document['frozen_native'][name+'_build_repo'])/'binocmesher/lib/core.so'
        require(file_sha(path) == document['frozen_native'][name+'_sha256'], 'FROZEN_NATIVE_CHANGED')
        bindings[str(path)] = file_sha(path)
    bindings[str(Path(protocol).resolve())] = file_sha(protocol)
    result = {'schema': 'visibility-screen-preregistration-seal-v1',
              'status': 'SEALED_BEFORE_NEW_SEGMENT_EXPERIMENTS',
              'sealed_at_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
              'protocol_sha256': file_sha(protocol), 'input_and_method_sha256': bindings,
              'new_scene_results_read': False, 'old_negative_result_preserved': True,
              'git_write_performed': False,
              'scope': 'File-level authoritative evidence seal; no assertion that untracked files have been committed.'}
    write_new(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = seal(args.protocol, args.output)
    print(result['status'], result['protocol_sha256'], len(result['input_and_method_sha256']))
