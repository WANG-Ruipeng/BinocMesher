"""Bounded private-cache and byte-exact array receipts for Forest experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024*1024):
            h.update(data)
    return h.hexdigest()


def cache_inventory(path):
    path = Path(path).resolve()
    result = {}
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            raise ValueError('Symlink source cache is not supported.')
        if item.is_file():
            stat = item.stat()
            result[item.relative_to(path).as_posix()] = {'bytes': stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'sha256': file_sha(item)}
    if not result:
        raise ValueError('Empty source cache.')
    return result


def content_inventory(inventory):
    return {key: {'bytes': value['bytes'], 'sha256': value['sha256']} for key, value in inventory.items()}


class PrivateCache:
    """Exactly one validated private directory, original cache remains read-only."""
    def __init__(self, original, *, parent='/home/warpwang/binoc-runs'):
        self.original = Path(original).resolve()
        self.before = cache_inventory(self.original)
        if sum(x['bytes'] for x in self.before.values()) > 2*1024**3:
            raise ValueError('Private copy exceeds declared 2 GiB budget.')
        self.parent = Path(parent).resolve()
        self.temporary = Path(tempfile.mkdtemp(prefix='forest-native-20260906-', dir=self.parent)).resolve()
        self.path = self.temporary/'cache'
        if self.temporary.parent != self.parent or not self.temporary.name.startswith('forest-native-20260906-'):
            raise ValueError('Unexpected private-cache target.')
        self.closed = False
        try:
            shutil.copytree(self.original, self.path)
            self.copy_before = cache_inventory(self.path)
            if content_inventory(self.before) != content_inventory(self.copy_before):
                raise ValueError('Private copy contents differ from the original cache.')
        except Exception:
            self.remove()
            raise

    def verify(self, query_count):
        after = cache_inventory(self.original)
        if after != self.before:
            raise ValueError('Original cache changed during the native campaign.')
        copied = cache_inventory(self.path)
        allowed = set(self.copy_before) | {'log.txt'}
        if not set(self.copy_before) <= set(copied) or set(copied)-allowed:
            raise ValueError('Native slicing changed the private cache file set.')
        for name, value in self.copy_before.items():
            if name != 'log.txt' and copied[name] != value:
                raise ValueError('Native slicing modified a non-log private cache input: '+name)
        old = self.copy_before.get('log.txt', {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()})
        new = copied.get('log.txt', old)
        appended = new['bytes']-old['bytes']
        if not 0 <= appended <= min(4*1024**2, 128*1024*max(query_count, 1)):
            raise ValueError('Private log append exceeded the declared bound.')
        digest = hashlib.sha256()
        if 'log.txt' in copied:
            with (self.path/'log.txt').open('rb') as stream:
                remaining = old['bytes']
                while remaining:
                    data = stream.read(min(1024*1024, remaining))
                    if not data:
                        raise ValueError('Private log truncated.')
                    digest.update(data); remaining -= len(data)
        if digest.hexdigest() != old['sha256']:
            raise ValueError('Private log prefix was overwritten.')
        return {'status': 'PASS', 'original_files_unchanged': len(after),
            'original_bytes': sum(x['bytes'] for x in after.values()),
            'private_nonlog_inputs_unchanged': True, 'private_log_appended_bytes': appended,
            'original_input_content_sha256': hashlib.sha256(json.dumps(content_inventory(after), sort_keys=True).encode()).hexdigest()}

    def remove(self):
        if self.closed:
            return
        target = self.temporary.resolve()
        if target.parent != self.parent or not target.name.startswith('forest-native-20260906-') or target == self.original or target in self.original.parents:
            raise ValueError('Refusing cleanup outside the exact private campaign directory.')
        shutil.rmtree(target)
        self.closed = True


def array_sha(array):
    a = np.asarray(array)
    if not a.flags.c_contiguous or a.dtype.hasobject:
        raise ValueError('Expected a contiguous numeric output array.')
    h = hashlib.sha256(json.dumps({'dtype': a.dtype.str, 'shape': list(a.shape)}, sort_keys=True).encode())
    if a.size:
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def mesh_receipt(meshes):
    return [{'vertices': len(v), 'faces': len(f),
             'sha256': {name: array_sha(a) for name, a in zip(('vertices', 'faces', 'tags'), (v, f, tags))}}
            for v, f, tags in meshes]


def apply_and_verify(meshes, plan):
    """Actually construct the patch once; do not return/publish mesh arrays."""
    element = int(plan['element'])
    v, f, tags = meshes[element]
    removed = tuple(sorted(map(int, plan['removed_face_rows'])))
    cycle = tuple(map(int, plan['boundary_actual_ids']))
    center = np.asarray(plan['center'], dtype=v.dtype)
    if len(removed) != 2 or removed[0] == removed[1] or len(cycle) != 4 or len(set(cycle)) != 4:
        raise ValueError('Malformed compact patch plan.')
    if any(i < 0 or i >= len(f) for i in removed) or any(i < 0 or i >= len(v) for i in cycle):
        raise ValueError('Compact patch plan index outside the actual baseline.')
    result_v = np.concatenate((v, center[None, :]))
    result_tags = np.concatenate((tags, np.asarray([1], dtype=tags.dtype)))
    fan = np.asarray([(cycle[i], cycle[(i+1) % 4], len(v)) for i in range(4)], dtype=f.dtype)
    result_f = np.concatenate((f, fan[2:]))
    result_f[list(removed)] = fan[:2]
    if array_sha(result_v[:-1]) != array_sha(v) or array_sha(result_tags[:-1]) != array_sha(tags):
        raise ValueError('Actual proposal changed old vertices or tags.')
    left = 0
    for right in (*removed, len(f)):
        if array_sha(result_f[left:right]) != array_sha(f[left:right]):
            raise ValueError('Actual proposal changed retained exterior face rows.')
        left = right+1
    if not np.array_equal(result_f[list(removed)], fan[:2]) or not np.array_equal(result_f[len(f):], fan[2:]):
        raise ValueError('Actual four-triangle patch construction differs from compact plan.')
    receipt = {'status': 'ACTUAL_ARRAYS_CONSTRUCTED_NOT_PUBLISHED',
        'element': element, 'vertices_before': len(v), 'faces_before': len(f),
        'vertices_after': len(result_v), 'faces_after': len(result_f),
        'old_vertices_and_tags_byte_identical': True, 'all_retained_face_rows_byte_identical': True,
        'other_elements_unchanged_by_object_identity': True,
        'output': mesh_receipt([(result_v, result_f, result_tags)])[0],
        'array_bytes_transient': result_v.nbytes+result_f.nbytes+result_tags.nbytes}
    return receipt
