#!/usr/bin/env python3
"""Snapshot the small existing C0 campaign and its source; never mutate inputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    repo, campaign, output = (x.resolve() for x in (args.repo, args.campaign, args.output))
    if output.exists():
        raise FileExistsError(output)
    validation = json.loads((campaign/'FULL_VALIDATION.json').read_text())
    if validation.get('pass') is not True:
        raise RuntimeError('C0 full validation has not passed')
    selected = ['binocmesher', 'experiments/source_splice', 'experiments/tv0_tv4', 'experiments/p1_p4']
    def git(*options):
        return subprocess.check_output(['git', '-C', str(repo), *options])
    tracked = git('ls-files', '-z', '--', *selected).decode().split('\0')
    paths = [(repo/name, 'source/'+name) for name in tracked if name and (repo/name).is_file()]
    paths += [(path, 'campaign/'+path.relative_to(campaign).as_posix()) for path in sorted(campaign.rglob('*')) if path.is_file()]
    paths.append((repo/'binocmesher/lib/core.so', 'runtime_at_freeze/core.so'))
    if any(path.is_symlink() for path, _ in paths):
        raise RuntimeError('Unexpected symlink in freeze inputs')
    total = sum(path.stat().st_size for path, _ in paths)
    if total > 100*1024**2:
        raise RuntimeError('Freeze input exceeds the 100 MiB storage budget')
    output.mkdir(parents=True)
    archive = output/'c0_snapshot.tar.gz'
    records = []
    with tarfile.open(archive, 'x:gz') as tar:
        for path, relative in paths:
            data = path.read_bytes()
            entry = tarfile.TarInfo(relative)
            entry.size = len(data)
            entry.mode = path.stat().st_mode & 0o777
            tar.addfile(entry, io.BytesIO(data))
            records.append({'archive_path':relative, 'source':str(path), 'bytes':len(data),
                            'sha256':hashlib.sha256(data).hexdigest()})
        diff = git('diff', '--binary', '--', *selected)
        entry = tarfile.TarInfo('provenance/current_source_diff.patch')
        entry.size = len(diff)
        tar.addfile(entry, io.BytesIO(diff))
    with tarfile.open(archive, 'r:gz') as tar:
        for record in records:
            data = tar.extractfile(record['archive_path']).read()
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise RuntimeError('Archive verification failed')
            if hashlib.sha256(Path(record['source']).read_bytes()).hexdigest() != record['sha256']:
                raise RuntimeError('Source changed during freeze')
    manifest = {'schema':'c1-lite-c0-freeze-v1', 'status':'PASS_C0_ARTIFACT_SNAPSHOT',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_checkout_head':git('rev-parse','HEAD').decode().strip(),
        'historical_validation_head':validation.get('head'),
        'scope':'Existing C0 full campaign plus current matching-checkout source. Core binary captured now, not asserted to be the historical test binary.',
        'not_included':'Environments, Infinigen, large Forest render/cache data, unrelated working-tree files.',
        'archive':str(archive), 'archive_bytes':archive.stat().st_size,
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'input_bytes':total, 'files':records}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__ == '__main__':
    main()
