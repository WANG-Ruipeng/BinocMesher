"""Preserve compact, hash-verified executed source files for a finished attempt."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(attempt, repository):
    attempt, repository = Path(attempt).resolve(), Path(repository).resolve()
    summary_path = attempt/'summary.json'
    summary = json.loads(summary_path.read_text())
    if not summary.get('private_cache_removed'):
        raise ValueError('Only archive a completed and cleaned-up campaign.')
    files = summary['executed_sources_sha256']
    planned = []
    for name, expected in files.items():
        source = Path(name).resolve()
        relative = source.relative_to(repository)
        if source.is_symlink() or not source.is_file() or digest(source) != expected:
            raise ValueError('Executed source is no longer the bound original: '+name)
        planned.append((source, relative, expected))
    if sum(p.stat().st_size for p, _, _ in planned) > 2*1024**2:
        raise ValueError('Compact source snapshot exceeds 2 MiB.')
    destination = attempt/'executed_source_snapshot'
    destination.mkdir(exist_ok=False)
    rows = []
    for source, relative, expected in planned:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError('Copied source hash mismatch.')
        rows.append({'path': relative.as_posix(), 'sha256': expected, 'bytes': target.stat().st_size})
    manifest = {'status': 'PASS_EXECUTED_SOURCE_SNAPSHOT',
                'attempt_summary_sha256': digest(summary_path), 'files': rows,
                'total_bytes': sum(row['bytes'] for row in rows)}
    with (destination/'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--repository', required=True)
    args = parser.parse_args()
    result = archive(args.attempt, args.repository)
    print(json.dumps({'status': result['status'], 'files': len(result['files']), 'bytes': result['total_bytes']}))
