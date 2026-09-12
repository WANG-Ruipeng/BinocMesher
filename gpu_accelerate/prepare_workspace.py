"""Copy only code into a fresh local replay workspace. Never copies scene data."""
import argparse
import json
import shutil
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    args=ap.parse_args()
    out=args.workspace.expanduser().resolve()
    if out.exists() or out.is_relative_to(PACKAGE.parent):
        ap.error('workspace must be a fresh path outside the repository')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('src','include'):
        shutil.copytree(PACKAGE/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('README.md','BUILD.md','API.md','HARNESS.md','requirements-checks.txt'):
        shutil.copy2(PACKAGE/name,out/name)
    for name in ('build','configs','inputs','logs','results','runs','cache','tmp'):
        (out/name).mkdir()
    (out/'.gitignore').write_text('*\n')
    (out/'STATUS.json').write_text(json.dumps({'status':'PREPARED_CODE_ONLY','inputs':'NOT_PROVIDED','GPU':'NOT_RUN'},indent=2)+'\n')
    (out/'configs/PERFORMANCE_AUTHORIZATION.json').write_text(json.dumps({'allowed':False},indent=2)+'\n')
    print(json.dumps({'workspace':str(out),'scene_data_copied':False,'GPU':'NOT_RUN'}))

if __name__=='__main__':
    main()
