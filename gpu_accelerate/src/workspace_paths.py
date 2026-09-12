"""Local-only inputs, outputs and installed tools; no GPU initialization."""
import os
import shutil
import sys
from pathlib import Path
SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("BM_GPU_WORKSPACE", str(SOURCE_ROOT))).expanduser().resolve()
INFINIGEN_ROOT = Path(os.environ.get("BM_INFINIGEN_ROOT", str(ROOT / "upstream/infinigen"))).expanduser().resolve()
REFERENCE_ROOT = Path(os.environ.get("BM_NATIVE_REFERENCE_ROOT", str(ROOT))).expanduser().resolve()
PYTHON = os.environ.get("BM_PYTHON", sys.executable)

def tool(name):
    override = os.environ.get("BM_" + name.upper().replace("-", "_"))
    return override or shutil.which(name) or name
