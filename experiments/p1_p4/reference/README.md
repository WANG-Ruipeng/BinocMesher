# Reference P1-P4 smoke

This directory contains the source-local Python oracle used for the quick P1-P4
contract tests. It is **not** the production BinocMesher intervention. The
production smoke is `../run_official_p1_smoke.py` plus the fixture functions
compiled into the patched official `core.so`.

Run:

```bash
BINOC_P1_P4_REFERENCE_OUT=/tmp/binoc-reference \
python run_p1_p4_reference.py
python validate_reference.py /tmp/binoc-reference
```
