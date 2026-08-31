# P1-P4 official-worktree smoke

This experiment has three independent layers:

1. `run_official_p1_smoke.py`: builds four complete official pipeline outputs
   for `(BINOC_PROVENANCE_V2, BINOC_EVENT_MODE) = 00/01/10/11`. It requires
   identical final meshes and canonical HP/HV cache semantics in every mode.
   Raw cross-process cache hashes remain diagnostic because upstream serializes
   unordered/padded C++ object representations.
2. `run_core_fixture_smoke.py`: calls deterministic P2-P4 functions compiled
   into the patched official `core.so`.
3. `reference/run_p1_p4_reference.py`: source-local mathematical oracle for the
   same schedule, quotient, relative-boundary, and sampling contracts.

The final marker `PASS_P1_P4_OFFICIAL_WORKTREE_SMOKE` is emitted only when all
three layers pass. It does not claim that the production intervention has
already been enabled on the paper scenes.

Cache reuse is mode/schema explicit. `slicing_preprocess.finish` is accepted
only with a matching `slicing_preprocess.manifest.json`; legacy or mismatched
caches fail closed and must be rebuilt in a fresh path. Processed cache and
registry files are published from temporary files only after checked close.

The pinned cloud gate is `.github/workflows/p1-p4-proof-ready.yml`. Locally on
the matching Linux environment, the equivalent full command is:

```bash
bash experiments/p1_p4/run_all.sh --repo "$PWD" \
  --output /tmp/binoc-p1-p4 --install-deps --full-official-smoke
```

The from-zero smoke was independently exercised on Linux x86-64 with GCC 11.4
and Python 3.13.15 at upstream commit
`8fae63707b6b128f1a4f9a35ec4d4a2bdc488e19`. The resulting official
`core.so` SHA-256 was
`5a92b758ee887ee492ad2d913187127bd9538d8fe78b179a0b2cee3485aff1ef`.
