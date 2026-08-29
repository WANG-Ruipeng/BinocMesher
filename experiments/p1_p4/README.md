# P1-P4 official-worktree smoke

This experiment has three independent layers:

1. `run_official_p1_smoke.py`: builds two complete official pipeline outputs,
   verifies byte-for-byte source-cache identity immediately before and after the
   instrumented observer call, and verifies identical final mesh bytes across
   baseline and instrumented runs. Cross-process raw cache hashes are recorded
   as non-gating diagnostics because upstream serializes unordered/padded C++
   object representations.
2. `run_core_fixture_smoke.py`: calls deterministic P2-P4 functions compiled
   into the patched official `core.so`.
3. `reference/run_p1_p4_reference.py`: source-local mathematical oracle for the
   same schedule, quotient, relative-boundary, and sampling contracts.

The final marker `PASS_P1_P4_OFFICIAL_WORKTREE_SMOKE` is emitted only when all
three layers pass. It does not claim that the production intervention has
already been enabled on the paper scenes.

The from-zero smoke was independently exercised on Linux x86-64 with GCC 11.4
and Python 3.13.15 at upstream commit
`8fae63707b6b128f1a4f9a35ec4d4a2bdc488e19`. The resulting official
`core.so` SHA-256 was
`5a92b758ee887ee492ad2d913187127bd9538d8fe78b179a0b2cee3485aff1ef`.
