# E2 runtime freeze — 2026-09-06

This commit freezes the tested E2 implementation and its small research evidence
on branch `experiment/c1-lite-pilot`. Local annotated tag:
`e2-runtime-freeze-20260906`. No remote push is part of this freeze.

## Frozen conclusion

`PASS_PRODUCTION_REQUESTED_SCHEDULE_ONLY`: 33 exact-rational and 41
physical-double queries tested separately under OMP 1/8. Actual native
identity/owner tracing, checked Python source-face replacement, unchanged
exterior arrays and whole-requested-batch fallback passed. The existing WSL
installed library is unchanged; the new library was tested in an isolated build.

Continuous-window admission remains false. The declared-arithmetic error
envelope and robust perturbation results are conditional supplements, not a
replacement for the missing all-time runtime bindings. No new rendering,
SSIM improvement or six-scene superiority claim is frozen by this commit.

## Contents

- Opt-in production C++ observer/discard ABI and Python runtime/geometry gates.
- Self-contained C1-lite scripts, tests, protocols and compact historical
  evidence, including failed cases and the limited Forest registry pilot.
- Required earlier source-splice sidewall/continuous-window guard and tests.
- Small JSON/CSV/log records and two existing scientific plot pairs (PNG/SVG).
  No environments, mesh caches, build binaries, temporary files or large renders.

The historical [verification record](verification.json) remains an immutable
record of the experiment, including its then-current `git_committed=false`.
This freeze document records the later version-control action; do not edit old
scientific results to make them appear to have been generated after a commit.

## Integrity and reuse

[verification.json](verification.json) binds 24 source/result files by raw
SHA256. The scoped `.gitattributes` files disable automatic text conversion for
these frozen files so Git preserves the actual bytes that were tested; no code
or result was normalized merely to create this commit. The staged blob bytes
are checked against the recorded SHA256 values before committing.

The complete newly added research tree and changed source files are frozen by
the commit's Git object IDs. A hash-bound fresh checkout of the 24 explicit
files must match their recorded SHA256 values. C1-lite regression: 274 tests;
source-splice regression: 14 tests; standalone BEB1 closure test: PASS; focused
C++ strict build: PASS with the documented pre-existing signedness warnings.

Use [REPRODUCE.md](REPRODUCE.md) for build and experiment commands. The source
cache and Miniforge environment stay local and are not embedded in Git.
