# E2 minimal production runtime — frozen scope

User authorized the E2 production runtime loop after the baseline-relative
repair. Keep the original E2 window [102/5,106/5], root 104/5, breakpoint 21,
four boundary SourceVIDs, two original faces, original anchors and raw mode.
Do not change the old C0/SSP1 API or activate a renderer/Forest experiment.

## Implementation contract

1. Add an explicitly enabled C++ identity ledger alongside ordinary slicing.
   It records actual ordered VID -> merged vertex index and every emitted raw
   owner -> final native face index. Disabled/unsupported instrumentation must
   not alter ordinary vertices, tags, faces, sorting or deduplication.
2. Generate complete ordinary baselines before proposing replacements. Resolve
   the boundary and both source faces from identities/owners, never nearest
   coordinates or a difference set inferred from the result. Suppress exactly
   all replicas of the independently determined source faces.
3. Reuse every old vertex/tag row. Replace the two source face rows with the
   first two oriented fan faces and append the other two; all other face rows
   remain bitwise identical. Append only the prescribed rounded center.
4. Check actual binary32 coordinates using exact rational predicates, including
   fan graph validity, interface links and contact with ALL retained triangles.
   Shared neighbours are checked, not skipped. A common actual vertex/edge may
   be the only allowed contact. Unknown is refusal, not success.
5. The initial public API is an atomic requested-schedule transaction. Precheck
   every requested frame before publishing any. Any invalid plan, unsupported
   mode, unknown geometry or failed witness returns every corresponding ordinary
   baseline, including root frames. Endpoints and outside times are unchanged.
   It is NOT an arbitrary-time/continuous-window admission; old window gates
   remain separate until actual full-window sufficient proofs exist.
6. Preserve original code/cache and build the new library in an isolated small
   directory. Test new-library disabled baseline against the preserved original
   library in separate processes. No overwrite of the installed WSL core.so.

## Fixed validation

- Original 33 rational times; all four breakpoints and adjacent representable
  physical-double values, including the two outside-window probes.
- Test exact-rational and ordinary physical-double entry points separately;
  report actual bit patterns and selectors, do not assume arrays coincide.
- OMP 1 and 8 in separate processes. One run each, no repeated hash campaign.
- Disabled, malformed plan, unsupported smooth, wrong/missing/partial owners,
  altered boundary/anchors, per-frame failure and state-reset negative tests.
- On acceptance: exact consumed-owner ledger, independent expected source rows,
  boundary reuse, all old vertex/tag rows and outside face rows unchanged.
- On refusal: same-entry/same-mode baseline shape, dtype and bytes unchanged
  for the entire requested schedule; no partially published meshes.

Arrays remain in RAM; compact JSON only. Temporary caches are ordinary copies,
not links, and removed automatically. New outputs capped at 20 MiB; build and
temporary data capped at 1 GiB, well below the 400 GB total storage ceiling.
If the all-time certificate remains unresolved, report the working production
schedule transaction distinctly; never promote sampled success to window PASS.
