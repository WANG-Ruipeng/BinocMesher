# Source contract mechanism and same-backend probe

Registered before experimental execution on 2026-09-07.
User specification: attachment `6a399d08-7473-49e1-bbee-c1d128f5ef70/pasted-text.txt`.
Frozen production revision: `54a7a7d235449858aac854a6da80f47d2021893b`.

## Scope and stopping rules

Run A, then B, then C0, once each. Do not start a later stage unless the
preceding stage's declared expectations all pass. Expected authorization and
contact refusals are negative controls, not campaign failures. An unexpected
exception, expectation mismatch, input-binding change or resource violation
stops the campaign. Do not repair and rerun a failed experiment in this campaign.
Development is parallel; all experiment execution is sequential and supervised.

The total work deadline is **2026-09-07 11:05:10 UTC**, 60 minutes after the
initial clock reading. Each experimental process is limited to 2 GiB RSS and
2 GiB address space. A/B/C0 wall caps are 120/180/300 seconds. Generated files,
including the small disposable cache, have a combined 1 GiB budget. The whole
repository must remain below 400,000,000,000 bytes. Initial repository size is
428,244,076 bytes (`du -sb`, before new experiment code).

No WMTK installation or migration, new scene, rendering, production edit,
contact-policy change, component-rule relaxation, full-schedule expansion or
certificate-reuse optimization is authorized by this first-stage protocol.
Do not publish complete mesh arrays. Retain compact receipts and any STOP.

## A: authorization mechanism

Ledger inputs are explicitly trusted and complete. Multiple owners in the
authorization fixture denote independently authorized sources, not implicitly
selected cache replicas. Test single-owner/full-owner positives, partial-owner
negative, equal-coordinate distinct identities and deterministic reindexing.
Positive authorization additionally requires the shared geometric policy to
pass. Compare production source resolution to independently implemented plain
grouping. Preserve every baseline input. No WMTK superiority or observer
completeness claim follows.

## B: joint contact, dependencies and publication

Use two synthetic square disks with coplanar collars, at z=0 and z=1, in
distinct element/identity namespaces. The inner square is [-1,1]^2 and outer
square [-2,2]^2. Obtain both individual certificates against the same complete
baseline before joint construction. Conflicting centers are z=3/4 and z=1/4;
safe centers are z=1/4 and z=3/4. All inputs are exact binary32 values.
The independent conflict witness is (1/3,0,1/2), with barycentric weights
(1/6,1/6,2/3) on the two right-side fans. The safe vertical separation is at
least 1/2. Verify safe and forbidden unions, input preservation, missing
component-membership and shared-boundary-support dependency controls, plus
late-query baseline-only publication. These controls do not isolate disjoint-boundary
raw-owner reuse or the complete dependency-graph construction.
Synthetic mechanisms are not real BEB1 events or new scene observations.

## C0: same-backend comparison

Use the frozen actual E2 source, cache and isolated native build listed in
`../c1_lite/artifacts/e2_runtime_20260906/REPRODUCE.md`.
Queries, in order: 103/5 (active), 104/5 (root), 102/5 (endpoint).
Both arms receive identical actual snapshots, complete ledgers, source-level
specification, center and contact policy. The plain arm must not call the
production resolver or consume its resolved face IDs. Both use the same array
executor and checker. Endpoint must pass its frozen contract and remain raw.
Report semantic agreement and stage costs. Logical work counts, measured
predicate calls and estimated/derived counts must be labeled separately.
Production resolver tracing is a separate untimed pass. Timed passes use no
tracing. Allow the same snapshot reuse to both arms; no cross-query certificate
reuse. A three-query probe is not a scaling or performance-superiority result.
Single-element public runtime measurements are not Forest-path measurements.

## Deliverables

Protocol, experiment code, per-stage compact JSON, supervised resource receipts
and a short final report. No Git commit/push or cleanup of unrelated files.
Whether completed or stopped, explicitly distinguish mechanism necessity,
same-backend equivalence and any still-unmeasured research advantage.
