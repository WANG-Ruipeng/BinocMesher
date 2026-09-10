# Source-contract mechanism probe 鈥?2026-09-07

Outcome: **A, B and C0 passed their frozen expectations and resource guards.**
No unexpected failure or budget stop occurred. Each stage ran once. No WMTK,
rendering, new scene, production edit or certificate-reuse optimization was run.

The result supports source-authorization and joint-contact mechanisms, but does
**not** establish an advantage over the independent ordinary reference.

## Evidence and scope

- [Protocol](PROTOCOL.md), frozen before execution.
- [A receipt](artifacts/run_20260907/A.json): five synthetic families / seven cases.
- [B receipt](artifacts/run_20260907/B.json): synthetic joint-contact and publication controls.
- [C0 receipt](artifacts/run_20260907/C0.json): three native E2 queries.
- [Compact summary](artifacts/run_20260907/summary.json).
- Each stage has a separate `*_supervisor.json` resource receipt and log.
- `campaign_bindings.json` binds the unchanged implementations across all stages.

Frozen production revision: `54a7a7d235449858aac854a6da80f47d2021893b`.

## A 鈥?source authorization

Both resolvers agreed on all seven cases: five accepted proposals and two
expected authorization refusals. Every positive also passed the shared actual
geometry/interface checker and produced byte-equal arrays. All inputs remained
unchanged. Two fixed reindexings preserved source-labelled output semantics.

The decisive matched pair holds the mesh and selected-owner request fixed.
Adding an independently authorized owner to one retired face changes both
resolvers from acceptance to refusal. The underlying geometric control still
passes. Complete provenance is therefore relevant to this authorization task;
mesh coordinates and triangles alone do not determine permission.

Owners here are independent authorization objects, not cache replicas. Ledger
completeness/correctness is a shared premise, not a tested emitter theorem.
The coincident-identity control uses an unused vertex and does not certify
overlapping occupied sheets. Positive outputs are proposals, not publication.
The ordinary reference enforces the same contract: this is not a novelty or
WMTK-incapability result.

## B 鈥?individual safety does not imply joint safety

Both fixtures use the same two baseline square disks and collars, at z=0 and
z=1. All four single-event certificates pass, each checking all 18 retained
faces: 10 strict AABB separations and 8 relative-edge certificates.

- Crossing centers z=3/4 and z=1/4: the union is rejected on its first exact
  replacement/replacement triangle test. An independent witness reconstructed
  from actual plan triangles is `(1/3, 0, 1/2)`, with strictly positive weights
  `(1/6, 1/6, 2/3)` in both right-side triangles.
- Safe centers z=1/4 and z=3/4: the union passes, consumes 4 faces and 6 raw
  owners once, and is unchanged by reversal of the two input events. All four
  emitted fan sectors bind the global gap `1 - phi/2 >= 1/2`, where
  `phi = 1 - max(abs(x), abs(y))` on the square.
- The safe pair needs 1 support-AABB event-pair exclusion, which excludes all
  32 triangle relations; **0 exact triangle-pair calls** are made. The crossing
  pair terminates on 1 exact call. These are different timing populations.
- Missing declared atomic membership and shared-boundary support are refused
  without changing the baseline. This does not isolate disjoint-boundary
  owner reuse or certify a complete/minimal dependency graph.
- With a synthetic slicer and the real schedule runtime, an injected identity
  failure at the second query discards the first proposal. All three outputs
  are baseline arrays; no partial edited result is published.

The low-level predicate correctly leaves
`new_contact_relative_to_baseline_proven=false`: it does not compare old
geometry. The independent fixture proof supplies the disjoint old layers and
intersecting new layers. We do not relabel the predicate's scope.

These are mechanism fixtures, not native BEB1 scene events or evidence of
visible improvement. Publication atomicity is reported separately from contact.

## C0 鈥?independent frontend, identical checker and array backend

Internal rational queries are **not seconds**. All three actual baselines have
1,831 vertices, 3,820 faces and 5,262 raw-owner rows. The native identity observer
preserves the ordinary baseline. Both arms independently resolve the same
support and obtain equal output digests/checker evidence; active outputs also
pass the frozen independent E2 identity/array validator.

Five alternating-order timing pairs per query, milliseconds, median:

| Query / role | Current resolver + shared backend | Ordinary resolver + shared backend |
|---|---:|---:|
| 103/5, active | 33.746 | 33.748 |
| 104/5, root | 33.017 | 33.756 |
| 102/5, endpoint / unchanged baseline | 11.737 | 11.647 |

Timings exclude native slicing, input binding, validation, output hashing,
tracing and serialization. Indices are rebuilt cold on the same warm immutable
snapshot in both arms; neither arm reuses a cross-query certificate. Separate
stage medians need not sum to the median total. The ordinary resolver retains
no-op `_count(None, ...)` function-call overhead even when counters are disabled;
sub-millisecond resolver differences are not evidence of algorithmic superiority.

For each active query, both arms perform the same principal work:

| Work per active query | Current | Ordinary |
|---|---:|---:|
| Vertex-ledger rows | 1,831 | 1,831 |
| Owner-ledger rows | 5,262 | 5,262 |
| Resolver face-row reads | 5,264 | 5,264 |
| Indices built / reused | 3 / 0 | 3 / 0 |
| Interface face slots | 3,820 | 3,820 |
| Retained AABB relations | 3,818 | 3,818 |
| Exact fan-to-retained calls | 28 | 28 |
| Explicit array payload bytes copied | 143,048 | 143,048 |
| Event pairs / inherited certificate relations | 0 / 0 | 0 / 0 |

Current face-row reads are derived from the frozen successful loop; other
current counts use a separate untimed trace. Ordinary resolver counts are
explicit logical counters. Exact fan-to-retained calls are not individual
determinant predicates or four triangle-pair calls. Copied bytes exclude Python
objects and allocation bookkeeping. The endpoint has no fan/contact scan or
array copy. Full definitions and all five timing samples are in C0.json.

There are two active agreements, one unchanged endpoint agreement and zero
decision mismatches. There are **no native rejection timing samples**; this is
not an accepted/rejected-population performance study. Equal geometry work is
expected by construction because the entire checker is shared: this probe isolates
source-resolution cost, not alternative joint-graph organization. Nor is it a Forest batch
path, scaling, complete-schedule publication or continuous-time experiment.

## Interpretation and stopping point

Current and ordinary implementations have equal principal work and nearby
timings on this probe. No substantive performance advantage has been established.
At active queries source resolution costs about 11.5鈥?2.4 ms, shared geometry
about 21 ms, and array assembly about 0.04 ms. Changing the array executor is not
supported as the measured bottleneck of this single-element path.

Joint certificate inheritance remains an unmeasured hypothesis, not a demonstrated
optimization: C0 has one event per query, and B's safe pair already avoids all
exact contact calls. This campaign therefore stops at the registered A/B/C0
boundary, without optimizing or expanding to WMTK, Forest or more rendering.
Earlier original-camera visual negative results remain unaffected.

## Resources and retained files

| Stage | Worker seconds | Supervised wall seconds | Peak worker RSS bytes |
|---|---:|---:|---:|
| A | 1.259 | 38.696 | 93,495,296 |
| B | 1.301 | 38.231 | 96,624,640 |
| C0 | 46.182 | 82.943 | 104,321,024 |

Supervised time includes repository-budget scans and is not algorithm latency.
All stages are below their 120/180/300 s caps and 2 GiB process memory cap.
The maximum observed worker RSS is about 99.5 MiB. The final C0 repository
snapshot is 428,514,573 bytes (~0.429 GB), before this small summary was added;
the experiment directory was 270,497 bytes. Both are far below declared limits.

The only disposable cache copied was 4,255,152 bytes. Its copy changed only
`log.txt`, was removed, and is reproducible from the unchanged original. No
environment, original cache, old experiment or production file was deleted.
No meshes, frames or large arrays were retained; no Git commit/push was made.

`supervise.py A`, then `B`, then `C0` are the original one-shot entrypoints using
the existing Miniforge `binoc-exp` Python under WSL. Completed receipts and the
absolute deadline deliberately prevent reusing this campaign. A future run
requires a separately approved budget/run identity; do not erase these receipts
to bypass the guard.
