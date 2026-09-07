# Forest: atomic requested sequence and original-camera visibility

The all-event support graph, component admission, and actual combined sequence
are complete. The original-camera visibility screen is also complete, but it
found **no directly visible candidate source support and no visible replacement
pixels** in the tested natural-frame windows. This is a positive result for
finite-schedule composition and a negative sample-selection result for visual
significance. It is not evidence of improved image quality.

The experiment implements steps 1–4 of the supplied next-stage plan. The
visible-component selection, post-displacement pilot, and subsequent image-quality
comparisons are paused because there is no qualifying visible sample here.

## Fixed scope

- Forest's existing 64-frame, 24 FPS source cache, five opaque elements, 960×540.
- The geometry was generated with the existing **6 px LOD threshold**, not the
  paper's 3 px final configuration. This is pre-surface-displacement raw geometry.
- All 131 canonical events at roots `3/2` and `5/2` remain in the denominator,
  including the 100 independently rejected events. The admission/contact policy
  was not relaxed.
- All 64 natural queries and two additional exact-root diagnostics were actually
  constructed for both OMP 1 and OMP 8. Each query has one combined five-element
  scene output; this is not sequential editing of previously modified baselines.
- This certifies the fixed requested queries and current selector domain, not an
  all-real-time theorem, an event-complete method, or final Infinigen rendering.

The [fixed protocol](../../FOREST_COMPONENT_SEQUENCE_PROTOCOL_20260906.md) and
[structural manifest](structural_coherence.json) define the experiment and evidence
chain. The latter verifies 159 hash-bound JSON artifacts; it assembles earlier
native/geometry evidence rather than independently rerunning a geometry solver.

## Joint admission: 25 events in 20 components

| Exact root | All events | Support components | Admitted components | Fallback components | Jointly admitted events |
| --- | ---: | ---: | ---: | ---: | ---: |
| `3/2` | 66 | 39 | 9 | 30 | 12 |
| `5/2` | 65 | 40 | 11 | 29 | 13 |
| Total | 131 | 79 | 20 | 59 | 25 |

Joint event admission is **25/131 = 19.0840%**; component admission is
**20/79 = 25.3165%**. Both roots have a resolved output and a nonempty admitted
subset (`2/2`), but neither root admits every event. Do not call this 100% event
or full-root-event coverage.

The previous **31/131 independent admission** remains a valid, different
experiment. Six of those events now fall back because their support components
include rejected members. Thus 106 events fall back: the original 100 plus six
component-propagated fallbacks. There are zero unknown actual support rows.
Fallback is a policy/domain decision, not a claim that 106 events contain errors.

The graph includes all 131 nodes and actual support evidence at 18 queries
(1,179 event-query footprints). It is a conservative dependency supergraph:
an edge need not be a collision, and the resulting components are not claimed
to be minimal. Original owner, source-face, boundary/interface identity,
retained-face and actual spatial dependencies are included. For the one event
without a source contract, the complete current candidate V-halo is reserved as
baseline-only support; no unknown future replacement is invented or bounded.
Actual modified `P ∪ Q` bounds and unchanged retained-star read dependencies have
different roles. Ideal cell boxes do not exclude actual binary32 interactions.

See [component certification](component01/summary.json),
[support graph and decisions](component01/support_graph.json), and the
[independent component audit](component01_independent_audit.json).
The supplemental audit rechecked 279 local geometries and 2,079 event pairs
(66,528 triangle pairs, including 1,344 exact-contact tests after broad-phase
exclusion). Five admitted components contain two events; fifteen are singletons.
Shared-owner/face/boundary cases remain unsupported; no general overlap solver
was introduced.

## Actual sequence and deterministic composition

- **16/64 natural frames change geometrically:** one-based frames 21–28 and 37–44.
- There are **200 actual joint event-frame replacements**, not 248 independent
  what-if interventions. Each of the 25 admitted events affects eight frames.
- The remaining 48 natural outputs preserve their baseline arrays exactly.
- All 18 active/root queries replay the complete relevant root population's
  source supports, not only the 25 admitted events, and match the graph evidence.
- Components commit all-or-none. Source faces/owners are consumed once; centers
  and fan faces are assigned canonical per-element identities. Rejected supports
  and retained arrays are preserved under the recorded contract.
- Local/interface checks, replacement–replacement checks and binding-verified
  inherited replacement–retained certificates pass under the fixed policy.
- **OMP 1 and OMP 8 agree on all 66 outputs**, canonical replacement records,
  owner consumption, identity mappings and complete support replays.
- The two root diagnostics run after all 64 natural frames, also checking a return
  to earlier query times without cross-frame mutation of the baseline.

Evidence: [OMP 1](sequence02_omp1/summary.json),
[OMP 8](sequence03_omp8/summary.json),
[independent comparison](omp1_omp8_comparison.json).
The successful marker is `PASS_FOREST_ATOMIC_REQUESTED_SEQUENCE`; the assembled
marker is `PASS_REQUESTED_SCHEDULE_STRUCTURAL_COHERENCE`.

## Original-camera visibility: zero direct image differences

All 16 modified natural frames were rasterized from the **original rendering
camera**, comparing ordinary baseline with the actual combined output. All 131
candidate source supports, not just admitted events, were tested at their eight
natural queries. The separate
[camera provenance](../../FOREST_VISIBILITY_CAMERA_PROVENANCE_20260906.md)
checks the saved Blender camera poses and intrinsics. Rendering uses `fx=fy=1500`;
the relaxed LOD intrinsics are retained unchanged as geometry provenance and are
not substituted for the rendering camera.

| Population | Event-frame queries | Strictly outside frustum: source support | Isolated pixels but occluded/depth tie | Visible source events |
| --- | ---: | ---: | ---: | ---: |
| All 131, including one candidate-halo proxy | 1,048 | 1,008 | 40 | 0 |
| 130 with a fixed source contract | 1,040 | 1,000 | 40 | 0 |
| 25 jointly admitted | 200 | 200 | 0 | 0 |

126 of all 131 events have source supports outside the frustum at every tested
natural frame. The remaining five account for the 40 isolated pixel-covering
queries, but their faces are not selected in the full-scene depth buffer.
The residual category of unclassified no-pixel-center cases is zero; these
results do not require an unverified blanket “subpixel” explanation.

The [independent projection audit](visibility01_independent_projection_audit.json)
recomputes strict same-clip-plane exclusion using exact rational arithmetic on
the saved binary64 camera and actual source coordinates. It uses the continuous
image rectangle, not a reduced pixel-center rectangle. It also checks the
world-coordinate path: native XYZ passes through the reader and ordinary wrapper
without an omitted spatial transform. Its independent geometric proof concerns
the **original source triangles**, not a new continuous-screen certificate for
every replacement fan. Full-scene occlusion/ties and replacement visibility are
inherited from the hash-bound rasterization evidence, not rerendered by this audit.

Actual rasterization additionally finds:

- Zero visible replacement pixels for all 25 jointly admitted events.
- **Identical depth, world-space flat normal, mask, element-ID, face-ID, and
  component-ID buffer hashes in every one of the 16 frames.**
- Zero depth/normal/mask-boundary changes, including unthresholded buffer checks.
- The visible-support admission fraction is **undefined `0/0` (`null`)**, not 0%.
  The compiled-source-only denominator gives the same undefined result.
- Pixel/impact-weighted artifact coverage is not measured: rejected replacement
  images are undefined, and visible source area is not an artifact-impact oracle.

This is pinhole, pixel-center, no-AA, pre-displacement geometric triage. It does
not establish equivalence of smooth-shaded/material RGB, shadows, motion blur,
or the final displaced scene. Component labels include fallback candidates and
are not isolated component-effect estimates. Numerical depth/normal thresholds
are not human-perception thresholds. No SSIM or quality-improvement claim follows.

See [visibility summary](visibility01/summary.json) and an actual
[normal comparison](visibility01/images/frame_0021-normal_pair.png),
[depth comparison](visibility01/images/frame_0021-depth_pair.png), and
[component image](visibility01/images/frame_0021-component_ids.png).
The 64 retained PNGs are scientific raster outputs, not generated illustrations.

## Measured costs, not production latency

| Measurement | OMP 1 | OMP 8 |
| --- | ---: | ---: |
| Entire verified-sequence harness wall time | 481.01 s | 353.29 s |
| Native ordinary/observer slicing subtotal | 233.63 s | 267.37 s |
| Additional frozen-baseline reference audit | 161.17 s | approximately 0 s |
| Resolve plus complete interface audit | 14.02 s | 14.02 s |
| Verified union, including exact rechecks and full-array audit | 4.56 s | 4.50 s |
| Baseline hashing subtotal | 1.04 s | 1.03 s |
| Peak RSS | 1,095,364,608 bytes | 1,091,469,312 bytes |

The OMP 1 run performs **44 additional frozen ordinary reference queries**;
OMP 8 reuses their evidence. Native call counts are therefore 110 and 66.
Their total wall times are **not a fair OMP speedup comparison**. CPU time,
initialization, supplemental certificate verification and cleanup have their
own scope; the rows above are not an exhaustive partition of wall time.

Separately, the new source-support inventory took 74.06 s, component certification
100.97 s, and 16-frame visibility triage 97.02 s. Component certification's peak
RSS was 1,115,623,424 bytes (about 1.04 GiB). These are incremental stages building
on the frozen per-event campaign, not a fresh end-to-end certification total.
The verified-union subtotal still includes audit work; pure production playback
overhead and `T_ours - T_baseline` remain unmeasured.

## Tests, preservation and storage

The [initial regression report](regression_tests.json) records 642 passing unit
tests plus the standalone critical-BEB1 combinatorial regression; later added
manifest/projection tests are included in the separate
[final regression report](regression_tests_final.json). The actual native,
geometry, sequence and visibility evidence above is separate from synthetic
unit-test success.

`sequence01_omp1` is retained as failed-startup evidence: a Python path-join typo
stopped it before native slicing or private cache construction. The experimental
runner was corrected and regression-tested, then rerun in fresh `sequence02_omp1`.
It is not an admission rejection or an overwritten successful run.

All 35 original cache files were verified unchanged after each native campaign.
Private cache copies were removed. No 64-frame mesh dump is retained: actual
arrays were constructed and audited in memory, with compact receipts/hashes saved.
This turn's eight temporary patch files were also removed; reusable experiment
scripts and failure evidence remain. Frozen production code/binaries, prior
admission results, environments, and the old `build_complete=false` record were
preserved. No Git commit or push was performed in this experiment.

The final storage scan counted approximately **40.41 GB** for the Windows
repository plus WSL VHDX file and migration backup combined, without double
counting files inside WSL. The new sequence artifacts are about **28.2 MB** before
the final small test report/README. This is well below the 400 GB limit.

## Next gate and paper claim

A defensible new claim is:

> On the fixed 64-frame Forest pre-displacement source cache, all 131 canonical
> events formed 79 conservative support components. Twenty components containing
> 25 events were jointly admitted and composed into actual outputs at every
> requested timestamp. The combined sequence modified 16 natural-frame meshes
> and agreed across OMP 1 and 8. None of the admitted replacements was directly
> visible in the tested original-camera geometric buffers.

Do not turn “16 modified meshes” into “16 visually improved frames.”

The next useful experiment is visibility-first screening of another genuine
paper-scene/time segment, with selection criteria fixed before quality results.
Increasing admission in this same candidate population does not by itself solve
its zero visible-source denominator. Do not widen the rendering FOV or replace
the camera with a diagnostic view under the original-camera claim.

The [post-displacement feasibility note](../../POST_DISPLACEMENT_PILOT_FEASIBILITY_20260906.md)
documents the existing per-element attribute path and why topology-dependent
vertex normals can invalidate pre-displacement identity/contact certificates.
It is design-only. No post-displacement pilot, contact-policy relaxation,
raw/extra_smooth/centroid/BEB1 RGB comparison, or long video was launched after
the visibility screen found no qualifying sample.
