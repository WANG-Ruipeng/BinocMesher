# Final visibility gate: Cave complete, Forest B running

This is a historical progress snapshot, not the final three-segment verdict.
Forest B attempt02 started on 2026-09-06 at approximately 16:06 UTC using the
same final 8 GiB / 90-minute native ceiling as Cave. The original campaign
deadline remains 19:28:39 UTC. Neither the scene nor Method policy was changed.

## Cave result

The fixed Cave seed 1, frames 1–64, 24 FPS, 960×540, 6 px, original-camera
pre-surface-displacement opaque screen completed. Native build: 3189.961 seconds;
complete cache: 1,513,840,506 bytes. Saved coarse, actual camera, kernel and gin
inputs match the first attempt. Original resource STOP records remain intact.

| Measurement | Cave |
| --- | ---: |
| Raw observations / logical incidences / canonical events | 522 / 301 / 147 |
| Source-ready / fixed source rejection | 123 / 24 |
| Independently admissible / direct rejection | 102 / 45 |
| Jointly admitted events | 87/147 (59.18%) |
| Jointly admitted components | 79/114 |
| Fallback events | 60 |
| Modified natural frames | 16/64 |
| Actual natural event-frame replacements | 696 |
| Original-camera candidate event/frame cases | 1176 |
| Visible candidate / admitted / replacement events | 0 / 0 / 0 |
| Unknown support visibility | 0 |
| Visible-conditional admission rate | null (0/0) |
| OMP 1/8 identical queries | 66 (64 natural + 2 root diagnostics) |

The 60 fallback events comprise 45 direct rejections, 2 independently valid
members falling back with rejected peers, and 13 independently valid members in
6 unsupported shared-support joint components. They are not all propagation.

Frames 21–28 each commit 43 event replacements; frames 37–44 each commit 44.
All six baseline/combined geometric buffer hashes match on all 16 changed frames.
Every tested image contains 518,400 foreground pixels: this is not an empty-frame
result. Of 1,176 candidate cases, 1,136 have no isolated pixel coverage and 40
are occluded or lose an exact depth tie. No-isolated-coverage is not proof that
every such event lies outside the frustum. Two selector-unsupported candidates
use the explicitly recorded candidate-domain proxy, not a hypothetical closure.

Source screening took 92.639 seconds; the OMP1 certification/sequence/visibility
harness took 1459.429 seconds; OMP8 sequence confirmation took 594.647 seconds.
These differ in work and are not an OMP speedup benchmark or production timing.

Evidence: [confirmed summary](cave/screen02_omp8/combined_summary.json),
[build/source audit](cave/build_source_independent_audit01.json),
[OMP1 funnel audit](cave/certification_funnel_audit01.json),
[visibility audit](cave/visibility_buffer_audit01.json),
[final OMP1/8 evidence-chain audit](cave/final_chain_audit01.json).
The OMP1-specific audits retain their original scope; the final chain audit binds
the subsequent OMP8 confirmation without rewriting those historical receipts.

## Still unresolved at this snapshot

Mountain remains the earlier completed negative result. Forest A remains prior
evidence and is not one of the three new preregistered segments. Forest B attempt02
is still building: no new admission or visibility count is available for it.
Consequently this snapshot does not issue STOP_ORIGINAL_CAMERA_VISUAL_HEADLINE
or select a quality pilot. No displacement, RGB, SSIM, new camera/seed/segment,
third resource increase, or Method extension was performed.
