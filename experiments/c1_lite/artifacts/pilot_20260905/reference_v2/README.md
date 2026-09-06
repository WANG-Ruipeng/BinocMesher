# Endpoint-order rerun: PASS_PREFLIGHT_ONLY

This rerun implements the approved coordinate evaluation-order correction.
It passes the fixed offline screen; it does not admit a production overlay.

Protocol: [pre-run amendment](../endpoint_order_v2_protocol.md).
Evidence: [result.json](result.json). The [original stopped result](../reference/result.json)
and [census](../census.json) remain byte-for-byte unchanged.

## Change and verification

The prospective binary32 center is now computed from the already quantized
source diagonal, rather than rounding the midpoint of a different binary64
diagonal. There is no tolerance-based acceptance or event-specific branch.
The binary64 reference trajectory is unchanged and has its own exact endpoint
collinearity gate. Neither the endpoint emulation nor the height comparison
executes the production C++ slicer.

- 16 unit/regression tests passed, including the old endpoint failure fixture
  and a binary64-fail/binary32-pass case.
- Both lower and upper endpoints pass the separate binary64 and binary32
  exact collinearity checks; prospective binary32 diagonal gap is zero.
- The root boundary positions and canonical center match the frozen C0 IR.
  This is a local input-geometry check, not a new whole-mesh or OMP execution.
- The event, original window, five time probes, grids 32/64/128, and strong
  per-probe quality rule are unchanged.
- All nine interior time/grid comparisons pass; endpoint comparisons agree
  with baseline to floating-point evaluation precision.
- `runtime_plan_emitted=false`, `render_started=false`, and
  `continuous_window_certified=false`.

## Local geometry signal

Reference: the original independent static demo terrain function. Metric:
mean absolute **vertical height error** on identical XY midpoint-grid samples
within the patch at each time. The following table reports the fixed 128 grid;
32 and 64 grids support the same improvement direction.

| Internal time | Position in window | Samples | Baseline mean error | C1-lite mean error | Reduction | Baseline sampled max | C1-lite sampled max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 117/11 | Left midpoint | 7699 | 0.21440279 | 0.20301984 | 5.309% | 0.45990515 | 0.44028500 |
| 120/11 | Exact root | 7342 | 0.21350101 | 0.19097801 | 10.549% | 0.45586179 | 0.43646735 |
| 123/11 | Right midpoint | 6976 | 0.21249308 | 0.20135340 | 5.242% | 0.45177282 | 0.43465957 |

This extends the previous root-only positive signal to two fixed neighboring
samples. The root numbers are close to, but not identical to, the earlier
saved-runtime-mesh experiment: this run evaluates parser binary64 boundaries,
not the earlier quantized whole-mesh arrays. Do not combine those measurements
as if their sample sets and coordinate models were identical.

## What the naive control establishes

The naive four-triangle fan interpolates between endpoint diagonal midpoints,
without the exact-event center. Its measured error agrees with baseline to
roundoff. The selected source faces use the same shared diagonal at all five
probes; the naive center lies on that diagonal, so the fan only subdivides
the original surface.

Thus the control supports the narrow interpretation that adding triangles
alone does not explain the observed geometric gain. It is a geometric null
control, not an independent strong smoothing method. It does not establish
that the exact-event center beats other sensible center-placement methods.
No pairing flip of this selected baseline patch is demonstrated here, and
these results must not be described as a repaired topology jump.

## Remaining paper-relevant gates

1. Establish the source branch/owner contract throughout this same window,
   including the census candidate breakpoint at tau=11; matching five samples
   is not an interval proof.
2. Check nondegeneracy and patch/exterior compatibility over the interval,
   not just convexity of the projected patch at the sampled times.
3. Before any visual claim, check original-camera ROI visibility and natural
   frame sampling, then validate the actual runtime coordinate/attribute path.
4. Broader events and meaningful alternative-center / original-method controls
   are needed before claiming general superiority or paper-level novelty.

No SSIM, Hausdorff, global-topology, six-scene coverage, or full-window error
claim follows from this result. No rendering, production edit, commit, or push
was performed in this rerun.

## Provenance

- Reference source SHA-256:
  `7fb812f7fdaa2488e17e7774017118b0cb0844c8d1aef8fc1a686ccc45af83eb`.
- Result SHA-256:
  `231dca42c361413c4b2ff43520a96c180be2de45d5b23e321eac4ba5aaa358f1`.
- Preserved original result SHA-256:
  `1c2a766bc2f0e6c2212bf589c9ab873c4c1b607751fb10b87ce246507cc4d12b`.
- Preserved census SHA-256:
  `51bd963cf70be3461808490c69c30b04d0b45b2502612c690fdb9c0da2a99164`.

For reproduction use the original reference command with the current source
and a **fresh** output directory. Do not overwrite either archived result. The
historical v1 README's expected STOP applies to the old source hash; the current
v2 source produces PASS_PREFLIGHT_ONLY for these inputs.
