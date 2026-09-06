# E2 actual-evaluation analytic envelope (conditional)

The authoritative final supplement is `result_center_checked.json` (14,872
bytes). `result.json` is retained unchanged as the first source-envelope
report; the final version additionally gates the separately specified
exact-anchor-then-RNE32 center bound.

Original window: `[102/5,106/5]`, root `104/5`, including the separate actual
singleton at 21. All three open affine cells and four singleton states were
audited without changing the window, anchors, old kernels, or production code.

Each unit has 5,262 emitted raw fan owners and 10,938 original polygon vertex
occurrences. Original serialized endpoint order is reparsed from bytes; it is
not inferred from canonical SourceVIDs. There are no reversed-time or
zero-effective-span emitted sources. Noncanonical original ordering is common
and valid: the canonical-to-original-order relation is unique. All replicas
share the exact ideal trajectory on each proven source branch. No-fan polygons
were included in the traversal (none are emitted in these seven units).

All four interface pairs remain strictly inside their effective interpolation
intervals. Minimum clamp clearance is `2/5`, far above the analytic time error.
Remote effective IDs may clamp at a slightly different real time after rounding;
continuous clamped coordinate trajectories remain covered by the error bound.
The report does not claim remote ideal/runtime identity equality.

Let `u=2^-53`, `gamma8=8u/(1-8u)`, `M=1100`, `L=275/8`, `T=106/5`.
The conservative rational-entry per-coordinate bound is

`L*T*gamma8 + M*gamma8 + 2^-24*(M+M*gamma8) + 2^-150`,

approximately `6.556511087718603e-5`, strictly below `1/1024`.
The center's original exact affine anchors followed by one RNE32 conversion
also fit `1/1024`; the runtime center implementation is a separate obligation.
The physical-entry model instead references its **actual computed binary64
discrete time** and omits the rational conversion term. It does not establish
equivalence of the physical selector with the rational selector.

The numeric profile is IEEE binary64/binary32, RNE, gradual underflow, no
fast-math reassociation, and fixed deltaT in `[2^-100,2^100]`. The supplied
deltaT is `0x1.eaabfa360338dp-6`; supplying it is not itself proof that a loaded
library uses it. Runtime compiler/library/cache binding, exact suppressor
equivalence, shared final-array reuse, endpoint contract and transactional
admission remain external obligations. `runtime_admitted` is **false**.

Reproduce from the repository, without initializing or changing the cache:

```sh
python -B experiments/c1_lite/actual_evaluation_envelope.py \
  --source-report experiments/c1_lite/artifacts/ab_20260906/window_audit/event-02-88ade47aa4fa/source.json \
  --cache-root /home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache \
  --output /tmp/e2_actual_envelope_fresh.json \
  --delta-t-hex 0x1.eaabfa360338dp-6
```

The output must not exist. Ten synthetic tests are in
`test_actual_evaluation_envelope.py`. This is an analytic all-time geometric
supplement, not a finite-sample proof or an automatic whole-window production
admission. Existing schedule-certified runtime results are not relabelled.
