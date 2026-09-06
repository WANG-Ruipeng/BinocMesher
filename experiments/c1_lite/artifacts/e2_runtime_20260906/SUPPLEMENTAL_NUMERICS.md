# Supplemental numerical analysis: not a runtime admission

These are separate, bounded offline analyses of the unchanged E2 source window.
The production requested-schedule transaction does not trust their PASS flags:
it rechecks each requested query's actual identities and binary32 coordinates.

## Conditional perturbation certificate

[Result](../repair_20260906/e2_conditional_perturbation/result.json),
[implementation](../../conditional_perturbation.py),
[runner](../../run_e2_conditional_perturbation.py).

For each of two proposed uniform absolute coordinate envelopes, `1/1024` and
`2^-20`, all 7 original units and 26,726 retained-candidate/unit contact checks
pass the sufficient conditional predicates. There are no UNKNOWN or REJECT
cases for those fixed envelopes. The 7 local fan/convexity checks also pass.
The units are 3 complete affine branches and 4 actual singletons, not a dense
time sample. Per envelope the contact certificate counts are:

- 26,530 fixed-direction robust convex-hull separations;
- 189 source-edge-anchored relative-plane certificates;
- 7 shared-corner relative-plane certificates.

The second count does not mean 189 distinct shared edges: that certificate
family can cover different allowed common features. Shared vertices remain
structural zeros in the perturbation argument; every nonshared point, including
the proposed center, must retain the required strict sign.

`1/1024` here is a hypothetical error-envelope bound, **not** a welding tolerance
and not an epsilon used to accept touching geometry in the production checker.
These results only imply safety if every actual occurrence and selected merged
coordinate satisfies the envelope, with the required actual identity/incidence
and source-face replacement correspondence.

## Analytic actual-evaluation envelope

[Implementation](../../actual_evaluation_envelope.py) separately reparses the
original ordered VID bytes, including every emitted polygon vertex, suppressed
source faces and polygons that produce no fan faces. Original endpoint order is
not reconstructed by canonicalizing SourceVID. `build_retained_unit` verifies
that the partition has no omitted internal selector/source threshold before a
midpoint is used merely to label an open branch.

The analysis distinguishes:

1. Exact-rational entry: comparison with the ideal source at the requested
   rational time, including the physical/discrete floating conversion error.
2. Physical-double entry: comparison with the ideal source at the **actual
   computed discrete double**, not at an independently assumed rational time.
   This does not establish physical-entry owner/selector equivalence.

Under the explicitly declared IEEE/RNE arithmetic and magnitude assumptions,
it uses a conservative interpolation forward-error bound, final binary32 cast
bound, and (for exact input) a Lipschitz source-motion bound times the time error.
Four interface identities additionally require a strict all-window clearance
from their effective clamping thresholds. The center has its own exact-anchor
then RNE32 bound; it must not be silently omitted from the proposed envelope.

Supplying the measured `deltaT` is not itself proof that every runtime execution
uses that arithmetic environment. The report deliberately leaves binary/mode
binding, source suppressor equivalence, center implementation binding, physical
selector equivalence and whole-window transaction admission unproved. See
`verification.json` for the final supplementary artifact paths and statuses.

Therefore neither analysis upgrades the old continuous-window admission gates.
The completed result in this directory remains **actual requested queries +
atomic batch publication**, with no all-real-time or render-quality claim.
