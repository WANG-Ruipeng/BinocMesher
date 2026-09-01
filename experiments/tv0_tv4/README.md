# Lightweight production-derived TV0--TV4 audit

This directory adds a read-only, falsifiable theory audit on top of the
existing P1 observer.  It does **not** modify `run_slicing()`.

The default profile uses 24 public-demo-style cameras, `pixels_per_cube=180`,
a vnoise-compatible deterministic terrain, exact rational event predicates,
and `OMP_NUM_THREADS=1`.

Stages:

- **TV0:** classify every source cell as regular-monotone, quotient-singular,
  invalid, or unresolved; census source grammar, temporal gaps, shared faces,
  and sampled embedding diagnostics.
- **TV1:** independently reconstruct temporal-face saddle events and prove, on
  exact source-labeled probes, that no tested event-free interval changes its
  local combinatorial signature.
- **TV2:** audit production quotient-singular cells over an exact epsilon
  ladder, plus order, HVID relabeling, degree-2 subdivision, and integer-chain
  cancellation invariants.
- **TV3:** compile one real cache-mined saddle into an offline 2-tet relative
  half-handle and audit its link, Gram volumes, and regular slices.
- **TV4:** map every raw/logical incidence of that canonical event to one shared
  offline block and audit ownership, relative-boundary agreement, and the
  2-to-1 component transition.

Run:

```bash
bash experiments/tv0_tv4/run_lightweight_tv0_tv4.sh \
  --repo "$PWD" --output /tmp/binoc-tv0-tv4
```

Expected markers:

```text
PASS_TV0_TV4_PRODUCTION_DERIVED_THEORY_VALIDATION
PASS_INDEPENDENT_TV0_TV4_VALIDATION
PASS_LIGHTWEIGHT_TV0_TV4_FROM_FRESH_CACHE
```

Scope limitations are intentional: TV3/TV4 are production-derived **offline**
blocks, not a production intervention; TV1 is a local source-labeled finite
oracle, not a formal whole-scene PL-isotopy theorem; the Jacobian/Gram outputs
are sampled diagnostics, not interval or Bernstein certificates.
