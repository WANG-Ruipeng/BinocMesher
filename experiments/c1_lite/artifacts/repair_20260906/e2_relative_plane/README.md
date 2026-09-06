# E2 relative-plane conditional feasibility

This is a small read-only-input diagnostic, **not a production admission**.
No v2/v3 kernel, previous report, source cache, or admission gate was changed.

## Frozen scope and result

- Original E2 window: `[102/5, 106/5]`, root `104/5`, additional breakpoint `21`.
- Three complete affine branches and four actual singleton states.
- Seven fixed retained-triangle types: six previously diagnosed zero-feature
  candidates and the adjacent triangle sharing the whole boundary edge.
- `result.json`: 49/49 ideal relative-plane certificates PASS. This is not a
  new scan or admission of every E2 retained triangle.
- The JSON records source/cache/script hashes and verifies frozen inputs were
  unchanged before publishing. Output: 501,198 bytes.
- SHA-256: `7a19f7ed0c51202794639ef1f081e19fcb1469b25aa9d6ae75f031069af7bb1e`.

The coordinate model is exact rational interpolation of serialized HV
binary32 values. Retained faces are the raw-original-time filtered canonical
SourceVID geometric quotient, not actual C++ array identities or extra_smooth.

## Exact geometric statement

Let `a = 159:8|311:0`, `b = 159:8|391:8`, `e = b-a`, and `d = x-a`.
The same rational tilt `lambda = 1` works across all seven states:

```text
L(x) = cross_XY(e,d) + cross_YZ(e,d)
     = e_x*d_y - e_y*d_x + e_y*d_z - e_z*d_y.
```

Every non-edge fan vertex, including the center, has
`L >= 462825/524288 > 0`. Every nonshared retained vertex has
`L <= -148225/524288 < 0`. The two boundary endpoints have structural `L = 0`.
All extrema are computed exactly for quadratic polynomials on the full closed
branches; singleton states are checked separately. This is not a sampling test.

Consequently the relative plane permits only the declared shared vertex/edge,
or no contact. It no longer requires a nonshared retained point's original
XY supporting-line equality to survive rounding.

## Conditional perturbation bound — not a measured runtime bound

Assume each corresponding coordinate, including both plane-defining endpoints
and the fan center, changes by at most `epsilon`; shared vertices must reuse
one common actual coordinate on both sides. The perturbed plane is defined
from those same actual endpoints. Then:

```text
|delta L| <= 2*epsilon*Bmax + 16*epsilon^2
Bmax = 825/64
minimum strict signed margin = 148225/524288.
```

The bound on `B` follows from maxima of sums of absolute affine functions at
the branch endpoints. Two hypothetical sufficient bounds are:

| Coordinate error bound | Functional error bound | Remaining signed margin |
| --- | --- | --- |
| `1/1024` | `1651/65536` | `135017/524288 > 0` |
| `1/128` | `829/4096` | `42113/524288 > 0` |

These bounds concern only these relative-plane signs. They do not certify
local fan embedding, other pairs, incidence links, endpoint realization, or
actual runtime coordinate errors. In particular, rounding physical time to
double near a source breakpoint may change selectors/SourceVIDs: a coordinate
error bound does not prove the necessary discrete identity correspondence.

Every certificate and the top-level result retain `runtime_admitted=false`.
Actual coordinate-error, SourceVID mapping, and double-time selector proofs
are explicitly absent.

## Reproduce

From Windows, using the existing Miniforge `binoc-exp` environment, select a
fresh output filename (the script refuses overwrite):

```powershell
wsl.exe -d Ubuntu -- /home/warpwang/miniforge3/envs/binoc-exp/bin/python -B /mnt/e/BinocMesher/experiments/c1_lite/audit_relative_plane_feasibility.py --source-report /mnt/e/BinocMesher/experiments/c1_lite/artifacts/ab_20260906/window_audit/event-02-88ade47aa4fa/source.json --cache-root /home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache --output /mnt/e/BinocMesher/experiments/c1_lite/artifacts/repair_20260906/e2_relative_plane/result-recheck.json --max-seconds 90
```

The diagnostic limits output to 512 KiB and the requested wall-time budget.
The original run completed in about seven seconds. Ten synthetic tests cover
structural zeros, aliases, inconsistent identities, the center constraint,
interior quadratic failure, exact error arithmetic, and input permutations:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/e/BinocMesher/experiments/c1_lite && /home/warpwang/miniforge3/bin/python -B -m unittest test_relative_plane_feasibility'
```
