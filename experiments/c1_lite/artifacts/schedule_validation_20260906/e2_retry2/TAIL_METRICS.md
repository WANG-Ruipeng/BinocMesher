# Local-footprint tail metrics

These are the same fixed diagnostic camera and original-source footprints as
README.md, against the selected 256-grid terrain reference. They are provided
alongside the mean errors, not as independent samples or statistical tests.
Depth is in scene units; normal angles are in degrees. Lower is better.

| Query | Method | Depth P95 | Depth RMSE | Max depth | Normal P95 | Max normal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| natural 15 | raw | 0.35072099 | 0.20104496 | 0.49600682 | 34.01446 | 36.40689 |
| natural 15 | centroid | 0.34689486 | 0.19610304 | 0.49602076 | 33.76920 | 36.60389 |
| natural 15 | BEB1 | 0.34571569 | 0.19347053 | 0.49603947 | 33.96413 | 36.88781 |
| exact root | raw | 0.36185614 | 0.20419200 | 0.50192055 | 34.18607 | 36.59361 |
| exact root | centroid | 0.35817566 | 0.19870214 | 0.50196144 | 33.89405 | 36.81752 |
| exact root | BEB1 | 0.35631294 | 0.19560666 | 0.50203856 | 34.37076 | 37.14495 |

Depth P95 and RMSE improve in these two queries. The largest depth error is
slightly higher; the mean-reference-refinement gain gate does not certify
maximum-error differences. Normal tails are not uniformly better either:
the BEB1 maximum normal error is higher in both queries, and its root normal
P95 is higher than both controls. The natural-frame BEB1 normal P95 lies between
raw and centroid. Thus even the local experiment does not support an
all-metric or worst-case-quality superiority claim.

The raw rasterizer counts 1,359 degenerate input faces at natural frame 15
and 1,372 at the exact root. Those are existing baseline degeneracies
retained outside the scoped repair,
not evidence that the output is a globally defect-free mesh. The window
validation checks newly proposed geometry and unchanged exterior separately;
it is not a cleanup pass for every baseline defect.
