# I0B0 existing six-hour checkpoint postprocessing

Only the existing t0/2h/4h/6h states were read. No integration, physical fill, original data or production source changes. Protocol definitions and arithmetic tolerances were frozen before computation; their SHA256 is `729f1138367c9f9095b69a7adef3d794e4778cbd332f0a5c2e244d1ad889d2f8`.

Historical source: `212df951c351f82dba74fbc43db5e52b0ad34c47`; original executed six-hour protocol SHA256: `9d9e756162c8f19ac1d3e32bbfb178d3c44d13c4f11ccf83e697b7e9c716f148`. All35 source hashes matched (the registered material diagnostic patch is explicit), original grid/parameters/initial state identities matched, and t0 fields matched the original archive byte-for-byte. Later I0B0 checkpoints matched registered COMMITTED markers and SHA256.

## Closed-domain redistribution

All32,087 wet columns form **one connected component** under the fixed common-surface wet connectivity. Its area is3.1859498418313675e14m2. Basin1 is exactly the full wet domain, not independent named ocean basins.

| Hours | Full eta volume change m3 | Full mean eta m | Positive volume m3 | Negative volume m3 | Prospective arithmetic bound m3 |
|---:|---:|---:|---:|---:|---:|
|0|0.000000000e+00|0.000000000e+00|0.000000000e+00|0.000000000e+00|9.094947018e-13|
|2|-5.493164062e-04|-1.724184101e-18|2.041991124e+13|-2.041991124e+13|3.714360217e+01|
|4|-7.324218750e-04|-2.298912134e-18|3.721699421e+13|-3.721699421e+13|6.769731810e+01|
|6|-1.953125000e-03|-6.130432358e-18|4.418495135e+13|-4.418495135e+13|8.037195829e+01|

Signed volumes are evaluated as a direct sum; separately rounded positive/negative totals need not subtract to that exact floating-point sum. No non-roundoff closed-component drift was found.

| Latitude band | 2h net volume m3 | 4h net volume m3 | 6h net volume m3 | 6h mean eta m |
|---|---:|---:|---:|---:|
|lat_-66_-58|-6.506078321e+12|-1.380970144e+13|-1.546110990e+13|-0.930284219|
|lat_-58_-30|4.496089008e+12|5.925990204e+12|-4.539266599e+11|-0.005515252|
|lat_-30_30|3.919429544e+12|1.341425980e+13|2.384218905e+13|0.132685503|
|lat_30_66|-1.909440232e+12|-5.530548556e+12|-7.927152496e+12|-0.198268776|

At6h the southern eight rows lose1.54611099e13m3, mean−0.930284m; all of this band is negative. The tropics gain2.38421891e13m3 net, offsetting both south and northern net deficits. The total positive eta-change volume4.41849513e13m3 exceeds the southern loss because other regions also lose water. These budgets locate compensation; they **do not trace parcels or prove a route from the south to any individual region**. Four longitude-quadrant net budgets and extrema locations are retained in the scalar JSON.

The296 retained points occupy1.69426055e12m2, approximately0.532% of the full wet area. Their−1.017m six-hour mean must never be described as the whole domain falling by one metre.

## Fixed-reference pressure and motion

Anomaly pressure is integrated from z=0 with the historical linear EOS, masked trapezoidal dz_3d and original dz_norm. Surface pressure is rho0*g*eta. Column-average pressure itself is descriptive; forces are averages of common-wet 3D gradients, **not gradients of column-average pressure**. Variable bathymetric weights do not commute with gradients. Reported anomaly pressure is not absolute seawater pressure.

| Hours | Global density-force RMS m/s2 | Surface-force RMS | Pressure-sum RMS | Pressure+Coriolis RMS | Coarse interval acceleration RMS | Maximum wet vector speed m/s |
|---:|---:|---:|---:|---:|---:|---:|
|0|4.447823127e-06|0.000000000e+00|4.447823127e-06|4.447823127e-06|n/a|0.000000000|
|2|4.425338460e-06|3.368996829e-06|2.781583084e-06|2.683786920e-06|1.609155130e-06|0.613338544|
|4|4.412646014e-06|3.836197853e-06|2.416405580e-06|2.113183158e-06|1.210369243e-06|1.064641538|
|6|4.402493998e-06|3.863745985e-06|2.526400294e-06|2.106496498e-06|1.259645897e-06|1.286173894|

Initially u=v=eta=0, so pressure due to eta and Coriolis are zero, but the density-force RMS is4.44782313e−6m/s2 globally and6.84138177e−6m/s2 in the south8rows. This is a direct at-rest **pressure imbalance**, not evidence that the WOA values are wrong or that a reference/unit error exists. Southern column-average density-anomaly pressure changes only40815.333→40811.062Pa over6h; its surface-pressure mean changes0→−9354.240Pa. Absolute mean-pressure comparisons do not establish cancellation of spatial gradients.

At6h the southern pressure-force RMS is3.91208153e−6m/s2, pressure+Coriolis RMS3.23022687e−6m/s2; both remain nonzero. At the historical failure location302.5E63.5S, density force is[−1.9049e−7,1.9972e−6], surface force[−3.9451e−6,−5.9090e−6], sum[−4.1356e−6,−3.9118e−6]m/s2. Thus eta-pressure can oppose one component and reinforce another. Max6h vector speed1.286174m/s is at198.5E56.5S,z=0, distinct from the future failure column. This is vector magnitude, unlike the earlier max-component gate.

Coriolis is[f*vbar,−f*ubar]. The7200s fixed-node differences are coarse Eulerian interval accelerations, not instantaneous split-step derivatives. Advection, wind, diffusion and friction were not ablated here; pressure+Coriolis is **not the complete momentum residual**. Neither pressure opposition nor a static near-cancellation would establish geostrophic or full dynamic equilibrium.

## Arithmetic and reference checks

For all four snapshots the NumPy hydrostatic oracle and historical JAX pressure match exactly in float64 at wet nodes. Common-wet spherical force error is at most1.490778e−19m/s2; a constant1m pressure-gauge shift changes forces by at most the same amount. Prospective pressure tolerance is about(9.18–10.11)e−8Pa and force tolerance(1.94–2.14)e−15m/s2. No counterexample to the stated volume, integration or constant-reference contracts was found. This checks this discrete operator, not physical suitability of the truncated closed southern boundary.

## Corrections and next hypothesis

The earlier85–89m3 six-hour volume-ledger allowance was a **posthoc arithmetic supplement**, not a preregistered scientific acceptance gate. The present4096eps postprocessing tolerances were defined before this diagnostic computation, but are also arithmetic checks and do not promote six-hour results to scientific qualification.

Current evidence favors closed-domain redistribution driven from a spatially nonuniform, initially pressure-unbalanced field; it does not identify the initial imbalance as erroneous or establish its causal share relative to wind and split dynamics. Next review should formulate a supported balanced-initialization/closed-boundary hypothesis and a complete momentum/energy budget before any further integration. No30h repair or industrial-quality/speed claim follows.

## Reproducibility and resource record

Run `scripts/controlled_window/postprocess_6h.py --help`; supply private run/archive/history-source paths and published protocols/artifact hashes explicitly. No private NPZ/WOA/checkpoint files are committed. The scalar JSON records each used input/checkpoint hash, component mask hashes, locations and limits. Execution used1CPU,4GiB/180s guards: wall4.375922s, peak working set516157440bytes, peak sampled private commit434405376bytes, exit0. One earlier3.2733s static attempt failed only at NumPy-bool console serialization; no final report was written. After explicit bool conversion the second execution completed. The frozen scientific definitions did not change.

Independent inspector reviewed code, provenance binding, final scalar budgets and inference limits without running integration.
