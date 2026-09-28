# Spatial scoring correction before climate qualification

2026-09-29. Register before implementation or rescoring any historical run.
The complete industrial objective and nonlinear/production requirements remain
unchanged. The specific live paired-reference process is not restarted or edited.

## Research and actual defect

[OMIP evaluation](https://gmd.copernicus.org/articles/13/3643/2020/), sections3,
4 and appendicesC/E, separates spin-up drift, multi-decade means, seasonal and
interannual variability, depth/basin structure and circulation. Section4.1
explicitly warns that scoring restored salinity against the restoration dataset
can underestimate bias. This is not evidence that any restored model is invalid.
[Xarray's official weighted example](https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html)
explains why equal-cell means overweight high-latitude regular-grid cells.
Cos(latitude) is a proxy on a REGULAR grid, not a substitute for arbitrary
cell geometry. OMIP's prescribed SSH filtering is not adopted as an SST formula.

Current `benchmark_metrics.py` uses equal-cell global/regional bias and RMSE;
its A2 window uses `round(2)` GRID CELLS irrespective of horizontal resolution
and includes finite land sentinels. `score_npz` scores against `T_init` without
an explicit reference-role warning. These scores cannot establish independent
climate qualification. Archived results must remain intact.

## Method change and witnesses

Introduce explicit metric definition `area_weighted_angular_box_v2`. Global and
regional bias/RMSE, zonal means, zonal pattern weights and pattern correlation
use wet-cell areas. Empty regions and non-finite wet values do not silently
improve the score by being dropped. Shape and nonpositive/non-finite weight
errors reject. Pure land values never enter wet statistics or the filter.

Use spherical rectangle area R^2*delta_lambda*(sin(phi_n)-sin(phi_s)). Provided
areas take precedence; midpoint/extrapolated edges inferred from ordered centers
are explicitly labelled as inferred, NOT independently verified model geometry.
Longitude ordering may cross the dateline; duplicate/nonmonotone centers reject.
Latitude edges inferred at a pole are clipped to the physical pole, not a metric
cosine floor. Supplied areas must be positive on every scored wet cell.

A2 is an area-weighted wet-only separable ANGULAR box with longitude periodic
distance and latitude bounded, each angular half-width2degrees. The same
coordinates/mask/weights/filter act on BOTH fields. It is not a constant-km
filter and does not claim mesoscale or eddy-by-eddy accuracy. Use sparse window
matrices to avoid a Python per-cell filter loop. A1 is zonal mean, then weighted
by each latitude band's scored wet area; weighted correlation is undefined for
constant patterns and is reported non-finite, not invented as a perfect score.

Independent NumPy small-grid witnesses must distinguish equal-cell/area scores,
coordinate/grid-cell windows, wraparound, land contamination, weighted pattern
correlation, invalid weights and wet missing data. Internal NPZ and external
NetCDF scoring must call the SAME spatial implementation. Tag reference role:
the existing T_init is an initialization/shared-initialization reference, not
automatically independent validation. Keep temporal averaging and budget-drift
definitions explicitly labelled as existing diagnostics, not closed budgets.

Comparison gate rejects different metric definitions and different spatial
comparison domains (coordinates, wet mask and weights). Legacy-versus-legacy
comparisons remain historical internal diagnostics with that explicit scope;
legacy-versus-v2 cannot pass using coincidentally similar numbers. A v2 PASS
still means internal comparison only, NOT century or climate certification.
Report-table rows identify the metric definition so a mixed table is not silently
interpreted as a like-for-like ranking.

## Remaining qualification requirements

This correction does NOT repair nonlinear momentum, moving kinetic/buoyancy
budgets, physical sources, production migration or global polar topology.
Before a formal climate experiment, register observational provenance/epochs,
train/tune/holdout separation, restoration targets, calendar/time bounds and
time-weighted multi-year means; evaluate raw and predeclared filtered errors,
volume-weighted 3D temperature/salinity, seasons, variability and circulation.
Do not replace those requirements with these SST spatial unit tests.
Century runs require actual full-model budget residuals and physical drift
attribution, restart and dt/precision/initial-state sensitivity. GPU/distributed,
forecast and whole-model differentiation remain separate full-objective gates.
