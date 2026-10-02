# Frozen-chart physical stock projection

## 2026-10-02: minimum contract before numerical execution

Start from merged main `ea0fd77257690ccf0b3e1f11d6c98f1e047c67d3`.
The new scope is an auxiliary weighted physical projection B and its direction,
not a conservative remap of the original inventories. Actual h/IT/IS/Mu/Mv,
the endpoint-prism area metric A=dL/2, and original endpoint PE stay authoritative
and unchanged. Production qualification and accepted-step count remain false/zero.

For each actual endpoint-layer velocity basis phi, over the complete physical
strip chart, map water volume, four stock integrals and anomaly gravity moment:

```text
B_H = integral(phi dV)
B_stock = integral(phi q dV)
B_G = integral(phi z rho_prime dV)
E_B = rho0*g*integral(eta^2/2 dx)*L + g*sum(B_G)
```

Density comes from actual historical-EOS inventory means in the stable affine
family; actual T is constant, so physical S follows that density authority.
Momentum uses actual P0 stock/h for the two strip owners, interpolated in x.
The weighted physical mass W=rho0*integral(phi_a*phi_b) gives B_M=W*u_actual
and the true interpolated volume KE. A declared 0.01-second fixed-W scratch
impulse tests mapped before/after momentum and midpoint pressure work. It does
not alter raw stocks or advance the physical model.

Partition unity gives physical global inventories. For sloping eta these do
not generally equal the original endpoint-prism sums. With k=(etaR-etaL)/d,
raw minus physical projected IS is
`L*d^3*(a1*k+s*k^2/2)/(6*rho0*beta)` and affine Mu defect is
`rho0*L*d^3*alpha*k/6`. The primary counterexample therefore has IS defect
50/779 PSU*m^3 and Mu defect 4.1 kg*m/s. Their original manufactured tangent
defects change at -3*alpha times the corresponding value. Water, constant-T
and constant-v total defects are zero. These differences must be exposed,
not silently transferred to the new state or folded into an energy residual.

The existing 629.17048125 J endpoint-versus-physical PE discrepancy stays a
negative control, including its separate reference free-PE part 628.453125 J
and anomaly part 0.71735625 J. E_B uses the existing physical PE definition;
it does not relabel the original endpoint functional. Per-basis gravity moments
are geometric projection diagnostics, not extra accepted conserved prognostic
stocks. Their physical evolution includes an explicit vertical-motion source.

The initial physical chart is frozen. Every raw interior interface and every
virtual-cut endpoint remains fixed; only actual top moves. Rebuilding normalized
cuts for each perturbed state gives a different chart and is forbidden. Owners
are independently resolved against raw interfaces. Crossing any frozen virtual
cut, altering raw interior interfaces, or admitting nonfinite geometry refuses.

Bdot is Reynolds' volume field direction plus actual-top shape. Local weighted
stock balance requires both outward ALE flux and the basis-divergence volume
term. For B_G the source additionally contains
`rho_prime*(z*u*phi_x+w*phi)`. Removing a real virtual-cut flux, this weighted
term, top shape, or the gravity-moment source must produce a resolved failure.
GCL is checked globally and for every weighted basis; raw endpoint hdot is not
silently substituted for physical projected volume direction.

Independent seven-point volume quadrature, original raw inventory sums and PE,
pressure virtual-work oracle, and an explicitly affine-preserving raw-stock
parameter curve check the map and its derivative. Arbitrary vertical shear is
covered only by static projection/mass/pressure work; the material tangent stays
in the validated affine flow family. Positive parameter epsilon is fixed in
the protocol; polynomial shape remainder and arithmetic roundoff are separate.
This curve is not a time-integrator order experiment.

The next industrial requirements remain raw conservative evolution/inverse
projection, original momentum control volumes, general P1/limiter derivatives,
variable-mass KE, full moving sigma/crossing and complete predict/12fast/replay,
real353 failure reproduction, and MOM6 same-condition equal-error speed work.
This focused contract neither enables nor claims those capabilities.

### Pre-freeze weighted gravity-balance clarification

The gravity-moment balance is the separate equation
`Gdot + weighted_ALE_flux = integral(rho_prime*(z*u*phi_x+w*phi))`.
Its right-hand side already includes the basis-divergence term. No second
`integral(z*rho_prime*u*phi_x)` is subtracted; equivalently one may write
`Gdot + flux - integral(z*rho_prime*u*phi_x) - integral(rho_prime*w*phi) = 0`.

### Pre-freeze case and finite-difference clarification

Actual T must equal Tref=15 within the inherited per-layer roundoff bound.
The JSON fixes all primary numbers, both PR25 fourteen-layer partitions and
the static shear constructions before execution. Zero spurious force requires
flat eta, zero horizontal density gradient and constant external pressure.
Owner-cut jumps of phi are retained in per-strip edge fluxes; the inside-strip
phi_x volume term does not replace them. The total PE parameter-difference
bound adds the reference free-PE remainder
`epsilon*g*rho0*L*integral(eta_dot^2/2 dx)` to the anomaly-moment remainder.

## 2026-10-02: verification implementation clarification

The frozen JSON formulas and thresholds are unchanged. Absolute polynomial
remainders are evaluated by exact rational arithmetic on the binary input
numbers. Each C2/C3/C4 polynomial must have a one-sign Bernstein coefficient
hull over [0,d]; then its nonnegative-basis absolute integral is an exact
signed polynomial integral. Conversion to a floating upper bound rounds
upward. A mixed hull refuses instead of replacing the absolute integral by
sampled Gauss values. The additional a0=0 control is a valid static projection
but deliberately cannot obtain this restricted finite-difference certificate.

The independent oracle binds every actual T and historical-EOS anomaly mean
to the declared affine specification. Material Bdot and FD certification also
bind every actual u and v to the declared affine/constant family. Static shear
remains supported for B/W/KE/pressure work, while material shear refuses.
Derived specific stocks, density bounds and expected velocity must be finite.

Raw-owner resolution and raw width sums are supplemented by sorted interval
adjacency for every endpoint-layer. Equal overlap and gap cannot be accepted
because their total widths happen to cancel. Local arithmetic bounds include
absolute quadrature terms before cancellation, actual raw-to-specific/EOS-fit
chains, moving-top operations and pressure traction operations. The original
endpoint inventory/PE functionals, source authority and qualification boundary
remain as frozen above.

Global strip pairing is also verified in a common bottom-to-top order at both endpoints, with continuous paired boundaries and exactly one final actual top. Independent endpoint width closure alone cannot prove that linear strips partition the interior physical domain.

## 2026-10-02: reviewed physical projection witness

Scientific implementation commit:
`c24ebcdd3d6a2116e187d033234786104fd23332`, following the protocol-only commit
`ca523ba6ae63e54d30a32d086a7acf3dcdb2b651`. The public scalar witness is
[slope_dual_stock_evidence.json](slope_dual_stock_evidence.json). It binds 245
source files, the frozen protocol and resource runner to the clean scientific
commit. Seven manufactured cases pass 11,362 independent gate rows. No private
archive arrays or machine paths are included. Documentation commits preserve
that original source lineage instead of rebinding earlier runs.

The positive sloped case retains the original endpoint PE 723.2564745000001 J
and physical PE 94.08599325 J. Their 629.1704812500001 J gap comprises reference
free-PE 628.453125 J and anomaly PE 0.7173562500000799 J. Its raw-minus-projected
IS defect is 0.06418485237486493 PSU*m^3 (analytic 50/779); Mu defect is
4.100000000000023 kg*m/s. The corresponding material defects are
-0.0015404364570024143 PSU*m^3/s and -0.09839999999999982 N. Endpoint PE direction
-80.2357884045 W remains separate from physical direction -60.08473631700005 W:
the -20.15105208749995 W discrepancy is still present. B is an auxiliary
physical projection, not a raw conservative remap that removes these gaps.

Independent per-basis Bdot and weighted local ALE balances have maximum
residuals 8.881784197001252e-16 and 7.216449660063518e-16 across both flow signs
and flat/sloped surfaces. Every row uses the frozen absolute-operation scale.
Removing actual-top shape, the weighted basis term, gravity vertical-motion
source or an active virtual cut produces a resolved local failure. Rebuilding
the moving normalized chart gives a different per-basis mass direction and
fails the frozen-chart comparison. Deep projected stocks change: positive
sloped maximum deep IS and Mu directions are 0.007660110300787457 and
0.17434297054710574 in their integrated stock units per second.

For the same case, the 0.01-second fixed-W auxiliary scratch has KE change
-5.984089514795747 J and independent midpoint pressure work
-5.9840895147957465 J. The actual projected initial momentum and solved final
momentum bind to the physical Gram matrix; maximum solve residual is
8.881784197001252e-16 and minimum eigenvalue 15.760350644769114. Arbitrary
vertical shear passes static B/W/KE and pressure-work probes but refuses the
material transport tangent. Pressure force depends on geometry/thermodynamic
state, never on velocity or a fitted energy residual.

The separate zero-depth-flux shear initially has both endpoint depth fluxes
exactly 0 m^2/s, independently bounded by 1.1368683772161603e-13 each. Its
initial volume pressure power is 0.1720679005106298 W versus roundoff bound
3.5836715509484527e-10 W. This is measured before the scratch impulse; its
later auxiliary kick work 2.847058124059359 J is a different quantity.

Positive sloped one-sided PE directions at epsilon 2^-8, 2^-9, 2^-10 s are
-60.01663586075301, -60.050686088448856, -60.06771120248595 W. Their analytic
truncation bounds are 0.06810045972572783, 0.03405022942760192,
0.017025114604988928 W, with total bounds 0.06810097871820923,
0.03405126742073915, 0.01702719059944372 W. This is an affine-preserving
parameter derivative test with physical shape truncation, not a fixed-endpoint
time integration or second-order claim. A valid static a0=0 case intentionally
refuses the restricted one-sign remainder certificate.

The independent inspector's code/mathematical gate passed before the scientific
commit. Controls cover forged raw owners, missing strips, equal overlap/gap,
crossed left/right pairing, thermodynamic-spec mismatch, actual material shear,
derived division overflow, changed raw interiors and crossing. Retained failures
and successes are recorded in [slope_dual_stock_resources.json](slope_dual_stock_resources.json).
Early 14-test passes were superseded by stronger independent controls. Initial
TDD collection lacked the new module; three oracle refusal controls then failed
before their fixes. Two later red runs were error-message regex mismatches:
existing EOS refusal and the stronger topology refusal both worked. No frozen
scientific threshold was relaxed to make these pass.

Reproduce in a project-local Python 3.12 environment with existing dependencies
and `research/experiments/material_top_band/affine_requirements.lock` constraints:

```text
python -m pytest tests/research/contracts/test_slope_dual_stock.py -q
python -m ruff check .
```

From a clean committed checkout, the measured Windows Job Object runner enforces
one CPU, 180 seconds and 4 GiB process-tree private memory. Save new receipts to
the ignored logs directory so that source provenance remains clean:

```text
.venv\Scripts\python.exe scripts\run_bounded_research_tests.py --module research.experiments.material_top_band.slope_dual_stock_evidence --output logs/slope_dual_stock/reproduced_witness.json
.venv\Scripts\python.exe scripts\run_bounded_research_tests.py tests/research/contracts/test_slope_dual_stock.py tests/research/contracts/test_affine_physical_pressure.py tests/research/contracts/test_inventory_pressure.py tests/research/contracts/test_material_real_geometry.py tests/research/contracts/test_fd_static_bridge.py tests/research/contracts/test_pressure_force_geometry.py -q
```

Qualification remains false, accepted steps and real-archive steps remain zero.
There is no inverse/conservative raw projection, original momentum-CV adapter,
variable-mass total-energy compatibility, general P1/limiter, full moving sigma,
complete predict/12fast/replay, real353 failure step, dt300/600, GPU, MOM6
same-condition/equal-error speed result or default-production switch. Those
industrial requirements remain open.


### Final bounded validation and review

The final focused suite passed **26 tests**, 9.45 s pytest / 9.844 s job,
peak interpreter RSS 55,824,384 bytes and tree private memory 71,589,888 bytes.
The clean-source scalar witness took **14.265 s**, RSS 39,849,984 bytes,
private 67,354,624 bytes. Independent evidence review verified all 245 actual
source hashes, the unmodified frozen protocol, all finite gate results and
qualification boundaries. Its gate passed without running another numerical
worker. Across the witness, 84 gates and 11,362 rows pass; maximum observed
residual/bound ratio is 0.9999983919163277, in the analytic FD envelope tests.

The full `tests/research/contracts` directory was attempted once and stopped at
the unchanged 180-second wall limit: job 180.156 s including termination,
RSS 1,062,567,936 bytes, private 1,217,372,160 bytes, exit 124. It is incomplete
and is not reported as a passing regression. The directly related six-file
command above subsequently passed **210 tests**, 14.31 s pytest / 14.781 s job,
RSS 59,035,648 bytes and private 75,292,672 bytes. Both official runs used the
clean scientific commit. Repository Ruff and patch whitespace checks pass.
All numerical invocations were serial with one CPU, 180 seconds and 4 GiB;
there were no accepted model steps or real archive integration.

The final source remains a research diagnostic with zero accepted states; the broader production and industrial gaps above remain open.
