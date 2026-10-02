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
