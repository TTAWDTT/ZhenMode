# Manufactured fixed-eta raw pressure slice

## 2026-10-02: feasibility and contract before execution

Start from merged main `9ed44b18fa6af4edb18e20bcccd0d1b6e8dacadd`.
For fixed h, raw momentum r=D*u and physical projected momentum mu=W*u give
the unique same-velocity map R=W*D^-1. W being positive does not make R
conservative. The existing sloped fixture has incompatible raw and physical
total impulse, IS and PE. One cannot append conservation constraints to this
unique inverse and then repair its residual. If velocity is allowed to change,
total impulse/energy constraints alone leave many degrees of freedom; that is
not preservation of the original physical field.

Flat eta gives W*ones=D*ones and hence algebraic global conservation of R and
its inverse. This does not identify arbitrary endpoint coefficients as raw CV
means. A general manufactured half-prism mean basis would require a new Q/Psi
contract, and moving geometry would also require Qdot/Psidot. Neither is part
of this round. The accepted subspace uses uniform actual velocity at all raw
layers and both columns, where means and coefficients coincide.

This round defines an explicit manufactured raw CV patch: adjacent half-prisms
x=[0,d/2] and [d/2,d] of length L, each with its actual fourteen horizontal
layers, common flat eta and bottom. It does not claim these are the original
global solver's momentum control volumes. T=Tref and common stable affine
rho_prime(z) permit exact P1 TS face integration. Open side fluxes are nonzero;
surface, bottom and fixed internal horizontal interfaces are impermeable.
Linear external pressure provides a nonzero, depth-independent pressure force.

Every raw face segment has one pressure integral and one integrated water/TS/M
flux. Common segments use the union of physical depths and are consumed by both
neighbors with opposite signs. Outer faces independently close every raw row.
The uniform pressure force lies in the compatible subspace:
Fraw=-(P/rho0)*D*ones and Fmu=-(P/rho0)*W*ones=R*Fraw. Full-vector equality is
tested, not only total impulse or pressure power. A finite research pressure
step updates actual raw Mu, verifies all finite-time ledgers and commits the
complete copied ColumnStocks once. Rejection rolls back the whole state,
time, accepted count and receipt. Production qualification remains false.

For actual before/after momenta with changing symmetric mass matrices, the
required identity is

```text
DeltaK = u_mid^T DeltaM - 0.5*u_after^T DeltaMass*u_before.
raw_r_dot = D_dot*u + D*W^-1*(mu_dot - W_dot*u).
```

The accepted slice has fixed mass. Moving snapshots are independent obstruction
and algebra tests, not accepted manufactured evolution. Moving-alpha weighted
physical flux requires a separately defined local raw quadrature-storage
transfer; global conservation of the inverse cannot replace that raw row proof.
The changing raw-minus-physical KE metric difference is recorded separately,
never called physical loss. Alpha, horizontal density gradients, shear and
sloped eta remain rejected. Original global CV identification, moving geometry,
general transport, fast/replay, forcing/mixing adapters and industrial/equal-error
speed qualification remain open.
`n### Pre-execution time-flux and transverse-boundary clarification`n`nUse exact J1, J2 and J3 as frozen in the JSON, not Q times midpoint u for Mu. The sign-crossing control has zero water flux and positive transported Mu. The slice is xz extruded with periodic y; V has no unreported outer-wall source. All face fluxes are computed and consumed before checking the zero row divergence.

## 2026-10-02: readable time-flux clarification and absolute face gates

The preceding pre-execution clarification contains literal newline escape
characters from serialization. Its mathematical meaning is restated here;
the historical prefix is preserved.

Use the frozen exact J1, J2 and J3 time integrals. Mu flux is rho0*L*dz*J2,
including a sign crossing with J1=0 and J2>0. The slice is xz extruded with
periodic y, so opposing transverse transports and pressure forces cancel.

The actual left and right inventory P1 hydrostatic pressures are integrated
with two-point Gauss; the common pressure is their arithmetic mean, once.
Actual pressure integrals, water/TS/M transport and KE/gravitational PE flux
at each shared and each outer face are independently bound before any raw
commit using seven-point time/depth quadrature. This gate reads actual raw
P1 fields and integrates overlying raw layers, and does not call the candidate
pressure primitive or candidate polynomial time moments. Each raw row consumes
its separate signed outer and common fluxes, including KE/PE diagnostic fluxes.
The frozen common affine thermodynamic field is qualified against every actual
raw stock; roundoff in reconstructing that field is covered only by primitive
EOS operation bounds. No residual is fitted or repaired.

Common flux errors that preserve zero divergence are explicitly rejected:
shared and both outer stock fluxes set to zero or P0, all three pressure values
given a common offset, and shared/outer KE or PE jointly erased. The entire
state, time, count and last receipt remain unchanged on these failures.
KE/PE flux diagnostics do not add a new prognostic energy stock. The accepted
raw h/IT/IS/Mu/Mv state remains the single authority.

The accepted family also requires inactive actual TS and density limiters. Actual P1 T/S endpoints must bind the common affine EOS field; the independent face gate integrates the actual auxiliary T/S reconstruction separately from direct EOS density P1.
