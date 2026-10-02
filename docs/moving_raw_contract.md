# Manufactured moving raw means and characteristic ALE slice

## 2026-10-02: design before numerical execution

The base is merged main `7223a5d533f95bc289a35b2dab62901c2b2776ae`.
The next pilot uses a new, explicitly declared manufactured raw mean authority.
It does not identify the original global momentum control volumes. The patch
has two adjacent half-prisms, each with its own fourteen horizontal raw layers.
The first raw layer changes thickness; interior cuts and the bottom stay fixed.
This is different from the original moving top-three-band design.

Actual raw `h, IT, IS, Mu, Mv` remain authoritative. Velocity means are located
at x=d/4 and 3d/4. The old fixture stores endpoint velocities; interpreting those
as half-CV means changes the physical field. The new fixture integrates actual
half-CV means explicitly and uses independent declared physical parameters.

### Mean basis and moving frame

Let O be the actual physical-depth overlap between the two raw layer partitions,
D=diag(rho0*A*h), A=d*L/2, and

```text
K = [[0, O/hL], [O.T/hR, 0]]
Q = .75*I + .25*K
Psi = N*Q^-1
W_N = D*(2*I + K)/3
M = Q^-T * W_N * Q^-1
```

Q maps endpoint coefficients to actual half-prism means. D*K is symmetric and
the D-weighted spectrum of K lies in [-1,1], so Q is invertible without a fitted
constraint or regularization. Q*ones=ones and M*ones=D*ones; the same-field
momentum map R=M*D^-1 and its inverse conserve global impulse. This is a new
mean basis, not reuse of the old endpoint metric. In exact geometry arithmetic,
D <= M <= (4/3)*D. M-D is the actual within-CV reconstruction covariance.

The domain interior endpoint indicators N stay fixed, while the top shape moves.
Both inverse-side derivatives are required in Mdot. Qdot contains overlap motion
and the own-volume denominator derivative; Rdot also contains Ddot. Qdot*c=0
on a depth-uniform affine field does not eliminate every test-basis frame term.
Separate general-basis/shear geometry probes test the full matrices and frame
terms, but do not qualify a general sheared flow. Their finite-difference
truncation envelope is separate from roundoff identities and time quadrature.

### Restricted finite-volume method and independent oracle

The accepted family has common flat eta and bottom, T=Tref, inactive actual
T/S and density limiters, common stable affine density in z, globally affine
horizontal velocity, and constant transverse V. There is no horizontal density
gradient. Every actual mean and both endpoints of each auxiliary P1 reconstruction
must qualify the canonical affine field within the frozen primitive-operation
envelope. The original stocks are never changed to impose that qualification.

The candidate follows constant-pressure-gradient horizontal particle acceleration
and the incompressible vertical Jacobian. At every physical face it traces
inverse particle labels, integrates physical transport and Pa tractions, and
updates IT/IS/Mu/Mv only by signed FV fluxes and pressure impulses. The analytic
end-state array is never used to replace accepted stocks. The ALE grid trajectory
defines h geometrically before flux evaluation. Every raw water row must match
the consumed Q/R ledger; failures reject, without changing Q/R or moving the
mesh to remove a residual. Local, global and cumulative floating-point GCL
defects and their predetermined bounds are retained.

One shared horizontal union-depth face is consumed by both raw owners with
opposite signs. Each column's internal vertical ALE R is consumed by both
adjacent layers. Surface and bottom relative material fluxes vanish. Open
inflow is specified by the globally affine initial-field extension.

The spatial transport authority is explicitly the qualified canonical affine
reconstruction. This is not literal advection of a possibly discontinuous
initial auxiliary P1 profile. Initial endpoint deviations and primitive-operation
bounds yield separate reconstruction majorants. They cover both velocity
components, alpha/w, T/S, quadratic momentum flux, cubic KE flux, Pa, pressure
work, terminal stock quotients and inactive-P1 reconstruction. Absolute affine
weights cover open-inflow extrapolation. These errors are not added after seeing
an acceptance residual and do not become a physical loss or a hidden projection.

The oracle has its own Eulerian PDE derivation and does not import candidate
label, path, polynomial, consumer, mean-map or after-stock helpers. It binds
actual before/after stocks by independent end-CV volume integration, checks PDE
residuals, all absolute physical face quantities, local ledgers and full boundary
work. Agreement with the same analytic path alone would be insufficient.

### Energy and time integration envelopes

Physical KE uses actual M and actual mean velocities; raw mean KE uses D.
The difference is explicitly retained. For globally affine velocity,

```text
K_physical - K_raw = rho0*L*H*d^3*alpha^2/96.
```

Each raw CV consumes physical KE and gravitational PE transport and pressure
energy flux. Internal vertical p*R cancels between adjacent cells only after
both local budgets pass. The material top has R=0 but nonzero external Pa work
through its actual motion. Local rho0*g*z potential sums to the global rho0
free-surface PE after adding a fixed bottom constant; rho-prime gravity PE is
counted once. No vertical kinetic-energy claim is made for this hydrostatic slice.

The exact before/after mass-chain identity is checked independently for D and M:

```text
DeltaK = a_mid.T*DeltaMomentum - .5*a_after.T*DeltaMass*a_before.
```

This algebraic impulse work is distinct from true time-integrated horizontal
pressure work. Midpoint velocity times integrated pressure impulse is separately
reported too. Their nonlinear differences are not called dissipation or loss.

Candidate face integrands use canonical rational time polynomials. Numerator
degree and lambda denominator power are mechanically audited per execution
channel against the protocol's degree table. The absolute coefficient-operation
polynomial includes all physical length and area factors before cancellation.
The 16-point remainder is dimensional: 2*dt*sum_abs_coeff*T16. The independent
32-point oracle has its own coefficient majorants and T32 bound. Input
reconstruction, time quadrature and roundoff are separate budgets. Neither
minmod reconstruction nor Q/M/R top-thickness denominators are covered by a
lambda-only time-tail claim. No generic time-order claim follows from the
restricted exact-characteristic family.

### Frozen checks and limits

The positive and negative cases use alpha=+/-0.08, U=+/-0.03, V=-0.02,
external pressure [80,200] or [200,80] Pa, and successive 0.02 and 0.03 s steps.
Zero-pressure moving and alpha=0 fixed-eta controls are retained. Geometry,
mean/frame, input/material, absolute-face, energy and rollback controls are
frozen in `moving_raw_protocol.json` before numerical execution.

Every numerical invocation is serial, single CPU, hard wall 180 s and owned
process-tree memory 4 GiB. The first batch totals at most 20 minutes. Evidence
must bind its actual clean scientific source, with failed development runs
retained and not retrospectively rebound. Default production, real353, the
original CV, original top-three geometry and predict/12fast/replay remain
unqualified. Mixing, biharmonic, FCT/filter, wind, heat, rotation and drag are
not silently applied or qualified by this pilot. No merge or license change.

### Exact manufactured inputs

The base-7223 fixture partitions fourteen layers using normalized left weights
1 through 14 and right weights (14 through 1)^1.3. Initial a0=1 is the global
z intercept, so q_b=a0+s*b. Inherited inventory EOS is rho0=1025 kg/m3,
Tref=15, Sref=35, alpha=2e-4, beta=7.6e-4, g=9.81 m/s2.
The nonzero V direction is periodic extrusion; its paired transverse faces
cancel. These choices are frozen inputs, not inferred from candidate results.
