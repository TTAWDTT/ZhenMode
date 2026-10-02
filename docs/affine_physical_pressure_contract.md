# Affine fixed-physical pressure continuation

## 2026-10-02: protocol before numerical execution

This independent instantaneous probe starts at main
`87b956f388f6a3e13366184778a6aea010c0bdfd`. PR21's real force failure,
PR22's constant-density sigma result and all historical evidence are retained.
The new protocol fixes cases and bounds before execution. Qualification is
false for production, original fourteen-slot momentum consumption and real353.

Authority is copied h/IT/IS/Mu/Mv with the historical EOS. Pressure uses density
means in the exact stable affine family, never separate tracer limiting.
Affine secant ties permit only affine-preserving tangents. General P1,
endpoint clipping, stepped bottoms, dry/periodic patches and crossing reject.

The dual is the full physical trapezoid between two endpoints, extruded by L.
Its initial partition includes both irregular fourteen-layer columns. Interior
and virtual cut endpoints remain fixed physical coordinates for the tangent;
only the actual free surface moves. Auxiliary endpoint areas are dL/2. They
are not the original global momentum control volumes.

Every endpoint-layer velocity basis receives weighted four-edge traction plus
the volume pressure times basis-divergence term. Independent volume pressure
gradient quadrature and epsilon strip-load geometry perturbations verify every
basis. The pressure field is frozen during load perturbations and analytically
extended; it is not re-equilibrated on deformed geometry. Discontinuous shear
bases give strip-load variations, not a smooth global material deformation or
a general inventory-PE gradient. Pressure cannot depend on current velocity,
Q or an observed energy residual.

The manufactured material tangent has u=U+alpha*x, w=-alpha*(z-b). Density and
free-surface directions follow the continuum transport equations. Exact P1
physical ALE flux is new diagnostic algebra, not PR22's P0 upwind transport.
All cuts and open-side fluxes remain explicit. Internal virtual fluxes are
checked locally before cancellation. No unexplained residual is called loss.

PE contains rho0 free PE and anomaly gravity moment once. Physical volume
B(y)/Bdot and endpoint-stock PE are evaluated separately. Flat eta and the
restricted affine tangent give an inventory positive control. Unequal eta
gives a prederived endpoint-versus-volume PE gap and refusal. External pressure
is the declared linear lift: endpoint compensation does not invent a quadratic
zero-pressure-gradient state. Outward open-side pressure plus gz*rho flux and
top external-pressure work are counted once.

One-sided raw stock perturbations independently reconstruct scalar density P1
and integrate layer PE, including ds*h^3/12+s*h^2*hdot/4. Their truncation
majorant is separate from arithmetic roundoff. Scratch fixed-mass impulses use
real M_before/M_after and midpoint velocity. Their KE is the nodal lumped
metric; the interpolated volume KE difference is reported separately.

The new probe returns no accepted state and executes no Euler/full step. Full
current-mass KE, general P1/limiter derivatives, crossing, bottom steps,
predict/12fast/replay, real353 and MOM6/equal-error-speed adapters remain open.
Numerical checks are serial, one CPU, 180 seconds and 4 GiB per invocation.
