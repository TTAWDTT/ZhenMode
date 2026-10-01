# Preregister source-carried momentum witness, not a frozen-kernel bug claim

2026-09-29, before execution. Whole industrial objective unchanged.
Current frozen kernel accepts volume/tracer sources but has no declared
incoming horizontal momentum velocity. It must not be promoted to physical
rain/source dynamics solely from its frozen energy PASS.

Use full-wet irregular spherical grid, uniform positive-east velocity0.1m/s,
zero north speed, zero density force/gravity/rotation, zero initial eta.
Apply uniform rain1e-6m/s for600s into the top layer, with constant incoming
tracer concentration and ASSUMED zero incoming east/north velocity in this
nonrotating experiment. No external axial torque and periodic/closed walls.

Independent exact physical axial momentum perrho0 is

    L = u*R^3*sum(dlambda*integral_phiS_phiN cos(phi)^2 dphi*H).

The source supplies ZERO axial momentum under the explicit assumed source
velocity. With zero other transport/forces, top speed should dilute by
Htop/(Htop+rain*dt); lower speeds stay fixed. Mixing also changes kinetic
energy. Frozen-M energy can stay constant while actual endpoint mass-weighted
momentum/energy are wrong for this source contract.

Record actual result, analytical expected source momentum, actual axial
change, frozen work, moving-mass energy and independent kinetic changes.
Do NOT fix it with arbitrary Mdot damping or a global velocity rescale here.
If axial relative change exceeds1e-12 (plus64eps arithmetic floor), this is
a required NEW nonlinear/source qualification FAIL, not failure of the
previously scoped frozen external-force equations or an old production claim.
The next method must evolve source-carried momentum and moving mass together.
