# Physical partial-face rotation witness before nonlinear production migration

Registered 2026-09-29 before running this additional physical witness. The
eight100-step density/linear-momentum cases pass their stated gates, including
energy preservation and the chosen sqrt(W) quadrature. Those results do NOT
establish physical consistency of that quadrature/interpolation on every partial
face. Do not infer correct acceleration simply from a skew matrix or reuse its
coefficient with f+relative_vorticity as complete nonlinear momentum.

Re-read [MOM6 thickness/PV Coriolis](https://mom6.readthedocs.io/en/main/api/generated/pages/Discrete_Coriolis.html)
and [MITgcm momentum sections2.14--2.15](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html):
energy-compatible cross interpolation includes physical thickness/transports;
vertical shear, Bernoulli gradient and metric terms require compatible volumes
and the actual continuity transport. MOM6's Lagrangian/PV formulas are not a
literal fixed-z partial-cell implementation. MITgcm also distinguishes its
energy-conserving and historical velocity-averaging options. Read equations,
not just the claim that both use C grids.

Additional input: longitude-global8x8, latitude edges +/-0.001degrees, vertical
interfaces [0,5,20,50]m. All columns50m except longitude column3, which is21m.
Its bottom layer is1m; neighbor bottom layers are30m. Set u=0, every open north
face v=2m/s, constant f=0.001/s. On thin east faces2/3, interior latitude rows2:6,
all four contributing north velocities are2 throughout the east face's common
wet interval20:21m; their vertical extent below21m must not amplify that local
Coriolis acceleration. The independent depth integral is f*2=0.002m/s2.
Require relative error <=1e-8 (same near-equatorial consistency tolerance as the
existing constant-depth witness); also report implicit60s rotation energy/error.

Write measured versus analytic accelerations, maximum ratio and energy gate,
source/runtime hashes; preserve FAIL. If it fails, repair physical cross-face
overlap/dual-volume consistency BEFORE nonlinear momentum or production cutover.
Do not fix by clipping a force, damping, removing shallow cells or relaxing the
registered equality. Establish the correct physical momentum quadrature and
moving-top common overlap, then re-run all old direct/reference gates. A pass
would still not prove nonlinear advection, whole-model conservation or climate.
