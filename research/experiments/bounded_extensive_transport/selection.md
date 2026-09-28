# Precision controls and initial method selection

2026-09-29, before any higher-order core implementation or accuracy result.
Protocol `8803e73`; frozen donor kernel `cb38ba7`.

The prescribed-Q control completes all twelve 100-step cases. Native32 bound
excursions are 3.101e-4 (constant) and 9.912e-5 (patterned); guard arithmetic
with32 storage gives the same excursions. V32/N64 and V64/N32 both still fail
the unchanged 2e-6 bound. V64/N64 and reconstructed two32 expansions show zero
global-bound excursion on this fixture. These implicate BOTH stored inventories
for this control, not just arithmetic. This is not a replay of the previous
barotropic trajectories or proof of their complete causal decomposition.

Choose explicit V/N64 inventory with eta/face velocity32 permitted, rather than
automatic whole-state promotion. The two32 expansion consumes exactly the same
73728 inventory bytes as V/N64 versus 36864 native32 bytes; it additionally
reconstructs and splits in64. This is not a GPU timing or full-model memory
comparison. Preserve the same-dtype donor path and reject unregistered mixtures.

Initial high-order candidate: centered metric-weighted linear slopes, zero slope
where either side is closed, with all-direction extensive FCT and SSPRK2.
The slope uses actual center distances; face extrapolation uses half cell widths
and current V/A vertical thickness. On uneven spacing, the derivative is
(d_right*g_left + d_left*g_right)/(d_left+d_right). No preliminary minmod is
applied: the shared multidimensional FCT limits the resulting face amounts.
Local bounds use old wet neighbors; only explicit forcing extends them by the
forced low-order state and its wet neighbors. Antidiffusive capacities use NEW
volume. This is a chosen candidate, not a demonstrated second-order method.

SSPRK2 averages the incoming state and the second Euler stage in BOTH V and N.
Q and sources are frozen across the two stages. Both stages must independently
pass the original physical/CFL gates. The registered cosine L2 ratios >=3.2,
budgets, extrema and mixed real-grid gates remain unchanged. If extrema limiting
destroys the convergence gate, retain the failure and research a new method;
do not weaken bounds or call SSPRK2 alone second-order qualification.

Primary equations checked: Zalesak 1979 sections IV/VI (positive/negative
multi-face capacities and a shared minimum per face), and Gottlieb/Shu/Tadmor
2001 sections 2.2/4 (convex Euler decomposition). Their scope does not qualify
our moving-volume, finite-precision implementation without independent tests.

Raw controls: `results/industrial_alignment/bounded_transport_precision.json`.
The tracked frozen-kernel probe reproduces them. Full industrial roadmap and
production-cutover requirements remain intact.
