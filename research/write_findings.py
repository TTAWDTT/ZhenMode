from pathlib import Path
p = Path('research/findings.md')
p.write_text(r'''
# Research findings

## Research question
What numerical, physical, and software choices make widely used ocean general
circulation models robust, scalable, and scientifically credible, and which should
`ocean_solver` adopt next?

## Current understanding
The strongest transferable signal is not a single exotic scheme. It is a stack of
conservative transport, disciplined masks, generalized vertical coordinates,
modular closures, and first-class diagnostics. The surveyed models differ a lot in
mesh and coordinate choice, but they converge on these same engineering principles.

## Key results

### Model families surveyed
| Model | Grid / coordinate | Time stepping | Transport / closures | Why it matters |
|---|---|---|---|---|
| MOM6 | structured, generalized vertical | split explicit + ALE | GM/Redi, MEKE, KPP/ePBL, rich lateral/vertical closures | Most directly relevant modern global OGCM |
| MITgcm | curvilinear/finite-volume | pressure method + implicit/free-surface variants | rich packages, adjoint | Strong numerical generality and ecosystem |
| NEMO | orthogonal curvilinear | split-explicit free surface | CEN/FCT/MUSCL/UBS/QUICKEST; TKE/OSM/GLS | Mature European community model |
| ROMS | curvilinear sigma | split explicit, FB AB3-AM4 or LF-AM3 | MPDATA/TVD, KPP/MY2.5/GLS | Excellent coastal/regional robustness |
| POP2/CESM | structured lat-lon, z | implicit free-surface barotropic solve | GM/Redi, KPP, harmonic/biharmonic | Mature CESM ocean; now being replaced by MOM6 |
| MPAS-Ocean | unstructured Voronoi | split explicit | ALE, monotone transport, GM/Redi, Leith, CVMix | Scalable variable-resolution mesh |
| FESOM2 | unstructured triangular, variable-resolution | split explicit | full 3D FCT, implicit vertical-advection stabilization, isoneutral diffusion | Excellent scalability and robust transport |
| ICON-O | icosahedral-triangular C-grid | explicit with implicit pieces | GM, harmonic/biharmonic, Smagorinsky/Leith, TKE, IDEMIX | Modern mimetic discretization and GPU support |
| HYCOM | hybrid isopycnal/z/sigma | hybrid-coordinate treatment | KPP, Kraus-Turner, Mellor-Yamada, PWP | Operational forecast system with DA |
| FVCOM | unstructured triangular | explicit/semi-implicit | MPDATA/TVD, wetting/drying, nesting | Coastal/regional production model |

### Dominant numerical pattern
Across the mature models, robustness comes from the combination of:
1. conservative finite-volume/finite-element operators,
2. a free-surface treatment that is explicitly implicit or split-explicit,
3. conservative/monotone tracer transport,
4. disciplined wet/dry and open/closed face masks,
5. flexible vertical coordinate treatment,
6. modular closures,
7. first-class diagnostics.

## Patterns and insights

- **Transport robustness is not just "monotone vs centered".**
  FESOM2 and NEMO both use low-order fallback plus higher-order base schemes,
  usually with a limiter. FESOM2 applies the limiter to full 3D fluxes, which is
  especially relevant because vertical advection can be the practical bottleneck
  near coastlines.

- **Vertical coordinate flexibility is common among the strongest models.**
  MOM6, MPAS, FESOM2, ICON-O, and HYCOM all use generalized vertical coordinate
  treatment or ALE/remap. This is a mature way to handle mixed layers, topography,
  and narrow seas without making the transport code grid-specific.

- **The closure stack is common, not exotic.**
  KPP, GM/Redi, harmonic/biharmonic, Smagorinsky/Leith, TKE, IDEMIX, and internal
  tide mixing recur across models. The important design idea is a clean closure
  interface, not one-off special cases.

- **Boundary discipline is the hidden foundation.**
  Mature models encode no-flux walls, wet/dry masks, open/closed faces, halo
  exchange, and partial cells as core invariants. This is where `ocean_solver`
  has already found several issues, and it should continue to be treated as a
  first-class concern.

- **Diagnostics are not optional.**
  MITgcm, MOM6, NEMO, ICON, MPAS, POP, and FESOM all separate diagnostics from the
  dynamical core. This makes it possible to test budgets, energy, transport,
  and parameterizations without touching the core solver.

## Lessons and constraints
- The next highest-leverage improvement is likely a full 3D FCT/MUSCL-style
  conservative transport option, not immediate unstructured-mesh work.
- A budget/diagnostic layer should be added before adding more closures.
- Vertical-coordinate flexibility is important, but it can be introduced gradually
  through conservative remap/z-star rather than a full rewrite.
- JAX remains a good implementation vehicle, but mature models suggest separating
  core, forcings, closures, and diagnostics more explicitly.

## Open questions
- What is the minimal ALE/z-star remap layer that helps `ocean_solver` without
  destabilizing the current 1° baseline?
- Which tracer transport limiter is easiest to implement correctly in JAX?
- Can the existing wet/dry face contract be made explicit enough to become a
  reusable test invariant?
- How much of the current performance comes from whole-step JAX compilation, and
  how much would be lost if the code is reorganized for modularity?

## Optimization trajectory
No numerical experiments yet. This phase is a literature/architecture survey.
''', encoding='utf-8')
print(p.read_text(encoding='utf-8')[:300])
