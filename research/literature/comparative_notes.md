
# Comparative notes on major ocean models

This note distills a review of public documentation and source trees from mature
ocean models into what is most transferable to `ocean_solver`. Third-party
source snapshots are no longer vendored in the repository.

## Scope and evidence

The models below are mature, widely used, and cover the main architectural families:
structured lat-lon finite-difference/finite-volume, unstructured finite-volume,
icosahedral/icosahedral-triangular C-grid, and hybrid-coordinate systems.

Evidence here is from official docs and source-code trees, not from informal blog posts.

## Cross-model comparison

| Model | Grid / coordinate | Core discretization | Time stepping | Transport / closures | Scalability and software discipline |
|---|---|---|---|---|---|
| MOM6 | structured/global, generalized vertical coordinate | layer-integrated vector-invariant hydrostatic primitive equations | split explicit + ALE/remap | GM/Redi, MEKE, backscatter, mixed-layer restratification; KPP/ePBL/CVMix | modular source tree, driver/coupler separation, strong diagnostics |
| MITgcm | curvilinear lat-lon/cube, hydrostatic + nonhydrostatic | finite-volume, C-grid | pressure method with implicit/free-surface variants; Adams-Bashforth 2/3 | rich package architecture, adjoint, flexible EOS | modular packages, domain decomposition, adjoint/tool ecosystem |
| NEMO | orthogonal curvilinear, mostly z/terrain-following | finite-difference/finite-volume hybrid, C-grid | explicit/semi-implicit split-explicit free surface | many tracer advection schemes: CEN, FCT, MUSCL, UBS, QUICKEST; GM/Redi; TKE/OSM/GLS | modular dirs, operational/climate use, broad community ecosystem |
| ROMS | curvilinear horizontal, sigma terrain-following | finite-difference/finite-volume hybrid | split explicit, FB AB3-AM4 or LF-AM3 | MPDATA/TVD, KPP/MY2.5/GLS, GM/Redi, wet/dry, OBCs | strong regional model, DA, nesting, boundary-condition discipline |
| POP2/CESM | structured lat-lon, z | finite-difference/finite-volume hybrid | implicit free-surface barotropic solve | GM/Redi, KPP, harmonic/biharmonic | mature block-structured MPI; CESM notes POP2 is being replaced by MOM6 |
| MPAS-Ocean | unstructured Voronoi, variable-resolution | finite-volume C-grid | split explicit | ALE, FCT-like monotone transport, GM/Redi, Leith, CVMix | scalable unstructured mesh, MPI+OpenMP, strong variable-resolution use cases |
| FESOM2 | unstructured triangular, variable-resolution | finite-volume | split explicit | full 3D FCT limiter, implicit stabilization of extreme vertical advection, isoneutral diffusion, backscatter | excellent scalability, partial cells, z-star default, Icepack, well-documented mesh tools |
| ICON-O | icosahedral-triangular C-grid, z/z-star | mimetic finite-volume with Hilbert-space compatible reconstructions | explicit with implicit pieces | GM, harmonic/biharmonic, Smagorinsky/Leith, TKE, IDEMIX | coupled to atmosphere/land/sea-ice, supports GPUs and local refinement |
| HYCOM | hybrid isopycnal/z/sigma | finite-volume / isopycnal-layer | hybrid-coordinate treatment | KPP, Kraus-Turner, Mellor-Yamada, PWP, detailed vertical mixing docs | operational global forecast system with DA |
| FVCOM | unstructured triangular, coastal/regional | finite-volume | explicit/semi-implicit, hydrostatic/nonhydrostatic | MPDATA/TVD, wetting/drying, nesting, rich tracer/biogeochemistry | coastal/estuarine production model with MPI/OpenMP |

## Lessons that matter most for ocean_solver

1. **Transport is the first robustness lever.**
   Mature models do not stop at donor-cell. They usually offer a bounded/monotone
   high-order scheme (FCT, MUSCL, TVD) and often apply the limiter on the full
   3D flux. FESOM2 is a particularly useful example: it uses first-order upwind
   as the low-order scheme, a high-order scheme as the base, and a 3D FCT limiter
   on the full fluxes.

2. **Vertical coordinate flexibility is a second robustness lever.**
   MOM6, MPAS, FESOM2, ICON-O, and HYCOM all decouple the physical equations from a
   fixed z-grid. z-star/ALE/remapping lets the model handle topography, mixed layer,
   and free-surface effects without exposing the transport code to raw vertical
   stretching. This is especially relevant when bottom topography or narrow seas
   create pathological columns.

3. **Split-explicit free surface is ubiquitous.**
   MOM6, ROMS, MPAS, FESOM2, ICON-O, NEMO, MITgcm, and POP2 all have mature
   free-surface/barotropic treatment. ocean_solver already has this, but the
   mature models make the implicit/explicit treatment, subcycling, and mask
   behavior more explicit.

4. **Parameterizations are modular, not model-specific.**
   KPP, GM/Redi, harmonic/biharmonic, Smagorinsky/Leith, TKE, IDEMIX, and internal
   tide mixing recur across models. A clean closure interface is more important
   than adding more closures quickly.

5. **Boundary/mask discipline is encoded everywhere.**
   Face-gated fluxes, wet/dry masks, no-flux walls, halo exchange, and partial
   bottom cells are not optional. They are the reason the mature models can run
   long integrations without tracer pileup at coasts.

6. **Diagnostics are first-class.**
   MITgcm, MOM6, NEMO, ICON, MPAS, POP, and FESOM all separate diagnostics from the
   dynamical core. This makes it possible to test budgets, energy, transport,
   and parameterizations without touching the core solver.

## What ocean_solver should do next

### Near term
- Add a full 3D flux-corrected-transport option for tracers, with a low-order
  monotone fallback and a high-order centered/PPM-like base. This is more robust
  than donor-cell alone.
- Add a budget/diagnostic layer that reports heat, salt, mass, and energy residuals
  per step, not just end-state maxima.
- Make the wet/dry and open/closed face masks a single, explicit, testable contract
  used by advection, diffusion, pressure, and free surface.

### Mid term
- Add a conservative remap layer for vertical coordinate flexibility, starting with
  z-star or partial bottom cells. This will reduce pathological narrow-sea behavior.
- Consider an optional implicit vertical-advection stabilizer for extreme vertical
  Courant numbers, following FESOM2's explicit/implicit vertical split.

### Later
- If multi-GPU is needed, use JAX sharding/`pjit` for structured lat-lon first
  rather than jumping to unstructured meshes.
- Only consider unstructured mesh after the structured solver has a mature,
  well-tested closure stack and diagnostics layer.

## What not to copy blindly
- Do not chase nonhydrostatic dynamics yet. Most global climate OGCMs are hydrostatic.
- Do not rewrite into unstructured mesh until the structured model's numerics are stable.
- Do not add more closures before the transport/coordinate layer is robust.
