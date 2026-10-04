# Numerical design notes — global FD solver

Current discretization choices and limits. D identifiers remain stable for source references. Historical investigation logs are available in Git, rather than duplicated here. These notes do not certify every climate metric or whole-model physical inventory closure.

## D1 — Meridional derivative at the N/S walls: mirror ghost, not one-sided

Meridional derivative stencils use an edge-replicated ghost at the closed latitude wall. Normal advective velocity is masked separately in the complete and free-surface steps. Longitude is periodic.

## D2 — Spherical mass conservation: meridional flux needs cos(face)

Area weighting on the latitude-longitude grid requires the meridional flux form `div_y = (1/cos_j) * d(v*cos_face)/dy`, with `cos_face = (cos_j+cos_{j+1})/2`. Zonal and meridional closed-face fluxes must telescope under the actual area weights.

## D3 — Energy-neutral free surface: the gradient must be the exact adjoint

The pressure gradient and conservative divergence must satisfy the discrete weighted adjoint relation `<u, grad(eta)>_A = -<eta, div(u)>_A`. Conservative gradients use wet face differences and the corresponding spherical metric. This operator relation alone is not a complete time-step energy theorem.

## D4 — Coastal flux gating for the 3D skew-flux divergence

Three-dimensional conservative divergence and gradient close a face unless both adjacent layer cells are wet. Land and below-bottom sentinel values must not enter an open-face pressure or isopycnal flux.

## D5 — Laplacian face gating at wet/ghost faces

Closed coastal and bottom faces must have zero diffusive flux. Latitude boundary ghosts replicate the boundary value. In opt-in `nodal_dual_v1`, scalar momentum Laplacian and its square use the closed wet-face cosine-flux operator. Legacy geometry retains its existing stencil. The nodal operator is a scalar reference-metric discretization, not a full spherical vector-viscosity theorem.

## D6 — Face-flux divergence: the w diagnosis must see what advection sees

Vertical transport diagnosis uses the same layer face transports as tracer advection. At a coast, a centered derivative and the gated face-flux divergence differ; combining them breaks the transport identity. Surface column transport remains visible in the boundary diagnostics.

## D7 — Vertical transport from the exact discrete divergence inverse

`_vertical_transport_iface` integrates horizontal layer divergence upward from the closed bed, using per-node `dz_node` weights. Interface distances `dz_3d` have length `nz-1` and cannot replace layer inventory weights.

## D8 — Seafloor ghost fill for the closure stencils

`_fill_ghost_bottom` extends the bottom wet value into below-bed stencil positions. All callers subsequently apply wet masks or wet-face gates. Its value in an entirely dry column has no physical meaning and must not enter wet-cell dynamics.

## D9 — Vertical diffusion must telescope (interface-flux form)

Opt-in `conservative_kv=True` selects interface-flux vertical diffusion, whose thickness-weighted column sum telescopes to the top and bottom boundary fluxes. The nonlinear residual subtracts diffusion already applied by the two linear half-steps. Biharmonic diffusion belongs only to those half-steps; adding it back into the residual would cancel its damping. The legacy option retains its declared stencil.

## D10 — Convective adjustment must telescope, and should be localized

`_conv_flux_tendency` exchanges tracer through vertical interfaces. `localize_conv=True` gates individual unstable interfaces; column-wide gating mixes the full column. These are different parameterization choices. The column-weighted internal exchange must cancel, and hard density gates have nonsmooth differentiation boundaries.

## D11 — Hydrostatic pressure: mask the ghost water before integrating

Hydrostatic pressure masks density anomaly below the seafloor before vertical integration. Pressure remains constant below the bottom. Horizontal pressure gradients also need closed wet-face gating.

## D12 — Mode split, and right-sizing the nu_h subcycle count

With `mode_split=True`, external free-surface dynamics use `n_subcyc` fast steps after the slow update. Horizontal momentum diffusion acts on the full three-dimensional velocity in linear half-steps. Its subcycle bound accounts for both `1/dx²` and `1/dy²`; sizing from meridional spacing alone is insufficient. `nu_nsub=None` retains the existing fast-step count; `cfl` uses the declared diffusion margin. Stability and speed remain configuration dependent.

## D13 — Stage-2 advection velocity and the column heat leak

Tracer stage-2 velocity is an explicit choice. `project_adv_vel=True` constrains the predictor through the native wet-layer transport operator described in D34–D36. `freeze_adv_vel=True` uses the old velocity. Neither choice alone establishes moving-volume/free-surface inventory consistency. Actual top tracer transport is recorded with its declared sign.

## D14 — Face-gated horizontal gradients for advection

Horizontal advection gradients close wet/dry faces so reference sentinel values cannot contaminate an open face. This gating uses the same boundary geometry as the conservative transport operators.

## D15 — Scalar advection must be flux form

Scalar transport is in flux form. Momentum retains its declared advective form. Below-bottom momentum stencils use ghost-filled values and wet masks; scalar and momentum equations are not interchangeable discretizations.

## D16 — Donor-cell vertical tracer flux

Vertical scalar face values are donor-cell values in every supported mode. Subcycling uses frozen velocities and horizontal transport over the declared interval. Flux signs are outflow minus inflow, with the tendency carrying the negative divergence. Surface transport uses the surface concentration and remains an explicit boundary term. Donor-cell monotonicity requires the appropriate substep CFL; it is not an unconditional complete-model guarantee.

## D17 — Slope limiter: DM95 taper, not a tanh clip

Isopycnal slopes use the DM95 taper and declared slope limit, rather than a `tanh` replacement. Apply wet masks and seafloor ghost treatment before computing slope-sensitive gradients.

## D18 — GM/Redi: one skew-flux operator, in interface-flux form

GM/Redi uses a shared skew-flux operator and interface-form vertical fluxes. Internal exchange must telescope with the actual layer weights and closed boundaries. Conservation of an isolated operator does not establish full parameterization or climate accuracy.

## D19 — The sponge relaxes toward the MASKED initial state, not the raw one

Sponge targets are masked initialized temperature and salinity. They must not pull a below-bottom or dry reference value into a wet column. Restoring is recorded as a source rather than internal conservative redistribution.

## D20 — The node-form vertical diffusion: zero-flux boundary and seafloor fill

Node-form vertical diffusion enforces the no-flux bottom boundary at the actual deepest wet node. Shallower columns require the appropriate one-sided stencil; a column wet to the deepest grid level keeps its declared boundary stencil. The node-form and opt-in interface-flux forms remain distinct.

## D21 — Polar cap: wet-point zonal mean, cos^2 taper, 3D mask, capped eta first

Polar caps average over wet cells only, use the three-dimensional mask for three-dimensional fields, and taper with `cos²` away from the pole. North-band weights reverse the south-band ordering. Cap `eta` before evaluating its pressure gradient, then cap updated barotropic velocities. Cast weights to the field precision to avoid implicit float64 promotion. Cap smoothing is a numerical treatment, not a physical source or a climate qualification.

## D22 — Free surface: forward-backward, implicit barotropic Coriolis, implicit drag

Free-surface coupling uses forward-backward updates, implicit barotropic Coriolis and implicit bottom drag. Gradient/divergence pairing and fast-step stability limits both matter. This does not turn the forward-backward time scheme into an exact energy-conserving integrator.

## D23 — Mass-conserving sponge and eta relaxation

Sponge and eta relaxation retain their area-weighted mass treatment. Diagnose eta displacement volume separately from fixed-node tracer stocks. Filtering, relaxation and physical sources must remain distinguishable in the ledger.

## D24 — Surface boundary conditions: bulk heat flux and salinity restoring

Surface forcing combines the declared prescribed/bulk heat and salinity restoring terms. Air temperature is external forcing, distinct from an SST target. Heat deposition weights use the selected surface-layer prescription; changing it changes the physical configuration.

## D25 — Grid-scale de-aliasing: 2/3 FFT in lon, 5-pt binomial in lat

Horizontal de-aliasing combines the declared longitude FFT truncation and latitude binomial filter. Masks and latitude boundary handling remain explicit. Filtering effects must remain visible in stage accounting rather than being silently relabeled as physical exchange.

## D26 — Barotropic density PGF: transport-consistent wet-column form

Barotropic density forcing is the wet-column transport-weighted average of the same gated three-dimensional pressure-gradient force used by momentum: `F_rho = sum(pgf_k * dz_k * wet_k) / H_sw`. Below-bottom constant pressure is excluded; normalization agrees with the barotropic velocity definition.

## D27 — Tracer transport: TVD/MUSCL flux limiter

`fct_adv` selects the existing TVD/MUSCL minmod reconstruction and takes precedence over `monotone_adv`. It is not a full multidimensional Zalesak FCT scheme. A slope requires all stencil points to be wet; latitude uses edge padding and longitude is periodic.

## D28 — Optional surface heat: account for wet volume and ice latent heat

Optional mixed-layer heat deposition is weighted by wet-node overlap, normalized so thickness-weighted heat tendency recovers the prescribed heat flux divided by `rho*cp`. It is deposition, not entrainment. Dynamic ice applies surface heat once in its surface operator, tracks sensible minus latent heat and uses actual ice change for brine/melt salt exchange. The ice model remains a surface phase-change approximation.

## D29 — Spatial tracer diffusivity belongs inside a face flux

Spatial tracer diffusivity belongs inside a wet-face flux: average the background plus coastal coefficient onto faces before taking the spherical conservative divergence. Multiplying a cell Laplacian by a spatially varying coefficient does not preserve the same inventory identity.

## D30 — Runtime and scores must preserve evidence, not manufacture PASS

Strict checkpoints include all six state fields; dynamic-ice restart cannot invent a missing ice field. Salt mass uses salinity/1000, and output schemas declare units. Heat residuals require integrated source data. Shared external scoring validates CF time, masks, coverage and finite errors; a scoring PASS is not a climate-skill or long-stability claim. Old raw/A2 and area-weighted v2 results retain distinct protocol identities.

## D31 — Default scalar diffusion must conserve too; square its flux operator

Temperature/salinity diffusion uses the wet-face flux operator `D1`; scalar biharmonic is `-kappa_bi * D1(D1(C))`. With the declared wet-volume inner product, `D1` is self-adjoint and nonpositive, so its negative square conserves content and dissipates variance. Biharmonic diffusion is not monotone and needs its own explicit stability bound, `kappa_bi*dt_half*abs(lambda_max(D1))² <= 2`.

## D32 — Record actual stages and sources; bookkeeping is not conservation

`audit.stages.make_budget_step` records the actual RK/Strang stages without replacing the core. Endpoint change minus stage changes checks bookkeeping; endpoint change minus declared sources checks a different budget. Internal transport and filtering are not external sources. Diagnostics are fixed-node water/ice enthalpy proxies, nominal salt mass and separate eta displacement volume; they do not prove moving-layer physical conservation.

## D33 — Attribute actual top transport; eta*C is not a conservation repair

Record actual RK/substep means of advection, convection, GM and Redi, including top transport. Retain signed and absolute residuals so cancellation cannot hide activity. Adding `area*eta*C`, deleting a top flux or offsetting a stock does not independently establish physical continuity/inventory closure.

## D34 — Project the native wet-face constraint with its volume adjoint

The projection constraint sums actual open wet-layer face transports. Its gradient is the wet-volume adjoint `G3 = -V^-1 * B^T * A`; the resulting operator uses that same native constraint. No pressure regularization, mean deletion, artificial diagonal or tracer correction is introduced. Algebraic projection accuracy is separate from physical volume/time consistency.

## D35 — Immutable projection settings and native Jacobi; measure convergence

Resolve the projection iteration cap, relative tolerance and preconditioner once at construction. An explicit cap overrides the environment default; record the dtype tolerance floor and cap origin. Jacobi uses the independently checked diagonal of the native operator without altering its physical matrix. Squared norms accumulate by addition; worst individual residuals accumulate by maximum. Convergence needs measured actual residuals, not only the solver return flag.

## D36 — Check actual transport and bound residual correction

After CG, check actual reverse-cumsum top transport. At most `projection_max_refinements` corrections are applied, using the existing RHS absolute stopping floor and bounded trigger. A refinement bound of zero retains the unrefined solve. No residual is filled back into a filtering term to manufacture closure. Bounded refinement is not a universal safe iteration-cap recommendation.

## D37 — Physical volumes and shared barotropic/tracer transport

Physical moving-volume, FV/C-grid and material-stock alternatives remain local research and are not the default production method. Their state and inventory semantics cannot be inferred from fixed-node FD output, and old node checkpoints cannot be relabeled as cell means or face velocities. Their candidate failures do not negate the historical FD production lineage.
