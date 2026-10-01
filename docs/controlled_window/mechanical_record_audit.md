# Existing six-hour records: mechanical budget audit

Object: PR7 head `19f0cd8fb56dd69155cdca65e5dab971eb344c4e`, I0B0 only.
The executed source is historical `212df951c351f82dba74fbc43db5e52b0ad34c47`,
with the registered material observation patch. All35 source hashes and all72
I0B0 committed checkpoint/marker hashes were checked read-only. No step was
advanced for this inventory.

| Actually retained | What can be obtained | What it does not supply |
|---|---|---|
| Full six-field state at t0 and every300s accepted endpoint, plus authority/hash | Recompute fixed-reference kinetic energy, eta quadratic potential energy, moving-top weight diagnostic and endpoint impulses at all73 times | Within-step forces, midpoint velocities, work attribution, density potential-energy conversion |
| `budget.observed_change`, `source_inputs`, exchanges and residuals | Moving-top temperature sensible-heat stock in J and nominal salt stock in kg | Mechanical energy, wind work, density/free-surface pressure work |
| `incoming_inventory`, `attempted_inventory` | The same two tracer stocks | Kinetic/free-surface/buoyancy PE stocks |
| `checks` and active/required subcycle counts | Gates, extrema, continuity, tracer face matching, nominal momentum plan count | Actual diffusion/RK/fast substage velocities or energies |
| `observed.pressure_before_*`, `pressure_pre_fast_*`, mean velocity and eta samples | One node's before/predictor density and eta forces | Domain work or the twelve fast midpoint force histories |
| Common geographic section fluxes and region volume change/nontransport residual | Regional volume budget; exterior north face saved as zero | Pressure-weighted face work, regional mechanical-energy flux, shear exchange |
| Four-snapshot postprocessing | Pressure/Coriolis norms and coarse Eulerian acceleration | Complete discrete momentum or energy closure |

`material_top.py:109-138` defines the moving tracer stock:
`h=h0+eta*surface_mask`, `Q_T=rho0*C_P*sum(A*h*T)` and
`Q_S=rho0*1e-3*sum(A*h*S)`. Its existing J units must not be mistaken for
mechanical energy. `stage_budgets.py` has additional recorder hooks, but this
material run did not call its `_StageRecorder`: corresponding fields are absent
from every saved I0B0 record, not measured zeros.

Actual retained momentum graph (`material_top.py:418-458`) is first bottom drag,
first material L, N predictor, predictor L, fast subcycle, final bottom drag,
wall mask. Tracer replay separately computes middle/end material L states;
`attempted=dynamical._replace(T=end.T,S=end.S)` discards their velocity outputs.
Do not book their momentum a second time. Tracer changes can still change density
pressure and any future compatible buoyancy-PE inventory.

Original operators: joint Heun viscosity (`material_top.py:249-274`);
split linear shear rotation (`jax_solver_global.py:1843-1961`);
two actual momentum RK stages (`2079-2098`);
full tendency and its explicitly subtracted mean/shear terms (`1237-1270`,
`1987-2046`); twelve25s constrained drift-kick-drift updates (`1635-1669`,
`2345-2391`); exact exponential bottom drag (`2414-2422`). Momentum advection
is advective-form then FFT/binomial dealiased (`994-1016`, `1071-1090`), so
closed walls alone do not establish conservative kinetic-energy advection.

Original wind stress is nonzero; horizontal/vertical viscosity and linear bottom
drag are active (`nu_h=2e6m2/s`, `nu_v=1e-4m2/s`, `r_bot=1e-3s-1`). Sponge,
eta relaxation, polar cap and momentum biharmonic are configured off. The flag
`project_adv_vel=true` does not establish an actual projection event: the
symmetric-fast early return bypasses the legacy nonlinear branch using it.
Projection and boundary masks must be tagged by actual call/map, not parameter
names. Face matching adjusts tracer fluxes rather than accepted velocity.

Initial u=v=eta=0 gives Kref=Eeta=0 and instantaneous pressure power0 even
with nonzero pressure acceleration. A finite kick can create kinetic energy;
use the update's average before/after velocity, not old velocity alone, to
measure its work. Missing density PE must remain an unclosed conversion, not
an invented source or proof of spurious energy.

The narrow proposed probe is frozen in `protocol_mechanical_probe.json`: one
independent original dt300 from t0 and one from6h, paired equivalence witnesses,
single CPU4GiB and600s total. No new6h/30h trajectory. Record reliable fast/drag
work and outer-stage increments first; explicitly leave unobserved slow-term
decomposition and incompatible buoyancy PE open. Protocol arithmetic bounds are
diagnostic, never new production or scientific acceptance gates.
