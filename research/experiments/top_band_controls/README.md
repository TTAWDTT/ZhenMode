# Top-band conversion controls: comparison, not a selected repair

This experiment implements A's requested two-column / three-control-volume
comparison. It imports only NumPy; production `src/`, the driver, gates and
existing PR3 are unchanged. Branch `codex/ocean-top-band-controls` starts at
`82e1ca4a2158d4c0ed20f91c04e80cdf43f8a36b`.

## Precise physical representation

Unit horizontal area, Boussinesq inertial density 1025 kg/m³, bottom z=-20 m,
top-band bottom z=-10 m, initial faces z=[0,-2.5,-10,-20]. The three slots
store thickness h and extensive N=h·[T,S,u,v]. These are finite-volume means,
**not the original FD point samples**. Heat content is represented by hT
(constant rho·Cp can be restored); salt by hS; momentum by rho·h[u,v].
No original private arrays are used. Full synthetic input is in config.json.

- Merge: before transport, combine top two positive control volumes and all
  four extensive quantities, retaining a zero inactive first slot. The active
  macrocell covers [eta,-10]; bottom and eta are unchanged. Side transport and
  sources are summed before donor reconstruction. The eliminated internal
  interface contributes no external exchange. No splitting back occurs on
  ascent; the restart persists this representation decision.
- Moving interface: band height B=10+eta must be positive. Define
  h=[B/4,3B/4,10], interface z1=eta-B/4. Both eta and the band bottom are physical
  boundaries. Evolve positive layer volumes and content on a Lagrangian
  intermediate grid, then use exact physical overlap lengths with piecewise
  constant donor means to remap all four N fields to this target. This is a
  specified top-band ALE **component**, not a full-column z-star model or a
  production coordinate implementation. It is not remapping into the old
  negative-thickness fixed target.
- Fixed target: h=[2.5+eta,7.5,10]. After crossing -2.5, requesting this target
  fails and returns the original bytes. It is retained only as a negative
  control, never repaired by a floor.

Both conversions are synchronous on this two-column stencil. One-sided
conversion is explicitly unsupported and rejects; its distinct geometry is
used ONLY in the pressure diagnostic. An absent neighbor, invalid band and
outflow >0.5·old volume reject before committing any state. This toy CFL is
declared for this new component, not a changed ocean acceptance threshold.
Water sources, sea ice, vertical mixing, pressure evolution and slow/fast
coupling are outside this component. Nonzero extensive heat/salt/momentum
sources are tested independently; no water is implicitly added by them.

## Same transport and trajectory

One shared face has q=[0.25,0.35,0] m³/s, positive left to right, no exterior
flux. Both schemes consume these SAME layer inputs (merge sums the first two).
Use dt=1 s, six descending and six returning steps, with sign chosen from the
persisted step counter. No prescribed eta overwrites occur: total column
volume determines eta and is driven by the actual shared face. The left column
goes 0→-3.6→0 m, right 0→+3.6→0 m; both cross the old limit while remaining
valid in their own representation. Unequal top flux fractions ensure that the
moving target actually requires remapping; differing column profiles exercise
donor mixing rather than an accidentally identity cycle.

Sources are transported as extensive rates. Scalar donor accounting and a
fixture-specific analytic overlap oracle independently check the update.
Uniform T/S/velocity retains its means (discrete GCL). Both initial profiles
have stable density increasing downward and velocity shear; no hydrostatic
evolution is claimed. Restart NPZ contains h, N, representation and accepted
step counter, allow_pickle=False. Before conversion, after conversion, descent
and return checkpoints reproduce the decisions and final bytes in one process.
This is not the original solver restart contract or a cross-process test.

## Retained quality losses and failed pressure qualification

The total conversion kinetic-energy loss for merge is **5354.82421875 J**,
matching the independently evaluated sum over columns of
`m0*m1/(2*(m0+m1))*|u0-u1|²`, including both velocity components.
Temperature variance drops 1251.875→1158.125 at conversion; salt and momentum
content totals remain conserved. Total inventory alone hides this irreversible
mixing. Moving conversion at the initial eta=0 has zero loss, since its initial
grid is the same; later transport/remap is dissipative.

After the common cycle, additional kinetic losses are about68.89074 J (merge)
and2730.86974 J (moving). Relative to each converted initial state, maximum
content changes are about2.64706 and3.82255 (mixed field units); final active
means are preserved in report.json. These are reported errors/losses, not
required byte reversibility of a lossy cycle. Geometry returns; profiles do not.
The cycle losses combine donor advection and remapping and do not isolate a
causal remap contribution. Do not rank physical quality from this one fixture.

Pressure counterexample uses the SAME analytic stationary density profile
T(z)=20+0.5z, S=35, zero velocity, eta=-2 in both columns. Each geometry gets
fresh exact physical cell averages, removing prior remap error. Integrate a
piecewise constant density anomaly to the COMMON physical depth -4 m, plus
rho0·g·eta; horizontal separation is1000 m. The analytic acceleration is zero,
but fixed-versus-merge gives -1.4715e-6 m/s² and fixed-versus-moving gives
+4.4145e-6 m/s². Identical target grids give zero force. These are explicitly
**diagnostic P0 reconstruction forces, not measured original FD forces**.
The heterogeneous topology is not supported by the transport component.
This counterexample prevents promoting a simple cell-mean pressure formula
to a coupled scheme; it does not prove every ALE or merge method impossible.
Both pressure qualifications and overall ocean qualifications remain false.

## Validation, costs and next decision

20 bounded CPU tests pass: four inventories, analytic kinetic loss/mixing,
uniform means, sources, stable initial stratification/shear, pressure negative
control, eight restart placements, one-sided/missing-neighbor/exhaustion/outflow
rejections and byte rollback. Linux Python3.12.14/NumPy2.5.3: pytest0.19 s;
measured wrapper0.417 s, child peakRSS38596 KiB. Audit0.00811 s, peakRSS25508 KiB.
No GPU, external science download or paid API; billing cost is unavailable.
The timing wrapper initially lacked /usr/bin/time; the retained final costs
use Python resource/subprocess instead. Ruff and whitespace checks pass.
JUnit, test log/cost, config/source SHA256 and report are in evidence/.

```sh
OPENBLAS_NUM_THREADS=1 timeout 60 python research/experiments/top_band_controls/audit.py --output /tmp/new_controls
timeout 60 python -m pytest tests/test_top_band_controls.py -q
```

The audit succeeds when its stated experiment completes and always reports
qualification_passed=false. Passing tests includes confirming pressure failure.
Stop if inventory, positive volume, bounds or rollback controls fail. Do not
connect either component to original FD states. Next decision waits for the
existing real initial-state/pressure-background audit; coupled pressure,
momentum geometry, source metrics, initialization and restart need a shared
physical reconstruction and parent coordination before any actual replay.
No real 1° repair, time order, gradients, long integration or industrial speed
qualification follows from this component.

Official design references, not validation of this implementation:
[MITgcm nonlinear free surface and vanishing surface layer](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html),
[MOM6 ALE and conservative remapping](https://mom6.readthedocs.io/en/main/api/generated/pages/ALE.html),
[MOM6 layer/barotropic transport coupling](https://mom6.readthedocs.io/en/main/api/generated/pages/Barotropic_Baroclinic_Coupling.html).
