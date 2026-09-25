# External industrial comparison — what changes now

Date: 2026-09-26  
Status: active

## Answer in one line

We are not yet comparing like with like. The next step is a fixed external
benchmark slice, starting with MOM6.

## Why

The current annual candidate is strong enough for internal A/B work, but it uses
its own forcing, domain, resolution, spin-up, and WOA-derived reference. That
cannot be used to claim superiority over MOM6/NEMO/FESOM2/ICON-Ocean.

## What I just added

- External benchmark protocol:
  `research/experiments/industrial_comparison_045/protocol.md`
- External target table:
  `research/experiments/industrial_comparison_045/targets.md`

## Next ordered steps

1. Freeze the current mixed-layer/ice 365d check.
2. Define one MOM6-compatible slice:
   same initial/reference field, same forcing, same duration, same metrics.
3. Run ocean_solver on that slice.
4. Run MOM6 on the same slice.
5. Publish a rerun-based comparison; no literature-only winner table.

## Honest positioning

- `ocean_solver` is currently a small, reproducible JAX research solver.
- Industrial models are mature systems with complete physics, sea ice, coupling,
  data infrastructure, and decades of validation.
- The fair near-term claim is not "we beat industrial models"; it is "we are
  competitive on one explicitly defined slice" or "we identify a specific gap."
