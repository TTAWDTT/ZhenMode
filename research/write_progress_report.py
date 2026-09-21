from pathlib import Path
p=Path('research/to_human/2026-09-21_ocean_model_survey.md')
p.write_text(r'''
# Progress report — mature OGCM survey

**Date:** 2026-09-21  
**Phase:** broad literature/architecture survey

## What I did
- Collected official docs and source material for MOM6, MITgcm, NEMO, ROMS,
  POP2/CESM, MPAS-Ocean, FESOM2, ICON-Ocean, HYCOM, and FVCOM.
- Reduced them to a comparative table in `research/literature/comparative_notes.md`.
- Updated `research/findings.md` with transferable lessons.

## Main conclusion
Mature ocean models do not win because of one exotic scheme. They converge on a
stack: conservative transport, disciplined masks, generalized vertical coordinates,
modular closures, and first-class diagnostics.

## Highest-leverage next step for ocean_solver
1. Prototype a full 3D flux-corrected-transport tracer option:
   - low-order monotone fallback,
   - higher-order centered/PPM-like base,
   - 3D FCT limiter on the full fluxes.
2. Add a budget/diagnostic layer for heat, salt, mass, and energy residuals.
3. Then explore z-star or partial-bottom-cell vertical-coordinate flexibility.

## What I recommend against now
- Do not rewrite into unstructured mesh yet.
- Do not chase nonhydrostatic dynamics yet.
- Do not add more closures before the transport/diagnostics layer is robust.

## Files
- `research/literature/comparative_notes.md`
- `research/findings.md`
- `research/research-log.md`
''', encoding='utf-8')
print(p.read_text(encoding='utf-8')[:200])
