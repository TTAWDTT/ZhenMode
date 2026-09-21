
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


## Follow-up in this loop
- Implemented `--fct-adv`, a compact TVD/MUSCL flux limiter for horizontal tracer
  transport.
- Added operator-level conservation and boundedness tests.
- Full test suite: 147 passed.
- The remaining experiment is a 30d/365d comparison against centered and donor-cell
  transport.


## 实验结果
- 30d / 365d 的 centered、monotone、FCT 对照都通过。
- FCT 在 1° 上与 centered 差异很小，`max|eta|` 略低，成本相近。
- 气候态 A2 RMSE 在三种方案里几乎相同，说明当前气候误差不是 transport 主导。
- 结论：`--fct-adv` 保留为非默认选项；下一步转向诊断和垂直坐标/地形结构。


## 新增诊断层
- `src/diagnostics.py` 记录 heat / salt / volume / depth。
- run driver 现在会在每个 snapshot 保存这些诊断。
- 365d centered / FCT 对照显示：体积严格守恒，热含量漂移约 0.7%，盐含量漂移约 0.0007%。
- 这进一步确认当前 1° 气候误差不是 transport 主导。
