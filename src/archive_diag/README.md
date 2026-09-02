# archive_diag/ — 诊断攻坚脚本的 append-only 档案

> 状态:历史档案(2026-08-29 归档)。这些脚本是全局 FD 求解器稳定性攻坚
> (见 `docs/long_run_climatology_report_zh.md` 的逐条记录)过程中产生的
> 单次诊断/复现脚本,`_` 前缀为临时草稿,`diag_*` 为早期历史诊断。
> 保留作证据链,不再作为开发主线。

## 核心线索(哪类脚本对应哪段历史)

| 前缀 | 主题 |
|------|------|
| `_diag_pgf*` `_diag_frho*` `_frho_*` | 斜压 PGF 注入链(step-45 eta blowup 根因)|
| `_diag_polcap*` | 极冠滤波(2D→3D wet_mask 修复)|
| `_conv_*` `_diag_conv*` | 对流调整反馈(ghost 底层)|
| `_s_*` `_adv_s*` | ITCZ 盐度失控 |
| `_diag_eta*` `_diag_mass*` | eta 增长/质量源定位 |
| `_diag_spectral*` | 区域谱求解器对照实验(排除离散化 vs 物理)|
| `_diag_circularity*` | A1/A2 循环论证暴露(no-restore 对照)|

每个文件头有 docstring 说明单次实验的假设与结论;配套证据见
`docs/long_run_climatology_report_zh.md` 与 git commit message。

## 仍在使用的活代码(不在本目录)

- 求解器: `../jax_solver.py`(区域谱)、`../jax_solver_global.py`(全球 FD)
- 驱动: `../run_long_integration.py`、`../run_long_integration_global.py`
- 气候态 bench: `../bench_climatology_compare.py`(区域)、`../bench_climatology_global.py`(全球)
