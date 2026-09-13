# archive_deepheat/ — 深层漏热攻坚脚本的 append-only 档案

> 状态:历史档案(2026-09-13 归档)。这些脚本是全局 FD 求解器「深层漏热」
> 攻坚(2026-09-10 ~ 09-12)过程中产生的一次性诊断 / 复现 / 实验脚本。
> 保留作证据链,不再作为开发主线。结论见
> `docs/deep-heat-poisoning-root-cause.md` 与相关 commit message。

## 攻克的问题(一句话)

生产物理配置下模型持续丢失约 -180 ZJ/yr 内部热量。分解确认损失全部来自
RK2 第 2 阶段——它用的对流速度是裸前向欧拉动量预测器 `u_pred = u + du1*dt`,
这个速度动量方案自身从未到达,柱散度极强,通量形式对流随之漏热。修复尝试
(冻结速度 / 守恒式垂直扩散)均默认关闭保留在 `jax_solver_global.py` 中。

## 核心线索(哪类脚本对应哪段历史)

| 前缀 | 主题 |
|------|------|
| `_budget_*` `_leak` `_decomp*` `_term*` `_terms_budget` `_resid` | 热量收支分解:定位损失落在 N 步 / RK2 哪一阶段 |
| `_adv_*` `_advcons` `_advonly` `_advfix_time` | 对流算子与第二阶段速度 |
| `_js_*` (8 个) | **solver 实验变体**(A/B gate、topzero、advgate、patched、V1/V2/V3) —— 整文件副本含独有改动,非冗余 |
| `_d2test` `_nuv_test*` `_dv_*` `_vert_test` `_conservative` | 垂直扩散:节点形式不守恒、界面通量形式 |
| `_shear*` `_shearHcol` | H_col 剪切投影(已否决的修复方向) |
| `_div_*` `_zf0*` `_fscheck` `_fsdecide` `_fsd*` | 柱散度、自由面、Fz[0] 与 eta 趋势的一致性 |
| `_conv_*` `_colgate` | 对流调整 / 柱闭合 |
| `_isoneutral` `_redi_np` `_kappa_eff` | 等密度面混合与 GM/Redi 有效系数 |
| `_ohc_*` `_remote_ohc` `_surfbud` `_flux_profile` | 热含量与地表通量 |
| `_diag_era3*` `_diag_slope005*` `_era*` `_relax_dump` | spinC era-3/4/5 归因(哪一 tag 在哪一年失稳) |
| `_equil_cause` `_deep_cause` `_close_attr` `_fix_*` `_fixtest` `_verify_fix` | 平衡态判定与修复验证 |
| `_probe_*` `_poll*` `_local_check*` `_arm_test` `_pin` | 集群 / 本机运行探针 |
| `_js_*.py` 及全部 `_*.py` | 运行方式见下 |

## 运行方式

这些脚本假设 **cwd = 仓库根**,大多数内部用 `sys.path.insert(0, "src")`
(与 `../archive_diag/` 先例一致)。归档后调用路径变为:

```
cd <repo-root>
python src/archive_deepheat/<name>.py
```

脚本本身未改动,`src/` 相对路径仍可解析。多数脚本需要本地或集群上已有的
`results/*.npz` 输入,单跑可能因缺数据而失败——这正是它们属于档案而非
测试的原因。

## 仍在使用的活代码(不在本目录)

- 求解器: `../jax_solver_global.py`(全球 FD)、`../jax_solver.py`(区域谱)
- 驱动: `../run_long_integration_global.py`
- 相关文档: `../../docs/deep-heat-poisoning-root-cause.md`
