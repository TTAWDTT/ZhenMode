# 修复 GM 闭合垂直 CFL 爆裂：bolus 平流形式 → skew-flux 形式

## 根因（已确认）

GM 闭合当前以 **bolus 平流形式** `-(u*·∇T + w*·∂T/∂z)` 实现（`_gm_tracer_transport`，jax_solver_global.py:703）。其垂直项 `w*·∂T/∂z` 有 **平流 CFL** `= |w*|·dt/dz`。

实测初场（kappa_gm=1000, dt=60s, 真实 WOA 层结）：
- k=0（表层 dz=5m）：w*=0.29 m/s → CFL = **3.49**（>1，严重不稳定）
- 前 6 层全部 >0.5，前 4 层全部 >1.0

症状：max|T| 在第 12-13 步指数爆裂（29→42→58→78→103→…），u/eta 保持有界。只有 T 爆——GM 是纯示踪项，符合预期。baseline（GM OFF）20 步完全平稳（max|T|=29.647 恒定）。

**这是数值问题，不是物理问题**：平流形式在非均匀垂直网格（表层 dz=5m）上对 w* 有严格 CFL 约束，w*=0.29 m/s 在 5m 层需要 dt < 17s。而 dt=60s 是外重力波 CFL 要求的下限（不能再降）。

## 修复方案：GM 用 skew-flux 残差形式（Griffies 1998 标准）

GM bolus + Redi 在数学上等价于一个 **skew-flux 张量**。将 GM 从平流形式改为 skew-flux 形式：
```
F^h = -κ_GM · S · ∂C/∂z            （水平斜压通量）
F^z = -κ_GM · (S·∇_h C + |S|²·∂C/∂z)  （垂直斜压通量）
tendency = -∇·F
```
垂直项是**扩散性**的，CFL = `κ_GM·|S|²·dt/dz²`。实测：kappa_gm=1000, |S|max=0.01（slope limiter），dz=5m → CFL = **0.48**（<0.5，稳定）。

这等价于把 `_redi_skew_flux_tendency` 的 κ 参数换成 κ_GM——结构上完全相同的代码，只是扩散性 CFL 而非平流性 CFL。这是所有主流 OGCM（MOM6/MITgcm/NEMO）的实际做法。

### 具体改动（只动 1 个文件：`src/jax_solver_global.py`）

**改动 1**：`_redi_skew_flux_tendency`（:735）加 `kappa` 参数（默认从 p.kappa_redi 读，但可显式传 κ_GM）：
```python
def _redi_skew_flux_tendency(tracer, S_x, S_y, p, kappa=None):
    if kappa is None:
        kappa = p.kappa_redi
    if kappa <= 0.0:
        return jnp.zeros_like(tracer)
    k = kappa
    ...  # 其余不变
```

**改动 2**：`_compute_tracer_tendency`（:804）的 GM 分支改为 skew-flux 形式，删掉对 `_gm_tracer_transport` 的调用：
```python
if p.kappa_gm > 0.0:
    S_x, S_y = _isopycnal_slope(state, p)
    gm_T = _redi_skew_flux_tendency(state.T, S_x, S_y, p, kappa=p.kappa_gm)
    gm_S = _redi_skew_flux_tendency(state.S, S_x, S_y, p, kappa=p.kappa_gm)
else:
    gm_T = 0.0; gm_S = 0.0
```
（Redi 分支 :816 不变——它用自己的 κ_redi 调同一函数，若 κ_redi>0 则额外加 Redi 耗散。但注意 Redi 分支会重算 slope，GM 分支已算了——可复用以省 3 个导数，但为最小改动先各算各的。）

**改动 3**：更新 :801-803 注释（"advection, so it survives residual" → "skew-flux residual, survives residual；扩散性 CFL"）。

**改动 4**：`_gm_tracer_transport`（:703）和 `_gm_bolus_velocity`（:676）**保留不删**——单元测试 `test_gm_closure.py` 直接测它们。但 bolus 平流形式不再接入 `_compute_tracer_tendency`，只是留作诊断工具。`_gm_tracer_transport` 在积分中不再被调用。

### 单元测试更新

`test_gm_closure.py` 的 10 个测试中：
- 8 个测 `_isopycnal_slope`/`_gm_bolus_velocity`/`_gm_tracer_transport`/`_redi_skew_flux_tendency` 本身——**不改动**（这些函数签名不变）
- `test_tendency_finite_with_gm`（:223）测 `_compute_tracer_tendency` 有限——仍应 PASS（skew-flux 也有限）
- `test_redi_is_dissipative_on_perturbation`（:271）——不改动

**新增 1 个测试**：`test_gm_skew_flux_stable`——验证 GM skew-flux 形式的垂直 CFL 在真实网格上 <0.5（回归保护，防止未来改回平流形式）。

## 验证流程

1. 改完后跑 `python tests/test_gm_closure.py`（10+1 测试全绿）
2. 跑全测试 `python -m pytest tests/`（65+1 测试全绿，无破坏）
3. 跑 20 步 stepping 诊断（max|T| 不再爆裂）
4. 跑 1 天 smoke test `--kappa-gm 1000`（不再 NaN）
5. 跑粗网格 gate sweep `kappa_gm ∈ {500, 1000, 2000, 5000}`（plan 2.1）
6. 进 10d → 90d → 365d gate（plan 2.2）

## 红线遵守

- 只动 `src/jax_solver_global.py`（worktree）和 `tests/test_gm_closure.py`（worktree）
- 不碰区域谱求解器（main 上的 jax_solver.py）
- 不碰 Python314/site-packages
- 不 push origin
- 判据预注册不变（365d, max|u|<10, max|T|<init+12, max|eta|<15, 末90天 KE 趋平）
- 不为凑 PASS 调病态值
