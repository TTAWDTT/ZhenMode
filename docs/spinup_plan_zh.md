# 平衡态加速自旋升技术方案（draft v1，09-09）

**目标**：把模式从当前漂移态推到准平衡气候态，使 AMOC / 深海 T / A1-A2 气候态检验
成为可与观测对比的平衡量。

**判据（先写死，防止中途挪门柱）**：
- 深海（z≤1000 m，3D 湿掩码）T 线性趋势 ≤ 0.005 °C/10yr（当前 +0.45）
- 且该趋势 ≤ 同段脉动 std（当前 last-yr resid 0.06 mC/yr 段内波动）——趋势淹没在变率里
- AMOC yr-10 均值漂移 ≤ 0.05 Sv/yr（当前 ~0.11/yr→1.11±0.15 Sv 稳定段）
- 满足后连续两个 10-yr 段都成立才算准平衡

## 现状基线（moc_tenyr_ms_gm.npz 实测）

| 量 | 当前值 | 速率 |
|---|---|---|
| deepT 漂移 | +0.045 °C/yr 三段恒定 | 无加速 → 扩散控制 |
| 上层 0-2 km T | −0.24 °C/10yr | 冷却收敛中 |
| AMOC | 1.11 ± 0.15 Sv（yr10） | ~0.011 Sv/yr 爬升 |
| 裸跑外推 | 深海扩散调整时标 1e6 m²/1e-5 = **3170 yr** | 不可裸跑 |

## 方案：两相加速自旋升

### Phase A — 加速深层调整（便宜档，先跑）

- `--kappa-v 2e-4`（×20）：深海扩散时标 3170 → **159 yr** 模式年
- 预算：54 min/10yr → **~14 h/百年**；跑 **300 yr ≈ 42 h**（分段 3×~14 h，
  每 100 yr 出 checkpoint + deepT 趋势报告，用 --checkpoint-days 25 保断点续算）
- dt/CFL 安全性：kappa_v=2e-4 时垂直扩散 CFL dt ≤ dz²/(2κ) = 25/(4e-4) ≈ 62500 s ≫ 3600 s，
  不触雷（runner 已有 --kappa-v 钩子，solver 走同一 _d2_dz2 路径，无新代码）
- 预期副作用（诚实记录）：×20 垂直扩散会过混合，层化偏弱、AMOC 偏强——
  **Phase A 的场只当"深层初值猜测"，不当气候态**

### Phase B — 标准参数精化（物理气候态）

- 从 Phase A 末态 restart（--restart-from，checkpoint 含全部 u,v,T,S,eta）
- `--kappa-v` 回落 1e-5，继续积分至判据满足
- 深层已接近平衡 → 残余调整时标大幅缩短；预计再 50-100 yr
- 这一步的平衡态才是 A1/A2、AMOC 对比的正式样本

### 备选（Phase A 太慢才启用）

- Bryan 式深层加速（扭曲时间坐标）：实现成本 ~1-2 天，收益 5-10×。
  仅当 Phase A+B 总预算超 3 天才值得做。

## 判据度量实现

- 复用 `results/_moc_ms.py` 管线：deepT/amoc 逐 10 天快照已存 npz
- 新增 `_spinup_check.py`：读任意 global_*.npz，输出判据四项（趋势/变率/AMOC 爬升/连续段），
  自动打 PASS/NOT-YET——避免每次手算

## 风险与对策

| 风险 | 对策 |
|---|---|
| 地中海 eta 漂移 ~500d 触看门狗（§10.4） | Phase A 用 --eta-relax-days（已有 flag），或 300 yr 内必触发 → 先短探针确认 |
| Phase A 过混合使层化不可逆退化 | Phase B 留 50-100 yr 让层化重建；对比 Phase A 前后 0-2km T 剖面 |
| C 盘仅 36 G | 每 phase 结束只留 checkpoint npz（~60 MB）+ snap 抽样，删中间 3D snap |
| 54 min/10yr 若再爆 CFL | conv_nsub=18、n_subcyc=24 余量已验证；CFL limiter (0fa9abd) 在 |

## 预算汇总

| 段 | 模式年 | wall | 产出 |
|---|---|---|---|
| Phase A | 300 yr | ~42 h（分段过夜×3） | 深层准平衡初值 |
| Phase B | 50-100 yr | ~7-14 h | 正式平衡气候态 |
| 合计 | — | **2-3 天 wall**（机器可挂后台） | 可对比的 AMOC/气候态 |

## 第一步（本周末前）

30-yr Phase A 探针：`--kappa-v 2e-4 --days 10950 --checkpoint-days 25`，
验证 (a) 不爆 (b) deepT 趋势确实压降 (c) Med eta 不触雷 —— 2.7 h 出结论，再放全量。
