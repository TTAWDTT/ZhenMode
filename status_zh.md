# ocean-solver 运行看板（status_zh.md）

> 由 `scripts/status_board.py` 从 `dashboard/runs.json` 生成。
> 人工只改 `dashboard/runs.json`；本文件每次刷新整体重写。

**生成时间**: 2026-09-20 13:23 (本地) · 刷新命令: `python scripts/status_board.py`

---

## 当前活动运行

（无活动运行 / 未查询集群）

## 注册表中的全部运行

- **spinAA_locc**（014/0）day ?/18250 — 无VERDICT — 【局部对流主实验】--localize-conv — +171 ZJ/yr, 比整列 spinS +236 降 27%
- **spinX_kv0**（014/5）day ?/18250 — 无VERDICT — κ_v=0 但 conv 保留 0.05 — +225 ZJ/yr
- **spinZ_alloff**（014/7）day ?/18250 — 无VERDICT — 【对照下限】全部混合关闭 (kv0 gm0 redi0 conv0) — κ 无关的深层增暖地板: +122 ZJ/yr
- **spinS_pj_k1000**（014/2）day ?/18250 — 无VERDICT — 【当前生产候选】kv1e-5 gm1000 redi0 整列对流 proj — +235 ZJ/yr
- **spinT_pj_k2000**（014/3）day ?/18250 — 无VERDICT — GM+Redi 双计 (κ_eff=2000) — +383 ZJ/yr (Defect 5 复现)
- **spinK_proj50**（014/1）day ?/18250 — 无VERDICT — GM+Redi 双计, 从 spinK 恢复 — +383 ZJ/yr
- **spinR_pj_k0**（014/1）day ?/18250 — 无VERDICT — κ 全场归零 (gm0 redi0) 但保留 kv=1e-5 — +130 ZJ/yr; 与 alloff 差 8 ZJ/yr = kv 的贡献
- **spinW_conv0**（014/4）day ?/18250 — 无VERDICT — conv=0 (整列对流关闭) — +183 ZJ/yr
- **spinY_nomix**（014/6）day ?/18250 — 无VERDICT — κ_v=0 + conv0 (整列) — +166 ZJ/yr
- **spinAB_locc_kvh**（014/1）day ?/18250 — 无VERDICT — 局部对流 + kv=5e-6 — +165 ZJ/yr (半垂直扩散)
- **spinAC_locc_twin**（014/2）day ?/18250 — 无VERDICT — 局部对流 + kv=0 — +158 ZJ/yr (与 spinY 同配置但 LOC conv, 对比 +168)
- **spinAD_locc_k2000**（014/3）day ?/18250 — 无VERDICT — 局部对流 + Redi=1000 (GM+Redi 双计) — +265 ZJ/yr, 确认 Defect 5 副作用仍在

---

