# dashboard — 集群运行状态面板（机器私有）

`npm start` → http://localhost:8399。单页 UI 列出 `runs.json` 中每个 run 的最新
进度、存活状态与判据结论，并画 deepT/AMOC 曲线、tail 实时日志。

## 数据来源

- `runs.json` —— **唯一的 run 注册表**。`scripts/status_board.py` 读的是同一份，
  不存在第二套注册表。改 run 清单只改这里。
- `public/index.html` —— 前端单页（Chart.js），无构建步骤。

## 依赖与限制（换机必读）

- **不是可移植工具**，只在能访问计算集群网关的机器上工作：
  - `server.js` 为每个节点 spawn 一个常驻 `cluster.py` 桥（网关 PTY）。
  - `cluster.py` 需要 `dashboard/secret.json`（已 gitignore，内容为网关连接信息）
    与同一 Python 环境里的 `paramiko`；两者缺一即启动失败并给出明确报错。
- 没有网关时，用 `python scripts/status_board.py --offline` 生成
  `status_zh.md`（同样的 `runs.json`，纯文本、无需集群）。
- 面板只读：所有状态都从集群上已有的日志/npz 里取，不会改动任何 run。
