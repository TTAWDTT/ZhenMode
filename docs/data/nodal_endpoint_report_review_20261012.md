# 第二轮报告两遍独立读回

当前报告数字与范围核对通过，科学/软件总体验收仍 OPEN。第二遍绑定报告 SHA256：d2dca38fd7d6e888e62f73e3218ee8448702f7349868c4b1a106395ae5d11dbc。

第一遍发现主比较表及若干正文使用普通舍入、非负显示有向下值；已通知作者。作者用 Decimal ROUND_CEILING 修正后，第二遍核查 38 个九位显示与 raw 值一致，32 行 JSON/CSV 与各原 native journal 的状态、源码、数值及 output hash 一致。没有重算之前的22，新增10已单独复算；当前32审查完成，早期source review字段future0只是较早空间/源码快照，不应被当成当前总运行计数。

六组排名直接使用原完整分数并与冻结 MOM/OA 原 journal 对齐：主网格 main/half 仍由 MOM6 领先，水平粗网格与垂向细网格四组由新Zhen较小。v3海面误差更小，但u/v比v2更大，报告准确披露；cubic 未选且其门槛通过不代表全面改善。thermal 独立性只覆盖初态/定义/几何/时钟/库存，完整 trajectory oracle仍共享，不能把热风全轨迹写成另一独立物理认证。

409隔离数学与新二维空间 PASS 的范围保持明确：只pure spatial scalar/continuity、未调用 pressure、不代表完整时间/海洋解。新1D单次原规格的 L1/加密/预算/sign 控制符合原门槛，但21/41节点strict constant RHS门槛失败，故 OPEN；最终constant漂移0不替代RHS门槛，量纲没有事后重设。非恒定端点漂移及显式水平补偿的库存变化已披露。post-result arithmetic定位属于development，不替代新1D结果，也不指导冻结409修改。

CI段诚实保留409pending、default push skip不计通过、旧afb/051结果不能替代409源码验收；既有1DOPEN使总体验收即使日后CI成功也不能自动写成所有声明门槛均过。本次只与已存pending快照及作者提供的CI日志结论对齐，未宣称验证409最终CI成功。

可读性/措辞建议（不影响当前数字核对）：将长的2D与1D段拆成两段；“只认证均匀全湿线性EOS的内部压力积分”宜改为“支持范围限于均匀全湿线性EOS，独立多项式一致性检查通过”，免得把有理权重证明/有界测试扩写成全部压力输入的物理认证。发布数据中可直接链接1D声明、OPEN结果和算术定位，后者注明development。

绑定记录：report-second-pass-alignment.json、report-second-pass-reviewed.md、prose_numeric_check.py。本记录未改报告/src、未重复native积分或已完成的原生复算。
