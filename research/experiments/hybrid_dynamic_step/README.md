# 混合全深动态两柱原型（独立实现；非完整原生产步）
基线8ad3552。权威顶部库存/实际质量动量，固定深层FD节点/参考体积，
深层库存和动量在演化时响应。未借用旧pressure-gradient/质量配对。
同物理顶部湿面平均压力、深层共同FD点压力，力为速度到Q的负伴随。
pressure kick的实际质量KE/力功交换单独在舍入界内；不证明总KE+PE闭合。

每步pressure半kick→由实际速度生成共同face Q→horizontal donor与
bottom-up deep continuity垂向flux（含-22.5交换）→top正厚remap→pressure半kick。
深层体积固定但库存演化；eta由全柱Q计算。压力下段以顶部底密度与第一个
deep节点梯形接续，此为新接口合同，不假称保留旧top节点梯形压力。
温盐top面用同P1；momentum采用donor均值，显式报告transport/remap KE代价。
全局温盐与momentum（扣压力impulse与extensive源）逐字段预算，joint-outflow
CFL拒绝；不clip、不改输入dt、不加阻尼。完整接受/深快照拒绝，版本1 restart。

固定研究EOS rho01025/g9.81/alpha2e-4/beta7.6e-4/Tref15/Sref35；1D normal=x。
暂缺2D经纬/海岸mask、非共享deep布局、Coriolis、原ice/mixing/forcing源适配、
完整总能量/时间阶、原生产restart迁移。不能调用本核声称历史353完整步修复。
九项CPU测试含static共同剖面/GCL、负伴随、deep响应、跨带交换、pressure-KE
功、restart、拒绝及合成750m近薄顶dt300，0.20s，ruff全库通过。
qualification=false。下一真实回放必须先由A审接口/动力精度并完成上述适配；
平底局部配对成功不要求全柱坐标重写，但也不证明真实地形可直接继承旧算子。
本分支供父任务审实现，不新增诊断PR；无私有原数组、长跑或外部算力。
