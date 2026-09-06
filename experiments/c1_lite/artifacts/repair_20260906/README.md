# Baseline-relative 修复与小规模验证 — 2026-09-06

本轮实现的是离线证书、接口检查及失败回退契约修复，不是已经接入生产的
连续窗口算法。没有修改四个 demo 的窗口、anchors、容差或网格构造来获得正结果；
没有清洗基线退化面，没有启动渲染，也没有提交或推送 Git。

**主要进展：E2 在完整理想 source 模型下的局部、接口及外部接触审计全部通过；
真实 Forest 得到可用的预表面位移事件登记数据。但运行时全窗口准入仍未完成，
本轮没有新的 SSIM 或生产画质提升结论。**

## 1. 实际修复了什么

- raw retained provider 加入原始 emission 时间区间及其断点，分别记录全部
  owner 的 retained / skipped / suppressed 状态；保留多 owner 到规范有向面的关系。
  这仍是 source 身份商模型，不是最终 C++ 顶点数组身份的证明。
- 增加通用精确接触充分判据：共享角点的正权半平面组合、斜向严格分离，
  以及 XY 支撑边零特征的三维分离。包括 fan 中心，覆盖完整仿射分段而非少量采样。
  不把不同 SourceVID 的坐标重合当成合法共享，也不跳过共享邻面。
- 检查原两面 patch 是有向 disk、完整边界附件、保留侧顶点 link 的有向单路径；
  拒绝 retained 面复用原 patch 内部对角线。几何、身份与拓扑证据分开。
- 明确第一版接口退化策略：保留面在接口处退化即拒绝整个窗口；远处原有缺陷
  可以保持，但不能因此宣称全网格无缺陷。没有删除它们以提高通过率。
- 新准入契约需要所有门均 PASS。缺失、UNKNOWN、REJECT 时整个窗口
  **包括精确 root** 都返回 baseline；旧 C0 root-only 是另一种历史模式。

代码入口：[raw provider](../../runtime_retained.py)、
[contact v3](../../window_contact_v3.py)、
[接口拓扑](../../interface_topology.py)、
[准入契约](../../window_admission.py)、
[集成审计](../../run_window_audit_v3.py)。这些模块不输出生产 replacement plan。

## 2. 固定四事件：证书覆盖改善，非网格质量改善

保留原 11 个仿射区间与 15 个实际断点，共 26 个审计单元，全部扫描完成。
下表分母是“外部候选三角形 × 审计单元”，不是独立事件数。
v2 与 v3 都使用修正后的 raw provider，因此可以直接比较计数。

| 事件 | v2 PASS / UNKNOWN / REJECT | v3 PASS / UNKNOWN / REJECT | v3 理想审计 | 实际窗口准入 |
|---|---:|---:|---|---|
| E0 | 30,007 / 2 / 2 | 30,007 / 2 / 2 | REJECT | FAIL_CLOSED |
| E1 | 21,420 / 0 / 0 | 21,420 / 0 / 0 | PASS | FAIL_CLOSED：端点表示不支持 |
| E2 | 26,678 / 48 / 0 | 26,726 / 0 / 0 | PASS | FAIL_CLOSED：运行时证据未齐 |
| E3 | 26,619 / 79 / 28 | 26,640 / 58 / 28 | REJECT | FAIL_CLOSED |
| 合计 | 104,724 / 129 / 30 | 104,793 / 60 / 30 | — | 0 个获准 |

两版分母均为 104,883；新增判据解决了 69 次未决检查。旧 A 的 109,552
是过滤前 processed-source 候选超集，不能把分母变化当成成功率提高。
v2 / v3 分别约 69.3 / 69.6 秒，均不包含原始 mesh 构建或历史 BEB1 anchor 成本。

E2 的原 48 项未决不是已发现的碰撞：一类需要斜向分离，其余涉及投影接触但
三维高度有严格间隔。判据是通用有限方向/特征搜索；它是在已有实例诊断启发下
补充的充分证书，不是独立测试集、完整 CCD 或普遍覆盖率证明。

保留的重要负例：

- E0 在实际 `tau=11` 有 retained `[A,A,C]` 接口面，不能擅自认为运行时会删掉。
  REJECT 计数也包含分段极限上的共享面退化，不表示新方法造成两次碰撞。
- E3 有四个不同 VID 但几何共线的接口三角形，跨七单元构成 28 次策略拒绝。
  “不同 VID”并不保证非退化；重复索引计数也不等于零面积面总数。
- E1 的理想构造可以通过，但当前要求的 binary32 内部对角点/四非退化 fan
  精确端点表示不受支持。没有用 epsilon、全局 double 或换事件掩盖此限制。

[v2 原始摘要](window_audit_v2/summary.json) ·
[v3 原始摘要](window_audit_v3/summary.json) ·
[raw owner / 退化证据](raw_retained/summary.json)

## 3. 实际运行时核验与尚未证明的部分

完成 11 个不同断点 × raw / extra_smooth × OMP 1 / 8 = **44 次实际基线切片**。
对应 OMP 1/8 mesh hash 全部相等，原始 cache 不变，临时 cache 副本均已删除。
raw 的面数与 source 商模型加回被抑制源面的计数在这些断点吻合。
E0 与 E3 的上述接口退化面在实际输出中都有精确有向坐标匹配。
这些是基线诊断，不是新窗口的运行时测试；坐标匹配、面数相同不能代替身份追踪。

另外对 E2 原先固定的 33 个时刻运行 raw / OMP 1 数值探针：预期边界坐标均
唯一匹配，所查两张 source face 各匹配一次，prospective fan 的 132 张三角形
精确非零面积；在这些时刻，中心/边界量化误差和由其产生的法向量误差为零，
两个端点的 prospective 中心精确落在实际对角线内部。最小三角形面积约
0.07298。这些时刻可精确表示，不意味着任意时间都没有舍入误差。

尚未完成的准入证据包括：同根事件隔离、实际 SourceVID / owner 等价映射、
时间选择器 conditioning、全窗口 binary32 接触及端点实现、未修改外部数组。
raw 与 extra_smooth 的几何/发射过程不同，不能共享未经证明的证书。
实际 C++ 时间输入含浮点转换，理想分段中的身份稳定也不能自动外推过去。
具体反例是 E2 下端完整 retained mesh 的理想零面积面数为 1,372，而实际
raw 为 1,370；局部探针的零误差不能推广为整个外部网格几何相同。

已只读核对 82 项既有输入/脚本/结果哈希绑定，全部相符；旧 A/B 保持不变。

对 E2 的七类共享支撑边候选，额外找到了以共同端点为原点的相对分离平面，
可以保留结构性零值，并对其余顶点的严格符号传播坐标误差界。这说明不一定
需要更换 fan 构造；它只是**给定统一坐标误差界及真实共享端点身份时的条件证据**，
尚未验证实际全窗口满足条件，不提升任何运行时 gate。

[OMP 断点结果](runtime_crosscheck/summary.json) ·
[E2 33 点探针](runtime_rounding_probe/result.json)

## 4. Forest：有真实事件数据，但流水线保留 STOP

复用既有 Forest seed-0 场景，在新路径生成 registry / provenance-enabled cache，
不改原场景或旧缓存，不渲染。固定最早 64 帧、24 fps、960×540、6 px LOD，
有两个时间组；**不是论文的 3 px / 480 帧完整实验**。

首次 24 帧构建碰到本轮自设的单文件 128 MiB 上限；且其零级时间树不足以
测试目标 temporal-neighbour 事件。该失败保留，不能计成零事件普查。
根据已确认的资源原因及时间树结构，预先锁定 64 帧重试：45 分钟、3 GiB
总量、2.5 GiB 提前停止、512 MiB 单文件上限，没有根据收益挑片段。

64 帧运行约 **20 分 3 秒**，峰值观测输出约 **1.783 GB**，结束输出约
**1.149 GB**。两套几何 cache、processed owner sidecars 和原生 registry 已生成。

| cache | raw observations | logical incidences | canonical events | 不同精确根 |
|---|---:|---:|---:|---:|
| OpaqueTerrain | 440 | 268 | 131 | 2 |
| atmosphere | 0 | 0 | 0 | 0 |

然而其后的 Infinigen surface kernel 因禁用属性生成后缺少 `eroded` 而报错。
**整条 driver 保持 STOP / build_complete=false**，没有重跑或追加渲染。
该后处理可能施加 displacement，不只是贴色，因此这里可用的是
**pre-surface-displacement 的 BinocMesher 源几何与事件登记**，不能称为完整
Infinigen 最终网格/渲染已成功。

131 个 canonical events 是登记层的 temporal-neighbour saddle，尚未编译完整
BEB1 closure、局部可嵌入条件或 C1-lite 窗口。BEB1 覆盖率为 NOT_ESTIMATED，
不是 131 个成功修复，也不是 0% 覆盖率；两个精确根不等于只有两个事件。
该结果首次提供这一短片段的真实分母，不代表六场景总体或实际质量收益。

[64 帧原始摘要](forest/stage64/summary.json) ·
[固定重试协议](forest/phase2_64_protocol.json) ·
[首次停止诊断](forest/stage1_diagnosis.json)

## 5. 能用于论文的结论与下一步

本轮没有改变网格构造，所以没有新的 treatment gain。旧 B 中 E2 的局部高度
误差积分相对 ordinary baseline 下降 **2.1125%**、相对 centroid 下降 **0.9261%**
仍只是既有离线有限采样结果；最大误差存在回退。E0 的旧正收益不能掩盖其
这轮接口拒绝，E3 的零收益和 E1 的 unsupported 均保留。没有新 SSIM。

当前可主张的是：受限 source 几何模型下更完整的合法接触认证、明确的
baseline-relative 条件及失败域、以及一个真实短片段的事件登记分母。
不能主张全局修复所有拓扑缺陷、整个浮点实现连续、真实场景全面优于基线，
也不能把 C1-lite 名称解释为已证明 C1 可微或一般 Morse 拓扑事件处理。

下一步应先复用新 Forest cache 做固定顺序的候选 eligibility / closure 筛查，
区分“登记事件”“BEB1 可构造”“窗口获准”三种分母；同时给 E2 补真实身份与
有界舍入的最小闭环。修复后处理适配器应作为独立、显式的新实验版本，
不能回改这次失败记录。完整运行时 gate 未齐前，不启动生产窗口/画质渲染。

## 6. 保留、清理与复现

旧 A/B 与 v2、v3 新证据均保留；各结果绑定输入、脚本、实际 core.so 的哈希。
Windows 和 WSL 两份 checkout 的既有差异未被覆盖。Miniforge 环境、原缓存、
有效旧结果及 WSL 迁移备份均未删除。

仅清理了本轮失败 24 帧试验的可重建缓存及私有 assets，共 **488,586,831 bytes**，
保留文件清单、SHA、日志与诊断；没有回收站副本，需要时可从原输入重新生成。
64 帧有效登记数据所在的约 1.15 GB 输出保留，避免再次花时间建树。
运行时诊断的临时副本自动删除；没有留下大规模 mesh 数组或渲染序列。

[清理记录](forest/stage1_cleanup.json) ·
[复现命令和输入限制](REPRODUCE.md) ·
[修复契约](../../REPAIR_PROTOCOL_20260906.md) ·
[v3 后续证书协议](../../CONTACT_V3_PROTOCOL_20260906.md)

最终验证：**184 项 C1-lite 单测 + 14 项相关单测全部通过**，另一个独立旧
BEB1 闭包组合回归也通过。新增条件性相对平面诊断为 49/49 PASS，仍不授权
运行时窗口。[相对平面说明与复现](e2_relative_plane/README.md)。

最终元数据写入前测得：Windows 仓库约 56.8 MB，本轮小型归档约 2.1 MB；
整个 WSL VHDX + 迁移备份 + Windows 仓库的保守宿主文件长度合计约
**40.27 GB**，小于 400 GB。Linux 内容已在 VHDX 内，不重复计算。
已应用的本轮临时补丁文本也已清除，实际代码与报告保留。
[测试、哈希核验及磁盘快照](validation.json)
