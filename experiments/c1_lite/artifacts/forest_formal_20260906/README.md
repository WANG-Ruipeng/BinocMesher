# Forest：真实有限输出序列准入实验已完成

最终结果：**31 / 131 = 23.6641%**，100 个固定政策回退，**UNKNOWN = 0**。
OMP 1 与 OMP 8 使用相同代码和输入，事件决策、失败见证、普通网格和获准输出证据一致。

这是当前预先声明的保守政策下，**逐事件、有限 requested schedule 的认证覆盖率**。
不是全实数时间定理、同根事件联合准入率、六场景总体覆盖率或视觉质量提升率。

## 输入与口径

- 复用 Forest 64 帧、24 FPS、960×540、`pixels_per_cube=6` 的既有缓存。
  保留原相机时间相位；不是论文六场景各 480 帧、3 px 的正式设置。
- 缓存为已生成的 pre-displacement opaque 几何，包含 ground、landtiles、
  sdf_trees、warped_rocks、voronoi_rocks 五个元素；atmosphere 不是第六个 opaque 元素。
  本轮没有重新生成场景、完成后续 Infinigen displacement 或运行长视频渲染。
- 固定 131 个 canonical registry events，两个离散根为 3/2 和 5/2。
  所有事件 ID、原缓存、相机文件及实际 native delta-T 均有绑定与始末核验。
- 每个事件独立对照同一 ordinary baseline。获准要求 8 个命中自然帧、
  2 个窗口外邻帧和 1 个独立 exact-root 查询全部通过，然后提交紧凑 schedule manifest。

## 真实漏斗

以下是按预定检查顺序的首次决定性拒绝，不能相加解释为所有缺陷的发生次数。

| 检查后剩余事件 | 数量 | 此阶段拒绝 |
| --- | ---: | --- |
| 固定 canonical 总体 / 局部 closure kernel | 131 | 0 |
| 既定 exhaustive source selector | 130 | 1：无候选 |
| 完整 source-quotient 接口 | 104 | 26：retained link 不构成所需路径 |
| 实际严格凸 XY 图面 | 94 | 10：不满足此政策前提 |
| 实际 regular retained 接口 | 85 | 9：接口已有退化面 |
| 两个 exact root 的五元素接触检查 | 35 | 50：精确禁止接触见证 |
| 全部自然帧及邻帧 | **31** | 4：自然帧上的精确禁止接触见证 |

因此总拒绝为 1 + 26 + 10 + 9 + 54 = 100。
104 个进入实际 native 阶段的事件中，条件提交率为 **31/104 = 29.8077%**；
论文主分母仍应保留 131，不能只展示这个条件率。

54 个接触拒绝中，52 个跨元素、2 个同元素。该见证只证明不满足当前共享特征接触政策，
**没有证明这些接触由 replacement 新增**。至少一个独立诊断见证位于原 baseline 的边界顶点。
严格凸 XY 和无退化接口也是当前证书的保守适用域，不是一般修复可行性的必要条件。
不能为了提高比例直接取消这些检查；完整口径见 [政策范围](../../FOREST_ADMISSION_POLICY_SCOPE_20260906.md)。

## 自然帧与结构结果

- 全部候选：131 个事件都有 8 个时间窗口命中，共 1048 个候选 event-frame 记录。
- 最终获准事件：**31 × 8 = 248** 个独立事件自然帧输出确实改变；不包含诊断根。
- 按不同帧号统计为 **16/64 = 25%**：21–28、37–44，帧号从 1 开始。
  这是独立事件实验中发生修改的帧集合，不是一次合并应用 31 个事件后的全局输出。
- 31 个获准事件共 341 个请求记录：279 个 active 查询实际构造并校验数组，
  62 个窗口外查询完整保留对应 baseline。原顶点和标签、所有 retained face rows、其他元素不变。
- 另有 12 个曾通过的 active 候选属于后来整份 schedule 回退的事件；它们被丢弃，未部分提交。
- 根处通过的 35 个事件中，4 个在自然帧失败，说明不能用 root-only 通过率替代 schedule 准入率。
- 本轮没有对 Forest 获准事件测可见像素收益。时间命中、数组改变和图像质量必须分开报告。

## 成本与线程对照

| 完整 native 认证实验 | OMP 1 | OMP 8 |
| --- | ---: | ---: |
| 含清理 wall time | 433.621 s | 474.913 s |
| 主进程 CPU time | 338.919 s | 412.180 s |
| 主进程峰值 RSS | 1,066,860,544 B | 1,054,896,128 B |
| 准入 / 拒绝 / 未知 | 31 / 100 / 0 | 31 / 100 / 0 |

每轮有 22 个唯一 native 查询、66 次 slicing 调用：每个查询分别运行冻结版 ordinary、
新观察器库 ordinary、观察器开启版本，验证完整五元素 baseline 一致。
这里包含副本、读取、完整几何认证、数组构造/哈希及清理，不是纯 slicing 性能。
CPU 列仅为主进程，未合并串行冻结版子进程 CPU；wall time 包含这些子进程。
每种线程数仅一轮，本次 OMP 8 未加速，不作普遍性能结论。

另计 source 编译/普查 77.257 s；独立完整 source-link 见证提取 25.450 s。
OMP 1 的 430 次事件–查询检查，按排序下取整样本分位数得到 P50/P95/P99
约 0.467 / 0.692 / 0.885 s；其中含快拒绝、窗口外检查及共享索引准备，不能当生产每帧时延。

## 最终证据入口

- [OMP 1 summary](native04/summary.json) 与 [逐事件索引](native04/events_index.json)。
- [OMP 8 summary](native05_omp8/summary.json) 与 [逐事件索引](native05_omp8/events_index.json)。
- [OMP 1 独立审计 v2](native04_independent_audit_v2.json) 与
  [OMP 8 独立审计](native05_omp8_independent_audit.json)：
  独立重建 26 个完整 source-link 图，重算 54 个接触、10 个非凸、9 个退化的具体见证；
  仅 1 个无候选事件继承冻结 compiler 的完整 exhaustive 证据。
- [OMP 1/8 对比](omp1_omp8_comparison.json)：131 个事件语义、22 个查询的五元素 baseline
  hashes，以及 31 个获准事件的 341 个请求记录一致；不是只比较平均数或事件计数。
- [完整 source-link 原始见证](source_link_witness01/summary.json)。
- [观察器构建 manifest](observer_sourcevid_build/manifest.json) 与
  [源码差异](observer_sourcevid_build/source.patch)。
- native03、native04、native05_omp8 中各自的 `executed_source_snapshot/manifest.json`
  保存已执行 Python 源码的精确快照与摘要绑定。

独立审计重算小型数学见证并核验保存的绑定收据；不是再次加载整份 native 网格。
实际整网格构造、外部数组不变性和缓存清理是在原实验进程中核验、再由收据绑定的。

## 历史结果更正及保留

旧 native02 的 0/131 **已撤销，不能引用为准入率**：bucket sort 原地减去字段最小值，
旧观察器导出了归一化 merger keys，而不是原 SourceVID，导致 103 个虚假的身份查找拒绝。
见 [native02 更正](native02/scope_correction.json)；旧 JSON 未被覆盖。

修正仅发生在新的隔离 observer-only build：恢复各元素/各查询的原有 ID 偏移，增加明确 v2 编码 ABI。
没有修改 merger keys、网格几何或已冻结生产代码；逐查询确认普通输出与冻结版完全一致。
旧 E2 原生库及原安装库的哈希保持不变。

native03 使用修正身份但仍只使用旧的充分分离判据，得到 24 准入、46 拒绝、61 UNKNOWN，
所以当时不报告政策准入率点估计。随后同一政策下补全有界精确三角形接触判定，
这 61 个待定项最终分为 7 个完整准入和 54 个明确拒绝；不是改变阈值来获取更好的数字。

## 存储、已有图像实验与下一步

大型私有缓存副本均已自动删除；清掉本轮 19 个一次性补丁，保留复现实验脚本和小型证据。
收尾时仓库约 156 MB，本 Forest 证据目录约 92 MB；仓库 + WSL VHDX + 迁移备份合计约
**40.37 GB**，未重复加算 VHDX 内的文件，远低于 400 GB。环境和历史原始缓存没有删除。

此前 [E2 schedule/图像实验](../schedule_validation_20260906/e2_retry2/README.md) 仍有效：
原相机的 active patch 不可见；固定诊断视角下 depth MAE 相对 raw 降约 4.26%、相对 centroid
降约 1.80%，但平均法线误差略变差，silhouette/boundary 不变。不能据此写全面视觉优越性。

下一步优先从这 31 个真正准入的事件中挑原相机可见样本，做很小的自然帧图像对比。
若研究提高覆盖率，应先区分现有跨元素接触与新增接触，再决定是否设计 baseline-relative
接触证书；本轮没有擅自放宽该政策，也没有扩跑长视频。
