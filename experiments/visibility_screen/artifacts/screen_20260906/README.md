# Visibility-first 实验结果：部分完成，尚无可见改进证据

本轮执行附件的第一至第四优先级，没有扩展 Method、放宽 contact policy、
改换种子/时间段/相机，也没有运行长视频或高质量 RGB。

**三个新预注册片段中，Mountain 完成全部筛查与 OMP 1/8 复验；Forest B、
Cave 因预注册资源上限中止。已完成片段没有合格可见组件，因此第二、第三
优先级的真实实验未启动。不能称为“四项实验全部完成”。**

## 1. 统一跨场景结果

所有行均为 24 FPS、960×540、6 px LOD 的 64 帧片段；只评估 opaque geometry，
处于 pre-surface-displacement 阶段。Forest A 是保留的旧负结果，不计入本轮
三个新片段的完成数或样本选择。

| 片段 | 绝对帧 | 状态 | Canonical / roots | Components | 联合准入 components / events | Source admission | 修改自然帧 | 可见候选 / 可见准入事件 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forest A（原结果） | 1–64 | 已完成，负结果 | 131 / 2 | 79 | 20 / 25 | 19.08% | 16 / 64 | 0 / 0 |
| Forest B | 97–160 | 构建资源上限 STOP | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| Cave | 1–64 | 构建时间上限 STOP | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| Mountain | 1–64 | 完成，OMP 1/8 一致 | 46 / 2 | 35 | 30 / 33 | 71.74% | 16 / 64 | 0 / 0 |

可见准入率的分母为零，记为 `null (0/0)`，不是 0%。未完成的 Forest B/Cave
不能记成零事件、零准入或零可见。不能把本表解释成“三个新片段都没有可见事件”。
不同场景的准入率差别也不是画质优劣或方法提升的证据。

机器可核验汇总：[coverage.json](coverage03_terminal_partial/coverage.json)，
可读表：[coverage.md](coverage03_terminal_partial/coverage.md)。
汇总器的 `PASS_HASH_BOUND_COVERAGE_LEDGER` 只表示证据绑定和统计通过，
不表示所有场景或全部研究目标通过。

## 2. Mountain 的实际结果

- 官方 Snowy Mountain seed 1；原始相机，帧 1–64；一个真实 opaque element：`landtiles`。
- 155 raw observations → 92 logical incidences → 46 canonical events，根为 `3/2`、`5/2`。
- 45 个 source-ready，1 个源接口门拒绝；source-ready 不等于 runtime admission。
- 完整有限 schedule 下，42 个事件独立通过，4 个直接拒绝。
- 35 个 conservative components：30 个单事件、3 个双事件、2 个五事件组件。
- 最终 30 个组件、33 个事件联合准入；13 个事件随 5 个组件回退。
- 实际输出全部 64 个自然查询和 2 个独立 exact-root diagnostics。
- 修改自然帧为 21–28、37–44；两窗口每帧分别提交 15、18 个事件替换，合计 264 次。
- OMP 1/8 的全部 66 个查询，网格数组、计划、组件成员和完整 root 支持回放一致。
- 原始 35 个缓存文件、851,327,836 bytes 全部保持不变；两次筛查的 private cache 均已删除。

主要证据：[完整结果](mountain/screen03_omp8/combined_summary.json)、
[OMP 复验](mountain/screen03_omp8/omp_confirmation.json)、
[构建/源阶段独立核查](mountain/build_source_independent_audit01.json)、
[认证漏斗独立核查](mountain/certification_funnel_audit01.json)。

### 回退原因不能混报

| 分类 | 事件数 | 含义 |
| --- | ---: | --- |
| 源接口/link 必要条件不通过 | 1 | 不能认证该源接口，不等于已发现新增网格碰撞 |
| 实际 source boundary 不满足严格凸 XY graph | 2 | 超出冻结构造的准入域 |
| 既有 retained 退化三角形接触接口 | 1 | baseline-relative 认证不能通过，不是新 fan 造成的碰撞 |
| 同组件拒绝成员连带回退 | 4 | 独立能通过，但组件必须 all-or-none |
| 共享边界联合构造不受支持 | 5 | 同一五事件组件均独立通过，但不能联合提交 |

最后两项是独立通过的 42 个事件减少至联合通过 33 个事件的原因，不能把全部
9 个都叫作“拒绝成员传播”。没有为提高数字而增加 shared-boundary solver。

### 可见性：真实修改没有投影成图像修改

16 个活跃自然帧覆盖 368 个 candidate-event/frame 查询，含全部 46 个事件。
可见 source-support events、可见 admitted events、可见 replacement pixels 均为零，
没有 unknown visibility。

这 16 帧的 depth、world-space flat normal、mask、element-ID、face-ID、component-ID
六类 buffer 的 baseline/combined SHA 均完全一致；depth、normal、silhouette
changed pixels 均为零。每帧有 374,466–411,795 个前景像素，并非空图。

368 个候选支持查询中，366 个没有 isolated pixel coverage，2 个在 isolated
投影后被遮挡或未赢得 depth tie。前者**没有被进一步全部判定为 out-of-frustum**，
不把亚像素/投影退化等可能性擅自排除。

逐帧 JSON 与小型 PNG 保存在 [OMP 1 输出目录](mountain/screen02_omp1/)。
这是原相机 pre-displacement 可见性诊断，不是 post-displacement 五方法质量实验。
没有把“相对 baseline 的差异”当作参考误差或改善，也没有用 diagnostic camera 替代原相机。

## 3. 第二、第三优先级为何未运行

冻结选择规则要求：组件已联合准入，至少两个自然帧各有 baseline component-support
union ≥16 pixels，且实际 replacement ≥1 pixel；按场景/首个合格帧/component ID
排序，不按质量 delta 排序。

本轮没有合格组件，`pilot_selection_ready=false`。因此：

- **第二优先级：真实 post-displacement pilot 未执行。** 已准备纯数组审核模块及
  测试，但它们不构成真实 kernel/attribute/geometry 认证。
- **第三优先级：五方法小视觉实验未执行。** 没有生成或声称 raw、extra_smooth、
  schedule-only、centroid、BEB1 的自然可见画质提升；独立质量参考也未冒充已有。

Mountain 有一项实际配置发现：`degrade_sdf_to_displacement=0`，
`mountain_collection`/`snow` 均以 `SDFPerturb` 进入 occupancy。冻结配置的后续
Displacement/BlenderDisplacement pass 列表为空，不能再次对网格施加这些扰动。
参见 [配置适用性核查](mountain/displacement_applicability01.json)。
这只是绑定配置的只读结论，不是 Gate B PASS；没有执行新中心属性求值或位移后认证。

当前只能说“已完成的数据尚无可见改进证据”。因为还有两个未完成片段，附件中
“三个完整片段均零可见则停止视觉 headline”的整体判据尚未完成评估。

## 4. 构建中止、真实成本与清理

| 阶段 | Wall time | Peak RSS | 结果 |
| --- | ---: | ---: | --- |
| Forest B native build | 2363.93 s | 4,292,042,752 B（进程组） | 达 4 GiB 输出预算的安全余量阈值，中止 |
| Cave native build | 2700.20 s | 2,766,114,816 B（进程组） | 达 45 分钟上限，中止 |
| Mountain coarse | 567.38 s | 2,107,686,912 B（进程组） | 完成 |
| Mountain native build | 1139.93 s | 2,609,541,120 B（进程组） | 完成 |
| Mountain source stage | 42.05 s | 220,839,936 B | 完成，仅源阶段 |
| Mountain OMP 1 harness | 702.53 s | 1,233,027,072 B（父进程） | 含 reference、认证、序列、图像、hash、清理 |
| Mountain OMP 8 replay | 359.12 s | 1,060,675,584 B（父进程） | 实际序列回放与证据复核，不重复 raster |

两次 harness 工作量不同，不能用其总 wall time 宣称 OMP 加速或 production playback
overhead。独立 performance benchmark 属于附件第五优先级，本轮没有冒充完成。

保留失败收据：[Forest B](forest_b/build01/summary.json)、[Cave](cave/build01/summary.json)。
Forest B 输出峰值 4,072,805,034 B，安全阈值 4,026,531,840 B（轮询监控），
未等到 4 GiB 硬额度才停止。Cave 是时间限制，不是磁盘或内存耗尽。

只删除了本轮 Forest B/Cave 的未完成 native cache payload，合计约 3.87 GiB allocated；
该 payload 不能直接恢复，需要重建。原 coarse scene、环境、资产、参数、相机、日志及
紧凑失败诊断保留，清理前后其余文件哈希一致。
收据：[Forest B 清理](forest_b/build01/cleanup.json)、[Cave 清理](cave/build01/cleanup.json)。
Mountain 完整缓存保留，未删除。没有删除旧实验、用户已有 tmp、环境或迁移备份。

所有重任务结束后的观测：仓库 + WSL VHDX + 迁移备份约 **41.15 GB**，
本轮 compact artifacts 约 **42.37 MB**，WSL 本轮持久目录约 **2.08 GB**；
分别低于 400 GB / 512 MiB / 32 GB 上限。WSL 文件不在 VHDX 之外重复计数。
详见 [存储快照](storage_snapshot02.json)。没有进行在线虚拟盘压缩。

## 5. 代码边界与下一步

预注册 protocol SHA 为 `9f8ec007632e763383d87ad544d4be2d0069bc506a89d3f38734e3c1bb6d2ae1`；
176 项冻结绑定复核未变。这是 file-level seal，不是 Git commit。本轮没有 commit/push。
新实验代码位于 `experiments/visibility_screen`，旧生产方法没有改动。

Mountain 首次 screen01 在新 Python adapter 的 SourceVID-shift 缓冲区接线处失败。
冻结 observer 使用固定 20-int transport；已修为正确 transport，同时只暴露真实
一个元素的 1×4 数据，没有 padding 场景，也没有改 C++。旧失败目录保留，
[真实单帧预检](mountain/native_identity_preflight01.json)通过后，才启动新的 screen02。

当前 143 项新 adapter 测试通过；旧 650+14 项单测和 1 项 standalone 结果是哈希绑定的
继承结果，明确未在最终回合重跑。参见 [回归收据](regression_tests02.json)。

**续跑需要先确认预算修订。** 已提出但尚未获得用户答复的修订是：三个固定片段统一
采用最多 8 GiB cache / 90 分钟 native build，其他场景、LOD、种子、准入条件和总存储
上限不变。当前仍严格使用原 4 GiB / 45 分钟协议，没有默许生效。

获准后需建立独立预算修订收据、保留旧 STOP、在新目录重建 Forest B/Cave；不能
静默覆盖旧协议或挑换片段。提高预算也不保证构建或可见性一定通过。
在补完这两段前，不继续增加理论或用新相机追求视觉正例。
