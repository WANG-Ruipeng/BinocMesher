# 固定 schedule / E2 图像 / Forest 准入筛选：本轮结果

E2 的结构清单和小型几何图像实验已完成；Forest 已完成全部 131 个
registry events 的前端核验和一个有限模板试验，但**完整 C1 的
closure-derived source-support eligibility 尚未完成，实际准入率为 N/A**。
不能把有限模板的 0/131 匹配误写成完整方法覆盖率为零。

生产代码仍冻结在 `000028e0fa5bc4922aeb51aa9fb9ee96b799301e`。
本轮只新增实验脚本、测试及小型结果；没有 RGB、长视频、场景重建、
环境变更、生产算法修改、提交或推送。

## 四项交付的实际状态

| 项目 | 本轮结果 | 适用范围 |
| --- | --- | --- |
| schedule-level structural-coherence manifest | PASS | E2 五个自然查询 + 独立 exact-root 查询；不是全实数时间定理 |
| E2 depth / normal / boundary | 已完成 | 原相机和固定局部诊断相机分开；三方法共用参考和度量域 |
| Forest 131-event eligibility funnel | 前端已完成；完整源支持阶段未评估 | 临界面直接四边形模板不是 E2 所用的闭包支持 |
| admission / fail-closed / hits / cost | 已记录可测项；未执行项明确 N/A | 不把未评估、模板试探失败、实验脚本 STOP 混为运行时失败 |

## E2：自然帧确实改变，但不是全指标改善

真实 24 FPS 相位中只有零起始 frame 15 命中窗口：全 24 帧的时间命中率
1/24 = 4.1667%；选取的五个自然帧中，修改率为 1/5 = 20%。其余四帧
与各自 ordinary raw 输出逐字节一致。精确根单独请求，不计入自然帧分母。

两种 window 方法均成功完成五自然帧的原子事务与独立根事务。
全部 3 个源时间分支和 4 个单点的身份/owner 关系通过；逐查询检查完整
源面抑制、外部数组不变和实际几何，并验证了提交前强制拒绝的整批回退。
该人工负例通过，不是观测到的自发失败率。

原相机看不到 E2 支持区域，可见支持为 0 像素，五自然帧的三种方法图像
完全一致。因此，本例不能作为原视角画质或时间 SSIM 改善的证据。

固定局部诊断视角使用原有 mesh，不重建 LOD。自然 frame 15 的原始两源面
足迹为 16,034 像素；它与 full raw 遮挡后的可见支持一致，三方法没有
缺失/新增覆盖。相对于独立地形参考：

| 方法 | Depth 平均绝对误差（场景单位） | 平均法线误差（度） |
| --- | ---: | ---: |
| ordinary raw | 0.172622 | 21.789374 |
| centroid-window | 0.168302 | 21.765772 |
| BEB1-anchor window | 0.165272 | 21.827989 |

BEB1 的 depth 平均误差相对 raw 降低 **4.2578%**，相对 centroid 降低
**1.8000%**；法线平均误差却分别增加 **0.038615°** 和 **0.062217°**。
法线回退不能当成参考细化噪声忽略。silhouette / pixel-boundary XOR 为 0。
深度 P95、RMSE 改善，但最大误差和法线尾部并非全部改善。

40,420 像素的完整 window-bbox ROI 另报，未用更小足迹代替预先声明的
主 ROI。参考 128/256 网格满足预先固定的覆盖、depth 和 normal 细化
阈值；没有修改相机或阈值来追求正结果。这是经验细化检查，不是完整
误差上界或统计显著性检验。

参见 [E2 详细报告](e2_retry2/README.md)、[尾部指标](e2_retry2/TAIL_METRICS.md)、
[结构 manifest](e2_retry2/structural_coherence.json)、[完整度量](e2_retry2/result.json)。
图示：[原相机](e2_retry2/original_natural_15_overview.png)、
[固定诊断相机](e2_retry2/diagnostic_natural_15_overview.png)。参考图的黑色外部
是有限参考域以外，不是候选网格的孔洞。

## Forest：前端通过，完整方法准入暂不能计算

输入是 64 帧、6 px 的 Forest 预位移缓存：440 raw observations、268
logical incidences、131 canonical events。两个 exact-root groups 分别是
3/2 上 66 个、5/2 上 65 个事件，不是六场景或论文 480 帧设置。

- 131/131 通过精确 registry saddle、引用来源记录及局部 completed-kernel
  非退化检查；这不等于完成 source-boundary 拼接或全局几何认证。
- 流式检查 3,081,976 个 processed records，只保留 2,399 个 halo polygons，
  来自 1,737 个记录，没有复制或重建约 1.10 GB 的整个缓存。
- 尝试以 critical face 四条边直接构造四边界的 two-face 模板，匹配为
  0/131。**该模板不是完整 C1 方法，不能据此报告 C1 覆盖率。**
- 关键自检反例就是已成功的 E2：critical face 包含
  `46:10, 311:0, 159:8, 391:8`，但实际闭包边界含 `46:10|376:8`。
  `376:8` 不在上述四顶点内；正式源支持来自完成后的 replacement plan，
  并非直接 critical-face quad。把这两种边界混用会错误解释筛选结果。
- 因而完整 source-support 编译、接口、局部窗口几何、外部几何及同根
  atomic plans 尚未建立。4,225 个同根候选对的兼容性仍为 UNKNOWN。
- 当前公开 window runtime 是单元素入口，Forest OpaqueTerrain 有五个
  元素；本轮没有执行 Forest runtime transaction，也没有生成其修改帧。

真正方法级别的结果解释以 [scope correction](forest/scope_correction.json)
及其说明为准。原始 [probe summary](forest/summary.json) 和
[逐事件记录](forest/events.json) 保留审计，其 `source_owner_closure=REJECT`
以及 `structurally_eligible_candidates=0` 只属于这个有限候选模板。

### 分母与 fail-closed 原因

| 统计 | 结果 | 不应如何解释 |
| --- | --- | --- |
| 前端局部核检查通过 | 131/131 | 不是完整 BEB1/source-support admission |
| 直接四边形模板匹配 | 0/131 | 不是完整方法覆盖率 0% |
| 正式闭包源支持资格率 | N/A | 正式源支持编译未执行 |
| Forest 实际 runtime 准入率 | N/A，尝试数 0 | 不是运行失败率 100% |
| Forest 实际修改自然帧 | 本轮未生成 | 不能由候选时间命中代替 |
| 未准入候选窗含自然帧 | 131/131；每窗 8 帧 | 不是认证窗口命中率或可见收益 |

这些未准入候选窗对应 frame numbers 21..28 或 37..44（Forest 使用
1-based 帧号）；共 1,048 个 event-frame incidences、16 个不同帧。
64 帧中的 16/64 仅表示候选时间区间并集，不是实际修改率。
时间映射从已执行 camera inputs 重建，并非 C++ 序列化时间参数。

本轮需要保留的原因是：direct-quad template mismatch、正式 closure
source-support compiler 尚未接入全 registry、外部/同根认证未建立、
单元素 runtime 不支持该五元素输入。它们不是已证实的网格缺陷。
没有测到 Forest 自发 fail-closed 次数；E2 的两次 harness STOP 是临时
日志检查规则问题，单列保留，不并入方法的事件拒绝统计。

## 实际成本、测试与数据保留

| Worker | 墙钟时间 | 峰值进程 RSS | 说明 |
| --- | ---: | ---: | --- |
| E2 成功实验 | 25.3869 s | 286,310,400 B | 含结构核验、参考细化和几何 PNG；OMP=1 |
| Forest 有限前端/probe | 36.1486 s | 330,682,368 B | 约 35.53 s 用于流式扫描 |

E2 另外两次 harness STOP 为 7.9169 s 和 8.0423 s；独立 prepare-only
预验也已通过。上述不是冷/热缓存严格微基准，也不包含脚本开发时间。
复用的 Forest 历史建缓存耗时 1203.2554 s，不能算免费或混入本轮扫描成本。

全部 **345 项测试通过**（331 C1-lite + 14 source-splice）。原生产文件
无差异；E2 原始缓存全文件检查不变。Forest 核查已用完整输入/引用记录
哈希和扫描文件时间戳，不宣称对无关 processed payload 做了完整 SHA。

本轮持久化结果约 **2.9 MB**，不保留 mesh 或全精度图像数组。E2 所有临时
缓存副本已删除，两项 worker 均已退出。仓库、WSL VHDX、迁移备份合计
约 **40.3 GB**，低于 400 GB；未删除环境、原始缓存或已有实验。

复现命令与限制见 [REPRODUCE.md](REPRODUCE.md)。

## 下一步边界

本轮证据支持 E2 的固定 schedule 闭环及一个视角上的局部 depth 改善，
同时保留 normal 回退和原视角零变化。要给出 Forest 的真实 admission
rate，下一步必须把 registry event 接到正式的闭包源支持/替换计划编译，
而不是继续使用直接四边形模板。本轮发现不等价后已停止将其用于方法
结论，**没有擅自开发或修改生产通用编译器**。
