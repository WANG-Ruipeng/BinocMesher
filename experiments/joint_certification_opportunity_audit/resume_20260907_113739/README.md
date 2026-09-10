# 联合认证剩余优化机会审计：完成

判决：**`INSUFFICIENT_COST_EVIDENCE`**。

确实找到了仍被重复检查的关系，但现有收据不能说明这些关系消耗了多少时间，
也没有建立一个值得为之优化的实际应用收益。因此本轮不启动优化、trace、native
重建、WMTK 或更大实验。这不是“证明永远没有优化机会”，而是当前证据不足以
支持继续扩张通用认证框架的论文主张。

本轮仅重新解析已有 JSON、核对执行源码快照并进行静态代码审阅。
[汇总清单](existing_receipt_inventory.json) 的状态为
`PASS_EXISTING_RECEIPT_INVENTORY`，不是新几何实验的 PASS。
只读汇总耗时 16.016 秒，峰值 RSS 26,660,864 字节，native 调用为 0。
[前次 STOP](../run_20260907_112855/STOP.json) 保留；本次是用户批准后的
[独立恢复协议](PROTOCOL.md)，没有删除旧保护或重跑旧 A/B/C0。

## 1. 贡献假设更新

| 主张 | 证据结论 | 论文处理 |
|---|---|---|
| 来源信息对指定授权合同必要 | A 支持，前提是共同可信、完整 ledger | 保留为机制基础 |
| 当前来源 resolver 有特殊能力或计算优势 | 普通独立参照复现全部受测决定及主要工作，耗时接近 | 暂时移出优势列表；不是全域统计等价证明 |
| 单项安全足以推出联合安全 | B 的严格反例否定这一推论 | 保留联合接触检查的必要性 |
| 当前联合检查/事务组织特别高效 | B 未测该比较，C0 每查询仅一个事件 | `UNMEASURED` |
| 数组编辑后端是已测关键瓶颈 | C0 装配约 0.04 ms，活跃路径约 33–34 ms | 不支持为性能先迁移后端 |
| BEB1-window 已提供明确应用收益 | 已测 Forest 原相机几何 buffers 未变 | 用途仍未建立，不能用认证 PASS 替代 |

准确数值直接取自 [C0.json](../../source_contract_probe/artifacts/run_20260907/C0.json)：
两个活跃查询的 resolver 中位数范围为 current **11.528799–11.603739 ms**、
ordinary **12.093740–12.405301 ms**。这些是“两个查询的中位数范围”，不是置信区间
或全部样本极值。这里恢复数值，不覆盖旧报告或原始收据。

## 2. 已有 Forest 工作清单

令 `P_i` 为退休原面，`Q_i` 为新 fan，`R = M0 \ union(P_i)` 为最终 retained mesh。
以下“实际播放”指已有研究验证版联合构造路径，不是已经测量的纯生产延迟。

### 2.1 先分清三种人口与工作类别

| 已有收据/路径 | event-query pairs | 逻辑三角形关系 | 外层 AABB 排除关系 | contact 函数调用 |
|---|---:|---:|---:|---:|
| 初级组件认证，18 个查询 | 2,079 | 66,528 | 65,184 | 1,344 |
| 每次实际播放，OMP 1 或 OMP 8 | 1,296 | 41,472 | 40,128 | 1,344 |
| 独立补充审计 | 2,079 | 66,528 | 65,184 | 1,344 |

初级组件认证与补充审计的数字恰好相同，但它们是分别保存、分别执行的证据。
本次从 `component01/summary.json` 指向的 `query_artifacts[].pair_proofs` 独立汇总了
第一行；第三行读取独立审计的 `counts`。不能因为数字相同就把两条路径混为一谈。

实际播放每次构造 64 个自然输出和 2 个根诊断；其中 18 个查询有编辑，包含
225 个 event-query plans，其余 48 个输出无编辑。OMP 1/8 的上述工作计数相同，
但不是新增的独立场景样本。两次播放分别继承自己的 225 份单项证书，不合并分母。

两类初级路径中，1,344 次调用都分为：`Q_Q=672`、`Q_A_OTHER_REMOVED_B=336`、
`Q_B_OTHER_REMOVED_A=336`。**这些是函数调用数，不是昂贵线性求解次数。**
[`check_triangle_contact`](../../c1_lite/forest_exact_contact.py:179) 内部仍可通过
`STRICT_EXACT_AABB` 返回；成功 union 收据没有保留这些内部快速退出/求解工作量。

### 2.2 按检查角色拆分

| 关系/工作 | 已有实现与本次确认 | 数量/复用口径 | 独立耗时 |
|---|---|---|---|
| `Q_i`–最终 `R` | 已继承同查询单项完整 retained 证书；播放没有重新调用全 exterior `audit_query` | 每次播放 225 份绑定计划；被继承的 primitive 关系总数 `UNMEASURED` | `UNMEASURED` |
| `Q_i`–其他将删除的 `P_j` | union 仍检查；支持不交且绑定一致时，单项完整证书已覆盖这些旧面 | 每次播放两方向 336+336 次 contact 调用；这些关系的继承尚未被 `_interactions` 使用 | `UNMEASURED` |
| `Q_i`–`Q_j` | 对单项证书而言是新增关系；组件认证检查过，验证版播放又检查 | 每次播放 672 次 contact 调用；对应 42 个未被外层 AABB 排除的 event-query pairs | `UNMEASURED` |
| Owner/来源索引 | 每个不可变 snapshot 内按 element 建索引并复用，再逐请求解析 | 实际索引读取/构建次数未记录；不是每事件完整重建全场景索引 | `UNMEASURED` |
| 接口/局部几何 | 完整 boundary star 按 element 共享全扫描；union 另验原盘、局部 graph、成员与不交支持 | 不能因接触证书可继承就移除 owner、接口、组件门 | `UNMEASURED` |
| Broad phase | 显式 pairs 枚举；一次严格 support AABB 排除该对全部 32 个关系 | 播放 1,296 对，其中 1,254 对被外层排除；不是 40,128 次单独 AABB 求解 | `UNMEASURED` |
| 绑定/存储 | 核对 query/mode、source、完整 baseline、fresh support、center、计划与组件证据 | 比较/读取字节、lookup、存储维护成本未分别记录 | `UNMEASURED` |
| 输出构造及保留性审计 | 构造数组，反复核对 vertices/tags、retained faces 与 baseline | 混在 verified-union 计时中，不能整段当作几何重复工作 | `UNMEASURED` |

关键代码定位：

- [完整单项 retained 检查](../../c1_lite/forest_native_patch.py:314)、
  [同查询证书绑定](../../c1_lite/forest_sequence_inputs.py:86)。
- [每 snapshot/element 来源索引](../../c1_lite/forest_native_patch.py:120)、
  [共享 boundary-star 扫描](../../c1_lite/forest_component_graph.py:69)。
- [组件 pair 认证](../../c1_lite/run_forest_component_certification.py:44)、
  [union 新新/新旧检查](../../c1_lite/forest_component_union.py:123)、
  [播放重新调用 union](../../c1_lite/run_forest_atomic_sequence.py:189)。
- [单独启动的补充审计](../../c1_lite/run_forest_atomic_sequence.py:115)、
  [数组保留性核对](../../c1_lite/forest_component_union.py:213)。

`Q_i`–`P_j` 的重复不是偷偷降低政策：单项 PASS 只接受严格分离或约定共享身份特征
上的接触；union 要求 boundary/source 支持不交，而 `P_j` 只用自己的 boundary
vertices。因此该对没有共享身份，两个路径要求的都是空交集。结论仍以完整单项
覆盖、相同 query/baseline/实际坐标/政策以及正确身份对应为前提。

对于 `Q_Q`，本次核对了两组播放中全部 225 个计划与组件认证的 query、完整 baseline、
plan 一致，且每个播放 pair 都有先前组件 PASS。不能因此直接跳过验证版重放：
派生中心的数值 ID 会在联合编排中改变，复用还需明确验证身份双射、共同特征与
certificate 适用范围。将独立复核静默删掉，也不构成算法优势。

### 2.3 真正可用的时间证据

| verified-union 子总体 | OMP 1 | OMP 8 |
|---|---:|---:|
| 全部 66 个输出 | 4.556426 s | 4.503537 s |
| 18 个有编辑/根查询 | 2.305221 s | 2.274712 s |
| 48 个无编辑查询 | 2.251205 s | 2.228825 s |

本次逐帧求和与 summary 完全吻合。这段函数即使收到空 records，也会核对/生成
整网格收据。因此不能将全部 4.56 秒视为多事件接触成本。**2.305 秒同样只是有编辑
union 的复合时间，不是可节省时间**；它包括准备、局部检查、接触调用、分配拷贝、
多次全数组核对等。

单独的解析/完整接口重放合计约 14.02 秒，也没有内部分类计时。native slicing、
额外 frozen-baseline reference、输入绑定、补充审计和清理另有范围，不能拼出一份
不存在的、完整且互不重叠的生产成本分解。独立补充审计的 54.585 秒尤其不能算成
“取消后算法就省下的成本”。

## 3. 优化机会与用途判决

**`INSUFFICIENT_COST_EVIDENCE`，不批准/启动 C1 原型或更大实验。**

已识别的是具体重复关系，不是已确认的昂贵热点：

| 需要决策的量 | 当前状态 |
|---|---|
| 同一真实 workload 上的纯方法 `T_base` | `UNMEASURED` |
| 可在同合同下移除的 `T_redundant` | `UNMEASURED` |
| 新增 binding/lookup/storage 开销 | `UNMEASURED` |
| `S_max = T_base / (T_base - T_redundant)` | `UNMEASURED`，不由关系个数猜测 |
| 明确的消费者、速度目标与收益门槛 | 尚未建立；不自动采用 1.2x |

原 Forest 结果仍是 16 个自然帧网格改变，但已测原相机 pre-displacement 几何 buffers
没有可见变化；该负结果不因认证分析而消失。当前合理论文定位是：

> 已验证机制；受测来源前端没有优势证据；联合认证收益和用途仍待建立。

如果以后另行批准定位成本，可固定已有 `frame_0027` 的 12 个 edits，单查询、无方法
改动地拆分计时。其历史复合 union 时间为 0.150851 秒，包含 66 个事件对和
96 次 contact 调用（48+24+24）；它只是一个明确候选，不是已确认瓶颈。
此次**没有运行该 trace**。

现有收据没有保留完整 mesh dump，两组每帧均记载 `persistent_mesh_files=0`。
紧凑 patch 证据与完整 baseline arrays 不同，不能从 hash 还原几何；如后续需要恢复
完整数组或 native 重建，必须单独批准。常规 memoization/索引应当是公平参照，
不能削弱冻结参照的既有继承能力制造优势。

## 证据完整性与停止边界

- 已验证全部 18 份组件查询、两组各 66 份输出的父收据哈希、分母、角色计数、状态与计时总计。
- 两组各 225 个播放计划与同 query/mode、完整 baseline、原组件计划匹配。
- 组件和两组播放各 7 条 executed-source 绑定与当前源码一致。
- native04 快照的 15 份源码（208,964 字节）、清单大小、原 summary 绑定及当前版本全部一致。
- 历史补充审计首次绑定的依赖仍按补充证据处理，没有追认早期未记录的哈希。
- 未修改生产代码、A/B/C0、历史 Forest 收据、环境或缓存；未删除、提交或推送。
- 上轮提取错误 STOP 保留；本次恢复后的汇总无异常，停止在已批准的只读审计边界。

复查入口为 [aggregate_existing.py](aggregate_existing.py)，只使用 Python 标准库读文件。
完成收据和绝对截止时间阻止本次入口再次执行，不应删除它们绕过终态。
