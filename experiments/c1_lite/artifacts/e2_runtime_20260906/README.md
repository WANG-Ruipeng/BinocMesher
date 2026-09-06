# E2 最小 production runtime 闭环 — 2026-09-06

**结果：真实 C++ 切片输出 + 原生 VID/owner ledger + Python 原子批次替换已跑通。**
新增公共入口 `BinocMesher.slice_window_batch`；保留原默认切片行为，必须显式调用。
测试范围是固定 E2、一个场景元素、raw 模式、一个给定查询批次。
这是 `PASS_PRODUCTION_REQUESTED_SCHEDULE_ONLY`，不是连续窗口准入。

## 实测结果

同一份冻结 E2 source、原窗口 `[102/5,106/5]`、root `104/5`、断点 `21`、
原 boundary / owners / anchors，不改构造、不改场景，不按结果挑选时间。
OMP 1 与 8 分别在独立进程运行一次；两种入口分开判断，不假设彼此输出相同。

| 入口 | 查询数 | 实际替换 | 端点原样 | 窗外原样 | 每批检查的 retained 面累计数 |
|---|---:|---:|---:|---:|---:|
| exact rational | 33 | 31 | 2 | 0 | 118,358 |
| physical double | 41 | 37 | 2 | 2 | 141,266 |

每个 OMP 配置覆盖同一张表；表中的面计数跨查询累计，不是不同事件数。
physical 查询包括原 33 个时刻转换值以及四个断点的相邻可表示 double，按位去重。
记录了实际物理时间、有效离散时间的十六进制表示和最终选择器。

- 新库普通 baseline 与保留的原库逐数组 hash 一致；启用只读 ledger 也不改变它。
- OMP 1 / 8 的 baseline、替换输出、VID / owner ledger hash 分别一致。
- 所有被替换查询通过独立 identity / array validator：实际 owner 恰好消费，
  不靠坐标焊接，四个边界顶点复用，原顶点与 tag 全部逐行逐位不变，
  两个被替换面之外的原面行 ID 与内容全部不变。
- 几何门使用实际 binary32 输出值的精确有理数谓词，检查局部有向 disk、
  非退化 fan、完整接口 link、与全部 retained 三角形的合法接触。
  共享邻面不跳过；没有充分分离证书就拒绝，不用 epsilon 放行。
- 公共 `mesher.slice_window_batch(...)` 与直接受审计的 transaction 输出一致。
- disabled、无效 proposal、unsupported smooth 全部保持对应 baseline；
  在批次后部故意破坏 owner 后，之前已准备的替换（包括 root）也全部丢弃。
  下一次有效调用恢复原结果，没有状态污染。

完整 worker 分别约 15.11 秒（OMP 1）与 15.12 秒（OMP 8），包括普通基线、
观测基线、正式替换、独立检查、公共 API 重跑和负例；两者并发运行。
**这些不是单帧性能或 OMP 加速比测量。** 当前精确谓词主要在 Python 端串行执行。

原始数据：[独立对比](comparison.json)、[原库 baseline](full_reference/result.json)、
[OMP 1](full_omp1/result.json)、[OMP 8](full_omp8/result.json)。
此前 smoke 结果也保留，不能当成额外独立事件。

## 实现与验证边界

流程是：完整 ordinary baselines → 实际身份映射 → 精确源面抑制 → 实际几何门
→ 整批发布；任一查询失败，整批返回其各自同入口、同模式的 ordinary baseline。
普通切片自身失败或超过内存限额时抛错且不发布，不伪造一个成功的 baseline。
支持 1–128 个查询，批次工作数组预算 512 MiB；底层 C++ 全局状态不支持并发
mesher 调用，多线程比较必须使用独立进程。旧 SSP1 干预不能与本入口叠加。

源缓存字节摘要在准备前与发布前均核对。C++ 记录的是实际 ordered VID 到合并后
索引及每个 raw owner 到去重后原生有向面的关系，保留 many-owner → one-face。
本次替换发生在 Python 端：核对完整 raw owner 等价类后，退休去重后的两张
source face，再写入四面 fan。它不是调用旧 C++ `should_suppress` 逐条跳过
raw emission；“owner 消费”在这里指完整等价类覆盖且对应源面仅退休一次。
C++ 负责生成不变的 baseline 和实际身份账本，Python 负责受检事务替换。
Python 在读完 ledger 后才取走 mesh；成功或异常均显式清除 pending 输出与观测状态。
原默认 API、旧 SSP1/C0 语义和已安装的 WSL `core.so` 均未替换。

代码：[公共入口](../../../../binocmesher/core.py)、
[批次运行时](../../../../binocmesher/window_runtime.py)、
[实际几何门](../../../../binocmesher/window_geometry_runtime.py)、
[C++ ledger](../../../../binocmesher/source/runtime_identity.h)、
[独立 validator](../../e2_runtime_validation.py)、[复现实验](../../run_e2_runtime.py)。

特别保留以下限制：

- `continuous_window_admitted=false`。逐查询认证不能替代任意实数时间的全窗口证明。
- 旧 `window_admission` 的连续时间门没有提升；其他三个 demo 的拒绝原因没有改动。
- 全窗口有界扰动的条件证书是补充分析，不是本次提交批次的授权条件；
  不能仅凭它推断所有实际浮点求值、选择器、owner 和最终数组映射已经全时域绑定。
- 没有渲染、SSIM、六场景覆盖率或新的保真提升测量，也不宣称全网格原有缺陷消失。
  历史 A/B 保真结果原样保留，不把它们记成本轮 runtime 提升。
- 新库仅在隔离路径验证；尚未部署覆盖日常 WSL checkout 的库。

## 复现与存储

[复现命令与输入绑定](REPRODUCE.md)。本轮 worker 只输出紧凑 JSON，数组不落盘。
每次只复制约 4.26 MB 的小缓存，结束自动删除；原始 cache 校验不变。
隔离 build 初始约 2.66 MB，保留供复现。没有清理环境、历史实验或迁移备份。
最终文件指纹和测试记录见本目录 `verification.json`。
