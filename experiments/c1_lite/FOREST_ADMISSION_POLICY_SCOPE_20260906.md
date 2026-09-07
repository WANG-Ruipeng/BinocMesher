# Forest 准入政策的证据范围

本文解释既定实验统计的含义，不修改构造、准入条件或正在运行的代码，也不预报最终准入数量。

## 统计的是固定证书政策的覆盖率

Forest 实验对固定的 131 个 canonical events，分别在同一个 ordinary baseline 上构造单事件替换；其他事件保持 baseline。每个事件检查既定自然帧及独立 exact-root 查询，统计的是 **保守、有限请求序列准入政策的覆盖率**。

这不是所有可能修复方案的存在性判定，不是同根多事件联合准入率，也不是任意实数时刻的连续窗口定理或画质提升率。单查询通过仍只是候选计划；全部所需查询和输入绑定检查通过之后，才能按请求序列原子提交。

## 两项容易误读的限制

1. **严格凸 XY 边界与中心严格内部。** 当前局部证书要求原支持边界的 XY 投影严格凸，实际 binary32 中心严格位于其内部。这提供一个容易精确检查的、四扇形非退化且同向的嵌入图面条件。严格凸性是当前明确采用的充分条件，不是一般四扇形嵌入的数学必要条件：某些凹但星形的边界也可由位于其可见核内的中心形成嵌入扇形。中心严格内部则是当前“凸 XY 图面、各扇区非退化且同向”合同的必要条件。违反这些条件，只能证明当前固定构造/证书政策不接纳该候选，不能证明其他支撑、投影或构造均不可能成功。

2. **与接口相接的 retained 退化三角形。** 当前相对保证要求有效的原支持 disk 和规则的 retained 接口邻域。实际 retained 三角形若与边界共享身份、且精确叉积为零，当前政策拒绝该候选。这里检查的是原 baseline 已经存在的 retained 面，不是替换新生成的面；这种拒绝不证明方法新增了退化，也不证明保留该既有退化的其他相对合同不可能成立。远离补丁的既有退化仍可保留，只要已证明它与补丁分离。扩大接口允许的退化情形需要另外定义并证明合同，不能仅为提高覆盖率而取消检查。

## 拒绝、未证与回退应分开报告

- **具体政策拒绝：** 具备原 SourceVID 身份和实际几何绑定，并有明确见证违反预先声明的源支持、局部图面或接口条件。这里的“必要”是指该固定准入政策要求的条件，不是所有可行修复的必要条件。
- **UNKNOWN：** 身份编码未验证、元数据或实现绑定缺失、资源界限耗尽，或者保守分离搜索没有找到证据。AABB 重叠或找不到分离面，都不是相交证明。旧 observer 的 merger-normalized key 也不能作为原 SourceVID 身份使用，其造成的拒绝不能计为方法失败。
- **Fail-closed：** 拒绝或未证时不发布该事件的部分替换，整个请求序列保留 ordinary baseline。该合同保持已有网格，不承诺修好 baseline 的其他既有缺陷。

因此，即使最终固定政策覆盖率为 `0/131`，准确含义也只是“这一预先声明的保守证书政策未接纳该总体中的事件”。不能改写为“全部事件都不可修复”“算法在数学上不可能”或“方法新增了这些 baseline 缺陷”。若仍有 UNKNOWN，则应同时报告其数量和准入率界限，而不是给出无条件的最终点估计。

可用于论文的简洁措辞：

> We report applicability under a predeclared conservative admission policy, not completeness over possible repairs; a policy rejection identifies a violated certificate precondition, not impossibility of an alternative repair or evidence that the replacement introduced a baseline defect.

代码与协议依据：`forest_native_patch.py` 的原身份编码门、实际局部检查与接口退化见证；`binocmesher/window_geometry_runtime.py` 的 graph-fan/interface/contact 判据；`FOREST_FORMAL_ADMISSION_PROTOCOL_20260906.md` 的固定总体、有限请求序列和 PASS/REJECT/UNKNOWN 语义。
