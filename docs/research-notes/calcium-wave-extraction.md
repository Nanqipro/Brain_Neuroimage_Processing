# 钙波提取算法研究：suite2p 与 CaImAn

> 这是一份方法调研记录，不是已实现的端到端钙波传播模块。链接指向上游项目；本仓库不再分发它们的完整源码副本。上游代码会变化，正式复现时请固定软件版本并记录参数。

- 结论：两个项目均未提供面向“钙波（spreading calcium waves）”的专用提取/传播检测模块。两者内置的是通用的钙成像处理管线（配准、源分离/ROI、去卷积、质量评估）。若需研究钙波，可基于它们的成像去噪、ROI/像素级时间序列与事件检测能力，组合实现“波”事件的检测与传播分析。

## suite2p：与钙波相关的能力与提取逻辑

- ROI（细胞）检测逻辑：
  - 通过对电影数据的协方差做 SVD 得到空间主成分（PCs），再进行空间平滑、噪声方差归一化，构造“相关性图”，在其上迭代找峰并扩展 ROI，期间按 raised-cosines 建模并回归去除 neuropil 贡献，直至停止条件满足。
  - 关键步骤与参数：
    - 数据分箱、SVD、相关性图、峰检测、ROI 迭代与 neuropil 处理：[suite2p cell detection 文档](https://github.com/MouseLand/suite2p/blob/main/docs/celldetection.rst)

- 去卷积（尖峰/瞬变提取）逻辑：
  - 先对校正荧光（`F - neucoeff*Fneu`）做去基线（“maximin”：高斯平滑→滑动最小→滑动最大），再用 OASIS 非负约束的活跃集算法做去卷积，输出 `spks` 作为瞬变/活动的稀疏估计。
  - 关键实现：
    - 原理与参数：[suite2p deconvolution 文档](https://github.com/MouseLand/suite2p/blob/main/docs/deconvolution.rst)
    - 实现：[suite2p extraction/dcnv.py](https://github.com/MouseLand/suite2p/blob/main/suite2p/extraction/dcnv.py)
    - 管线入口：[suite2p run_s2p.py](https://github.com/MouseLand/suite2p/blob/main/suite2p/run_s2p.py)

- 与“钙波”关系：
  - suite2p 的“Spike detection/去卷积”在 ROI 层面给出时间序列的瞬时活动。并无“传播波前/波速/空间扩散”专用分析；但可基于所有 ROI 的去卷积轨迹，检测群体同步事件或按空间邻近的起始时间构造波前。

## CaImAn：与钙波相关的能力与提取逻辑

- CNMF/CNMF-E 源分离与时间成分提取：
  - 将电影分解为空间（A）与时间（C）成分，并联合背景与噪声建模；时间维上采用约束 FOOPSI/OASIS 做去卷积与活动估计，支持 AR(1)/AR(2) 指示器模型与在线版本。
  - 关键实现：
    - 时间去卷积（并行 FOOPSI 调用）：[CaImAn temporal.py](https://github.com/flatironinstitute/CaImAn/blob/main/caiman/source_extraction/cnmf/temporal.py)
    - OASIS Cython 实现（AR 模型与活跃集池合并）：[CaImAn oasis.pyx](https://github.com/flatironinstitute/CaImAn/blob/main/caiman/source_extraction/cnmf/oasis.pyx)

- 组件质量评估与事件区间：
  - 通过 SNR、空间相关与 CNN 置信度筛选成分；同时提供从时间序列提取峰/事件区间的工具（用于质量评估）。
  - 相关代码：
    - 质量筛选接口：[CaImAn estimates.py](https://github.com/flatironinstitute/CaImAn/blob/main/caiman/source_extraction/cnmf/estimates.py)
    - 峰/事件区间工具：[CaImAn components_evaluation.py](https://github.com/flatironinstitute/CaImAn/blob/main/caiman/components_evaluation.py)

- 光流（行为模块）：
  - 提供与光流相关的函数，适合用于估计时空活动的运动场，可用于构造波前传播的方向/速度，但未与钙成像 CNMF 管线直接耦合。
  - 入口文件：[CaImAn behavior.py](https://github.com/flatironinstitute/CaImAn/blob/main/caiman/behavior/behavior.py)

- 与“钙波”关系：
  - CaImAn 强项在于像素/补丁级的源分离与时间反卷积，输出组件时间序列与事件；并无现成的“钙波传播检测”模块，但可结合光流与局部相关图进行波前追踪。

## 面向钙波的建议分析流程（基于现有能力组合）

- 预处理与对齐：
  - 使用 NoRMCorre（CaImAn）或 suite2p 的注册模块完成运动校正。
- 时序活动提取：
  - ROI 级：直接用 suite2p 的 `spks` 或 CaImAn 的 `S`（FOOPSI/OASIS）作为活动轨迹。
  - 像素/网格级：在 CaImAn 中以小块/像素为单位提取 `C`/`S`，或用局部相关图（local correlation）作为功能模板；具体接口应以所固定版本的 CaImAn 示例与文档为准。
- 波事件检测：
  - 在所有轨迹上进行峰检测，记录第一次超过阈值的“起始时间”；对空间邻近像素/ROI 进行连通标记，形成同一事件的空间簇；以起始时间的空间梯度拟合波前。
- 传播方向与速度：
  - 在原始/去噪帧序列上计算光流（CaImAn 行为模块），在事件窗口内对光流向量场空间平均，得到主传播方向与速度估计。
- 指标与可视化：
  - 波前速度（μm/s）、方向分布、事件覆盖面积/持续时间；叠加等时线/光流箭头于局部相关图或均值图。

## 何时选择哪套方案

- 数据以清晰细胞为主（2P，背景较低）：优先 suite2p（ROI 检测 + dcnv 简洁高效），波分析基于 ROI。 
- 1P/endoscope、强背景/重叠源：优先 CaImAn（CNMF-E），必要时在像素网格上做时序提取，再结合光流做波分析。

## 结语

- 两套工具均未内置“钙波”专用提取算法；它们提供的是可用于构建传播分析的时空信号与事件检测基础。真正的钙波模块仍需明确事件连接规则、空间标定、速度估计、误差评估与合成基准。
