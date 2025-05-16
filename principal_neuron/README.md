# 小鼠脑神经元行为关联可视化工具

本项目旨在分析和可视化小鼠在特定行为（如"Close"，"Middle"，"Open"）中关键神经元的活动数据。
它通过处理效应大小数据来识别关键神经元，并在神经元的相对空间位置上绘制这些神经元，以及它们之间的共享和特有关系。

## 项目结构

```
.
├── data/                     # 存放原始CSV数据文件 (用户需自行准备)
│   ├── EMtrace01-3标签版.csv   # 神经元效应大小数据
│   └── EMtrace01_Max_position.csv # 神经元相对位置数据
├── output_plots/             # 生成的图表将保存于此
├── src/                      # 源代码目录
│   ├── main_emtrace01_analysis.py # 主分析和绘图流程脚本
│   ├── data_loader.py        # 数据加载和初步处理模块
│   ├── config.py             # 配置文件 (阈值、颜色等)
│   ├── plotting_utils.py     # 绘图工具函数模块
│   └── __init__.py           # (可选, 使src成为一个包)
└── README.md                 # 本说明文件
```

## 如何运行

1.  **环境准备**:
    *   确保已安装 Python 和必要的库 (pandas, matplotlib, seaborn)。
    *   可以通过 `pip install pandas matplotlib seaborn` 进行安装。

2.  **数据准备**:
    *   将您的神经元效应大小数据文件命名为 `EMtrace01-3标签版.csv` 并放入 `data` 文件夹。
        *   该文件应包含行为名称，以及对应不同神经元 (`Neuron_X`) 的效应大小值。
    *   将您的神经元位置数据文件命名为 `EMtrace01_Max_position.csv` 并放入 `data` 文件夹。
        *   该文件应包含神经元编号 (`number`) 及其相对X (`relative_x`) 和Y (`relative_y`) 坐标。
    *   *注意*: 当前 `src/data_loader.py` 中的加载函数包含硬编码的示例数据。如需使用您自己的文件，请确保文件存在于 `data/` 目录下，并根据需要调整 `data_loader.py` 中的文件读取逻辑（移除或修改硬编码部分）。

3.  **运行脚本**:
    在项目根目录下执行以下命令：
    ```bash
    python src/main_emtrace01_analysis.py
    ```
    生成的图表将保存在 `output_plots` 文件夹中。

## 配置说明 (`src/config.py`)

*   `EFFECT_SIZE_THRESHOLD`: 用于筛选关键神经元的效应大小阈值。默认根据数据分析建议设置为 `0.4407`。
*   `BEHAVIOR_COLORS`: 定义了不同行为在图表中的基础颜色。
    *   示例: `{'Close': 'red', 'Middle': 'green', 'Open': 'blue'}`
*   `MIXED_BEHAVIOR_COLORS`: 定义了行为对共享神经元在Scheme B图中的混合颜色。
    *   键为按字母顺序排序的行为名称元组。
    *   示例: ` {('Close', 'Middle'): 'yellow', ('Close', 'Open'): 'magenta', ...}`

## 绘图功能及选项 (`src/plotting_utils.py`)

脚本会生成以下9种图表：

1.  **图1-3: 单一行为的关键神经元图**
    *   文件名示例: `plot_close_key_neurons.png`
    *   显示特定行为中，效应大小超过阈值的关键神经元在其相对位置的分布。
    *   选项 (`plot_single_behavior_activity_map` 函数):
        *   `show_title` (布尔值, 默认 `True`): 控制是否显示图表标题。可在 `main_emtrace01_analysis.py` 中调用时修改。

2.  **图4-6: 两行为间共享的关键神经元图**
    *   文件名示例: `plot_shared_close_middle_schemeB.png`
    *   显示两两行为之间共享的关键神经元。
    *   选项 (`plot_shared_neurons_map` 函数):
        *   `scheme` (字符串, 默认 `'B'`):
            *   `'A'`: 仅显示共享的神经元。
            *   `'B'`: 显示两个行为的所有关键神经元（非共享部分半透明），并高亮显示共享的神经元（颜色混合、标记点更大、边框更粗）。
        *   `show_title` (布尔值, 默认 `True`): 控制是否显示图表标题。
        *   可在 `main_emtrace01_analysis.py` 中修改 `scheme_to_use` 变量来切换方案，或为不同方案生成不同的图。

3.  **图7-9: 单一行为的特有关键神经元图**
    *   文件名示例: `plot_unique_close_neurons.png`
    *   显示仅在特定行为中效应大小超过阈值，而不存在于其他行为关键神经元列表中的特有神经元。
    *   选项 (`plot_unique_neurons_map` 函数):
        *   `show_title` (布尔值, 默认 `True`): 控制是否显示图表标题。

## 未来可能的扩展

*   支持更多行为的分析。
*   实现三行为共享神经元的可视化。
 