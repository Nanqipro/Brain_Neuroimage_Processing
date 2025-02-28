# 交互式神经元网络可视化

本文档说明如何生成和使用交互式神经元网络可视化工具。

## 功能特点

- 交互式可视化神经元功能网络
- 鼠标悬停在神经元上时高亮显示其连接关系
- 可缩放、拖动和选择节点
- 保留原有的颜色分组（相同颜色表示属于同一功能模块）
- 显示神经元之间的相关性权重
- 优化的视觉布局，节点大小适中，连接线透明度合理
- 清晰的视觉层次结构，更接近于静态图的美观度

## 视觉优化

最新版本进行了以下视觉优化：

- 减小了节点大小，使网络图更清晰、更美观
- 降低了连接线的透明度，减少了视觉上的拥挤感
- 使用了 Barnes-Hut 物理引擎，提供更均匀的节点分布
- 优化了节点间的间距，减少了重叠
- 添加了标题、操作说明和图例
- 对不同类型的交互增加了视觉反馈

## 使用方法

### 生成交互式可视化

有两种方法可以生成交互式可视化：

1. **通过运行分析脚本**（如果您尚未运行过分析）：
   ```
   python src/analysis_results.py
   ```
   这将执行完整的分析流程，并在分析结束时自动生成交互式可视化。

2. **通过专用脚本**（如果您已经运行过分析）：
   ```
   python src/create_interactive_network.py
   ```
   或者双击`run_interactive_network_visualization.bat`批处理文件。

### 查看交互式可视化

生成的交互式可视化保存在`results/analysis/interactive/interactive_neuron_network.html`文件中。您可以使用任何现代浏览器打开此文件。

### 交互操作说明

在交互式可视化中，您可以：

- **悬停在神经元上**：突出显示与该神经元相连的所有连接
- **点击神经元**：固定选择该神经元及其连接
- **使用鼠标滚轮**：缩放网络
- **拖动空白区域**：移动整个网络
- **拖动节点**：重新排列网络结构

## 技术说明

交互式可视化基于以下技术构建：
- PyVis：Python的交互式网络可视化库
- vis.js：底层JavaScript可视化库
- NetworkX：Python的图形分析库

## 故障排除

如果您在生成或查看交互式可视化时遇到问题：

1. 确保已安装PyVis库：`pip install pyvis`
2. 确保已运行分析并生成了`network_analysis_results.json`文件
3. 如果HTML文件无法正确显示，请尝试使用不同的浏览器（推荐Chrome或Firefox）

### 常见错误解决方案

#### JSON解析错误

如果遇到类似此错误：
```
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column X (char Y)
```

这通常是由于在设置网络选项时JSON格式不正确导致的。最新版本已修复此问题，使用直接的JSON字符串而非Python字典来设置选项。如果您仍然遇到此问题，请尝试：

1. 确保使用的是最新版本的代码
2. 检查`visualization.py`和`create_interactive_network.py`中的`set_options()`部分是否使用正确的JSON字符串
3. 确保JSON字符串中不包含JavaScript风格的注释 (//)

#### Python字典错误

如果遇到类似此错误：
```
AttributeError: 'dict' object has no attribute 'replace'
```

这是因为尝试将Python字典直接传递给pyvis的`set_options`方法，但该方法期望接收的是字符串。解决方法：

1. 使用本项目最新版本的代码，已经修复了这个问题
2. 如果您自行修改了代码，请确保使用正确格式的JSON字符串调用`set_options`方法：
   ```python
   # 正确方式
   net.set_options("""
   {
     "physics": {
       "enabled": true
     }
   }
   """)
   
   # 或者使用json.dumps转换字典
   options = {"physics": {"enabled": True}}
   net.set_options(json.dumps(options))
   ``` 