# matplotlib字体警告解决方案

## 问题描述

运行神经元状态分析时出现大量字体警告：
```
Font 'default' does not have a glyph for '\u2212' [U+2212], substituting with a dummy symbol.
```

## 原因分析

1. **Unicode负号问题**：`\u2212`是Unicode标准的负号字符
2. **字体缺失**：系统默认字体没有该字符的字形
3. **matplotlib配置**：中文字体配置不完整

## 解决方案

### 方案1：已修复的代码配置
代码中已添加以下配置来解决问题：

```python
# 设置中文字体支持和数学符号处理
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = ['sans-serif']

# 额外的字体配置
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.default'] = 'regular'

# 禁用字体相关警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
```

### 方案2：手动字体配置
如果仍有问题，可以手动配置：

```python
import matplotlib.pyplot as plt
import matplotlib

# 禁用Unicode负号
plt.rcParams['axes.unicode_minus'] = False

# 设置备用字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

# 忽略警告
import warnings
warnings.filterwarnings('ignore')
```

### 方案3：系统字体安装
在不同系统上安装合适的字体：

**Windows:**
- 确保安装了微软雅黑或SimHei字体
- 重启Python环境

**macOS:**
- 使用系统自带的PingFang SC字体
- 或安装SimHei字体

**Linux:**
- 安装文泉驿字体：`sudo apt-get install fonts-wqy-microhei`
- 或使用：`sudo yum install wqy-microhei-fonts`

## 验证解决效果

运行字体测试脚本：
```bash
python font_config.py
```

这将：
1. 自动配置最适合的字体
2. 生成测试图片验证效果
3. 显示当前字体配置信息

## 预期效果

修复后应该：
- ✅ 不再出现字体警告信息
- ✅ 中文文本正常显示
- ✅ 负号和数学符号正确渲染
- ✅ 图表质量保持高清

## 如果问题仍然存在

1. **清除matplotlib缓存**：
```python
import matplotlib
matplotlib.font_manager._rebuild()
```

2. **使用其他字体**：
```python
plt.rcParams['font.family'] = 'serif'  # 或 'monospace'
```

3. **完全禁用Unicode**：
```python
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
```

## 注意事项

- 字体配置需要在导入matplotlib后、绘图前设置
- 不同系统的字体名称可能不同
- 建议在脚本开头统一配置字体设置 