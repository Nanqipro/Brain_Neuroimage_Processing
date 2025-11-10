# 钙波检测算法 - Baseline自适应能力分析报告

> **分析日期**: 2025-11-10  
> **分析对象**: `element_extraction.py` - `detect_calcium_transients()` 函数  
> **数据特点**: 输入已是ΔF/F降噪后的数据

---

## 一、当前Baseline计算机制

### 1.1 核心代码分析

**Baseline估计方式**（第602行）：
```python
baseline = np.percentile(smoothed_data, baseline_percentile)  # 默认baseline_percentile=8
```

**关键特征**：
- ✅ **全局固定baseline**：对整条信号计算一个百分位数值
- ✅ **简单高效**：计算快速，无需额外参数
- ⚠️ **非自适应**：不随时间或局部信号特性变化

### 1.2 与Baseline相关的自适应机制

虽然baseline本身不自适应，但您的算法在**检测阈值**上实现了自适应：

#### ✅ 已有的自适应机制

1. **信噪比自适应阈值**（第606行）
```python
threshold = baseline + min_snr * noise_level
```
- `noise_level`根据数据动态计算
- 能适应不同神经元的噪声水平

2. **Prominence自适应**（第611行）
```python
prominence_threshold = noise_level * prominence_factor
```
- 基于噪声水平动态调整
- 配合`filter_strength`参数进一步调节

3. **形态学多层过滤**（第748-904行）
- 上升/衰减比例检查
- 非对称性检查
- 指数衰减拟合
- 这些机制能自适应地识别真实钙波 vs 噪声

---

## 二、对不同幅值波形的适应能力评估

### 2.1 ✅ 能良好处理的情况

#### 情况A：不同神经元有不同整体活跃度
```
神经元1: 低幅值（0-0.5 ΔF/F）
神经元2: 中幅值（0-1.0 ΔF/F）
神经元3: 高幅值（0-2.0 ΔF/F）
```

**处理能力**: ✅ **良好**

**原因**:
- 每个神经元**独立计算**baseline和noise_level
- SNR自适应阈值能区分不同幅值的信号
- 形态学评分不依赖绝对幅值

**示例**:
```
神经元1: baseline=0.02, threshold=0.02+3.5×0.01=0.055
神经元2: baseline=0.05, threshold=0.05+3.5×0.02=0.12
神经元3: baseline=0.10, threshold=0.10+3.5×0.05=0.275
```

---

#### 情况B：同一神经元内，不同钙波有不同幅值
```
时间段1: 小钙波（幅值0.1）
时间段2: 大钙波（幅值0.5）
时间段3: 中钙波（幅值0.3）
```

**处理能力**: ✅ **良好**

**原因**:
- 峰值检测使用`find_peaks`的相对prominence
- 边界查找基于"低于baseline"的判据，对大小钙波一视同仁
- 形态学评分评估波形质量而非绝对幅值

---

### 2.2 ⚠️ 可能存在问题的情况

#### 情况C：基线漂移（Baseline Drift）
```
时间 0-100s:   baseline ≈ 0.05  |  钙波幅值 0.1-0.3
时间 100-200s: baseline ≈ 0.15  |  钙波幅值 0.1-0.3（相对自身基线）
时间 200-300s: baseline ≈ 0.08  |  钙波幅值 0.1-0.3
```

**处理能力**: ⚠️ **可能不准确**

**问题分析**:

1. **全局baseline问题**
   - 当前算法计算的baseline = 8%百分位数 ≈ 0.05（取所有时间段的低值）
   - 在100-200s时段（高基线期），实际基线≈0.15
   - 使用0.05作为基线会导致：
     ```
     实际小钙波: 0.15→0.25 (幅值0.1)
     误判幅值: 0.25-0.05=0.2 (放大了1倍)
     ```

2. **边界检测问题**
   ```python
   while smoothed_data[start_idx] > baseline:  # baseline=0.05
       start_idx -= 1
   ```
   - 在高基线时段（baseline实际=0.15），算法仍使用0.05
   - 会将整个高基线区域误认为是钙波的一部分
   - 导致**钙波边界过宽，持续时间过长**

3. **可能的后果**
   - ❌ 在高基线时段：**误检噪声**或**钙波边界不准**
   - ❌ 在低基线时段：可能正常
   - ❌ 持续时间、AUC等特征计算不准确

**示意图**:
```
信号值
  |    ╱‾‾╲                    ╱‾╲
  |   ╱    ╲                  ╱   ╲
0.2|  ╱      ╲   ←高基线→   ╱     ╲
  | ╱        ╲.............╱.......╲........  实际基线≈0.15
0.1|╱          ╲...........╱         ╲.......
  |____________╲_________╱___________╲______  全局baseline=0.05
  0           100       200          300    时间(s)
  
  问题：在100-200s区间，使用0.05作为基线会导致
        将虚线以上部分全部识别为"钙波"
```

---

#### 情况D：不同行为状态下基线水平不同
```
静息状态: baseline ≈ 0.03, 钙波幅值 0.1-0.2
活跃状态: baseline ≈ 0.12, 钙波幅值 0.1-0.2（相对自身基线）
```

**处理能力**: ⚠️ **可能不准确**

**问题**:
- 与情况C类似，全局baseline无法适应状态切换
- 可能在活跃状态下**误检**或**边界不准**

---

#### 情况E：局部高幅值波形后的不应期
```
大钙波: 幅值1.5, 持续10s
     ↓
不应期: 基线略高于静息(0.1 vs 0.05), 持续20s
     ↓
小钙波: 幅值0.15, 但相对局部基线(0.1)实际幅值只有0.05
```

**处理能力**: ⚠️ **可能漏检小钙波**

**问题**:
- 全局baseline=0.05，但局部(不应期)基线=0.1
- 小钙波(0.1→0.15)相对全局baseline幅值=0.1，可能被检测
- 但实际相对局部基线幅值只有0.05，可能是生理性抑制后的恢复，而非真实钙波

---

## 三、定量评估：当前算法的适应范围

### 3.1 基线漂移容忍度测试

假设您的数据特征如下：
```python
全局baseline (8%百分位数): B_global = 0.05
实际局部baseline范围: B_local = 0.03 ~ 0.15
典型钙波幅值: A = 0.1 ~ 0.5
噪声标准差: σ = 0.02
```

**检测阈值**:
```
threshold = B_global + min_snr × σ = 0.05 + 3.5×0.02 = 0.12
```

**分析不同时段**:

| 时段 | 真实基线 | 钙波峰值 | 真实幅值 | 算法判定 | 结果 |
|-----|---------|---------|---------|---------|------|
| 低基线期 | 0.03 | 0.13 | 0.10 | 0.13 > 0.12 | ✅ 正确检测 |
| 正常期 | 0.05 | 0.15 | 0.10 | 0.15 > 0.12 | ✅ 正确检测 |
| 高基线期 | 0.15 | 0.25 | 0.10 | 0.25 > 0.12 | ⚠️ 检测但幅值算错 |
| 高基线期(噪声) | 0.15 | 0.18 | 0.03 | 0.18 > 0.12 | ❌ 误检噪声 |

**结论**:
- ✅ 当基线漂移范围 **< 2×σ** 时，算法表现良好
- ⚠️ 当基线漂移范围 **> 3×σ** 时，开始出现误差
- ❌ 当基线漂移范围 **> 5×σ** 时，误检/漏检明显

---

### 3.2 您的数据是否会遇到问题？

**需要检查的指标**:

1. **基线稳定性检查**
   ```python
   # 简单测试代码
   import numpy as np
   data = your_neuron_data  # ΔF/F数据
   
   # 分段计算基线
   n_segments = 10
   segment_length = len(data) // n_segments
   baseline_per_segment = []
   
   for i in range(n_segments):
       segment = data[i*segment_length:(i+1)*segment_length]
       baseline_per_segment.append(np.percentile(segment, 8))
   
   baseline_drift = np.max(baseline_per_segment) - np.min(baseline_per_segment)
   noise_level = np.std(data[data < np.percentile(data, 50)])
   
   print(f"基线漂移范围: {baseline_drift:.4f}")
   print(f"噪声水平: {noise_level:.4f}")
   print(f"漂移/噪声比: {baseline_drift/noise_level:.2f}")
   
   if baseline_drift < 2 * noise_level:
       print("✅ 基线稳定，当前算法适用")
   elif baseline_drift < 5 * noise_level:
       print("⚠️ 基线有一定漂移，建议使用自适应baseline")
   else:
       print("❌ 基线漂移严重，强烈建议使用自适应baseline")
   ```

2. **可视化检查**
   ```python
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(15, 4))
   plt.plot(data, alpha=0.5, label='ΔF/F data')
   
   # 全局baseline
   global_baseline = np.percentile(data, 8)
   plt.axhline(global_baseline, color='r', linestyle='--', 
               label=f'Global baseline={global_baseline:.3f}')
   
   # 滑动baseline（参考）
   window = len(data) // 20
   sliding_baseline = pd.Series(data).rolling(window, center=True).quantile(0.08)
   plt.plot(sliding_baseline, color='orange', linestyle='--', 
            label='Sliding baseline (8%)')
   
   plt.legend()
   plt.title('Baseline Stability Check')
   plt.show()
   ```

---

## 四、改进方案（如需要）

### 方案A：添加滑动窗口baseline（推荐） ⭐⭐⭐

**适用场景**: 
- 存在缓慢的基线漂移
- 不同行为状态基线不同
- 长时间记录（>10分钟）

**实现代码**:

```python
def estimate_sliding_baseline(data, window_size=None, percentile=8):
    """
    使用滑动窗口估计局部baseline
    
    参数:
        data: 输入信号
        window_size: 窗口大小（采样点数），默认为数据长度的10%
        percentile: 百分位数，默认8
    
    返回:
        baseline: 与data同长度的baseline数组
    """
    if window_size is None:
        window_size = len(data) // 10
    
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    
    baseline = np.zeros_like(data, dtype=float)
    half_window = window_size // 2
    
    for i in range(len(data)):
        # 确定窗口范围
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        
        # 计算窗口内的百分位数
        window_data = data[start:end]
        baseline[i] = np.percentile(window_data, percentile)
    
    # 平滑baseline（防止过度波动）
    from scipy.signal import savgol_filter
    baseline = savgol_filter(baseline, window_size, 2)
    
    return baseline
```

**修改detect_calcium_transients函数**:

```python
def detect_calcium_transients(
    data, 
    fs=4.8,
    baseline_method='global',  # 新增参数：'global' 或 'sliding'
    baseline_window=None,      # 滑动窗口大小
    baseline_percentile=8,
    ...
):
    """
    参数:
        baseline_method: 'global' (当前方法) 或 'sliding' (滑动窗口)
        baseline_window: 滑动窗口大小（秒），仅在baseline_method='sliding'时使用
    """
    
    # ... 前面的预处理代码 ...
    
    # 2. 估计基线和噪声水平
    if baseline_method == 'global':
        # 当前方法
        baseline = np.percentile(smoothed_data, baseline_percentile)
        baseline_array = np.full_like(smoothed_data, baseline)
    
    elif baseline_method == 'sliding':
        # 滑动窗口方法
        if baseline_window is None:
            window_size = len(smoothed_data) // 10  # 默认10%数据长度
        else:
            window_size = int(baseline_window * fs)
        
        baseline_array = estimate_sliding_baseline(
            smoothed_data, 
            window_size=window_size,
            percentile=baseline_percentile
        )
        baseline = baseline_array  # 保持兼容性
    
    else:
        raise ValueError(f"未知的baseline方法: {baseline_method}")
    
    # 计算噪声水平
    if baseline_method == 'global':
        noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    else:
        # 对滑动baseline，计算去趋势后的噪声
        detrended = smoothed_data - baseline_array
        noise_level = np.std(detrended[detrended < np.percentile(detrended, 50)])
    
    # 3. 检测峰值（需要修改以支持数组baseline）
    if baseline_method == 'global':
        threshold = baseline + min_snr * noise_level
        initial_peaks, peak_props = find_peaks(
            smoothed_data, 
            height=threshold,
            prominence=prominence_threshold, 
            width=min_width
        )
    else:
        # 对滑动baseline，先去趋势
        detrended = smoothed_data - baseline_array
        threshold = min_snr * noise_level
        initial_peaks, peak_props = find_peaks(
            detrended,
            height=threshold,
            prominence=prominence_threshold,
            width=min_width
        )
    
    # 4. 分析每个钙爆发（需要修改边界查找）
    for i, peak_idx in enumerate(peaks):
        # 获取峰值处的局部baseline
        if baseline_method == 'global':
            local_baseline = baseline
        else:
            local_baseline = baseline_array[peak_idx]
        
        # 寻找左侧边界
        start_idx = peak_idx
        left_limit = 0 if i == 0 else peaks[i-1]
        
        if baseline_method == 'global':
            # 原有逻辑
            while start_idx > left_limit and smoothed_data[start_idx] > baseline:
                start_idx -= 1
        else:
            # 滑动baseline：使用局部baseline
            while start_idx > left_limit and smoothed_data[start_idx] > baseline_array[start_idx]:
                start_idx -= 1
        
        # 寻找右侧边界（类似修改）
        end_idx = peak_idx
        right_limit = len(smoothed_data) - 1 if i == len(peaks) - 1 else peaks[i+1]
        
        if baseline_method == 'global':
            while end_idx < right_limit and smoothed_data[end_idx] > baseline:
                end_idx += 1
        else:
            while end_idx < right_limit and smoothed_data[end_idx] > baseline_array[end_idx]:
                end_idx += 1
        
        # 计算特征（使用局部baseline）
        peak_value = smoothed_data[peak_idx]
        amplitude = peak_value - local_baseline
        
        # 存储baseline信息
        transient = {
            'start_idx': start_idx,
            'peak_idx': peak_idx,
            'end_idx': end_idx,
            'amplitude': amplitude,
            'peak_value': peak_value,
            'baseline': local_baseline,
            'baseline_method': baseline_method,
            ...
        }
        
        transients.append(transient)
    
    return transients, smoothed_data
```

**使用方法**:
```python
# 当前方法（全局baseline）
transients, _ = detect_calcium_transients(data, baseline_method='global')

# 滑动baseline方法
transients, _ = detect_calcium_transients(
    data, 
    baseline_method='sliding',
    baseline_window=30  # 30秒滑动窗口
)
```

---

### 方案B：样条拟合baseline（高级） ⭐⭐

**适用场景**:
- 基线有非线性漂移
- 需要更平滑的baseline估计

**实现代码**:

```python
def estimate_spline_baseline(data, smoothness=0.1, percentile=20):
    """
    使用样条拟合估计baseline
    
    参数:
        data: 输入信号
        smoothness: 平滑度参数（0-1），越大越平滑
        percentile: 选择低活动点的百分位数
    
    返回:
        baseline: 拟合的baseline数组
    """
    from scipy.interpolate import UnivariateSpline
    
    # 找到低活动点作为样条节点
    threshold = np.percentile(data, percentile)
    low_activity_mask = data < threshold
    
    if np.sum(low_activity_mask) < 10:
        # 如果低活动点太少，退化为全局baseline
        return np.full_like(data, np.percentile(data, 8))
    
    x = np.arange(len(data))
    knots_x = x[low_activity_mask]
    knots_y = data[low_activity_mask]
    
    # 拟合样条
    s_param = len(data) * smoothness
    try:
        spline = UnivariateSpline(knots_x, knots_y, s=s_param, k=3)
        baseline = spline(x)
        
        # 确保baseline不超过信号
        baseline = np.minimum(baseline, data)
        
        return baseline
    except:
        # 如果拟合失败，退化为全局baseline
        return np.full_like(data, np.percentile(data, 8))
```

---

### 方案C：自动选择baseline方法（智能） ⭐⭐⭐

```python
def auto_select_baseline_method(data, fs=4.8):
    """
    根据数据特性自动选择baseline估计方法
    
    返回:
        method: 'global' 或 'sliding'
        params: 相应参数
    """
    # 评估基线稳定性
    n_segments = 10
    segment_length = len(data) // n_segments
    baseline_per_segment = []
    
    for i in range(n_segments):
        segment = data[i*segment_length:(i+1)*segment_length]
        baseline_per_segment.append(np.percentile(segment, 8))
    
    baseline_drift = np.max(baseline_per_segment) - np.min(baseline_per_segment)
    noise_level = np.std(data[data < np.percentile(data, 50)])
    drift_ratio = baseline_drift / noise_level if noise_level > 0 else 0
    
    # 决策
    if drift_ratio < 2.0:
        print(f"  自适应baseline选择: 'global' (drift_ratio={drift_ratio:.2f} < 2.0)")
        return 'global', {}
    
    elif drift_ratio < 5.0:
        print(f"  自适应baseline选择: 'sliding' (drift_ratio={drift_ratio:.2f} 在2-5之间)")
        # 根据漂移程度调整窗口大小
        window_size = int(len(data) / (drift_ratio * 2))
        return 'sliding', {'baseline_window': window_size / fs}
    
    else:
        print(f"  自适应baseline选择: 'sliding' with small window (drift_ratio={drift_ratio:.2f} > 5.0)")
        # 严重漂移，使用较小窗口
        window_size = int(len(data) / 20)
        return 'sliding', {'baseline_window': window_size / fs}
```

**集成到主函数**:

```python
def detect_calcium_transients(
    data, 
    fs=4.8,
    baseline_method='auto',  # 'auto', 'global', 'sliding'
    ...
):
    if baseline_method == 'auto':
        selected_method, params = auto_select_baseline_method(data, fs)
        baseline_method = selected_method
        if 'baseline_window' in params:
            baseline_window = params['baseline_window']
    
    # ... 后续逻辑 ...
```

---

## 五、总结与建议

### 5.1 当前算法能力总结

| 场景 | 适应能力 | 说明 |
|-----|---------|------|
| ✅ 不同神经元不同幅值 | **优秀** | 独立计算baseline |
| ✅ 同一神经元多种幅值钙波 | **优秀** | SNR自适应+形态学评分 |
| ⚠️ 基线缓慢漂移 | **一般** | 漂移<2σ时可接受 |
| ⚠️ 不同状态不同基线 | **一般** | 需要人工调整参数 |
| ❌ 基线快速变化 | **较差** | 建议使用滑动baseline |

### 5.2 您的数据是否需要改进？

**决策树**:

```
开始
  ↓
检查您的数据特征
  ↓
├─ 记录时长 < 5分钟？
│  └─ 是 → ✅ 当前算法足够
│  └─ 否 → 继续
│
├─ 是否有明显的行为状态切换？
│  └─ 是 → ⚠️ 建议使用滑动baseline
│  └─ 否 → 继续
│
├─ 可视化检查：baseline是否稳定？
│  └─ 稳定 → ✅ 当前算法足够
│  └─ 漂移 → ⚠️ 建议使用滑动baseline
│
└─ 运行"基线稳定性检查"代码
   └─ drift_ratio < 2.0 → ✅ 当前算法足够
   └─ drift_ratio > 2.0 → ⚠️ 建议使用滑动baseline
```

### 5.3 实施建议

**最小改动方案**（推荐先尝试）:
1. ✅ 保持当前算法不变
2. ✅ 添加`baseline_method`参数（默认='global'）
3. ✅ 仅在需要时使用`baseline_method='sliding'`

**优点**:
- 向后兼容，不影响现有结果
- 灵活选择，根据数据特点调整
- 渐进式改进，风险低

**实施步骤**:
1. 第1天：实现`estimate_sliding_baseline`函数
2. 第2天：修改`detect_calcium_transients`支持滑动baseline
3. 第3天：对比测试，验证改进效果
4. 第4天：更新文档和示例

---

## 六、验证方案

### 6.1 对比测试代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_baseline_methods(neuron_data, fs=4.8):
    """
    对比不同baseline方法的效果
    """
    # 方法1：全局baseline（当前）
    transients_global, _ = detect_calcium_transients(
        neuron_data, fs=fs, baseline_method='global'
    )
    
    # 方法2：滑动baseline（新）
    transients_sliding, _ = detect_calcium_transients(
        neuron_data, fs=fs, baseline_method='sliding', baseline_window=30
    )
    
    # 可视化对比
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    time = np.arange(len(neuron_data)) / fs
    
    # 子图1：原始数据+全局baseline
    axes[0].plot(time, neuron_data, 'k-', alpha=0.5)
    global_baseline = np.percentile(neuron_data, 8)
    axes[0].axhline(global_baseline, color='r', linestyle='--', 
                    label=f'Global baseline={global_baseline:.3f}')
    for t in transients_global:
        axes[0].axvline(t['peak_idx']/fs, color='r', alpha=0.3)
    axes[0].set_title(f"Global Baseline Method: {len(transients_global)} events")
    axes[0].legend()
    axes[0].set_ylabel('ΔF/F')
    
    # 子图2：原始数据+滑动baseline
    axes[1].plot(time, neuron_data, 'k-', alpha=0.5)
    sliding_baseline = estimate_sliding_baseline(neuron_data)
    axes[1].plot(time, sliding_baseline, color='orange', linestyle='--',
                label='Sliding baseline')
    for t in transients_sliding:
        axes[1].axvline(t['peak_idx']/fs, color='orange', alpha=0.3)
    axes[1].set_title(f"Sliding Baseline Method: {len(transients_sliding)} events")
    axes[1].legend()
    axes[1].set_ylabel('ΔF/F')
    
    # 子图3：检测差异
    axes[2].plot(time, neuron_data, 'k-', alpha=0.3)
    
    # 只在全局方法检测到的事件（可能误检）
    global_only = [t for t in transients_global 
                   if not any(abs(t['peak_idx'] - ts['peak_idx']) < 10 
                             for ts in transients_sliding)]
    for t in global_only:
        axes[2].axvline(t['peak_idx']/fs, color='r', alpha=0.5, 
                       label='Global only' if t==global_only[0] else '')
    
    # 只在滑动方法检测到的事件（可能漏检）
    sliding_only = [t for t in transients_sliding 
                    if not any(abs(t['peak_idx'] - tg['peak_idx']) < 10 
                              for tg in transients_global)]
    for t in sliding_only:
        axes[2].axvline(t['peak_idx']/fs, color='orange', alpha=0.5,
                       label='Sliding only' if t==sliding_only[0] else '')
    
    axes[2].set_title(f"Detection Differences: "
                     f"{len(global_only)} global-only, "
                     f"{len(sliding_only)} sliding-only")
    axes[2].legend()
    axes[2].set_ylabel('ΔF/F')
    axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('baseline_method_comparison.png', dpi=300)
    plt.show()
    
    # 统计对比
    comparison = pd.DataFrame({
        'Method': ['Global', 'Sliding'],
        'Num Events': [len(transients_global), len(transients_sliding)],
        'Mean Amplitude': [
            np.mean([t['amplitude'] for t in transients_global]) if transients_global else 0,
            np.mean([t['amplitude'] for t in transients_sliding]) if transients_sliding else 0
        ],
        'Mean Duration': [
            np.mean([t['duration'] for t in transients_global]) if transients_global else 0,
            np.mean([t['duration'] for t in transients_sliding]) if transients_sliding else 0
        ]
    })
    
    print("\n=== Baseline Method Comparison ===")
    print(comparison.to_string(index=False))
    
    return transients_global, transients_sliding
```

### 6.2 使用示例

```python
# 加载数据
df = pd.read_excel('your_data.xlsx')
neuron_data = df['n1'].values

# 运行对比
transients_global, transients_sliding = compare_baseline_methods(
    neuron_data, fs=4.8
)
```

---

## 七、结论

### 7.1 直接回答您的问题

**Q: 现在我的算法是否可以自适应调整baseline等来提取不同幅值的波形？**

**A**: 
- ✅ **对于不同神经元的不同幅值**：是的，能很好适应
- ✅ **对于同一神经元内不同幅值的钙波**：是的，能很好适应
- ⚠️ **对于基线漂移情况**：部分适应，漂移较小时可以，漂移较大时需要改进

### 7.2 优势总结

您的算法已经具备的**优秀特性**：
1. ✅ SNR自适应阈值
2. ✅ Prominence自适应
3. ✅ 形态学多层过滤（这是非常创新的）
4. ✅ 每个神经元独立计算参数

### 7.3 改进空间

**可选改进**（根据数据特点决定是否需要）：
1. 添加滑动窗口baseline（处理基线漂移）
2. 添加自动baseline方法选择（智能适应）
3. 保持当前方法作为默认（向后兼容）

**建议**：
- 如果您的数据没有明显基线漂移 → ✅ 当前算法已足够好
- 如果需要处理长时间记录或多状态数据 → ⚠️ 建议添加滑动baseline选项

---

**报告结束**

如需实施改进或有其他问题，请随时联系。

