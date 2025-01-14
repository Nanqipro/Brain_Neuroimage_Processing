# 导入必要的库
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置Matplotlib以支持中文字符
# 选择系统中已安装的中文字体，例如 'SimHei' 或 'Microsoft YaHei'
# 您可以根据实际安装的字体名称进行修改

# 方法一：全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 方法二：在绘图时单独指定字体
# 例如，在plt.title()中使用 fontproperties 参数

# 示例：
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)
# plt.title('标题', fontproperties=font)

# 其余代码保持不变
# 1. 读取Excel数据
neuron_activity_df = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day3\Day3.xlsx', sheet_name='Sheet1')
behavior_label_df = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day3\Day3 behavior.xlsx', sheet_name='Sheet1')


# 2. 数据预处理和对齐
neuron_activity_df.rename(columns=lambda x: x.strip(), inplace=True)
behavior_label_df.rename(columns=lambda x: x.strip(), inplace=True)

neuron_activity_df['Time'] = pd.to_numeric(neuron_activity_df['Time'], errors='coerce')
behavior_label_df['Time'] = pd.to_numeric(behavior_label_df['Time'], errors='coerce')

neuron_activity_df.dropna(subset=['Time'], inplace=True)
behavior_label_df.dropna(subset=['Time'], inplace=True)

merged_df = pd.merge(neuron_activity_df, behavior_label_df, on='Time', how='inner')

print(f"合并后的数据行数: {merged_df.shape[0]}")
print(merged_df.head())

# 3. 数据重塑
neuron_columns = [col for col in merged_df.columns if col not in ['Time', 'Behavior']]
long_df = merged_df.melt(id_vars=['Time', 'Behavior'], value_vars=neuron_columns,
                         var_name='NeuronID', value_name='Activity')

print(f"长格式数据行数: {long_df.shape[0]}")
print(long_df.head())

# 4. 统计检验
neurons = long_df['NeuronID'].unique()
behavior_states = long_df['Behavior'].unique()
num_states = len(behavior_states)

results = []


def cohens_d(x, y):
    """计算Cohen's d效应大小"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std != 0 else np.nan


def eta_squared(f_stat, df_between, df_within):
    """计算Eta-squared效应大小"""
    return f_stat * df_between / (f_stat * df_between + df_within)


for neuron in neurons:
    neuron_data = long_df[long_df['NeuronID'] == neuron]
    groups = [neuron_data[neuron_data['Behavior'] == state]['Activity'].dropna().values for state in behavior_states]

    if any(len(group) == 0 for group in groups):
        print(f"神经元 {neuron} 的某些行为状态下没有数据，跳过统计检验。")
        continue

    if num_states == 2:
        group1, group2 = groups
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)  # Welch’s t-test
        effect_size = cohens_d(group1, group2)
        test = 't-test'
        statistic = t_stat
    elif num_states > 2:
        try:
            f_stat, p_val = f_oneway(*groups)
            df_between = num_states - 1
            df_within = len(neuron_data) - num_states
            effect_size = eta_squared(f_stat, df_between, df_within)
            test = 'ANOVA'
            statistic = f_stat
        except Exception as e:
            print(f"神经元 {neuron} 的ANOVA检验出错: {e}")
            continue
    else:
        print(f"行为状态数量不足以进行统计检验。")
        continue

    results.append({
        'NeuronID': neuron,
        'Test': test,
        'Statistic': statistic,
        'p-value': p_val,
        'EffectSize': effect_size
    })

# 5. 多重比较校正
results_df = pd.DataFrame(results)
results_df.dropna(subset=['p-value'], inplace=True)
results_df['p-adjusted'] = multipletests(results_df['p-value'], method='fdr_bh')[1]
alpha = 0.05
results_df['Significant'] = results_df['p-adjusted'] < alpha

# 6. 结果展示与保存
significant_neurons = results_df[results_df['Significant']]
print(f"显著的神经元数量: {significant_neurons.shape[0]} / {len(neurons)}")
print(significant_neurons)

results_df.to_csv('neuron_statistical_results.csv', index=False)
significant_neurons.to_csv('significant_neurons.csv', index=False)

# 7. 可视化

# 7.1. 调整后的p值分布
plt.figure(figsize=(10, 6))
sns.histplot(results_df['p-adjusted'], bins=50, kde=True)
plt.axvline(alpha, color='red', linestyle='--', label=f'alpha = {alpha}')
plt.xlabel('调整后的 p 值')
plt.ylabel('计数')
plt.title('调整后的 p 值分布')
plt.legend()
plt.tight_layout()
plt.show()

# 7.2. 显著神经元的活跃度分布
for idx, row in significant_neurons.iterrows():
    neuron = row['NeuronID']
    neuron_data = long_df[long_df['NeuronID'] == neuron]
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Behavior', y='Activity', data=neuron_data)
    sns.swarmplot(x='Behavior', y='Activity', data=neuron_data, color=".25")
    plt.title(f'神经元 {neuron} 在不同行为状态下的活跃度分布\n调整后的 p 值 = {row["p-adjusted"]:.4e}')
    plt.xlabel('行为状态')
    plt.ylabel('钙离子浓度')
    plt.tight_layout()
    plt.show()
