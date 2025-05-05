import csv
import os
import matplotlib.pyplot as plt

# 定义 CSV 文件路径
csv_file_path = "results.csv"

# 初始化三个字典
was = {}
ras = {}
aas = {}

# 读取 CSV 文件
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
    # 使用 DictReader 读取 CSV 文件
    reader = csv.DictReader(file)

    # 遍历每一行
    for row in reader:
        protocol = row["Protocol"]  # 获取协议名称
        was[protocol] = float(row["Write_Average"])  # 写入平均值
        ras[protocol] = float(row["Read_Average"])  # 读取平均值
        aas[protocol] = float(row["All_Average"])  # 综合平均值

# 输出结果
print("was:", was)
print("ras:", ras)
print("aas:", aas)



#
# Set font for academic papers (remove Chinese font settings)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False  # For minus sign display

# X-axis labels (English version)
x_labels = ['Load', 'Workload A', 'Workload B', 'Workload C']

# Algorithm latency data
tradition_raft = [was['Raft'], aas['Raft'], was['Raft']*0.05 + ras['Raft']*0.95, ras['Raft']]     # Raft
pigpaxos = [was['PigPaxos'], aas['PigPaxos'], was['PigPaxos']*0.05 + ras['PigPaxos']*0.95, ras['PigPaxos']]           # PigPaxos
mencius_raft = [was['Mencius'], aas['Mencius'], was['Mencius']*0.05 + ras['Mencius']*0.95, ras['Mencius']]       # Mencius
fr = [was['FR'], aas['FR'], was['FR']*0.05 + ras['FR']*0.95, ras['FR']]                  # FR
cdraft = [was['CD-Raft'], aas['CD-Raft'], was['CD-Raft']*0.05 + ras['CD-Raft']*0.95, ras['CD-Raft']]               # CD-Raft

print("Traditional Raft:", tradition_raft)
print("PigPaxos:", pigpaxos)
print("Mencius:", mencius_raft)
print("FR:", fr)
print("CD-Raft:", cdraft)

# 计算提升百分比
def calculate_improvement(current, comparison):
    improvement = []
    for i in range(len(current)):
        if comparison[i] == 0:  # 避免除零错误
            improvement.append(0)
        else:
            improvement.append((comparison[i] - current[i]) / comparison[i] * 100)
    return improvement

# CD-Raft 相较于其他算法的提升
cd_vs_tradition = calculate_improvement(cdraft, tradition_raft)
cd_vs_pigpaxos = calculate_improvement(cdraft, pigpaxos)
cd_vs_mencius = calculate_improvement(cdraft, mencius_raft)
cd_vs_fr = calculate_improvement(cdraft, fr)

# FR 相较于其他算法的提升
fr_vs_tradition = calculate_improvement(fr, tradition_raft)
fr_vs_pigpaxos = calculate_improvement(fr, pigpaxos)
fr_vs_mencius = calculate_improvement(fr, mencius_raft)
fr_vs_cdraft = calculate_improvement(fr, cdraft)

# 打印 CD-Raft 的提升
print("\nCD-Raft 相较于其他算法的提升百分比:")
print("vs Traditional Raft:", [f"{x:.2f}%" for x in cd_vs_tradition])
print("vs PigPaxos:", [f"{x:.2f}%" for x in cd_vs_pigpaxos])
print("vs Mencius:", [f"{x:.2f}%" for x in cd_vs_mencius])
print("vs FR:", [f"{x:.2f}%" for x in cd_vs_fr])

# 打印 FR 的提升
print("\nFR 相较于其他算法的提升百分比:")
print("vs Traditional Raft:", [f"{x:.2f}%" for x in fr_vs_tradition])
print("vs PigPaxos:", [f"{x:.2f}%" for x in fr_vs_pigpaxos])
print("vs Mencius:", [f"{x:.2f}%" for x in fr_vs_mencius])
print("vs CD-Raft:", [f"{x:.2f}%" for x in fr_vs_cdraft])


# Bar plot parameters
bar_width = 0.12
x = range(len(x_labels))

# Create figure with professional layout
fig, ax = plt.subplots(figsize=(15, 8))

# Plot bars with academic style
ax.bar([i - 2.5*bar_width for i in x], tradition_raft, width=bar_width,
       label='Raft', color='white', edgecolor='#d62728',
       hatch='///', linewidth=1.5)
ax.bar([i - 1.5*bar_width for i in x], pigpaxos, width=bar_width,
       label='PigPaxos', color='white', edgecolor='#2ca02c',
       hatch='xxx', linewidth=1.5)
ax.bar([i - 0.5*bar_width for i in x], mencius_raft, width=bar_width,
       label='Mencius', color='white', edgecolor='#ff7f0e',
       hatch='...', linewidth=1.5)
ax.bar([i + 0.5*bar_width for i in x], fr, width=bar_width,
       label='FR', color='#aec7e8', edgecolor='#1f77b4', linewidth=1.5)
ax.bar([i + 1.5*bar_width for i in x], cdraft, width=bar_width,
       label='CD-Raft', color='#1f77b4', edgecolor='navy', linewidth=1.5)

# Axis labels with professional formatting
ax.set_xlabel('Workload Type', fontsize=22, labelpad=15)
ax.set_ylabel('Average Latency (ms)', fontsize=22, labelpad=15)
ax.set_xticks(ticks=x)
ax.set_xticklabels(x_labels, fontsize=20)
ax.tick_params(axis='y', labelsize=18)

# Y-axis configuration
plt.ylim(0, 60)
ax.set_yticks(range(0, 60, 10))

# Grid style for academic papers
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Professional legend configuration
legend = ax.legend(loc='upper center',
                  bbox_to_anchor=(0.5, 1.18),
                  ncol=5,
                  fontsize=18,
                  frameon=True,
                  edgecolor='#333333',
                  handletextpad=0.5,
                  columnspacing=1.2,
                  title='Protocols',
                  title_fontsize='19')

# Layout optimization
fig.tight_layout(pad=3)

# Export settings (uncomment to use)
output_path = './results'
if not os.path.exists(output_path):
    os.makedirs(output_path)
plt.savefig(f'{output_path}/all.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white')

plt.show()