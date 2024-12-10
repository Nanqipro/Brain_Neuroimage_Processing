import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import scipy.signal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap


# ========== 0. 配置 Matplotlib 字体 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 数据读取和清理 ==========
# 读取钙离子浓度数据
calcium_file_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\Tensorflow 2  小鼠神经元\数据集合\calcium_data Day6.xlsx'  # 替换为钙离子浓度数据文件路径
calcium_df = pd.read_excel(calcium_file_path)

# 读取行为标签数据
behavior_file_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\Tensorflow 2  小鼠神经元\数据集合\Day6 behavior.xlsx'
behavior_df = pd.read_excel(behavior_file_path, sheet_name='Sheet1')

# 清理行为标签数据
cleaned_behavior_data = behavior_df.dropna(subset=['Behavior'])

# ========== 2. 时间格式和归一化 ==========
# 确保时间列一致
calcium_df['Time'] = calcium_df['Time'].astype(int)
cleaned_behavior_data['Time'] = cleaned_behavior_data['Time'].astype(int)

# 提取钙离子浓度数据
time_values = calcium_df['Time'].values
calcium_values = calcium_df.iloc[:, 1:].values

# 钙离子浓度归一化
scaler = MinMaxScaler()
calcium_normalized = scaler.fit_transform(calcium_values)

# 将归一化后的数据保存为 DataFrame
calcium_normalized_df = pd.DataFrame(calcium_normalized, columns=calcium_df.columns[1:])
calcium_normalized_df.insert(0, 'Time', time_values)

# ========== 3. 滑动窗口处理 ==========
# 定义窗口大小和步长
window_size = 50
step_size = 10

# 滑动窗口提取
windows = []
window_times = []

calcium_normalized_values = calcium_normalized_df.iloc[:, 1:].values  # 排除时间列

for start in range(0, len(calcium_normalized_values) - window_size + 1, step_size):
    end = start + window_size
    window_data = calcium_normalized_values[start:end, :]
    window_time = time_values[start:end]
    windows.append(window_data)
    window_times.append(window_time)

# 转换为 NumPy 数组
windows = np.array(windows)
window_times = np.array(window_times)

# ========== 4. 匹配行为标签 ==========
# 定义匹配函数
labels_time = cleaned_behavior_data['Time'].values
labels_value = cleaned_behavior_data['Behavior'].values

def get_label_for_window(window_time):
    middle_time = window_time[len(window_time) // 2]
    time_diff = np.abs(labels_time - middle_time)
    min_index = np.argmin(time_diff)
    return labels_value[min_index]

# 为每个窗口匹配行为标签
window_labels = [get_label_for_window(window_time) for window_time in window_times]
window_labels = np.array(window_labels)

# ========== 5. 标签编码 ==========
# 使用 LabelEncoder 对行为标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(window_labels)

# 独热编码
num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded, num_classes)

# ========== 6. 特征提取与加权 ==========
# 定义特征及其权重
feature_weights = {
    'amplitude': 0.40,
    'peak': 0.10,
    'latency': 0.15,
    'frequency': 0.25,
    'decay_time': 0.05,
    'rise_time': 0.05
}


# 定义特征提取函数
def compute_amplitude(window):
    return np.max(window, axis=0) - np.min(window, axis=0)


def compute_peak(window):
    return np.max(window, axis=0)


def compute_latency(window, time_array):
    peak_indices = np.argmax(window, axis=0)
    latency = time_array[peak_indices] - time_array[0]
    return latency


def compute_frequency(window, time_array, threshold=0.5):
    frequency = []
    for neuron_data in window.T:
        peaks, _ = scipy.signal.find_peaks(neuron_data, height=threshold * np.max(neuron_data))
        frequency.append(len(peaks))
    return np.array(frequency)


def compute_decay_time(window, time_array, half_max=True):
    decay_times = []
    for neuron_data in window.T:
        peak = np.max(neuron_data)
        peak_index = np.argmax(neuron_data)
        if half_max:
            half_max = peak / 2
            decay_indices = np.where(neuron_data[peak_index:] <= half_max)[0]
            if len(decay_indices) > 0:
                decay_time = time_array[peak_index + decay_indices[0]] - time_array[peak_index]
            else:
                decay_time = time_array[-1] - time_array[peak_index]
            decay_times.append(decay_time)
        else:
            decay_times.append(0)
    return np.array(decay_times)


def compute_rise_time(window, time_array, half_max=True):
    rise_times = []
    for neuron_data in window.T:
        baseline = np.min(neuron_data)
        peak = np.max(neuron_data)
        if half_max:
            half_max = baseline + (peak - baseline) / 2
            rise_indices = np.where(neuron_data >= half_max)[0]
            if len(rise_indices) > 0:
                rise_time = time_array[rise_indices[0]] - time_array[0]
            else:
                rise_time = time_array[-1] - time_array[0]
            rise_times.append(rise_time)
        else:
            rise_times.append(0)
    return np.array(rise_times)


# 定义窗口对应的时间数组（假设时间间隔均匀）
time_step = (window_times[0][-1] - window_times[0][0]) / window_size
time_array = np.linspace(0, window_size * time_step, window_size)

# 初始化特征列表
X_features = []

for i in range(len(windows)):
    window = windows[i]  # 当前窗口的神经元数据，形状 (window_size, num_neurons)

    # 计算各特征
    amplitude = compute_amplitude(window)  # (num_neurons,)
    peak = compute_peak(window)  # (num_neurons,)
    latency = compute_latency(window, time_array)  # (num_neurons,)
    frequency = compute_frequency(window, time_array)  # (num_neurons,)
    decay_time = compute_decay_time(window, time_array)  # (num_neurons,)
    rise_time = compute_rise_time(window, time_array)  # (num_neurons,)

    # 聚合特征（例如取平均值）
    amplitude_mean = np.mean(amplitude)
    peak_mean = np.mean(peak)
    latency_mean = np.mean(latency)
    frequency_mean = np.mean(frequency)
    decay_time_mean = np.mean(decay_time)
    rise_time_mean = np.mean(rise_time)

    # 应用权重
    weighted_amplitude = feature_weights['amplitude'] * amplitude_mean
    weighted_peak = feature_weights['peak'] * peak_mean
    weighted_latency = feature_weights['latency'] * latency_mean
    weighted_frequency = feature_weights['frequency'] * frequency_mean
    weighted_decay_time = feature_weights['decay_time'] * decay_time_mean
    weighted_rise_time = feature_weights['rise_time'] * rise_time_mean

    # 组合加权特征
    window_feature = [
        weighted_amplitude,
        weighted_peak,
        weighted_latency,
        weighted_frequency,
        weighted_decay_time,
        weighted_rise_time
    ]

    X_features.append(window_feature)

# 转换为 NumPy 数组
X_features = np.array(X_features)

print("特征矩阵 X_features 的形状：", X_features.shape)

# 使用之前的 y_categorical 作为标签
y = y_categorical

print("独热编码后的标签 Y 的形状：", y.shape)

# ========== 7. 划分训练集和测试集 ==========
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

print("训练集特征形状：", X_train.shape)
print("测试集特征形状：", X_test.shape)
print("训练集标签形状：", y_train.shape)
print("测试集标签形状：", y_test.shape)

# ========== 8. 构建神经网络模型 ==========
# 定义模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 查看模型结构
model.summary()

# ========== 9. 模型训练 ==========
# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
)

# # ========== 10. 模型评估 ==========
# # 绘制准确率曲线
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['accuracy'], label='训练准确率')
# plt.plot(history.history['val_accuracy'], label='验证准确率')
# plt.title('模型准确率')
# plt.xlabel('轮数')
# plt.ylabel('准确率')
# plt.legend()
# plt.show()
#
# # 绘制损失曲线
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['loss'], label='训练损失')
# plt.plot(history.history['val_loss'], label='验证损失')
# plt.title('模型损失')
# plt.xlabel('轮数')
# plt.ylabel('损失')
# plt.legend()
# plt.show()
#
# # 在测试集上评估模型
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print('测试集准确率：{:.2f}%'.format(test_accuracy * 100))
#
# # ========== 11. 模型解释与特征重要性分析 ==========
# # 创建一个背景样本
# background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
#
# # 创建 DeepExplainer
# explainer = shap.DeepExplainer(model, background)
#
# # 计算 SHAP 值
# shap_values = explainer.shap_values(X_test[:10])
#
# # 绘制特征重要性
# shap.summary_plot(shap_values, X_test[:10], feature_names=list(feature_weights.keys()))

# ========== 10. 模型评估 ==========
# 绘制准确率曲线
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('轮数')
plt.ylabel('准确率')
plt.legend()
plt.show()

# 绘制损失曲线
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('轮数')
plt.ylabel('损失')
plt.legend()
plt.show()

# 在测试集上评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('测试集准确率：{:.2f}%'.format(test_accuracy * 100))

# ========== 11. 模型解释与特征重要性分析 ==========
import shap

# 创建一个背景样本
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# 创建 DeepExplainer
explainer = shap.DeepExplainer(model, background)

# 计算 SHAP 值
shap_values = explainer.shap_values(X_test[:10])

# 检查 shap_values 的结构
print(f"Type of shap_values: {type(shap_values)}")
print(f"Number of classes: {len(shap_values)}")
print(f"Shape of shap_values for first class: {shap_values[0].shape}")

# 选择一个类别的 SHAP 值进行绘图，例如第一个类别
shap.summary_plot(shap_values[0], X_test[:10], feature_names=list(feature_weights.keys()))

# 或者循环绘制所有类别的 SHAP summary plot
for i, class_shap in enumerate(shap_values):
    shap.summary_plot(class_shap, X_test[:10], feature_names=list(feature_weights.keys()))
    plt.title(f'SHAP Summary for Class: {label_encoder.classes_[i]}')
    plt.show()