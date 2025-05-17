import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from scipy.stats import pearsonr
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from tqdm import tqdm

def load_data(data_path, min_samples=50):
    # 根据文件扩展名决定使用哪种方法读取数据
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        data = pd.read_excel(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}，请使用.csv或.xlsx/.xls格式")
    
    # 自动检测神经元特征列（假设所有神经元列都是以'n'开头后跟数字的格式）
    neuron_cols = [col for col in data.columns if col.startswith('n') and col[1:].isdigit()]
    
    if not neuron_cols:
        raise ValueError("未找到神经元特征列，请确保神经元列名以'n'开头后跟数字")
    
    # 按照编号顺序排序神经元列
    neuron_cols.sort(key=lambda x: int(x[1:]))
    print(f"自动检测到{len(neuron_cols)}个神经元特征列: {neuron_cols[:5]}...等")
    
    # 提取特征和标签
    features = data.loc[:, neuron_cols].values
    
    if 'behavior' not in data.columns:
        raise ValueError("未找到'behavior'标签列，请确保数据包含此列")
    
    labels = data['behavior'].values

    # 统计每个标签的样本数量
    class_counts = Counter(labels)
    print(f"原始类别分布: {class_counts}")
    
    # 过滤掉样本数量少于min_samples的标签
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
    if len(valid_classes) < len(class_counts):
        removed_classes = [cls for cls, count in class_counts.items() if count < min_samples]
        print(f"已移除样本数少于{min_samples}的标签: {removed_classes}")
        
        # 创建过滤后的数据掩码
        valid_mask = np.isin(labels, valid_classes)
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # 更新类别统计
        class_counts = Counter(labels)
        print(f"过滤后类别分布: {class_counts}")
    
    if len(class_counts) < 2:
        raise ValueError(f"过滤后的数据集中只有{len(class_counts)}个类别，至少需要2个类别进行分类")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
    class_weights = torch.FloatTensor(class_weights)
    print(f"类别权重: {class_weights}")

    return features_scaled, labels_encoded, class_weights, encoder.classes_

def oversample_data(features, labels, ramdom_state=42, method='smote'):
    """
    对不平衡数据进行重采样
    
    Parameters
    ----------
    features : np.ndarray
        特征矩阵
    labels : np.ndarray
        标签向量
    ramdom_state : int, optional
        随机种子, by default 42
    method : str, optional
        重采样方法, by default 'smote'，可选值包括:
        - 'smote': 标准SMOTE过采样
        - 'adasyn': 自适应合成采样
        - 'borderline_smote': 边界SMOTE
        - 'svm_smote': 基于SVM的SMOTE
        - 'random_under': 随机下采样
        - 'near_miss': 最近邻下采样
        - 'smote_enn': SMOTE + ENN清洁
        - 'smote_tomek': SMOTE + Tomek链接清洁
        - 'combined': 组合策略（下采样多数类+过采样少数类）
        
    Returns
    -------
    tuple
        (重采样后的特征, 重采样后的标签)
    """
    print(f"原始数据分布: {Counter(labels)}")
    
    # 选择合适的重采样方法
    if method == 'smote':
        resampler = SMOTE(random_state=ramdom_state)
        print("使用SMOTE过采样方法")
    elif method == 'adasyn':
        resampler = ADASYN(random_state=ramdom_state)
        print("使用ADASYN自适应合成采样方法")
    elif method == 'borderline_smote':
        resampler = BorderlineSMOTE(random_state=ramdom_state)
        print("使用BorderlineSMOTE边界过采样方法")
    elif method == 'svm_smote':
        resampler = SVMSMOTE(random_state=ramdom_state)
        print("使用SVM-SMOTE过采样方法")
    elif method == 'random_under':
        # 计算采样策略，保留所有少数类，降低多数类数量
        class_counts = Counter(labels)
        majority_class = max(class_counts, key=class_counts.get)
        minority_classes = [c for c in class_counts.keys() if c != majority_class]
        if minority_classes:
            # 将多数类下采样到少数类数量的3倍
            target_count = max([class_counts[c] for c in minority_classes]) * 3
            sampling_strategy = {majority_class: min(target_count, class_counts[majority_class])}
            for c in minority_classes:
                sampling_strategy[c] = class_counts[c]
            resampler = RandomUnderSampler(random_state=ramdom_state, sampling_strategy=sampling_strategy)
            print(f"使用随机下采样方法，多数类下采样至: {target_count}")
        else:
            # 如果只有一个类，则使用SMOTE
            resampler = SMOTE(random_state=ramdom_state)
            print("只有一个类别，无法下采样，使用默认SMOTE方法")
    elif method == 'near_miss':
        resampler = NearMiss(version=2)  # 版本2通常效果更好
        print("使用NearMiss最近邻下采样方法")
    elif method == 'smote_enn':
        resampler = SMOTEENN(random_state=ramdom_state)
        print("使用SMOTE+ENN组合重采样方法")
    elif method == 'smote_tomek':
        resampler = SMOTETomek(random_state=ramdom_state)
        print("使用SMOTE+Tomek Links组合重采样方法")
    elif method == 'combined':
        # 先下采样再过采样的组合策略
        # 1. 先对多数类进行下采样
        class_counts = Counter(labels)
        majority_class = max(class_counts, key=class_counts.get)
        minority_classes = [c for c in class_counts.keys() if c != majority_class]
        if minority_classes:
            # 将多数类下采样到少数类平均数量的5倍
            avg_minority = sum([class_counts[c] for c in minority_classes]) / len(minority_classes)
            target_count = int(avg_minority * 5)
            sampling_strategy = {majority_class: min(target_count, class_counts[majority_class])}
            for c in minority_classes:
                sampling_strategy[c] = class_counts[c]
            undersampler = RandomUnderSampler(random_state=ramdom_state, sampling_strategy=sampling_strategy)
            features_under, labels_under = undersampler.fit_resample(features, labels)
            print(f"第1步: 随机下采样，多数类从{class_counts[majority_class]}下采样至{target_count}")
            print(f"下采样后分布: {Counter(labels_under)}")
            
            # 2. 然后应用SMOTE进行过采样
            oversampler = SMOTE(random_state=ramdom_state)
            features_resampled, labels_resampled = oversampler.fit_resample(features_under, labels_under)
            print(f"第2步: SMOTE过采样")
            print(f"最终重采样后分布: {Counter(labels_resampled)}")
            
            return features_resampled, labels_resampled
        else:
            # 如果只有一个类，则使用SMOTE
            resampler = SMOTE(random_state=ramdom_state)
            print("只有一个类别，无法应用组合策略，使用默认SMOTE方法")
    else:
        # 默认使用SMOTE
        resampler = SMOTE(random_state=ramdom_state)
        print(f"未知方法: {method}，使用默认SMOTE方法")
    
    # 执行重采样
    features_resampled, labels_resampled = resampler.fit_resample(features, labels)
    print(f"重采样后分布: {Counter(labels_resampled)}")
    
    return features_resampled, labels_resampled

# 添加数据增强方法 - 对神经元数据添加随机噪声
def augment_with_noise(features, labels, noise_level=0.01, n_samples=1):
    """
    通过添加随机噪声生成新样本
    
    Parameters
    ----------
    features : np.ndarray
        特征矩阵
    labels : np.ndarray
        标签向量
    noise_level : float, optional
        噪声水平，特征标准差的比例, by default 0.01
    n_samples : int, optional
        每个样本生成的新样本数, by default 1
        
    Returns
    -------
    tuple
        (增强后的特征, 增强后的标签)
    """
    # 计算噪声标准差
    feature_std = np.std(features, axis=0)
    noise_std = feature_std * noise_level
    
    # 生成新样本
    aug_features = []
    aug_labels = []
    
    for i in range(len(features)):
        # 添加原始样本
        aug_features.append(features[i])
        aug_labels.append(labels[i])
        
        # 生成新样本
        for _ in range(n_samples):
            # 生成随机噪声
            noise = np.random.normal(0, noise_std)
            # 创建新样本
            new_sample = features[i] + noise
            aug_features.append(new_sample)
            aug_labels.append(labels[i])
    
    return np.array(aug_features), np.array(aug_labels)

# 增强神经元时间序列
def augment_timeseries(features, labels, aug_methods=['jitter', 'scaling'], aug_ratio=0.3):
    """
    对神经元时间序列数据进行增强
    
    Parameters
    ----------
    features : np.ndarray
        特征矩阵
    labels : np.ndarray
        标签向量
    aug_methods : list, optional
        增强方法列表，可选值包括:
        - 'jitter': 添加抖动噪声
        - 'scaling': 放缩变换
        - 'permutation': 随机排列
        - 'rotation': 旋转变换
        by default ['jitter', 'scaling']
    aug_ratio : float, optional
        每个类别增强的比例，by default 0.3
        
    Returns
    -------
    tuple
        (增强后的特征, 增强后的标签)
    """
    # 按类别分组
    unique_labels = np.unique(labels)
    all_features = []
    all_labels = []
    
    for label in unique_labels:
        class_mask = (labels == label)
        class_features = features[class_mask]
        class_labels = labels[class_mask]
        
        # 计算需要增强的样本数
        n_aug = max(int(len(class_features) * aug_ratio), 1)
        
        # 添加原始样本
        all_features.append(class_features)
        all_labels.append(class_labels)
        
        # 随机选择样本进行增强
        indices = np.random.choice(len(class_features), n_aug, replace=False)
        selected_features = class_features[indices]
        
        # 应用选定的增强方法
        aug_features = []
        
        for method in aug_methods:
            if method == 'jitter':
                # 添加小的随机噪声
                sigma = 0.05 * np.std(selected_features, axis=0)
                jitter = np.random.normal(0, sigma, selected_features.shape)
                aug_features.append(selected_features + jitter)
            
            elif method == 'scaling':
                # 随机缩放
                scale_factors = np.random.uniform(0.9, 1.1, (len(selected_features), 1))
                aug_features.append(selected_features * scale_factors)
            
            elif method == 'permutation':
                # 随机交换一小部分特征（每行内部）
                permuted = selected_features.copy()
                for i in range(len(permuted)):
                    segment_length = int(0.1 * selected_features.shape[1])  # 交换10%的特征
                    if segment_length >= 2:
                        idx1, idx2 = np.random.choice(selected_features.shape[1], 2, replace=False)
                        permuted[i, idx1], permuted[i, idx2] = permuted[i, idx2], permuted[i, idx1]
                aug_features.append(permuted)
            
            elif method == 'rotation':
                # 在高维空间中旋转（主要用于图像，但可以尝试）
                if selected_features.shape[1] >= 3:  # 至少需要3维才能旋转
                    rotated = selected_features.copy()
                    for i in range(len(rotated)):
                        # 仅在随机选择的3个维度上应用旋转
                        dims = np.random.choice(selected_features.shape[1], 3, replace=False)
                        angle = np.random.uniform(0, np.pi/18)  # 最多旋转10度
                        # 简单的3D旋转矩阵
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        rotation = np.array([
                            [cos_a, -sin_a, 0],
                            [sin_a, cos_a, 0],
                            [0, 0, 1]
                        ])
                        rotated[i, dims] = np.dot(rotated[i, dims], rotation)
                    aug_features.append(rotated)
        
        # 合并增强样本
        for aug in aug_features:
            all_features.append(aug)
            all_labels.append(np.ones(len(aug)) * label)
    
    # 合并所有类别的样本
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels

# 定义GAN模型相关组件
class Generator(nn.Module):
    """神经元数据生成器网络"""
    def __init__(self, input_dim, neuron_count, hidden_dim=128):
        """
        初始化生成器网络
        
        Parameters
        ----------
        input_dim : int
            输入噪声维度
        neuron_count : int
            神经元数量（输出特征维度）
        hidden_dim : int, optional
            隐藏层维度, by default 128
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 4),
            
            nn.Linear(hidden_dim * 4, neuron_count),
            nn.Tanh()  # 输出范围[-1, 1]，根据需要可换成Sigmoid或其他激活函数
        )
        
    def forward(self, z):
        """前向传播"""
        return self.model(z)

class Discriminator(nn.Module):
    """神经元数据判别器网络"""
    def __init__(self, neuron_count, hidden_dim=128):
        """
        初始化判别器网络
        
        Parameters
        ----------
        neuron_count : int
            神经元数量（输入特征维度）
        hidden_dim : int, optional
            隐藏层维度, by default 128
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(neuron_count, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)

# 定义VAE模型
class VAE(nn.Module):
    """神经元数据变分自编码器"""
    def __init__(self, neuron_count, latent_dim=20, hidden_dim=128):
        """
        初始化VAE网络
        
        Parameters
        ----------
        neuron_count : int
            神经元数量（输入/输出特征维度）
        latent_dim : int, optional
            潜在空间维度, by default 20
        hidden_dim : int, optional
            隐藏层维度, by default 128
        """
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(neuron_count, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )
        
        # 均值和对数方差层
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, neuron_count),
            # 根据数据特点选择激活函数
            nn.Tanh()  # 输出范围[-1,1]
        )
        
        self.latent_dim = latent_dim
        
    def encode(self, x):
        """编码过程：输入到潜在空间"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """解码过程：潜在空间到输出"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# 使用GAN进行数据增强
def augment_with_gan(features, labels, noise_dim=100, batch_size=32, epochs=50, 
                   samples_per_class=None, device=None):
    """
    使用GAN生成神经元数据样本
    
    Parameters
    ----------
    features : np.ndarray
        特征矩阵
    labels : np.ndarray
        标签向量
    noise_dim : int, optional
        噪声维度, by default 100
    batch_size : int, optional
        批次大小, by default 32
    epochs : int, optional
        训练轮数, by default 50
    samples_per_class : dict, optional
        每个类别需要生成的样本数, by default None
        如果为None，则自动平衡各类别
    device : torch.device, optional
        训练设备, by default None
        
    Returns
    -------
    tuple
        (增强后的特征, 增强后的标签)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用GAN进行数据增强，设备: {device}")
    
    # 标准化数据
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 按类别组织数据
    class_counts = Counter(labels)
    unique_labels = np.unique(labels)
    
    # 如果未指定每个类别生成的样本数，则自动平衡各类别
    if samples_per_class is None:
        max_count = max(class_counts.values())
        samples_per_class = {label: max_count - class_counts[label] for label in unique_labels}
    
    # 存储原始数据和生成的数据
    all_features = np.copy(features)
    all_labels = np.copy(labels)
    
    # 为每个类别训练一个GAN并生成样本
    for label in unique_labels:
        # 如果不需要生成额外样本，则跳过
        if samples_per_class[label] <= 0:
            continue
        
        print(f"为类别 {label} 训练GAN，目标生成 {samples_per_class[label]} 个样本")
        
        # 获取当前类别的样本
        mask = (labels == label)
        class_features = features_scaled[mask]
        
        # 如果样本太少，使用过采样增加训练数据
        if len(class_features) < 10:
            print(f"类别 {label} 的样本数量过少 ({len(class_features)})，使用SMOTE增加训练数据")
            sm = SMOTE(random_state=42, k_neighbors=min(5, len(class_features)-1))
            try:
                class_features_ext, _ = sm.fit_resample(
                    class_features, 
                    np.ones(len(class_features))
                )
                # 限制样本数量以避免过拟合
                if len(class_features_ext) > 50:
                    indices = np.random.choice(len(class_features_ext), 50, replace=False)
                    class_features_ext = class_features_ext[indices]
                class_features = class_features_ext
            except Exception as e:
                print(f"SMOTE处理出错: {e}，使用原始数据")
        
        # 创建PyTorch数据集
        tensor_x = torch.FloatTensor(class_features).to(device)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, len(class_features)), shuffle=True)
        
        # 初始化模型
        neuron_count = features.shape[1]
        generator = Generator(noise_dim, neuron_count).to(device)
        discriminator = Discriminator(neuron_count).to(device)
        
        # 优化器
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 损失函数
        criterion = nn.BCELoss()
        
        # 训练GAN
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch_idx, (real_samples,) in enumerate(dataloader):
                batch_size = real_samples.size(0)
                
                # 真实样本的标签
                real_labels = torch.ones(batch_size, 1).to(device)
                # 虚假样本的标签
                fake_labels = torch.zeros(batch_size, 1).to(device)
                
                # 训练判别器
                d_optimizer.zero_grad()
                
                # 真实样本的损失
                outputs = discriminator(real_samples)
                d_loss_real = criterion(outputs, real_labels)
                
                # 生成虚假样本
                noise = torch.randn(batch_size, noise_dim).to(device)
                fake_samples = generator(noise)
                
                # 虚假样本的损失
                outputs = discriminator(fake_samples.detach())
                d_loss_fake = criterion(outputs, fake_labels)
                
                # 判别器总损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # 训练生成器
                g_optimizer.zero_grad()
                
                # 生成虚假样本并计算损失
                outputs = discriminator(fake_samples)
                g_loss = criterion(outputs, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"D Loss: {np.mean(d_losses):.4f}, "
                      f"G Loss: {np.mean(g_losses):.4f}")
        
        # 生成新样本
        generator.eval()
        with torch.no_grad():
            n_samples = samples_per_class[label]
            noise = torch.randn(n_samples, noise_dim).to(device)
            fake_samples = generator(noise).cpu().numpy()
        
        # 逆标准化
        fake_samples = scaler.inverse_transform(fake_samples)
        
        # 添加到总数据集
        all_features = np.vstack([all_features, fake_samples])
        all_labels = np.append(all_labels, np.ones(n_samples) * label)
    
    print(f"GAN增强后的数据分布: {Counter(all_labels)}")
    return all_features, all_labels

# 使用VAE进行数据增强
def augment_with_vae(features, labels, latent_dim=20, batch_size=32, epochs=50, 
                   samples_per_class=None, device=None):
    """
    使用VAE生成神经元数据样本
    
    Parameters
    ----------
    features : np.ndarray
        特征矩阵
    labels : np.ndarray
        标签向量
    latent_dim : int, optional
        潜在空间维度, by default 20
    batch_size : int, optional
        批次大小, by default 32
    epochs : int, optional
        训练轮数, by default 50
    samples_per_class : dict, optional
        每个类别需要生成的样本数, by default None
        如果为None，则自动平衡各类别
    device : torch.device, optional
        训练设备, by default None
        
    Returns
    -------
    tuple
        (增强后的特征, 增强后的标签)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用VAE进行数据增强，设备: {device}")
    
    # 标准化数据
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 按类别组织数据
    class_counts = Counter(labels)
    unique_labels = np.unique(labels)
    
    # 如果未指定每个类别生成的样本数，则自动平衡各类别
    if samples_per_class is None:
        max_count = max(class_counts.values())
        samples_per_class = {label: max_count - class_counts[label] for label in unique_labels}
    
    # 存储原始数据和生成的数据
    all_features = np.copy(features)
    all_labels = np.copy(labels)
    
    # 为每个类别训练一个VAE并生成样本
    for label in unique_labels:
        # 如果不需要生成额外样本，则跳过
        if samples_per_class[label] <= 0:
            continue
        
        print(f"为类别 {label} 训练VAE，目标生成 {samples_per_class[label]} 个样本")
        
        # 获取当前类别的样本
        mask = (labels == label)
        class_features = features_scaled[mask]
        
        # 如果样本太少，使用过采样增加训练数据
        if len(class_features) < 10:
            print(f"类别 {label} 的样本数量过少 ({len(class_features)})，使用SMOTE增加训练数据")
            sm = SMOTE(random_state=42, k_neighbors=min(5, len(class_features)-1))
            try:
                class_features_ext, _ = sm.fit_resample(
                    class_features, 
                    np.ones(len(class_features))
                )
                # 限制样本数量以避免过拟合
                if len(class_features_ext) > 50:
                    indices = np.random.choice(len(class_features_ext), 50, replace=False)
                    class_features_ext = class_features_ext[indices]
                class_features = class_features_ext
            except Exception as e:
                print(f"SMOTE处理出错: {e}，使用原始数据")
        
        # 创建PyTorch数据集
        tensor_x = torch.FloatTensor(class_features).to(device)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, len(class_features)), shuffle=True)
        
        # 初始化模型
        neuron_count = features.shape[1]
        vae = VAE(neuron_count, latent_dim).to(device)
        
        # 优化器
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        
        # 训练VAE
        for epoch in range(epochs):
            vae.train()
            train_loss = 0
            recon_loss = 0
            kl_loss = 0
            
            for batch_idx, (data,) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # 前向传播
                recon_batch, mu, logvar = vae(data)
                
                # 重构损失
                reconstruction_loss = F.mse_loss(recon_batch, data, reduction='sum')
                
                # KL散度
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # 总损失
                beta = 0.5  # KL散度的权重
                loss = reconstruction_loss + beta * kl_divergence
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                recon_loss += reconstruction_loss.item()
                kl_loss += kl_divergence.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Loss: {train_loss / len(dataset):.4f}, "
                      f"Recon: {recon_loss / len(dataset):.4f}, "
                      f"KL: {kl_loss / len(dataset):.4f}")
        
        # 生成新样本
        vae.eval()
        with torch.no_grad():
            n_samples = samples_per_class[label]
            z = torch.randn(n_samples, latent_dim).to(device)
            samples = vae.decode(z).cpu().numpy()
        
        # 逆标准化
        samples = scaler.inverse_transform(samples)
        
        # 添加到总数据集
        all_features = np.vstack([all_features, samples])
        all_labels = np.append(all_labels, np.ones(n_samples) * label)
    
    print(f"VAE增强后的数据分布: {Counter(all_labels)}")
    return all_features, all_labels

# 更新增强平衡数据集函数以包含GAN和VAE
def enhance_balanced_dataset(features, labels, methods=None):
    """
    使用多种数据增强方法组合处理不平衡数据集
    
    Parameters
    ----------
    features : np.ndarray
        特征矩阵
    labels : np.ndarray
        标签向量
    methods : list, optional
        增强方法列表，默认为['combined', 'timeseries']
        可选值包括:
        - 'combined': 组合重采样（下采样+过采样）
        - 'smote': 标准SMOTE过采样
        - 'borderline': 边界SMOTE过采样
        - 'adasyn': ADASYN自适应合成采样
        - 'noise': 添加随机噪声
        - 'timeseries': 时间序列特定增强
        - 'gan': 生成对抗网络增强
        - 'vae': 变分自编码器增强
        
    Returns
    -------
    tuple
        (增强后的特征, 增强后的标签)
    """
    if methods is None:
        methods = ['combined', 'timeseries']
    
    print("开始多步骤数据增强处理...")
    print(f"原始数据分布: {Counter(labels)}")
    
    current_features, current_labels = features, labels
    
    for method in methods:
        if method == 'combined':
            # 使用组合重采样（下采样+过采样）
            current_features, current_labels = oversample_data(
                current_features, current_labels, 
                ramdom_state=42, method='combined'
            )
        elif method == 'smote':
            # 使用标准SMOTE
            current_features, current_labels = oversample_data(
                current_features, current_labels, 
                ramdom_state=42, method='smote'
            )
        elif method == 'borderline':
            # 使用边界SMOTE
            current_features, current_labels = oversample_data(
                current_features, current_labels, 
                ramdom_state=42, method='borderline_smote'
            )
        elif method == 'adasyn':
            # 使用ADASYN
            current_features, current_labels = oversample_data(
                current_features, current_labels, 
                ramdom_state=42, method='adasyn'
            )
        elif method == 'noise':
            # 添加随机噪声
            current_features, current_labels = augment_with_noise(
                current_features, current_labels, 
                noise_level=0.01, n_samples=1
            )
            print(f"添加噪声后数据分布: {Counter(current_labels)}")
        elif method == 'timeseries':
            # 添加时间序列增强
            current_features, current_labels = augment_timeseries(
                current_features, current_labels, 
                aug_methods=['jitter', 'scaling'], 
                aug_ratio=0.2
            )
            print(f"时间序列增强后数据分布: {Counter(current_labels)}")
        elif method == 'gan':
            # 使用GAN进行增强
            current_features, current_labels = augment_with_gan(
                current_features, current_labels,
                epochs=30,  # 减少训练轮数以加快处理速度
                batch_size=16
            )
        elif method == 'vae':
            # 使用VAE进行增强
            current_features, current_labels = augment_with_vae(
                current_features, current_labels,
                epochs=30,  # 减少训练轮数以加快处理速度
                batch_size=16
            )
    
    print(f"最终增强后数据分布: {Counter(current_labels)}")
    return current_features, current_labels

# 计算特征之间的相关性矩阵, Pearson 相关系数
def compute_correlation_matrix(features):
    features_T = features.T
    num_neurons = features_T.shape[0]
    correlation_matrix = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(i, num_neurons):
            corr, _ = pearsonr(features_T[i], features_T[j])
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr

    return correlation_matrix

def generate_graph(sample_features, correlation_matrix, threshold=0.4):
    num_neurons = len(sample_features)
    edges_src = [] # 源节点
    edges_dst = [] # 目标节点
    edges_weights = []

    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            corr = correlation_matrix[i, j]
            if abs(corr) > threshold:
                edges_src.append(i)
                edges_dst.append(j)
                edges_weights.append(abs(corr))
                # 无向图，所以需要添加反向边
                edges_src.append(j)
                edges_dst.append(i)
                edges_weights.append(abs(corr))

    # 如果没有边，则添加一些简单的边
    if len(edges_src) == 0:
        print("No edges found in the graph here!")
        for i in range(num_neurons):
            j = (i + 1) % num_neurons
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
            edges_weights.extend([0.1, 0.1])

    # 将边的列表转换为 PyTorch 张量，(2, num_edges)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    # 将边的权值转换为 PyTorch 张量，(num_edges,)
    edge_attr = torch.tensor(edges_weights, dtype=torch.float)

    return edge_index, edge_attr

# 生成 PyG 数据对象
def create_pyg_dataset(features, labels, correlation_matrix, threshold=0.4):
    data_list = []
    for i in range(len(features)):
        sample = features[i]
        x = torch.tensor(sample.reshape(-1, 1), dtype=torch.float)  # 钙离子浓度 (num_neurons, 1)
        edge_index, edge_attr = generate_graph(sample, correlation_matrix, threshold)
        data = Data(
            x=x, # 钙离子浓度, (num_neurons, 1)
            edge_index=edge_index, # 边索引, (2, num_edges)
            edge_attr=edge_attr, # 边权值, (num_edges,)
            y=torch.tensor(labels[i], dtype=torch.long), # 标签, (1,)
        )
        
        data_list.append(data)
    return data_list
    
# 将生成的拓扑图可视化
def visualize_graph(data, sample_index=0, title="Neuron Connection Graph", result_dir='result', position_file=None):
    plt.figure(figsize=(10, 10))
    graph_data = data[sample_index]
    G = nx.Graph()
    for i in range(graph_data.x.shape[0]):
        node_value = float(graph_data.x[i][0])
        G.add_node(i, value=node_value)
    for i in range(graph_data.edge_index.shape[1]):
        src = int(graph_data.edge_index[0, i])
        dst = int(graph_data.edge_index[1, i])
        weight = float(graph_data.edge_attr[i]) if graph_data.edge_attr is not None else 1.0
        G.add_edge(src, dst, weight=weight)

    # 如果提供了位置文件，则从中读取真实的空间位置
    if position_file and os.path.exists(position_file):
        try:
            # 读取位置数据
            position_data = pd.read_csv(position_file)
            print(f"位置数据文件包含 {len(position_data)} 个神经元的位置信息")
            
            # 创建一个从神经元ID到位置的映射
            position_map = {}
            
            # 检查是否存在神经元编号与索引不一致的问题
            neuron_ids_in_csv = set([int(row['number']) for _, row in position_data.iterrows()])
            nodes_in_graph = set(range(len(G.nodes)))
            
            # 查看最小和最大神经元ID，可能的偏移量
            min_neuron_id = min(neuron_ids_in_csv) if neuron_ids_in_csv else -1
            max_neuron_id = max(neuron_ids_in_csv) if neuron_ids_in_csv else -1
            print(f"CSV中神经元ID范围: {min_neuron_id} 到 {max_neuron_id}")
            print(f"图中节点ID范围: 0 到 {len(G.nodes)-1}")
            
            # 检测是否需要应用偏移量 (如果CSV从1开始而图从0开始)
            offset = 0
            if min_neuron_id == 1 and 0 in nodes_in_graph and 0 not in neuron_ids_in_csv:
                offset = -1
                print(f"检测到神经元ID偏移: 应用 {offset} 的偏移量")
            
            missing_nodes = []
            for i in range(len(G.nodes)):
                csv_id = i - offset  # 计算CSV中的对应ID
                row = position_data[position_data['number'] == csv_id]
                if len(row) > 0:
                    position_map[i] = (float(row['relative_x'].values[0]), float(row['relative_y'].values[0]))
                else:
                    missing_nodes.append(i)
            
            # 如果我们有足够的位置信息，使用真实的空间位置
            if len(position_map) == len(G.nodes):
                pos = position_map
                print(f"使用真实空间位置可视化 {len(pos)} 个神经元")
            else:
                print(f"警告：位置数据不完整 ({len(position_map)}/{len(G.nodes)})")
                if missing_nodes:
                    print(f"缺失节点ID: {missing_nodes}")
                    
                # 尝试使用可用的位置数据，对缺失的使用布局算法补充
                if len(position_map) > len(G.nodes) * 0.8:  # 如果有超过80%的节点有位置数据
                    print("使用部分真实位置数据，缺失部分使用布局算法补充")
                    # 为缺失的节点生成位置
                    try:
                        # 使用布局算法为缺失节点生成位置
                        temp_g = G.copy()
                        for node in position_map:
                            temp_g.remove_node(node)
                        
                        if temp_g.number_of_nodes() > 0:
                            temp_pos = nx.spring_layout(temp_g, seed=42)
                            # 合并两个位置字典
                            pos = {**position_map, **temp_pos}
                        else:
                            pos = position_map
                    except Exception as e:
                        print(f"补充缺失节点位置时出错: {e}")
                        pos = position_map
                else:  # 如果缺失太多，使用布局算法
                    print("缺失位置数据过多，使用布局算法")
                    try:
                        pos = nx.kamada_kawai_layout(G)
                    except:
                        pos = nx.spring_layout(G, seed=42)
        except Exception as e:
            print(f"读取位置数据时出错：{e}，将使用布局算法")
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G, seed=42)
    else:
        # 如果没有位置文件，使用布局算法
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)

    # 获取节点值以用于颜色映射
    node_values = [G.nodes[i]['value'] for i in range(len(G.nodes))]
    vmin = min(node_values)
    vmax = max(node_values)
    
    # 根据边权重确定边的宽度
    edge_weights = [G.edges[edge]['weight'] * 3 for edge in G.edges]
    
    # 创建颜色映射
    cmap = plt.cm.coolwarm

    # 绘制节点
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_values,
        cmap=cmap,
        node_size=350,
        alpha=0.9,
        vmin=vmin,
        vmax=vmax,
        edgecolors='black',
        linewidths=0.5
    )
    
    # 绘制边 - 解决connectionstyle警告
    edges = nx.draw_networkx_edges(
        G, pos,
        width=edge_weights,
        edge_color='gray',
        alpha=0.6,
        arrows=True  # 使用arrows=True替代connectionstyle
    )
    
    # 绘制节点标签
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_family='sans-serif',
        font_weight='bold'
    )
    
    # 添加颜色条
    cbar = plt.colorbar(nodes, label='Calcium Concentration', shrink=0.8)
    cbar.ax.tick_params(labelsize=9)
    
    # 添加标题和信息
    behavior_label = graph_data.y.item()
    plt.title(f"{title}\nSample Label: {behavior_label}", fontsize=14, fontweight='bold')
    plt.text(0.02, 0.02, f"Number of Nodes: {G.number_of_nodes()}, Number of Edges: {G.number_of_edges()}",
             transform=plt.gca().transAxes, fontsize=10)
    
    # 添加坐标位置信息
    position_source = "Real Space Position" if position_file and os.path.exists(position_file) else "Layout Algorithm Generated"
    plt.text(0.02, 0.06, f"Position Source: {position_source}",
             transform=plt.gca().transAxes, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()