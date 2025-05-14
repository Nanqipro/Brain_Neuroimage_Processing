import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest

def extract_advanced_features(features, pca_model=None, is_train=True):
    """提取高级特征，确保训练集和测试集使用相同的转换"""
    n_samples, n_features = features.shape
    
    # 1. 统计特征
    mean_features = np.mean(features, axis=1).reshape(-1, 1)
    std_features = np.std(features, axis=1).reshape(-1, 1)
    max_features = np.max(features, axis=1).reshape(-1, 1)
    min_features = np.min(features, axis=1).reshape(-1, 1)
    
    # 2. 非线性变换
    log_features = np.log1p(np.abs(features))  # log(1+x)
    
    # 3. PCA特征
    if is_train:
        # 训练集上拟合PCA模型
        pca_model = PCA(n_components=0.95)
        pca_features = pca_model.fit_transform(features)
        print(f"PCA保留了{pca_features.shape[1]}个主成分")
    else:
        # 测试集上使用训练好的PCA模型
        if pca_model is None:
            raise ValueError("测试集转换需要提供训练好的PCA模型")
        pca_features = pca_model.transform(features)
    
    # 组合所有特征
    combined_features = np.hstack([
        features,  # 原始特征
        mean_features, std_features, max_features, min_features,  # 统计特征
        pca_features  # PCA特征
    ])
    
    print(f"特征扩展: 从{n_features}维扩展到{combined_features.shape[1]}维")
    
    return combined_features, pca_model

def select_features(features, labels, k=40, selector=None, is_train=True):
    """基于互信息选择最重要的特征"""
    if is_train:
        # 训练集上拟合选择器
        selector = SelectKBest(mutual_info_classif, k=k)
        selected_features = selector.fit_transform(features, labels)
        
        # 获取选择的特征索引
        top_indices = selector.get_support(indices=True)
        print(f"特征选择: 从{features.shape[1]}维减少到{k}维")
        print(f"选择的特征索引: {top_indices}")
    else:
        # 测试集上使用训练好的选择器
        if selector is None:
            raise ValueError("测试集转换需要提供训练好的选择器")
        selected_features = selector.transform(features)
    
    return selected_features, selector