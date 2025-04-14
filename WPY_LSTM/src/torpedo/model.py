import torch
import torch.nn as nn
import numpy as np # 保留 numpy 以便设置随机种子

def set_random_seed(seed):
    """
    设置随机种子以确保结果可重现
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

# --- 自编码器 ---
class NeuronAutoencoder(nn.Module):
    """
    神经元自编码器
    """
    def __init__(self, input_size, hidden_size, latent_dim):
        super(NeuronAutoencoder, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        original_shape = x.shape
        if x.size(-1) != self.input_size:
            # 处理维度不匹配 (简化处理，实际应用中应更严格)
            padding_size = self.input_size - x.size(-1)
            if padding_size > 0:
                padding = torch.zeros(*x.shape[:-1], padding_size, device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[..., :self.input_size]
        
        try:
            if len(original_shape) == 3: # (batch, seq, features)
                batch_size, seq_len, features = original_shape
                x_reshaped = x.reshape(-1, features)
                encoded = self.encoder(x_reshaped)
                decoded = self.decoder(encoded)
                encoded = encoded.reshape(batch_size, seq_len, self.latent_dim)
                decoded = decoded.reshape(original_shape)
            elif len(original_shape) == 2: # (batch, features) or (features)
                 # 如果是单个样本 (features)，需要 unsqueeze
                 if len(original_shape) == 1:
                      x = x.unsqueeze(0)
                 encoded = self.encoder(x)
                 decoded = self.decoder(encoded)
                 if len(original_shape) == 1:
                      encoded = encoded.squeeze(0)
                      decoded = decoded.squeeze(0)
            else:
                 raise ValueError(f"不支持的输入维度: {len(original_shape)}")
                 
            return encoded, decoded
                
        except RuntimeError as e:
            print(f"自编码器前向传播错误: {e}")
            print(f"输入形状: {x.shape}, 期望输入大小: {self.input_size}")
            # 返回零张量以允许流程继续，但在日志中应记录此错误
            encoded_zeros = torch.zeros(*original_shape[:-1], self.latent_dim, device=x.device)
            decoded_zeros = torch.zeros_like(x.view(original_shape)) # 确保形状正确
            return encoded_zeros, decoded_zeros

# --- 多头注意力 ---
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        if input_dim % num_heads != 0:
             raise ValueError(f"input_dim ({input_dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1) # 获取序列长度
        
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            # 确保 mask 维度兼容
            # mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.output_linear(context)
        
        return output, attention_weights

# --- 时间注意力 ---
class TemporalAttention(nn.Module):
    """
    时间注意力机制
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        attention_scores = self.attention(hidden_states).squeeze(-1) # (batch, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1) # (batch, seq_len, 1)
        attended = torch.sum(hidden_states * attention_weights, dim=1) # (batch, hidden_size)
        return attended, attention_weights.squeeze(-1) # 返回 (batch, seq_len) 的权重

# --- 增强型 LSTM 模型 ---
class EnhancedNeuronLSTM(nn.Module):
    """
    增强版神经元LSTM模型
    整合了自编码器、双向LSTM和多层注意力机制
    
    [Simplified Version: Only BiLSTM + Classifier]
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 latent_dim=32, num_heads=4, dropout=0.2): # 保留签名以兼容旧配置加载，但 latent_dim/num_heads 不再使用
        """
        参数:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层大小
            num_layers (int): LSTM层数
            num_classes (int): 输出类别数
            latent_dim (int): [不再使用]
            num_heads (int): [不再使用]
            dropout (float): Dropout比率
        """
        super(EnhancedNeuronLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        # self.latent_dim = latent_dim # 不再使用
        
        # --- 移除/注释掉 AE 和 Attention --- 
        # self.autoencoder = NeuronAutoencoder(input_size, hidden_size, latent_dim)
        
        # 双向LSTM (直接接收 input_size)
        self.lstm = nn.LSTM(
            input_size=input_size, # 直接使用原始输入维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # self.multihead_attention = MultiHeadAttention(hidden_size * 2, num_heads)
        # self.temporal_attention = TemporalAttention(hidden_size * 2)
        
        # 批标准化 (接收 LSTM 输出: hidden_size * 2)
        # 注意：LSTM 输出是 (batch, seq, hidden*2)，我们需要对最后一个时间步的输出或某种聚合进行标准化
        # 这里假设我们取最后一个时间步的输出进行分类
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 分类器 (接收 BatchNorm 输出: hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 记录模型配置
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
            # 'latent_dim': latent_dim,
            # 'num_heads': num_heads,
            'dropout': dropout,
            'architecture': 'SimpleBiLSTM' # 标记简化架构
        }
        
    def forward(self, x):
        """
        前向传播 (简化版)
        """
        batch_size = x.size(0)
        
        # 检查输入维度
        if x.size(-1) != self.input_size:
            padding_size = self.input_size - x.size(-1)
            if padding_size > 0:
                padding = torch.zeros(batch_size, x.size(1), padding_size, device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[..., :self.input_size]
            
        try:
            # --- 移除 AE 和 Attention 调用 ---
            # encoded, decoded = self.autoencoder(x)
            
            # 通过LSTM处理序列
            # lstm_out: (batch, seq_len, hidden_size * 2)
            lstm_out, (h_n, c_n) = self.lstm(x) 
            
            # --- 取最后一个时间步的输出进行分类 ---
            # lstm_out[:, -1, :]: (batch, hidden_size * 2)
            last_time_step_out = lstm_out[:, -1, :]
            
            # --- 移除 Attention 调用 --- 
            # attended, attention_weights = self.multihead_attention(lstm_out, None)
            # temporal_context, temporal_weights = self.temporal_attention(attended)
            
            # 批标准化 (对最后一个时间步的输出进行)
            normalized = self.batch_norm(last_time_step_out)
            
            # 分类
            output = self.classifier(normalized)
            
            # 返回分类输出，注意力权重设为 None
            return output, None, None 
            
        except RuntimeError as e:
            if "size mismatch" in str(e) or "dimension" in str(e):
                print(f"前向传播中的维度错误: {str(e)}")
                print(f"输入形状: {x.shape}, 模型输入大小: {self.input_size}")
                return (
                    torch.zeros(batch_size, self.num_classes, device=x.device),
                    None, None
                )
            else:
                raise e

    def summary(self):
        """
        打印模型摘要信息
        """
        print("\n--- 模型摘要 ---")
        print(f"输入大小 (Input Size): {self.input_size}")
        print(f"LSTM 隐藏层大小 (Hidden Size): {self.hidden_size}")
        print(f"LSTM 层数 (Num Layers): {self.num_layers}")
        print(f"类别数 (Num Classes): {self.num_classes}")
        # print(f"自编码器潜在维度 (Latent Dim): {self.latent_dim}")
        # print(f"多头注意力头数 (Num Heads): {self.num_heads}")
        print(f"Dropout 率: {self.config['dropout']}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print("---------------")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params
        }

if __name__ == '__main__':
    # 测试模型是否能正确初始化和执行前向传播
    print("\n--- 测试模型初始化和前向传播 ---")
    # 假设配置
    INPUT_SIZE = 43
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    NUM_CLASSES = 15
    LATENT_DIM = 32
    NUM_HEADS = 4
    DROPOUT = 0.2
    SEQ_LEN = 10
    BATCH_SIZE = 5
    
    # 检查 hidden_size * 2 是否能被 num_heads 整除
    if (HIDDEN_SIZE * 2) % NUM_HEADS != 0:
         print(f"警告: (hidden_size * 2) = {HIDDEN_SIZE * 2} 不能被 num_heads = {NUM_HEADS} 整除。测试可能失败或与实际配置不符。")
         # 可以选择调整 HIDDEN_SIZE 或 NUM_HEADS 使其可整除以进行测试
         # 例如： HIDDEN_SIZE = 256 -> 256*2=512. 512 % 4 == 0. OK.
         # 例如： HIDDEN_SIZE = 128 -> 128*2=256. 256 % 4 == 0. OK.

    try:
        model = EnhancedNeuronLSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT
        )
        model.summary()
        
        # 创建假的输入数据
        dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
        print(f"\n输入虚拟数据形状: {dummy_input.shape}")
        
        # 执行前向传播
        output, attention_weights, temporal_weights = model(dummy_input)
        
        print(f"模型输出形状: {output.shape} (预期: [{BATCH_SIZE}, {NUM_CLASSES}])")
        print(f"多头注意力权重形状: {attention_weights.shape} (预期: [{BATCH_SIZE}, {NUM_HEADS}, {SEQ_LEN}, {SEQ_LEN}])")
        print(f"时间注意力权重形状: {temporal_weights.shape} (预期: [{BATCH_SIZE}, {SEQ_LEN}])")
        
        print("\n--- 模型测试成功 --- ")
        
    except Exception as e:
        print(f"\n--- 模型测试失败 --- ")
        import traceback
        traceback.print_exc() 