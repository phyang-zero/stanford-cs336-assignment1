import torch
import torch.nn as nn
from einops import einsum, rearrange

from typing import Optional
import math

class Linear(nn.Module):
    """
    无偏置线性层：y = xW^T
    输入：(..., in_features)
    输出：(..., out_features)
    权重参数 W，形状 (out_features, in_features)
    权重初始化为截断正态分布
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """
        初始化 Linear 层参数。
        in_features: 输入维度
        out_features: 输出维度
        device, dtype: 张量设备和类型
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        a = -30 / std
        b = 30 / std
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：y = xW^T
        输入：x (..., in_features)
        输出：(..., out_features)
        """
        return einsum(x, self.W, "... i, o i -> ... o")
    
class Embedding(nn.Module):
    """
    嵌入层：将 token ID 映射为向量
    输入：(..., ) 整数索引
    输出：(..., embedding_dim) 嵌入向量
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        初始化嵌入层参数。
        num_embeddings: 词汇表大小
        embedding_dim: 嵌入向量维度
        device, dtype: 张量设备和类型
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播：根据 token ID 查找嵌入向量
        输入：token_ids (...,)
        输出：(..., embedding_dim)
        """
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)
    输入: (..., d_model)
    输出: (..., d_model)
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        初始化 RMSNorm 层。
        d_model: 模型维度
        eps: 防止除以零的小数
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        输入: x (..., d_model)
        输出: (..., d_model)
        """
        # 保存原始数据类型，并将输入提升为 float32 以保证计算稳定性
        original_dtype = x.dtype
        x_float = x.to(torch.float32)
        
        # 计算输入的均方根的倒数
        rrms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # 归一化并乘以可学习的权重，然后转换回原始数据类型
        normalized_x = x_float * rrms * self.weight
        return normalized_x.to(original_dtype)

class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数模块
    输入: (..., d_model)
    输出: (..., d_model)
    """
    def __init__(self, d_model: int, d_ff: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        初始化 SwiGLU 层。
        d_model: 输入和输出维度
        d_ff: 中间隐藏层维度
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype) # 门控上采样层
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype) # 下采样层
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) # 数据流上采样层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        输入: x (..., d_model)
        输出: (..., d_model)
        """
        main_stream = self.w1(x) # 主数据流通过 w1
        gate_stream = self.w3(x) # 门控流通过 w3
        # SiLU(z) = z * sigmoid(z)
        activated_stream = main_stream * torch.sigmoid(main_stream)
        gated_output = activated_stream * gate_stream
        return self.w2(gated_output)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    带数值稳定性的 Softmax 函数。
    """
    # 减去最大值以防止指数爆炸
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算缩放点积注意力。
    """
    # 1. 计算 Q 和 K 的转置的点积，然后缩放
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 应用掩码 (如果提供)
    if mask is not None:
        # --- 这是需要修改的地方 ---
        # 检查 mask 的类型，以决定如何应用它
        if mask.dtype == torch.bool:
            # 如果是布尔掩码，使用 masked_fill
            scores = scores.masked_fill(mask == False, float('-inf'))
        else:
            # 如果是浮点数掩码 (带有 -inf)，直接相加
            scores = scores + mask

    # 3. 对缩放后的分数应用你自己的 softmax 函数
    attn_weights = softmax(scores, dim=-1)

    # 4. 将注意力权重与 V 相乘
    output = torch.matmul(attn_weights, V)

    return output

class RoPE(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding)
    """
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        初始化 RoPE 层。
        d_k: Query 或 Key 的维度
        max_seq_len: 支持的最大序列长度
        theta: RoPE 的基准参数
        """
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算旋转频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        t = torch.arange(max_seq_len, device=device, dtype=dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # 注册为 buffer，这样它会随模型移动到不同设备，但不是模型参数
        self.register_buffer('freqs_cos', freqs.cos())
        self.register_buffer('freqs_sin', freqs.sin())

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对输入张量应用 RoPE。
        输入:
            x (..., seq_len, d_k): 输入的 Q 或 K 矩阵
            token_positions (..., seq_len): 可选的 token 位置信息
        输出:
            (..., seq_len, d_k): 经过旋转编码的 Q 或 K 矩阵
        """
        seq_len = x.shape[-2]
        
        # 获取对应位置的 cos 和 sin 值
        if token_positions is None:
            cos = self.freqs_cos[:seq_len]
            sin = self.freqs_sin[:seq_len]
        else:
            cos = self.freqs_cos[token_positions]
            sin = self.freqs_sin[token_positions]

        # 将 x 的特征维度两两一组进行处理
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # 应用旋转公式
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 将旋转后的结果拼接起来
        rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1)
        return rearrange(rotated_x, '... d two -> ... (d two)')

class MultiheadSelfAttention(nn.Module):
    """
    多头自注意力机制
    输入: x (..., sequence_length, d_model)
    输出: (..., sequence_length, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, rope: Optional[RoPE] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        初始化多头自注意力层。
        d_model: 模型维度
        num_heads: 注意力头的数量
        rope: 可选的 RoPE 模块
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        # 使用单个线性层一次性完成 Q, K, V 的投影
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        # 输出投影层
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        """
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, '... s (three h d) -> three ... h s d', three=3, h=self.num_heads)
        if self.rope is not None:
            q = self.rope(q, token_positions=token_positions)
            k = self.rope(k, token_positions=token_positions)
        if mask is None:
            seq_len = x.shape[-2]
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=q.dtype), diagonal=1)
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)
        output = rearrange(attn_output, '... h s d -> ... s (h d)')
        output = self.o_proj(output)

        return output

class TransformerBlock(nn.Module):
    """
    Transformer 模块，包含多头自注意力和前馈网络。
    输入: x (..., sequence_length, d_model)
    输出: (..., sequence_length, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: Optional[RoPE] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        初始化 TransformerBlock。
        d_model: 模型维度
        num_heads: 注意力头的数量
        d_ff: 前馈网络中间层维度
        rope: 可选的 RoPE 模块
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = MultiheadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        输入:
            x (..., sequence_length, d_model): 输入序列
            mask (..., sequence_length, sequence_length): 可选的注意力掩码
            token_positions (..., sequence_length): 可选的 token 位置信息 (用于 RoPE)
        输出:
            (..., sequence_length, d_model): Transformer 模块的输出
        """
        # --- Attention Sub-layer ---
        normalized_x = self.norm1(x)
        attention_output = self.attention(normalized_x, mask=mask, token_positions=token_positions)
        x = x + attention_output

        # --- Feed-Forward Sub-layer ---
        normalized_x2 = self.norm2(x)
        ffn_output = self.ffn(normalized_x2)
        x = x + ffn_output

        return x

class TransformerLM(nn.Module):
    """
    一个完整的 Transformer 语言模型。
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_seq_len: int, rope_theta: float = 10000.0, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        初始化 TransformerLM。
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_layers: TransformerBlock 的层数
        num_heads: 注意力头的数量
        d_ff: 前馈网络中间层维度
        max_seq_len: 最大序列长度
        rope_theta: RoPE 的基准参数
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RoPE(d_model // num_heads, max_seq_len, theta=rope_theta, device=device, dtype=dtype)
        
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        
        self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        输入:
            token_ids (batch_size, sequence_length): 输入的 token ID
            token_positions (batch_size, sequence_length): 可选的 token 位置信息
        输出:
            logits (batch_size, sequence_length, vocab_size): 预测的 logits
        """
        seq_len = token_ids.shape[-1]
        
        # 1. 创建注意力掩码
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=token_ids.device), diagonal=1)
        
        # 2. 获取词嵌入
        x = self.embedding(token_ids)
        
        # 3. 依次通过所有 TransformerBlock
        for block in self.blocks:
            x = block(x, mask=mask, token_positions=token_positions)
            
        # 4. 应用最终的层归一化
        x = self.norm_final(x)
        
        # 5. 应用 LM Head 得到 logits
        logits = self.lm_head(x)
        
        return logits