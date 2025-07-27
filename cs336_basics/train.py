import torch
from torch import Tensor, from_numpy
from torch.optim.optimizer import Optimizer
import math
from jaxtyping import Float, Int
import numpy.typing as npt
import numpy as np
import os
from typing import IO, BinaryIO

def cross_entropy_loss(
    inputs: Float[Tensor, "... vocab_size"], 
    targets: Int[Tensor, " ..."]
) -> Float[Tensor, ""]:
    """
    计算数值稳定的交叉熵损失。

    Args:
        inputs (torch.Tensor): 模型的输出 logits，形状为 (..., vocab_size)，
                               其中 '...' 代表任意数量的批处理维度。
        targets (torch.Tensor): 真实的目标 token ID，形状为 (...)。

    Returns:
        torch.Tensor: 一个标量 (scalar)，代表整个批次的平均交叉熵损失。
    """
    # 1. 为保证数值稳定性，从 logits 中减去最大值。
    #    这不会改变 softmax 的结果，但可以防止 exp() 溢出。
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    stable_logits = inputs - max_logits

    # 2. 计算 log(sum(exp(logits)))，这是 log-softmax 的分母部分。
    #    利用公式 log(sum(exp(x))) = c + log(sum(exp(x - c)))，其中 c 是 max_logits。
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1))
    log_softmax_denominator = max_logits.squeeze(-1) + log_sum_exp

    # 3. 获取目标 token 对应的 logit 值。
    #    我们需要使用 torch.gather 来根据 targets 的索引，从 inputs 中精确地挑选出正确的 logit。
    #    targets.unsqueeze(-1) 是为了让 targets 的维度与 inputs 匹配以进行 gather 操作。
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 4. 计算每个样本的损失。
    #    loss = log(sum(exp(logits))) - target_logit
    loss_per_example = log_softmax_denominator - target_logits

    # 5. 返回整个批次的平均损失。
    return torch.mean(loss_per_example)

class AdamW(Optimizer):
    """
    实现了 AdamW 优化器。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        初始化 AdamW 优化器。

        Args:
            params (iterable): 模型的参数。
            lr (float): 学习率 alpha [cite: 929]。
            betas (Tuple[float, float]): 用于计算一阶和二阶矩的 beta 参数 [cite: 929]。
            eps (float): 为保证数值稳定性加到分母上的 epsilon [cite: 929]。
            weight_decay (float): 权重衰减系数 lambda [cite: 929]。
        """
        # 对输入超参数进行合法性检查
        # if not 0.0 <= lr:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        # if not 0.0 <= eps:
        #     raise ValueError(f"Invalid epsilon value: {eps}")
        # if not 0.0 <= betas[0] < 1.0:
        #     raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        # if not 0.0 <= betas[1] < 1.0:
        #     raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # if not 0.0 <= weight_decay:
        #     raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 将超参数打包成字典，并调用父类的构造函数
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()  # 使用 no_grad 装饰器，因为我们是手动修改参数，不需要 PyTorch 跟踪梯度
    def step(self, closure=None):
        """
        执行一步优化。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有参数组 (通常只有一个)
        for group in self.param_groups:
            # 获取当前组的超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # 遍历组内的每一个参数 p
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad  # 获取当前参数的梯度 g [cite: 921]

                # 获取或初始化该参数的状态
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0  # 时间步 t
                    state['exp_avg'] = torch.zeros_like(p)      # 一阶矩 m
                    state['exp_avg_sq'] = torch.zeros_like(p)   # 二阶矩 v

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                t = state['step']

                # m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * (math.sqrt(bias_correction2) / bias_correction1)
                
                # 计算分母 sqrt(v_t) + epsilon
                denom = exp_avg_sq.sqrt().add_(eps)

                # theta_t <- theta_{t-1} - alpha_t * m_t / (sqrt(v_t) + eps)
                p.addcdiv_(exp_avg, denom, value=-alpha_t)

                # theta_t <- theta_t - alpha * lambda * theta_{t-1}
                # 注意：这里使用的是原始学习率 lr (alpha)，而不是 alpha_t
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)
        
        return loss
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    根据给定的参数和当前迭代步数，计算学习率 [cite: 970]。
    该调度器包含三个阶段：线性预热、余弦退火和退火后 。

    Args:
        it (int): 当前的迭代步数 t。
        max_learning_rate (float): 最大学习率 alpha_max。
        min_learning_rate (float): 最小学习率 alpha_min。
        warmup_iters (int): 线性预热的步数 T_w。
        cosine_cycle_iters (int): 余弦退火周期的总步数 T_c。

    Returns:
        float: 当前迭代步数应该使用的学习率。
    """
    # 阶段 1: 线性预热 (Warm-up)
    # 如果当前步数小于预热步数，则学习率线性增加 。
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # 阶段 3: 退火后 (Post-annealing)
    # 如果当前步数大于余弦退火周期，则学习率保持在最小值 。
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 阶段 2: 余弦退火 (Cosine Annealing)
    # 如果当前步数在预热和退火结束之间，学习率按余弦曲线衰减 。
    
    # 首先，计算在余弦周期内的进度 (从 0 到 1)
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    # 接着，根据进度计算余弦衰减的比例 (从 1 到 0)
    # 公式为 0.5 * (1 + cos(progress * pi))
    decay_ratio = 0.5 * (1 + math.cos(math.pi * progress))
    
    # 最后，将衰减比例应用到学习率的变化范围上
    lr_range = max_learning_rate - min_learning_rate
    
    return min_learning_rate + lr_range * decay_ratio

def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> torch.Tensor:
    # 过滤掉没有梯度的参数
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.)

    # 设置一个小的 epsilon 以保证数值稳定性
    eps = 1e-6  # PyTorch 默认值 [cite: 981]

    # 计算所有梯度的总 L2 范数
    total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g.detach(), ord=norm_type) for g in grads]), ord=norm_type)

    # 计算裁剪比例
    clip_coef = max_norm / (total_norm + eps)

    # 如果总范数超过了最大值，则按比例缩小梯度
    if clip_coef < 1:
        for g in grads:
            g.detach().mul_(clip_coef.to(g.device))

    # return total_norm

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集中采样语言建模输入序列及其对应的标签。

    Args:
        dataset (np.array): 1D numpy 数组，包含数据集中的整数 token ID。
        batch_size (int): 期望的批处理大小。
        context_length (int): 每个采样示例的上下文长度。
        device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'）。

    Returns:
        Tuple of torch.LongTensors，形状为 (batch_size, context_length)。第一个元组项是采样的输入序列，
        第二个元组项是对应的语言建模标签。
    """
    # 随机选择起始位置
    start_indices = np.random.randint(0, dataset.shape[0] - context_length, size=batch_size)
    input_sequences = [
        dataset[start_idx:start_idx + context_length] for start_idx in start_indices
    ]
    labels = [
        dataset[start_idx + 1:start_idx + context_length + 1] for start_idx in start_indices
    ]

    # 将输入序列和标签转换为 PyTorch 张量并移动到指定设备
    input_tensor = from_numpy(np.array(input_sequences, dtype=np.int64)).to(device)
    label_tensor = from_numpy(np.array(labels, dtype=np.int64)).to(device)

    return input_tensor, label_tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    将模型、优化器和迭代次数序列化并保存到磁盘。

    Args:
        model (torch.nn.Module): 要序列化的模型。
        optimizer (Optimizer): 要序列化的优化器。
        iteration (int): 已完成的训练迭代次数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 保存路径或文件对象。
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(state, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    从磁盘加载模型和优化器的状态。

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 源路径或文件对象。
        model (torch.nn.Module): 要加载状态的模型。
        optimizer (Optimizer): 要加载状态的优化器。

    Returns:
        int: 加载的迭代次数。
    """
    state = torch.load(src, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return state['iteration']