from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor

from cs336_basics.tokenizer import train_bpe
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import (
    Linear,
    Embedding,
    RMSNorm,
    SwiGLU,
    scaled_dot_product_attention,
    MultiheadSelfAttention,
    RoPE,
    TransformerBlock,
    TransformerLM,
    softmax,
)
from cs336_basics.train import cross_entropy_loss, AdamW, get_lr_cosine_schedule, clip_grad_norm, get_batch, save_checkpoint, load_checkpoint

def run_linear(d_in, d_out, weights, in_features):
    linear = Linear(d_in, d_out)
    linear.W.data = weights
    return linear(in_features)


def run_embedding(vocab_size, d_model, weights, token_ids):
    embedding = Embedding(vocab_size, d_model)
    embedding.weight.data = weights
    return embedding(token_ids)


def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    swiglu = SwiGLU(d_model, d_ff)
    swiglu.w1.W.data = w1_weight
    swiglu.w2.W.data = w2_weight
    swiglu.w3.W.data = w3_weight
    return swiglu(in_features)


def run_scaled_dot_product_attention(Q, K, V, mask):
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features):
    module = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, rope=None)
    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    state_dict = {
        "qkv_proj.W": qkv_weight,
        "o_proj.W": o_proj_weight,
    }
    module.load_state_dict(state_dict)
    module.to(in_features.device)
    return module(in_features, mask=None, token_positions=None)


def run_multihead_self_attention_with_rope(
    d_model, num_heads, max_seq_len, theta,
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
    in_features, token_positions=None,
):
    d_k = d_model // num_heads
    rope_module = RoPE(d_k=d_k, max_seq_len=max_seq_len, theta=theta)
    mha_module = MultiheadSelfAttention(
        d_model=d_model, num_heads=num_heads, rope=rope_module
    )
    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    state_dict_to_load = {
        "qkv_proj.W": qkv_weight,
        "o_proj.W": o_proj_weight,
    }
    mha_module.load_state_dict(state_dict_to_load, strict=False)
    mha_module.to(in_features.device)
    return mha_module(in_features, mask=None, token_positions=token_positions)


def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    module = RoPE(d_k=d_k, max_seq_len=max_seq_len, theta=theta)
    module.to(in_query_or_key.device)
    return module(in_query_or_key, token_positions=token_positions)


def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    rope = RoPE(d_k=d_model // num_heads, max_seq_len=max_seq_len, theta=theta)
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope)
    qkv_weight = torch.cat(
        [weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"]], dim=0
    )
    state_dict_to_load = {
        "norm1.weight": weights["ln1.weight"],
        "attention.qkv_proj.W": qkv_weight,
        "attention.o_proj.W": weights["attn.output_proj.weight"],
        "norm2.weight": weights["ln2.weight"],
        "ffn.w1.W": weights["ffn.w1.weight"],
        "ffn.w2.W": weights["ffn.w2.weight"],
        "ffn.w3.W": weights["ffn.w3.weight"],
    }
    block.load_state_dict(state_dict_to_load, strict=False)
    block.to(in_features.device)
    return block(in_features, mask=None, token_positions=None)


def run_transformer_lm(
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
    weights,
    in_indices,
):
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=context_length,
        rope_theta=rope_theta,
    )
    state_dict_to_load = {
        "embedding.weight": weights["token_embeddings.weight"],
        "norm_final.weight": weights["ln_final.weight"],
        "lm_head.W": weights["lm_head.weight"],
    }
    for i in range(num_layers):
        qkv_weight = torch.cat(
            [
                weights[f"layers.{i}.attn.q_proj.weight"],
                weights[f"layers.{i}.attn.k_proj.weight"],
                weights[f"layers.{i}.attn.v_proj.weight"],
            ],
            dim=0,
        )
        state_dict_to_load[f"blocks.{i}.attention.qkv_proj.W"] = qkv_weight
        state_dict_to_load[f"blocks.{i}.attention.o_proj.W"] = weights[f"layers.{i}.attn.output_proj.weight"]
        state_dict_to_load[f"blocks.{i}.norm1.weight"] = weights[f"layers.{i}.ln1.weight"]
        state_dict_to_load[f"blocks.{i}.norm2.weight"] = weights[f"layers.{i}.ln2.weight"]
        state_dict_to_load[f"blocks.{i}.ffn.w1.W"] = weights[f"layers.{i}.ffn.w1.weight"]
        state_dict_to_load[f"blocks.{i}.ffn.w2.W"] = weights[f"layers.{i}.ffn.w2.weight"]
        state_dict_to_load[f"blocks.{i}.ffn.w3.W"] = weights[f"layers.{i}.ffn.w3.weight"]
    model.load_state_dict(state_dict_to_load, strict=False)
    model.to(in_indices.device)
    return model(in_indices, token_positions=None)


def run_rmsnorm(d_model, eps, weights, in_features):
    module = RMSNorm(d_model=d_model, eps=eps)
    module.load_state_dict({"weight": weights})
    module.to(in_features.device)
    return module(in_features)


def run_silu(in_features):
    # SiLU(x) = x * sigmoid(x)
    return in_features * torch.sigmoid(in_features)


def run_get_batch(dataset, batch_size, context_length, device):
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features, dim):
    return softmax(in_features, dim=dim)


def run_cross_entropy(inputs, targets):
    return cross_entropy_loss(inputs, targets)


def run_gradient_clipping(parameters, max_l2_norm):
    clip_grad_norm(parameters, max_l2_norm)


def get_adamw_cls():
    return AdamW


def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    return load_checkpoint(src, model, optimizer)

def get_tokenizer(vocab, merges, special_tokens):
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    return train_bpe(input_path, vocab_size, special_tokens)