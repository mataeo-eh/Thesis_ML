"""Bidirectional RoPE transformer backbone using PyTorch SDPA."""

from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Llama 3.1-style frequency-scaled rotary embeddings.

    The configured frequencies are fixed at initialization and can be evaluated
    at arbitrary sequence lengths. This is the Llama 3.1 scaled-RoPE variant,
    which is distinct from the separately named YaRN algorithm.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        base: float = 500_000.0,
        scaling_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_context: int = 8192,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        if base <= 0:
            raise ValueError("RoPE base must be positive")
        if scaling_factor < 1.0:
            raise ValueError("RoPE scaling_factor must be at least 1")
        if low_freq_factor <= 0 or high_freq_factor <= low_freq_factor:
            raise ValueError("RoPE frequency factors must satisfy 0 < low < high")
        if original_context <= 0:
            raise ValueError("RoPE original_context must be positive")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        wavelengths = 2 * math.pi / inv_freq
        low_freq_wavelength = original_context / low_freq_factor
        high_freq_wavelength = original_context / high_freq_factor

        scaled_inv_freq = torch.where(wavelengths > low_freq_wavelength, inv_freq / scaling_factor, inv_freq)
        smooth = (original_context / wavelengths - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth) * scaled_inv_freq / scaling_factor + smooth * scaled_inv_freq
        medium_frequency = (wavelengths >= high_freq_wavelength) & (wavelengths <= low_freq_wavelength)
        scaled_inv_freq = torch.where(medium_frequency, smoothed_inv_freq, scaled_inv_freq)

        self.register_buffer("inv_freq", scaled_inv_freq, persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        # Llama uses split-half rotation, so each half receives the same phase.
        phases = torch.cat((freqs, freqs), dim=-1)
        cos = phases.cos().to(dtype=dtype)
        sin = phases.sin().to(dtype=dtype)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    first_half, second_half = x.chunk(2, dim=-1)
    rotated_half = torch.cat((-second_half, first_half), dim=-1)
    return x * cos + rotated_half * sin


class MultiHeadSelfAttention(nn.Module):
    """Vanilla MHA, not GQA."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float = 0.0,
        *,
        qk_norm: bool = True,
        rope_theta: float = 500_000.0,
        rope_scaling_factor: float = 8.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 4.0,
        rope_original_context: int = 8192,
    ) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qk_norm = qk_norm
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(
            self.head_dim,
            base=rope_theta,
            scaling_factor=rope_scaling_factor,
            low_freq_factor=rope_low_freq_factor,
            high_freq_factor=rope_high_freq_factor,
            original_context=rope_original_context,
        )
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.qk_norm:
            # QK-norm can flatten attention; keep copy-class loss visible when toggling it.
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_mask = None
        if attention_mask is not None:
            # SDPA bool mask uses True for keys that participate in attention.
            attn_mask = attention_mask[:, None, None, :].to(torch.bool)
        # CUDA must use a fused linear-memory kernel. Excluding MATH turns an
        # unsupported dtype/shape/mask into an immediate error instead of an
        # O(seq^2) allocation and Windows shared-memory spillover. CPU retains
        # its only available implementation for unit tests and diagnostics.
        kernel_context = (
            sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
            if q.is_cuda
            else nullcontext()
        )
        with kernel_context:
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out(attended)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        *,
        qk_norm: bool = True,
        rope_theta: float = 500_000.0,
        rope_scaling_factor: float = 8.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 4.0,
        rope_original_context: int = 8192,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model,
            heads,
            dropout=dropout,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rope_low_freq_factor=rope_low_freq_factor,
            rope_high_freq_factor=rope_high_freq_factor,
            rope_original_context=rope_original_context,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        *,
        qk_norm: bool = True,
        rope_theta: float = 500_000.0,
        rope_scaling_factor: float = 8.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 4.0,
        rope_original_context: int = 8192,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    heads,
                    ffn_dim,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    rope_theta=rope_theta,
                    rope_scaling_factor=rope_scaling_factor,
                    rope_low_freq_factor=rope_low_freq_factor,
                    rope_high_freq_factor=rope_high_freq_factor,
                    rope_original_context=rope_original_context,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda hidden, block=layer: block(hidden, attention_mask=attention_mask),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, attention_mask=attention_mask)
        return self.final_norm(x)
