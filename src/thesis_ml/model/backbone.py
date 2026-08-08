"""Bidirectional RoPE transformer backbone using PyTorch SDPA."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class FrozenInputKV:
    """Everything the canvas pass needs so the input pass never has to rerun.

    Produced by :meth:`BidirectionalTransformer.forward` when the
    ``frozen_input_kv`` ablation is enabled and ``return_cached_input_kv=True``,
    and accepted back by the same method as ``cached_input_kv=`` so a caller
    that keeps the input region fixed (iterative denoising, where only the
    canvas changes between steps) pays for the input region exactly once.

    Fields:
        layer_kv: one ``(key, value)`` pair per transformer block, in block
            order. Each tensor is ``[batch, heads, input_len, head_dim]`` and is
            already RoPE-rotated and already QK-normed, which is precisely why
            the canvas pass can concatenate it onto its own live keys without
            needing a second set of key position ids.
        hidden: the input region's own pre-``final_norm`` hidden states,
            ``[batch, input_len, d_model]``. Retained so a cached forward can
            still emit a full ``[input | canvas]`` output rather than padding
            the input half with zeros, which keeps every ``logits[:, input_len:]``
            slice downstream valid and keeps cached and uncached forwards
            numerically identical.
    """

    layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    hidden: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6, *, scale: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model)) if scale else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized if self.weight is None else normalized * self.weight


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

    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the fixed RoPE frequencies at a set of token positions.

        Parameters:
            seq_len: number of positions to generate when ``position_ids`` is
                omitted. Ignored (and may be any value) when ``position_ids`` is
                supplied, because the explicit ids then define both the count
                and the values.
            device: device to build the position/phase tensors on.
            dtype: dtype of the returned cos/sin (matches the activations).
            position_ids: optional explicit ``[batch, seq_len]`` long tensor of
                per-example token positions. Used by the
                ``per_segment_positions`` ablation, where the input region and
                the canvas region each count from zero, so positions differ
                between batch rows and can no longer be a shared ``arange``.

        Returns:
            ``(cos, sin)``. Shape ``[seq_len, head_dim]`` when ``position_ids``
            is None (the historical, batch-independent path) and
            ``[batch, seq_len, head_dim]`` when it is supplied. ``apply_rope``
            dispatches on exactly that rank difference.

        Calls: nothing beyond torch primitives; reads the ``inv_freq`` buffer
        built in ``__init__``.
        """

        if position_ids is None:
            # Untouched baseline path. Kept verbatim (torch.arange + torch.outer)
            # so a toggles-off model executes exactly the tensor ops it always has.
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device=device))
        else:
            positions = position_ids.to(device=device, dtype=self.inv_freq.dtype)
            # torch.outer only accepts 1-D inputs, so the batched path needs an
            # explicit broadcast multiply. This was verified bitwise-identical to
            # torch.outer on the 1-D case, so the two branches do not diverge
            # numerically -- they differ only in rank.
            freqs = positions[..., :, None] * self.inv_freq.to(device=device)
        # Llama uses split-half rotation, so each half receives the same phase.
        phases = torch.cat((freqs, freqs), dim=-1)
        cos = phases.cos().to(dtype=dtype)
        sin = phases.sin().to(dtype=dtype)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``[batch, heads, seq_len, head_dim]`` activations by RoPE phases.

    Parameters:
        x: query or key tensor, ``[batch, heads, seq_len, head_dim]``.
        cos, sin: phases from ``RotaryEmbedding.forward``. Either
            ``[seq_len, head_dim]`` (positions shared by every batch row) or
            ``[batch, seq_len, head_dim]`` (per-example positions).

    Returns:
        The rotated tensor, same shape as ``x``.

    Calls: nothing. Dispatches purely on the rank of ``cos``.
    """

    if cos.dim() == 2:
        # Shared-position path: broadcast one position table over batch and heads.
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        # Per-example path: batch is already axis 0, so only the head axis is new.
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Bidirectional multi-head self-attention with optional prepended keys.

        Parameters:
            x: ``[batch, query_len, d_model]`` activations. Under the
                ``frozen_input_kv`` ablation this is only the canvas region on
                the second pass, so ``query_len`` is shorter than the key axis.
            attention_mask: boolean key mask, ``True`` where a key participates.
                Its width must match the FULL key axis, i.e.
                ``cached_len + query_len`` when ``cached_kv`` is supplied.
            position_ids: optional ``[batch, query_len]`` explicit RoPE
                positions, forwarded to ``self.rope``.
            cached_kv: optional ``(key, value)``, each
                ``[batch, heads, cached_len, head_dim]``, already RoPE-rotated
                and already QK-normed, prepended to this call's own keys and
                values. Because the cache is stored POST-RoPE, the rotation the
                canvas queries expect is already baked in and no key-side
                position ids are needed here.
            return_kv: when True, also return this call's own (uncached,
                post-RoPE, post-QK-norm) ``(key, value)`` so a caller can build
                a cache from an input-only pass.

        Returns:
            ``[batch, query_len, d_model]``, or ``(output, (key, value))`` when
            ``return_kv`` is True.

        Calls: ``self.qkv``, ``self.q_norm``/``self.k_norm``, ``self.rope``,
        module-level ``apply_rope``, ``F.scaled_dot_product_attention``,
        ``self.out``.
        """

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
        cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype, position_ids=position_ids)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Snapshot BEFORE any cache concatenation, so a cache built here holds
        # exactly this region's own keys/values and never a growing prefix.
        returned_kv = (k, v) if return_kv else None
        if cached_kv is not None:
            # Tensors are [batch, heads, seq, head_dim] after the transposes
            # above, so the key/value axis to grow is dim 2, not dim 1.
            k = torch.cat([cached_kv[0], k], dim=2)
            v = torch.cat([cached_kv[1], v], dim=2)

        attn_mask = None
        if attention_mask is not None:
            # SDPA bool mask uses True for keys that participate in attention.
            attn_mask = attention_mask[:, None, None, :].to(torch.bool)
        # CUDA must use a fused linear-memory kernel. Excluding MATH turns an
        # unsupported dtype/shape/mask into an immediate error instead of an
        # O(seq^2) allocation and Windows shared-memory spillover. CPU retains
        # its only available implementation for unit tests and diagnostics.
        #
        # Backend ORDER IS THE PREFERENCE ORDER (`set_priority=True`):
        # FlashAttention first because it is the fastest and most
        # memory-efficient option, then memory-efficient attention as the
        # fallback. Both are fused and linear-memory, so either is acceptable;
        # this is a performance preference, not a correctness one.
        #
        # Do not read the presence of FLASH_ATTENTION here as a promise that it
        # runs. Measured 2026-08-07 on this project's target box (RTX 3070,
        # sm_86, torch 2.12.1+cu130, Windows): the Windows wheel is built
        # WITHOUT any FlashAttention kernel, so every call silently falls back
        # to EFFICIENT_ATTENTION -- that is the backend actually serving
        # training and inference today, and it performs fine. PyTorch reports
        # this only as a `UserWarning: Torch was not compiled with flash
        # attention` from sdp_utils.cpp. Note that
        # `torch.backends.cuda.flash_sdp_enabled()` still returns True on such
        # a build: it reports the PREFERENCE, not availability, and is
        # misleading if used to check whether flash will actually run. To test
        # availability, restrict to [SDPBackend.FLASH_ATTENTION] and see
        # whether the call raises "No available kernel".
        #
        # Keeping flash listed first costs nothing and means a future
        # flash-capable build (a Linux cloud trainer, or a rebuilt wheel) picks
        # it up with no code change. Also note FlashAttention rejects a
        # non-None `attn_mask` outright, and this project always passes a
        # padding mask, so masked calls route to EFFICIENT_ATTENTION even where
        # flash IS compiled in.
        kernel_context = (
            sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
                set_priority=True,
            )
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
        # Reshape against the QUERY length, not the unpacked seq_len. They are
        # equal on every path except a cached second pass, where the key axis is
        # longer than the query axis and seq_len would silently mis-view.
        attended = attended.transpose(1, 2).contiguous().view(batch, q.shape[2], d_model)
        output = self.out(attended)
        if returned_kv is not None:
            return output, returned_kv
        return output


class GeGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


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
        self.pre_attn_norm = RMSNorm(d_model)
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
        self.post_attn_norm = RMSNorm(d_model)
        self.pre_ffn_norm = RMSNorm(d_model)
        self.ffn = GeGLU(d_model, ffn_dim)
        self.post_ffn_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One sandwich-norm attention + GeGLU block.

        Parameters:
            x: ``[batch, query_len, d_model]`` block input.
            attention_mask: boolean key mask spanning the full key axis; see
                ``MultiHeadSelfAttention.forward``.
            position_ids: optional explicit per-example RoPE positions for the
                queries in ``x``.
            cached_kv: optional frozen ``(key, value)`` prepended to this
                block's keys/values inside its attention module.
            return_kv: when True, also return this block's own attention
                ``(key, value)``.

        Returns:
            ``[batch, query_len, d_model]``, or ``(output, (key, value))`` when
            ``return_kv`` is True.

        Calls: ``self.pre_attn_norm``, ``self.attn``, ``self.post_attn_norm``,
        ``self.pre_ffn_norm``, ``self.ffn``, ``self.post_ffn_norm``.
        """

        attention_output = self.attn(
            self.pre_attn_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            cached_kv=cached_kv,
            return_kv=return_kv,
        )
        if return_kv:
            attention_branch, layer_kv = attention_output
        else:
            attention_branch = attention_output
        x = x + self.post_attn_norm(attention_branch)
        ffn_branch = self.ffn(self.pre_ffn_norm(x))
        x = x + self.post_ffn_norm(ffn_branch)
        if return_kv:
            return x, layer_kv
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
        frozen_input_kv: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        # Ablation toggle. When on (and the caller tells us where the canvas
        # starts via `input_len`), the single joint forward becomes two passes:
        # the input region alone, then the canvas attending to the input's
        # cached keys/values. Off by default so the baseline is untouched.
        self.frozen_input_kv = frozen_input_kv
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        input_len: int | None = None,
        cached_input_kv: FrozenInputKV | None = None,
        return_cached_input_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, FrozenInputKV]:
        """Run every block over the concatenated ``[input | canvas]`` sequence.

        Parameters:
            x: ``[batch, input_len + canvas_len, d_model]`` embeddings.
            attention_mask: boolean key mask over the same full width, ``True``
                where a key participates.
            position_ids: optional ``[batch, input_len + canvas_len]`` explicit
                RoPE positions. ``None`` means the historical shared
                ``arange(seq_len)``.
            input_len: index at which the canvas begins. Required to take the
                frozen-KV path; used only to slice ``x``, ``attention_mask``
                and ``position_ids`` into their two regions.
            cached_input_kv: a ``FrozenInputKV`` from an earlier forward over
                the SAME input region. Supplying it skips the input pass
                entirely, which is the inference win: the input region is
                identical across every denoising step.
            return_cached_input_kv: when True, also return the ``FrozenInputKV``
                the input pass produced so a caller can feed it back later.

        Returns:
            ``[batch, input_len + canvas_len, d_model]`` -- ALWAYS full length,
            because every downstream consumer slices the canvas out by index
            with ``logits[:, input_len:, :]`` and a short tensor would silently
            misalign rather than raise. Returns ``(hidden, FrozenInputKV)``
            when ``return_cached_input_kv`` is True.

        Raises:
            ValueError: if a cache is requested or supplied while the
                ``frozen_input_kv`` ablation is off or ``input_len`` is missing,
                since no cache exists on the joint path.

        Calls: each ``TransformerBlock`` in ``self.layers``, ``self.final_norm``,
        and ``torch.utils.checkpoint.checkpoint`` when activation checkpointing
        is enabled and the module is training.
        """

        use_checkpoint = self.gradient_checkpointing and self.training
        # `input_len == 0` (canvas-only forwards) has no input region to freeze,
        # so it falls back to the joint path where it is trivially correct.
        frozen_path = self.frozen_input_kv and input_len is not None and input_len > 0

        if not frozen_path:
            if cached_input_kv is not None or return_cached_input_kv:
                raise ValueError(
                    "frozen input KV cache requires frozen_input_kv=True and a positive input_len"
                )
            # ---- unchanged joint path; stays bit-identical to the baseline ----
            for layer in self.layers:
                if use_checkpoint:
                    x = checkpoint(
                        lambda hidden, block=layer: block(
                            hidden,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        ),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = layer(x, attention_mask=attention_mask, position_ids=position_ids)
            return self.final_norm(x)

        # ---- frozen-KV: split the joint forward into two passes ----
        # Slicing recovers the two region masks exactly, because the caller
        # concatenated them in [input | canvas] order in the first place.
        input_mask = None if attention_mask is None else attention_mask[:, :input_len]
        input_positions = None if position_ids is None else position_ids[:, :input_len]
        canvas_positions = None if position_ids is None else position_ids[:, input_len:]

        if cached_input_kv is not None:
            # Reuse path: the input region is unchanged from a previous forward,
            # so both its hidden states and its per-layer K/V are already known.
            if len(cached_input_kv.layer_kv) != len(self.layers):
                raise ValueError(
                    "cached input KV has "
                    f"{len(cached_input_kv.layer_kv)} layers but the backbone has {len(self.layers)}"
                )
            input_hidden = cached_input_kv.hidden
            layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...] = cached_input_kv.layer_kv
        else:
            # Pass 1: the input region alone, attending only to itself.
            input_hidden = x[:, :input_len, :]
            collected_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer in self.layers:
                if use_checkpoint:
                    # `block=layer` guards late binding exactly as on the joint
                    # path. K/V are RETURNED, never appended from inside the
                    # wrapped body: checkpoint re-executes that body during
                    # backward, so a side-channel append would fire twice and
                    # the first-run tensors are the discarded ones.
                    input_hidden, layer_kv = checkpoint(
                        lambda hidden, block=layer: block(
                            hidden,
                            attention_mask=input_mask,
                            position_ids=input_positions,
                            return_kv=True,
                        ),
                        input_hidden,
                        use_reentrant=False,
                    )
                else:
                    input_hidden, layer_kv = layer(
                        input_hidden,
                        attention_mask=input_mask,
                        position_ids=input_positions,
                        return_kv=True,
                    )
                collected_kv.append(layer_kv)
            layer_caches = tuple(collected_kv)

        # Pass 2: the canvas region, attending to concat(cached_input_K, canvas_K).
        # The mask passed here is the FULL-WIDTH combined mask, because that is
        # the width of the concatenated key axis; SDPA broadcasts it over the
        # shorter query axis, so no query dimension is needed.
        canvas_hidden = x[:, input_len:, :]
        for index, layer in enumerate(self.layers):
            if use_checkpoint:
                # `cache=layer_caches[index]` MUST be a default argument for the
                # same late-binding reason as `block=layer`; read from the
                # enclosing scope it would hand every layer the last cache.
                canvas_hidden = checkpoint(
                    lambda hidden, block=layer, cache=layer_caches[index]: block(
                        hidden,
                        attention_mask=attention_mask,
                        position_ids=canvas_positions,
                        cached_kv=cache,
                    ),
                    canvas_hidden,
                    use_reentrant=False,
                )
            else:
                canvas_hidden = layer(
                    canvas_hidden,
                    attention_mask=attention_mask,
                    position_ids=canvas_positions,
                    cached_kv=layer_caches[index],
                )

        # Full-length output is a hard requirement, not a convenience: the input
        # half carries the input-only-attention representations, and every
        # downstream consumer slices by index and needs the canvas to start at
        # column input_len.
        hidden = self.final_norm(torch.cat([input_hidden, canvas_hidden], dim=1))
        if return_cached_input_kv:
            return hidden, FrozenInputKV(layer_kv=layer_caches, hidden=input_hidden)
        return hidden
