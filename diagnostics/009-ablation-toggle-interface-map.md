# 009 — Ablation Toggle Interface Map

Read-only investigation. **No source was modified.** Every code change described here is a
proposal for downstream implementation workers, not an applied edit.

## Scope and conditions

- Repository: `Thesis_ML` (submodule), branch `main`, commit `a6297a6`.
- Interpreter: `.venv/Scripts/python.exe` → `torch 2.12.1+cu130`, CUDA 13.0, `cuda.is_available() == True`.
- GPU used for the SDPA/autograd probes: **NVIDIA GeForce RTX 3070** (sm_86), Windows 11.
- Target profile: `configs/local_overfit_v2.yaml` (extends `configs/local_overfit.yaml` →
  `config/default.yaml`).
- Diagnosis being implemented against is taken as given (RoPE is the only positional signal;
  left padding is load-bearing; nothing marks the input/canvas boundary). It was not re-derived.

The three toggles under design, all defaulting to `false`, all-off required to be **bit-identical**
to today:

1. `model.frozen_input_kv`
2. `model.segment_embeddings`
3. `model.per_segment_positions`

---

## 1. Signature chain

### 1.1 Current chain, verbatim

**`src/thesis_ml/model/model.py:72-81`**

```python
    def forward(
        self,
        *,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        input_attention_mask: torch.Tensor | None = None,
        canvas_attention_mask: torch.Tensor | None = None,
        input_features: InputFeatures | None = None,
        canvas_self_conditioning: torch.Tensor | None = None,
    ) -> ModelOutput:
```

Its single backbone call, **`src/thesis_ml/model/model.py:94`**:

```python
        hidden = self.backbone(embeddings, attention_mask=attention_mask)
```

**`src/thesis_ml/model/backbone.py:257`** — `BidirectionalTransformer.forward`

```python
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
```

**`src/thesis_ml/model/backbone.py:211`** — `TransformerBlock.forward`

```python
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
```

**`src/thesis_ml/model/backbone.py:123`** — `MultiHeadSelfAttention.forward`

```python
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
```

Two more signatures in the same chain matter because both toggles touch them:

**`src/thesis_ml/model/backbone.py:69`** — `RotaryEmbedding.forward`

```python
    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
```

**`src/thesis_ml/model/backbone.py:79-84`** — `apply_rope`

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    first_half, second_half = x.chunk(2, dim=-1)
    rotated_half = torch.cat((-second_half, first_half), dim=-1)
    return x * cos + rotated_half * sin
```

Both are monkeypatched by tests (`tests/test_model.py:427`, `:443`, `:450`, `:474`), so their
existing parameter names and call forms must be preserved, not replaced.

### 1.2 Minimal new parameters

Total: **two** new optional parameters on `BidirectionalTransformer.forward`, **three** on
`TransformerBlock.forward` and `MultiHeadSelfAttention.forward`, **one** on
`RotaryEmbedding.forward`, and — this is the important one — **zero** on
`SC2StrategyDiffusionModel.forward`.

#### `SC2StrategyDiffusionModel.forward` — NO signature change

Per-example input lengths do **not** need to be threaded in. `input_lengths` is exactly
recoverable from `input_attention_mask`, by construction:

`src/thesis_ml/data/collate.py:119-121`

```python
        input_token_ids[row, max_input_len - length :] = example.input_token_ids
        input_attention_mask[row, max_input_len - length :] = True
        input_lengths[row] = length
```

so `input_attention_mask.sum(dim=1) == input_lengths` identically. Inside `forward`, after
`_combine_attention_masks` has already substituted `torch.ones_like(...)` for a `None` input mask
(`model.py:119-122`), the derivation is total:

```python
        # Per-example count of real (non batch-shape-padded) input tokens.
        # Derived from the mask rather than taken as a parameter so that the
        # sampler, the eval harness, and every existing test keep working
        # without touching a single call site. Equal to batch.input_lengths by
        # construction (see data/collate.py:119-121).
        input_lengths = attention_mask[:, :input_len].sum(dim=1)
```

Take this route. The explicit-parameter alternative
(`input_lengths: torch.Tensor | None = None`) costs a `.to(active_device)` addition at
`inference/sampler.py:124`, `:217`, and `:353` (the sampler moves only `input_token_ids`,
`input_attention_mask`, `canvas_attention_mask`, and `input_features` to the device —
`sampler.py:183-186` — it never moves `batch.input_lengths`), plus edits to
`tests/test_model.py`, `tests/test_dataset.py`, `tests/test_windowing.py`, and
`tests/test_pipeline_hardening.py`. Deriving avoids all of it.

`model.py:forward` does gain **local variables** (not parameters):

```python
        input_len = input_token_ids.shape[1]      # NEW local; see §6.2
        position_ids = None                       # [B, input_len + canvas_len] when toggle 3 is on
```

and its backbone call becomes:

```python
        hidden = self.backbone(
            embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_len=input_len,
        )
```

Passing `input_len` unconditionally is safe: `BidirectionalTransformer` ignores it unless its
constructor flag `frozen_input_kv` is set.

#### `BidirectionalTransformer.forward` — 2 new parameters

```python
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        input_len: int | None = None,
    ) -> torch.Tensor:
```

- `position_ids`: `torch.Tensor | None`, default `None`. Shape `[B, S]`, dtype `torch.long`,
  where `S == x.shape[1] == input_len + canvas_len`. `None` means "use `torch.arange(S)`",
  which is today's behavior exactly. Placed **after** `attention_mask` and behind a bare `*`
  so nothing can be passed positionally by accident.
- `input_len`: `int | None`, default `None`. The width of the input region, i.e. the index at
  which the canvas starts. Used only to slice `position_ids` and `attention_mask` into their two
  regions for the frozen-KV two-pass path. `None` (or `self.frozen_input_kv == False`) keeps the
  single joint pass.

Both region masks are recovered by slicing, so no extra mask parameters are needed —
`_combine_attention_masks` (`model.py:113-123`) concatenates in exactly `[input | canvas]` order:

```python
        input_mask = None if attention_mask is None else attention_mask[:, :input_len]
        canvas_mask = None if attention_mask is None else attention_mask[:, input_len:]
```

`frozen_input_kv: bool = False` goes on the **constructor** (`backbone.py:220-235`), alongside the
existing `gradient_checkpointing: bool = False` at `backbone.py:234`, and is wired from
`model.py:52-65` as `frozen_input_kv=model_config.frozen_input_kv`.

#### `TransformerBlock.forward` — 3 new parameters

```python
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
```

- `position_ids`: `torch.Tensor | None = None`, `[B, x.shape[1]]` long — positions of **this
  block's own tokens** (queries and its own keys).
- `cached_kv`: `tuple[torch.Tensor, torch.Tensor] | None = None` — this layer's frozen input
  `(K, V)`, each `[B, heads, input_len, head_dim]`, **already RoPE-rotated and already
  QK-normed**. Caching post-RoPE keys is what keeps the position contract from needing a second
  set of key position ids.
- `return_kv`: `bool = False` — when `True` the block additionally returns its own post-RoPE
  `(K, V)` so pass 1 can capture them.

When `return_kv=True` the return is `(hidden, (k, v))`. Do **not** make the return type vary in
any other way — a checkpointed function with a conditionally-shaped return is a debugging trap.

#### `MultiHeadSelfAttention.forward` — the same 3 parameters

```python
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
```

Body changes, in order:

1. `cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype, position_ids=position_ids)`
   (line 134).
2. `q = apply_rope(q, cos, sin)` / `k = apply_rope(k, cos, sin)` unchanged (lines 135-136).
3. Immediately after RoPE, capture `k_out, v_out = k, v` if `return_kv`.
4. If `cached_kv is not None`: `k = torch.cat([cached_kv[0], k], dim=2)` and
   `v = torch.cat([cached_kv[1], v], dim=2)`. **Key axis is dim 2** — tensors are `[B, H, S, D]`
   at this point (transposed at lines 127-129).
5. Mask construction at lines 138-141 is unchanged. See §3.
6. The reshape at line 163 must use `q.shape[2]`, not the `seq_len` unpacked at line 124 — those
   are the same today but differ in pass 2. Concretely:
   `attended.transpose(1, 2).contiguous().view(batch, q.shape[2], d_model)`.

`attention_mask` semantics are widened, not changed: it is a **key-only** mask whose width equals
the **key** axis (`input_len + canvas_len` in pass 2), not the query axis. It already broadcasts
over queries, so no code change is needed for that.

#### `RotaryEmbedding.forward` — 1 new parameter

```python
    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate rotary cos/sin at either arange positions or explicit ids.

        Parameters:
            seq_len: sequence length, used only when position_ids is None.
            device / dtype: placement and output dtype (unchanged).
            position_ids: optional [B, S] long tensor of PER-EXAMPLE positions.
                None reproduces the pre-toggle torch.arange(seq_len) behavior
                bit-for-bit.
        Returns:
            (cos, sin), each [S, head_dim] when position_ids is None and
            [B, S, head_dim] otherwise.
        Calls: nothing beyond torch primitives; consumed by apply_rope.
        """

        if position_ids is None:
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        else:
            positions = position_ids.to(device=device, dtype=self.inv_freq.dtype)
        # torch.outer only accepts 1-D inputs, so the batched path needs an
        # explicit broadcast multiply. Verified bitwise-identical to
        # torch.outer on the 1-D path (see diagnostics 009 §1.3), so the
        # toggle-off numerics are unchanged.
        freqs = positions[..., :, None] * self.inv_freq.to(device=device)
```

`apply_rope` gains a rank branch and keeps its `(x, cos, sin)` signature (the monkeypatch at
`tests/test_model.py:429-431` depends on it):

```python
    # cos/sin are [S, D] in the default path and [B, S, D] once per-example
    # position ids are in play; x is always [B, heads, S, D].
    if cos.dim() == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
```

### 1.3 Verified: the `torch.outer` → broadcast swap is bitwise-identical

Measured with the real `RotaryEmbedding(64)` buffer:

```
S=    7 dt=torch.float32  outer==broadcast bitwise: True
S=    7 dt=torch.bfloat16 outer==broadcast bitwise: True
batched freqs shape: (3, 11, 32)
batched cos rank: 3 -> needs cos[:, None, :, :], giving (3, 1, 11, 64) vs x (3, 4, 11, 64)
```

`torch.equal(torch.outer(pos, inv), pos[..., :, None] * inv)` is `True`. The toggle-off path is
therefore preserved exactly.

### 1.4 Per-segment position id construction (toggle 3)

Built once in `SC2StrategyDiffusionModel.forward`, never inside the backbone:

```python
        # Input real content is LEFT-padded (collate.py:119-121), so example i's
        # real tokens occupy slots [input_len - L_i, input_len). Subtracting that
        # offset gives 0..L_i-1 on the real slots. Padded slots go negative and
        # are clamped to 0; they are excluded from attention as keys and their
        # logits are never scored, so their position value is arbitrary — 0 is
        # chosen only so the value stays deterministic and in-range.
        offsets = input_len - input_lengths                     # [B]
        input_positions = (
            torch.arange(input_len, device=device)[None, :] - offsets[:, None]
        ).clamp_min(0)                                          # [B, input_len]
        # The canvas RESTARTS at 0 at canvas index 0 — that is the whole point:
        # it makes "canvas position 0" (the win/loss token) a fixed RoPE phase
        # instead of a phase that drifts with the input length.
        canvas_positions = (
            torch.arange(canvas_len, device=device)[None, :].expand(batch_size, canvas_len)
        )                                                       # [B, canvas_len]
        position_ids = torch.cat([input_positions, canvas_positions], dim=1)
```

---

## 2. Gradient checkpointing

### 2.1 Current code, verbatim

**`src/thesis_ml/model/backbone.py:257-267`**

```python
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
```

`block=layer` is a **late-binding guard**, not an autograd device: without it every lambda would
resolve the free variable `layer` at call time and every checkpoint would execute the *last*
block. `attention_mask` is captured by closure and requires no grad.

Import: `from torch.utils.checkpoint import checkpoint` (`backbone.py:12`), monkeypatched by
`tests/test_model.py:507` as `backbone_module.checkpoint`, so the module-level name must stay.
`configs/local_overfit.yaml:65` sets `gradient_checkpointing: true`, so this path is live on the
target profile.

### 2.2 Do extra tensors have to be explicit positional args? **No.**

This was measured, not assumed. Under `use_reentrant=False`, a closure-captured **non-leaf**
tensor with grad history (exactly the frozen-KV cache situation — pass-1 K/V produced by shared
weights) yields gradients **identical to the non-checkpointed reference**:

```
closure   loss_match=True w1_grad_match=True w2_grad_match=True w1_grad_maxdiff=0.000e+00
explicit  loss_match=True w1_grad_match=True w2_grad_match=True w1_grad_maxdiff=0.000e+00
```

Both routes are correct. The repo already relies on this (`attention_mask` is closure-captured).
Keep `use_reentrant=False` — the classic "closure captures get no gradient" restriction belongs to
the reentrant implementation, and switching to `use_reentrant=True` is not on the table.

Two hazards **are** real and the implementer must respect both:

**(a) Late binding on the cache.** A per-layer cache entry read from inside the lambda must be
bound as a default argument, exactly like `block=layer`. `cache=cached_kv[index]` — not
`cached_kv[index]` read from the enclosing scope.

**(b) The wrapped function IS re-executed during backward, so side-effect capture is broken.**
Measured on a 5-layer chain:

```
calls after forward :  5
calls after backward: 10
```

Pass 1 must therefore **return** its K/V from the block (a tuple return through `checkpoint()` is
supported and verified) and must **never** append them to an outer list from inside the
checkpointed body — that list would be appended a second time during recomputation, and the
first-run tensors are the discarded ones.

A consequence worth stating plainly: because pass 1's K/V become checkpoint **outputs**, they are
retained rather than recomputed. That is inherent to caching them at all, not a defect.

### 2.3 Exact rewritten `checkpoint(...)` calls

```python
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        input_len: int | None = None,
    ) -> torch.Tensor:
        """Run the block stack, optionally as two frozen-KV passes.

        Parameters:
            x: [B, S, d_model] embeddings over the concatenated [input | canvas].
            attention_mask: [B, S] bool key mask, True where a key participates.
            position_ids: optional [B, S] long per-example RoPE positions; None
                keeps the pre-toggle torch.arange(S) behavior.
            input_len: width of the input region, i.e. where the canvas starts.
        Returns:
            [B, S, d_model] — ALWAYS full length, even on the frozen-KV path.
            See diagnostics 009 §4: five call sites slice the canvas off by
            index and would silently take the wrong window otherwise.
        Calls: TransformerBlock.forward (directly or through
            torch.utils.checkpoint.checkpoint) and self.final_norm.
        """

        use_checkpoint = self.gradient_checkpointing and self.training

        if not (self.frozen_input_kv and input_len is not None):
            # ---- unchanged joint path; must stay bit-identical ----
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
        # Slicing recovers the two region masks exactly, because
        # model._combine_attention_masks concatenated them in [input | canvas]
        # order in the first place (model.py:113-123).
        input_mask = None if attention_mask is None else attention_mask[:, :input_len]
        canvas_mask = None if attention_mask is None else attention_mask[:, input_len:]
        input_positions = None if position_ids is None else position_ids[:, :input_len]
        canvas_positions = None if position_ids is None else position_ids[:, input_len:]

        # Pass 1: the input region alone, attending only to itself.
        input_hidden = x[:, :input_len, :]
        cached_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            if use_checkpoint:
                # `block=layer` guards late binding exactly as in the joint path.
                # K/V are RETURNED, never appended from inside the body: the
                # wrapped fn is re-executed during backward (measured 5 forward
                # calls -> 10 total), so a side-channel append would fire twice.
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
            cached_kv.append(layer_kv)

        # Pass 2: the canvas region, attending to concat(cached_input_K, canvas_K).
        canvas_hidden = x[:, input_len:, :]
        for index, layer in enumerate(self.layers):
            if use_checkpoint:
                # `cache=cached_kv[index]` MUST be a default argument for the
                # same late-binding reason as `block=layer`; reading it from the
                # enclosing scope would give every layer the last layer's cache.
                canvas_hidden = checkpoint(
                    lambda hidden, block=layer, cache=cached_kv[index]: block(
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
                    cached_kv=cached_kv[index],
                )

        # Full-length output is a hard requirement, not a convenience. The input
        # half carries the input-only-attention representations; nothing scores
        # or reads it, but every downstream consumer slices by index and needs
        # the canvas to start at column input_len.
        return self.final_norm(torch.cat([input_hidden, canvas_hidden], dim=1))
```

Note that pass 2 passes the **full-width** `attention_mask` (width `input_len + canvas_len`), not
`canvas_mask` — that is the concatenated key axis. See §3.

---

## 3. SDPA masking — DEFINITIVE

### 3.1 Current code, verbatim

**`src/thesis_ml/model/backbone.py:138-141`**

```python
        attn_mask = None
        if attention_mask is not None:
            # SDPA bool mask uses True for keys that participate in attention.
            attn_mask = attention_mask[:, None, None, :].to(torch.bool)
```

**`src/thesis_ml/model/backbone.py:146-162`**

```python
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
```

### 3.2 Required pass-2 mask shape

**`[B, 1, 1, input_len + canvas_len]`, dtype `torch.bool`, `True` = key participates.**

Query axis is `canvas_len`; the mask carries **no** query axis and broadcasts over it. Line 141
already produces exactly this shape given a width-`(input_len + canvas_len)` `attention_mask` —
which is precisely the tensor `_combine_attention_masks` already builds. **No change to the
mask-building code is required.** Pass the full combined mask into pass 2 and leave lines 138-141
alone.

### 3.3 Does `sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION])` tolerate it on CUDA? **YES.**

Measured on this machine and this wheel. Reproduce with a script calling
`F.scaled_dot_product_attention` under `sdpa_kernel([...], set_priority=True)`.

| configuration | dtype | q_len | k_len | mask | result |
|---|---|---|---|---|---|
| FLASH+EFFICIENT (baseline, square) | fp32 / bf16 | 60 | 60 | `[B,1,1,60]` bool | **OK** |
| FLASH+EFFICIENT, non-square | fp32 | 23 | 60 | `[B,1,1,60]` bool | **OK** (fwd+bwd) |
| FLASH+EFFICIENT, non-square | bf16 | 23 | 60 | `[B,1,1,60]` bool | **OK** (fwd+bwd) |
| EFFICIENT only, non-square | fp32 / bf16 | 23 | 60 | `[B,1,1,60]` bool | **OK** |
| FLASH only, non-square | fp32 / bf16 | 23 | 60 | `[B,1,1,60]` bool | FAIL — `No available kernel` |
| FLASH only, non-square | fp32 / bf16 | 23 | 60 | **none** | FAIL — `No available kernel` |
| FLASH only, **square** | bf16 | 60 | 60 | `[B,1,1,60]` bool | FAIL — `No available kernel` |
| FLASH+EFFICIENT, expanded mask `[B,1,C,I+C]` | fp32 / bf16 | 23 | 60 | bool | **OK** |
| FLASH+EFFICIENT, unaligned key lens 61 / 63 / 127 / 1001 | bf16 | 23, 37, 512 | — | `[B,1,1,K]` bool | **OK** (all, fwd+bwd) |
| FLASH+EFFICIENT, realistic scale | bf16 | 2048 | 6144 | `[B,1,1,6144]` bool | **OK** (fwd+bwd) |

Numerics: `EFFICIENT` vs `MATH` reference on the non-square masked case, fp32 —
**max abs diff 1.49e-06**. A fully-`False` key row produces **no NaN**.

### 3.4 The load-bearing discovery

PyTorch emitted, at `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:810`:

```
UserWarning: Torch was not compiled with flash attention.
```

**This `torch 2.12.1+cu130` Windows wheel contains no FlashAttention kernel at all.** Confirmed
independently: restricting to `[SDPBackend.FLASH_ATTENTION]` fails with `No available kernel`
even with a **square** shape and **no mask**. `torch.backends.cuda.flash_sdp_enabled()` returns
`True` — that flag reports the *preference*, not kernel availability, and is misleading here.

Two consequences:

1. Today's production attention **already runs 100% on `EFFICIENT_ATTENTION`**. The
   `SDPBackend.FLASH_ATTENTION` entry at `backbone.py:148` is a no-op on this install. The
   `AGENTS.md` contract ("prefers fused Flash SDPA and falls back only to memory-efficient SDPA
   with a broadcast boolean key mask") is being satisfied by its fallback branch, always.
2. The frozen-KV pass-2 shape lands on the same backend that already serves every training step.
   It is not a new code path for the kernel selector.

Even if the wheel were rebuilt with flash, FlashAttention rejects any non-`None` `attn_mask`
outright, so a masked call would still route to `EFFICIENT_ATTENTION`. The conclusion is robust
across that change.

### 3.5 Decision

**No fallback is needed. Do not add `SDPBackend.MATH`. Do not expand the mask to
`[B, 1, canvas_len, input_len + canvas_len]`.** Keep `backbone.py:138-141` and `:146-162` exactly
as they are; just feed pass 2 the full-width combined mask.

Evidence basis: direct execution on the target GPU (RTX 3070, sm_86) with the installed wheel,
covering fp32 and bf16, forward and backward, aligned and unaligned key lengths, small and
realistic sequence sizes, plus a MATH-reference numeric cross-check.

---

## 4. Hidden-state / logits consumers

### 4.1 `ModelOutput.hidden_states` — zero consumers

A repository-wide grep for `hidden_states` across all `.py` files (excluding `.venv`) returns
exactly two hits, both in the producer:

- `src/thesis_ml/model/model.py:20` — the dataclass field declaration.
- `src/thesis_ml/model/model.py:96` — `return ModelOutput(logits=logits, hidden_states=hidden)`.

Nothing in `train/`, `eval/`, `inference/`, `viz/`, or `tests/` reads it. It is write-only today.
Its shape is nevertheless coupled to `logits` (both come from the same `hidden`), so the
full-length requirement below governs it too.

### 4.2 `ModelOutput.logits` — five direct consumers, all index-based

| file:line | code | assumes full `[input \| canvas]` length? |
|---|---|---|
| `src/thesis_ml/train/loop.py:1123` | `estimate.logits[:, input_len:, :]` | **YES** |
| `src/thesis_ml/train/loop.py:1139` | `canvas_logits = output.logits[:, input_len:, :]` | **YES** |
| `src/thesis_ml/inference/sampler.py:132` | `output.logits[:, input_token_ids.shape[1]:, :]` | **YES** |
| `src/thesis_ml/inference/sampler.py:227` | `raw_canvas_logits = output.logits[:, input_token_ids.shape[1]:, :]` | **YES** |
| `src/thesis_ml/inference/sampler.py:363` | `final_output.logits[:, input_token_ids.shape[1]:, :].detach().cpu()` | **YES** |

`input_len` is defined at `src/thesis_ml/train/loop.py:1105`:

```python
            input_len = batch.input_token_ids.shape[1]
```

**`src/thesis_ml/eval/` — no direct consumers.** `eval/harness.py:143-144` reads
`sampled.final_canvas_logits`, which `inference/sampler.py:165` and `:374` have already sliced to
canvas-only. Indirect dependency only.

**`src/thesis_ml/viz/` — no direct consumers.** `viz/diagnostics.py:1014` reads
`item.result.final_canvas_logits` (`viz/diagnostics.py:1003-1056`, `write_logits_json`), which
comes from the same already-sliced eval path. `viz/diagnostics.py:352` and `:1148` only forward an
`include_canvas_logits` flag. Indirect dependency only.

Tests that assert the full length directly:

- `tests/test_model.py:249` — `output.logits.shape == (batch, input_ids.shape[1] + canvas_ids.shape[1], 128)`
- `tests/test_model.py:820` — same assertion in the RoPE extrapolation smoke test
- `tests/test_model.py:804` — indexes `base[:, first_canvas_index]` where
  `first_canvas_index = input_ids.shape[1]`
- `tests/test_dataset.py:385-387` — `logits.shape[1] == expected_length`
- `tests/test_pipeline_hardening.py:201-204` — `output.shape[:2] == (B, input + canvas)`
- `tests/test_windowing.py:482-500` — slices `batched_output[input_pad : batch.input_token_ids.shape[1]]`,
  which depends on both the full length **and** the left-padding offset

### 4.3 Confirmed: the frozen-KV path CANNOT return a shorter tensor

**Confirmed.** `BidirectionalTransformer.forward` must return `[B, input_len + canvas_len, d_model]`
on the frozen-KV path, and `SC2StrategyDiffusionModel.forward` must keep emitting full-length
logits. Implementation: `torch.cat([input_hidden, canvas_hidden], dim=1)` before `final_norm`
(see §2.3).

**Blast radius if this is gotten wrong — and why it is worse than it looks.** All five consumers
slice with `[:, input_len:, :]`. Python slicing past the end of a dimension does **not** raise; it
returns an empty or truncated tensor. So a canvas-only return produces:

- when `input_len >= canvas_len`: a **zero-width** logits tensor → downstream shape errors
  surface far from the cause, inside `loss.py` or `_allowed_probabilities`;
- when `input_len < canvas_len`: a **silently wrong suffix** of the canvas —
  `canvas_len - input_len` columns of real logits, misaligned against `target_canvas`,
  `class_labels`, `canvas_loss_mask`, and `canvas_prediction_distances`. Training would run,
  loss would be finite, and every per-class and rare-class metric in `epoch_metrics.csv` would be
  quietly meaningless. This is the failure mode to fear.

The input half's values are unconstrained in content — loss is canvas-only (`scored_mask` from
`batch.canvas_loss_mask`, `loop.py:1099-1102`) and no consumer reads input-region logits — but
they must be **present, correctly shaped, and finite**.

---

## 5. Manifest stamps — **PASS**

### 5.1 What the stamps actually hash

**`src/thesis_ml/data/windowing.py:324-337`**

```python
def manifest_config_stamp(config: ProjectConfig) -> str:
    stamp_fields = {
        "manifest_version": MANIFEST_VERSION,
        "tokenized_artifact_version": TOKENIZED_ARTIFACT_VERSION,
        "target_semantics": _target_semantics(config),
        "sampling_interval_s": config.data.sampling_interval_s,
        "input_budget_tokens": config.data.input_budget_tokens,
        "canvas_budget_tokens": config.data.canvas_budget_tokens,
        "canvas_recon_fraction": config.data.canvas_recon_fraction,
        "within_type_tiebreak": config.data.within_type_tiebreak,
    }
    encoded = json.dumps(stamp_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Every input is a module constant or a `config.data.*` field. `_target_semantics`
(`windowing.py:371-372`) reads only `config.data.debut_mode`. **`config.model` is never touched.**

**`src/thesis_ml/data/windowing.py:351-359`**

```python
def vocabulary_stamp(vocabulary: ContentVocabulary) -> str:
    """Bind persisted token IDs to the vocabulary that wrote them."""

    tokens = [
        (token.name, token.token_id, token.source_id, token.kind)
        for token in vocabulary.tokens
    ]
    encoded = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Input is the vocabulary object alone. No config of any kind.

### 5.2 The rebuild trigger

`src/thesis_ml/pipeline/train_pipeline.py:510-522` rebuilds when any of:
`manifest_version`, `tokenized_artifact_version`, `config_stamp`, `replay_source_stamp`,
`replay_count`, `perspectives`, or `vocabulary_stamp` disagrees with the manifest header.
`src/thesis_ml/data/windowing.py:273-287` raises on the same three version/stamp mismatches at
load time. None of these seven inputs is a function of `config.model`.

### 5.3 Empirical proof against the real on-disk manifest

Loaded `configs/local_overfit_v2.yaml`, mutated **every** `model.*` field
(`d_model`, `layers`, `heads`, `ffn`, `qk_norm`, `self_conditioning`, `gradient_checkpointing`,
`rope_theta`), and recomputed:

```
stamp(base)    : 0678919d776b1648142a6ea2b91cc18b571a5c4af5d046687e6ae4c5172992ea
stamp(mutated) : 0678919d776b1648142a6ea2b91cc18b571a5c4af5d046687e6ae4c5172992ea
EQUAL          : True
on-disk stamp  : 0678919d776b1648142a6ea2b91cc18b571a5c4af5d046687e6ae4c5172992ea
matches config : True
```

The stamp is unchanged under arbitrary `model.*` mutation **and** matches the header of the actual
`data/processed/local/overfit_window_manifest.jsonl` on disk.

### 5.4 Verdict

**PASS.** Adding `model.frozen_input_kv`, `model.segment_embeddings`, and
`model.per_segment_positions` cannot alter `manifest_config_stamp` or `vocabulary_stamp`, cannot
invalidate `data/processed/local/overfit_window_manifest.jsonl`, and cannot trigger a rebuild
across the 943 replays. Flipping any toggle is free at the data layer.

---

## 6. Supporting details for the implementers

### 6.1 Where the attention masks come from

Both are built in `src/thesis_ml/data/collate.py::collate_diffusion_examples`:

- **`input_attention_mask`** — allocated `collate.py:114`, filled `collate.py:120`.
  Shape `[B, max_input_len]`, dtype `torch.bool`.
  **LEFT-padded**: `input_attention_mask[row, max_input_len - length:] = True`, so the `True`
  block is flush **right**.
- **`canvas_attention_mask`** — allocated `collate.py:127`, filled `collate.py:133`.
  Shape `[B, max_canvas_len]`, dtype `torch.bool`.
  **RIGHT-padded**: `canvas_attention_mask[row, :length] = True`, so the `True` block is flush
  **left**. `canvas_loss_mask` is a clone of it (`collate.py:137`).

Combined at `src/thesis_ml/model/model.py:113-123`:

```python
def _combine_attention_masks(
    input_token_ids: torch.Tensor,
    canvas_token_ids: torch.Tensor,
    input_attention_mask: torch.Tensor | None,
    canvas_attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if input_attention_mask is None:
        input_attention_mask = torch.ones_like(input_token_ids, dtype=torch.bool)
    if canvas_attention_mask is None:
        canvas_attention_mask = torch.ones_like(canvas_token_ids, dtype=torch.bool)
    return torch.cat([input_attention_mask.to(torch.bool), canvas_attention_mask.to(torch.bool)], dim=1)
```

Result: `[B, input_len + canvas_len]` bool. Because the concatenation order is fixed, slicing at
`input_len` recovers both regions exactly — no new mask parameters needed anywhere.

### 6.2 Where input and canvas are split today

**`SC2StrategyDiffusionModel.forward` does not split them at all.** There is no `input_len`
variable in `src/thesis_ml/model/model.py`. The split lives entirely in the callers:

- `src/thesis_ml/train/loop.py:1105` — `input_len = batch.input_token_ids.shape[1]`
- `src/thesis_ml/inference/sampler.py:132`, `:227`, `:363` — inline `input_token_ids.shape[1]`

Adding a local `input_len = input_token_ids.shape[1]` inside `model.forward` is new code, not a
rename of something existing.

### 6.3 `input_lengths` availability

- Declared: `src/thesis_ml/data/collate.py:38`, dtype `torch.long`, shape `[B]`.
- Filled: `collate.py:115` (alloc), `collate.py:121` (`input_lengths[row] = length`).
- Pinned: `collate.py:70`.
- **On device for training**: `src/thesis_ml/train/loop.py:2139` —
  `input_lengths=batch.input_lengths.to(device, non_blocking=non_blocking)`.
- **NOT on device for sampling**: `inference/sampler.py:183-186` moves only `input_token_ids`,
  `input_attention_mask`, `canvas_attention_mask`, and `input_features`. `denoise_canvas_once`
  (`sampler.py:110-131`) likewise. Any explicit-parameter design must add three `.to()` calls.

This asymmetry is the practical argument for deriving lengths from `input_attention_mask` inside
`model.forward` (§1.2) instead of threading a parameter.

### 6.4 `ARCHITECTURE_ID` and `validate_checkpoint_compatibility`

**`src/thesis_ml/model/model.py:14`**

```python
ARCHITECTURE_ID = "uniform-gemma4-dense-v1"
```

Assigned to the instance at `model.py:41` (`self.architecture_identity = ARCHITECTURE_ID`).

**`src/thesis_ml/model/model.py:138-158`**

```python
def validate_checkpoint_compatibility(
    checkpoint: dict,
    model: nn.Module,
    checkpoint_path: str,
) -> None:
    """Fail closed on retired or cross-process checkpoint metadata."""

    expected_architecture = getattr(model, "architecture_identity", None)
    observed_architecture = checkpoint.get("architecture_identity")
    if observed_architecture != expected_architecture:
        raise ValueError(
            f"checkpoint {checkpoint_path} architecture identity mismatch: "
            f"expected {expected_architecture!r}, got {observed_architecture!r}"
        )
    expected_process = getattr(model, "diffusion_process", None)
    observed_process = checkpoint.get("diffusion_process")
    if observed_process != expected_process:
        raise ValueError(
            f"checkpoint {checkpoint_path} diffusion process mismatch: "
            f"expected {expected_process!r}, got {observed_process!r}"
        )
```

Callers: `train/loop.py:1212` (`load_checkpoint`), `train/loop.py:1272` (`load_model_weights`),
`inference/sampler.py:400`, `viz/diagnostics.py:198`.

**Checkpoint-compatibility note.** `model.segment_embeddings` adds parameters, changing the
`state_dict` key set. `validate_checkpoint_compatibility` inspects only `architecture_identity`
and `diffusion_process`, so it will **not** catch a toggle mismatch; the strict
`load_state_dict` at `loop.py:1214` / `:1270` will raise on missing or unexpected keys instead.
That failure is loud and acceptable. **Do not bump `ARCHITECTURE_ID`** — that would reject every
existing checkpoint even with all toggles off, which contradicts the all-off-is-identical
requirement.

Mitigating factor for diagnostics: `viz/diagnostics.py:167-170` takes `model.*` from the
checkpoint's own stored config (`replace(config, model=model_config)`), so a checkpoint trained
with a toggle on rebuilds with that toggle on automatically.

### 6.5 `reset_joint_output()` re-application

**`src/thesis_ml/model/model.py:67-70`**

```python
        self._init_weights(model_config.layers)
        # The general initializer above intentionally initializes every Linear;
        # restore the exact zero-output joint residual after it completes.
        self.embedding.reset_joint_output()
```

The method itself, **`src/thesis_ml/model/embedding.py:145-150`**:

```python
    def reset_joint_output(self) -> None:
        """Make the initialized joint branch exactly zero, preserving E."""

        output = self.joint_mixer[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
```

`_init_weights` (`model.py:98-110`) walks `self.named_modules()` and re-initializes **every**
`nn.Linear` and `nn.Embedding` with `std = 0.02` (or `residual_std` for `attn.out` / `ffn.down`).
A new `nn.Embedding(2, d_model)` for segment embeddings **will** be caught by that sweep and
initialized to `std=0.02`, which is non-zero and therefore **not** bit-identical to today when the
toggle is off — except that the module does not exist at all when the toggle is off, so the
default path is unaffected. Still, a `reset_segment_embeddings()` zeroing the table belongs right
next to line 70 so that enabling the toggle starts from an exact no-op residual, matching the
`reset_joint_output` precedent:

```python
        self._init_weights(model_config.layers)
        # The general initializer above intentionally initializes every Linear;
        # restore the exact zero-output joint residual after it completes.
        self.embedding.reset_joint_output()
        # Same reasoning for the segment table: _init_weights re-initializes
        # every nn.Embedding at std=0.02, so zero it afterwards to make the
        # toggle's first step an exact no-op relative to the joint path.
        self.embedding.reset_segment_embeddings()
```

`reset_segment_embeddings()` must be a **no-op when the toggle is off** (guard on
`self.segment_embedding is None`), so it is safe to call unconditionally.

### 6.6 Config plumbing — a mandatory step that is easy to miss

`src/thesis_ml/config.py:401-405`:

```python
        if field.name not in raw:
            if is_optional:
                values[field.name] = None
                continue
            raise ConfigError(f"{field_path} is required")
```

**Dataclass field defaults are ignored by the config builder.** A new non-`Optional` field on
`ModelConfig` (`config.py:57-66`) that is absent from the merged YAML mapping raises
`ConfigError`. And `config.py:382-384` rejects unknown YAML keys. Therefore the three toggles must
land in the **same change** as:

- three `bool` fields on `ModelConfig` in `src/thesis_ml/config.py`, **and**
- three keys in `config/default.yaml` under `model:` (after line 27's
  `gradient_checkpointing: false`), each `false`.

Every profile in `configs/` transitively extends `config/default.yaml` (`local_overfit.yaml:1` →
`../config/default.yaml`; `local_overfit_v2.yaml:1` → `local_overfit.yaml`), and `_deep_merge`
(`config.py:365-372`) merges nested mappings, so adding the defaults once covers all four
profiles. No profile currently declares a `model:` section except `local_full.yaml` and
`local_overfit.yaml` (both only for `gradient_checkpointing: true`).

No `ModelConfig(...)` is constructed positionally anywhere in `src/` or `tests/` — every site uses
`load_config` plus `dataclasses.replace`, so new fields with defaults are safe there.

### 6.7 Tests that will need attention

- `tests/test_model.py:493-519` (`test_gradient_checkpointing_is_config_gated`) asserts
  `calls == 1` for a 1-layer model. Unaffected while `frozen_input_kv` defaults false; with the
  toggle on it would be 2 (one per pass).
- `tests/test_model.py:427-441` and `:470-490` monkeypatch `backbone_module.apply_rope` and
  `backbone_module.F.scaled_dot_product_attention`. Both patch points and both call signatures
  must survive.
- `tests/test_model.py:449-468` calls `attention.rope(seq_len, device=x.device, dtype=x.dtype)`
  and reconstructs attention by hand — this is the bit-identity guard for the toggle-off path.
- `tests/test_windowing.py:475-500` cross-checks batched vs. alone forwards through the left-pad
  offset. Per-segment positions are specifically designed to keep this equality (real content gets
  `0..L_i-1` regardless of batch padding width), so it should be a *stronger* pass, not a break —
  worth asserting explicitly in the new toggle tests.

---

## 7. Summary of decisions the implementers should not re-litigate

| Question | Answer | Basis |
|---|---|---|
| Must extra tensors be explicit `checkpoint()` args? | **No** — closure capture is correct under `use_reentrant=False` | measured, 0.000e+00 grad diff vs reference |
| Can pass 1 stash K/V in an outer list? | **No** — the body re-runs in backward (5 → 10 calls) | measured |
| Pass-2 mask shape | `[B, 1, 1, input_len + canvas_len]` bool, key-only | existing line 141 already emits it |
| Does FLASH+EFFICIENT tolerate the non-square mask on CUDA? | **Yes**, fwd+bwd, fp32+bf16, aligned and unaligned | measured on RTX 3070 / torch 2.12.1+cu130 |
| Is a MATH fallback needed? | **No** | same measurements |
| Is FLASH even available here? | **No** — wheel not compiled with it; everything already runs on EFFICIENT | `sdp_utils.cpp:810` warning + FLASH-only failures |
| Can the frozen-KV path return canvas-only logits? | **No** — 5 index-based consumers, silent misalignment | grep + slice semantics |
| Any consumer of `hidden_states`? | **None** repo-wide | grep |
| Can a toggle invalidate the window manifest? | **No — PASS** | stamp recomputation vs. the real on-disk header |
| Does `model.forward` need an `input_lengths` parameter? | **No** — derive from `input_attention_mask.sum(dim=1)` | `collate.py:119-121` construction |
