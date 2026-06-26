<objective>
Build the model: the embedding stack that fuses token identity with input-only contextual encodings, the vanilla bidirectional transformer backbone with RoPE, the output head, and the canvas-only loss utility with mandatory per-class loss decomposition. This is the core of the system — a single bidirectional stack that conditions on a clamped input region and denoises a canvas region.

No training loop, no diffusion noising, no optimizer here — this prompt builds the model and the loss function it will be trained with. The diffusion process that drives them is prompt 005.
</objective>

<context>
- Python + PyTorch, `src/` layout, uv-managed. Read `./CLAUDE.md`.
- Architectural source of truth is `./SPEC.md`. READ IT IN FULL. Directly relevant: §2 (model family), §3 (loss), §6 (input representation — the two position concepts and the embedding pipeline), §11 (config), §14 (ban list). SPEC wins on conflict.
- Consumes prompt 002 (vocabulary object + special-token IDs; token records carrying raw fields) and prompt 003 (collate output: batched input region + canvas region with a region/length marker, target canvas x0, per-position class labels, padding mask). USE those interfaces; do not reinvent them. If they're missing, STOP and say so.
- Localized task: one model module. Single agent, no orchestration.
</context>

<constraints_from_spec>
Non-negotiable (§2, §6, §14). Violating any is a failed task:
- ONE dense bidirectional transformer in the LLaDA/LLaMA lineage (RMSNorm, SwiGLU, RoPE, vanilla MHA). Single stack. FULL bidirectional attention over the entire `[input region][canvas region]` sequence — no causal masking anywhere. Padding is masked out; nothing else is. **Vanilla MHA, not GQA** (LLaDA's deliberate choice for diffusion; see requirements).
- Conditioning is CLAMPING, not a separate encoder. There is NO encoder-decoder split, NO cross-attention, NO separate prompt encoder. The input region is just the prefix of the same sequence.
- NO SSM layers in v1.
- NO classification head. The output is generative: a linear projection to the shared vocabulary, scored at canvas positions.
- NO copy mechanism (pointer net, copy gate, copy loss). Copying is learned via the loss.
- NO set aggregators / pooling-over-entities. Every entity is its own token; embeddings are per-token.
- **Two position concepts, never conflated (§6):** *sequence position* is RoPE (rotary), applied to queries/keys in attention — the only sequence-position mechanism, no learned absolute position table. *Map position* is an entity FEATURE, encoded as an additive contextual encoding. They are different axes.
- **Embedding pipeline (§6), input-region tokens only:** token embedding (learned) + additive contextual encodings (map position, unit stats, absolute clock) + team-flag embedding (self vs enemy). Continuous encodings (clock, map position) use extrapolation-friendly features (Fourier/sinusoidal), NOT learned lookups over fixed bins. CANVAS tokens receive the token embedding ONLY — never the contextual encodings or team flag.
- Loss is position-wise cross-entropy on CANVAS positions only; input positions never receive loss. Per-token-class loss logging is mandatory (§3).
</constraints_from_spec>

<requirements>
1. **Embedding module.**
   - Shared token embedding table over the full vocabulary (special + content tokens from 002).
   - Additive input-only contextual encodings: map position (2D, Fourier/sinusoidal → projected to d_model), unit stats (projected to d_model), absolute game clock (continuous, Fourier/sinusoidal → projected). These are summed into the token embedding for INPUT tokens only.
   - Team-flag embedding (self vs enemy), added to INPUT tokens only (the canvas is enemy-only, so it needs no team flag).
   - Canvas tokens: token embedding only. Assert this is structurally true — there should be no code path that adds contextual encodings to canvas positions.
   - Raw field values come from the batched records (002/003). The module turns values into vectors; it is the value→vector step SPEC §6 assigns to the model.

2. **Backbone — explicit, LLaDA/LLaMA lineage.**
   - A dense bidirectional transformer mirroring the published **LLaDA** architecture (the dense from-scratch masked-diffusion LM in this project's citations), downscaled. Components: **RMSNorm (pre-norm)**, **SwiGLU FFN**, **RoPE** for sequence position, **vanilla multi-head attention (MHA)** — NOT grouped-query attention.
   - **MHA, not GQA — deliberate.** GQA's payoff is shrinking the KV cache during autoregressive decoding; full-canvas diffusion does not decode autoregressively, so LLaDA uses vanilla MHA and trims FFN width to control the parameter budget. Follow that: equal query/key/value head counts. Do NOT implement GQA in v1. (If prompt-prefix KV caching of the clamped input is added much later, GQA could be revisited then — not now.)
   - RoPE applied to queries/keys in each attention layer, indexed by sequence position over the full `[input][canvas]` sequence. Input length varies per example, so the canvas begins at a different absolute index per example; RoPE is relative, so this is fine — add no fixed-length assumption.
   - Full attention with a padding mask only. No causal mask. No sliding window.
   - **Provisional config (PROVISIONAL, from SPEC §11):** d_model 1536, 16 layers, 12 heads (head_dim 128), SwiGLU FFN ~4096 (~450M, aspect ratio 96). Read ALL of these from config; the model class must accept any of these dims without code changes so size can track dataset scale. To scale up at a standard aspect ratio, widen d_model rather than only stacking layers (see SPEC §11 sizing note).

3. **Attention and efficiency — explicit.**
   - Route attention through **PyTorch `scaled_dot_product_attention` (SDPA)**, which dispatches FlashAttention kernels on supported GPUs and supports non-causal (full bidirectional) attention with a padding/attention mask. Do not hand-roll a naive softmax-attention matmul as the default path. Verify the current SDPA API and the correct way to pass a boolean/additive padding mask for full attention before writing it.
   - The optional `flash-attn` package (varlen path) may be used if SDPA proves insufficient for packed variable-length batches, but SDPA is the default — do not add the dependency unless a measured need appears.
   - **Fused cross-entropy:** expose it as an OPTIONAL config flag in the loss utility, OFF by default. Note in a comment that its memory benefit scales with vocabulary size and this project's vocabulary is tiny, so the saving is marginal — it is not a priority and must not pull in a heavy dependency.

4. **Output head.** Linear projection from d_model to vocab size, producing logits at every position. Loss uses only the canvas-position logits.

5. **Loss utility.** A function/module computing position-wise cross-entropy against the target canvas (x0), restricted to canvas positions, with:
   - **Per-class decomposition (mandatory):** using the per-position class labels from 003 (`enemy-observed`, `enemy-fogged`, `enemy-future`, `[DELIMITER]`, `[END]`, `[PAD]`), return the mean CE per class alongside the aggregate, for logging from the very first run.
   - **Per-class weights** from config (`class_loss_weights`, §11, default all 1.0), multiplying each position's loss by its class weight.
   - **Loop-fillable hooks:** accept (a) a per-position scored-mask selecting which canvas positions contribute (the diffusion loop will pass the masked-position set; default = all canvas positions) and (b) an optional per-position loss weight vector (the loop will pass MDLM schedule weights; default = ones). Keep these as parameters so 005 can supply diffusion specifics without 004 knowing about t.
   - Returns: aggregate scalar loss (for backprop) and a per-class breakdown dict (for logging). It does NOT sample t, does NOT apply `[MASK]`, does NOT know the noise schedule.

6. **Region handling.** The model reads the input/canvas split from the collate marker to (a) apply contextual encodings to the input region only and (b) let the loss target the canvas region only. The split is data, passed in — not inferred.

7. **Config-driven.** All dims, layer counts, head counts, FFN width, and class weights come from config. Nothing hardcoded.
</requirements>

<implementation>
- This is a LLaDA-shaped dense bidirectional transformer (RMSNorm, SwiGLU, RoPE, MHA) with a custom input embedding front-end. Before writing RoPE, verify the current idiomatic rotary application to q/k; the rotation/interleaving convention is easy to get subtly wrong. Verify the current PyTorch SDPA signature and how to pass a full-attention padding mask.
- Build the model class so d_model / layers / heads / FFN width are all config-driven and the model instantiates correctly at any of them. Final parameter count should track dataset scale (tiny vocab ⇒ nearly all params are backbone), so size must be trivially changeable via config — never wire in a fixed dimension.
- Do NOT add a learned input-vs-canvas segment embedding in v1. The input/canvas distinction is already carried by the presence/absence of contextual encodings and, at inference, by the all-`[MASK]` canvas init. Adding a segment embedding is a complexity the burden of proof hasn't met yet; leave it out unless first runs demonstrate a need. (Noted as a deliberate omission, not an oversight.)
- Keep the contextual encodings ADDITIVE per §6 — sum into the token embedding, do not concatenate-and-project (that changes the interface and isn't what SPEC specifies).
- Do NOT build: the diffusion noising, t-sampling, MDLM schedule weighting, optimizer, training loop, checkpointing (all 005); the sampler (006); evaluation (007). Build the model and the loss utility, nothing downstream.
- Avoid premature abstraction: no model registry, no config-driven layer-type plugin system, no SSM hooks (the §12 SSM fallback is shelved — do not stub it). One concrete model class.
</implementation>

<output>
Create, using relative paths:
- `./src/<pkg>/model/embedding.py` — token + additive input-only contextual encodings + team flag
- `./src/<pkg>/model/backbone.py` — RoPE bidirectional transformer (or co-locate; document choice)
- `./src/<pkg>/model/model.py` — the assembled model (embedding → backbone → head) with the region split
- `./src/<pkg>/model/loss.py` — canvas-only CE with per-class decomposition, per-class weights, loop-fillable scored-mask and per-position weight hooks
- `./tests/test_model.py` — tests covering the verification below
</output>

<verification>
Run these and report each PASS/FAIL with command and result:

1. **Forward shapes:** A forward pass on a small synthetic batch (use config defaults at reduced scale) produces logits of shape `[batch, seq_len, vocab_size]`. PASS only if shapes are correct and the pass runs without error.

2. **Contextual encodings are input-only:** Assert that canvas-position embeddings equal the pure token embedding (no contextual/team-flag component), and that perturbing an input token's raw fields (map position / stats / clock) changes that input position's embedding but changes NO canvas-position embedding before attention. PASS only if the input-only property holds structurally.

3. **Loss is canvas-only:** Construct a batch where input-region target ids differ from predictions; assert input positions contribute ZERO to the loss and only canvas positions do. PASS only if input positions are provably excluded.

4. **Per-class logging populated:** Assert the loss returns a per-class breakdown dict with an entry for every class present in the batch's labels, and that the class means combine (weightedly) to the aggregate. PASS only if per-class logging is present and consistent.

5. **Bidirectionality (no causal leak the wrong way):** Assert attention is full, not causal — e.g. changing a LATER sequence position measurably changes the logits at an EARLIER canvas position. PASS only if earlier positions depend on later ones (confirming bidirectional attention).

6. **RoPE length extrapolation smoke:** Run a forward pass on a sequence LONGER than the one used in check 1 (more canvas tokens and more input entities) and assert it runs and returns correct shapes with no fixed-length crash. This confirms the model is built for unseen lengths. PASS only if the longer sequence works.

For each: state what was run, the result, PASS/FAIL. If any fails, fix and re-run ALL. Use realistic shapes, not 1-token toys.
</verification>

<success_criteria>
- SPEC was read; nothing from §14 implemented (no encoder-decoder, cross-attention, SSM, classification head, copy mechanism, or set pooling); no segment embedding added.
- Single bidirectional stack in the LLaDA lineage (RMSNorm, SwiGLU, RoPE, vanilla MHA — no GQA); full attention via PyTorch SDPA/FlashAttention with a padding mask; map position is a separate additive feature, not conflated with sequence position.
- Model dims (d_model/layers/heads/FFN) are fully config-driven and the model instantiates at the provisional LLaDA-shape (~450M) and at other sizes without code changes.
- Embedding fuses token + additive input-only contextual encodings + team flag on the input region only; canvas tokens get token embedding only.
- Output head projects to the shared vocabulary; loss is canvas-only position-wise CE.
- Loss utility produces per-class decomposition (from 003's labels) and applies per-class weights, with loop-fillable scored-mask and per-position-weight hooks, while remaining ignorant of t and the noise schedule.
- All six verification checks PASS at realistic scale.
</success_criteria>
