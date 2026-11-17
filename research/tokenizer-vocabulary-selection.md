# Vocabulary endpoint for owner/timestep BPE

Research and design prepared 2026-11-17. No full-corpus OSCER result has been
measured yet. The runnable handoff is [scripts/oscer/README.md](../scripts/oscer/README.md).

Owner framing: this corpus is text for an NLP experiment, with **three
languages: Protoss, Terran and Zerg**, and one player/timestep block as a word.
Random resolves to one of those languages. The intended next consumer on OSCER
is a **generic autoregressive LLM trained from scratch**, not the existing diffusion model. The job
here is framework-neutral tokenizer research; it implements no model trainer and
does not change either established experiment arm's framework contract.

## What the literature establishes

The strongest practical evidence is the reasoning published by the model teams,
not merely their vocabulary totals. These are version-specific documented
examples, not claims about the latest release of each family.

| Deployed open-weight family | Stated tokenizer rationale | What is not publicly established |
|---|---|---|
| [DeepSeek-V3, §4.1](https://arxiv.org/html/2412.19437v2#S4.SS1) | Expands to roughly 128k, changes tokenizer training data and pre-tokenization for multilingual compression, and allows punctuation/newline combinations. It also addresses the boundary bias introduced by those combinations. | No published vocabulary-size sweep or stopping equation proves 128k optimal. The boundary treatment is an independent design decision, not a consequence of the count. |
| [Qwen, §2.2](https://arxiv.org/html/2309.16609v1#S2.SS2), [Qwen2, §2.1](https://arxiv.org/html/2407.10671v4#S2.SS1), [Qwen3, §2](https://arxiv.org/html/2505.09388v1#S2) | Qwen starts from an existing vocabulary, adds frequent Chinese and other-language text, compares compression across languages and reports preliminary downstream checks showing no degradation from expansion. Qwen2 reuses the tokenizer across model sizes; Qwen3 reports 151,669 entries. | Neither a universal 152k target nor an exact marginal-gain threshold is prescribed. Family reuse and regular/control-token accounting also affect the published number. |
| [Gemma 3, §2.2](https://arxiv.org/html/2503.19786v1#S2.SS2) | Uses the same tokenizer as Gemini 2.0, with approximately 262k entries, split digits, preserved whitespace and byte-level encodings. The report states improved balance for non-English languages and separately describes handling language imbalance in training data. | It does not publish a controlled 32k-versus-262k tokenizer sweep or demonstrate that this size independently optimizes every Gemma model. Reusing a family tokenizer is part of the implemented design. |

The transferable reasoning is: **encode useful material efficiently across all
target languages, preserve important boundaries/representability, account for
vocabulary-dependent model costs, and check downstream quality**. The precise
128k/152k/262k totals are outcomes of their corpora and model families. Treating
one of those numbers as a recipe would skip the reasoning the owner requested.

For this project, the counterpart of multilingual compression is atomic content
IDs per learned token and tokens per player/timestep word, measured separately
for Protoss, Terran and Zerg. Fewer tokens carry more replay content through a
fixed context and require fewer serial AR decoding steps for the same decoded
replay. Those benefits must be weighed against larger embeddings and output
projections. This is an engineering inference from the published design goals;
none of the reports establishes a numerically optimal size for this corpus.

Additional algorithmic and experimental evidence:

| Primary source | Evidence | Implication here |
|---|---|---|
| [Sennrich, Haddow & Birch, ACL 2016](https://aclanthology.org/P16-1162/) | BPE starts from atomic symbols, repeatedly merges frequent adjacent pairs, and controls inventory growth through a merge budget; rare words remain decomposable. | Use content IDs as the alphabet, each player/timestep as a word, and preserve a fully reversible atomic fallback. |
| [Ding, Renduchintala & Duh, MTSummit 2019](https://aclanthology.org/W19-6620/) | Systematically tests merge settings against architecture, language pair and resource conditions rather than trusting an inherited recipe. | Sweep size on this corpus; choose using dev data rather than assume text defaults transfer. |
| [Xu et al., ACL 2021, VOLT](https://aclanthology.org/2021.acl-long.571/) | Proposes an information-theoretic/optimal-transport approach to vocabulary selection without trial model training and demonstrates substantial inventory reductions in translation experiments. | A data-based shortlist is defensible, but this implementation uses a transparent compression curve, not VOLT or a claim to reproduce its objective. |
| [Schmidt et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.40/) | Experiments across 64 language models show that minimum token count alone does not establish best downstream quality; pre-tokenization and segmentation matter. The paper describes 30–50k as a common model-library range and examines 32,768 / 40,960 / 49,152. | This is evidence about common text conventions, not a current exhaustive market survey or a universal SC2 optimum. Compression is a screening metric; prediction quality must ultimately decide. |
| [Tao et al., NeurIPS 2024](https://arxiv.org/abs/2407.13623) | Vocabulary allocation interacts with model size, data and compute. The authors compare fixed-compute settings, derivative estimates and loss fitting. | Include embedding-size cost and sequence compression together. Do not transplant fitted natural-language scaling coefficients into these much smaller structured-state models. |
| [Touvron et al., Llama 2, 2023](https://arxiv.org/abs/2307.09288) | Uses a 32,000-token SentencePiece BPE vocabulary. | Direct AR LLM precedent for choosing a 32k-scale shared inventory as the initial reference. |
| [Groeneveld et al., OLMo, 2024](https://arxiv.org/html/2402.00838v3) | Uses 50,280 tokenizer entries, with the embedding table padded to 50,304 for throughput. | Approximately 50k is another established AR choice; distinguish learned vocabulary from allocated embedding rows. |
| [Dubey et al., Llama 3, 2024](https://arxiv.org/html/2407.21783v3) | Uses 128k entries, including 28k added for non-English language support, and measures compression and downstream effects. | AR multilingual work can justify larger inventories, but three languages do not imply multiplying a monolingual vocabulary by three. |
| [Conneau et al., XLM-R, ACL 2020](https://aclanthology.org/2020.acl-main.747/) | Studies shared multilingual vocabulary and language-sampling trade-offs across 100 languages (an MLM rather than an AR model). | Inspect each race language's fertility and representation separately; aggregate compression can hide an underrepresented language. |

There is **no established universal terminal pair frequency, vocabulary size,
or percent-compression threshold**. A **32,768-ID comparator** remains in the
sweep, but no size is automatically selected from convention. The first-run
endpoint is chosen from observed multilingual efficiency under an explicit
project tolerance. Frequency 2/3 on a small replay
is also not a corpus-scale endpoint: extending the corpus grows the repeated
tail. A frequency at merge selection is not final compound usage, because later
merges consume earlier compounds. Overlapping pair frequency is not the number
of deleted tokens for repeated runs.

## Concrete schema

1. Reuse the canonical serializer's finite-position entity-presence rules,
   unit/structure types, upgrade/research tokens and ascending content-ID order.
   Preserve native one-second sampled timesteps and repeated IDs: a multiset of
   five SCVs is five atomic IDs. No count quantization, bag pooling, position,
   health, game time or instance IDs enter token identity.
2. One **word** is all content belonging to one player in one recorded timestep.
   Thus `[scv scv marine]` may become one or several learned subword IDs. This
   is a word analogy, not a whole-block dictionary: an unseen composition still
   decomposes into known atoms and known compounds. Empty blocks stay empty and
   retain their structural ownership marker. Existing upgrades are included to
   preserve the shared content grammar; this is broader than combat units alone.
3. Retain all 297 current joint base IDs, including unused reserved tokens and
   all race tokens. Append one ID per ranked merge. No ID shifts or unknown-token
   loss. Structural tokens, owners, timesteps and replay ends are hard boundaries.
   All compounds decode exactly to their original sorted atomic sequence. The
   resulting *compound IDs* need not be numerically sorted; ordering refers to
   the decoded atoms, not re-sorting after encoding.
4. Reuse shared replay splitting (seed 1234, default 15% test then 10% of the
   remainder dev). Fit on training replays only, both owners once per replay.
   Do not duplicate both perspectives during fitting: they contain the same
   words and would merely double every frequency. Freeze the same tokenizer
   artifact for both model arms. Existing local-run subsets are not reused as
   this study's split. Save exact replay membership and SHA-256 source identities.
   Pool languages with their observed training frequencies for the first fit;
   no implicit race balancing or independent per-race vocabularies. Report
   per-race player-stream/word counts, content compression and tokens per word
   on train/dev and the selected full-corpus result. If metadata says Random,
   resolve from a unique starting worker/townhall type in the first recorded
   timestep; ambiguous or missing evidence fails explicitly. The reserved
   `[RANDOM]` base ID remains allocated for compatibility, not a fourth language.
5. Adjacent pair selection uses current overlapping occurrence counts weighted
   by word multiplicity, descending frequency, ascending `(left ID, right ID)`
   ties. Replacement is left-to-right and non-overlapping. This remains the
   existing preview algorithm. SQLite now maintains its exact inverted index,
   rewrites only affected words, and commits each merge for restartability.

## Vocabulary endpoint experiment

A metadata-only inventory of the current 943-replay corpus on 2026-11-17 found
565 Terran, 613 Protoss and 708 Zerg player streams (1,886 total). The unchanged
shared seeded split gives 722 train / 80 dev / 141 test replays; its player-stream
counts are train T=429/P=483/Z=532, dev T=52/P=47/Z=61 and test
T=84/P=83/Z=115. No unresolved Random was found in that inventory. These are
language-coverage counts, not before/after tokenization results. The imbalance
is modest but real; per-language compression should still be visible.

`config/tokenizer_study.yaml` defines **total IDs**, not additional merges:

| Candidate total IDs | Appended merges with base 297 | Tied token-table parameters at width 384 |
|---:|---:|---:|
| 297 (atomic control) | 0 | 114,048 |
| 512 | 215 | 196,608 |
| 1,024 | 727 | 393,216 |
| 2,048 | 1,751 | 786,432 |
| 4,096 | 3,799 | 1,572,864 |
| 8,192 | 7,895 | 3,145,728 |
| 16,384 | 16,087 | 6,291,456 |
| 32,768 | 32,471 | 12,582,912 |
| 49,152 | 48,855 | 18,874,368 |
| 65,536 | 65,239 | 25,165,824 |

These are illustrative vocabulary costs (`V × 384`), not an assumed hidden
width or architecture for the generic AR model and not parameter changes to the
current conditioned model (291 IDs). Larger vocabularies also widen dense logits
and the output projection; this table does not estimate total runtime or memory.
An untied head would add its own vocabulary-dependent parameters. Any later
arm comparison keeps the selected vocabulary and non-embedding budget common.

Fit one ranked trajectory to the largest target (65,536) or exhaustion at pair
frequency 2, whichever comes first. Evaluate deterministic ranked prefixes, with
identical full-replay framing at every size. Default `selection_rule:
multilingual_savings` chooses the **smallest inventory preserving 99% of the best
observed dev savings overall AND for each race language separately**. All three
languages must occur in both train and dev; missing coverage fails before
fitting. This applies the model teams' compression and multilingual-coverage
reasoning while minimizing allocated vocabulary among the qualifying candidates.
If the corpus runs out of repeated pairs, report the actual achieved inventory;
do not invent unused merges to hit a conventional number.

If selection lands at the largest explored size without pair exhaustion, it is
a search-boundary result. Expand the configured targets (for example to 131,072
and 262,144) if the per-language curves still improve materially. The job is
restartable; no claim is made that 65,536 is an inherent limit of this language.

Also report a pooled dev-compression recommendation: the smallest candidate
retaining **99% of the best observed dev token savings**:

`savings(V) = dev_atomic_full_tokens - dev_bpe_full_tokens(V)`

`V* = min { V : savings(V) >= 0.99 × max_candidate savings }`

For the default rule, impose the analogous inequality on each language's
**content** token savings as well. A language with zero achievable savings
contributes a vacuous constraint; it still requires measured coverage. Pooling
alone can hide poor compression for a less represented race.

The 99% rule is an explicit **project heuristic**, not a threshold claimed by
DeepSeek, Qwen or Gemma. It expresses a 1% tolerance on attainable token savings,
not a 1% model-quality or runtime tolerance. The report includes the entire curve
so the owner can inspect the trade-off. `selection_rule: dev_savings` uses only
the pooled recommendation; `selection_rule: reference` chooses the predeclared
32,768 comparator (or the actual inventory if fitting exhausts first). The atomic
control wins if all observed savings are zero. Frequency 2 is the exploration floor, not a chosen production stopping
frequency. A requested size can be unattainable; report actual achieved size and
stop inventing candidates after pair exhaustion. The upper search bound can
limit the recommendation; inspect the curve and expand it if the last size is
still yielding material gains. Report final train/dev usage, including compounds
used fewer than 100 times (a configurable descriptive threshold, not a filter).
Do not prune unused interior merge IDs: later rules can depend on them.

Every candidate reports illustrative token-table parameters (`V × d`, with the
configurable reporting width `d=384`) and a `tokens × V` dense-head work proxy.
For a fixed width, dense output projection work scales with that product;
backbone work, attention, packing, padding, kernel behavior and optimizer state
are omitted. It is not a runtime estimate or a complete compute-optimality
criterion. No actual model architecture has been selected by this study. The
endpoint remains a tokenizer shortlist until a from-scratch AR comparison checks
semantic quality and measured throughput, following Qwen's validation logic.

Lock `selection.json` before inspecting the test split. Candidate curves contain
train/dev only. Then measure the selected tokenizer on the entire corpus and
separately on train/dev/test. Test metrics are descriptive and must not be used
to revise the endpoint; further tuning consumes dev, not test. Exact round-trip
checks are required for every distinct owner/timestep word in each replay.

The final modeling decision should compare the multilingual recommendation, the 32k reference, approximately
16k/49k/65k alternatives, the compression recommendation and the atomic control
with the generic AR LLM under matched data exposure and compute,
reporting opponent-state/build-order/outcome performance and actual throughput.
Per-token loss/perplexity across different vocabularies is not directly
comparable; use semantic decoded metrics and, where probabilistically valid,
loss normalized by a common atomic representation. Diffusion denoising CE under
sampled corruption is not an AR next-token likelihood. No such training comparison
is implemented or claimed by this first tokenization job.

## Accounting and limits

Before/after totals each count one original replay once: content plus
`3T + 11` structural IDs for `T` timesteps. Header/opening delimiter = 6, footer
plus END = 5; each timestep has SELF, ENEMY and DELIMITER = 3. Empty player words
contribute no content but retain those markers. Re-windowing overhead, padding,
and the later second perspective are not silently added. Doubling both full
totals gives the two-perspective exposure before window framing. The old
conditioned estimator has different input/canvas framing, so its headline total
is not interchangeable with this joint-stream count.

Report total, average, min, max, p50/p90/p95/p99 tokens per replay before/after;
content/structural counts, compression factor and percent reduction; word counts,
empty words, tokens per word (all and nonempty); used content IDs, rare/unused
compound usage; per-replay and per-split figures; actual pair frequencies and
true merge deletions; hashes/config/split/timing and exact decode checks.

Only a bounded synthetic corpus and single-real-replay parsing checks are local
verification. The corpus-scale counts and useful regeneration timing must come
from the user-submitted OSCER job. This study creates no training manifests,
normalization statistics, model weights or production checkpoint compatibility.
The chosen tokenizer is a reusable artifact; wiring it into joint training and
sampling remains separate work under the sequence-format contract.
