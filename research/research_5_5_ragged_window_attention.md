# Research: Ragged Window Attention (Per-Token Variable Lookback)
## ID: 5.5

---

## Executive Summary

**Novelty verdict:** PARTIAL — ~65% covered; per-head variable span (Sukhbaatar 2019 [1], MoA [8]), per-query non-contiguous selection (Quest [10], TokenSelect [22], Twilight [16], H2O [13]), variable-length hardware kernels (FlashInfer [17], Jagged Flash Attention [18]), and binary head classification (DuoAttention [21]) exist; residual novelty is per-token-position variable *contiguous* lookback with learned/heuristic w_i predictor + hardware-efficient ragged kernel at serving scale — an engineering-system gap, not a conceptual gap ([Sukhbaatar 2019, 1], [MoA, 8], [Jagged Flash Attention, 18]).

Idea 5.5 proposes per-token variable-width contiguous lookback windows in causal attention, where each token position independently determines how many past KV entries to attend to. The core mechanism (~65% covered) exists in per-head form (Sukhbaatar 2019, MoA 2024) and non-contiguous form (Quest, TokenSelect, Twilight), but the complete **per-token-position variable contiguous window system at LLM serving scale** is not published.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 validates the local-window requirement inside aggressive compressed attention. CSA/HCA both add an uncompressed sliding-window branch (`nwin=128`) because compressed blocks cannot preserve fine-grained local dependencies or same-block causality. This does not publish per-token variable contiguous windows, but it makes "compressed global + local exact window" a required baseline for any ragged-window proposal.


### Key Comparison Tables

**vs Baseline A1 (Qwen3.5-27B Hybrid)** — *RWA on 16 full-attention layers only; 48 Gated DeltaNet layers unchanged*

| Metric | Baseline A1 | RWA (w_avg=2K, s=32K) | Change | Notes |
|--------|------------|----------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L/4·s·d/H + L·d²) | O(L/4·w_avg·d/H + L·d²) | ↓ 16× attn; ~4% total | Attn on 25% of layers; MLP/Gated DeltaNet unchanged |
| Memory bandwidth (decode) | O(L/4·s·d_kv + L·d²) | O(L/4·w_avg·d_kv + L·d²) | ↓ KV BW on 16 layers | KV BW at 32K: ~2.15 GB (3.7% of total ~58 GB); 16× reduction → ~3.5% total BW savings |
| KV cache STORAGE | ~2.15 GB (32K ctx) | **~0.134 GB** (if w_max=2K) | ↓ ~16× IF w_max cap applied | 16 × 2 × 4 KV heads × **256 head_dim** × 2048 × 2 bytes; requires hard cap |
| KV cache BANDWIDTH/token | ~2.15 GB equivalent | **~0.134 GB avg** | ↓ ~16× (avg) | Uses w_avg=2K; no hard cap needed; varies per token |
| KV cache (262K ctx) | ~17.3 GB | ~1.08 GB (w_avg) | ↓ ~16× | At native context; KV fraction rises |
| Weight memory | O(L·d·d_ff) | = | = | No weight change |
| Training cost | 1.0× | ~1.0× | = | Attn small fraction at 27B scale |
| TTFT (8K prompt) | ref | ↓ <2% total | ↓ negligible | Attn on 25% of layers; MLP dominates |
| TPOT (batch=1) | ref | ↓ (⚠ assumes zero-overhead ragged kernel; practical hardware overhead unvalidated) **~1.03× at 32K** | ↓ negligible at 32K | KV is 3.7% of A1 total BW; at 262K context rises to ~23%, giving ~1.17× TPOT |

**vs Baseline A2 (Qwen3-32B Dense)** — *RWA on all 64 layers — highest-impact application*

| Metric | Baseline A2 | RWA (w_avg=2K, s=32K) | Change | Notes |
|--------|------------|----------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(s·d+d·d_ff)) | O(L·(w_avg·d+d·d_ff)) | ↓ 16× attn; net ~10% | Attn ~10% of FLOPs at 32K |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+w_avg·d_kv)) | ↓ KV BW 16× | KV fraction at 32K: 8.59/(64+8.59)=11.8%; SwiGLU 3-matrix MLP: KV/total ~14.6%; 16× KV reduction |
| KV cache STORAGE | **~8.59 GB** (32K ctx) | **~0.537 GB** (if w_max=2K) | ↓ ~16× IF w_max cap | Requires hard cap at w_max; else storage = ~8.59 GB full |
| KV cache BANDWIDTH/token | ~8.59 GB equivalent | **~0.537 GB avg** | ↓ ~16× (avg) | Uses w_avg; per-token benefit without hard cap |
| KV cache (262K ctx) | ~68.7 GB | ~4.3 GB (w_avg) | ↓ ~16× | Enables long-context deployment |
| Weight memory | O(L·d·d_ff) | = | = | Unchanged |
| Training cost | 1.0× | ~0.95–1.0× | ≈ = | ~9% attn FLOPs saved if attn=10% of training |
| TTFT (8K prompt) | ref | ↓ ~5% total | ↓ slight | 16× attn reduction; MLP dominates TTFT |
| TPOT (batch=1) | ref | ↓ (⚠ assumes zero-overhead ragged kernel; practical hardware overhead unvalidated) **~1.13× at 32K** | ↓ moderate at 32K | (64+8.59)/(64+0.54) = 72.59/64.54 ≈ 1.12×; at 128K: ~1.52× |

**vs Baseline B (Qwen3.5-397B-A17B MoE)** — *RWA on 15 global-attention layers only*

| Metric | Baseline B | RWA (w_avg=2K, s=32K) | Change | Notes |
|--------|-----------|----------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L/4·s·d/H + L·k_moe·d·d_e) | ↓ 16× on 15 attn layers | ↓ slight overall | MoE FFN dominates compute |
| Memory bandwidth (decode) | O(L/4·s·d_kv + active weights) | ↓ KV BW on 15 layers | ↓ small | Active weight BW (~34 GB) dominates |
| KV cache STORAGE | ~1.0 GB (32K ctx) | **~0.063 GB** (if w_max=2K) | ↓ ~16× IF cap | 15 × 2 × 2 KV heads × **256 head_dim** × 2048 × 2 bytes |
| KV cache BANDWIDTH/token | ~1.0 GB equivalent | **~0.063 GB avg** | ↓ ~16× (avg) | Already small fraction of total BW |
| Weight memory | O(L·E·d·d_e) | = | = | Unchanged |
| Training cost | 1.0× | ~1.0× | = | Attn tiny fraction of MoE training |
| TTFT (8K prompt) | ref | ≈ ref | = | MoE FLOPs dominate |
| TPOT (batch=1) | ref | ↓ (⚠ assumes zero-overhead ragged kernel; practical hardware overhead unvalidated) **~1.03× at 32K** | ↓ negligible at 32K | KV ~2.9% of total BW; 16× reduction → ~2.7% BW savings |

**vs Baseline C (K2 family, LLM360)** — *RWA on all 80 layers*

| Metric | Baseline C | RWA (w_avg=2K, s=32K) | Change | Notes |
|--------|-----------|----------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·(s·d+d·d_ff)) | O(L·(w_avg·d+d·d_ff)) | ↓ 16× attn | Attn fraction smaller at 8B hidden dim |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+w_avg·d_kv)) | ↓ KV BW 16× | KV fraction at 32K: 10.0/(145.1+10.0)=6.4%; 16× reduction |
| KV cache STORAGE | **~10.0 GiB** (32K ctx) | **~0.625 GiB** (if w_max=2K) | ↓ ~16× IF cap | 80 × 2 × 8 × 128 × 2048 × 2 bytes |
| KV cache BANDWIDTH/token | ~10.0 GiB equivalent | **~0.625 GiB avg** | ↓ ~16× (avg) | |
| KV cache (262K ctx) | ~80.0 GiB | ~5.0 GiB (w_avg) | ↓ ~16× | |
| Weight memory | O(L·d·d_ff) | = | = | Unchanged |
| Training cost | 1.0× | ~0.95–1.0× | ≈ = | |
| TTFT (8K prompt) | ref | ↓ ~3% total | ↓ slight | MLP dominates |
| TPOT (batch=1) | ref | ↓ (⚠ assumes zero-overhead ragged kernel; practical hardware overhead unvalidated) **~1.06× at 32K** | ↓ modest at 32K | (145.1+10.0)/(145.1+0.625) = 155.1/145.7 ≈ 1.065×; at 262K: ~1.33× |

---

## 1. Idea Description

Variable-size sliding window per token — some tokens look back at few positions, others at many. Sparse attention with a learned or heuristic per-token lookback count.

**Inferred inference-speedup intent:** At decode (batch=1), standard full attention requires reading O(s·d_kv) KV bytes per step. If token i has window width w_i << s, KV reads reduce to O(w_i·d_kv). Average bandwidth reduction = s/w_avg. At s=32K, w_avg=2K: 16× KV bandwidth reduction.

**Critical design distinction (from review):**
- **KV bandwidth reduction**: Achieved when average attended window is smaller; no hard cap required; each token reads only its w_i KV entries; bandwidth benefit = s/w_avg on average
- **KV storage reduction**: Requires a hard w_max cap — the KV cache stores at most w_max entries per position; this IS equivalent to a fixed sliding window. Without a hard cap, the KV cache still grows to s entries (you need to retain all positions because some future token might have a large w_i)
- The two benefits have different implementations. Variable window bandwidth savings are achievable without storage savings, but storage savings require committing to a maximum window.

**Sink tokens**: Any implementation must retain initial attention sink tokens (StreamingLLM finding) — pure per-token variable windows without sink token preservation degrade quality catastrophically for sequences beyond the window.

---

## 2. Literature Review

Adaptive Attention Span in Transformers[1]: Sukhbaatar et al., ACL 2019 (arXiv:1905.07799). Per-head learnable span with soft masking; large model achieves 0.98 bpc (SOTA) on enwiki8 with ~70% FLOPs reduction. Dynamic extension z_t = S·σ(v^T·x_t + b) introduces per-token span as token-function. Seminal closest prior art — but per-head not per-token-position.

Generating Long Sequences with Sparse Transformers[2]: Child et al., arXiv 2019 (arXiv:1904.10509). Introduces strided and local window attention patterns; establishes fixed-window prior art lineage.

Longformer: The Long-Document Transformer[3]: Beltagy et al., arXiv 2020 (arXiv:2004.05150). Fixed sliding window + global tokens; SOTA on long-document NLP. Per-layer window varies but not per-token.

Big Bird: Transformers for Longer Sequences[4]: Zaheer et al., NeurIPS 2020 (arXiv:2007.14062). Random + window + global sparse attention; proven Turing-complete.

LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models[5]: Han et al., NAACL 2024 (arXiv:2308.16137). Fixed Λ-shaped mask (128 sinks + 6,144 local); 2.72× decode speedup, 7.5× memory savings; strong fixed-window baseline for comparison.

Mixture of Depths: Dynamically Allocating Compute in Transformer Models[6]: Raposo et al., arXiv 2024 (arXiv:2404.02258). Per-token adaptive compute by routing tokens to bypass layers; validates per-token variable budget concept. Direct analogue for training stability analysis.

Efficient Streaming Language Models with Attention Sinks (StreamingLLM)[7]: Xiao et al., ICLR 2024 (arXiv:2309.17453). Attention sinks: initial tokens must always be retained. 22.2× speedup over sliding-window recomputation. Critical finding: any windowed scheme must retain sink tokens.

Mixture of Attention Spans (MoA)[8]: Fu et al., CoLM 2025 (arXiv:2406.14909). Per-head heterogeneous window assignment; 6.6–8.2× decode throughput improvement, narrows gap from full attention from 9–36% to within 5%. Closest published system to idea 5.5 at the per-head level — per-token is the natural next extension.

Native Sparse Attention (NSA)[9]: Yuan et al., ACL 2025 (arXiv:2502.11089). Three parallel branches: coarse compression + fine-grained token selection + sliding window. Abstract reports substantial speedups over Full Attention on 64K-length sequences across decoding/forward/backward; specific per-op multiples are paper-body tables. Input-conditioned selection at block granularity.

Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference[10]: Tang et al., ICML 2024 (arXiv:2406.10774). Per-query page-level KV selection. Effectively per-token variable non-contiguous attention. Up to 2.23× self-attention speedup, 7.03× inference latency reduction.

PagedAttention / vLLM[11]: Kwon et al., SOSP 2023 (arXiv:2309.06180). KV page management for serving. Variable per-token windows require dynamic page allocation/eviction that conflicts with PagedAttention's fixed-size page model; must address in implementation.

FlexAttention (PyTorch)[12]: PyTorch 2.4+ block_mask API; supports arbitrary block-sparse masks with near-FlashAttention performance. Most practical implementation path for prototyping per-token variable windows; reduces "requires custom CUDA kernel" barrier significantly.

H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models[13]: Zhang et al., NeurIPS 2023 (arXiv:2306.14048). Heuristic per-position KV selection retaining heavy hitters + recency; >95% of attention is sparse. Closest heuristic analog.

Lost in the Middle: How Language Models Use Long Contexts[14]: Liu et al., arXiv 2023 (arXiv:2307.03172). LLMs struggle with information in middle of context; attention concentrates at beginning and end. Supports the design choice of retaining sinks + recent window as the minimal viable retention set.

Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention[15]: Munkhdalai et al., arXiv 2024 (arXiv:2404.07143). Compressive memory + local window; addresses quality cliff from aggressive windowing by compressing (not discarding) distant context. Relevant for mitigating the quality cliff risk of idea 5.5.

Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning[16]: Lin et al., NeurIPS 2025 (arXiv:2502.02770). Per-query variable budget via top-p criterion; 15.4× self-attention acceleration. Per-token adaptive budget at inference; non-contiguous.

FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving[17]: Ye et al., MLSys 2025 (arXiv:2501.01005). Ragged tensor support, paged KV, block-sparse patterns; primary implementation substrate for variable-length attention patterns.

Jagged Flash Attention[18]: Xu et al., ACM RecSys 2024 (arXiv:2409.15373). Variable-length batches via jagged tensors; 3× speedup over dense FlashAttention. Demonstrates hardware feasibility of ragged-length attention computation.

The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs[19]: Nawrot et al., arXiv 2025 (arXiv:2504.17768). Comprehensive evaluation of 6 sparse methods across 9 tasks, 128K sequences, sparsity to 0.95. Key findings: (1) variable budget outperforms fixed budget; (2) longer sequences tolerate higher sparsity; (3) "even moderate sparsity levels often result in significant performance degradation on at least one task" — quality cliff risk is real.

SWAT: Sliding Window Attention Training for Efficient Large Language Models[20]: arXiv 2025 (arXiv:2502.18845). Fixed sliding window training; 98% performance retention, 50% less memory, 30% faster training. Training methodology baseline for any window-based approach.

DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads[21]: Xiao et al., ICLR 2025 (arXiv:2410.10819). Identifies retrieval heads (require full KV) vs streaming heads (only sinks + recent window); applies hard w_max only to streaming heads. Up to 2.45× memory reduction for MHA models. Direct structural analogue: per-head variable window depth is DuoAttention's binary special case; idea 5.5 generalizes to continuous per-token windows within each head.

TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection[22]: Wei et al., EMNLP 2025 (arXiv:2411.02886). Per-head token-level KV selection using QK dot-product criticality scores; 23.84× attention speedup; non-contiguous selection. Closest published per-token selection system; distinguishing factor for idea 5.5 is contiguous window constraint enabling hardware-efficient ragged kernels vs non-contiguous page selection.

DeepSeek-V4 CSA/HCA[23]: DeepSeek-AI, 2026. Both CSA and HCA combine compressed global attention with a 128-token sliding-window branch. This validates the quality need for local exact context under heavy compression and becomes the first baseline for any ragged-window extension.

---

## 3. Prior Art Classification

- **Status**: PARTIAL (~70% overlap after DeepSeek-V4)
- **Exists (per-head variable span)**: Sukhbaatar 2019 [1] — 70% FLOPs reduction, SOTA character LM
- **Exists (per-query non-contiguous selection)**: Quest [10], TokenSelect [22], Twilight [16], H2O [13] — per-token variable effective scope at inference
- **Exists (hardware variable-length kernels)**: FlashInfer [17], Jagged Flash Attention [18]
- **Exists (per-head variable window per layer)**: MoA [8] — 6.6–8.2× decode throughput; DuoAttention [21] — binary retrieval/streaming head split
- **Partial (per-token variable span)**: Sukhbaatar's dynamic extension z_t — introduced but not primary experiment and not serving-scale
- **Novel**: Per-token-position variable **contiguous** lookback window in causal autoregressive LLM at serving scale, with learned/heuristic predictor of w_i and hardware-efficient ragged kernel. Novel gap is an engineering system gap, not a conceptual gap.
- **No longer novel alone**: combining compressed long-range attention with a fixed local exact window. DeepSeek-V4 CSA/HCA uses this pattern at 1M context.

---

## 4. Technical Analysis

### 4.1 Storage vs Bandwidth Distinction (Critical)

| Design Choice | KV Storage Reduction | KV BW Reduction | Quality Impact |
|-------------|---------------------|-----------------|---------------|
| Hard cap w_max (= fixed sliding window) | ↓ s/w_max × | ↓ s/w_max × | Fixed window quality risks for any token needing long lookback |
| Variable w_avg without cap | ↓ 0× (full KV retained) | ↓ s/w_avg × on average | Better quality: tokens with long needs use long window |
| Variable w_avg with cap w_max | ↓ s/w_max × | ↓ s/w_avg × on average | Middle ground; depends on w_max << s |

**The 16× headline figure is a KV bandwidth figure (w_avg=2K/32K), not a storage figure unless a hard cap is applied.**

### 4.2 Compute Analysis

- **Prefill FLOPs reduction**: Σ_i w_i × d_kv vs s(s+1)/2 × d_kv. With uniform w_avg: ratio = 2·w_avg/s = 12.5% at 2K/32K. MLP dominates at LLM scale (attn ~10%), so total prefill FLOPs reduce ~1.14×. (Original §7 claim of "4×" is wrong; §4.2 value of 12.5% is correct.)
- **Decode FLOPs**: Bandwidth-bound at batch=1. FLOPs from attention = O(w_i × d/H) per layer.
- **Training FLOPs**: Attention FLOPs reduce by 16×. If attn=10% of training, total training FLOPs reduce ~9%.

### 4.3 TPOT Formula Applied (SwiGLU 3-matrix accounting)

**MLP bandwidth for A2**: SwiGLU uses 3 weight matrices (gate_proj, up_proj, down_proj). A2: ~750 MB/layer (vs ~524 MB/layer under a 2-matrix approximation).

| Baseline | Ctx | weight_BW | KV_before | KV_after (w_avg=2K) | TPOT | Notes |
|---------|-----|-----------|-----------|---------------------|------|-------|
| A1 | 32K | ~58 GB (56+correction) | ~2.15 GB | ~0.134 GB | (58+2.15)/(58+0.134) ≈ **1.034×** | KV is 3.6% |
| A2 | 32K | ~64 GB (SwiGLU ~750MB×64L = ~48GB + attn projections) | ~8.59 GB | ~0.537 GB | (64+8.59)/(64+0.537) ≈ **1.12×** | KV fraction ~11.8% |
| A2 | 128K | ~64 GB | ~34.4 GB | ~0.537 GB | (64+34.4)/(64+0.537) ≈ **1.52×** | KV fraction ~35% at 128K; 64× reduction (128K/2K) |
| B | 32K | ~34 GB | ~1.0 GB | ~0.063 GB | (34+1.0)/(34+0.063) ≈ **1.027×** | KV is 2.9% |
| C | 32K | ~145.1 GB | ~10.0 GiB | ~0.625 GiB | (145.1+10.0)/(145.1+0.625) ≈ **1.065×** | KV fraction 6.4% |

---

## 5. Implementation Considerations

- **Primary implementation path**: **FlexAttention** (PyTorch 2.4+, block_mask API) is the most accessible path, supporting arbitrary block-sparse masks with near-FlashAttention performance. It reduces prototyping time from months to weeks relative to a custom Triton kernel.

- **Production path**: FlashInfer [17] supports block-sparse patterns and ragged tensors; variable per-token windows can be expressed as per-row variable-width block-sparse masks. True 16× speedup requires avoiding padding (ragged kernel with no padding to w_max), or accepting speedup = s/w_max (storage cap).

- **PagedAttention compatibility**: Variable per-token windows require dynamic page allocation/eviction that conflicts with PagedAttention's fixed-size page model [11]. vLLM integration requires PagedAttention modification or bypassing to FlashInfer's block-sparse kernel.

- **Sink token integration**: Must retain initial 4–128 attention sink tokens [7] regardless of window width. Any implementation should add an invariant: window always includes sink positions.

- **Ragged batching overhead**: True per-token variable window creates ragged KV access lengths across positions; padding to w_max eliminates the per-token variability benefit. Jagged tensor representation (offset arrays) avoids padding but requires custom scheduling.

- **Training stability**: Soft masking (Sukhbaatar-style continuous gradient) is well-validated for gradient flow through variable spans. Learned predictor: small MLP or linear layer predicting w_i from hidden state; L1 span penalty encourages compression. Curriculum training (start with large windows, progressively reduce) recommended to avoid quality collapse.

- **Compatibility**:
  - Orthogonal to: KV quantization (5.1), MoE routing, weight pruning (5.7, 5.8)
  - Synergistic with: 5.4 (ragged window covers local; linked attention covers global distal — strongest combination)
  - No benefit to: Gated DeltaNet/linear attention layers (Baseline A1/B) — these have no KV cache to window

---

## 6. Synergies

- **Combines well with**:
  - 5.1 (TurboQuant KV quantization): multiplicative savings — (w_avg/s) × (bits/16) reduction; e.g., w_avg=2K and 4-bit KV: 16× × 4× = 64× KV memory reduction
  - 5.4 (Linked Attention): local window + global distal top-k; combined ~6.6% retention with both locality and semantic relevance; creates NSA-like architecture
  - 5.3 (Grammar/FSM Attention): grammar implies local window for syntactically local tokens; natural synergy
  - MoA-style per-head: combining per-head window heterogeneity with per-token variability gives 2D variable span

- **Conflicts/redundancy with**:
  - 5.2 (LSTM-Gated Attention): LSTM forgetting mechanism already compresses history; variable window is redundant for gated-recurrent layers
  - Baseline A1/B Gated DeltaNet layers: no KV cache to window

---

## 7. Risk Assessment

- **Technical risk**: MEDIUM — Core mechanism well-understood (Sukhbaatar 2019, MoA 2024, NSA). Novel element is the complete per-token-position contiguous window system at serving scale. Primary risk: quality cliff (Sparse Frontier 2025: "even moderate sparsity levels often result in significant performance degradation on at least one task"). Secondary risk: ragged batching hardware efficiency.

- **Primary benefit**: KV **bandwidth** reduction (s/w_avg × on average) enabling faster decode at long contexts; KV **storage** reduction only if hard w_max cap applied (= fixed sliding window trade-off). Most compelling use case: long-context serving (128K–262K) where KV bandwidth fraction rises to 35–52%.

- **Critical questions before committing (confirmed from review)**:
  1. Does contiguous per-token window preserve quality at w_avg=2K on LongBench/RULER? (Answer in days with offline experiment)
  2. Can FlexAttention express per-position variable windows without custom CUDA? (YES — elevated as primary path)
  3. Is per-token variable worth complexity over per-head variable (MoA)? (Need empirical answer)
  4. Is the design storage-capped (w_max, = fixed window) or bandwidth-only variable (no cap, storage unchanged)?

- **Implementation effort**: HIGH — FlexAttention prototype: 2–4 weeks. Production-quality ragged kernel: 3–6 months. Full training from scratch with learned predictor: 6–12 months.

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: MoA per-head heterogeneous windows at w_avg=2K/32K: LongBench 41.9 vs full 43.8 (−1.9 absolute, −4.3% relative) at 6.6× decode throughput [MoA, CoLM 2025, 8]; Adaptive Attention Span achieves 0.98 bpc SOTA on enwiki8 with ~70% FLOPs reduction [Sukhbaatar et al., ACL 2019, 1]; LM-Infinite's Λ-shape (128 sinks + 6144 local): 2.72× decode speedup, 7.5× memory, <1 PPL increase [LM-Infinite, NAACL 2024, 5]; SWAT fixed-sliding-window training: 98% retention at 50% memory reduction [SWAT, 20]; BigBird at 12.5% sparsity within 0.3–1.1 points of full attention on GLUE/SQuAD [BigBird, NeurIPS 2020, 4].
- **Known failure modes**: Sparse-Frontier evaluation finds "even moderate sparsity levels often result in significant performance degradation on at least one task" — at 90% sparsity, 4–12% relative average drop with some tasks >30% [Sparse Frontier, 19, arXiv 2025]; BigBird at higher sparsity (w=128, s=4096) drops 2–4 points on extractive QA [BigBird, 4]; post-hoc window truncation on a model not trained with it incurs ~2–5× the quality cost of training-time constraints [Sparse Frontier, 19]; catastrophic failure on long-document retrieval / needle-in-haystack at >32K context without sink preservation [LM-Infinite, 5]; multi-turn chat where distant context carries state degrades sharply.
- **Empirical evidence**: MoA Table (LongBench 41.9 vs 43.8 at 6.6× throughput) [MoA, 8]; Sukhbaatar Table (0.98 bpc enwiki8, 70% FLOPs reduction) [Sukhbaatar et al., 1]; LM-Infinite §Results (2.72× decode, <1 PPL increase) [LM-Infinite, 5]; BigBird §Results (0.3–1.1 point gap at 12.5% sparsity) [BigBird, 4]; Sparse Frontier §Results (variable > fixed budget by 1–3 points multi-task avg) [Sparse Frontier, 19]; SWAT §Results (98% retention at 50% memory reduction) [SWAT, 20].
- **Mitigations**: Always preserve sink tokens (128 positions suffice for Λ-mask) [LM-Infinite, 5]; train with the variable window constraint from scratch rather than post-hoc truncating — closes 60–80% of the quality gap [Sparse Frontier, 19]; use soft masking (Sukhbaatar-style) or curriculum (progressive w reduction) for best recovery [Sukhbaatar, 1]; set w_avg ≥ typical inter-relevant-span distance for the domain (1K–4K for NLP); compose with Linked Attention (5.4) to restore global long-range coverage for retrieval-dependent workloads; fall back to InfLLM-style CPU compressive memory for needle-in-haystack at 128K+ [InfLLM]; use page-level granularity via PagedAttention/vLLM for hardware efficiency [PagedAttention, 11].

### Detailed breakdown

- **Reported quality delta (from closest analogues)**:
  - Mixture of Attention Spans / MoA (Fu et al., CoLM 2025) [8]: per-head heterogeneous window assignment at w_avg=2K/32K reduces LongBench average from 43.8 (full attention) to 41.9 at 6.6× decode throughput improvement — a −1.9 absolute gap (−4.3% relative). Narrowing the gap from full attention requires the model to be trained with MoA from scratch; post-hoc window truncation produces larger quality drops of ~3–6 LongBench points at the same retention ratio.
  - BigBird (Zaheer et al., NeurIPS 2020) [4]: random + window + global sparse attention achieves Turing-completeness theoretically and GLUE/SQuAD performance within 0.3–1.1 points of full attention at 12.5% sparsity (w=512, s=4096). At higher sparsity (w=128, s=4096), BigBird drops 2–4 points on extractive QA — an early indicator that the quality cliff is real at high window compression.
  - Adaptive Attention Span (Sukhbaatar et al., ACL 2019) [1]: per-head learnable span achieves 0.98 bpc on enwiki8 (SOTA at time of publication) with ~70% FLOPs reduction by learning spans that average far shorter than the maximum. This is the gold standard for learned window compression: when the span is learned end-to-end, quality is maintained or improved vs fixed-window baselines, because the model can allocate longer spans to heads that need them.
  - The Sparse Frontier (Nawrot et al., arXiv 2025) [19]: comprehensive evaluation of 6 sparse attention methods on 9 tasks at 128K sequences, sparsity up to 0.95. Key finding: "even moderate sparsity levels often result in significant performance degradation on at least one task." Variable-budget methods outperform fixed-budget by 1–3 points on multi-task averages at the same sparsity. At 90% sparsity (w_avg=12.8K/128K), average task performance drops 4–12% relative vs full attention, with some tasks showing >30% degradation. The quality cliff is task-specific and unpredictable without empirical evaluation.
  - SWAT (arXiv 2025) [20]: fixed sliding window training achieves 98% performance retention on standard benchmarks at 50% memory reduction, providing a strong fixed-window training baseline. The 2% residual gap suggests that contiguous fixed windows incur a small but non-zero quality cost even when trained from scratch.
  - LM-Infinite (Han et al., NAACL 2024) [5]: Λ-shaped mask (128 sinks + 6,144 local window) achieves 2.72× decode speedup and 7.5× memory savings with <1 PPL increase on standard language modeling benchmarks. This is the closest published result to aggressive fixed-window deployment at production scale, confirming that a carefully chosen fixed window with sink preservation is within 1 PPL of full attention.

- **Monotonicity**: Quality degradation is broadly monotone with decreasing w_avg: shorter average windows cause higher PPL and lower task scores. However, the rate of degradation is non-linear and task-dependent. Tasks that rely on long-range semantic coherence (multi-document QA, long summarization) degrade steeply when w_avg drops below the typical inter-relevant-segment distance. Tasks that are locally structured (code generation, short-form QA) are highly tolerant of short windows (w_avg=1K–2K) with minimal quality loss. The Sparse Frontier (2025) confirms that variable-budget approaches are more quality-efficient than fixed-budget at the same FLOPs reduction, because they can allocate longer windows to tokens that empirically need them.

- **Recovery**: Recoverable via longer training or larger w_avg. Post-hoc window truncation (applying a window to a model not trained with it) incurs larger quality costs than training-time window constraints; the gap is ~2–5× in quality impact at equivalent sparsity. Training with soft masking (Sukhbaatar-style) or curriculum (progressively reducing w) produces the best recovery. For tasks with catastrophic failure under short windows (exact retrieval, multi-hop reasoning over long documents), full recovery requires either increasing w_max or adding a complementary mechanism like Linked Attention (5.4) for global long-range coverage, or InfLLM-style compressive external memory to retain evicted distant context.

- **Conditions for acceptable degradation**: Degradation is acceptable when (1) the task is locally structured and does not require long-range semantic retrieval (code completion, short QA, translation); (2) w_avg ≥ typical relevant-span distance for the target domain (often 1K–4K tokens for many NLP tasks); (3) the model is trained from scratch with the variable window constraint rather than post-hoc truncated; (4) sink tokens are always preserved, preventing catastrophic positional-encoding collapse. Degradation is not acceptable for: long-document retrieval, needle-in-haystack at >32K context, or multi-turn chat where distant context carries state.

---

<!-- CITATION MANIFEST -->
[1] Adaptive Attention Span in Transformers: Sukhbaatar et al., ACL 2019; arXiv:1905.07799; per-head learnable span; 0.98 bpc SOTA enwiki8 with ~70% FLOPs reduction; z_t dynamic extension closest per-token prior art
[2] Generating Long Sequences with Sparse Transformers: Child et al., arXiv 2019; arXiv:1904.10509; foundational local window attention; establishes sparse attention prior art lineage
[3] Longformer: The Long-Document Transformer: Beltagy et al., arXiv 2020; arXiv:2004.05150; fixed sliding window + global tokens; layer-wise window size variation
[4] Big Bird: Transformers for Longer Sequences: Zaheer et al., NeurIPS 2020; arXiv:2007.14062; random + window + global sparse attention; Turing-complete
[5] LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models: Han et al., NAACL 2024; arXiv:2308.16137; 128 sinks + 6144 local window; 2.72× speedup; fixed-window baseline
[6] Mixture of Depths: Dynamically Allocating Compute in Transformer Models: Raposo et al., arXiv 2024; arXiv:2404.02258; per-token adaptive compute via layer-skip routing; training stability reference for per-token variable budget
[7] Efficient Streaming Language Models with Attention Sinks (StreamingLLM): Xiao et al., ICLR 2024; arXiv:2309.17453; attention sinks must always be retained; 22.2× speedup; critical design constraint for windowed attention
[8] Mixture of Attention Spans (MoA): Fu et al., CoLM 2025; arXiv:2406.14909; per-head heterogeneous window assignment; 6.6–8.2× decode throughput; closest per-head system to idea 5.5
[9] Native Sparse Attention (NSA): Yuan et al., ACL 2025; arXiv:2502.11089; coarse + fine-grained + sliding window branches; abstract reports substantial decode/forward/backward speedups on 64K sequences (specific per-op multiples paper-body tables); input-conditioned at block granularity
[10] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference: Tang et al., ICML 2024; arXiv:2406.10774; page-level query-aware KV selection; 2.23× self-attention speedup, 7.03× inference latency reduction; per-query non-contiguous selection
[11] PagedAttention / vLLM: Kwon et al., SOSP 2023; arXiv:2309.06180; KV page management for serving; variable per-token windows conflict with fixed-size page model
[12] FlexAttention: PyTorch 2.4+ block_mask API; arbitrary block-sparse attention masks with near-FlashAttention performance; primary prototyping path for ragged window attention
[13] Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models (H2O): Zhang et al., NeurIPS 2023; arXiv:2306.14048; >95% of attention sparse on ~5% of tokens; heuristic per-position variable selection
[14] Lost in the Middle: How Language Models Use Long Contexts: Liu et al., arXiv 2023; arXiv:2307.03172; LLMs struggle with middle-context access; supports sink + recent window as minimal retention set
[15] Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention: Munkhdalai et al., arXiv 2024; arXiv:2404.07143; compressive memory + local window; mitigates quality cliff by compressing rather than discarding distant context
[16] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning: Lin et al., NeurIPS 2025; arXiv:2502.02770; per-query top-p variable budget; 15.4× attention acceleration; non-contiguous per-token variable budget at inference
[17] FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving: Ye et al., MLSys 2025; arXiv:2501.01005; ragged tensor support; block-sparse patterns; primary production kernel substrate
[18] Jagged Flash Attention: Xu et al., ACM RecSys 2024; arXiv:2409.15373; variable-length attention via jagged tensors; 3× speedup over dense FlashAttention; hardware-feasibility demonstration
[19] The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs: Nawrot et al., arXiv 2025; arXiv:2504.17768; comprehensive 6-method 9-task evaluation; variable budget > fixed budget; quality cliff at moderate sparsity; "token-to-page importance unfeasible during prefill"
[20] SWAT: Sliding Window Attention Training for Efficient Large Language Models: arXiv 2025; arXiv:2502.18845; fixed sliding window training methodology; 98% performance retention at 50% memory reduction; training baseline reference
[21] DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads: Xiao et al., ICLR 2025; arXiv:2410.10819; per-head retrieval vs streaming head distinction; streaming heads use w_max cap; up to 2.45× memory reduction; binary per-head special case of per-token variable windows
[22] TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection: Wei et al., EMNLP 2025; arXiv:2411.02886; per-head token-level KV selection via QK criticality; 23.84× attention speedup; non-contiguous; distinguishes from idea 5.5 contiguous window approach
