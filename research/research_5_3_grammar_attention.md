# Research: Grammar/State-Machine Structured Attention (KV Compression)
## ID: 5.3

---

## Executive Summary

Idea 5.3 proposes using a learned grammar or FSM to constrain which tokens attend to which, storing only grammar-permitted KV entries and reducing KV cache size from O(s·d_kv) to O(|S|·d_kv) per token, where |S| << s. The primary benefit is KV cache size reduction enabling longer effective contexts within a fixed memory budget.

**Key finding:** Fixed sparse attention patterns are well-validated (Longformer, BigBird). KV eviction based on attention scores is production-deployed (H2O, SnapKV, PyramidKV). The specific novel contribution of an **end-to-end trained differentiable FSM/grammar generating attention masks with structured KV storage** is PARTIAL — no published paper does this exact combination, though NSA and MoSA approach it closely.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4's CSA/HCA makes generic "trained sparse/compressed attention with local fallback" substantially less novel. CSA compresses every 4 tokens into one KV entry, sparse-selects top-k compressed entries with a learned lightning indexer, and adds a sliding-window branch. HCA compresses every 128 tokens into one KV entry and attends densely over the compressed stream. Idea 5.3 remains distinct only if the FSM/grammar state itself controls the mask or cache layout; otherwise it overlaps with CSA/HCA/NSA.

**Key quantitative anchors:**
- A1 KV at 32K: **~2.15 GB** (head_dim=256 per canonical baseline)
- A2 KV at 32K: **~8.59 GB** (head_dim=128)
- TPOT improvement at 32K is modest: KV is ~11.8% of total bandwidth for A2; 10% retention → ~1.1× total TPOT
- MoSA "27% perplexity improvement" claim treated as speculative (unreviewed preprint)


### Key Comparison Tables

**vs Baseline A1 (Qwen3.5-27B Hybrid)** — *Grammar-structured attention on 16 full-attention layers only*

Note: Figures apply to learned-grammar variant only. CFG variant is infeasible due to O(s) per-token parsing overhead — see §4.3 (L244–249).

| Metric | Baseline A1 | This Idea (10% retention) | Change | Notes |
|--------|------------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L/4·s·d/H + L·d²) | O(L/4·\|S\|·d/H + L·d²) | ↓ on attn | 10× attn FLOP reduction on 25% of layers; MLP/linear-attn unchanged |
| Memory bandwidth (decode) | O(L/4·s·d_kv + L·d²) | O(L/4·\|S\|·d_kv + L·d²) | ↓ KV reads | KV in 16 full-attn layers: ~2.15 GB → ~0.215 GB at 10% retention |
| KV cache (32K ctx) | **~2.15 GB** | **~0.215 GB** (10%) | ↓ ~10× | 16 × 2 × 4 KV heads × **256 head_dim** × 32768 × 2 bytes = ~2.15 GB |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff + FSM params) | ≈ = | FSM parameters negligible vs billions of FFN weights |
| Training cost | 1.0× | ~1.1×–1.3× | ↑ | Differentiable grammar routing overhead |
| TTFT (8K prompt) | ref | = ref | = | Grammar eviction is decode-only; prefill unchanged |
| TPOT (batch=1) | ref | ↑ ↓ ~1.5%–2% ceiling (⚠ FSM overhead unknown at 27B+ scale) | ↓ negligible at 32K | KV at 32K is ~3.8% of total BW (~56 GB weight + ~2.15 GB KV); 10% retention → 0.9× KV savings → ~3.4% total BW reduction |

**vs Baseline A2 (Qwen3-32B Dense)** — *Grammar-structured attention on all 64 layers*

Note: Figures apply to learned-grammar variant only. CFG variant is infeasible due to O(s) per-token parsing overhead — see §4.3 (L244–249).

| Metric | Baseline A2 | This Idea (10% retention) | Change | Notes |
|--------|------------|--------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·s·d/H + L·d·d_ff) | O(L·\|S\|·d/H + L·d·d_ff) | ↓ on attn | 10× attn FLOP reduction; MLP unchanged |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+\|S\|·d_kv)) | ↓ on KV | KV at 32K: ~8.59 GB → ~0.86 GB; weight BW ~64 GB unchanged |
| KV cache (32K ctx) | **~8.59 GB** | **~0.86 GB** (10%) | ↓ ~10× | 64 × 2 × 8 × 128 × 32768 × 2 bytes = ~8.59 GB |
| KV cache (262K ctx) | ~68.7 GB | ~6.87 GB (10%) | ↓ ~10× | At native context length |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff + FSM params) | ≈ = | FSM negligible vs 32B weights |
| Training cost | 1.0× | ~1.1×–1.3× | ↑ | MoSA-style routing overhead |
| TTFT (8K prompt) | ref | = ref | = | Grammar eviction is decode-only |
| TPOT (batch=1) | ref | ↑ ↓ **~1.1× at 32K** (⚠ FSM overhead unknown at 27B+ scale) | ↓ modest at 32K | KV fraction 8.59/(64+8.59)=11.8%; 10% retention → 10.6% BW savings → ~1.11× TPOT. At 262K: KV 68.7/(64+68.7)=52%; ~1.83× TPOT at 10% retention |

**vs Baseline B (Qwen3.5-397B-A17B MoE)** — *Grammar-structured attention on 15 global-attention layers only*

Note: Figures apply to learned-grammar variant only. CFG variant is infeasible due to O(s) per-token parsing overhead — see §4.3 (L244–249).

| Metric | Baseline B | This Idea (10% retention) | Change | Notes |
|--------|-----------|--------------------------|--------|-------|
| Compute (FLOPs/token) | O(L/4·s·d/H + L·k_moe·d·d_e) | ↓ on 15 attn layers | ↓ | MoE FFN dominates |
| Memory bandwidth (decode) | O(L/4·s·d_kv + L·k_moe·d·d_e) | ↓ on KV | ↓ small | KV ~1.0 GB of ~35 GB total; 2.9% fraction; 10% retention → 2.6% BW savings → ~1.03× TPOT |
| KV cache (32K ctx) | **~1.0 GB** | **~0.1 GB** (10%) | ↓ ~10× | 15 × 2 × 2 KV heads × **256 head_dim** × 32768 × 2 bytes |
| Weight memory | O(L·E·d·d_e) | O(L·E·d·d_e + FSM params) | ≈ = | FSM negligible vs 397B |
| Training cost | 1.0× | ~1.05×–1.15× | ↑ | Only 25% of layers use attention |
| TTFT | ref | = ref | = | Decode-only eviction |
| TPOT (batch=1) | ref | ↑ ↓ ~1.03× at 32K (⚠ FSM overhead unknown at 27B+ scale) | ↓ negligible | MoE weight loading dominates |

**vs Baseline C (K2 family, 72.55B dense Llama-arch)** — *Grammar-structured attention on all 80 layers*

Note: Figures apply to learned-grammar variant only. CFG variant is infeasible due to O(s) per-token parsing overhead — see §4.3 (L244–249).

| Metric | Baseline C | This Idea (10% retention) | Change | Notes |
|--------|-----------|--------------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L·(\|S\|·d+d·d_ff)) | ↓ on attn | 10× attn FLOPs; MLP FLOPs unchanged |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+\|S\|·d_kv)) | ↓ on KV | KV at 32K: ~10.0 GiB → ~1.0 GiB; weight BW ~145.1 GB unchanged |
| KV cache (32K ctx) | **~10.0 GiB** | **~1.0 GiB** (10%) | ↓ ~10× | 80 × 2 × 8 × 128 × 32768 × 2 bytes |
| KV cache (262K ctx) | ~80.0 GiB | ~8.0 GiB (10%) | ↓ ~10× | K2-Think-V2 native max |
| Weight memory | ~145.1 GB | ~145.1 GB + FSM params | ≈ = | FSM negligible |
| TPOT (batch=1) | ref | ↑ ↓ **~1.06× at 32K** (⚠ FSM overhead unknown at 27B+ scale) | ↓ modest | KV 10.74/(145.1+10.74)=6.9%; 10% retention → ~1.06× TPOT. At 262K: KV=85.9 GB decimal (~80.0 GiB binary); (145.1+85.9)/(145.1+8.59)=231.0/153.69 ≈ ~1.50× at 10% retention (all values in GB decimal for ratio consistency) |

---

## 1. Idea Description

**Novelty verdict: PARTIAL — end-to-end trained differentiable grammar/FSM generating attention masks with KV cache structured to store only grammar-permitted entries has no published precedent; component prior art (Longformer [4], BigBird [5], Sartran et al. [6], StructFormer [7], Willard & Louf [8], Zhao et al. [22]) covers the mask side and the parse-constrained-attention side independently.**

**Grammar/State-Machine Structured Attention (KV Compression):** Use a learned grammar or FSM to constrain which tokens attend to which, storing only grammar-permitted KV entries and reducing KV cache size from O(s · d_kv) to O(|S| · d_kv) per token, where |S| << s.

Standard decode step: Read K, V for all s tokens × H_kv heads × L layers = O(s × d_kv × H_kv × L) memory reads per token.

Grammar/FSM-structured decode step: Compute FSM state for current token position → Identify allowed attendee set S(i) via FSM transitions → Read K, V only for tokens in S(i) = O(|S(i)| × d_kv × H_kv × L) memory reads per token.

**Reduction factor**: s / |S(i)|

- **Fixed grammar (sliding window)**: |S| = w + g. Validated, safe, large reduction.
- **Learned grammar (MoSA/NSA style)**: |S| determined end-to-end during training. Best quality.
- **CFG parse-tree grammar**: |S| = set of syntactic ancestors. Works for sentences; breaks for documents.

---

## 2. Literature Review

### Sparse Transformer (Child et al., 2019) [1]
- **URL**: arXiv:1904.10509
- **Summary**: Factorized sparse attention (strided + fixed). O(n√n) complexity. State-of-the-art on Enwik8, CIFAR-10, ImageNet-64. First rigorous demonstration that pre-specified sparse attention patterns preserve generation quality.

### Adaptive Attention Span (Sukhbaatar et al., 2019) [2]
- **URL**: arXiv:1905.07799
- **Summary**: Per-head learnable attention span: z_t = S·σ(v^T·x_t + b). Lower layers average ~32 chars; upper layers span longer. 0.98 bpc vs. 0.99 bpc (Transformer-XL) on enwiki8 at ~70% fewer FLOPs. Most direct prior art to per-token variable window.

### Reformer (Kitaev et al., 2020) [3]
- **URL**: arXiv:2001.04451
- **Summary**: LSH-based content-determined sparse attention — attend only within hash buckets. O(n log n) complexity. 2020 precursor to content-based learned grammar selection; establishes that content-based sparse attention predates MoSA by 5 years. ICLR 2020.

### Longformer (Beltagy et al., 2020) [4]
- **URL**: arXiv:2004.05150
- **Summary**: Sliding window (w=512–4096) + global tokens. O(n·w). Consistently outperforms RoBERTa on long document tasks. At w=512, s=32768: (512 + g) / 32768 ≈ 1.6% of full KV.

### BigBird (Zaheer et al., 2020) [5]
- **URL**: arXiv:2007.14062
- **Summary**: Random + local + global tokens. O(n). Universal approximator, Turing complete. At r=3, w=64, g=2, s=4096: ~1.7% of full KV. NeurIPS 2020.

### Transformer Grammars (Sartran et al., 2022) [6]
- **URL**: arXiv:2203.00633
- **Summary**: Grammar-constrained attention via CFG parse trees at sentence level. Works within sentences; "recursive syntactic composition bottleneck harms perplexity on document-level language modeling." Document-level grammar failure documented. TACL 2022.

### StructFormer (Shen, Tay et al., 2021) [7]
- **URL**: arXiv:2012.00857
- **Summary**: Jointly induces dependency and constituency structures from masked language modeling, integrating the induced dependency relations into the transformer via a differentiable dependency-constrained self-attention mechanism. Closest prior art to Idea 5.3's full combination (grammar structure + attention masking). Establishes that grammar-guided attention existed in 2021.

### Efficient Guided Generation for LLMs / Outlines (Willard & Louf, 2023) [8]
- **URL**: arXiv:2307.09702
- **Summary**: Foundational paper for FSM-based grammar-constrained LLM generation. Compiles any CFG into a finite state machine and applies it to constrain token probabilities. Directly enables converting grammar specifications into attention masks at inference time. Most direct technical predecessor to FSM-attention masking in Idea 5.3.

### StreamingLLM / Attention Sinks (Xiao et al., 2023) [9]
- **URL**: arXiv:2309.17453
- **Summary**: Attention sinks (first 4 tokens receive disproportionate attention). These global tokens must never be evicted or catastrophic quality loss results. ICLR 2024.

### ScissorHands (Liu et al., 2023) [10]
- **URL**: arXiv:2305.17118
- **Summary**: KV eviction via importance-persistence hypothesis. Up to 5× KV memory reduction. NeurIPS 2023.

### H2O (Zhang et al., 2023) [11]
- **URL**: arXiv:2306.14048
- **Summary**: KV eviction retaining heavy-hitter tokens (>20% KV retention maintains quality). NeurIPS 2023.

### LongNet (Ding et al., 2023) [12]
- **URL**: arXiv:2307.02486
- **Summary**: Dilated attention with exponentially growing window sizes. Enables 1B-token sequences.

### FSM-Attention Equivalence (Yang et al., 2023) [13]
- **URL**: arXiv:2310.13897
- **Summary**: Masked hard-attention Transformers recognize exactly the star-free languages (a subset of regular languages). Formal connection between attention masks and FSM expressiveness. NeurIPS 2024.

### SnapKV (Li et al., 2024) [14]
- **URL**: arXiv:2404.14469
- **Summary**: Observation-window voting for KV compression. "Negligible drops in accuracy" at 92% compression on 16 long-sequence datasets including Needle-in-a-Haystack. NeurIPS 2024.

### PyramidKV (Cai et al., 2024) [15]
- **URL**: arXiv:2406.02069
- **Summary**: Layer-adaptive KV compression. Full performance at 12% cache on LongBench; 100% Needle accuracy on LLAMA-3-70B at 128 KV entries.

### Native Sparse Attention / NSA (Yuan et al., 2025) [16]
- **URL**: arXiv:2502.11089
- **Summary**: Production-grade hardware-aligned structured sparse attention. Block-aligned tokens for GPU efficiency. Closest deployable analog of Idea 5.3. ACL 2025.

### Twilight (Lin, Tang et al., 2025) [17]
- **URL**: arXiv:2502.02770
- **Summary**: Adaptive top-p attention pruning. 98% attention pruning with 3.9× end-to-end speedup and maintained quality. NeurIPS 2025. **Note**: The 3.9× speedup is specifically at 98% pruning; at 90% pruning (10% retention), end-to-end speedup would be significantly less.

### The Sparse Frontier (Nawrot et al., 2025) [18]
- **URL**: arXiv:2504.17768
- **Summary**: Systematic study of sparse attention methods. "Even moderate sparsity levels often result in significant performance degradation on at least one task." Larger sparse models outperform smaller dense ones at equivalent cost.

### MoSA: Mixture of Sparse Attention (Piekos et al., 2025) [19]
- **URL**: arXiv:2505.00315
- **Summary**: Expert-choice learned sparse attention trained end-to-end. Claims 27% perplexity improvement vs. dense at equal compute. **[NOTE: May 2025 arXiv preprint, not peer-reviewed; treat as speculative until confirmed at venue.]** Best current demonstration of end-to-end learned grammar-structured attention.

### Grammar-Aligned Decoding (Park et al., 2024) [20]
- **URL**: arXiv:2405.21047
- **Summary**: Grammar-constrained decoding "can distort the LLM's distribution." Attention-level grammar constraints carry analogous risk. NeurIPS 2024.

### Stack Attention (DuSell & Chiang, 2024) [21]
- **URL**: arXiv:2310.01749
- **Summary**: Stack attention augments transformers with pushdown-automaton stacks, enabling recognition of arbitrary context-free languages without syntactic supervision. Outperforms baseline transformers on CFLs; improves NL modeling under constrained parameter budget. ICLR 2024 (spotlight). Closest theoretical prior art showing a grammar mechanism integrated directly into attention rather than applied as an external mask.

### Dependency Transformer Grammars (Zhao, Lou & Tu, 2024) [22]
- **URL**: arXiv:2407.17406
- **Summary**: Integrates dependency-based syntactic structures into Transformer LMs by simulating dependency transition systems via constrained attention masks and relative positional encodings. Achieves better generalization at comparable perplexity when trained on dependency-annotated data; outperforms constituency-based syntactic models. 2024. Direct prior art for dependency-constrained attention masking — overlaps significantly with Idea 5.3's novel contribution.

### FlexAttention (Dong et al., 2024) [23]
- **URL**: arXiv:2412.05496
- **Summary**: PyTorch programming model that compiles arbitrary attention mask functions into optimized FlashAttention-like kernels via Triton, supporting block-sparse masks with performance competitive with hand-written kernels. 2024. Key practical implementation backbone for grammar-induced irregular attention patterns without custom CUDA kernels.

### DeepSeek-V4 CSA/HCA (DeepSeek-AI, 2026) [24]
- **URL**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro and `DeepSeek_V4.pdf`
- **Summary**: Training-native hybrid compressed attention. CSA combines sequence compression, sparse top-k compressed-block selection, shared-KV MQA, grouped output projection, and a sliding-window branch. HCA uses much heavier sequence compression with dense attention over compressed entries. DeepSeek reports operational 1M-token context with much lower FLOPs/KV than DeepSeek-V3.2.
- **Relevance**: New strongest production-scale baseline for learned sparse/compressed attention. Grammar/FSM variants must show a specific structure advantage over CSA/HCA, not merely KV reduction.

---

## 3. Prior Art Classification

- **Status**: PARTIAL-HIGH overlap after DeepSeek-V4

| Component | Status | Best Reference |
|-----------|--------|---------------|
| Fixed sparse attention patterns | EXISTS AND WELL-VALIDATED | Longformer [4], BigBird [5] |
| Learned per-head span control | EXISTS | Sukhbaatar et al. [2] |
| Content-based learned sparse | EXISTS | Reformer [3], MoSA [19] |
| Grammar-constrained attention (parse-tree) | EXISTS, PARTIAL — sentence-level only | Sartran et al. [6], StructFormer [7] |
| Dependency-constrained attention masking | EXISTS | Zhao et al. [22] |
| CFL-recognizing grammar in attention | EXISTS | DuSell & Chiang [21] |
| FSM-derived attention masking | EXISTS (theory) | Willard & Louf [8], Yang et al. [13] |
| FSM + KV compression (end-to-end) | DOES NOT EXIST | — |
| Trained compressed sparse attention + sliding window | EXISTS | DeepSeek-V4 CSA/HCA [24], NSA [16] |
| KV eviction via attention scores | EXISTS AND WELL-VALIDATED | H2O [11], SnapKV [14], PyramidKV [15] |

**Novel contribution**: End-to-end trained model where a differentiable learned FSM/grammar generates the per-token attention mask, the KV cache is structured to store only grammar-permitted entries, and training jointly optimizes the grammar and the language model quality. Note: Zhao et al. [22] demonstrates dependency-constrained attention masks trained on annotated data, and DeepSeek-V4 [24] demonstrates trained compressed sparse attention with local fallback. The remaining gap is the combined differentiable grammar + KV-structured storage without external parse annotation and with structure that CSA/HCA cannot capture.

---

## 4. Technical Analysis

**Novelty verdict: PARTIAL — end-to-end trained differentiable grammar/FSM generating attention masks with KV cache structured to store only grammar-permitted entries has no published precedent; component prior art (Longformer [4], BigBird [5], Sartran et al. [6], StructFormer [7], Willard & Louf [8], Zhao et al. [22]) covers the mask side and the parse-constrained-attention side independently.**

### 4.1 KV Reduction Mechanism

```
Standard:  O(s × d_kv × H_kv × L) memory reads per decode step
Grammar:   O(|S| × d_kv × H_kv × L) memory reads per decode step
Reduction: s / |S|
```

**Realistic attendance rates** (from literature):
| Method | Practical attendance | Quality impact |
|--------|--------------------|--------------|
| Longformer (w=512) | ~1.6% at s=32K | Low for local tasks |
| BigBird (r+w+g) | ~1.7% at s=4K | Low with global tokens |
| SnapKV | ~8% | Negligible on standard benchmarks |
| PyramidKV | ~12% | Full quality at 12% |
| H2O | ~20% | Negligible on generation |
| MoSA (best perf.) | ~30% | +27% PPL improvement [speculative] |

**Realistic average-case estimate**: 15–30% attendance, not 10%. At 20% attendance:
- A2 TPOT at 32K: 8.59×0.8/(64+8.59×0.2) = 6.87/(64+1.718) ≈ (64+8.59)/(64+1.718) = 1.10× TPOT

### 4.2 TPOT Analysis

Using canonical formula: TPOT_ceiling = (weight_BW + KV_BW_before) / (weight_BW + KV_BW_after)

| Baseline | Context | weight_BW | KV_before | KV_after (10%) | TPOT ceiling |
|---------|---------|-----------|-----------|----------------|-------------|
| A1 | 32K | ~54 GB | ~2.15 GB | ~0.215 GB | ~1.035× |
| A2 | 32K | ~64 GB | ~8.59 GB | ~0.86 GB | ~1.11× |
| A2 | 262K | ~64 GB | ~68.7 GB | ~6.87 GB | ~1.83× |
| B | 32K | ~34 GB | ~1.0 GB | ~0.1 GB | ~1.026× |
| C | 32K | ~145.1 GB | ~10.74 GB | ~1.0 GB | ~1.058× |
| C | 262K | ~145.1 GB | ~85.9 GB | ~8.59 GB | ~1.50× |

**Key insight**: Grammar-structured attention primarily enables **longer effective context within a fixed memory budget**, not dramatic TPOT improvement at short contexts. Primary value is memory capacity enablement.

### 4.3 CFG Parsing Overhead

For the CFG grammar variant:
- Online incremental parsing: O(s) per token (Earley-style)
- Batch CFG parsing (CYK): O(s³) or O(s²·G) for grammar G
- The O(s) per-token cost of running an online parser during decode **would dominate all other savings at long context**
- **This makes the CFG variant infeasible for production.** Only learned grammar variants avoid this overhead.

**TPOT overhead caveat:** The TPOT improvements shown in the tables above assume the KV reduction benefit exceeds the per-token grammar computation overhead. For the learned grammar variant (the relevant case for LLM deployment), this has not been verified empirically at 27B+ scale. The overhead of running a grammar automaton per decode step must be benchmarked before citing these figures. Until empirical verification is available, treat all TPOT improvement numbers as upper bounds conditioned on negligible grammar overhead.

---

## 5. Implementation Considerations

- **Recommended design (combining best evidence)**:
  - [70%] Sliding window (w=512): O(w × d_kv) KV per token
  - [20%] Learned content-selected tokens (MoSA-style): O(k × d_kv) KV per token
  - [10%] Attention sinks (4-8 global tokens)
  - Total KV per token: (512 + 64 + 8) / 32768 ≈ **1.78% of full KV**

- **Integration**:
  - A1: Apply to full-attention layers (25% of layers). Gated DeltaNet layers unaffected.
  - A2: Apply to all 64 layers. Primary benefit target.
  - B: Apply to global attention layers only (25% of layers).
  - C: Apply to all 80 layers.

- **Hardware considerations**: NSA [16] provides block-aligned sparse attention for GPU efficiency. FlexAttention [23] compiles arbitrary attention mask functions into optimized Triton/FlashAttention kernels, enabling grammar-induced irregular patterns without hand-written CUDA. Grammar-induced irregular patterns should target either block alignment (constrain grammar to block-aligned masks) or FlexAttention-based compilation.

---

## 6. Synergies

- **Combines well with**: 5.1 (TurboQuant — quantize the retained KV entries); 5.2 (LSTM-Gated — gated layers need no KV, sparse attention on remaining full-attn layers).
- **Note**: the practical variant of grammar-constrained attention reduces to a reframing of NSA [16]; the incremental novelty over NSA requires explicit justification.

---

## 7. Risk Assessment

- **Technical risk**: MEDIUM — Learned sparse attention (MoSA, NSA) is the viable path. External grammar parsers are infeasible for production. The novelty contribution (differentiable FSM attention) remains to be demonstrated.
- **Potential impact**: PRIMARY = KV memory capacity enablement (longer contexts at fixed hardware). SECONDARY = TPOT improvement at very long contexts (>128K).
- **Implementation effort**: MEDIUM — NSA [16] and FlexAttention [23] (arXiv:2412.05496) provide building blocks; FlexAttention in particular compiles arbitrary mask functions to optimized Triton kernels, lowering the implementation barrier for grammar-induced irregular patterns. Block-aligned learned sparse masks are the practical implementation target.

---

## 8. Accuracy / Quality Tradeoff

**Novelty verdict: PARTIAL — end-to-end trained differentiable grammar/FSM generating attention masks with KV cache structured to store only grammar-permitted entries has no published precedent; component prior art (Longformer [4], BigBird [5], Sartran et al. [6], StructFormer [7], Willard & Louf [8], Zhao et al. [22]) covers the mask side and the parse-constrained-attention side independently.**

| Method | KV vs Full | Quality Impact |
|--------|-----------|----------------|
| Longformer (w=512, s=32768) | ~1.6% | Low for local tasks, high for long-range |
| BigBird (r+w+g) | ~1.7% | Low (universal approx. guaranteed) |
| SnapKV | ~8% | "Negligible accuracy drop" on 16 datasets [14] |
| PyramidKV | ~12% | Full performance on LongBench [15] |
| H2O | ~20% | Negligible on generation tasks [11] |
| MoSA (claimed) | ~10–30% | +27% PPL improvement **[speculative]** [19] |
| Twilight (98% pruning) | ~2% | 3.9× speedup with maintained quality [17] |

**Critical risk**: Document-level dependency loss (cross-sentence context severed by pure syntactic grammar). Mitigation: global sink tokens + sliding window fallback.

---

<!-- CITATION MANIFEST -->
[1]: Sparse Transformer — Child, Gray, Radford, Sutskever. arXiv:1904.10509. 2019. O(n√n) factorized sparse attention.
[2]: Adaptive Attention Span — Sukhbaatar, Grave, Bojanowski, Joulin. arXiv:1905.07799. ACL 2019. Per-head learnable span; 0.98 bpc on enwiki8.
[3]: Reformer — Kitaev, Kaiser, Levskaya. arXiv:2001.04451. ICLR 2020. LSH-based content-determined sparse attention; O(n log n).
[4]: Longformer — Beltagy, Peters, Cohan. arXiv:2004.05150. 2020. Sliding window + global tokens; O(n·w); ~1.6% KV at w=512, s=32K.
[5]: BigBird — Zaheer et al. arXiv:2007.14062. NeurIPS 2020. Random + local + global; O(n); universal approximator; Turing complete.
[6]: Transformer Grammars — Sartran, Barrett, Kuncoro et al. arXiv:2203.00633. TACL 2022. CFG parse tree attention; works at sentence level; fails at document level.
[7]: StructFormer — Shen, Tay, Zheng, Bahri, Metzler, Courville. arXiv:2012.00857. 2021. Joint unsupervised induction of dependency and constituency structure with differentiable dependency-constrained self-attention; closest prior art for grammar + attention masking.
[8]: Efficient Guided Generation (Outlines) — Willard, Louf. arXiv:2307.09702. 2023. Foundational FSM-based grammar-constrained LLM generation; CFG→FSM compilation for attention masking.
[9]: StreamingLLM / Attention Sinks — Xiao, Tian, Chen, Han, Lewis. arXiv:2309.17453. ICLR 2024. Attention sinks must be preserved; first 4 tokens receive global attention.
[10]: ScissorHands — Liu, Desai, Liao et al. arXiv:2305.17118. NeurIPS 2023. Importance-persistence KV eviction; up to 5× reduction.
[11]: H2O — Zhang et al. arXiv:2306.14048. NeurIPS 2023. Heavy-hitter KV eviction; >20% retention maintains quality.
[12]: LongNet — Ding et al. arXiv:2307.02486. 2023. Dilated attention with exponential window growth; 1B-token sequences.
[13]: FSM-Attention Equivalence — Yang, Chiang, Angluin. arXiv:2310.13897. NeurIPS 2024. Hard-attention Transformers recognize exactly star-free languages.
[14]: SnapKV — Li, Huang, Yang et al. arXiv:2404.14469. NeurIPS 2024. Observation-window voting; negligible accuracy drop at 92% compression.
[15]: PyramidKV — Cai, Zhang, Gao et al. arXiv:2406.02069. 2024. Layer-adaptive KV compression; full performance at 12% cache.
[16]: Native Sparse Attention (NSA) — Yuan et al. arXiv:2502.11089. ACL 2025. Production-grade block-aligned structured sparse attention; closest deployable Idea 5.3 analog.
[17]: Twilight — Lin, Tang, Yang et al. arXiv:2502.02770. NeurIPS 2025. Adaptive top-p pruning; 3.9× speedup at 98% pruning; NOT directly extrapolable to 90% pruning.
[18]: The Sparse Frontier — Nawrot et al. arXiv:2504.17768. 2025. Systematic study; variable-budget sparse attention outperforms fixed-budget.
[19]: MoSA: Mixture of Sparse Attention — Piekos, Csordás, Schmidhuber. arXiv:2505.00315. May 2025 preprint (not peer-reviewed). Learned sparse attention; claims 27% PPL improvement at equal compute [speculative until confirmed].
[20]: Grammar-Aligned Decoding — Park, Wang et al. arXiv:2405.21047. NeurIPS 2024. Grammar-constrained decoding can distort LLM distribution; analogous risk for attention-level grammar constraints.
[21]: Stack Attention — DuSell, Chiang. arXiv:2310.01749. ICLR 2024 (spotlight). Pushdown-automaton stacks integrated into attention; enables recognition of arbitrary CFLs without syntactic supervision; outperforms baseline transformers on CFLs.
[22]: Dependency Transformer Grammars — Zhao, Lou, Tu. arXiv:2407.17406. 2024. Dependency-constrained attention masks simulating dependency transition systems; better generalization at comparable perplexity; direct prior art overlapping with Idea 5.3.
[23]: FlexAttention — Dong, Feng, Guessous, Liang, He. arXiv:2412.05496. 2024. PyTorch compiler model generating optimized Triton/FlashAttention kernels from arbitrary attention mask functions; practical backbone for grammar-induced irregular attention patterns.
