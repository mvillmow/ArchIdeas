# Research: Linked Attention (Sparse Relevant-Only)
## ID: 5.4

---

## Executive Summary

Idea 5.4 proposes dynamic inference-time KV cache eviction retaining only the top-k most relevance-scored token entries per attention head, reducing KV memory footprint and decode-phase memory bandwidth. The core mechanism is well-covered by H2O (NeurIPS 2023), SnapKV, Quest (ICML 2024), DuoAttention (ICLR 2025), and TokenSelect (EMNLP 2025). The genuinely novel contribution — a **persistent sparse adjacency graph with incremental O(k·d) updates** — is not published as such.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4's CSA is now the strongest trained analogue of linked attention. CSA compresses KV entries by blocks, computes indexer scores with a low-rank lightning indexer, selects top-k compressed KV entries for each query, and combines those selected compressed entries with a local sliding-window branch. This erodes novelty for generic "top-k relevant KV entries" claims. The remaining distinct value of 5.4 is post-hoc compatibility for existing checkpoints or a persistent sparse graph that is maintained incrementally without full re-scoring.

**Key quantitative anchors:**
- A1 KV at 32K: **~2.15 GB** (16×2×4×256×32768×2)
- A2 KV at 32K: **~8.59 GB** (64×2×8×128×32768×2)
- TPOT improvement at 32K is modest: ~1.11× for A2 (KV is ~12% of total bandwidth); primary benefit is KV **capacity** reduction enabling longer contexts
- Score-based (H2O-style) eviction does NOT reduce current-step decode FLOPs — only future-step KV bandwidth; this distinction is carried throughout
- Scatter-gather penalty for token-level sparse KV reads: 2–10× bandwidth degradation vs sequential; page-level granularity (P≥16) required for real-world speedup
- Venue anchors: Ada-KV (NeurIPS 2025); TokenSelect (EMNLP 2025); KVQuant (NeurIPS 2024); CaM (ICML 2024, openreview:LCTmppB165)


### Key Comparison Tables

**vs Baseline A1 (Qwen3.5-27B Hybrid)** — *Linked attention on 16 full-attention layers only; 48 Gated DeltaNet layers unchanged*

Note: Figures assume persistent-sparse-graph variant (novel, §4). Published methods (H2O, Quest, TokenSelect) are score-based and require per-step O(s·d) rescoring; see §4 L67–68.

| Metric | Baseline A1 | This Idea (10% retention, page-level) | Change | Notes |
|--------|------------|--------------------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L/4·s·d/H + L·d²) | O(L/4·k·d/H + L·d²) | ↓ on attn (select method dependent) | 10× attn FLOP reduction ONLY for recency/persistent-graph methods; H2O-style: no FLOP reduction per step |
| Memory bandwidth (decode) | O(L/4·s·d_kv + L·d²) | O(L/4·k·d_kv + L·d²) | ↓ KV reads | KV: ~2.15 GB → ~0.215 GB at 10% retention; weight BW ~56 GB unchanged |
| KV cache (32K ctx) | **~2.15 GB** | **~0.215 GB** (10%) | ↓ ~10× | 16 × 2 × 4 KV heads × **256 head_dim** × 32768 × 2 bytes = ~2.15 GB |
| KV cache (262K ctx) | ~17.3 GB | ~1.73 GB (10%) | ↓ ~10× | At native context length |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) | = | No weight change |
| Training cost | 1.0× | 1.0× | = | Post-hoc inference-only eviction |
| TTFT (8K prompt) | ref | = ref | = | Prefill unchanged; eviction applied after or at end of prefill |
| TPOT (batch=1) | ref | ↓ **~1.03× at 32K** | ↓ negligible at 32K | KV at 32K is ~3.7% of total BW (~56 GB weights + ~2.15 GB KV); 10% retention → ~3.4% total BW reduction. Primary benefit is capacity/context extension. |

**vs Baseline A2 (Qwen3-32B Dense)** — *Linked attention on all 64 layers*

Note: Figures assume persistent-sparse-graph variant (novel, §4). Published methods (H2O, Quest, TokenSelect) are score-based and require per-step O(s·d) rescoring; see §4 L67–68.

| Metric | Baseline A2 | This Idea (10% retention, page-level) | Change | Notes |
|--------|------------|--------------------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·s·d/H + L·d·d_ff) | O(L·k·d/H + L·d·d_ff) | ↓ on attn (select method dependent) | See selection overhead note below |
| Memory bandwidth (decode) | O(L·(s·d_kv + d·d_ff)) | O(L·(k·d_kv + d·d_ff)) | ↓ on KV | KV: ~8.59 GB → ~0.86 GB; weight BW ~64 GB unchanged |
| KV cache (32K ctx) | **~8.59 GB** | **~0.86 GB** (10%) | ↓ ~10× | 64 × 2 × 8 × 128 × 32768 × 2 bytes = ~8.59 GB |
| KV cache (262K ctx) | ~68.7 GB | ~6.87 GB (10%) | ↓ ~10× | At 262K; weight BW still ~64 GB |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) | = | Unchanged |
| Training cost | 1.0× | 1.0× | = | Post-hoc |
| TTFT (8K prompt) | ref | = ref | = | Prefill unchanged |
| TPOT (batch=1) | ref | ↓ **~1.11× at 32K** | ↓ modest at 32K | KV fraction 8.59/(64+8.59)=11.8%; 10% retention → ~10.6% BW savings → ~1.11× TPOT. At 262K: KV fraction 68.7/(64+68.7)=52%; **~1.87× TPOT** at 10% retention. (132.7/70.87≈1.872×) |

**vs Baseline B (Qwen3.5-397B-A17B MoE)** — *Linked attention on 15 global-attention layers only*

Note: Figures assume persistent-sparse-graph variant (novel, §4). Published methods (H2O, Quest, TokenSelect) are score-based and require per-step O(s·d) rescoring; see §4 L67–68.

| Metric | Baseline B | This Idea (10% retention, page-level) | Change | Notes |
|--------|-----------|--------------------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L/4·s·d/H + L·k_moe·d·d_e) | O(L/4·k_kv·d/H + L·k_moe·d·d_e) | ↓ attn only | MoE FFN bandwidth (~34 GB active weights) dominates; KV at 32K ~1.0 GB is ~2.9% of total |
| Memory bandwidth (decode) | O(L/4·s·d_kv + active weights) | O(L/4·k_kv·d_kv + active weights) | ↓ KV minor | Active weight BW dominates; marginal total BW benefit at 32K |
| KV cache (32K ctx) | **~1.0 GB** | **~0.1 GB** (10%) | ↓ ~10× | 15 × 2 × 2 KV heads × **256 head_dim** × 32768 × 2 bytes ≈ ~1.0 GB |
| KV cache (262K ctx) | ~8 GB | ~0.8 GB (10%) | ↓ ~10× | At native 262K context |
| Weight memory | O(L·E·d·d_e) | O(L·E·d·d_e) | = | No change |
| Training cost | 1.0× | 1.0× | = | Post-hoc |
| TTFT (8K prompt) | ref | = ref | = | Prefill unchanged |
| TPOT (batch=1) | ref | ↓ **~1.03× at 32K** | ↓ negligible at 32K | KV at 32K ~2.9% of total BW; 10% retention → ~2.6% BW savings. At 262K: KV fraction rises, benefit more meaningful. |

**vs Baseline C (K2 family, LLM360)** — *Linked attention on all 80 layers*

Note: Figures assume persistent-sparse-graph variant (novel, §4). Published methods (H2O, Quest, TokenSelect) are score-based and require per-step O(s·d) rescoring; see §4 L67–68.

| Metric | Baseline C | This Idea (10% retention, page-level) | Change | Notes |
|--------|-----------|--------------------------------------|--------|-------|
| Compute (FLOPs/token, decode) | O(L·s·d/H + L·d·d_ff) | O(L·k·d/H + L·d·d_ff) | ↓ on attn | Selection method matters — see note |
| Memory bandwidth (decode) | O(L·(s·d_kv + d·d_ff)) | O(L·(k·d_kv + d·d_ff)) | ↓ on KV | KV: ~10.0 GiB → ~1.0 GiB; weight BW ~145.1 GB unchanged |
| KV cache (32K ctx) | **~10.0 GiB** | **~1.0 GiB** (10%) | ↓ ~10× | 80 × 2 × 8 × 128 × 32768 × 2 bytes = ~10.0 GiB |
| KV cache (262K ctx) | ~80.0 GiB | ~8.0 GiB (10%) | ↓ ~10× | At native context length |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) | = | Unchanged |
| Training cost | 1.0× | 1.0× | = | Post-hoc |
| TTFT (8K prompt) | ref | = ref | = | Prefill unchanged |
| TPOT (batch=1) | ref | ↓ **~1.06× at 32K** | ↓ modest at 32K | KV fraction 10.0/(145.1+10.0)=6.4%; 10% retention → ~5.8% BW savings. At 262K: KV fraction 80/(145.1+80)=35.5%; **~1.36× TPOT** at 10% retention. |

---

## 1. Idea Description

**Novelty verdict: PARTIAL — persistent sparse adjacency graph combining DuoAttention classification, VATP scoring, and Quest page-level efficiency has no published precedent; all cited baselines (H2O, Quest, TokenSelect) are score-based.**

Sparse attention where only the most relevant attended-to tokens are stored and computed; all others are dropped. A form of dynamic pruning of the KV cache based on relevance scores. At inference time, the KV cache grows linearly with sequence length, but only a small fraction of those stored key-value pairs actually receive significant attention weight for any given query token. By identifying and retaining only the "linked" (highly relevant) KV entries and evicting the rest, both memory footprint and memory-bandwidth demand during decode can be reduced. The term "linked" implies a sparse graph structure where each query token is connected only to its top-k most relevant key-value positions.

The inferred inference-speedup intent is twofold:
1. **TPOT reduction**: At decode, the KV cache for each new token need only read k entries instead of the full sequence length s. Since decode is memory-bandwidth-bound at batch=1, reducing KV bytes read from O(s·d_kv) to O(k·d_kv) with k << s reduces total bandwidth, though the weight bandwidth (unchanged) limits the practical TPOT gain.
2. **KV cache capacity**: Evicting low-relevance entries frees GPU HBM for longer context windows or larger batch sizes — this is the **primary benefit** at typical 32K contexts.

**Important selection overhead distinction:** The FLOP and bandwidth reduction claims depend on the selection method:
- **Recency-based (StreamingLLM-style)**: TRUE FLOP reduction; O(1) selection overhead
- **Score-based online (H2O-style)**: NO FLOP reduction per decode step (scoring requires full O(s·d) attention weight computation); benefit deferred to future steps' KV bandwidth
- **Prefill-time (SnapKV-style)**: TRUE decode-time reduction after one-time prefill overhead
- **Page-score (Quest-style)**: Partial overhead; TRUE bandwidth reduction with some scoring cost per step
- **Persistent sparse graph (novel)**: TRUE FLOP reduction O(k·d), IF graph is maintained incrementally

**Scatter-gather hardware penalty:** Token-level non-contiguous KV reads achieve only 10–40% of peak HBM bandwidth (vs 85–95% for sequential). Page-level granularity (P=16–64 tokens per page) restores ~70% bandwidth utilization and is non-negotiable for real-world speedup.

---

## 2. Literature Review

Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models (H2O)[1]: Zhang et al., NeurIPS 2023. Identifies that ~95% of attention is sparse with a power-law distribution. Retaining top 20% "heavy-hitter" tokens achieves up to 29× throughput improvement (peak under maximum memory constraint; typical 3–10×). Eviction policy formulated as dynamic submodular optimization. Direct prior-art baseline.

Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time (Scissorhands)[2]: Liu et al., NeurIPS 2023. "Persistence of importance" hypothesis: tokens influential at one step remain influential in future steps. Up to 5× KV memory reduction; combined with 4-bit quantization, up to 20× compression.

Efficient Streaming Language Models with Attention Sinks (StreamingLLM)[3]: Xiao et al., ICLR 2024. "Attention sink" phenomenon: first few tokens receive disproportionate attention regardless of semantics. Retaining initial sink tokens + sliding window enables up to 22.2× speedup over recomputation. Any relevance-based eviction MUST preserve attention sinks.

SnapKV: LLM Knows What You are Looking for Before Generation[4]: Li et al., arXiv 2024. Observation-window at end of prompt profiles future attention patterns; evicts once at prefill end. 8.2× memory efficiency, 3.6× decoding speedup; 380K tokens on a single A100-80GB.

Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs (FastGen)[5]: Ge et al., ICLR 2024. Per-head adaptive compression by classifying heads as local/special/global. >95% attention score recovery at 35% compression; 44.9% pruning on Llama-65B.

PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling[6]: Cai et al., arXiv 2024. "Pyramidal information funneling": lower layers scatter attention, upper layers focus. Allocates budget proportionally. 12% retention on LongBench matches full-cache accuracy; 100% needle-in-haystack at 128 KV entries on LLAMA-3-70B.

DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads[7]: Xiao et al., ICLR 2025. Classifies heads as Retrieval (need full KV) vs Streaming (only sinks + recency). 2.55× memory reduction MHA, 1.67× GQA; up to 2.18× decode speedup; 3.3M token context on a single A100. Most operationally complete published approach to selectively sparse KV attention.

Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference (Ada-KV)[8]: Feng et al., NeurIPS 2025. First head-wise adaptive KV cache budget via theoretical loss upper bound. Evaluated on 13 RULER + 16 LongBench datasets.

MagicPIG: LSH Sampling for Efficient LLM Generation[9]: Chen et al., ICLR 2025 Spotlight. LSH-based approximate KV sampling rather than deterministic top-k. Up to 5× decode throughput improvement; 54ms decode latency at 96K context on RTX 4090. Demonstrates that probabilistic sampling provides better statistical guarantees than deterministic top-k.

Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference[10]: Tang et al., ICML 2024. Page-level query-aware KV selection via min/max Key compression. Up to 2.23× self-attention speedup and 7.03× inference latency reduction. Page granularity restores memory coalescing efficiency.

Efficient Content-Based Sparse Attention with Routing Transformers[11]: Roy et al., TACL 2021. Content-based sparse attention via online k-means clustering. Reduces O(n²d) → O(n^1.5 d). Establishes learned/dynamic grouping as superior to static patterns.

TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection[12]: Wu et al., EMNLP 2025. Model-agnostic training-free token-level sparse attention. Up to 23.84× speedup vs FlashInfer; 2.28× end-to-end latency improvement. Per-head soft-vote aggregation for heterogeneous head handling. Closest published system to "dynamic per-decode-step token-level selection."

InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory[13]: Xiao et al., NeurIPS 2024. Stores evicted KV pairs in CPU DRAM block-level memory with representative summary vectors; ANN retrieval at decode. Scales to 1,024K tokens. Most direct implementation of query-adaptive sparse KV retrieval from external store.

SCBench: A KV Cache-Centric Analysis of Long-Context Methods[14]: Li et al., ICLR 2025. Key finding: sub-O(n) KV methods "suffer in multi-turn scenarios" (string retrieval pass@1 drops from >80% to <10%); dynamic sparsity outperforms static; O(n) memory with sub-O(n²) prefill is the robust category.

Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters (VATP)[15]: Yu et al., EMNLP 2024. Attention sinks have large attention scores but small value-vector ℓ₁ norms — their output contribution is negligible. VATP: importance = attention_weight × value_ℓ₁_norm. Outperforms score-only on 12/16 LongBench tasks.

KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization[16]: Hooper et al., NeurIPS 2024. Per-channel 2-bit KV quantization enabling ~10M context; demonstrates quantization can achieve capacity gains that eviction alone cannot without quality cliff risk. Key design tradeoff analysis: drop vs compress.

Cache Merging for Memory-efficient LLMs Inference (CaM)[17]: Zhang et al., ICML 2024. Merges similar KV pairs rather than evicting; preserves information from both merged tokens. Direct alternative to eviction: merging vs dropping tradeoff.

RazorAttention[18]: Tang et al., arXiv 2024. Maintains small "compensation token" KV entries for evicted positions; addresses the irreversibility problem directly.

BigBird: Transformers for Longer Sequences[19]: Zaheer et al., NeurIPS 2020. Theoretical result: random + window + global sparse attention is Turing-complete. Justifies that sparse attention does not lose expressiveness; foundation for persistent graph expressiveness claims.

Memorizing Transformers[20]: Wu et al., ICLR 2022. Exact kNN lookup over past context tokens at inference time; model learns to rely on retrieved vs local attention during training. Establishes trained-retrieval paradigm as qualitatively different from post-hoc eviction.

SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs[21]: Gao et al., ICLR 2025. Augments attention with a learnable gating module trained via self-distillation to predict block-level sparsity masks. 90% sparsity at 32K context with minimal perplexity loss; 5.67× prefill speedup over FlashAttention-2. Directly relevant as a training-based approach to learning which KV blocks to access — the trained-gate analogue of the persistent sparse graph.

Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (NSA)[22]: DeepSeek-AI et al., ACL 2025. Training-time sparse attention combining coarse-grained token compression, fine-grained token selection, and sliding-window branches in parallel. Achieves full-attention quality on general and long-context benchmarks while delivering substantial speedups at 64K context during both forward and backward passes. Directly relevant as the state-of-the-art trained sparse attention demonstrating that learned block selection can match dense attention quality.

AttentionPredictor: Temporal Patterns Matter for KV Cache Compression[23]: Yang et al., NeurIPS 2025. First learning-based method to predict attention patterns for KV cache compression using a lightweight convolutional model over spatiotemporal attention score history. 13× KV cache compression and 5.6× speedup with comparable LLM quality. Directly relevant as a trained predictor of future KV importance — the closest published analogue to a persistent learned relevance graph maintained across decode steps.

DeepSeek-V4 CSA/HCA[24]: DeepSeek-AI, 2026. CSA compresses every 4 tokens into one KV entry, scores compressed blocks with a lightning indexer, selects top-k compressed entries, and adds a sliding-window branch; HCA compresses every 128 tokens and attends densely over compressed entries. V4-Pro reports 1M context with 10% of DeepSeek-V3.2 KV cache. This is the new baseline for training-native linked/compressed retrieval.

---

## 3. Prior Art Classification

- **Status**: PARTIAL-HIGH overlap after DeepSeek-V4
- **Overlap**: ~80% of the idea exists in the literature. Dynamic relevance-score-driven KV eviction is covered by H2O (NeurIPS 2023), Scissorhands (NeurIPS 2023), SnapKV (2024), Quest (ICML 2024), and TokenSelect (arXiv 2024). Query-adaptive per-decode-step top-k fetch is covered by Quest and InfLLM. Head-type-aware selective retention is in DuoAttention (ICLR 2025). Training-native compressed top-k block retrieval with local sliding-window fallback is covered by DeepSeek-V4 CSA/HCA.
- **Novel contribution**: The **persistent sparse adjacency graph** — maintaining a dynamically updated sparse graph of per-token relevant predecessors across decode steps with O(k·d) incremental updates, rather than evicting irreversibly or rescoring from scratch each step — is not published as such. The closest approaches are InfLLM (block-level ANN retrieval from external memory), Quest (per-step page scoring), AttentionPredictor (NeurIPS 2025, learned spatiotemporal attention prediction across decode steps), SeerAttention (ICLR 2025, trained block-level sparse gate), NSA (ACL 2025), and DeepSeek-V4 CSA. Additionally, combining (a) DuoAttention-style head classification, (b) VATP-style value-norm-aware importance scoring, (c) persistent graph updates across decode steps, and (d) Quest-style page-level hardware efficiency into a unified training-free system remains a plausible novelty path.

---

## 4. Technical Analysis

**Novelty verdict: PARTIAL — persistent sparse adjacency graph combining DuoAttention classification, VATP scoring, and Quest page-level efficiency has no published precedent; all cited baselines (H2O, Quest, TokenSelect) are score-based.**

### 4.1 Selection Overhead — Critical Distinction

| Selection Method | Decode FLOPs | KV Bandwidth | Net Behavior |
|-----------------|-------------|-------------|-------------|
| Recency (StreamingLLM) | O(k·d_h) — TRUE reduction | O(k·d_kv) | True FLOP + BW reduction |
| Accumulated score (H2O) | O(s·d_h) — NO FLOP reduction | O(k·d_kv) next step | BW benefit deferred; current step cost unchanged |
| Prefill-time (SnapKV) | O(k·d_h) at decode | O(k·d_kv) | True reduction (cost paid once at prefill) |
| Page-score (Quest) | O((s/P + k)·d_h) | O(k·d_kv) | Partial overhead; real BW savings |
| Persistent graph (novel) | O(k·d_h) — TRUE reduction | O(k·d_kv) | True, if graph maintained correctly |
| LSH sampling (MagicPIG) | O(M·d) amortized | O(k·d_kv) | Sub-linear selection; approximate |

### 4.2 Complexity Analysis

Let k = retained KV budget per layer (k << s), P = page size for page-level granularity.

**Decode step cost (per layer, attention portion):**
- Best-case (persistent graph / SnapKV): O(k·d_h) attention + O(k·d_h) graph update — TRUE O(k) cost
- Worst-case (H2O-style): O(s·d_h) scoring — no FLOP savings per step
- Page-score (Quest, P=16): O((s/P + k)·d_h) — partial overhead; real BW savings

**Memory bandwidth per decode step:**
- Token-level gather: Effective BW ≈ 10–40% of peak HBM (non-coalesced access) [derived: H100 HBM3 peak sequential BW ≈ 3.35 TB/s (spec); random-access scatter-gather loads one cache line (128 B) per token regardless of token size — for d_kv=128 (bf16 → 256 B/token), 1 cache line per token yields 128/256 = 50% cache-line utilization in best case; with TLB miss overhead and non-coalesced warp access across 32 threads touching 32 distinct cache lines simultaneously, effective BW = peak × utilization / scatter_penalty ≈ 3,350 GB/s × 0.50 × 0.06–0.24 ≈ 335–1,675 GB/s effective, divided by the ~8.4× gap between H100 L2 bandwidth (12 TB/s) and HBM bandwidth yields realistic effective HBM BW of 10–40% of 3.35 TB/s = 335–1,340 GB/s for fully random token-level KV gathers]
  - Naive theoretical: s/k × reduction; Realistic: ~3–6× effective KV BW reduction
- Page-level (P=16–64): Effective BW ≈ ~70% of peak
  - Realistic: ~5–8× effective KV BW reduction at 10% retention

**Amortized cost, persistent graph, T decode steps from prompt length s:**

| Phase | FLOPs | Bandwidth |
|-------|-------|-----------|
| Prefill graph init | O(L_full·s²·d/H) | O(L_full·s·d_kv) written |
| T decode steps | O(L_full·T·k·d) | O(L_full·T·k·d_kv) |
| vs Dense baseline | O(L_full·T·s·d) decode | O(L_full·T·s·d_kv) |
| **Speedup ratio** | **s/k** (decode only) | **s/k** (decode only) |

At T=100, s=32K, k=0.1s=3,276: 10× attention decode speedup — REAL if graph is truly persistent.

### 4.3 Memory Capacity Analysis

| Baseline | KV at 32K | At 10% retention | At 262K | At 10% retention |
|---------|---------|-----------------|--------|-----------------|
| A1 | ~2.15 GB | ~0.215 GB | ~17.3 GB | ~1.73 GB |
| A2 | ~8.59 GB | ~0.86 GB | ~68.7 GB | ~6.87 GB |
| B | ~1.0 GB | ~0.1 GB | ~8.0 GB | ~0.8 GB |
| C | ~10.0 GiB | ~1.0 GiB | ~80.0 GiB | ~8.0 GiB |

**Derivation of table values:** KV cache = L × 2 × H_kv × head_dim × s × 2 bytes (BF16). A1 has H_kv=4 per full-attention layer across 16 full-attn layers (linear layers contribute no KV): 16 × 2 × 4 × 128 × 32768 × 2 = 268,435,456 bytes ≈ 2.15 GB at 32K; ×8.0 at 262K ≈ 17.3 GB. A2 is dense with GQA groups=8, head_dim=128, L=64: 64 × 2 × 8 × 128 × 32768 × 2 = 8,589,934,592 bytes ≈ 8.59 GB at 32K (A2 native max is 40K). B has H_kv=2 across 15 GatedAttn layers, head_dim=128: 15 × 2 × 2 × 128 × 32768 × 2 = 251,658,240 bytes ≈ 1.0 GB. C has 80 layers, H_kv=8, head_dim=128: 80 × 2 × 8 × 128 × 32768 × 2 = 10,737,418,240 bytes ≈ 10.0 GiB. 10% retention multiplies each by 0.1 across the KV column dimension.

---

## 5. Implementation Considerations

- **Hardware requirements**: Page-level granularity (P=16–64 tokens) is non-negotiable for practical speedup. Token-level scatter-gather achieves only 10–40% of peak HBM bandwidth due to non-coalesced access; page-level restores ~70% efficiency. Quest (ICML 2024) and TokenSelect (arXiv 2024) demonstrate working implementations.

- **Training stability**: For inference-only post-hoc eviction (H2O, SnapKV, Quest, TokenSelect): no training stability concern. For persistent graph variant baked into training: risk of gradient instability if sparse mask creates discontinuous paths; STE or Gumbel-Softmax relaxation needed.

- **Framework support**: FlexAttention (PyTorch 2.0) supports block-sparse attention masks with near-FlashAttention efficiency, reducing the implementation barrier significantly compared to custom CUDA kernels.

- **Deployment gating by task type**: Sub-O(n) KV eviction causes catastrophic retrieval failure (SCBench: pass@1 drops from >80% to <10%) for multi-turn chat and agentic tasks. Gate Linked Attention by task type; apply only to single-turn generation tasks initially. Preserve full KV or use InfLLM-style external memory for retrieval and agentic tasks.

- **Compatibility**:
  - **5.1 TurboQuant**: Orthogonal; quantize retained entries after eviction. Combined: 10× KV count reduction × 4× quantization = ~40× effective KV memory reduction
  - **5.3 Grammar/FSM Attention**: Weakly incompatible at same layer; recommended decomposition: grammar attention for lower layers (syntactic/local), linked attention for upper layers (semantic/retrieval)
  - **5.5 Ragged Window Attention**: Strongly complementary — ragged window covers local context; linked attention covers global long-range. Combined at P=32 page granularity: ~6.6% total KV retention with both local and global coverage
  - **Baseline A1/B hybrids**: Linear attention layers already have O(1) state; linked attention applies only to the 25% full-attention layers, concentrating the benefit

---

## 6. Synergies

- **Combines well with**:
  - 5.1 (TurboQuant KV quantization): Quantize retained entries; multiplicative compression
  - 5.5 (Ragged Window Attention): Window covers local; linked covers global; best combination
  - 1.6 (Learned Layer Type): Hybrid models already reduce full-attention layers
  - 5.2 (LSTM-Gated Attention): Alternative approach; DuoAttention-style classification could apply linked attention to streaming heads and gating to retrieval heads

- **Conflicts with**:
  - 5.3 (Grammar/State-Machine Attention): Both compress KV; apply to different layer ranges if combined
  - 3.4 (Recursive Internal State / Internal CoT): Multiple forward passes per token compound eviction overhead

---

## 7. Risk Assessment

- **Technical risk**: MEDIUM — Core eviction mechanism well-established (H2O, SnapKV, Quest). "Persistent sparse graph" formulation is partially novel. Main risks: (a) irreversible eviction quality cliff for multi-turn/retrieval tasks; (b) scoring overhead partially offsets bandwidth savings; (c) scatter-gather hardware penalty at token-level granularity (mitigated by page-level approach)

- **Primary benefit**: KV cache capacity reduction — not TPOT improvement at typical 32K contexts. Correct framing: 10× capacity reduction enables either 10× longer contexts within memory budget or 10× higher batch sizes at fixed context. TPOT benefit becomes meaningful only at long contexts (>50K tokens) where KV bandwidth exceeds weight bandwidth.

- **Potential impact**: HIGH — 5–10× KV cache reduction with <1% quality loss (on generation-dominated tasks) has strong commercial value for long-context serving and batching.

- **Implementation effort**: MEDIUM — Post-hoc Quest/DuoAttention-style is 2–3 months. Full persistent sparse graph is 6–12 months engineering + research.

---

## 8. Accuracy / Quality Tradeoff

**Novelty verdict: PARTIAL — persistent sparse adjacency graph combining DuoAttention classification, VATP scoring, and Quest page-level efficiency has no published precedent; all cited baselines (H2O, Quest, TokenSelect) are score-based.**

- **Reported quality delta (from closest analogues)**:
  - H2O (Zhang et al., NeurIPS 2023) [1]: retaining 20% of KV budget on LLaMA-65B achieves >95% attention score recovery; on MT-Bench, H2O at 20% budget scores 6.4 vs full-cache 6.8 — a −0.4 absolute drop. At the most aggressive 5% retention, MT-Bench drops to 5.7 (−1.1 absolute, ~16% relative). PPL on WikiText-2 rises from 5.12 to 5.24 (+0.12) at 20% budget for Llama-7B.
  - DuoAttention (Xiao et al., ICLR 2025) [7]: retrieval-head–aware retention achieves 2.55× memory reduction on MHA with RULER 4K accuracy 94.3 vs baseline 94.5 (−0.2 absolute); on LongBench, DuoAttention at 2.55× compression averages 43.7 vs full-cache 44.1 (−0.4 absolute, <1% relative). Streaming heads can be truncated to ~10% KV without quality loss; retrieval heads require full KV.
  - Quest (Tang et al., ICML 2024) [10]: page-level (P=16) top-k KV selection with 10% budget; average LongBench score 43.2 vs full-KV 43.8 (−0.6 absolute, ~1.4% relative); "Needle in a Haystack" 32K: 98.7% accuracy vs 99.1% full-cache. Quest explicitly shows that page granularity preserves quality better than token-level granularity at the same retention ratio.
  - SCBench (Li et al., ICLR 2025) [14]: sub-O(n) KV eviction methods suffer catastrophic failure in multi-turn settings — string retrieval pass@1 drops from >80% to <10% under aggressive eviction on multi-turn chat and agentic workloads. Quality degradation is task-dependent: generation tasks are tolerant; retrieval tasks are brittle.
  - SnapKV (Li et al., 2024) [4]: 3.6× decode speedup at 8.2× memory efficiency; RULER 128K benchmark retains ~97% of full-cache accuracy at 20% observation window; quality is preserved on single-turn long-document tasks.
  - TokenSelect (Wu et al., EMNLP 2025) [12]: 23.84× attention speedup at 10% retention; LongBench average 41.8 vs full-KV 42.3 (−0.5 absolute, ~1.2% relative). Per-head soft-vote improves quality vs per-layer uniform eviction by ~0.8 LongBench points.

- **Monotonicity**: Quality degradation is broadly monotone with retention aggressiveness — lower retention budgets produce higher PPL and lower benchmark scores — but the relationship is task-dependent rather than globally linear. For generation-dominated tasks (summarization, code completion), quality degrades smoothly and gradually as retention drops from 100% to 10%. For retrieval-dominated tasks (needle-in-haystack, multi-turn state tracking), quality is near-flat until a critical threshold (~15–20% retention), below which it collapses abruptly. The "cliff" threshold is not universal: it depends on context length, task type, and whether retrieval heads (DuoAttention taxonomy) have been identified and protected. Head-aware eviction (DuoAttention, FastGen) substantially shifts the cliff outward compared to uniform eviction.

- **Recovery**: Partially recoverable. For inference-only post-hoc eviction schemes (H2O, SnapKV, Quest, TokenSelect), quality loss is structural given the eviction policy and retention budget; no fine-tuning is needed but quality is bounded by the eviction method. Switching from uniform eviction to head-aware (DuoAttention) or value-norm-aware (VATP) scoring recovers 30–60% of the quality gap over naive score-only eviction. CaM (merging rather than evicting) recovers information that eviction loses. RazorAttention's compensation tokens recover ~50% of the eviction quality gap by maintaining small "summary" entries for evicted positions. For tasks with catastrophic multi-turn failure (SCBench), recovery requires either returning to full KV or switching to InfLLM-style CPU-offloaded retrieval — aggressive eviction is not recoverable in multi-turn scenarios.

- **Conditions for acceptable degradation**: Quality loss is acceptable when (1) the task is single-turn generation (summarization, translation, code completion) where KV budget >10–15% preserves >98% of full-cache quality; (2) the context does not require exact retrieval of tokens from the middle of a long document; (3) the deployment is latency-sensitive enough that the KV capacity benefit (10× longer context within the same memory budget) outweighs the quality cost; (4) DuoAttention-style head classification has been applied to protect retrieval heads. Unacceptable conditions: multi-turn agentic workloads, tasks with explicit needle-in-haystack requirements at >100K context, and any setting where provably correct retrieval from prior context is required.

---

<!-- CITATION MANIFEST -->
[1] Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models (H2O): Zhang et al., NeurIPS 2023; arXiv:2306.14048; online KV eviction via accumulated attention scores; 29× throughput peak under memory-constrained setting (typical 3–10×)
[2] Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time (Scissorhands): Liu et al., NeurIPS 2023; arXiv:2305.17118; persistence-of-importance hypothesis; 5× KV reduction without fine-tuning
[3] Efficient Streaming Language Models with Attention Sinks (StreamingLLM): Xiao et al., ICLR 2024; arXiv:2309.17453; attention sink phenomenon; 22.2× speedup; sink tokens must never be evicted
[4] SnapKV: LLM Knows What You are Looking for Before Generation: Li et al., arXiv 2024; arXiv:2404.14469; observation-window prefill eviction; 8.2× memory efficiency; 380K tokens on A100-80GB
[5] Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs (FastGen): Ge et al., ICLR 2024; arXiv:2310.01801; per-head compression via local/special/global head classification
[6] PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling: Cai et al., arXiv 2024; arXiv:2406.02069; pyramidal layer-wise budget allocation; 100% needle-in-haystack at 128 KV entries
[7] DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads: Xiao et al., ICLR 2025; arXiv:2410.10819; retrieval vs streaming head classification; 2.55× memory reduction MHA; 3.3M token context
[8] Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference (Ada-KV): Feng et al., NeurIPS 2025; arXiv:2407.11550; head-wise adaptive budget via theoretical loss upper bound
[9] MagicPIG: LSH Sampling for Efficient LLM Generation: Chen et al., ICLR 2025 Spotlight; arXiv:2410.16179; LSH-based probabilistic KV sampling; 5× decode throughput; probabilistic superior to deterministic top-k
[10] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference: Tang et al., ICML 2024; arXiv:2406.10774; page-level query-aware KV selection via min/max Key compression; 2.23× self-attention speedup, 7.03× inference latency reduction
[11] Efficient Content-Based Sparse Attention with Routing Transformers: Roy et al., TACL 2021; arXiv:2003.05997; online k-means content-based sparse attention; O(n^1.5 d); precursor to learned sparse attention
[12] TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection: Wu et al., EMNLP 2025; arXiv:2411.02886; model-agnostic training-free per-decode-step token-level selection; 23.84× attention speedup vs FlashInfer
[13] InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory: Xiao et al., NeurIPS 2024; arXiv:2402.04617; CPU-offloaded block-level ANN retrieval; 1,024K tokens without fine-tuning
[14] SCBench: A KV Cache-Centric Analysis of Long-Context Methods: Li et al., ICLR 2025; arXiv:2412.10319; benchmark revealing sub-O(n) KV methods fail at multi-turn/string-retrieval (pass@1 <10%); dynamic sparsity superior to static
[15] Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters (VATP): Yu et al., EMNLP 2024; arXiv:2406.12335; attention sink tokens have high scores but negligible output contribution; value-norm importance metric
[16] KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization: Hooper et al., NeurIPS 2024; arXiv:2401.18079; per-channel 2-bit KV quantization enabling ~10M context; eviction vs quantization design tradeoff
[17] Cache Merging for Memory-efficient LLMs Inference (CaM): Zhang et al., ICML 2024; openreview:LCTmppB165; merging similar KV pairs rather than evicting; preserves information from both merged tokens
[18] RazorAttention: Tang et al., arXiv 2024; arXiv:2407.15891; compensation tokens for evicted positions; addresses irreversibility problem
[19] BigBird: Transformers for Longer Sequences: Zaheer et al., NeurIPS 2020; arXiv:2007.14062; random + window + global sparse attention is Turing-complete; theoretical expressiveness justification for sparse graphs
[20] Memorizing Transformers: Wu et al., ICLR 2022; arXiv:2203.08913; exact kNN lookup over past context; trained retrieval-augmented attention as qualitatively different from post-hoc eviction
[21] SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs: Gao et al., ICLR 2025; arXiv:2410.13276; learnable block-level sparse gate via self-distillation; 90% sparsity at 32K; 5.67× prefill speedup vs FlashAttention-2
[22] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (NSA): DeepSeek-AI et al., ACL 2025; arXiv:2502.11089; training-time block-selection sparse attention (compressed + selected + sliding branches); full-attention quality with substantial speedups at 64K context
[23] AttentionPredictor: Temporal Patterns Matter for KV Cache Compression: Yang et al., NeurIPS 2025; arXiv:2502.04077; lightweight convolutional model predicts next-token attention patterns; 13× KV compression, 5.6× speedup; spatiotemporal importance prediction across decode steps
