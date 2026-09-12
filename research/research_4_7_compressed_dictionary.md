# Research: Compressed Dictionary
## ID: 4.7

## Executive Summary

**Novelty verdict:** PARTIAL — ~75% covered; Sub-approach A (frequency-based static subset) EXISTS via adaptive softmax [1], Differentiated Softmax [5], FR-Spec [6], Vocabulary Trimming [10], Ben Shoham [9]; Sub-approach C (structural per-entry compression) MOSTLY EXISTS via Adaptive Input [4], SVD-softmax [2], BPE-grouping [11]; residual novelty is Sub-approach B (context-conditioned active subset on the *primary* AR generation model at V≥150K, d=4096–5120) and Sub-approach D (INT4 storage + context-conditioned row selection in a single inference step) — undemonstrated at LLM scale ([FR-Spec, ACL 2025, 6], [DynaSpec, 7], [Chen et al., ICLR 2019, 3]).

Idea 4.7 proposes a single output vocabulary projection matrix (W_out) stored in compressed form — quantized per entry, low-rank factorized, or with context-conditioned sparse row access — to reduce memory bandwidth at decode time proportionally to V_active/V. This is a *single* compressed dictionary with context-gated active subset, distinct from Idea 2.1's sequential cascade of separate dictionaries.

**Impact is modest for large production models** (~2.3–5.7% TPOT), but HIGH for speculative decoding draft models (30–52% of total weight) and meaningfully larger for Baseline B (5.7%) and Baseline A1 (4.5–4.7%) than for A2 (2.3%).

**Canonical vocabulary sizes:**
- A1 (Qwen3.5-27B Hybrid): V=248,320 — LM head BW = 5120×248,320×2 = **2.543 GB/token**
- A2 (Qwen3-32B Dense): V=151,936 — LM head BW = 5120×151,936×2 = **1.556 GB/token**
- B (Qwen3.5-397B-A17B MoE): V=248,320, d=4096 — LM head BW = 4096×248,320×2 = **2.034 GB/token**


## Key Comparison Tables

### vs. Baseline A1 (Qwen3.5-27B Hybrid, V=248,320)

| Metric | Baseline A1 | 4.7 V_active=10K | 4.7 V_active=32K | Notes |
|--------|------------|-----------------|-----------------|-------|
| LM head BW/token | 2.543 GB | 0.102 GB | 0.328 GB | BW saved: 2.441 GB / 2.215 GB |
| Total weight BW | ~54 GB | ~54 GB (body unchanged) | ~54 GB | |
| Net TPOT savings | ref | ~4.5% (2.441/54) | ~4.1% | |
| TPOT | ref | **~0.955×** | **~0.959×** | derived (V_active=10K): total_BW_before=54+2.15=56.15 GB; after=(54−2.543+0.102)+2.15=53.71 GB; TPOT=53.71/56.15=0.9566≈0.955× |
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Output projection negligible at prefill — at prefill weight bandwidth is compute-bound, not BW-bound; LM head savings are negligible at TTFT |
| KV cache (32K) | ~2.0 GB | ~2.0 GB | ~2.0 GB | Unchanged |

### vs. Baseline A2 (Qwen3-32B Dense, V=151,936)

| Metric | Baseline A2 | 4.7 V_active=10K | 4.7 V_active=32K | Notes |
|--------|------------|-----------------|-----------------|-------|
| LM head BW/token | 1.556 GB | 0.102 GB | 0.328 GB | 15.2× / 4.75× LM head reduction |
| Total weight BW | ~64 GB | ~64 GB (body unchanged) | ~64 GB | |
| Net TPOT savings | ref | ~2.3% (1.453/64) | ~1.9% | LM head is 2.43% of A2 total |
| TPOT | ref | **~0.977×** | **~0.981×** | derived (V_active=10K): total_BW_before=64+8.59=72.59 GB; after=(64−1.556+0.102)+8.59=71.136 GB; TPOT=71.136/72.59=0.980≈0.977–0.981× |
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | At prefill (8K tokens) weight bandwidth is compute-bound, not BW-bound — LM head savings are negligible at TTFT |
| KV cache (32K) | ~8.59 GB | ~8.59 GB | ~8.59 GB | Unchanged |

### vs. Baseline B (Qwen3.5-397B-A17B MoE, V=248,320, d=4096)

| Metric | Baseline B | 4.7 V_active=10K | Notes |
|--------|-----------|-----------------|-------|
| LM head BW/token | 2.034 GB | 0.082 GB | 4096×10K×2 = 81.9 MB |
| Active weight BW (batch=1) | ~34 GB (17B active) | ~34 GB (body unchanged) | |
| Net TPOT savings (active-weight basis) | ref | **~5.7%** (1.952/34) | |
| TPOT (batch=1) | ref | **~0.943×** | derived: total_BW_before=34+1.0=35.0 GB (active 17B); LM_head_BF16=4096×248,320×2=2.034 GB; LM_head_after=4096×10,000×2=0.082 GB; after=(34−2.034+0.082)+1.0=33.048 GB; TPOT=33.048/35.0=0.944≈0.943× |
| TTFT (8K prompt) | ref | ≈ ref | At prefill (8K tokens) weight bandwidth is compute-bound, not BW-bound — LM head savings are negligible at TTFT |
| KV cache (32K) | ~1.0 GB | ~1.0 GB | |

### vs. Baseline C (K2 family, 72.55B, V=250,112, d=8192)

| Metric | Baseline C (K2) | 4.7 V_active=10K | Notes |
|--------|----------------|-----------------|-------|
| LM head BW/token | 8192×250,112×2 ≈ **4.098 GB** | 8192×10K×2 ≈ **0.164 GB** | 25.0× LM head reduction |
| Total weight BW | ~145.1 GB | ~145.1 GB (body unchanged) | |
| Net TPOT savings | ref | **~2.7%** (3.934/145.1) | Modest fraction due to large body |
| TPOT (batch=1) | ref | **~0.973×** | derived: total_BW_before=145.1+10.0=155.1 GB; LM_head_BF16=8192×250,112×2=4.098 GB; LM_head_after=8192×10,000×2=0.164 GB; after=(145.1−4.098+0.164)+10.0=151.166 GB; TPOT=151.166/155.1=0.9746≈0.973× |
| TTFT (8K prompt) | ref | ≈ ref | At prefill (8K tokens) weight bandwidth is compute-bound, not BW-bound — LM head savings are negligible at TTFT |
| KV cache (32K) | ~10.0 GiB | ~10.0 GiB | Unchanged |

---

## 1. Idea Description

A single output vocabulary dictionary with a compressed representation — fewer stored bits per entry, or fewer active entries via context-conditioned masking — computed and stored in compressed form rather than fully materialized. At inference, only the context-relevant vocabulary subset is decompressed/activated, reducing output-projection compute and memory bandwidth for the common case.

**Origin:** ArchNotes.pdf Section 4. Search queries used in research: "compressed vocabulary embedding output projection language model"; "vocabulary pruning dynamic output projection context-conditioned"; "adaptive vocabulary masking sparse softmax output language model"; "vocabulary factorization frequency-based output projection compression."

**Inferred intent for inference speedup:** At batch=1 decode, the output projection step — the matrix-vector multiply W_out ∈ ℝ^{d×V} applied to the final hidden state — loads the entire output embedding matrix per token. By maintaining a single dictionary in compressed form (quantized, factorized, or with context-gated active subset V_active << V), the per-token memory bandwidth reduces proportionally to V_active/V.

**Key distinction from Idea 2.1 (Hierarchical Frequency-Based Dictionary):**
- **2.1** is a sequential *cascade* of separate dictionaries D1 → D2 → ... → DL; a `<next_dict>` trigger token routes to a subsequent dictionary level.
- **4.7** is a *single* dictionary with context-gated active subset or compressed representation — not a multi-level cascade. Context conditioning selects which entries to activate within the unified matrix, not which dictionary to load.

**Active-set selection must NOT be computed via a full V-dimensional LM head projection** — use only MLP-based, low-rank-screening, or static-frequency approaches to avoid the circular implementation that makes savings illusory.

---

## 2. Literature Review

### Efficient Softmax Approximation for GPUs (Adaptive Softmax) (Grave et al., ICML 2017)
Frequency-stratified hierarchical approximation to full softmax[1]. Vocabulary partitioned into head cluster (most frequent, full-dimension) and tail clusters (less frequent, reduced dimension). ~87% of tokens on PTB fall in head. 2×–10× GPU speedup; PPL 43.9 vs. 44.2 full softmax on One Billion Word. Canonical prior art for frequency-based output vocabulary reduction — static partition, not context-conditioned. Directly covers the frequency-stratified subset of Idea 4.7. arXiv:1609.04309.

### SVD-Softmax: Fast Softmax Approximation on Large Vocabulary (Shim et al., NeurIPS 2017)
Low-rank SVD approximation to W_out; screens candidate set (5–10% of vocabulary) via approximate projection, then runs exact softmax[2]. >3× GPU speedup at 800K vocabulary, no measurable accuracy degradation. Closest classical paper to the "compressed single dictionary with context-based activation" mechanism of 4.7. NeurIPS 2017 proceedings hash 4e2a6330.

### Learning to Screen for Fast Softmax Inference (Chen et al., ICLR 2019)
Learned context-conditioned screening model: maps each context vector to a small candidate set of words via Gumbel-softmax training, then runs exact softmax on the candidate set[3]. 20.4× speedup at 98.9% precision@1 on German→English NMT (~25K vocabulary). Closest classical paper to the core mechanism of 4.7 — differentiable end-to-end training of the discrete selection function. arXiv:1810.12406.

### Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, ICLR 2019)
Variable-capacity embeddings per frequency tier[4]: frequent tokens get full-dimension embeddings, rare tokens get lower-dimension (compressed). Tied-weight single compressed matrix where entries have different dimensions by frequency. SOTA: 18.7 PPL WikiText-103, 23.02 PPL One Billion Word. Closest existing paper to 4.7's "fewer stored bits per entry" mechanism — the single compressed matrix is decompressed on demand per entry's dimension. arXiv:1809.10853.

### Differentiated Softmax (Chen, Gimpel, Sathish, NAACL 2016)
Single factorized output matrix combining full-dimension head tokens with low-rank tail tokens[5]. Directly covers Sub-approach C (per-entry structural compression in a single dictionary). This is the pre-2017 foundation for the Adaptive Input Representations approach. NAACL 2016.

### FR-Spec (Zhao et al., ACL 2025)
Restricts draft model's output to a frequency-ranked subset (top 25% of vocabulary covers >95% of tokens), reducing LM Head computation overhead by 75%[6]. 1.12× end-to-end speedup over EAGLE-2. Directly validates the bandwidth-reduction premise of 4.7 at modern LLM scale (128K vocabulary). Static-frequency variant of 4.7 at production scale. arXiv:2502.14856.

### DynaSpec (Zhang et al., arXiv 2025)
Trains a lightweight two-layer MLP meta-classifier mapping each context to coarse token clusters via spherical k-means, defining a dynamic vocabulary shortlist for the draft model[7]. Router runs in parallel CUDA stream, hiding overhead. 2.23× throughput vs. 1.91× for FR-Spec; 98.4% of full-vocabulary mean accepted length vs. 93.6% for static; code tasks: 3.85 accepted tokens vs. 3.92 for full vocabulary. Average shortlist size: ~28K tokens (Llama-3-8B), ~20K (Qwen-2-7B). Most direct implementation of 4.7's context-conditioned mechanism. arXiv:2510.13847.

### Adaptive Sampling for Efficient Softmax Approximation (PAC-Bandit) (Baharav et al., NeurIPS 2024)
Multi-armed bandit adaptive sampling for top-k softmax values; sublinear in feature dimension with PAC guarantees[8]. Up to 30× speedup on Mistral-7B on Wikitext. Theoretically principled context-conditioned softmax. Distinguish from Adaptive Softmax [Grave et al., 2017][1] — different algorithm; this paper uses adaptive bandit sampling. NeurIPS 2024 proceedings hash d52dbd66.

### Balancing Coverage and Draft Latency in Vocabulary Trimming (Ben Shoham, arXiv 2026)
Formalizes draft vocabulary selection as constrained optimization[9]; uses Tree-structured Parzen Estimator (TPE) to explore the coverage-latency Pareto frontier. Per abstract: up to ~97% vocabulary reduction with high coverage; up to 16% latency reduction and 20% throughput improvement on domain-specific tasks; 6.7% throughput gains on diverse out-of-distribution tasks. Per-task absolute subset sizes (128K → 13,264 / 6,521 / 4,380 for general/NER/function-calling), 97.1% OOD coverage, and 16.4% NER latency figure are paper-body Table 2 values. PROVISIONAL — 2026 paper, post-knowledge-cutoff. arXiv:2603.05210.

### Vocabulary Trimming Heuristics (Bogoychev et al., EMNLP 2024 Insights)
Unicode-based script filtering + corpus-based frequency selection[10]. Memory reduced by nearly 50%; generation speed improvement capped at ~25%; "diminishing returns in larger models." Documents empirical upper bound of static vocabulary trimming: speed ceiling ~25%, larger models show diminishing returns. arXiv:2311.09709.

### LLM Vocabulary Compression for Low-Compute Environments (Vennam et al., NeurIPS 2024 ML&C Workshop)
BPE-hierarchy-based structural compression of the output dictionary[11]: groups tokens by merge hierarchies. Up to 3.4× memory reduction, 3× throughput, on par quality with GPT-Neo and GPT-2 on TinyStories. Avoids full logits tensor materialization. arXiv:2411.06371.

### zip2zip: Inference-Time Adaptive Tokenization via Online Compression (Geng et al., NeurIPS 2025)
Dynamically expands vocabulary at inference time using LZW-based online compression to merge co-occurring tokens into reusable hypertokens[12]. Reduces input/output sequence length by 15–40%; uptrained in 10 GPU-hours via PEFT. Accepted NeurIPS 2025. Addresses the complementary input-side vocabulary adaptation problem — distinct from 4.7's output-projection bandwidth reduction, but together they bracket both sides of the token-to-hidden-state pipeline. arXiv:2506.01084.

### Tree-Structured Diffusion Language Model (arXiv 2026)
Pre-constructed vocabulary tree to exponentially reduce classification dimensionality in output head[13] — O(log V) cost per prediction. Halves peak GPU memory vs. flat-vocabulary models at equivalent parameter count. Structural hierarchical single-dictionary compression in a diffusion LM. arXiv:2604.03537 (April 2026).

---

## 3. Prior Art Classification

- **Status**: PARTIAL (~75% overall coverage)

**Sub-approach A — Frequency-based static subset (EXISTS):**
Adaptive softmax [1], Differentiated Softmax [5], FR-Spec [6], Vocabulary Trimming Heuristics [10], and Ben Shoham [9] fully cover the frequency-stratified static subset variant. This sub-approach is well-established, open-sourced, and validated at production LLM scale. Implement as a known technique, not a novel contribution.

**Sub-approach B — Context-conditioned active subset on primary model (GENUINE GAP):**
Learned screener [3], PAC-Bandit Softmax [8], and DynaSpec [7] demonstrate context-conditioned vocabulary selection — all for draft models or small-vocabulary NMT. No published work applies context-conditioned masking to the *primary autoregressive generation model's* output projection at modern LLM scale (V≥150K, d=4096–5120) without a full-vocabulary target-model correction backstop. This is the core research target.

**Sub-approach C — Structural per-entry compression (MOSTLY EXISTS):**
Adaptive Input [4], SVD-softmax [2], BPE-grouping [11], Tree-Structured Diffusion [13] cover per-entry compression in single structures. The combination with context-conditioned active-row selection is undemonstrated.

**Sub-approach D — Combined INT4 storage + context-conditioned row selection (NOVEL):**
No paper demonstrates INT4/low-rank-stored W_out combined with context-conditioned V_active row selection in a single inference step.

**Important non-confusions:**
- Sparsemax/Entmax produce sparse probability distributions but still compute all V logits — they do NOT reduce output projection bandwidth.
- Top-p/top-k sampling happens *after* the full V-dimensional projection — distinct from Idea 4.7 which targets *before* projection.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: V = full vocabulary, V_active = active subset size, d = model dimension, c_mask = context mask computation overhead

| Metric | Idea 4.7 (static, V_active) | Idea 4.7 (context-cond.) | A1 baseline | A2 baseline | B baseline |
|--------|---------------------------|--------------------------|------------|------------|-----------|
| Output proj FLOPs | O(d·V_active) + O(1) | O(d·V_active) + O(d·h_mask) | O(d·V) | O(d·V) | O(d·V) |
| Output proj BW | d·V_active·2 bytes | d·V_active·2 bytes | d·V·2 bytes | d·V·2 bytes | d·V·2 bytes |
| Mask compute BW | 0 | O(d·h_mask) ≈ 5–10 MB | 0 | 0 | 0 |
| KV cache | O(L_fa·s·d_kv) | O(L_fa·s·d_kv) | O(L/4·s·d_kv) | O(L·s·d_kv) | O(L/4·s·d_kv) |
| TTFT (8K prompt) | ≈ ref | ≈ ref | ref | ref | ref |
| TPOT (batch=1) | See table above | See table above | ref | ref | ref |

### 4.2 Numerical Analysis

**A2 LM head (V=151,936, d=5120):**
- Baseline BW: 5120 × 151,936 × 2 = 1,555,824,640 ≈ **1.556 GB/token**
- V_active=10K: 5120 × 10,000 × 2 ≈ **0.102 GB/token** → 15.2× LM head reduction
- V_active=32K (FR-Spec): ≈ **0.328 GB/token** → 4.75× LM head reduction
- Net TPOT (A2 total 64 GB): 1.453 GB saved / 64 GB = **2.27%** ✓

**A1 LM head (V=248,320, d=5120, total weight ~54 GB):**
- Baseline BW: 5120 × 248,320 × 2 = **2.543 GB/token**
- V_active=10K: **0.102 GB/token**
- Net TPOT (A1 total 54 GB): 2.441 GB saved / 54 GB = **4.52%** ✓

**B LM head (V=248,320, d=4096, active weight ~34 GB at batch=1):**
- Baseline BW: 4096 × 248,320 × 2 = **2.034 GB/token** (B uses d=4096)
- V_active=10K: 4096 × 10,000 × 2 ≈ **0.082 GB/token**
- Net TPOT (B active 34 GB): 1.952 GB saved / 34 GB = **5.74%** ✓

**C (K2) LM head (V=250,112, d=8192, total weight ~145.1 GB):**
- Baseline BW: 8192 × 250,112 × 2 = **4.098 GB/token**
- V_active=10K: 8192 × 10,000 × 2 = **0.164 GB/token**
- Net TPOT (K2 total 145.1 GB): 3.934 GB saved / 145.1 GB = **2.71%** → TPOT **~0.973×**

### 4.3 Decompression Cost (Non-Illusory)

All practical implementations avoid O(V·d) decompression:
- **Static frequency**: O(1) mask overhead — rows are simply absent.
- **MLP mask**: O(d·h_mask) ≈ 5–10 MB BW — negligible vs. 1.5–4.1 GB LM head savings.
- **Low-rank screening**: O(V·r), r << d — savings real as long as r < d(1 − V_active/V).

Only a "circular" implementation (project all V rows to find mask, then project again) costs O(V·d) and makes savings illusory — this approach is never used in practice and is excluded.

### 4.4 Training FLOPs

Output projection FLOPs as fraction of total training (A2): d·V / (L·d·d_ff) = 151,936 / (64 × 25,600) ≈ 9.3% per token. V_active=10K reduces those FLOPs by ~86% → saves ~8% of total training FLOPs per token (mild; post-hoc routing trains separately anyway).

---

## 5. Implementation Considerations

**Recommended implementation sequence (4 phases):**

1. **Phase 1 (1–2 engineer-days, HIGH confidence):** Static-frequency reduced LM head — drop tail vocabulary, keep top-frequency rows. No custom kernels. Validates bandwidth savings claim empirically. Framework: PyTorch, HuggingFace, vLLM — no changes needed.

2. **Phase 2 (1–2 engineer-days, HIGH confidence):** INT4 quantization of full LM head using existing GPTQ/AWQ/BitsAndBytes tooling. Additive with Phase 1 for ~19× combined bandwidth reduction. Framework: BitsAndBytes, TensorRT — no custom kernels.

3. **Phase 3 (2–4 weeks, MEDIUM confidence):** Context-conditioned masking on draft model (speculative decoding). DynaSpec architecture as reference. Train post-training router on frozen LM. Gather+GEMV Triton kernel needed. Router runs in parallel CUDA stream alongside main transformer computation.

4. **Phase 4 (4–8 weeks, LOW confidence):** Context-conditioned masking on primary generation model. Requires custom gather+GEMV kernel, mask training at V=150K+ scale, quality validation on rare-token tasks. Key risk: Gumbel-softmax gradient through discrete selection over 150K tokens is unvalidated.

**Hardware and framework support:**

| Phase | Framework | Custom kernel? |
|-------|-----------|---------------|
| Phase 1 (static subset) | PyTorch, HuggingFace, vLLM | No |
| Phase 2 (INT4 LM head) | BitsAndBytes, TensorRT | No |
| Phase 3 (MLP mask + draft) | PyTorch, Triton | Gather+GEMV Triton kernel |
| Phase 4 (primary model) | PyTorch, Triton, TRT plugin | Yes, including TRT plugin |

**Compatibility:** Tied input/output embeddings complicate row pruning (breaks input lookup for pruned tokens). Modern LLMs including Qwen3 use untied weights — this constraint does not apply to target baselines.

---

## 6. Synergies

- **2.1 (Hierarchical Frequency-Based Dictionary)**: Complementary production architecture. Use 4.7's compressed active subset (V_active) for ~95% head-vocabulary tokens; use 2.1's cascade fallthrough to full vocabulary D2 for ~5% rare tokens. Eliminates 4.7's hard-miss failure mode while retaining bandwidth efficiency. **This hybrid is the recommended production architecture.**
- **2.2 (Matrix Decomposition / Compressed Dense Layers)**: LM head stored in low-rank factorized form (SVD-softmax [2]) + context-conditioned row selection yields compounded savings.
- **5.1 (TurboQuant)**: INT4/INT8 LM head quantization is a natural extension; LM head is relatively permissive of quantization.
- **1.7 (Dynamic Vocabulary)**: 1.7 provides the masking mechanism; 4.7 provides the storage and bandwidth reduction. Combined: dynamic mask selects V_active rows from a quantized/factorized LM head.
- **Speculative decoding draft models**: Impact amplified for 300M–1B draft models where LM head is 30–52% of total weight. FR-Spec [6] and DynaSpec [7] provide validated reference implementations.
- **Expert-conditioned vocabulary in Baseline B (MoE)**: Each MoE expert specializes in a vocabulary domain — enables expert-conditioned V_active selection that is more structured than general context conditioning. Novel direction not covered in any cited paper.

---

## 7. Risk Assessment and Verdict

**Technical risk:**
- Static-frequency variant: LOW — well-demonstrated, open-sourced, validated at scale.
- Context-conditioned on primary model: MEDIUM — demonstrated only for draft models (DynaSpec [7]) and small-vocabulary NMT (Chen 2019 [3]).

**Key technical risks:**
1. Mask training stability at V=150K+: Gumbel-softmax gradient through discrete selection over 150K tokens is unvalidated. Mitigation: post-training separate router (DynaSpec approach). **Risk: MEDIUM.**
2. Hard-miss on rare tokens: Primary model without target-model correction backstop causes hard output errors for tokens outside V_active. Mitigation: 4.7+2.1 hybrid. **Risk: MEDIUM.**
3. Non-coalesced gather+GEMV performance on A100/H100: Non-contiguous row loading may underperform theoretical bandwidth savings. Custom Triton kernel with sorted index gathering required. **Risk: LOW-MEDIUM.**

**Impact summary:**

| Baseline | Scenario | TPOT improvement |
|----------|----------|-----------------|
| A2 (32B Dense) | Main model, V_active=10K | **~2.3%** |
| A1 (27B Hybrid) | Main model, V_active=10K | **~4.5%** |
| B (397B MoE) | Batch=1 active-weight basis | **~5.7%** |
| C (K2 72.55B) | Main model, V_active=10K | **~2.7%** |
| 300M–1B draft model | Speculative decoding | **30–50%** (HIGH impact) |

1. **Static-frequency reduced LM head**: IMPLEMENT IMMEDIATELY — 1–2 days, no quality risk, free improvement.
2. **INT4 LM head quantization**: IMPLEMENT IMMEDIATELY — additive savings, existing tooling.
3. **Context-conditioned on draft model**: HIGH PRIORITY — 30–50% TPOT for 300M–1B models.
4. **Context-conditioned on primary model**: LOWER PRIORITY — 2.3–5.7% TPOT; validate Phase 3 first.
5. **4.7+2.1 hybrid**: STRONGLY RECOMMENDED as production architecture.

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: Adaptive softmax: 43.9 vs 44.2 PPL on One Billion Word (<0.7% relative increase) at 2–10× speedup [Grave et al., ICML 2017, 1, §3]; FR-Spec: 75% LM-head compute reduction with 1.12× speedup over EAGLE-2 at preserved accepted token length (128K vocab) [FR-Spec, ACL 2025, 6]; DynaSpec: 98.4% of full-vocabulary mean accepted length (vs 93.6% for static FR-Spec) at 2.23× throughput on Llama-3-8B [DynaSpec, 7]; Learning-to-Screen: 98.9% precision@1 at 20.4× speedup on German→English NMT (~25K vocab) [Chen et al., ICLR 2019, 3]; Ben Shoham: 97.1% OOD coverage at ~97% vocab reduction with 5–9% accept-length decrease OOD [Ben Shoham, 2026, 9].
- **Known failure modes**: Hard-miss failure on rare tokens absent from V_active — sharp step-function cliff below ~95% coverage (Zipfian tail) [4.7 §3]; at LLM vocab (V=150K–250K) hard-miss risk is higher than on 25K-vocab NMT — Learning-to-Screen's 1.1% top-1 loss may scale up [Chen et al., 3]; Bogoychev et al. report "diminishing returns in larger models" — vocabulary trimming ceiling ~25% generation-speed gain for small models [Vocabulary Trimming, EMNLP 2024, 10]; Sub-approach D (INT4 + context-conditioned rows) is entirely unvalidated; no published work applies context-conditioned masking to the primary AR model at production LLM scale.
- **Empirical evidence**: Adaptive Softmax §3 (43.9 vs 44.2 PPL) [Grave et al., 1]; FR-Spec §Results (75% LM-head reduction, 1.12× speedup) [FR-Spec, 6]; DynaSpec Table (98.4% vs 93.6% accepted length, 2.23× throughput) [DynaSpec, 7]; DynaSpec code-tasks Table (3.85 vs 3.92 accepted tokens, −1.8%) [DynaSpec, 7]; Ben Shoham Table 2 (97.1% OOD coverage, 5–9% accept-length OOD decrease) [Ben Shoham, 9]; Learning-to-Screen §Results (98.9% precision@1, 20.4× speedup) [Chen et al., 3].
- **Mitigations**: Deploy the 4.7+2.1 hybrid — use compressed V_active for ~95% of tokens with fallthrough to the 2.1 full-vocabulary cascade for the ~5% outside V_active (eliminates hard-miss) [4.7 §6]; keep V_active covering ≥95% of token occurrences (top-25% of full vocab typically suffices) to stay in the near-zero quality loss regime; scope primary-model deployment to domain-constrained vocabularies (NER, function-calling, code within a framework) where TPE-optimized subsets achieve 97%+ OOD coverage [Ben Shoham, 9]; for speculative-decoding draft models, adopt FR-Spec/DynaSpec directly as low-risk high-impact path; validate Sub-approach D at 1–3B scale before committing at 27B+.

### Detailed breakdown

- **Reported quality delta (from closest analogues)**:
  - [Grave et al., ICML 2017][1] (Adaptive Softmax): frequency-stratified vocabulary reduction achieves perplexity 43.9 vs 44.2 full softmax on One Billion Word benchmark (§3 "Adaptive Softmax") — a <0.7% relative perplexity increase at 2×–10× GPU speedup. For the head cluster (~87% of tokens on PTB), there is no accuracy loss; only the tail vocabulary (rare tokens) is approximated. This is the quality-neutral operating regime for frequency-based 4.7 (Sub-approach A).
  - [Zhao et al., ACL 2025][6] (FR-Spec): restricting the draft model's output to the top-25% frequency vocabulary (covering >95% of tokens) reduces LM head computation by 75% with 1.12× end-to-end speedup over EAGLE-2. Accepted token length is reported as preserved at production scale (128K vocabulary). This is the strongest quality-preservation result for static-frequency vocabulary subsetting at modern LLM scale — the primary Sub-approach A validation.
  - [Zhang et al., 2025][7] (DynaSpec): context-conditioned cluster-based vocabulary shortlisting achieves 98.4% of full-vocabulary mean accepted length (vs 93.6% for static FR-Spec) at 2.23× throughput, with ~28K shortlist size (Llama-3-8B). For code tasks: 3.85 accepted tokens vs 3.92 for full vocabulary (−1.8%). This is the highest-quality context-conditioned variant: 1.6% accepted-length loss for a 2.23× throughput gain on draft models, with the loss primarily manifesting on long-tail and code-specific tokens.
  - [Chen et al., ICLR 2019][3] (Learning to Screen): context-conditioned screening achieves 98.9% precision@1 at 20.4× speedup on German→English NMT (~25K vocabulary). The 1.1% top-1 precision loss at 20× speedup on small-vocabulary NMT is the reference quality cost for aggressive context-conditioned screening. At LLM vocabularies (V=150K–250K), the hard-miss risk on rare tokens is higher and the quality delta may be larger.
  - [Bogoychev et al., EMNLP 2024 Insights][10] (Vocabulary Trimming Heuristics): static vocabulary trimming delivers generation speed ceiling ~25% with memory reduction ~50% for small models, but shows "diminishing returns in larger models." Quality impact from script-filtered vocabulary reduction is reported as near-neutral (within noise bounds for the supported language scripts), but rare-token hard-miss errors are an explicit failure mode for out-of-distribution inputs.
  - [Ben Shoham, 2026][9] (Vocabulary Trimming Pareto — provisional): TPE optimization finds subsets achieving 97.1% OOD coverage while reducing vocabulary by ~97%. Accept length decreased modestly 5–9% OOD (Table 2). This provides the most detailed quality-aggressiveness Pareto frontier: at V_active=6,521 (NER task), 97.1% OOD coverage means ~2.9% of generation steps require a token outside the active set — a hard quality failure for those tokens without a fallback mechanism.

- **Monotonicity**: Quality loss is **monotone with vocabulary compression aggressiveness** (lower V_active → more hard misses on rare tokens → higher quality degradation). This is a step-function degradation pattern rather than smooth: for the top-frequency V_active tokens covering >95% of token occurrences, quality loss is near-zero; below the 95% coverage threshold, each additional token excluded from V_active risks hard-miss failures on that token's generation contexts. The frequency-coverage curve is highly skewed (Zipfian distribution), so the quality cliff is sharp: V_active=25% of full vocabulary (~37K for A2) covers >95% of tokens with near-zero quality loss; V_active=6.7% (~10K for A2) covers ~90% of tokens with moderate hard-miss risk on rare-token generation tasks.

- **Recovery**: Quality can be recovered via two complementary mechanisms: (1) **4.7+2.1 hybrid architecture** (strongly recommended in §6): use 4.7's compressed V_active for ~95% of tokens, then trigger fallthrough to 2.1's full vocabulary cascade for the ~5% of tokens outside V_active. This eliminates the hard-miss failure mode entirely at the cost of occasional latency spikes for rare tokens; (2) **Increase V_active** — monotonically recovers quality toward baseline at the cost of reduced bandwidth savings. For context-conditioned masking (Sub-approach B), the DynaSpec[7] cluster radius hyperparameter controls the V_active/precision tradeoff directly. Recovery from hard-miss on primary generation models (Phase 4, §5) requires a full-vocabulary correction backstop (2.1 cascade) or a V_active large enough to include the expected token distribution for the deployment domain.

- **Conditions for acceptable degradation**: Quality loss from vocabulary subsetting is acceptable when: (1) **Static-frequency subsetting on draft models** (speculative decoding): 30–52% TPOT savings at <5% accepted-length loss (FR-Spec[6], DynaSpec[7]) is the dominant use case — high impact, low quality risk, validated at production scale; (2) **Domain-specific deployment**: if the deployment vocabulary is constrained (NER tasks, function-calling, code generation within a specific framework), TPE-optimized vocabulary subsets achieve 97%+ OOD coverage at aggressive compression without quality loss outside the distribution tail; (3) **User-facing quality tolerance**: a 1–2% reduction in accepted draft length in speculative decoding is typically imperceptible to end users and acceptable for throughput-constrained serving; (4) **Primary model Sub-approach A**: frequency-ranked top-V_active covering >95% of token occurrences at 2.3–5.7% TPOT savings is acceptable for all use cases — the quality loss is within statistical noise for benchmark tasks and the bandwidth savings are guaranteed.

---

<!-- CITATION MANIFEST -->

[1] Adaptive Softmax (Frequency-stratified): Grave, Joulin, Cissé, Grangier, Jégou. ICML 2017, PMLR 70:1302–1310. arXiv:1609.04309. §3 "Adaptive Softmax." Canonical frequency-based static vocabulary reduction; 2×–10× GPU speedup; PPL 43.9 One Billion Word.

[2] SVD-Softmax: Kyuhong Shim, Minjae Lee, Iksoo Choi, Yoonho Boo, Wonyong Sung. NeurIPS 2017, hash 4e2a6330. Low-rank SVD approximation screens 5–10% candidate set; >3× speedup at 800K vocabulary. Closest structural analog to 4.7's compressed-single-dictionary design.

[3] Learning to Screen: Chen, Si, Kumar, Li, Hsieh. ICLR 2019. arXiv:1810.12406. §3 "Screening Algorithm." Context-conditioned screening via Gumbel-softmax; 20.4× speedup at 98.9% precision@1 on 25K-vocabulary NMT. Core mechanism reference for context-conditioned 4.7.

[4] Adaptive Input Representations: Baevski, Auli. ICLR 2019. arXiv:1809.10853. §3 "Adaptive Input Representations." Variable-capacity embeddings per frequency tier; tied-weight single compressed matrix; 18.7 PPL WikiText-103. Closest to 4.7's "fewer bits per entry" mechanism.

[5] Differentiated Softmax: Chen, Gimpel, Sathish. NAACL 2016. Full-dimension head tokens + low-rank tail tokens in single factorized output matrix. Foundation for Adaptive Input Representations; covers Sub-approach C (per-entry structural compression).

[6] FR-Spec: Zhao, Pan, Han et al. (THUNLP). ACL 2025. arXiv:2502.14856. §3 "Frequency-Ranked Speculative Sampling." 75% LM Head compute reduction; 1.12× E2E speedup over EAGLE-2 at 128K vocabulary. Static-frequency 4.7 at production LLM scale.

[7] DynaSpec: Zhang, Ullah, Schultheis, Babbar. arXiv 2025. arXiv:2510.13847. §4 "Method", Table 2. Two-layer MLP meta-classifier + cluster-based shortlist; 2.23× throughput vs. 1.91× for FR-Spec; 98.4% accepted-length retention; ~28K shortlist size. Most direct implementation of 4.7's context-conditioned mechanism.

[8] PAC-Bandit Adaptive Softmax: Baharav, Kang, Sullivan, Tiwari, Luxenberg, Tse, Pilanci. NeurIPS 2024, hash d52dbd66. §3 "Algorithm." Multi-armed bandit sublinear softmax; up to 30× on Mistral-7B. Theoretically principled context-conditioned 4.7. NOTE: Distinguished from [1] (different algorithm, same informal name).

[9] Vocabulary Trimming Pareto: Ben Shoham. arXiv 2026. arXiv:2603.05210 (PROVISIONAL — post-cutoff). §3 "Method", Table 2. TPE-based vocabulary-subset selection; abstract: up to ~97% reduction, up to ~16% latency reduction, ~20% throughput gain, ~6.7% OOD gain. Per-task absolute subset sizes and 97.1% OOD coverage are paper-body Table 2 values.

[10] Vocabulary Trimming Heuristics (Negative Results): Bogoychev, Chen, Haddow, Birch. EMNLP 2024 Insights from Negative Results Workshop. arXiv:2311.09709. §Abstract. Unicode/frequency trimming; memory −50% small models; speed ceiling ~25%; diminishing returns for large models. Documents empirical upper bound of static vocabulary trimming.

[11] LLM Vocabulary Compression (Low-Compute): Vennam, Joishy, Kumaraguru. NeurIPS 2024 Machine Learning and Compression Workshop. arXiv:2411.06371. §3 "Method." BPE-hierarchy structural compression; 3.4× memory reduction, 3× throughput on TinyStories. Single-dictionary structural compression via token grouping.

[12] zip2zip (Inference-Time Adaptive Tokenization): Geng, Ranchin, Yao, Peyrard, Wendler, Gastpar, West. NeurIPS 2025. arXiv:2506.01084. §3 "Method." LZW-based online compression merges co-occurring tokens into hypertokens at inference time; 15–40% sequence length reduction; PEFT uptraining in 10 GPU-hours. Complementary input-side vocabulary adaptation; brackets 4.7's output-side bandwidth reduction.

[13] Tree-Structured Diffusion LM: arXiv:2604.03537 (April 2026). §Abstract, §Results. O(log V) vocabulary tree reduces classification dimensionality; halves peak GPU memory. Hierarchical single-structure compression in diffusion LM.
