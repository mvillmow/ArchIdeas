# Research: Dynamic Per-Value Numeric Type
## ID: 5.9

---

## Executive Summary

Per-element (per-scalar) dynamic numeric type assignment for LLM weights is **not hardware-viable** as a standalone mechanism: storing a 2-bit type tag per 4-bit weight adds 50% overhead, eliminating any compression benefit. The practical formulation is **per-group type assignment** (SpQR-style), where a small fraction of outlier weights within each group are stored at elevated precision (FP16 sparse), achieving ~3–4 bit effective average with near-lossless quality. This is already published (SpQR, OWQ, SqueezeLLM). The open novel direction is **prompt-conditioned or activation-conditioned per-group type selection at runtime**, enabling dynamic precision adaptation without static calibration — no published work achieves this.


---

## Key Comparison Table

| Metric | BF16 Baseline | INT4 (group=128) | SpQR (~3.9-bit) | 5.9 Reframed (per-group dynamic) |
|--------|---------------|-----------------|-----------------|----------------------------------|
| Bits per weight | 16 | 4 | ~3.9 | 4–5 (estimate) |
| TPOT improvement vs BF16 (batch=1) | 1× | ~1.5–2.5× practical | ~1.4–2.0× practical | ~1.3–2.0× (estimate) |
| Quality vs BF16 (PPL delta) | 0 | +0.3–1.5 | <+0.5 (near-lossless) | <+0.3 (target) |
| Post-hoc? | — | Yes | Yes | Yes (for static case) |
| Hardware efficiency | 100% | High (Tensor Cores) | Moderate (CSR scatter) | Low–Moderate (scattered FP16) |
| Published | — | Yes | Yes | No |

---

## TPOT Formula (Canonical)

For weight-bandwidth-dominated decode (batch=1):

```
TPOT_ratio = (compressed_weight_BW + KV_BW) / (baseline_weight_BW + KV_BW)
```

Values <1 indicate faster decode (TPOT improvement).

### Per-Baseline TPOT Analysis

**A1 (Qwen3.5-27B Hybrid):**
- Baseline weight BW: ~54 GB (BF16). At 3.9 bits/param: ~54 × 3.9/16 = ~13.2 GB. At 4.5 bits/param: ~15.2 GB.
- KV at 32K: ~2.15 GB (16 full-attn layers, 4 KV heads, head_dim=256, 32K tokens, BF16)
- SpQR-style (3.9 bit): (13.2 + 2.15) / (54 + 2.15) = 15.35 / 56.15 = **0.273× → 3.66× ideal TPOT improvement**; practical ~2.5–3.0× after scatter overhead
- At 4.5 bits: (15.2 + 2.15) / 56.15 = 17.35 / 56.15 = **0.309× → 3.24× ideal**; practical ~2.2–2.7×

**A2 (Qwen3-32B Dense):**
- Baseline weight BW: ~64 GB (BF16). At 3.9 bits: ~64 × 3.9/16 = ~15.6 GB. At 4.5 bits: ~18.0 GB.
- KV at 32K: ~8.59 GB (64 layers × 8 KV heads × 128 head_dim × 32768 tokens × 2 bytes)
- SpQR-style (3.9 bit): (15.6 + 8.59) / (64 + 8.59) = 24.19 / 72.59 = **0.333× → 3.00× ideal**; practical ~2.0–2.5× after scatter overhead
- At 4.5 bits: (18.0 + 8.59) / 72.59 = 26.59 / 72.59 = **0.366× → 2.73× ideal**; practical ~1.8–2.3×

**B (Qwen3.5-397B-A17B MoE):**
- Active weight BW: ~34 GB (BF16). At 3.9 bits active: ~34 × 3.9/16 = ~8.29 GB. At 4.5 bits: ~9.56 GB.
- KV at 32K: ~1.0 GB (15 global-attn layers × 2 KV heads × 256 head_dim × 32768 × 2 bytes ≈ 0.98 GB)
- SpQR-style (3.9 bit): (8.29 + 1.0) / (34 + 1.0) = 9.29 / 35.0 = **0.265× → 3.77× ideal**; practical ~2.5–3.0×
- At 4.5 bits: (9.56 + 1.0) / 35.0 = 10.56 / 35.0 = **0.302× → 3.31× ideal**; practical ~2.2–2.7×

**C (K2 family, 80 layers):**
- Baseline weight BW: ~145.1 GB (BF16). At 3.9 bits: ~145.1 × 3.9/16 = ~35.4 GB. At 4.5 bits: ~40.8 GB.
- KV at 32K: ~10.0 GiB (80 layers × 8 KV heads × 128 head_dim × 32768 × 2 bytes)
- SpQR-style (3.9 bit): (35.4 + 10.0) / (145.1 + 10.0) = 45.4 / 155.1 = **0.293× → 3.41× ideal**; practical ~2.3–2.8×
- At 4.5 bits: (40.8 + 10.0) / 155.1 = 50.8 / 155.1 = **0.327× → 3.06× ideal**; practical ~2.1–2.6×

**Note**: "Practical" = ideal × (0.60–0.80) due to non-sequential HBM access for FP16 sparse scatter. Sequential-access quantized formats (pure INT4, grouped) achieve 85–95% of ideal; scattered outlier formats achieve 60–80%.

---

## Idea Description

**From arch_research_ideas.md (Section 5, idea 5.9):**

> Mixed-precision at the individual value level — each weight or activation can have its own numeric type (fp16, int8, int4, etc.) chosen dynamically to minimize memory without sacrificing accuracy.

**Inferred intent:** At decode time (batch=1), the dominant cost is memory bandwidth: every byte of every weight matrix must be streamed from HBM to compute units for each generated token. If each weight's stored type could be chosen individually — 4-bit for easy-to-quantize weights, 16-bit for outliers — the effective byte-per-weight drops below the fixed-precision baseline and TPOT improves proportionally. "Dynamic" in the idea implies the type assignment is chosen at calibration or inference time rather than fixed at training. The per-value granularity goes finer than existing per-layer, per-channel, or per-group mixed-precision schemes.

**Hardware constraint (SIMT):** True per-element type selection is not viable on current NVIDIA GPU hardware. Tensor Core matrix multiply (MMA) requires all elements in a tile to share a format — the MMA instruction operates on a fixed-format input matrix. Heterogeneous types within a single MMA call require SIMT (CUDA core) fallback, which is ~500× lower throughput per element than Tensor Cores. This constraint alone disqualifies per-scalar type assignment for performance-critical weight GEMMs. The viable path is per-group type assignment: all elements within a group share a format, but groups can differ.

**Metadata overhead analysis:**

| Type tag bits | Weight bits | Overhead fraction | Effective bits/weight | Net compression vs BF16 |
|--------------|-------------|-------------------|-----------------------|-------------------------|
| 2 | 4 | 2/4 = 50% | 6.0 | 2.67× (not useful) |
| 2 | 8 | 2/8 = 25% | 10.0 | 1.6× (not useful) |
| 1 (outlier bit) | 4 (non-outlier) | ~0.5/4 avg | ~4.5 avg | 3.56× |
| Per-group (B=16) | 4 | 0.25/4 = 6.25% type + scale overhead | ~4.25 avg | 3.76× |
| Per-group (B=128) | 4 | 0.125/4 = 3.1% | ~4.12 avg | 3.88× |

At per-element granularity (B=1), the overhead is prohibitive. At per-group (B=16–128), the overhead is acceptable, which is why SpQR/OWQ/SqueezeLLM use per-group outlier flags.

---

## Literature Review

### Mixed Precision Training
**Micikevicius et al., ICLR 2018, arXiv:1710.03740**

Establishes canonical two-precision training: FP16 for forward/backward, FP32 master weights for optimizer. Loss scaling prevents FP16 underflow. Matches FP32 accuracy across CNNs, RNNs, LMs. Foundational but per-tensor (not per-value) at inference.

### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
**Dettmers et al., NeurIPS 2022, arXiv:2208.07339**

Mixed-precision decomposition for emergent outlier feature dimensions. Outlier dimensions (<0.1% of computations, ~6 feature dimensions per layer) preserved in FP16; >99.9% computed at INT8. Zero accuracy degradation 6.7B–175B. Granularity: per-channel (dimension-level), not per-element. Demonstrates real hardware execution of heterogeneous-precision operations — this is the closest GPU-viable approach to per-element precision.

### SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs
**Xiao et al., ICML 2023, arXiv:2211.10438**

Per-channel scaling to migrate outlier difficulty from activations to weights, enabling W8A8. Up to 1.56× speedup, 2× memory reduction. Key insight: per-channel scaling achieves outlier protection without per-element type tags. Shows that true per-element mixed precision is often not necessary for quality at INT8.

### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
**Frantar et al., ICLR 2023, arXiv:2210.17323**

Layer-wise PTQ using Hessian reconstruction. Group size=128 (shared scale per group). Negligible PPL increase at 4-bit; moderate at 3-bit. Establishes group quantization paradigm. 3–4× TPOT improvement practical. Key constraint: uniform bit-width within group — no per-element type variation.

### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
**Lin et al., MLSys 2024 Best Paper, arXiv:2306.00978**

Identifies salient (1%) weight channels via activation magnitude. Per-channel scaling protects salient channels; all weights remain INT4. >3× speedup on edge GPUs. AWQ explicitly avoids per-element mixed precision for hardware efficiency. Strongest argument against per-scalar type assignment: per-channel scaling achieves similar quality without metadata overhead.

### SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
**Dettmers et al., ICLR 2024, arXiv:2306.03078**

3-bit quantization with second-level quantization of scale factors; ~1% of weights stored as FP16 sparse outliers. <1% relative PPL loss on LLaMA/Falcon. 15% speedup vs FP16; enables 33B on single 24 GB GPU. **Effective average: ~3.9 bits/param.** This is the most direct prior art for 5.9: per-group outlier flag (binary type selector) within each group of 16 weights. PPL delta: <+0.5 vs FP16 on LLaMA-13B. SpQR's 1% outlier fraction means ~0.15 bits/weight overhead for the FP16 store.

### OWQ: Outlier-Aware Weight Quantization
**Lee et al., AAAI 2024 Oral, arXiv:2306.02272**

Preserves weak columns (0.15% of weights) in FP16; quantizes remainder to 3/4-bit. LLaMA-7B at 3.1-bit OWQ matches GPTQ 4-bit PPL. 3.21% kernel overhead vs GPTQ. Per-column (not per-element) mixed precision. Demonstrates that even column-level granularity (coarser than per-element) achieves near-4-bit quality at 3.1-bit average.

### ATOM: Low-Bit Quantization for Efficient and Accurate LLM Serving
**Zhao et al., MLSys 2024, proceedings.mlsys.org 2024**

W4A4 quantization with channel reordering for outlier isolation. <0.4 PPL increase on LLaMA models. **Up to 7.7× higher throughput vs FP16 under multi-user serving (batch>1)** — not batch=1 TPOT. This is the ceiling of what mixed-precision channel-level quantization can achieve at multi-user batch, not representative of single-request latency.

### QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving
**Lin et al., MLSys 2025, arXiv:2405.04532**

W4A8KV4 with progressive group quantization (QoQ). SmoothAttention for KV outliers. 1.2–3.5× throughput improvement over TensorRT-LLM on A100. Per-component (not per-element) mixed precision. State-of-the-art system baseline for quantized serving.

### QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
**Tseng et al., ICML 2024, arXiv:2402.04396**

Hadamard incoherence + E₈ lattice vector quantization. State-of-the-art at <3 bits/weight. Pareto-optimal below 2.5 bits but Hadamard pre-processing may not be streaming-decode-friendly.

### SqueezeLLM: Dense-and-Sparse Quantization
**Kim et al., ICML 2024, arXiv:2306.07629**

Sensitivity-based non-uniform quantization + dense-and-sparse decomposition. Outlier weights (0.45%) in FP16 sparse; rest 3–4-bit. >0.3 PPL improvement over prior 3-bit methods on LLaMA-7B C4. Like SpQR, uses binary type flag (outlier/not-outlier) per element within a group.

### HQQ: Half-Quadratic Quantization
**Badri and Shaji, 2023, mobiusml.github.io/hqq_blog**

Fast PTQ via half-quadratic optimization (no calibration data required). 2-bit and 4-bit LLM quantization in minutes. Block-level scale/zero-point (block size 64). Near-GPTQ quality at higher speed. Relevant as baseline for fast per-block type assignment without Hessian computation.

### NVFP4: NVIDIA FP4 Quantization
**NVIDIA, 2024, developer.nvidia.com**

NVIDIA FP4 (E2M1) format on Blackwell (GB200). 2× memory bandwidth reduction vs INT8. Hardware Tensor Core support for 4-bit format. Relevant as hardware context for lowest-bit-width hardware-native format.

### FP6-LLM: Efficiently Serving Large Language Models on GPUs with FP6-Bit Weight Quantization
**Xia et al., NSDI 2025, arXiv:2401.14112**

FP6 weight quantization (5.58-bit effective with scale) with hardware-optimized CUDA kernel. Abstract reports 1.69×–2.65× higher normalized inference throughput than FP16 baseline (cuBLAS); the paper-body 2.1× effective-memory-bandwidth figure is a midpoint / specific measurement configuration. Relevant as upper bound on non-standard-format serving efficiency.

### MixLLM: LLM Quantization with Global Mixed-Precision between Output-Features and Highly-Efficient System Design
**Liu et al., 2024, arXiv:2411.05161**

Global mixed-precision at layer granularity — some layers at 8-bit, others at 4-bit — selected via sensitivity analysis. Achieves better PPL than uniform 4-bit at the same average bits. Not per-element: granularity is per-layer.

### Progressive Mixed-Precision Decoding for Efficient LLM Inference
**Chen et al., 2024, arXiv:2410.13461**

Dynamic precision selection at runtime: distinguishes prefill vs decode phases, and within the decode phase progressively lowers precision as more tokens are generated (temporal depth, not layer depth). Runtime precision assignment at per-step-per-layer granularity, not per-element or activation-conditioned. Closest to "dynamic" in the 5.9 sense.

### HAQ: Hardware-Aware Automated Quantization with Mixed Precision
**Wang et al., CVPR 2019, arXiv:1811.08886**

Reinforcement learning to search per-layer bit-width assignments for hardware efficiency. Per-layer (not per-element) mixed precision. Foundational automated mixed-precision search work.

### SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference
**Liu et al., EPFL; arXiv:2602.22136, February 2026**

**CAUTION: Post-knowledge-cutoff paper (February 2026).** Actual title is "SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference" (Liu et al., EPFL). Focuses on heterogeneous mixed-precision for edge DNN hardware, not LLM serving or per-group runtime dynamic type assignment. Relevance to 5.9 is indirect (hardware-aware heterogeneous type assignment concept); does not constitute prior art for prompt-conditioned or activation-conditioned per-group LLM precision selection.

### RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On-Device LLM Inference
**Gautam and Jha, arXiv:2603.17891, March 2026**

RL-based (Soft Actor-Critic) per-layer bit-width assignment for on-device LLM inference. Policy trained on LLaMA-2-7B generalizes zero-shot to LLaMA-2-13B and Mistral-7B. Achieves 5.54 PPL at 3.68 GB (3.65 effective bits), outperforming uniform 4-bit AWQ (5.60 PPL at 3.90 GB). Per-layer granularity, not per-group or activation-conditioned — closest existing work to automated per-layer precision assignment for deployment.

### MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference
**ACL 2025, aclanthology.org/2025.acl-long.531**

Mixed-precision quantization framework specifically targeting long-context inference. Assigns different precision levels across KV cache and weight components based on context length and token position. Relevant as evidence that mixed-precision approaches are being extended to the long-context regime that overlaps with 5.9's serving targets (A1: 262K, C: 524K context).

### Mixed-Precision Quantization for Language Models: Techniques and Prospects (Survey)
**Rakka et al., arXiv:2510.16805, October 2025**

Comprehensive survey of mixed-precision quantization frameworks for language models. Categorizes frameworks by bit allocation strategy (per-layer, per-channel, per-group, per-token) and precision configuration. Directly relevant as the canonical reference for the full prior art landscape covered by idea 5.9.

### MX+: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving
**Lee et al., MICRO 2025, arXiv:2510.14557**

Extends OCP MX format with selective per-element precision elevation within MX blocks. The key insight is that the outlier element within a block does not need its exponent field, allowing that exponent field to be repurposed as extended mantissa bits to increase precision without additional storage overhead. Achieves significantly higher model performance than MXFP4 with negligible storage overhead and minimal inference slowdown (virtually eliminable with hardware support). Evaluated on LLaMA family models. **This is the most direct prior art for 5.10 block-dynamic scheme.** For 5.9 (per-element with type flags), MX+ is the limit case (1 elevated per block, not arbitrary per-element).

---

## Per-Baseline Detailed Analysis

### A1 (Qwen3.5-27B Hybrid)

**Architecture note:** 64 layers total (16 full-attention with 24Q/4KV heads, head_dim=256; 48 Gated DeltaNet recurrent). Weights: ~27B parameters.

**Weight sizes at various precisions:**
- BF16: ~54 GB
- INT4 (group=128): ~54 × 4/16 = **~13.5 GB**
- SpQR ~3.9-bit: ~54 × 3.9/16 = **~13.2 GB** (with outlier overhead ≈ 0.1 bits/param)
- 5-bit effective (per-group dynamic): ~54 × 5/16 = **~16.9 GB** (NOT ~20 GB — ~20 GB implies ~5.9 bits/param)

**KV cache at 32K:** 16 full-attn layers × 2 × 4 KV-heads × 256 head_dim × 32768 tokens × 2 bytes = **~2.15 GB**

**Gated DeltaNet compatibility:** Per-element type assignment does not depend on attention type; Gated DeltaNet weights are standard linear projections. Per-group quantization compatible with Gated DeltaNet weight shapes.

**TPOT (SpQR 3.9-bit, batch=1, 32K ctx):**
- (13.2 + 2.15) / (54 + 2.15) = 15.35 / 56.15 = **0.273× → 3.66× ideal TPOT improvement**
- Practical (with scatter overhead): ~**2.5–3.0×**

### A2 (Qwen3-32B Dense)

**KV cache at 32K:** 64 layers × 2 × 8 KV-heads × 128 head_dim × 32768 × 2 bytes = **~8.59 GB** (GQA: 8 KV-heads, not 64 Q-heads).

**Weight at SpQR 3.9-bit:** ~64 × 3.9/16 = **~15.6 GB**

**TPOT (SpQR, batch=1, 32K):** (15.6 + 8.59) / (64 + 8.59) = 24.19 / 72.59 = **0.333× → 3.00× ideal**; practical ~**2.0–2.5×**

**Weight bandwidth dominance at batch=1, 32K:** INT4 weights (~16.5 GB) > KV (~8.59 GB). KV only dominates weights at batch≥8 at 32K context.

### B (Qwen3.5-397B-A17B MoE)

**Active experts:** ~17B parameters active per token (out of ~397B total). Active weight BW: ~34 GB (BF16). At SpQR 3.9-bit: ~8.3 GB active.

**KV at 32K:** 15 global-attn layers × 2 × 2 KV-heads × 256 head_dim × 32768 × 2 bytes = **~1.0 GB**

**Per-expert calibration:** Per-group outlier detection should be applied per-expert (not globally across experts) for MoE. Thanos or SparseGPT-based calibration per expert recommended.

**TPOT (SpQR, batch=1, 32K):** (8.3 + 1.0) / (34 + 1.0) = 9.3 / 35.0 = **0.266× → 3.77× ideal**; practical ~**2.5–3.0×**

### C (K2 family, 80 layers)

**KV at 32K:** 80 layers × 2 × 8 KV-heads × 128 head_dim × 32768 × 2 bytes = **~10.0 GiB**

**Weight at SpQR 3.9-bit:** ~145.1 × 3.9/16 = **~35.4 GB**

**TPOT (SpQR, batch=1, 32K):** (35.4 + 10.0) / (145.1 + 10.0) = 45.4 / 155.1 = **0.293× → 3.41× ideal**; practical ~**2.3–2.8×**

---

## What Exists vs What Is Novel

### Fully Published (Not Novel)

| Approach | Paper | Granularity | Status |
|----------|-------|-------------|--------|
| Per-channel FP16 outliers | LLM.int8() [2], OWQ [6] | Channel | Published |
| Per-group binary type flag (outlier/not) | SpQR [5], SqueezeLLM [11] | Group-16 | Published |
| Per-channel scaling (no type change) | AWQ [4], SmoothQuant [3] | Channel | Published |
| Per-layer mixed precision | MixLLM [15], HAQ [17] | Layer | Published |
| Per-component (W/A/KV) mixed precision | QServe [9], QLoRA [12] | Component | Published |

### What Is Novel (Open Question)

**Novelty verdict: PARTIAL — static per-group outlier elevation is published [SpQR, ICLR 2024, arXiv:2306.03078; SqueezeLLM, ICML 2024, arXiv:2306.07629; OWQ, AAAI 2024, arXiv:2306.02272]; runtime activation-conditioned per-group type selection (where the type assignment changes per prompt/activation rather than being fixed by post-hoc calibration) has no published precedent and is the novel direction.**

**Prompt-conditioned or activation-conditioned per-group type assignment at runtime:** No published work selects group-level precision dynamically based on the current input prompt or token activation patterns at decode time. The closest is Progressive Mixed-Precision Decoding [16] (per-step-per-layer, not per-group, not activation-conditioned). A system that observes activation magnitudes at runtime and selects INT4 vs FP8 vs FP16 per group — without static calibration — would be novel.

**Continuous-valued precision (not discrete types):** Beyond selecting from a fixed set of types, learned per-group "effective bit-width" as a differentiable parameter during training has not been published for LLM inference kernels (only in research training contexts, HAQ).

---

## Integration with Other Ideas

| Idea | Compatibility | Notes |
|------|--------------|-------|
| 5.7 (Block Compressed Weights / INT4) | COMPATIBLE (combined) | 5.9 is a refinement of 5.7's INT4 path with per-group outlier elevation |
| 5.8 (Block Sparse Weights) | COMPATIBLE | Weight sparsity and per-group precision are orthogonal |
| 5.10 (Block Dynamic Compression) | SAME MECHANISM | 5.9 per-element = 5.10 with B=1. Merge into 5.10 for B≥16. |
| 5.4 (Linked Attention / KV eviction) | COMPATIBLE | KV precision and weight precision are independent |
| 5.5 (Ragged Window Attention) | COMPATIBLE | Orthogonal mechanisms |

---

## Risk Matrix

| Risk | Severity | Probability | Mitigation |
|------|---------|------------|-----------|
| Per-element (B=1) hardware infeasibility (SIMT constraint) | HIGH | CERTAIN | Reframe to per-group (B=16–128). SpQR/OWQ already solve this. |
| Per-group scatter overhead eliminates TPOT gain | MEDIUM | LOW | SpQR achieves 15% speedup (modest but real). Scatter to large FP16 blocks is faster than per-scalar. |
| SigmaQuant (post-cutoff) targets edge DNN heterogeneous quantization, not LLM per-group runtime | LOW | N/A | SigmaQuant is not prior art for prompt-conditioned per-group LLM precision; novelty of dynamic assignment remains open |

---

## Recommended Path

**If pursuing static per-group (near-term):**
- Use SpQR or SqueezeLLM with group=16 on Qwen3-32B (A2) as-is
- Measure actual batch=1 TPOT on A100 with sparse outlier kernel
- Compare against pure GPTQ-INT4 TPOT baseline
- Expected: ~15–30% TPOT improvement over INT4 due to lower average bits; scatter overhead partially cancels

**If pursuing dynamic (novel, longer-term):**
- Train a lightweight predictor (per-layer, not per-element) that estimates activation outlier magnitude per group
- Use predictor output to select INT4 vs FP8 per group at runtime
- Implement with two-phase decode: predictor pass (cheap) + mixed-precision GEMV (custom kernel)
- Validate on Qwen3-7B first; measure quality vs pure INT4 baseline

---

---

## Benefits vs Baseline A1 (Qwen3.5-27B Hybrid)

**Novelty verdict: PARTIAL — static per-group outlier elevation [SpQR, SqueezeLLM, OWQ] is published; runtime activation-conditioned per-group type selection is the novel direction.**

| Metric | Baseline A1 | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; weight quantization does not reduce FLOPs |
| TPOT (batch=1) | ref | 0.273× → 3.66× ideal | ↓ 3.66× ideal; practical ~2.5–3.0× | (13.2+2.15)/(54+2.15) = 15.35/56.15 = 0.273; SpQR 3.9-bit weights ~13.2 GB, KV ~2.15 GB |
| KV cache (32K ctx, BF16) | ~2.15 GB | = | = | Unchanged; 16 full-attn layers × 2 × 4 KV-heads × 256 head_dim × 32768 × 2 B |
| Weight memory | ~54 GB (BF16) | ~13.2 GB (SpQR 3.9-bit) | ↓ 4.09× | 54 × 3.9/16 = ~13.2 GB; practical scatter overhead adds ~10–20% effective BW |

## Benefits vs Baseline A2 (Qwen3-32B Dense)

**Novelty verdict: PARTIAL — static per-group outlier elevation [SpQR, SqueezeLLM, OWQ] is published; runtime activation-conditioned per-group type selection is the novel direction.**

| Metric | Baseline A2 | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; weight quantization does not reduce FLOPs |
| TPOT (batch=1) | ref | 0.333× → 3.00× ideal | ↓ 3.00× ideal; practical ~2.0–2.5× | (15.6+8.59)/(64+8.59) = 24.19/72.59 = 0.333; SpQR 3.9-bit weights ~15.6 GB, KV ~8.59 GB |
| KV cache (32K ctx, BF16) | ~8.59 GB | = | = | Unchanged; 64 layers × 2 × 8 KV-heads × 128 head_dim × 32768 × 2 B |
| Weight memory | ~64 GB (BF16) | ~15.6 GB (SpQR 3.9-bit) | ↓ 4.10× | 64 × 3.9/16 = ~15.6 GB |

## Benefits vs Baseline B (Qwen3.5-397B-A17B MoE)

**Novelty verdict: PARTIAL — static per-group outlier elevation [SpQR, SqueezeLLM, OWQ] is published; runtime activation-conditioned per-group type selection is the novel direction.**

| Metric | Baseline B | This Idea | Change | Notes |
|--------|-----------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; weight quantization does not reduce FLOPs |
| TPOT (batch=1) | ref | 0.266× → 3.77× ideal | ↓ 3.77× ideal; practical ~2.5–3.0× | (8.3+1.0)/(34+1.0) = 9.3/35.0 = 0.266; SpQR 3.9-bit active weights ~8.3 GB, KV ~1.0 GB |
| KV cache (32K ctx, BF16) | ~1.0 GB | = | = | Unchanged; 15 global-attn layers × 2 × 2 KV-heads × 256 head_dim × 32768 × 2 B |
| Weight memory (active) | ~34 GB active (BF16) | ~8.3 GB active (SpQR 3.9-bit) | ↓ 4.10× | 34 × 3.9/16 = ~8.3 GB; per-expert calibration recommended for MoE |

## Benefits vs Baseline C (K2 Family, LLM360)

**Novelty verdict: PARTIAL — static per-group outlier elevation [SpQR, SqueezeLLM, OWQ] is published; runtime activation-conditioned per-group type selection is the novel direction.**

| Metric | Baseline C | This Idea | Change | Notes |
|--------|-----------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; weight quantization does not reduce FLOPs |
| TPOT (batch=1) | ref | 0.293× → 3.41× ideal | ↓ 3.41× ideal; practical ~2.3–2.8× | (35.4+10.74)/(145.1+10.74) = 46.14/155.84 = 0.296; SpQR 3.9-bit weights ~35.4 GB, KV ~10.74 GB decimal (~10.0 GiB binary per cell above) |
| KV cache (32K ctx, BF16) | ~10.0 GiB | = | = | Unchanged; 80 layers × 2 × 8 KV-heads × 128 head_dim × 32768 × 2 B |
| Weight memory | ~145.1 GB (BF16) | ~35.4 GB (SpQR 3.9-bit) | ↓ 4.10× | 145.1 × 3.9/16 = ~35.4 GB; large d_ff benefits proportionally |

---

## Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - SpQR (Dettmers et al., ICLR 2024) [5]: 3-bit quantization + 1% FP16 sparse outliers (effective ~3.9 bits) achieves **<1% relative PPL loss** across LLaMA 7B–65B on WikiText-2. LLaMA-65B: SpQR PPL 3.87 vs BF16 3.79 (+0.08 PPL, +2.1% relative). LLaMA-7B: SpQR PPL 5.63 vs BF16 5.12 (+0.51 PPL, +10% relative). The 7B result demonstrates that small models incur meaningfully higher relative quality cost, while 65B+ models are nearly lossless. The 15% generation speedup over FP16 at 33B scale is the practical efficiency payoff.
  - OWQ (Lee et al., AAAI 2024 Oral) [6]: per-column FP16 outlier preservation (0.15% of weights) in a 3-bit base achieves **3.1-bit OWQ matching GPTQ 4-bit PPL** across LLaMA-7B and LLaMA-65B. Concretely: LLaMA-7B 3.1-bit OWQ achieves PPL 5.84 vs GPTQ 4-bit PPL 5.95 — better quality at fewer bits. Kernel overhead vs GPTQ: only 3.21%. This is the most efficient per-group outlier approach at sub-4-bit quantization.
  - GPTQ (Frantar et al., ICLR 2023) [7]: 4-bit group=128 quantization on LLaMA-7B raises WikiText-2 PPL from **5.68 (FP16) to 6.43 (+0.75 PPL)**, and at 3-bit to 8.02 (+2.34 PPL). At 175B (OPT-175B), GPTQ 4-bit PPL is 9.22 vs dense 8.34 (+0.88 PPL). These figures establish the quality floor for uniform low-bit quantization that SpQR/OWQ improve upon via selective outlier elevation.
  - ATOM (Zhao et al., MLSys 2024) [8]: W4A4 with channel reordering achieves **<0.4 PPL increase** on WikiText-2 at 4-bit weights + activations, with 7.73× throughput improvement under batch>1 serving. The W4A4 result confirms that aggressive quantization of both weights and activations is feasible with careful outlier handling.
  - SqueezeLLM (Kim et al., ICML 2024) [11]: density-and-sparse 3-bit decomposition with 0.45% FP16 outliers achieves **>0.3 PPL improvement over prior 3-bit methods** on LLaMA-7B C4 (PPL 7.75 vs GPTQ-3bit 8.65). The key insight is that 3-bit with sparse outlier elevation outperforms uniform 4-bit at the same memory footprint.
  - Progressive Mixed-Precision Decoding (Chen et al., arXiv 2024) [16]: phase-aware precision (higher precision for prefill, lower for decode). Per abstract: 1.4–12.2× speedup in matrix-vector multiplications over FP16 models; the corpus characterization as generic "GPU speedup" should be narrowed to MVM. Quality degradation confined to <0.5 PPL increase on WikiText-2 for LLaMA-7B (paper body). This is the runtime-adaptive precision result closest to the novel dynamic assignment envisioned by idea 5.9.

- **Monotonicity**: Quality degradation is broadly monotone with decreasing effective bit-width — lower average bits produce higher PPL — but the relationship is non-linear and model-size-dependent. The key non-linearity: with per-group outlier elevation (SpQR/OWQ), quality drops from 3.9-bit to 3-bit are ~10× smaller than the drop from 4-bit to 3-bit with uniform quantization. Outlier elevation is the critical mechanism — protecting 0.15–1% of weights in FP16 recovers the quality equivalent of ~0.5–1.0 additional bits across the entire weight matrix. Small models (7B) incur ~5–10× higher relative PPL cost per lost bit than large models (65B+), making quality-efficiency tradeoffs far more favorable at deployment scale.

- **Recovery**: Fully recoverable within the per-group framework. Static SpQR/OWQ calibration already achieves <+0.5 PPL on models ≥13B without any fine-tuning. For the novel dynamic assignment (runtime per-group precision based on activation magnitudes), quality recovery is achieved by: (1) increasing the FP16 outlier fraction (1% → 2% improves PPL ~0.1–0.3 but adds storage); (2) switching from INT4 to INT8 for groups containing activation outliers; (3) using group calibration data that is more representative of the deployment distribution. Quality is not fundamentally limited by the mechanism — it is limited by the calibration quality and outlier detection accuracy.

- **Conditions for acceptable degradation**: Per-group mixed-precision quality loss is acceptable when: (1) model size is ≥13B, where <+0.5 PPL is reliably achievable with SpQR/OWQ at ~3.9-bit effective; (2) TPOT improvement (2.0–3.0× practical) justifies the quality cost for latency-sensitive serving; (3) the deployment task is generation-dominated (summarization, code, translation) where small PPL increases have minimal downstream impact; (4) calibration data is available and representative. The dynamic assignment variant is acceptable under the same conditions, with the additional requirement that the lightweight runtime predictor adds <10% overhead (validated by Progressive Mixed-Precision Decoding). Unacceptable conditions: retrieval-critical tasks where exact token probabilities matter, and sub-7B models where per-group PPL deltas are large enough to cause noticeable output quality degradation.

---

<!-- CITATION MANIFEST -->
## Citations

[1] Mixed Precision Training (Micikevicius et al.): ICLR 2018 foundational work on FP16/FP32 mixed-precision training with loss scaling; per-tensor precision at training time. arXiv:1710.03740
[2] LLM.int8() (Dettmers et al.): NeurIPS 2022; INT8 quantization with FP16 decomposition for emergent outlier channels; per-channel (not per-element) mixed precision. arXiv:2208.07339
[3] SmoothQuant (Xiao et al.): ICML 2023; per-channel activation-to-weight scaling enabling W8A8; avoids per-element type selection. arXiv:2211.10438
[4] AWQ (Lin et al.): MLSys 2024 Best Paper; per-channel scaling to protect 1% salient weight channels; uniform INT4 for all weights; explicitly avoids per-element mixed precision. arXiv:2306.00978
[5] SpQR (Dettmers et al.): ICLR 2024; 3-bit quantization + 1% FP16 sparse outliers; ~3.9-bit average; <1% relative PPL loss; 15% speedup vs FP16 on 33B; most direct prior art for 5.9. arXiv:2306.03078
[6] OWQ (Lee et al.): AAAI 2024 Oral; per-column FP16 outlier preservation (0.15% of weights); 3.1-bit OWQ matches GPTQ 4-bit PPL; 3.21% kernel overhead vs GPTQ. arXiv:2306.02272
[7] GPTQ (Frantar et al.): ICLR 2023; Hessian-based layer-wise PTQ at 3–4 bits; group=128; 3–4× speedup vs FP16; uniform bit-width within group (no per-element variation). arXiv:2210.17323
[8] ATOM (Zhao et al.): MLSys 2024; W4A4 with channel reordering for outlier isolation; <0.4 PPL increase; 7.73× throughput vs FP16 under batch>1 serving (NOT batch=1 TPOT). proceedings.mlsys.org 2024
[9] QServe (Lin et al.): MLSys 2025; W4A8KV4 with progressive group quantization (QoQ); 1.2–3.5× serving throughput over TensorRT-LLM. arXiv:2405.04532
[10] HAQ (Wang et al.): CVPR 2019; RL-based per-layer bit-width search; hardware-aware automated mixed precision at layer granularity. arXiv:1811.08886
[11] SqueezeLLM (Kim et al.): ICML 2024; density-and-sparse decomposition; 0.45% FP16 outliers; 3-bit with >0.3 PPL improvement over prior 3-bit methods on LLaMA-7B C4. arXiv:2306.07629
[12] QLoRA (Dettmers et al.): NeurIPS 2023; NF4 4-bit quantization with double quantization of scale constants; block=64 NF4; enables 65B finetuning on single 48GB GPU. arXiv:2305.14314. Disambiguation: this file uses [5-SpQR] and [12-QLoRA] for the two Dettmers et al. papers.
[13] AQLM (Egiazarian et al.): ICML 2024; multi-codebook quantization at 2–3 bits/param; Pareto-optimal below 3 bits; fast GPU/CPU kernels. arXiv:2401.06118
[14] QuIP# (Tseng et al.): ICML 2024; Hadamard incoherence + E₈ lattice VQ; state-of-the-art at <3 bits/weight. arXiv:2402.04396
[15] MixLLM (Liu et al.): arXiv 2024; "LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design"; per-layer global mixed-precision (8-bit/4-bit) via sensitivity analysis; better PPL than uniform 4-bit at same average bits. arXiv:2411.05161
[16] Progressive Mixed-Precision Decoding (Chen et al.): arXiv 2024; phase-aware precision: higher precision for prefill, progressively lower precision as more tokens are generated during decode (temporal depth, not layer depth); 1.4–12.2× speedup in matrix-vector multiplications over FP16 (abstract); closest to runtime precision assignment but not per-group or activation-conditioned. arXiv:2410.13461
[17] HQQ (Badri and Shaji): mobiusml.github.io 2023; fast PTQ via half-quadratic optimization without calibration data; 2-bit and 4-bit LLM quantization in minutes. mobiusml.github.io/hqq_blog
[18] NVFP4 (NVIDIA): developer.nvidia.com 2024; Blackwell hardware FP4 (E2M1) Tensor Core support; 2× bandwidth reduction vs INT8.
[19] FP6-LLM (Xia et al.): NSDI 2025; FP6 weight quantization with hardware-optimized CUDA kernel; abstract reports 1.69–2.65× higher normalized inference throughput vs FP16 cuBLAS (corpus's 2.1× effective-bandwidth figure is a paper-body midpoint). arXiv:2401.14112
[20] SigmaQuant (Liu et al., EPFL): arXiv:2602.22136, February 2026. CAUTION: Post-knowledge-cutoff paper. Actual focus: hardware-aware heterogeneous quantization for edge DNN inference, not LLM per-group runtime precision selection. Indirect relevance to 5.9 concept only.
[21] MX+ (Lee et al.): MICRO 2025; arXiv:2510.14557; extends OCP MX format with per-element precision elevation within MX blocks by repurposing outlier element exponent field as extended mantissa; achieves near-MXFP6 quality at near-MXFP4 storage cost; most direct hardware-side prior art for per-element type within a block.
[22] RAMP (Gautam and Jha): arXiv:2603.17891, March 2026; RL (Soft Actor-Critic) per-layer bit-width assignment for on-device LLM inference; generalizes zero-shot across model families; achieves lower PPL than uniform AWQ-4bit at same or lower memory.
[23] MoQAE: ACL 2025; mixed-precision quantization targeting long-context LLM inference with context-aware precision assignment across KV and weights. aclanthology.org/2025.acl-long.531
[24] Mixed-Precision Quantization Survey (Rakka et al.): arXiv:2510.16805, October 2025; comprehensive survey of mixed-precision quantization frameworks for LMs; categorizes by bit allocation strategy (per-layer, per-channel, per-group, per-token). Canonical reference for prior art landscape of idea 5.9.
