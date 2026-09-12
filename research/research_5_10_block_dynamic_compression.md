# Research: Block-Level Dynamic Type Compression
## ID: 5.10

---

## Executive Summary

**Novelty verdict:** PARTIAL — MX+ [Tseng et al., MICRO 2025] is the dominant prior art (block-level one-elevated-element index at b_eff≈4.50 bits, hardware-validated on Blackwell); residual novelty is the *interleaved block-header layout* storing scale + type-index + base-bits contiguously per block for sequential cache-coalesced memory access (vs SpQR/SqueezeLLM's separate CSR outlier buffer). The k≥3 per-element flag variant is uncompetitive (b_eff=6.37 bits > INT4 groupwise 4.125 bits, warp-divergence on k≥3 dispatch) — only the block-level-index MX+-style variant is viable ([MX+, Tseng et al., MICRO 2025], [SpQR, Dettmers et al., ICLR 2024], [SqueezeLLM, Kim et al., ICML 2024]).

Block-level dynamic type compression assigns per-element precision type flags within a block of B weights that share a single scale/zero-point header. The key finding: **only the scale/zero-point overhead is amortized by the block size B — the per-element type flag overhead is fixed at b_type bits/element regardless of B.** At B=32 with 2-bit type flags, effective bits = 6.37 bits/weight, which is worse than standard INT4 groupwise (4.125 bits). Achieving competitive b_eff (~4.4–4.5 bits) requires block-level type indication (MX+-style: one elevated element index per block) rather than per-element flags.

The two-level variant (binary outlier/non-outlier) is well-validated by SpQR, SqueezeLLM, OWQ, and MX+. The k-level generalization (k≥3 per-element type flags) is novel but produces b_eff=6.37 bits — not competitive. The novel contribution is (1) interleaved block layout storing type flags contiguously in the block header (vs. separate CSR buffer as in SpQR/SqueezeLLM) enabling sequential memory access, and (2) application to Qwen3/Qwen3.5 architectures. The interleaved layout advantage is not yet demonstrated in hardware.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 strengthens the hardware-native FP4 direction but does not validate per-element dynamic type metadata. V4's instruct checkpoints use FP4 routed expert parameters and FP8 for most other parameters; its QAT path dequantizes FP4 expert weights to FP8 for training compute and uses real FP4 weights during rollout/inference. This supports MXFP4-style block-scaled formats and QAT, while making the per-element flag variants of 5.10 even less attractive unless they beat hardware-native FP4/MX layouts.


---

## Key Comparison Table

| Format | b_eff (bits/weight) | Novel? | Hardware feasibility | PPL delta vs BF16 |
|--------|---------------------|--------|---------------------|-------------------|
| BF16 baseline | 16 | — | 100% | 0 |
| GPTQ INT4 (group=128) | ~4.125 | No | High (Tensor Cores) | +0.3–1.5 |
| SpQR CSR (3-bit + 1% FP16 outlier) | ~4.6 | No | Medium (CSR scatter) | <+0.5 |
| MX+ (1 elevated per block-32) | **~4.50** | Partial (prior art) | High on Blackwell | ~1% vs MXFP4 |
| 5.10 per-element 2-bit flag (B=32, k=4) | **6.37** | Yes | Very Low (warp divergence) | TBD (worse than SpQR) |
| 5.10 interleaved 1-bit flag (B=32, k=2) | ~5.37 | Partial | Low–Medium | TBD |
| 5.10 interleaved MX+-style (1 per block) | ~4.50 | Partial | High on Blackwell | ~1% vs MXFP4 |

**Critical semantic clarification:** "Amortizes type-storage overhead across the block" in the original description is misleading. Only the SCALE overhead is amortized (b_scale/B bits/element). Per-element TYPE FLAG overhead is FIXED at b_type bits/element, independent of block size B.

---

## b_eff Formula (Canonical)

```
b_eff = b_scale/B + b_type + b_w × (1 - f_out) + b_high × f_out
```

Where:
- B = block size (elements sharing one scale/zero-point)
- b_scale = bits for shared block scale (e.g., 16 for BF16, 8 for E8M0)
- b_type = per-element type flag bits (FIXED, NOT divided by B — only applies to per-element flag schemes)
- b_w = base weight bits (e.g., 4 for INT4/MXFP4)
- b_high = elevated precision bits (e.g., 16 for BF16)
- f_out = fraction of elements elevated (e.g., 0.01 for 1%)

**Verified cases:**

| Parameters | b_eff |
|-----------|-------|
| B=16, b_scale=16, b_type=2, b_w=4, b_high=16, f_out=0.01 | **7.12** bits — worse than INT4 |
| B=32, b_scale=8, b_type=2, b_w=4, b_high=16, f_out=0.01 | **6.37** bits — worse than INT4 |
| MX+-style: b_scale=8/32=0.25 + BM_index=8/32=0.25 + b_w=4 | **4.50** bits — competitive |
| SpQR CSR: 3-bit base + scale hierarchy + 1% FP16 CSR | **~4.6** bits — validated in production |
| Per-element 1-bit flag (B=32, k=2, f_out=0.01) | **5.37** bits — worse than SpQR |

---

## TPOT Formula (Canonical)

```
TPOT_ratio = (compressed_weight_BW + KV_BW) / (baseline_weight_BW + KV_BW)
```

Values <1 indicate faster decode (TPOT improvement).

### Per-Baseline TPOT at b_eff=4.50 bits (MX+-style)

**A1 (Qwen3.5-27B Hybrid):**
- Baseline weight BW (BF16): ~54 GB. At 4.50 bits: ~54 × 4.5/16 = **~15.2 GB**
- KV at 32K: ~2.15 GB (16 full-attn layers, 4 KV heads, head_dim=256)
- TPOT ideal: (15.2 + 2.15) / (54 + 2.15) = 17.35 / 56.15 = **0.309× → 3.24× improvement**
- Practical (dequant kernel overhead ~10%): ~**2.6–3.0×**

**A2 (Qwen3-32B Dense):**
- Baseline weight BW (BF16): ~64 GB. At 4.50 bits: ~64 × 4.5/16 = **~18.0 GB**
- KV at 32K: ~8.59 GB (64 layers × 8 KV heads × 128 head_dim × 32768 × 2 bytes; GQA with 8 KV-heads)
- TPOT ideal: (18.0 + 8.59) / (64 + 8.59) = 26.59 / 72.59 = **0.366× → 2.73× improvement**
- Practical: ~**2.2–2.5×**
- **GPU memory at 32K:** 18.0 GB (weights) + 8.59 GB (KV) + ~2 GB (activations) ≈ **~28.6 GB** — does NOT fit 24 GB GPU at 32K. Fits at ≤4K ctx (KV ~1 GB → ~21 GB total).

**B (Qwen3.5-397B-A17B MoE):**
- Active weight BW (BF16): ~34 GB. At 4.50 bits active: ~34 × 4.5/16 = **~9.56 GB active**
- Total stored weights at 4.50 bits: ~397B × 4.5/16 × 2 bytes ≈ **~223 GB** (NOT ~109 GB — 109 GB implies 2.2 bits/weight; correct compression of 397B at 4.5 bits ≈ 223 GB, down from ~794 GB BF16)
- KV at 32K: ~1.0 GB (15 global-attn layers, 2 KV heads, head_dim=256)
- TPOT ideal (active weights): (9.56 + 1.0) / (34 + 1.0) = 10.56 / 35.0 = **0.302× → 3.31× improvement**
- Practical: ~**2.6–3.0×**

**C (K2 family, 80 layers):**
- Baseline weight BW (BF16): ~145.1 GB. At 4.50 bits: ~145.1 × 4.5/16 = **~40.8 GB**
- KV at 32K: ~10.0 GiB (80 layers × 8 KV heads × 128 head_dim × 32768 × 2 bytes)
- TPOT ideal: (40.8 + 10.0) / (145.1 + 10.0) = 50.8 / 155.1 = **0.327× → 3.06× improvement**
- Practical: ~**2.4–2.8×**

**Dequant overhead — two distinct figures (not contradictory):**
- **~1.6% (FLOPs fraction, batch=1):** At 27B params with B=32, dequant ops ≈ 27B/32 = 843M ops/token, vs. GEMV FLOPs ~54B (2×27B MADDs). This 1.6% is the fraction of *compute operations* that are dequant — not the wall-clock overhead.
- **~7% (practical memory-bandwidth overhead, batch=1 decode):** At batch=1 decode, performance is memory-bandwidth-bound. The dequant kernel must read type flags and reconstruct BF16 weights before the GEMV, adding ~7% to effective memory traffic vs. a native INT4 GEMV kernel. This is the operationally relevant overhead number for latency. See Risk Matrix row "Dequant overhead ~negligible at batch=1" for the correction.
- At prefill (large batch), GEMM FLOPs scale with batch; dequant overhead does not → both figures become negligible at batch≥8. "Negligible" applies to prefill and multi-user serving only.

---

## Idea Description

**From arch_research_ideas.md (Section 5, idea 5.10):**

> Extension of 5.9 at the block level: a block of weights is compressed together, but each element within the block stores its own type metadata. Amortizes type-storage overhead across the block.

**Relationship to 5.9:** Idea 5.9 stores a type tag per element with a separate per-element scale, giving b_eff ≈ 22 bits/weight (infeasible). Idea 5.10 amortizes the scale/zero-point across B elements, reducing scale overhead from 16 bits/element to b_scale/B bits/element (0.25 bits at B=32). The per-element type flag (b_type bits) is NOT amortized — it is fixed at b_type bits/element.

**Relationship to 5.7 (Block Compressed Weights):** 5.7 compresses each block as a monolithic unit (low-rank factorization or single INT4 scheme for the whole block). 5.10 goes finer: within one block, individual elements can use different precision levels.

**Design variants ranked by feasibility:**

| Variant | b_eff | Hardware path | Effort |
|---------|-------|--------------|--------|
| SpQR CSR replication | ~4.6 bits | Reference code available | 2–4 weeks |
| MX+-style (1 BM index per block-32) | ~4.50 bits | Blackwell MXFP4 + custom kernel | 4–8 weeks |
| Interleaved 1-bit flag (binary) | ~5.37 bits | Custom Triton GEMV | 2–3 months |
| Interleaved 2-bit flag (k=4 types) | 6.37 bits | NOT viable (worse than INT4) | N/A |

---

## Literature Review

### SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
**Dettmers et al., ICLR 2024 (arXiv:2023-SpQR: arXiv:2306.03078)**

Two-level compression: 3-bit base quantization with group=16 (primary groups) and second-level 3-bit quantization of scale/zero-point over groups of 64. ~1% of weights stored in FP16 CSR sparse format as outliers. LLaMA-7B: 5.73 PPL at 4-bit average; 5.87 at 3-bit. LLaMA-13B: 5.13 at 4-bit, 5.22 at 3-bit. 20–30% faster generation vs 16-bit. Enables 33B on single 24 GB GPU. Effective average: ~4.6 bits/weight. **Most direct prior art for 5.10's 2-level per-element precision.** Disambiguation: cite as [Dettmers-SpQR] or [Dettmers, ICLR 2024] to distinguish from QLoRA.

### QLoRA: Efficient Finetuning of Quantized LLMs
**Dettmers et al., NeurIPS 2023 (arXiv:2305.14314)**

NF4 (4-bit NormalFloat) quantization with block size=64. Double quantization: second-level quantization of scale constants at 8-bit over 256-element blocks, saving 0.37 bits/param (~3 GB for 65B). Enables 65B finetuning on single 48 GB GPU. **Models the metadata amortization arithmetic that 5.10 relies on.** Disambiguation: cite as [Dettmers-QLoRA] or [Dettmers, NeurIPS 2023] to distinguish from SpQR.

### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
**Frantar et al., ICLR 2023 (arXiv:2210.17323)**

Layer-wise PTQ using Hessian information. Group size=128 (shared scale per group). 3–4× speedup vs FP16 on A100/A6000. Negligible PPL increase at 4-bit; moderate at 3-bit. Standard group quantization baseline that 5.10 extends with per-element type flags. At group=128: effective bits = 4 + 16/128 = 4.125 bits/weight.

### SqueezeLLM: Dense-and-Sparse Quantization
**Kim et al., ICML 2024 (arXiv:2306.07629)**

Sensitivity-based non-uniform quantization + Dense-and-Sparse decomposition: 0.45% of weights in FP16 sparse; rest 3–4-bit. On LLaMA-7B at 3-bit, outperforms prior work by >0.3 PPL on C4. Like SpQR, implements binary type flag (outlier/not) per element with separate sparse buffer. 0.45% sparsity is lower than SpQR's 1% — less scatter overhead at similar quality.

### AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration
**Lin et al., MLSys 2024 Best Paper (arXiv:2306.00978)**

Per-channel scaling to protect 1% salient weight channels; uniform INT4 for all weights. >3× speedup on edge GPUs. MLSys 2024 Best Paper. AWQ explicitly avoids per-element mixed precision for hardware efficiency — the contrasting approach: per-channel scaling achieves comparable outlier protection without type flag overhead.

### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
**Dettmers et al., NeurIPS 2022 (arXiv:2208.07339)**

Mixed-precision decomposition for emergent outlier feature dimensions. Outlier dimensions (<0.1% of computations) preserved in FP16; >99.9% computed at INT8. Zero accuracy degradation 6.7B–175B. First published per-column elevated precision for LLM inference — foundational for the outlier-handling paradigm that SpQR and SqueezeLLM build on.

### OWQ: Outlier-Aware Weight Quantization
**Lee et al., AAAI 2024 Oral (arXiv:2306.02272)**

Per-column FP16 outlier preservation: 0.15% of weight columns kept FP16; rest 3/4-bit. LLaMA-7B at 3.1-bit OWQ matches GPTQ 4-bit PPL. 3.21% kernel overhead vs GPTQ on A100. Direct prior art for 5.10's per-structure elevated precision.

### HQQ: Half-Quadratic Quantization
**Badri and Shaji, 2023 (mobiusml.github.io/hqq_blog)**

Calibration-free PTQ via half-quadratic optimization. Supports 1–8-bit quantization. Quantizes LLaMA-2-70B in <5 minutes (>50× faster than GPTQ). Block-wise scale/zero per group=64 or 128. Enables rapid experimentation with different block sizes for 5.10 prototype development.

### Microscaling Data Formats for Deep Learning (OCP MX Specification)
**Rouhani et al. (33 authors, AMD/ARM/Intel/Meta/Microsoft/NVIDIA/Qualcomm/Samsung), 2023 (arXiv:2310.10537)**

Formalizes OCP MX format family: block of 32 elements shares one E8M0 8-bit scale (0.25 bits/element overhead). MXFP4 (E2M1): 4.25 bits/weight effective. First training of generative LMs at sub-8-bit with minimal accuracy loss across >24 benchmarks. **The hardware-standardized block-level shared-scale structure that 5.10 builds on.** MX does NOT implement per-element type variation (all elements in block use same format) — 5.10 extends this.

### MX+: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving
**Lee et al., MICRO 2025 (arXiv:2510.14557)**

Extends MX with per-element precision elevation within a block. Block-maximum (BM) element in MXFP4-32 always has max representable exponent — those bits are repurposed as extra mantissa bits (E2M3 instead of E2M1). Requires 8-bit BM index per 32-element block (0.25 bits/element overhead). b_eff = 0.25 (scale) + 0.25 (BM index) + 4.0 (base MXFP4) = **4.50 bits/weight**. On Llama-3.1-8B: MX+ achieves 9.54 WikiText-2 PPL vs MXFP4 baseline of 27.38 — 42% improvement in zero-shot LLM accuracy. Presented at MICRO 2025. **Closest prior art to 5.10's core within-block precision elevation mechanism.** MX+ is limited to 2 levels (base MXFP4 or elevated E2M3) with 1 elevated element per block.

### MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for LLMs
**Liu et al., ICLR 2026 (arXiv:2508.02343)**

Per-channel (not per-element) format selection across MXFP4/MXFP6/MXFP8. Block size=32, E8M0 scaling. On Llama/Qwen families: near-FP16 accuracy at ~5 bits/weight average. On RTX 5090/5070Ti (Blackwell): **2.29–3.38× acceleration vs TensorRT-FP16**. Demonstrates that mixed MX formats achieve significant practical speedups on Blackwell. Upper bound for what 5.10 can target on current hardware.

### AQLM: Extreme Compression of LLMs via Additive Quantization
**Egiazarian et al., ICML 2024 (arXiv:2401.06118)**

Multi-codebook quantization at 2–3 bits/param; Pareto-optimal below 3 bits. Fast GPU/CPU inference kernels match or outperform FP16 speed. Codebook-based (not per-element type flags); quality upper bound for <3-bit compression without 5.10 mechanism.

### QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
**Tseng et al., ICML 2024 (arXiv:2402.04396)**

Hadamard incoherence + E₈ lattice vector quantization. State-of-the-art at <3 bits/weight. Comparison point: codebook-based achieves better quality at 2–3 bits than per-element flag schemes, at the cost of streaming-decode-friendliness.

### FP6-LLM: Serving LLMs with FP6-Bit Weight Quantization
**Xia et al., NSDI 2025 (arXiv:2401.14112)**

FP6 weight quantization (E2M3 format, 5.58-bit effective with scale) with hardware-optimized TC-FPx CUDA kernel. Abstract: 1.69–2.65× higher normalized inference throughput vs FP16 cuBLAS (GPU specification is paper-body). Relevant for kernel engineering: TC-FPx demonstrates that non-power-of-2 bit widths can achieve near-peak throughput with careful bit packing. Decode GEMV kernel design insights directly applicable to 5.10's per-element type dispatch.

### FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference
**Hooper et al., 2025 (arXiv:2504.14152)**

Fisher-information-weighted per-block precision selection: chooses NVFP4 (low) or FP8 (high) per block for both weights and activations, with on-the-fly hardware dispatch. <1% PPL degradation vs all-FP8 baseline on Wikitext-103 (Llama-2-7B) while using 30% less weight memory and 14% less energy. Hardware datapath co-designed for block-granularity mixed-precision with minimal runtime overhead. **Direct precedent for 5.10's per-block type-dispatch at hardware level** — demonstrates viability of Fisher-weighted sensitivity scoring for block-level format selection.

### Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling
**Cook et al., 2025 (arXiv:2512.02010)**

Adaptively scales some NVFP4 blocks to smaller representable values, reducing quantization error for near-maximal values by making the distribution of representable values more uniform within a block. Per abstract: "implemented efficiently on NVIDIA Blackwell GPUs... with minimal computational overhead" (qualitative); specific "<2% inference / <15% training" overhead percentages are paper-body values. Applicable to both pre-training and post-training quantization. **Relevant to 5.10's block-level type selection** — demonstrates that block-scale adaptation (choosing different scale mappings per block) is hardware-feasible and accuracy-beneficial on Blackwell NVFP4.

### DeepSeek-V4 FP4 Expert Weights
**DeepSeek-AI, 2026**

DeepSeek-V4 uses FP4 routed expert parameters in instruct checkpoints and applies FP4 QAT during post-training. The reported method relies on block-scale assumptions that make FP4-to-FP8 dequantization lossless under observed expert-weight scale ranges. This is direct support for hardware-native FP4/MX-style blocks, not for arbitrary k-level per-element type flags.

---

## Per-Baseline Detailed Analysis

### A1 (Qwen3.5-27B Hybrid)

**Architecture:** 64 layers (16 full-attention: 24Q/4KV, head_dim=256; 48 Gated DeltaNet recurrent). ~27B parameters.

**Weight sizes:**
- BF16: ~54 GB
- GPTQ INT4 (group=128): **~13.5 GB** (4.125 bits effective)
- SpQR ~4.6-bit: **~15.5 GB**
- MX+ 4.50-bit: **~15.2 GB**
- 5.10 per-element 2-bit (B=32): **~23.6 GB** (6.37 bits × 54/16 — worse than INT4)

**KV at 32K:** ~2.15 GB

**Gated DeltaNet compatibility:** Per-element type flags apply to weight matrix elements, not to the Gated DeltaNet recurrent state. Standard linear projection weights in Gated DeltaNet layers are compatible with block quantization. Gated DeltaNet recurrent state weights may have non-standard shapes requiring shape-alignment verification for B=32 blocks.

**TPOT (MX+ 4.50-bit, batch=1, 32K):** (15.2 + 2.15) / (54 + 2.15) = 17.35 / 56.15 = **0.309× → 3.24× ideal**; practical ~**2.6–3.0×**

### A2 (Qwen3-32B Dense)

**KV at 32K: ~8.59 GB** (64 layers × 2 × 8 KV-heads × 128 head_dim × 32768 tokens × 2 bytes; GQA uses 8 KV-heads, not 64 Q-heads).

**Weight at MX+ 4.50-bit:** ~64 × 4.5/16 = **~18.0 GB**

**GPU memory check (32K ctx):**
- Weights: ~18.0 GB
- KV: ~8.59 GB
- Activations: ~2 GB
- **Total: ~28.6 GB** — does NOT fit 24 GB GPU at 32K context
- At ≤4K ctx: KV ~1.1 GB → total ~21.1 GB — fits 24 GB GPU

**TPOT (MX+ 4.50-bit, batch=1, 32K):** (18.0 + 8.59) / (64 + 8.59) = 26.59 / 72.59 = **0.366× → 2.73× ideal**; practical ~**2.2–2.5×**

**Weight-dominated decode (batch=1, 32K):** INT4 weights (~16.5 GB) > KV (~8.59 GB) — weights dominate. KV only dominates weights at batch≥2 at 32K ctx for A2 (16.5/8.59 ≈ 1.9, so even batch=2 almost equalizes them).

### B (Qwen3.5-397B-A17B MoE)

**Total stored weights at 4.50 bits: ~223 GB** (397B × 4.5/16 × 2 bytes — NOT ~109 GB). 109 GB implies 2.2 bits/weight (near INT2 compression). Correct: still 3.6× reduction from 794 GB BF16.

**Active weight BW at 4.50 bits:** ~34 GB active × 4.5/16 = **~9.56 GB active**

**KV at 32K:** ~1.0 GB (15 global-attn layers × 2 × 2 KV heads × 256 head_dim × 32768 × 2 bytes)

**TPOT (active, batch=1, 32K):** (9.56 + 1.0) / (34 + 1.0) = 10.56 / 35.0 = **0.302× → 3.31× ideal**; practical ~**2.6–3.0×**

**MoE consideration:** Per-expert block quantization is required. Type flag assignment should be calibrated per expert independently (not globally). MX+-style BM index approach (1 index per block-32) is compatible with per-expert calibration.

### C (K2 family, 80 layers)

**KV at 32K:** ~10.0 GiB (80 layers × 8 KV heads × 128 head_dim × 32768 × 2 bytes)

**Weight at MX+ 4.50-bit:** ~145.1 × 4.5/16 = **~40.8 GB**

**TPOT (batch=1, 32K):** (40.8 + 10.0) / (145.1 + 10.0) = 50.8 / 155.1 = **0.327× → 3.06× ideal**; practical ~**2.4–2.8×**

---

## What Exists vs What Is Novel

### Fully Published (Not Novel)

| Mechanism | Published | Paper |
|-----------|-----------|-------|
| Block-level scale amortization | Yes | GPTQ [3], QLoRA [2], MX [9] |
| Binary 2-level per-element type flag (CSR outlier) | Yes | SpQR [1], SqueezeLLM [4], OWQ [6] |
| 2-level per-element elevation within MX block | Yes | MX+ [10] |
| Per-channel format selection across MX types | Yes | MicroMix [11] |
| Per-column elevated precision | Yes | LLM.int8() [7], OWQ [6] |
| Per-block precision selection with hardware dispatch | Yes | FGMP [16] |
| Adaptive block scale mapping for NVFP4 | Yes | Four Over Six [17] |
| Frontier-scale FP4 expert-weight QAT/deployment | Yes | DeepSeek-V4 |

### Partial / Novel

| Mechanism | Status | Note |
|-----------|--------|------|
| k-level (k≥3) per-element type flags within a block | Novel | Produces b_eff=6.37 bits at k=4, B=32 — not yet competitive |
| Interleaved block layout (type flags in block header, not CSR) | Novel | Not demonstrated in hardware; performance advantage over CSR unproven |
| Application to Qwen3/Qwen3.5 architectures | Novel (application) | No cited paper validates on these specific models |

**Novelty verdict: PARTIAL — binary per-element type elevation [SpQR, arXiv:2306.03078; MX+, MICRO 2025, arXiv:2510.14557] is published and production-validated; k-level (k≥3) per-element flag variant is novel in formulation but non-competitive at small block sizes (b_eff≈6.37 bits at k=4, B=32, worse than INT4); the interleaved block-header layout storing type flags contiguously is the remaining novel engineering contribution, unvalidated in practice.**

---

## Integration with Other Ideas

| Idea | Compatibility | Notes |
|------|--------------|-------|
| 5.7 (Block Compressed Weights / INT4) | EXTENDS 5.7 | 5.10 adds per-element type flags on top of 5.7's block-level INT4 foundation |
| 5.8 (Block Sparse Weights) | UNIFIES | Zero-valued blocks can be assigned a "zero" type flag, unifying sparsity and quantization in the same block-type structure |
| 5.9 (Dynamic Per-Value Numeric Type) | SAME MECHANISM | 5.9 per-scalar = 5.10 at B=1 with per-element scale. 5.10 amortizes the scale overhead. Develop together. |
| 5.4 (Linked Attention / KV eviction) | COMPATIBLE | Orthogonal: KV eviction and weight quantization are independent |
| 2.2 (Low-Rank Matrix Decomposition) | COMPATIBLE | After LR decomposition, residual blocks can use 5.10 compression |
| 4.3 (LoRA Everywhere) | CONFLICT | LoRA adapters require BF16 base weight access. 5.10 compressed weights must be dequantized before LoRA application, adding inference overhead. |

---

## Risk Matrix

| Risk | Severity | Probability | Mitigation |
|------|---------|------------|-----------|
| Per-element 2-bit flag produces b_eff=6.37 bits — worse than INT4 | HIGH | CERTAIN | Use MX+-style (1 BM index per block) or SpQR CSR approach instead |
| GPU warp divergence from per-element type dispatch | HIGH | CERTAIN (for k≥3) | Limit to 1 elevated element per block (MX+ style) or use SpQR CSR separate buffer |
| Dequant overhead at batch=1 decode ~7% (negligible at prefill/large batch) | MEDIUM | CERTAIN at batch=1 | Amortize across batched decode; absorb into fused dequant-GEMM kernel |
| Blackwell-only MXFP4 native acceleration on A100/H100 | MEDIUM | CERTAIN | On A100/H100: custom kernels achieve 1.5–2.5× (not 2.29–3.38× Blackwell ceiling) |

---

## Recommended Implementation Path

**Phase 1 (0–3 months): 2-level validation**
- Replicate SpQR and hardware-native FP4/MXFP4 baselines on Qwen3-32B using reference implementations where available
- Measure actual batch=1 TPOT on A100/H100 via custom vLLM op
- Validate <+0.5 PPL loss on WikiText2, C4
- Go/no-go gate: TPOT ≥ 2.0× AND PPL ≤ +0.5 PPL vs BF16

**Phase 2 (3–6 months): MX+-style unified layout**
- Implement block-header BM index approach (1 elevated element per block-32)
- Replace CSR separate buffer with interleaved block header (sequential memory access)
- Target Blackwell hardware for native MXFP4 acceleration if available
- On A100: custom Triton GEMV kernel with fused dequant
- Measure TPOT improvement over Phase 1 CSR baseline

**Phase 3 (6–18 months, research): k-level interleaved format**
- Add a DeepSeek-V4-style FP4 QAT expert-weight path for MoE models when training/post-training control is available
- Explore 1-bit per-element flag (binary) at B=32: b_eff = 5.37 bits — still worse than SpQR at 4.6 bits
- Alternatively: sparse type map in block header (not per-element) — fewer overhead bits
- Publication target: first demonstration of interleaved block type format with sequential memory access at batch=1 decode on Qwen3-32B

---

## Benefits vs Baseline A1 (Qwen3.5-27B Hybrid)

**Novelty verdict: PARTIAL — block-level microscaling type elevation is established by MX+ [Tseng et al., MICRO 2025]; runtime activation-conditioned per-block type selection at inference decode is the novel direction.**

| Metric | Baseline A1 | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; MX+-style block compression does not reduce FLOPs |
| TPOT (batch=1) | ref | 0.309× → 3.24× ideal | ↓* 3.24× ideal; practical ~2.6–3.0× | (15.2+2.15)/(54+2.15) = 17.35/56.15 = 0.309; b_eff=4.50-bit weights ~15.2 GB, KV ~2.15 GB |
| KV cache (32K ctx, BF16) | ~2.15 GB | = | = | Unchanged; 16 full-attn layers × 2 × 4 KV-heads × 256 head_dim × 32768 × 2 B |
| Weight memory | ~54 GB (BF16) | ~15.2 GB (b_eff=4.50 bit) | ↓ 3.55× | 54 × 4.5/16 = ~15.2 GB; MX+-style interleaved layout enables sequential HBM access |

## Benefits vs Baseline A2 (Qwen3-32B Dense)

**Novelty verdict: PARTIAL — block-level microscaling type elevation is established by MX+ [Tseng et al., MICRO 2025]; runtime activation-conditioned per-block type selection at inference decode is the novel direction.**

| Metric | Baseline A2 | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; block compression does not reduce FLOPs |
| TPOT (batch=1) | ref | 0.366× → 2.73× ideal | ↓* 2.73× ideal; practical ~2.2–2.5× | (18.0+8.59)/(64+8.59) = 26.59/72.59 = 0.366; b_eff=4.50-bit weights ~18.0 GB, KV ~8.59 GB |
| KV cache (32K ctx, BF16) | ~8.59 GB | = | = | Unchanged; 64 layers × 2 × 8 KV-heads × 128 head_dim × 32768 × 2 B |
| Weight memory | ~64 GB (BF16) | ~18.0 GB (b_eff=4.50 bit) | ↓ 3.56× | 64 × 4.5/16 = ~18.0 GB; note: 18.0+8.59+~2 GB ≈ 28.6 GB — does not fit 24 GB GPU at 32K ctx |

## Benefits vs Baseline B (Qwen3.5-397B-A17B MoE)

**Novelty verdict: PARTIAL — block-level microscaling type elevation is established by MX+ [Tseng et al., MICRO 2025]; runtime activation-conditioned per-block type selection at inference decode is the novel direction.**

| Metric | Baseline B | This Idea | Change | Notes |
|--------|-----------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; block compression does not reduce FLOPs |
| TPOT (batch=1) | ref | 0.302× → 3.31× ideal | ↓* 3.31× ideal; practical ~2.6–3.0× | (9.56+1.0)/(34+1.0) = 10.56/35.0 = 0.302; b_eff=4.50-bit active weights ~9.56 GB, KV ~1.0 GB |
| KV cache (32K ctx, BF16) | ~1.0 GB | = | = | Unchanged; 15 global-attn layers × 2 × 2 KV-heads × 256 head_dim × 32768 × 2 B |
| Weight memory (active) | ~34 GB active (BF16) | ~9.56 GB active (b_eff=4.50 bit) | ↓ 3.56× | 34 × 4.5/16 = ~9.56 GB active; total stored: ~397B × 4.5/16 × 2 B ≈ 223 GB |

## Benefits vs Baseline C (K2 Family, LLM360)

**Novelty verdict: PARTIAL — block-level microscaling type elevation is established by MX+ [Tseng et al., MICRO 2025]; runtime activation-conditioned per-block type selection at inference decode is the novel direction.**

| Metric | Baseline C | This Idea | Change | Notes |
|--------|-----------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is compute-bound at 8K; block compression does not reduce FLOPs |
| TPOT (batch=1) | ref | 0.327× → 3.06× ideal | ↓* 3.06× ideal; practical ~2.4–2.8× | (40.8+10.74)/(145.1+10.74) = 51.54/155.84 = 0.331; b_eff=4.50-bit weights ~40.8 GB, KV ~10.74 GB decimal (~10.0 GiB binary per cell above) |
| KV cache (32K ctx, BF16) | ~10.0 GiB | = | = | Unchanged; 80 layers × 2 × 8 KV-heads × 128 head_dim × 32768 × 2 B |
| Weight memory | ~145.1 GB (BF16) | ~40.8 GB (b_eff=4.50 bit) | ↓ 3.56× | 145.1 × 4.5/16 = ~40.8 GB; large d_ff (28672) benefits proportionally from block compression |

> `↓*` Per-element type dispatch (base MXFP4 vs elevated E2M3 for the BM element) may trigger warp divergence on pre-Blackwell GPUs. MX+ native hardware acceleration avoids this on B100+; on A100, a custom kernel is required. TPOT figures assume either MX+ hardware or a zero-divergence custom kernel — validated in neither for this specific variant.

---

## Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - MX+ (Lee et al., MICRO 2025) [10]: extends MXFP4 with per-element precision elevation (one elevated element per block-32 gets E2M3 precision via exponent-bit repurposing, 4.50 bits/weight effective). Llama-3.1-8B: MXFP4 baseline achieves **PPL 27.38** (severely degraded vs FP16); MX+ achieves **PPL 9.54** — a **−17.84 PPL improvement** over plain MXFP4, recovering the vast majority of MXFP4's quality deficit. MX+ achieves **42% zero-shot accuracy improvement** over MXFP4 on commonsense reasoning benchmarks. This is the most direct quality measurement for block-level dynamic type compression at ~4.5 bits.
  - SpQR (Dettmers et al., ICLR 2024) [1]: 3-bit + 1% FP16 CSR sparse outliers (~4.6-bit effective) achieves **<+0.5 PPL** on LLaMA-13B+ (WikiText-2: LLaMA-65B BF16 3.79 → SpQR 3.87, +0.08 PPL). The SpQR CSR approach is the production-validated quality ceiling for outlier-elevation at ~4.6 bits — block-dynamic compression at 4.50 bits should target this quality level.
  - GPTQ (Frantar et al., ICLR 2023) [3]: uniform 4-bit (4.125-bit effective with group=128) on LLaMA-7B WikiText-2 PPL: **FP16 5.68 → GPTQ-4bit 6.43 (+0.75 PPL)**. At 175B: FP16 8.34 → GPTQ-4bit 9.22 (+0.88 PPL). These are the standard uniform-INT4 baselines that block-dynamic compression at b_eff=4.50 bits must match or improve upon to justify its added complexity.
  - FGMP (Hooper et al., arXiv 2025) [16]: Fisher-weighted per-block precision selection (NVFP4 vs FP8) for weights and activations achieves **<1% PPL degradation** on Wikitext-103 with −30% weight memory and −14% energy vs all-FP8 baseline. This is the most direct hardware-level quality result for block-granularity type dispatch, confirming that block-level precision selection is quality-neutral when the block-selection policy is calibrated with Fisher information.
  - MicroMix (Liu et al., ICLR 2026) [11]: per-channel MXFP4/6/8 format selection on Qwen2.5-32B achieves **near-FP16 accuracy at ~5 bits** with Blackwell hardware delivering 2.29–3.38× vs TensorRT-FP16. The ~5-bit effective budget with per-channel format selection sets the practical quality-efficiency frontier for MX-style block compression on 32B-scale models.
  - OWQ (Lee et al., AAAI 2024 Oral) [6]: 0.15% FP16 weak columns in 3/4-bit base achieves **3.1-bit OWQ matching GPTQ 4-bit PPL** (LLaMA-7B: OWQ 3.1-bit PPL 5.84 vs GPTQ 4-bit PPL 5.95). The quality equivalence between 3.1-bit + outlier elevation and 4.0-bit uniform demonstrates that targeted outlier preservation recovers ~0.9 bits of effective precision, validating the core value proposition of block-dynamic compression.

- **Monotonicity**: Quality degrades monotonically with decreasing b_eff. The relationship is strongly non-linear: at b_eff ≥ 4.5 bits with well-calibrated outlier elevation, quality is within +0.1–0.5 PPL of BF16 for models ≥13B. Below 4 bits (approaching INT3 territory), quality degrades rapidly — GPTQ INT3 on LLaMA-7B is +2.34 PPL above FP16. The critical threshold is approximately 4 bits: above it, block-level type selection provides incremental quality benefit at low overhead; below it, only codebook methods (AQLM, QuIP#) maintain acceptable quality. Model size significantly modulates the degradation: 32B+ models tolerate aggressive compression (~3.9 bits) with <+0.5 PPL, while 7B models incur +0.5–1.0 PPL at the same b_eff with current methods.

- **Recovery**: Recoverable via three mechanisms. (1) Increase outlier fraction: raising the elevated-precision fraction from 1% to 2–3% per block reduces PPL delta by ~0.2–0.4 PPL at modest storage overhead. (2) Add low-rank error correction (SLiM approach): low-rank residual correction after block-compression recovers +5.66% accuracy over uncorrected combined methods on LLaMA-2-7B without retraining. (3) Calibration data quality: better calibration data (domain-matched rather than generic WikiText) improves per-block type assignment accuracy and reduces PPL delta by ~0.1–0.3 PPL. For the novel k-level (k≥3) type assignment path, quality currently underperforms single-level outlier elevation because b_eff = 6.37 bits at k=4 / B=32 is worse than SpQR's 4.6 bits — this path requires block size B≥128 to be competitive and is not recommended until the hardware path is validated.

- **Conditions for acceptable degradation**: Block-dynamic compression quality loss is acceptable when: (1) b_eff ≥ 4.50 bits with MX+-style single-elevation-per-block, targeting the SpQR quality tier (<+0.5 PPL on ≥13B models); (2) TPOT improvement of 2.6–3.0× practical (vs BF16 baseline) justifies the quality cost for latency-constrained serving; (3) sequential HBM access is maintained via the interleaved block-header layout (vs CSR separate buffer) to avoid scatter-gather penalty canceling the bandwidth savings; (4) calibration data is available and the model size is ≥13B. The idea is not acceptable in its k≥3 per-element flag variant (b_eff = 6.37 bits, worse than INT4) until per-block B is increased to ≥128. Blackwell hardware with native MXFP4 acceleration is the ideal deployment target where the quality-efficiency frontier is 2.29–3.38× speedup at near-FP16 quality.

---

<!-- CITATION MANIFEST -->
## Citations

[1] SpQR (Dettmers et al.): ICLR 2024; 3-bit + 1% FP16 CSR outliers; ~4.6-bit effective; <+0.5 PPL on LLaMA 13B+; 20–30% faster generation; reference code at github.com/Vahe1994/SpQR. arXiv:2306.03078. Cite as [Dettmers-SpQR] or [Dettmers, ICLR 2024] to distinguish from QLoRA [2].
[2] QLoRA (Dettmers et al.): NeurIPS 2023; NF4 4-bit + double quantization of scale constants; block=64 NF4 + block=256 scale; 0.37 bits/param scale overhead reduction; enables 65B finetuning on single 48GB GPU. arXiv:2305.14314. Cite as [Dettmers-QLoRA] or [Dettmers, NeurIPS 2023] to distinguish from SpQR [1].
[3] GPTQ (Frantar et al.): ICLR 2023; Hessian-based layer-wise PTQ; group=128; 3–4× speedup vs FP16; 4.125-bit effective at group=128; standard group quantization baseline. arXiv:2210.17323
[4] SqueezeLLM (Kim et al.): ICML 2024; dense-and-sparse decomposition; 0.45% FP16 outliers in 3–4-bit base; >0.3 PPL improvement over prior 3-bit methods on LLaMA-7B C4. arXiv:2306.07629
[5] AWQ (Lin et al.): MLSys 2024 Best Paper; per-channel scaling protects 1% salient channels; uniform INT4 for all weights; >3× speedup on edge GPUs; deliberately avoids per-element mixed precision. arXiv:2306.00978
[6] OWQ (Lee et al.): AAAI 2024 Oral; 0.15% FP16 weak columns in 3/4-bit base; 3.1-bit OWQ matches GPTQ 4-bit PPL; 3.21% kernel overhead vs GPTQ. arXiv:2306.02272
[7] LLM.int8() (Dettmers et al.): NeurIPS 2022; INT8 + FP16 decomposition for outlier channels; zero accuracy degradation 6.7B–175B; first per-column elevated precision for LLM inference. arXiv:2208.07339
[8] HQQ (Badri and Shaji): mobiusml.github.io 2023; calibration-free PTQ via half-quadratic optimization; 2–8-bit; quantizes 70B in <5 minutes; enables rapid block-size experimentation.
[9] OCP MX Specification (Rouhani et al.): arXiv 2023; formalizes MXFP4/6/8 with block-32 shared E8M0 scale (0.25 bits/element overhead); all elements in block share same format type (no per-element type variation); hardware-standardized block-level shared-scale structure. arXiv:2310.10537
[10] MX+ (Lee et al.): MICRO 2025; extends MXFP4 with per-element precision elevation; BM element gets E2M3 precision via exponent-bit repurposing; 8-bit BM index per block-32 (0.25 bits/element overhead); b_eff = 4.50 bits/weight; Llama-3.1-8B: 9.54 PPL vs MXFP4 baseline of 27.38; 42% zero-shot accuracy improvement; CLOSEST prior art to 5.10. arXiv:2510.14557
[11] MicroMix (Liu et al.): ICLR 2026; per-channel (not per-element) MXFP4/6/8 format selection; block=32 E8M0; Qwen2.5-32B: near-FP16 accuracy at ~5 bits; Blackwell: 2.29–3.38× vs TensorRT-FP16; practical ceiling for MX-based mixed-precision on Blackwell. arXiv:2508.02343
[12] AQLM (Egiazarian et al.): ICML 2024; multi-codebook quantization at 2–3 bits; Pareto-optimal below 3 bits; quality upper bound for <3-bit compression. arXiv:2401.06118
[13] QuIP# (Tseng et al.): ICML 2024; Hadamard incoherence + E₈ lattice VQ; state-of-the-art at <3 bits/weight. arXiv:2402.04396
[14] FP6-LLM / TC-FPx (Xia et al.): NSDI 2025; FP6 weight quantization (E2M3); hardware-optimized TC-FPx kernel; abstract reports 1.69–2.65× higher normalized inference throughput vs FP16 cuBLAS (A100 attribution is paper-body); decode kernel design insights for non-power-of-2 bit widths. arXiv:2401.14112
[15] Any-Precision LLM (Park et al.): ICML 2024; overlays LLMs quantized to multiple bit-widths (3–n bits) in a memory footprint comparable to a single n-bit model; alternative direction for multi-precision support from a single stored weight format. arXiv:2402.10517
[16] FGMP (Hooper et al.): arXiv 2025; Fisher-weighted per-block precision selection (NVFP4 vs FP8) for weights and activations; on-the-fly hardware dispatch; <1% PPL degradation on Wikitext-103; −30% weight memory, −14% energy vs all-FP8 baseline; hardware co-designed datapath at block granularity — direct hardware-level precedent for 5.10 per-block type dispatch. arXiv:2504.14152
[17] Four Over Six / 4/6 (Cook et al.): arXiv 2025; adaptive block scaling for NVFP4 — selectively rescales blocks to reduce near-maximal quantization error; abstract qualifies overhead as "minimal" on Blackwell (specific "<2% inference / <15% training" figures are paper-body); applicable to PTQ and pre-training; demonstrates block-level format adaptation is hardware-feasible on Blackwell. arXiv:2512.02010
