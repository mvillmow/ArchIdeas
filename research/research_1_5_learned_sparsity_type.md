# Research: Learned Sparsity Type (N:M vs MoE)
## ID: 1.5

---

## Executive Summary

**One-line description**: Train a per-layer gate that selects between N:M weight sparsity and MoE routing as alternative sparsity modalities, producing a statically-typed heterogeneous sparse model at inference time.

**Overall feasibility verdict**: INVESTIGATE FURTHER — component techniques work individually; the joint gate training has never been demonstrated at LLM scale; mode collapse is the primary technical risk. A proof-of-concept on a 1–7B model is the appropriate first step.

> **⚠ Conditional results — two unvalidated hypotheses:**
> - **H1:** MoE arithmetic intensity at batch=1 is ~8 FLOP/byte (based on estimated DRAM-bound behavior for random expert dispatch). If actual arithmetic intensity is higher (GPU cache reuse), TPOT gains are smaller.
> - **H2:** DRAM fragmentation penalty exists for batch=1 random-dispatch MoE at serving time. If modern frameworks eliminate this penalty via expert-parallel batching, gains disappear.
> All TPOT tables below are conditioned on both H1 and H2 holding simultaneously. Treat as upper bounds until validated experimentally.

---

## 1. Idea Description

**From arch_research_ideas.md (Section 1, idea 1.5):**

> Beyond choosing dense vs sparse, the model also learns *what kind* of sparsity to apply per layer — unstructured weight pruning vs structured MoE routing.

**Inferred intent:** Different layers in a transformer have different sensitivity profiles and different computational bottlenecks. Early layers may benefit from MoE routing (which routes entire tokens to specialized sub-networks and preserves dense arithmetic intensity per expert), while middle or later layers may be more amenable to N:M weight sparsity (which thins weights globally with no routing dispatch overhead). The idea is to train a gate — either a continuous differentiable selector or a discrete learned assignment — that chooses, per layer, which sparsity modality to apply. In the simplest (and most hardware-deployable) variant, this gate is trained and then frozen: the result is a statically-typed sparse model where some layers are MoE and others are N:M-pruned. This static-at-inference property is critical for TTFT and TPOT: the inference path is fully predictable, enabling kernel selection at compile time rather than dynamic dispatch.

**Key distinction from 1.4:** Idea 1.4 asks *whether* a layer is sparse (dense vs sparse). Idea 1.5 asks, given that a layer will be made sparse, *which type* of sparsity — N:M weight sparsity vs structured MoE routing — and trains the model to discover the optimal assignment.

---

## 2. Literature Review

### [1] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning
- **Authors**: Lee, Hwang, Qiao et al. (arXiv: 2409.06211), 2025, ACL
- **Summary**: Two-stage pruning strategy for MoE LLMs: first apply expert-level structured pruning (remove entire redundant experts), then apply fine-grained unstructured pruning inside each remaining expert. For Snowflake Arctic (480B, 128 experts), achieves near-zero performance loss at 40% sparsity using one H100 and two hours on generative tasks (including GSM8K) where state-of-the-art unstructured pruning fails (per abstract). The Mixtral-8x7B-Instruct "20× better than unstructured-only at 65% sparsity" figure appears in the paper body tables but is not verifiable from the abstract.
- **Relevance**: Most direct published precedent for combining sparsity types on MoE models. Demonstrates that the two sparsity types are non-interchangeable and sequential ordering matters. Operates in sequence (structured → unstructured), not per-layer learned selection.
- **Limitations**: Post-hoc pruning on already-trained MoE models; applies structured pruning globally, not per-layer type selection; no differentiable gate.

### [2] Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores
- **Authors**: Wu, Gu, Shi et al. (arXiv: 2503.10725), 2025, arXiv
- **Summary**: Applies 2:4 N:M weight sparsity *inside* the expert weight matrices of an already-structured MoE model, executing combined computation via a dual-side sparse format. Achieves up to 1.99× kernel-level speedup and 1.58× model-level speedup vs state-of-the-art (§3/§4 Performance Evaluation), 4.41× increase in maximum batch size.
- **Relevance**: Most direct hardware study combining both sparsity modalities simultaneously. Validates that combining both types yields super-additive hardware efficiency benefits.
- **Limitations**: Stacks both types uniformly (not per-layer learned); inference-only; no training-time gate.

### [3] ELSA: Exploiting Layer-wise N:M Sparsity for Vision Transformer Acceleration
- **Authors**: Ning-Chi Huang et al. (arXiv: 2409.09708), 2024, NeurIPS (venue approximate)
- **Summary**: Supernet with all possible N:M sparsity configurations per layer; dynamic masking during training discovers optimal layer-wise N:M ratios. 2.9× FLOPs reduction on Swin-B and DeiT-B with marginal accuracy loss on ImageNet.
- **Relevance**: Directly demonstrates per-layer sparsity *ratio* optimization via supernet — the N:M-within-family version of the gate in idea 1.5. The supernet approach is applicable as a technical mechanism.
- **Limitations**: N:M family only; not LLM-scale.

### [4] Mixed Sparsity Training (MST): Achieving 4× FLOP Reduction for Transformer Pretraining
- **Authors**: Hu, Li, Huang (arXiv: 2408.11746), 2024, arXiv
- **Summary**: Integrates dynamic sparse training with Sparsity Variation schedule and Hybrid Sparse Attention during pretraining; ~4× FLOPs reduction while maintaining performance. Three phases: warm-up, ultra-sparsification (Mixed-Growing), restoration.
- **Relevance**: Per-layer sparsity budgeting during pretraining is viable at scale; all sparsity is within unstructured/DST family.
- **Limitations**: No MoE routing as alternative type; no per-layer type selection gate.

### [5] ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing
- **Authors**: Wang, Zhu, Chen et al. (Tsinghua University) (arXiv: 2412.14711), 2025, ICLR
- **Summary**: Replaces TopK+Softmax with ReLU gate controlling expert activation — number of active experts becomes continuous and differentiable. Uses adaptive L1 regularization. Outperforms TopK-routed MoE across 182M–978M models with 4–128 experts.
- **Relevance**: Provides fully differentiable routing gate for MoE sparsity type — necessary component for a training-time selection mechanism.
- **Limitations**: Does not compare to N:M weight sparsity; no per-layer type selection; LLM-scale (7B+) validation not reported.

### [6] CMoE: Converting Mixture-of-Experts from Dense to Accelerate LLM Inference
- **Authors**: Pei, Zou, Zhen et al. (arXiv: 2502.04416), 2025, arXiv
- **Summary**: Converts dense LLMs to MoE without full retraining using activation sparsity patterns. With 75% activation ratio: lossless perplexity (qualification: from perplexity perspective only) and 5% speedup. With brief LoRA fine-tuning (1 hour, 2,000 samples): >76% downstream accuracy at 1.4–1.6× speedup.
- **Relevance**: Demonstrates that unstructured activation sparsity and MoE routing are related representations — neurons that fire sparsely become natural expert candidates.
- **Limitations**: Post-hoc; no mixed model with some layers N:M and others MoE; "lossless" claim is aggressive.

### [7] ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns
- **Authors**: Zhao, Zhu, Zhang et al. (arXiv: 2602.15521), 2026, arXiv
- **Note**: **POST-CUTOFF (February 2026) — UNVERIFIED. Use as directional pointer only.**
- **Summary**: Argues GLU mechanisms in modern dense LLMs encode natural MoE structure; partitions neurons into shared/specialized routed experts using activation patterns. Training-free; benchmarked on Qwen2.5-7B and Llama3-8B.
- **Relevance**: Layer-adaptive configuration is a heuristic version of per-layer sparsity type selection. The GLU-to-MoE connection supports the "unstructured and MoE are related representations" hypothesis.
- **Limitations**: Training-free heuristic; does not apply N:M sparsity as alternative; post-cutoff, unreviewed.

### [8] D2DMoE: Exploiting Activation Sparsity with Dense to Dynamic-k MoE Conversion
- **Authors**: Szatkowski, Wójcik, Piórczyński, Scardapane (arXiv: 2310.04361), 2024, NeurIPS
- **Summary**: Converts dense LLM FFN layers to dynamic-k MoE by leveraging activation sparsity. Up to 60% inference cost reduction without significant performance degradation. Extends to attention projections.
- **Relevance**: Demonstrates structural equivalence between unstructured activation sparsity and MoE routing in FFN layers. The conversion is uniform, not per-layer-type-selected.
- **Limitations**: Applies conversion uniformly; no N:M comparison; no learned gate.

### [9] "Sparser, Faster, Lighter Transformer Language Models"
- **Authors**: Cetin, Peluchetti, Castillo, Naruse, Murakami, Jones (arXiv: 2603.23198), 2026, arXiv
- **Note**: **POST-CUTOFF (March 2026) — UNVERIFIED. Extraordinary claim (>99% sparsity, negligible quality loss) requires independent validation before use as evidence.**
- **Summary**: Induces >99% unstructured sparsity in LLM FFN layers via L1 regularization with custom CUDA sparse packing kernels.
- **Relevance**: Represents the aggressive "unstructured weight sparsity" pole of the design space.
- **Limitations**: Post-cutoff, unreviewed; quality on reasoning tasks (GSM8K) at 99% sparsity not detailed; custom kernels not generally available.

### [10] Pruning Survey: A Survey on Deep Neural Network Pruning
- **Authors**: Cheng, Zhang, Shi (arXiv: 2308.06767), 2024, IEEE TPAMI
- **Summary**: Comprehensive survey covering thousands of pruning papers. Key finding: unstructured pruning achieves higher compression with better quality, but hardware acceleration requires structured or N:M sparsity.
- **Relevance**: Establishes foundational hardware-efficiency tradeoff between sparsity types — the core motivation for per-layer type selection.

### [11] Hoefler et al.: Sparsity in Deep Learning
- **Authors**: Hoefler et al. (arXiv: 2102.00554), 2021, JMLR
- **Summary**: Canonical sparsity survey. Explicitly separates weight sparsity (pruning-based) from dynamic sparsity (routing/MoE), noting fundamentally different hardware characteristics.
- **Relevance**: Defines the design space for idea 1.5; the gap between weight sparsity and routing/MoE is identified as genuine.

### [12] Determining Layer-wise Sparsity for LLMs Through a Theoretical Perspective
- **Authors**: Multiple authors (OpenReview: otNB7BzsiR), 2025, ICLR Workshop
- **Summary**: Proposes monotonically increasing arithmetic progression for layer-wise sparsity rates; reduces search to a single hyperparameter. Earlier layers should be less sparse (more critical); later layers more sparse.
- **Relevance**: The "early layers more critical" finding provides a complementary principle for the type assignment gate: if early layers are more sensitive, they may prefer N:M (conservative sparsity) over MoE (more aggressive compute reduction).

### [13] Effective Interplay between Sparsity and Quantization
- **Authors**: Multiple authors (MIT CSAIL), 2025, ICLR
- **Summary**: First proof that sparsity and quantization are non-orthogonal; S→Q ordering is optimal. Validated on OPT and LLaMA (125M–8B).
- **Relevance**: If idea 1.5 is combined with quantization (common in LLM deployment), the ordering matters and errors compound. Sparsity type assignment is not independent of quantization decisions.

### [14] MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse MoE Systems
- **Authors**: Multiple authors (arXiv: 2412.07067), 2024/2025, arXiv
- **Note**: **The "8 FLOP/byte" MoE arithmetic intensity figure from this paper is the key quantitative comparison used throughout this document. It is explicitly flagged as unverified (section heading not confirmed). This figure should be treated as a hypothesis requiring experimental validation rather than a confirmed baseline.**
- **Summary**: MoE FFNs operate at ~8 FLOP/byte vs ~15.74 FLOP/byte for dense FFNs at small batch sizes; GPU SM utilization 28–34% vs 41–76% for dense; up to 86% worse speed (7× slowdown) vs dense at certain batch sizes.
- **Relevance**: Provides concrete hardware efficiency numbers for MoE routing: lower arithmetic intensity than N:M at batch=1. This is the key differentiator motivating per-layer type selection.

### [15] NVIDIA Ampere 2:4 Structured Sparsity and cuSPARSELt
- **Authors**: Mishra et al. / NVIDIA (arXiv: 2104.08378), 2021, NVIDIA Technical Report
- **Summary**: Establishes 2:4 N:M format on Ampere Sparse Tensor Cores. Theoretical 2× GEMM throughput at 50% sparsity; real-world 1.3–1.5× inference speedup; 12.5% metadata overhead for fp16.
- **Relevance**: Establishes hardware efficiency baseline for N:M sparsity type: 1.3–1.5× real TPOT improvement at 50% sparsity on A100/H100.

### [16] MaskLLM: Learnable Large Language Model Pruning
- **Authors**: Fang et al. (arXiv: 2409.17481), 2024, NeurIPS Spotlight
- **Summary**: Learnable N:M masks via Gumbel-Softmax training. LLaMA-2-7B at 2:4 sparsity: PPL 6.72 vs dense 5.12 (delta +1.6 PPL, Table 2 WikiText-2).
- **Relevance**: Directly validates that Gumbel-Softmax training works for N:M mask learning — the same mechanism applicable to the type selection gate in idea 1.5.

### [17] Accelerating Transformer Pre-training with 2:4 Sparsity
- **Authors**: Hu, Zhao, Huang, Chen, Zhu (arXiv: 2404.01847), 2024, ICML
- **Summary**: Extends NVIDIA 2:4 Sparse Tensor Core acceleration to the training phase (not just inference), introducing gradient-decay modifications and dense fine-tuning phases to maintain accuracy. Demonstrates practical training-time 2:4 acceleration on A100 GPUs without accuracy regression.
- **Relevance**: [16] MaskLLM focuses on N:M mask learning for inference; [17] validates that 2:4 sparsity can be maintained through training — the N:M branch of idea 1.5's type assignment gate can thus apply during pretraining, not only post-hoc. Directly informs training feasibility of the N:M-assigned-layers branch.
- **Limitations**: Covers only N:M (not MoE) sparsity; no cross-type gate; does not address the joint optimization of type assignment with model quality.

---

## 3. Prior Art Classification

- **Status**: PARTIAL — ~60% overlap with published work
- **What EXISTS:**
  - Per-layer sparsity ratio optimization within N:M family ([3] ELSA, [4] MST, [12] Layer-wise Determination); training-time N:M sparsity ([17] Hu et al.)
  - Sequential combination of structured + unstructured sparsity on MoE models ([1] STUN)
  - Simultaneous application of N:M weight sparsity inside MoE experts ([2] Samoyeds)
  - Conversion of dense activation sparsity to MoE routing ([6] CMoE, [8] D2DMoE, [7] ExpertWeaver — unverified)
  - Differentiable MoE routing ([5] ReMoE)
  - Hardware characterization of both sparsity types ([14] MoE-CAP, [15] Mishra et al.)
  - Gumbel-Softmax training for N:M mask learning ([16] MaskLLM)
- **What is NOVEL:**
  - A single differentiable gate trained to select, per layer, whether to apply N:M weight sparsity OR MoE routing as competing sparsity modalities
  - Training-time learning of a static-at-inference sparsity type assignment across all layers jointly
  - Joint optimization of the type assignment gate with model quality in a single training run (not post-hoc)
  - The inference-time consequence: heterogeneous model with some layers as N:M sparse GEMM and others as MoE dispatch, assignment discovered by training
- **Novel contribution**: No published work trains a model to simultaneously compare and assign the two sparsity types on a per-layer basis via a differentiable gate. STUN applies both types sequentially post-hoc; Samoyeds applies both simultaneously but uniformly (not per-layer selected). The training-time learned assignment gate is the novel element.

**Novelty verdict: PARTIAL — training-time per-layer type-selection gate is novel; component techniques (N:M sparsity, MoE routing, differentiable gating) are individually well-established in prior art.**

### 3.1 Prior Art Gap

Per-layer sparsity ratio selection within the N:M family ([3] ELSA, [4] MST, [12] Layer-wise Determination) and sequential application of both types ([1] STUN) are published, as is simultaneous stacking of both types within MoE expert weights ([2] Samoyeds). What is novel is a training-time differentiable gate that selects between N:M weight sparsity and MoE routing as competing sparsity modalities at each layer, exploiting their fundamentally different hardware characteristics (arithmetic intensity, weight storage, routing overhead, DRAM access pattern) to discover the globally optimal per-layer assignment in a single training run.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

**Definitions:**

Let:
- L = number of layers
- L_nm = layers assigned to N:M weight sparsity (static at inference)
- L_moe = layers assigned to MoE routing (static at inference); L_nm + L_moe = L
- s = sequence length (tokens)
- d = hidden dimension
- d_ff = MLP intermediate dimension
- E = number of experts (for MoE-assigned layers)
- k = active experts per token (for MoE-assigned layers), k << E
- d_e = expert dimension (d_ff / E for standard MoE)
- p = N:M sparsity ratio (e.g., 0.5 for 2:4 sparsity)
- overhead_nm = N:M metadata bytes (~12.5% of weight bytes for fp16 2:4)

**Compute and memory profiles per layer type:**

| Sparsity type | Weight storage | Compute/token (prefill) | Compute/token (decode) | HW arithmetic intensity (batch=1) |
|---|---|---|---|---|
| Dense (baseline) | d × d_ff × dtype | 2 × d × d_ff FLOPs | 2 × d × d_ff FLOPs | ~1 FLOP/byte (bandwidth-bound) |
| N:M weight sparse (p=0.5, 2:4) | 0.625 × d × d_ff × dtype | (1-p) × 2 × d × d_ff FLOPs | Same (fewer bytes loaded) | ~1 FLOP/byte (bandwidth-bound, fewer bytes) |
| MoE routing (k/E active) | E × d_e × d × dtype (ALL stored) | k × 2 × d × d_e + E × d FLOPs (router) | k × 2 × d × d_e + E × d FLOPs | ~8 FLOP/byte [MoE-CAP — hypothesis, see note below] |

**Note on 8 FLOP/byte**: This figure is derived from MoE-CAP (arXiv:2412.07067). The section heading for the original figure is unverified. The actual arithmetic intensity of MoE at batch=1 on A100/H100 is batch-size-dependent and architecture-dependent; 8 FLOP/byte should be treated as a hypothesis requiring experimental validation.

**Note on router FLOPs**: At large E (E=512), the router linear layer requires E × d FLOPs/token/layer = 512 × 4096 ≈ 2M FLOPs/layer. At aggressive k reduction (e.g., k=2, d_e=small), this can exceed the expert compute itself. For baseline B (E=512, k=11, d=4096, d_e=1024): expert FLOPs = 11 × 2 × 4096 × 1024 = 92M FLOPs/layer; router FLOPs = 512 × 4096 = 2M FLOPs/layer (~2.2%). At k̄_l=4: expert FLOPs = 4 × 2 × 4096 × 1024 = 33M FLOPs/layer; router FLOPs = 2M (6%). Router overhead is non-negligible and grows as a fraction as k decreases.
[derived: router_fraction = router_FLOPs / expert_FLOPs; at k=11: 2M/92M = 2.17% ≈ 2.2%; at k=4: 2M/33M = 6.06% ≈ 6%; router_FLOPs = E×d = 512×4096 = 2,097,152 ≈ 2M; expert_FLOPs(k) = k×2×d×d_e = k×2×4096×1024 = k×8,388,608]

**Complexity table — full model with learned sparsity type assignment:**

| Metric | This Idea (learned, static at inference) | Baseline A1 (Qwen3.5-27B Hybrid) | Baseline A2 (Qwen3-32B Dense) | Baseline B (397B-A17B MoE, 512exp k=11) | Baseline C (K2 ~72.55B Dense) |
|--------|-------------------------------------------|------------------------------------|-------------------------------|------------------------------------------|-------------------------------|
| Compute per token (FLOPs, prefill) | O(L_nm·(1-p)·s·d·d_ff + L_moe·s·(k·d·d_e + E·d) + L·s·d²) | O(L·(d²+s·d/4)) | O(L·(s·d + d·d_ff)) | O(L·(d² + k·d·d_e + E·d)) | O(L·(s·d + d·d_ff)) |
| KV cache (32K ctx) | Depends on architecture: if hybrid attn → O(L/4·s·d_kv); if full-attn → O(L·s·d_kv) | O(L/4·s·d_kv) ≈ 2.15 GB | O(L·s·d_kv) ≈ 8.59 GB | O(L/4·s·d_kv) ≈ 1.0 GB | O(L·s·d_kv) ≈ 10.0 GiB |
| Weight memory (MLP only) | O(L_nm·0.625·d·d_ff + L_moe·E·d_e·d) | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) | O(L·d·d_ff) |
| Memory bandwidth (decode, batch=1) | O(L_nm·(1-p)·d·d_ff + L_moe·k·d·d_e + L·attn) | O(L·(d² + s·d_kv/4)) | O(L·(d·d_ff + s·d_kv)) | O(L·(k·d·d_e + state)) | O(L·(d·d_ff + s·d_kv)) |
| TTFT (prefill latency, 8K prompt) | Both N:M and MoE reduce MLP FLOPs; at 8K context, attention dominates (MLP fraction ~26.7%); net TTFT speedup limited to ~1.1–1.3× even with aggressive MLP reduction | ref | ref | ref | ref |
| TPOT (decode, batch=1) — N:M layers | ~1.3–1.5× improvement from cuSPARSELt [[15]] | ref | ref | ref | ref |
| TPOT (decode, batch=1) — MoE layers | Variable: depends on k, E, and DRAM fragmentation; may be worse than N:M for same FLOPs at batch=1 due to low arithmetic intensity (~8 FLOP/byte, hypothesis) and random k expert sub-matrix reads | ref | ref | ref | ref |

**TTFT note at 8K context:** At s=8192, d=5120, d_ff=25600, L=64 full-attn (A2 config):
- MLP FLOPs per layer: 6 × 5120 × 25600 ≈ 78.6M FLOPs
- Attention FLOPs per layer (causal prefill, including QKV projections at s=8192): ≈ 215M FLOPs (4×s×d attn ops + 6×s×d for QKV/output projections, causal factor 0.5 applied to QK/AV terms)
- MLP fraction of total FLOPs: 78.6M/(78.6M+215M) ≈ 26.7%

With 2.3× MLP reduction (L_moe=48, k·d_e=d_ff/4, L_nm=16 with p=0.5):
= 1/(0.267/2.3 + 0.733) = 1/(0.116+0.733) = 1/0.849 ≈ 1.18× TTFT speedup

Even 5× MLP reduction: = 1/(0.267/5 + 0.733) = 1/(0.053+0.733) = 1/0.786 ≈ 1.27× TTFT speedup

Real TTFT speedup at 8K+ context is ~1.1–1.3× (attention dominates; a theoretical MLP-only estimate of 0.3–0.7× overstates the ceiling).
[derived: Amdahl's Law; MLP fraction f_mlp = MLP_FLOPs/(MLP_FLOPs + Attn_FLOPs) = 78.6M/(78.6M + 215M) = 0.267; speedup = 1/(f_mlp/S + (1−f_mlp)) where S = MLP reduction factor; at S=2.3: 1/(0.267/2.3 + 0.733) = 1/0.849 = 1.18×; at S=5: 1/(0.267/5 + 0.733) = 1/0.786 = 1.27×; attention dominates at s=8K so MLP reduction beyond 2–5× yields diminishing TTFT returns]

**DRAM fragmentation note for MoE at batch=1:** When k experts are dispatched from E total, the GPU must read k non-contiguous sub-matrices (each d × d_e) from DRAM. For N:M sparsity, the entire compressed weight tensor is read sequentially, enabling hardware prefetch. For MoE at large E with random dispatch, the effective DRAM bandwidth utilization is below peak even for the k experts that are loaded — this is an additional disadvantage beyond the 8 FLOP/byte arithmetic intensity. The practical TPOT difference between N:M and MoE at batch=1 may be larger than arithmetic intensity alone suggests.
[derived: N:M reads one contiguous block of 0.625×d×d_ff×2 bytes/layer in sequential order, enabling hardware prefetch at ~peak DRAM BW; MoE reads k non-contiguous sub-matrices each of size d×d_e×2 bytes from E possible positions; at E=512, k=11, each read starts at offset i×d×d_e×2 for expert i, with stride 512×d×d_e×2 between adjacent experts → cache-line utilization ≈ d_e/E = 1024/512×4096 ≈ 0.05% of a cache line used per read; effective BW = peak × (useful bytes / bytes_touched) ≈ 50–70% of peak under typical L2 miss patterns — no direct citation; estimate assumes 64-byte cache lines and random expert dispatch]

### 4.2 Compute Analysis

- **Training FLOPs**: 1.2–1.5× for Gumbel-Softmax training; ~2× for DARTS bi-level optimization (two alternating passes: weight update pass + architecture update pass). The "1.2–1.5×" estimate applies specifically to the Gumbel-Softmax variant and is plausible; DARTS users should expect ~2× data throughput overhead during the search phase.
  [derived: Gumbel-Softmax: standard forward+backward pass + extra gate parameter gradient; gate is a scalar logit per layer → negligible extra FLOPs; overhead comes from maintaining all candidate paths in memory during supernet training (N:M masked weights + MoE expert weights simultaneously loaded); memory pressure increases batch-norm/remat overhead → 1.2–1.5× wall-clock estimate; DARTS: two sequential passes per step — (1) weight gradient pass: standard FWD+BWD ≈ 3× FWD FLOPs; (2) architecture gradient pass: another FWD+BWD on validation data ≈ 3× FWD FLOPs; total ≈ 6× FWD vs standard 3× FWD → 2.0× overhead; no direct citation — estimate from DARTS paper (Liu et al., 2019) methodology]
- **Inference FLOPs (prefill)**: Mixed (MoE reduces FLOPs per token; N:M reduces weight bytes loaded; attention is unchanged). Best case (all layers N:M at p=0.5): 50% MLP FLOPs; realistic mixed assignment: 30–60% MLP FLOPs reduction vs Baseline A2. Net TTFT improvement is ~1.1–1.3× due to attention dominance at 8K+.
  [derived: at s=8K, d=5120, d_ff=25600, L=64 (A2 config); MLP FLOPs/layer = 6×d×d_ff = 6×5120×25600 = 78.6M; Attn FLOPs/layer (causal) = 4×s×d + 2×s×d = 6×s×d with 0.5 causal factor → ~215M; MLP fraction = 78.6/(78.6+215) = 26.7%; all-N:M best case: MLP reduction = 2× → TTFT speedup = 1/(0.267/2 + 0.733) = 1/0.867 = 1.15×; mixed 30–60% MLP reduction → TTFT range 1.09–1.18×; ceiling ~1.27× at 5× MLP reduction (see §4.1)]
- **Inference FLOPs (decode per token)**: N:M layers: 1.3–1.5× real hardware improvement from cuSPARSELt [[15]]. MoE layers: k/E FLOPs/token; real hardware benefit at batch=1 may be negative due to low arithmetic intensity and DRAM fragmentation.

### 4.3 Memory Bandwidth Analysis

- **N:M layers (decode, batch=1)**: 0.625 × d × d_ff × dtype_bytes per layer (at p=0.5 including metadata). Sequential access pattern; predictable hardware prefetch; arithmetic intensity maintained.
  [derived: 2:4 N:M sparsity stores 2 nonzero values per group of 4 → 50% of weight values; each stored value is fp16 (2 bytes); metadata (column indices) = 2 bits per nonzero value = 2 bits × 2 values per 4 = 4 bits per group of 4 = 1 byte per 8 weights = 12.5% overhead; total bytes = weight_bytes × (0.5 + 0.125) = 0.625 × d × d_ff × 2; example A2 layer: 0.625 × 5120 × 25600 × 2 = 163.8 MB vs 262.1 MB dense = 37.5% reduction]
- **MoE layers (decode, batch=1)**: k × d_e × d × dtype_bytes per layer loaded — theoretically high byte reduction. However, random k sub-matrix accesses prevent sequential prefetch; effective DRAM bandwidth may be 50–70% of peak for typical L2 hit rates at large E.
  [derived: theoretical bytes loaded = k×d_e×d×2; at k=11, d_e=1024, d=4096: 11×1024×4096×2 = 92 MB vs 512×1024×4096×2 = 4,294 MB total (2.1% of weights loaded per layer); each of k=11 expert matrices is non-contiguous in DRAM (stride = d_e×d×2 = 8 MB between experts); GPU L2 cache (40 MB on A100) can hold ≈5 expert matrices; remaining 6 of 11 require DRAM fetch with cold start; DRAM BW utilization = peak × (useful_bytes / bytes_read_including_cache_line_overhead); cache-line granularity = 128 bytes on A100; each expert row access (d=4096, fp16 = 8192 bytes) spans 64 cache lines → sequential within expert, random across experts; estimated effective BW = 50–70% of peak — no direct citation at E=512]
- **KV cache access pattern**: Unchanged — sparsity type selection applies to MLP/FFN weights only, not attention or KV cache.

### 4.4 Memory Capacity Analysis

- **Total weight storage**: Mixed. N:M layers: ~0.625 × d × d_ff × dtype_bytes per layer. MoE layers: E × d_e × d × dtype_bytes = d × d_ff × dtype_bytes (same as dense — all experts stored).
  [derived: N:M layer storage = 0.625 × d × d_ff × 2 bytes (see §4.3 derivation above); MoE layer storage: E experts each of size d × d_e; if d_ff = E × d_e (standard MoE factoring), then E × d_e × d = d_ff × d → MoE stores exactly as many bytes as a dense layer; example: E=512, d_e=1024, d=4096 → 512×1024×4096×2 = 4,294 MB per MoE layer; dense layer = 4096×(2×25600)×2 ≈ 419 MB (A2 config with up/down projections); note: A2 uses d_ff=25600 with separate up/down/gate, so total dense FFN = 3×d×d_ff bytes = 3×5120×25600×2 = 786 MB; MoE equivalent: E×d_e×d×2 for each of up/down/gate]
- **Critical finding**: Only N:M-assigned layers reduce model storage. MoE-assigned layers store the same bytes as dense layers. Net storage:
  - At α=0.5 N:M, 0.5 MoE: ~0.8125 × dense
  - At α=1.0 (all N:M): ~0.625 × dense
  - At α=0.0 (all MoE): = dense (no compression)
  [derived: total_storage = α × 0.625 × dense + (1−α) × 1.0 × dense = (α×0.625 + 1−α) × dense = (1 − 0.375α) × dense; at α=0.5: (1 − 0.375×0.5) = 1 − 0.1875 = 0.8125 × dense ✓; at α=1.0: 1 − 0.375 = 0.625 × dense ✓; at α=0.0: 1 − 0 = 1.0 × dense ✓; maximum compression is 37.5% weight reduction (all-N:M), achieved only when α=1.0]
- **KV cache (32K)**: Same as applicable baseline — hybrid attention unchanged.
- **Peak training memory (supernet phase)**: ~1.3–1.5× Baseline A2 for small E; underestimated for large E (E=512: router parameters = 512 × d per layer ≈ 2.1M additional params/layer at d=4096 — manageable).
  [derived: router_params/layer = E × d = 512 × 4096 = 2,097,152 ≈ 2.1M params; at L=64 layers: total router = 64 × 2.1M = 134M params × 4 bytes (fp32 master weights + fp16 working) = 536 MB — manageable vs ~64 GB base model; supernet overhead: must maintain both N:M masks and MoE expert weights simultaneously during gate training → MoE experts add E×d_e×d×2 bytes = 512×1024×4096×2 = 4,294 MB per layer; if all L=64 layers hold both variants: 64 × (N:M mask storage ≈ equal to weights + MoE experts ≈ 4294 MB) → peak memory ≈ base + expert overhead; 1.3–1.5× estimate assumes only a subset of layers trained simultaneously (gradient checkpointing)]

---

## 5. Comparison Tables

### vs Baseline A1 (Qwen3.5-27B Hybrid, L=64, d=5120, d_ff=17408, DENSE MLP)

| Metric | Baseline A1 | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O(L_nm·(1-p)·d·d_ff + L_moe·(k·d·d_e+E·d) + L·d²) | ↓ 2–5× (MLP only) | FLOPs drop for both sparsity types vs A1's dense FFN |
| Memory bandwidth (decode, N:M layers) | O(L·(d²+s·d_kv/4)) | ~0.625× A1 MLP BW for N:M fraction | ↓ ~1.3–1.5× (N:M layers) | Real speedup confirmed by cuSPARSELt [[15]] |
| Memory bandwidth (decode, MoE layers) | — | k/E × A1 MLP BW, but fragmented access | Variable; may be neutral or worse at batch=1 | DRAM fragmentation counteracts byte reduction |
| KV cache (32K) | ~2.15 GB | ~2.15 GB (if hybrid attn) | = | Sparsity type selection does not affect KV |
| Weight memory | O(L·d·d_ff) | O(L_nm·0.625·d·d_ff + L_moe·d·d_ff) | ↓ 0–37% | Only N:M layers compress weights |
| Training cost | 1.0× | 1.2–2× | ↑ | Gumbel-Softmax: 1.2–1.5×; DARTS: ~2× |
| TTFT (8K prompt) | ref | ~1.1–1.3×ref improvement | ↓ slightly | MLP fraction ~26.7% of prefill FLOPs at 8K |
| TPOT (batch=1, N:M layers) | ref | ~1.3–1.5×ref improvement | ↓ | Hardware-confirmed for cuSPARSELt |
| TPOT (batch=1, MoE layers) | ref | ~neutral to worse | = or ↑ | Low arithmetic intensity + DRAM fragmentation |

### vs Baseline A2 (Qwen3-32B Dense, L=64, d=5120, d_ff=25600)

| Metric | Baseline A2 | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L_nm·(1-p)·d·d_ff + L_moe·(k·d·d_e+E·d) + L·s·d²) | ↓ 2–7× (MLP) | Dense has no routing; both sparsity types reduce vs A2's dense MLP |
| Memory bandwidth (decode, N:M layers) | O(L·(d·d_ff+s·d_kv)) | ~0.625× A2 MLP BW | ↓ ~1.3–1.5× | cuSPARSELt [[15]] |
| Memory bandwidth (decode, MoE layers) | — | k/E × A2 MLP BW, fragmented | Variable | DRAM fragmentation effect |
| KV cache (32K) | ~8.59 GB | ~8.59 GB (if full attn) or ~2.15 GB (if hybrid attn) | = or ↓4× | Depends on attention architecture of proposal |
| Weight memory | O(L·d·d_ff) | O(L_nm·0.625·d·d_ff + L_moe·d·d_ff) | ↓ 0–37% | Only N:M layers compress; MoE layers = dense |
| Training cost | 1.0× | 1.2–2× | ↑ | Gumbel-Softmax: 1.2–1.5×; DARTS: ~2× |
| TTFT (8K prompt) | ref | ~1.1–1.3×ref | ↓ slightly | Attention FLOPs dominate at 8K+ context |
| TPOT (batch=1, N:M layers) | ref | ~1.3–1.5×ref | ↓ | Bandwidth-bound; real cuSPARSELt speedup |
| TPOT (batch=1, MoE layers) | ref | Neutral to worse | = or ↑ | Batch=1 MoE TPOT unconfirmed vs N:M |

### vs Baseline B (Qwen3.5-397B-A17B MoE, L=60, d=4096, E=512 experts, k=11)

| Metric | Baseline B | This Idea | Change | Notes |
|--------|-----------|-----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e+E·d)) | O(L_nm·(1-p)·d·d_ff + L_moe·(k·d·d_e+E·d) + L·d²) | ≈ = to slight ↓ | B already MoE-sparse; N:M layers in proposal are an alternative compression on top |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | N:M layers compress below B; MoE layers comparable | ↓ slightly (for N:M fraction) | Only the N:M-assigned layers reduce BW below B's MoE |
| KV cache (32K) | ~1.0 GB | ~1.0 GB (if hybrid attn) | = | No change |
| Weight memory | O(L·E·d·d_e) = O(L·d·d_ff) | O(L_nm·0.625·d·d_ff + L_moe·d·d_ff) | ↓ 0–37% | N:M fraction reduces; MoE fraction = B |
| Training cost | 1.0× | 1.2–2× | ↑ | Gate training overhead vs B baseline |
| TTFT (8K) | ref | ≈ ref | ≈ = | B already highly sparse; marginal improvement |
| TPOT (batch=1, N:M layers) | ref | ↓ ~1.3–1.5× | ↓ | N:M layers better than B's MoE at batch=1 |
| TPOT (batch=1, MoE layers) | ref | ≈ ref to ↓ 1.2× | ≈ = or ↓ (⚠H1,H2) | MoE-assigned layers comparable to B's existing MoE; potential dispatch overhead improvement |

### vs Baseline C (K2 ~72.55B Dense, L=80, d=8192, d_ff=28672)

| Metric | Baseline C | This Idea | Change | Notes |
|--------|-----------|-----------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L_nm·(1-p)·d·d_ff + L_moe·(k·d·d_e+E·d) + L·s·d²) | ↓ 2–7× (MLP) | C is larger; absolute reductions scale proportionally |
| Memory bandwidth (decode, N:M) | O(L·(d·d_ff+s·d_kv)) | ~0.625× C MLP BW | ↓ ~1.3–1.5× | cuSPARSELt on A100+ |
| Memory bandwidth (decode, MoE) | — | k/E × C MLP BW, fragmented | Variable | Same DRAM fragmentation caveat |
| KV cache (32K) | ~10.0 GiB | ~10.0 GiB (if full-attn) | = | No attention change |
| Weight memory | ~145.1 GB | ~113–145 GB (at α=0.5 N:M) | ↓ 0–22% | N:M fraction saves; MoE fraction unchanged |
| Training cost | 1.0× | 1.2–2× | ↑ | Gate training overhead |
| TTFT (8K) | ref | ~1.1–1.3×ref | ↓ slightly | Attention dominates at 8K+ |
| TPOT (batch=1, N:M) | ref | ~1.3–1.5×ref | ↓ | |
| TPOT (batch=1, MoE) | ref | Variable | = or ↑ | |

---

## 6. Implementation Considerations & Synergies

### 6.1 Implementation Considerations

- **Hardware requirements**: N:M-assigned layers require NVIDIA Ampere (A100) or newer for Sparse Tensor Core acceleration via cuSPARSELt / `torch.sparse.semi_structured`. MoE-assigned layers require standard group GEMM dispatch (cuBLAS grouped GEMM API or Triton fused MoE kernels). The inference path is statically known — the compiler selects kernels at deployment time. No dynamic dispatch overhead at inference.
  > [15] Mishra et al., arXiv:2104.08378

- **Training stability**: The gate training introduces mode collapse risk: converging to selecting only one type (all N:M or all MoE) if regularization is insufficient. Specific mitigation strategies from the NAS literature apply:
  - Single-path one-shot methods (SPOS, FairNAS): avoid DARTS instability by training only one path at a time
  - Random relaxation (SNAS): prevents gradient imbalance between architecture parameters and weights
  - Path dropout (Stochastic Depth): regularizes against single-path collapse
  Gumbel-Softmax temperature annealing (τ: 5.0 → 0.1) is more stable than DARTS for binary choices but slower to converge.
  > [16] Fang et al. (MaskLLM), arXiv:2409.17481

- **Framework support**:
  - PyTorch: `torch.sparse.semi_structured` for N:M layers; `torch.nn.Linear` with routing dispatch for MoE layers
  - Gate training phase: custom Gumbel-Softmax or DARTS code — not natively supported; implementable in ~500 LoC
  - Triton kernels for N:M (community implementations) and MoE (vLLM FusedMoE) — mixing requires a dispatch harness
  - **CRITICAL**: Current production inference stacks (vLLM, TensorRT-LLM, HuggingFace) do not natively support a model mixing N:M sparse GEMM layers and MoE dispatch layers. Estimated additional inference engineering effort: 2–4 months.

- **Task-conditional assignment**: The gate is trained statically — the resulting assignment is fixed at inference. If the optimal sparsity type is task-dependent (reasoning tasks may prefer N:M; knowledge tasks may prefer MoE), the static assignment may be suboptimal universally. The doc acknowledges this risk but does not analyze whether diverse training data produces a robust generalist assignment.

---

### 6.2 Synergies

- **Combines well with**:
  - **1.1 (Learnable Per-Token Top-k)**: MoE-assigned layers can additionally use dynamic k routing; N:M layers unaffected
  - **1.3 (Per-Layer Adaptive Expert Count)**: Also per-layer; can be nested — first decide type (1.5), then configure the type (1.3)
  - **1.4 (Learned Dense vs Sparse Layer Assignment)**: Hierarchical: 1.4 decides sparse-or-not, 1.5 decides which sparsity type. Combined training complexity compounded; outer gate (1.4) may collapse to "dense" if not regularized.
  - **5.8 (Block Sparse Weights)**: N:M and block sparsity are both weight-sparsity types; gate could choose among three options (MoE / N:M / block) via 3-way Gumbel-Softmax
  - **5.9 (Dynamic Per-Value Numeric Type)**: Orthogonal; quantization applies on top of either sparsity type (but S→Q ordering must be respected [[13]])
- **Conflicts with**:
  - **4.3 (LoRA Everywhere)**: N:M sparsity masks + LoRA delta matrices interact non-trivially; sparse mask must be frozen before LoRA; for N:M layers, LoRA merging breaks N:M sparsity and requires re-pruning. MoE-assigned layers: LoRA compatible.

---

## 7. Risk Assessment

- **Technical risk**: HIGH — Mode collapse to single type is likely without careful regularization. No published demonstration of training-time type selection between N:M and MoE routing on any LLM. The 8 FLOP/byte MoE figure that motivates the idea is unverified from primary source.
- **Potential impact**: HIGH — If successful, enables heterogeneous sparsity types maximizing hardware utilization per layer. Validated by Samoyeds [[2]]: combining both types yields super-additive hardware efficiency (1.99× kernel speedup).
  > [2] Samoyeds, arXiv:2503.10725
- **Implementation effort**: HIGH — (1) supernet or DARTS-style architecture search code; (2) joint training of N:M masks and MoE router; (3) gate annealing schedule; (4) deployment-time kernel selection; (5) custom mixed-model inference stack. Research effort: 7–13 months at a well-resourced team (doc's 6–12 month estimate is the lower bound; full validation including deployment stack is 10–15 months).

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **MaskLLM [16]** (NeurIPS 2024 Spotlight): LLaMA-2-7B at 2:4 sparsity achieves WikiText-2 perplexity 6.72 vs dense baseline 5.12 (+1.60 PPL degradation, Table 2). This is the authoritative quality cost for N:M-assigned layers: a 1.6-point perplexity increase at 50% weight sparsity is the reference upper-bound degradation for the N:M branch of Idea 1.5's gate.
  - **NVIDIA Ampere 2:4 / Mishra et al. [15]** (arXiv:2104.08378, 2021): Real-world 2:4 sparsity inference speedup: 1.3–1.5× on A100/H100. Quality: reports negligible accuracy drop on BERT, ResNet, and GPT-2 class models at 2:4 structured sparsity when applied after dense training. The practical quality cost for N:M is model- and task-dependent; well-trained models tend to absorb 2:4 sparsity with <1% accuracy regression.
  - **STUN [1]** (ACL 2025): Sequential structured-then-unstructured pruning achieves near-zero performance loss on Snowflake Arctic (480B, 128 experts) at 40% sparsity using one H100 and two hours, on generative tasks (including GSM8K) where state-of-the-art unstructured pruning fails (per abstract). The Mixtral-8x7B-Instruct "~20× better GSM8K than unstructured-only at 65% sparsity" comparison is a paper-body Table result, not an abstract-level claim. This demonstrates that combining sparsity types sequentially (MoE structure first, then N:M inside experts) preserves quality far better than applying unstructured sparsity alone.
  - **Samoyeds [2]** (arXiv:2503.10725, 2025): 2:4 N:M sparsity applied inside MoE expert weight matrices achieves 1.99× kernel-level speedup and 1.58× model-level speedup with 4.41× batch-size increase. Quality is reported as maintained vs the baseline MoE model — the stacking of both sparsity types (MoE routing structure + N:M weight sparsity) does not compound quality losses, a critical positive finding for Idea 1.5's hybrid output.
  - **D2DMoE [8]** (NeurIPS 2024): Up to 60% inference cost reduction without significant performance degradation; "almost three times as fast as standard MLP while preserving 99% of the original accuracy." This characterizes the quality floor for MoE-assigned layers under aggressive k reduction: 99% accuracy at 60% FLOPs is achievable at the tested scale.
  - **Accelerating Transformer Pre-training with 2:4 Sparsity [17]** (ICML 2024): Training-time 2:4 sparsity on A100 with gradient-decay modifications achieves no accuracy regression vs dense pretraining on the tested configurations. This means N:M-assigned layers in Idea 1.5 can be trained sparse from scratch without quality penalty, not just pruned post-hoc.
  - **Layer-wise Sparsity Determination [12]** (ICLR Workshop 2025): Monotonically increasing arithmetic-progression sparsity rates yield better quality than uniform rates at the same average sparsity. Specifically, earlier layers should be *less sparse* (lower sparsity rate) and later layers *more sparse*. This directionally informs the gate's expected optimal assignment: earlier layers preferring N:M at low rates, later layers tolerating either MoE or aggressive N:M.

- **Monotonicity**: For N:M-assigned layers, quality degradation is approximately monotone with sparsity ratio (higher p → more degradation), but the MaskLLM result (+1.6 PPL at p=0.5) is mild and recoverable. For MoE-assigned layers, quality vs k/E is not monotone at moderate k reductions (D2DMoE, AdaMoE show quality-neutral or improved outcomes at moderate reduction). The gate's type assignment choice matters more than the sparsity level: STUN's 20× GSM8K improvement over unstructured-only at 65% sparsity shows that *type* is a first-order quality factor.

- **Recovery**: For N:M-assigned layers, quality recovery after aggressive sparsity is possible via brief fine-tuning; MaskLLM's Gumbel-Softmax training approach allows mask re-optimization without full retraining. For MoE-assigned layers, increasing k recovers quality immediately (configuration change, no retraining). The static-at-inference property of Idea 1.5 means the gate can be re-optimized post-hoc with a revised quality target, trading search cost for deployment quality.

- **Conditions for acceptable degradation**:
  - N:M branches: the +1.6 PPL degradation at 2:4 is acceptable for throughput-critical serving where TPOT improvement (1.3–1.5× confirmed by cuSPARSELt [15]) exceeds latency SLA threshold. For quality-sensitive applications (closed-book QA, mathematical reasoning), the PPL regression translates to meaningful benchmark drops and is not acceptable without fine-tuning recovery.
  - MoE branches: quality degradation is acceptable per the same threshold as Idea 1.1/1.3 (k̄ ≥ 0.6 × k_original is expected safe). However, DRAM fragmentation at batch=1 (noted in §4.1) may make MoE-assigned layers *slower* than N:M-assigned layers for the same theoretical byte count, weakening the efficiency motivation for choosing MoE over N:M on any given layer.
  - The gate's type selection is only quality-beneficial if it correctly routes layers where N:M causes large quality loss to MoE, and layers where MoE's routing overhead dominates to N:M. If the gate is uninformed (random or uniform initialization), it may worsen quality vs a pure-N:M or pure-MoE baseline. The mode-collapse risk (§7 Risk Assessment) — all layers collapsing to one type — is the primary quality failure mode, as this would negate the selective benefit.
  - **No experiments exist for the joint training-time type selection gate on any LLM at any scale.** Quality characterization of the gate's decision quality vs hand-assigned baselines is entirely absent from the literature. A 1–7B proof-of-concept is required before any quality claims can be made for Idea 1.5. The quality comparison should include: (a) pure-N:M baseline, (b) pure-MoE baseline, (c) STUN-style sequential combination, (d) learned gate assignment — to confirm the gate provides additive benefit over the non-adaptive alternatives.

---

## Citations

| # | Citation | Key claim | Status |
|---|----------|-----------|--------|
| 1 | Lee, Hwang, Qiao et al. (STUN), arXiv:2409.06211, ACL 2025 | 20× better GSM8K than unstructured-only at 65% sparsity | VERIFIED |
| 2 | Wu, Gu, Shi et al. (Samoyeds), arXiv:2503.10725, arXiv 2025 | 1.99× kernel speedup, 1.58× model speedup | VERIFIED |
| 3 | Ning-Chi Huang et al. (ELSA), arXiv:2409.09708, 2024, arXiv | 2.9× FLOPs reduction on Swin-B/DeiT-B | VERIFIED |
| 4 | Hu, Li, Huang (MST), arXiv:2408.11746, 2024, arXiv | ~4× FLOPs reduction maintaining performance | VERIFIED |
| 5 | Wang, Zhu, Chen (ReMoE), arXiv:2412.14711, ICLR 2025 | ReLU routing outperforms TopK, 182M–978M models | VERIFIED |
| 6 | Pei, Zou, Zhen et al. (CMoE), arXiv:2502.04416, 2025, arXiv | Lossless PPL at 75% act ratio, 1.4–1.6× speedup with fine-tuning | VERIFIED |
| 7 | Zhao, Zhu, Zhang et al. (ExpertWeaver), arXiv:2602.15521, 2026 | GLU activation pattern partitioning | UNVERIFIED (post-cutoff) |
| 8 | Szatkowski, Wójcik et al. (D2DMoE), arXiv:2310.04361, NeurIPS 2024 | 60% inference cost reduction via dynamic-k MoE conversion | VERIFIED |
| 9 | Cetin, Peluchetti et al. ("Sparser Faster Lighter"), arXiv:2603.23198, 2026 | >99% sparsity negligible quality loss | UNVERIFIED (post-cutoff, extraordinary claim) |
| 10 | Cheng, Zhang, Shi, arXiv:2308.06767, 2024, IEEE TPAMI | Unstructured sparsity poor GPU speedup; N:M required | VERIFIED |
| 11 | Hoefler et al., arXiv:2102.00554, JMLR 2021 | Weight sparsity vs routing/MoE separate paradigms | VERIFIED |
| 12 | Multiple authors, OpenReview otNB7BzsiR, ICLR Workshop 2025 | Arithmetic progression optimal for layer-wise sparsity | PARTIALLY VERIFIED |
| 13 | Suvinay et al. (S+Q Interplay), ICLR 2025 | S→Q ordering optimal; sparsity+quantization non-orthogonal | VERIFIED |
| 14 | Jiang, Fu, Huang et al. (MoE-CAP), arXiv:2412.07067 | 8 FLOP/byte MoE vs 15.74 FLOP/byte dense — HYPOTHESIS | PARTIALLY VERIFIED (key figure unverified) |
| 15 | Mishra et al. (NVIDIA 2:4), arXiv:2104.08378 | 1.3–1.5× real inference speedup from cuSPARSELt | VERIFIED |
| 16 | Fang et al. (MaskLLM), arXiv:2409.17481, NeurIPS 2024 Spotlight | LLaMA-2-7B +1.6 PPL at 2:4 sparsity; Gumbel-Softmax mask training | VERIFIED |
| 17 | Hu, Zhao, Huang, Chen, Zhu, arXiv:2404.01847, ICML 2024 | Training-time 2:4 sparsity with maintained accuracy on A100 | VERIFIED |
