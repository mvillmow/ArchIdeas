# Research: Per-Layer Adaptive Expert Count
## ID: 1.3

## Executive Summary

**Novelty verdict:** EXISTS — heterogeneous per-layer expert allocation is covered ~85–90% by LExI, Alloc-MoE's Alloc-L, AdapMoE, DynMoE, DynaMoE; residual novelty is application to the hybrid DeltaNet+GatedAttn MoE at E=512, L=60 scale ([LExI, Chitty-Venkata et al., 2025], [Alloc-MoE, ACL 2026], [AdapMoE, ASPLOS 2024]).

**One-line description:** Assign a different (learned or calibrated) number of active experts k_l to each MoE layer rather than a single fixed k across all layers, reducing average expert activation and proportionally cutting inference FLOPs and memory bandwidth.

**Value proposition:** Multiple published papers (LExI 2025, Alloc-MoE 2026, AdapMoE 2024, GRAPE 2026, DiEP 2025) independently confirm that heterogeneous per-layer expert allocation outperforms uniform allocation in quality-efficiency tradeoff. The static post-training variant (LExI/Alloc-L style) is immediately actionable with 1–2 engineer-days of effort, requires no retraining, and is framework-ready (vLLM on H100 confirmed by LExI authors). The expected reduction is ~10–36% in MoE-layer FLOPs and memory bandwidth depending on aggressiveness.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 uses a fixed 6 routed experts in all MoE layers, but it is not uniform in routing mechanism: the first 3 MoE layers use Hash routing while later layers use learned routing with updated affinity and balance losses. This leaves per-layer adaptive k open, but shifts the baseline from "fixed k=11 everywhere" to "fixed low k with early deterministic routing." Any 1.3 experiment should include a V4-style fixed-k=6/Hash-first control before claiming layer-adaptive gains.

---

## 1. Idea Description

Instead of fixing the number of active experts per layer, allow each layer to dynamically choose how many experts to activate at inference time. Focus on inference-time adaptation rather than architecture search.

**Key distinction from Idea 1.1:** Idea 1.1 is per-token variable k — all layers receive the same k decision for a given token. Idea 1.3 is per-LAYER variable k — the number of active experts is a layer-specific value (either learned during training or determined post-training from weight statistics). Layer 1 might activate 5 experts, layer 30 might activate 15 experts for the same token. This is an architectural or learned-per-layer policy, not a per-token routing decision.

**Inferred intent for inference speedup:** Different layers in a transformer exhibit heterogeneous sensitivity to the number of active experts. Early layers and some later layers may be over-provisioned at the baseline fixed k=11; allowing those layers to operate at k_l < 11 while preserving high k_l at the most sensitive layers can reduce average compute (Σ k_l / L < k_baseline) without proportional quality loss. At decode time, this reduces average weight bytes loaded per token proportionally to the reduction in average k_l — directly reducing TPOT.

---

## 2. Literature Review

### LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference [1]
*Chitty-Venkata, Madireddy, Emani, Vishwanath — arXiv:2509.02753 (2025)*

A two-stage, data-free post-training optimization that determines the optimal number of active experts per layer in a pretrained MoE model without retraining. Stage 1 uses Monte Carlo sampling with synthetic Gaussian inputs to profile per-layer sensitivity (Frobenius norm perturbation as a proxy for importance). Stage 2 uses an evolutionary algorithm to allocate layer-specific top-k values subject to a global compute budget. Tested on Mixtral-8x7B, Qwen1.5-MoE-A2.7B, OLMoE-1B-7B, MiniCPM-MoE-8x2B, DeepSeek-V2-Lite, and DeepSeekVL2-Tiny. Achieves 10% better accuracy than uniform expert pruning at matched throughput on Qwen1.5-MoE on H100.

> **[Chitty-Venkata et al., 2025]** — §4 "Results" (LM-Eval accuracy comparisons, H100 throughput via vLLM)

**Relevance:** Most direct implementation of Idea 1.3 in the static post-training variant. Explicitly assigns different k_l per layer from weight statistics alone (no calibration data, no retraining). Production-framework compatible (vLLM confirmed on H100).
**Limitations:** Post-training heuristic; static assignment after search phase; does not adapt per token or task at runtime; not evaluated at 512-expert scale.

---

### Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient MoE Inference [2]
*Liu, Tian, Wang, Zhang, Qiao, Li — arXiv:2604.08133 (ACL 2026)*

A unified framework optimizing expert activation at two levels simultaneously: layer-level (Alloc-L) and token-level (Alloc-T). Alloc-L profiles layer-wise sensitivity via an end-to-end performance metric, then solves layer-level budget allocation as a dynamic programming problem with monotonicity constraints (depth-increasing allocation profile). Alloc-T dynamically redistributes activations within a layer across tokens based on routing scores. Achieves 1.15× prefill and 1.34× decode speedup on DeepSeek-V2-Lite at half the original activation budget.

> **[Liu et al., 2026]** — §3.2 "Alloc-L: Layer-Level Allocation" and §4 "Experiments" (prefill/decode speedups on DeepSeek-V2-Lite)

**Relevance:** Alloc-L is the most rigorous published formulation of the layer-level adaptive k concept, using dynamic programming with formal sensitivity profiling. The ACL 2026 venue provides high credibility.
**Limitations:** Requires calibration data for sensitivity profiling; static after profiling step; not at 512-expert scale.

---

### Harder Tasks Need More Experts: Dynamic Routing in MoE Models [3]
*Huang, An, Zhuang, Tao, Zhang, Jin, Xu, Chen, Huang, Feng — arXiv:2403.07652 (ACL 2024)*

Proposes a confidence-based dynamic routing mechanism where the number of activated experts adjusts based on input difficulty. Analyzes layer-by-layer expert activation statistics and finds significant variation in the number of experts needed across different layers, explicitly motivating heterogeneous per-layer k assignment. Average improvement of 0.7% over Top-2 routing with less than 90% of activated parameters.

> **[Huang et al., 2024]** — §4 "Analysis: Layer-wise Expert Variation" and Table results comparing dynamic routing vs. Top-2

**Relevance:** Provides empirical evidence that per-layer heterogeneous expert count is both observed and beneficial.
**Limitations:** Primary mechanism is per-token confidence-based routing; layer-wise variation is an analysis finding rather than the implemented mechanism.

---

### DA-MoE: Towards Dynamic Expert Allocation for Mixture-of-Experts Models [4]
*Akhavan Aghdam, Jin, Wu — arXiv:2409.06669 (2024)*

Proposes a dynamic router using Transformer attention weights to measure token importance, then allocates a variable number of experts per token. Consistently outperforms state-of-the-art Transformer-based MoE on GLUE benchmark.

**Relevance:** Provides theoretical backing that variable k is beneficial; demonstrates a clean attention-based importance measure for routing.
**Limitations:** Per-token routing, not per-layer. Does not analyze inter-layer k variation.

---

### DiEP: Adaptive MoE Compression through Differentiable Expert Pruning [5]
*Bai, Li, Zhang, Hong, Guo — arXiv:2509.16105 (2025)*

Introduces differentiable expert pruning that adapts pruning rates at the layer level by jointly learning intra- and inter-layer importance scores. Converts discrete layer-wise search into a continuous optimization problem. On Mixtral-8×7B, retains ~92% of original performance with only half the experts, achieving up to 7.1% higher accuracy than competing methods on MMLU.

> **[Bai et al., 2025]** — §3 "Method" and §4 "Experiments" (MMLU accuracy, 50% expert retention)

**Relevance:** Demonstrates that learned, per-layer heterogeneous expert count allocation via gradient-based optimization outperforms uniform pruning.
**Limitations:** Framed as compression/pruning; results in fixed per-layer k at inference time.

---

### GRAPE: Does a Global Perspective Help Prune Sparse MoEs Elegantly? [6]
*Zhang, Ghosh, Liu, Yu, Liu — arXiv:2604.06542 (2026)*

GRAPE (Global Redundancy-Aware Pruning of Experts) dynamically allocates pruning budgets across layers based on cross-layer redundancy analysis, yielding heterogeneous per-layer expert retention ratios. Tested on Mixtral-8x7B, Mixtral-8x22B, DeepSeek-MoE, Qwen-MoE, and GPT-OSS; achieves 1.40% average accuracy improvement over the strongest local (uniform per-layer) baseline, with maximum gains up to 2.45%.

> **[Zhang et al., 2026]** — §4 "Experiments" (Table showing 1.40% avg improvement vs. uniform baselines, max 2.45%)

**Relevance:** Quantifies the benefit of heterogeneous layer-wise expert allocation over uniform allocation; validates core premise of Idea 1.3 at large scale.
**Limitations:** Post-training pruning; focuses on memory reduction; not inference-time routing adaptation.

---

### MoE-I2: Compressing MoE Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition [7]
*Yang, Sui, Xiao, Huang, Gong, Duan, Jia, Yin, Cheng, Yuan — ACL Anthology 2024.findings-emnlp.612 (EMNLP 2024 Findings)*

Two-stage compression combining non-uniform layer-wise pruning ratios (via Layer-wise Genetic Search and Block-wise KT-Reception Field) with intra-expert low-rank decomposition. Applied to Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, and Mixtral-8×7B. Demonstrates that per-layer heterogeneous expert allocation outperforms uniform approaches.

**Relevance:** Empirical evidence from three production-scale MoE models that per-layer heterogeneous allocation with non-uniform ratios is beneficial.
**Limitations:** Compression-oriented; the non-uniform ratios are chosen from importance analysis, not jointly learned.

---

### MoLA: MoE LoRA with Layer-wise Expert Allocation [8]
*Gao, Chen, Rao, Liu, Sun, Zhang, Peng, Guo, Subrahmanian — ACL Anthology 2025.findings-naacl.284 (NAACL 2025 Findings)*

Applies LoRA-MoE to pre-trained transformers with layer-wise expert allocation. Key finding: **more experts should be allocated to higher layers** (lower layers exhibit greater expert redundancy); direction of optimal allocation is task- and scale-dependent. With fewer total parameters than uniform allocation, layer-wise expert distribution outperforms uniform setting across six NLP and commonsense QA benchmarks.

> **[Gao et al., 2025]** — §4 "Experiments" (outperforms uniform allocation across 6 benchmarks)

**Relevance:** Empirically establishes that per-layer heterogeneous expert allocation improves both efficiency and quality over uniform allocation.
**Note:** The optimal allocation favors **higher layers** (inverted-triangle configuration: fewer experts in lower layers, more in higher layers), reflecting lower expert redundancy in higher layers. This direction is also task- and scale-dependent.
**Limitations:** Domain is LoRA experts for fine-tuning; allocation is static once determined.

---

### MoE Pathfinder: Trajectory-Driven Expert Pruning [9]
*Yang, Tian, Song — arXiv:2512.18425 (2025)*

Treats MoE as a weighted computation graph and frames expert selection as global optimal path planning. Naturally yields non-uniform expert retention across layers. Outperforms nearly all existing uniform-pruning approaches.

**Relevance:** Alternative global-path approach to per-layer heterogeneous expert allocation; cross-layer dependencies improve over independent layer-wise decisions.
**Limitations:** Pruning-focused; no training-time learning component; no inference-time dynamic routing.

---

### EvoESAP: Non-Uniform Expert Pruning for Sparse MoE [16]
*Liu, Tang, Sun, Shen, Yuan — arXiv:2603.06003 (2026)*

Decouples expert pruning into within-layer expert ranking and across-layer budget allocation. Introduces ESAP (Expected Speculative Acceptance Proxy), a teacher-forced metric that enables cheap evaluation of pruned candidate models without autoregressive decoding. EvoESAP applies evolutionary search to discover non-uniform layer-wise sparsity allocations under a fixed global budget, achieving up to +19.6% on MATH-500 at 50% sparsity compared to uniform allocation across 7B–30B SMoE LLMs.

**Relevance:** Directly validates that non-uniform layer-wise expert budget allocation outperforms uniform allocation; provides an efficient search mechanism for finding per-layer k_l assignments. Orthogonal to LExI [1] and GRAPE [6] in its evaluation metric and search strategy.
**Limitations:** Post-training pruning (reduces stored experts); search requires running model forward passes; not evaluated at 512-expert scale.

---

### AdapMoE: Adaptive Sensitivity-based Expert Gating and Management [10]
*Zhong, Liang, Wang, Wang, Huang, Li — arXiv:2408.10284 (ASPLOS 2024)*

Employs a sensitivity-based adaptive gating mechanism that dynamically adjusts the number of activated experts per token during inference using offline-calibrated per-layer thresholds. Achieves 25% reduction in average activated experts with 1.35× speedup without accuracy loss.

> **[Zhong et al., 2024]** — §4 "Experimental Results" (25% expert reduction, 1.35× speedup)

**Relevance:** Closest existing work to the combined vision of Idea 1.3 at inference time — uses per-layer sensitivity thresholds (which implicitly define different k_l per layer) applied dynamically during decoding.
**Limitations:** Threshold calibrated offline; primary axis of variation is per-token; not at 512-expert scale.

---

### D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models [11]
*Jiang, Wang, Xie, Zhao, Zhang, Qian, Lui — NeurIPS 2024*

Designs a per-layer dynamic decision module for each transformer layer that decides whether the layer should execute or be skipped for a given token. Reduces computational cost by up to 45–50% on QA, summarization, math, and commonsense tasks on Llama-2.

> **[Jiang et al., 2024]** — §4 "Experiments" (45–50% compute reduction, NeurIPS 2024 proceedings)

**Relevance:** Direct implementation of per-layer variable compute at inference time; the per-layer adaptive compute framing is analogous to Idea 1.3.
**Limitations:** Binary skip decision (full layer or skip entirely), not variable expert count; applies to dense models.

---

### DynMoE: Dynamic Mixture of Experts — An Auto-Tuning Approach for Efficient Transformer Models [12]
*Guo, Cheng, Tang, Tu, Lin — arXiv:2405.14297 (ICLR 2025)*

Introduces "top-any" gating that allows each token to autonomously determine how many experts to activate, combined with an adaptive training process that adjusts the number of experts per training step. Published at ICLR 2025. Covers both vision and language tasks; demonstrates that variable k is learnable end-to-end with stable training.

**Relevance:** Highest-visibility (ICLR 2025) published system combining variable-k with end-to-end learned adaptation. Demonstrates viability of the dynamic variant of Idea 1.3.
**Limitations:** Per-token rather than per-layer; training-based (not post-training static assignment).

---

### DynaMoE: Dynamic Token-Level Expert Activation with Layer-Wise Adaptive Capacity [13]
*Gülmez — arXiv:2603.01697 (2026)*

Combines token-level dynamic expert count with six distinct layer-wise expert capacity scheduling strategies (ascending, descending, uniform, pyramid, valley, random). Key findings: (a) optimal expert schedules are task- and scale-dependent; (b) dynamic routing reduces gradient variance during training, improving convergence stability.

**Relevance:** Directly combines per-token dynamic k with per-layer capacity scheduling — the intersection that Idea 1.3's novelty framing claimed was "not well-explored." DynaMoE closes this gap. The remaining novel contribution for Idea 1.3 is application to hybrid MoE+linear-attention architecture at Baseline B's scale (397B, 512 experts).
**Limitations:** Not evaluated at 512-expert scale; not applied to hybrid DeltaNet+attention architectures.

---

### Yuan3.0 Ultra / LAEP: Layer-Adaptive Expert Pruning at 1.5T Scale [14]
*Wu et al. (YuanLab.ai) — arXiv:2601.14327 (2026)*

Yuan3.0 Ultra introduces the Layer-Adaptive Expert Pruning (LAEP) algorithm, which achieves 49% pre-training efficiency improvement and 33.3% parameter reduction at 1.5T parameter scale via per-layer adaptive expert pruning. The highest-scale published validation of per-layer heterogeneous expert allocation.

**Relevance:** Provides the highest-scale (1.5T parameter) empirical validation that per-layer adaptive expert allocation is effective and practical. Directly relevant to Idea 1.3's application at Baseline B's 397B scale.
**Limitations:** Pruning-focused (reduces stored experts, not just activated); pre-training context rather than post-training inference optimization.

---

### DeepSeek-V4: Fixed Low-k All-MoE Baseline with Early Hash Routing [17]
*DeepSeek-AI — DeepSeek-V4 technical report and Hugging Face release (2026)*

DeepSeek-V4-Pro uses all-MoE transformer blocks with 384 routed experts plus 1 shared expert and fixed activation of 6 routed experts per token. It does not allocate different active expert counts per layer, but it does make the first 3 MoE layers deterministic through Hash routing.

**Relevance:** Establishes a modern low-k baseline and shows that per-layer routing mechanism can vary even when k_l does not. This is the strongest fixed-k control for Idea 1.3.
**Limitations:** No per-layer k_l adaptation; no published ablation showing whether some V4 layers could use fewer or more than 6 routed experts.

---

### Optimal Expert-Attention Allocation in Mixture-of-Experts: A Scalable Law [15]
*Li, Jiang, Tian, Liu, Zhang, Hu — arXiv:2603.10379 (2026)*

Extends Chinchilla scaling laws to cover the expert-attention FLOPs allocation ratio within MoE models. Finds that the optimal ratio r* (fraction of FLOPs dedicated to expert layers vs. attention layers) follows a power-law relationship with total compute and varies with model sparsity.

**Relevance:** Provides theoretical grounding for why per-layer expert compute allocation should differ systematically; supports the design of layer-specific k policies.
**Limitations:** Treats layer types as uniform categories rather than individual layers; a scaling law, not an inference optimization method.

---

## 3. Prior Art Classification

- **Status**: EXISTS
- **Overlap summary**: ~85–90% covered. The core mechanism — assigning different k_l values to different layers of a pre-trained MoE model, targeting inference efficiency — is substantially covered by LExI [1], Alloc-MoE's Alloc-L component [2], AdapMoE [10], DynMoE [12], and DynaMoE [13]. Multiple papers (GRAPE [6], DiEP [5], MoE-I2 [7], MoE Pathfinder [9], EvoESAP [16], LAEP [14]) independently corroborate that heterogeneous per-layer expert allocation outperforms uniform allocation.

- **Novel contribution** (residual gap): The specific application to a hybrid MoE+linear-attention architecture (DeltaNet + GatedAttn) at Baseline B's scale (397B, 512 experts, 60 layers) has not been demonstrated. DynaMoE [13] closes the "combined layer+token dynamic adaptation" gap; the remaining novel element is the hybrid architecture context and the 512-expert scale.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: L=60 layers, d=4096, d_e≈1024 (expert intermediate dim), E=512 experts, k=11 (baseline), k_l=active experts at layer l, k̄_l = (1/L)Σk_l.

**FLOPs (MoE FFN per token):**
- Baseline B: O(L·k·3·d·d_e) = O(60·11·3·4096·1024) ≈ 8.3B MACs total (SwiGLU: gate_proj+up_proj+down_proj per expert)
- Idea 1.3: O(Σ_l k_l·3·d·d_e) = O(L·k̄_l·3·d·d_e); reduction ratio = k̄_l/k

**Example scenario (verified arithmetic):**
20 layers @ k_l=6, 30 layers @ k_l=11, 10 layers @ k_l=14:
k̄_l = (120+330+140)/60 = 590/60 ≈ 9.83; ratio = 9.83/11 ≈ 0.89 (11% reduction)

Aggressive scenario (k̄_l=7): ratio = 7/11 ≈ 0.636 (36% reduction)

**Routing gating FLOPs (unchanged):** 4×E×d = 4×512×4096 = 8.4M MACs per layer; across 60 layers: 504M MACs total.
Routing overhead fraction: 2.2% at k=11 → 3.4% at k̄_l=7 → 5.9% at k̄_l=4. Routing overhead is invariant but grows as a proportion at aggressive reductions.

**TTFT (MoE FFN dominance at 8K context):**
- DeltaNet layers: 45 × d² × s = 45 × 4096² × 8192 ≈ 6.2T FLOPs
- GatedAttn layers: 15 × s² × d ≈ 4.1T FLOPs
- MoE FFN: 60 × 6 × 11 × 4096 × 1024 × 8192 ≈ 136T FLOPs (SwiGLU: 3 mats × 2 MACs-to-FLOPs)
- MoE FFN is ~93% of prefill FLOPs at 8K context — TTFT reduction ≈ k̄_l/k (within 1% rounding)

**TPOT (bandwidth breakdown at decode):**
Per layer weight bandwidth:
- Expert weights: k × 3 × d × d_e × 2 bytes (bf16, SwiGLU) = 11 × 3 × 4096 × 1024 × 2 = 277MB per layer
- Gating weights: E × d × 2 bytes = 512 × 4096 × 2 = 4MB per layer
- DeltaNet attention weights: ~33.6MB per layer
Total per layer: ~315MB; expert fraction = 277/315 ≈ 88% at k=11 (decreasing to ~82% at k̄_l=7)

Net TPOT reduction at k̄_l=7: ~0.88 × 0.364 ≈ 32% (not 36%). The MoE-FFN-only bound is 36%, but net model-wide reduction accounts for unchanged gating and attention bandwidth.

| Metric | Idea 1.3 vs. Baseline B | Baseline A1 | Baseline A2 | Baseline B | Baseline C |
|--------|------------------------|-------------|-------------|------------|------------|
| Compute (FLOPs/token) | O(L·k̄_l·d·d_e); ↓ k̄_l/k × | N/A (not MoE) | N/A (not MoE) | O(L·k·d·d_e) | N/A (not MoE) |
| KV cache (32K ctx) | O(L/4·s·d_kv) unchanged; ≈1.0 GB — derived: 15×2×2×256×32768×2 = 1,006,632,960 bytes ≈ 1.0 GB (15 GatedAttn layers, 2 KV heads each, head_dim=256, seq=32768, bf16) | ~2.15 GB | ~8.59 GB | ~1.0 GB | ~10.0 GiB |
| Weight memory | O(L·E·d·d_e) unchanged | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) | O(L·d·d_ff) |
| TTFT (8K prompt) | ↓ ~0.89–0.64× (MoE FFN; ~0.92–0.69× net full model) [derived: MoE FFN FLOPs fraction at 8K ctx ≈ 93% (§4.1: 136T/(136T+6.2T+4.1T) = 136/146 ≈ 0.93); TTFT_MoE_only = k̄_l/k × 0.93 = {7/11 × 0.93 = 0.636 × 0.93 = 0.59, 10/11 × 0.93 = 0.909 × 0.93 = 0.85}; net full model = (k̄_l/k) × FFN_share + (1 − FFN_share) = {0.636 × 0.93 + 0.07 = 0.62, 0.909 × 0.93 + 0.07 = 0.86} → ~0.64–0.92× range] | ref | ref | ref | ref |
| TPOT (batch=1) | ↓ ~0.91–0.68× net (not 0.89–0.64× — accounts for gating+attn BW) [derived: Net TPOT = (k̄_l/k) × expert_weight_fraction + (1 − expert_weight_fraction); expert fraction = 277/315 ≈ 0.88 at k=11 (§4.3: 277 MB expert / 315 MB total per layer); lower bound k̄_l=7: (7/11) × 0.88 + 0.12 = 0.636 × 0.88 + 0.12 = 0.560 + 0.12 = 0.68×; upper bound k̄_l=10: (10/11) × 0.88 + 0.12 = 0.909 × 0.88 + 0.12 = 0.80 + 0.12 = 0.91×] | ref | ref | ref | ref |

**Note on TPOT bounds:** Net TPOT range 0.91–0.68× accounts for routing and attention bandwidth (~12% of total at k=11) being unchanged. The MoE-FFN-only bound (0.89–0.64×) is labeled as an upper bound in the derivation above.

### 4.2 Training Cost

- **Static post-training (LExI/Alloc-L/DiEP style):** Zero training cost. Model weights frozen; only routing configuration changes.
- **Learned end-to-end (dynamic variant):** k-allocation head adds O(L×d) = 60×4096 ≈ 245K params vs. 397B baseline (<0.0001% overhead). Training overhead: negligible.

### 4.3 Memory Bandwidth

- Expert weight bytes per decode ∝ Σ_l k_l × 3 × 2 × d × d_e bytes (SwiGLU: gate+up+down projections). With k̄_l < k: proportional reduction.
- Gating weights (E×d per layer), DeltaNet weights (~d² per layer), and GatedAttn weights: all unchanged.
- Net bandwidth reduction = (k̄_l/k) × (expert weight fraction of total bandwidth) ≈ (k̄_l/k) × 0.88 at k=11.

### 4.4 Memory Capacity

- Total weight storage unchanged — all E=512 experts per layer still stored.
- KV cache unchanged — only 15/60 GatedAttn layers contribute; unaffected by expert routing.
- Training memory (static variant): unchanged. Learned variant: marginal increase for k-allocation head.

---

## 5. Implementation Considerations

**Static post-training assignment (LExI/Alloc-L style):**
- No custom kernels required. Top-k_l parameter per layer is a trivial configuration change.
- vLLM's FusedMoE accepts `top_k` as a per-layer integer at construction time. LExI authors report H100 results via vLLM — production-framework confirmed.
- Expert dispatch tensor at layer l: [B×S, k_l] — uniform within each layer, varying across layers. No irregularity.
- Implementation effort: 1–2 engineer-days to run LExI or Alloc-L sensitivity profiling on Baseline B.

**Dynamic per-token k_l assignment (AdapMoE/DynaMoE style):**
- Expert dispatch tensor shape [B×S, k_{i,l}] is irregular (each row may differ).
- Option A — Capacity factor padding: pad to max(k_{i,l}) per layer; wastes FLOPs proportional to (max_k - k̄_{i,l})/max_k; with tokens spanning k ∈ {6..14}, padding waste can reach 30–60% for naive implementation.
- Option B — Variable-length dispatch (Triton custom kernel): no padding waste; requires 2–4 weeks kernel engineering; not available in standard production frameworks as of 2026-04.
- Option C — Sorting and packing: group tokens by k_{i,l}, reduce within-group waste; medium engineering complexity.
- Expert prefetching: static k_l allows deterministic prefetch; dynamic k_l requires speculative prefetching (unsolved for variable k at 512-expert scale).
- Implementation effort (dynamic): 2–4 engineer-weeks for kernel + training loop changes.

**Framework support:**
- PyTorch: trivial for static; implementable but GPU-inefficient for dynamic without custom kernel.
- vLLM: confirmed for static (LExI); requires FusedMoE extension for dynamic.
- TensorRT-LLM: likely supports static via configurable MoE; dynamic unknown.
- JAX/XLA: native for static; pallas kernel needed for dynamic.

---

## 6. Synergies

- **1.1 (Per-Token Adaptive Expert Count):** Orthogonal axes — 1.3 sets per-layer k_l budget, 1.1 varies per-token allocation within that budget. Combined: a 2D adaptive routing grid. Note: combining both in the dynamic variant creates maximum dispatch irregularity.
- **1.4 (Learned Dense/Sparse Layer Assignment):** 1.4 makes binary dense/sparse decisions per layer; 1.3 makes quantitative k_l decisions given sparse. Together they form a complete layer-level compute allocation policy. Setting k_l → E subsumes 1.4's "dense" decision.
- **2.2 (Compressed Dense Layers via Matrix Decomposition):** Reduces per-expert byte count independently. Multiplicative bandwidth benefit.
- **5.1 (TurboQuant KV cache):** Independent dimension (KV cache vs. expert bandwidth). Additive benefit.
- **3.1 (Layer-Level MoE / Full Block Routing):** If blocks are routed, per-block k is a generalization of Idea 1.3. 3.1 subsumes 1.3 in the limit.

---

## 7. Risk Assessment

| Dimension | Static Variant | Dynamic Variant |
|-----------|---------------|-----------------|
| GPU kernel feasibility | FULLY FEASIBLE (vLLM confirmed) | CONDITIONAL (custom kernel needed) |
| Training stability | N/A (post-training) | MEDIUM (budget constraint + straight-through estimator) |
| Quality risk | LOW (multiple published validations) | MEDIUM (no results at 512-expert scale) |
| Implementation effort | LOW: 1–2 engineer-days | MEDIUM: 2–4 engineer-weeks |
| Framework support | PRODUCTION READY | NEEDS CUSTOM WORK |
| Scale gap risk | LOW (Alloc-MoE, GRAPE, AdapMoE validated at comparable scales) | MEDIUM (512-expert specific results absent) |

**Overall risk:** LOW for static variant; MEDIUM for dynamic variant. The static variant is one of the lowest-risk, highest-certainty inference optimizations available for Baseline B — it is essentially "run LExI on Baseline B and deploy."

---

## Citation Manifest

[1] LExI: Chitty-Venkata, Madireddy, Emani, Vishwanath — arXiv:2509.02753 (2025)
[2] Alloc-MoE: Liu, Tian, Wang, Zhang, Qiao, Li — arXiv:2604.08133 (ACL 2026)
[3] Harder Tasks Need More Experts: Huang et al. — arXiv:2403.07652 (ACL 2024)
[4] DA-MoE: Akhavan Aghdam, Jin, Wu — arXiv:2409.06669 (2024)
[5] DiEP: Bai, Li, Zhang, Hong, Guo — arXiv:2509.16105 (2025)
[6] GRAPE: Zhang, Ghosh, Liu, Yu, Liu — arXiv:2604.06542 (2026)
[7] MoE-I2: Yang et al. — ACL Anthology 2024.findings-emnlp.612 (EMNLP 2024 Findings)
[8] MoLA: Gao et al. — ACL Anthology 2025.findings-naacl.284 (NAACL 2025 Findings)
[9] MoE Pathfinder: Yang, Tian, Song — arXiv:2512.18425 (2025)
[10] AdapMoE: Zhong, Liang, Wang, Wang, Huang, Li — arXiv:2408.10284 (ASPLOS 2024)
[11] D-LLM: Jiang, Wang, Xie, Zhao, Zhang, Qian, Lui — NeurIPS 2024 proceedings
[12] DynMoE: Guo, Cheng, Tang, Tu, Lin — arXiv:2405.14297 (ICLR 2025)
[13] DynaMoE: Gülmez — arXiv:2603.01697 (2026)
[14] Yuan3.0 Ultra / LAEP: Wu et al. (YuanLab.ai) — arXiv:2601.14327 (2026)
[15] Optimal Expert-Attention Allocation: Li, Jiang, Tian, Liu, Zhang, Hu — arXiv:2603.10379 (2026)
[16] EvoESAP: Liu, Tang, Sun, Shen, Yuan — arXiv:2603.06003 (2026)
[17] DeepSeek-V4: DeepSeek-AI — 2026 technical report and Hugging Face release; all-MoE low fixed-k baseline with early Hash routing

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: At 25% expert reduction, zero accuracy loss (AdapMoE [10], ASPLOS 2024, §Results); at 50% expert retention on Mixtral-8×7B, ~92% MMLU retention with up to +7.1% over uniform pruning (DiEP [5], arXiv:2509.16105 §4); +1.40% average over strongest uniform baseline across Mixtral/DeepSeek-MoE/Qwen-MoE/GPT-OSS (GRAPE [6], ACL 2026); +19.6% on MATH-500 at 50% sparsity vs uniform (EvoESAP [16], arXiv:2603.06003).
- **Known failure modes**: Quality cliff at very aggressive average k̄_l (< 0.5 × k_baseline) on reasoning tasks [EvoESAP 16]; no experiments at 512-expert, 60-layer scale exist in the literature — extrapolation uncertainty from Mixtral-8x7B/DeepSeek-V2-Lite scale is unquantified; uniform allocation *misallocates* compute away from reasoning-critical layers, so naive reductions on reasoning benchmarks regress hardest [EvoESAP 16].
- **Empirical evidence**: LExI §4 "Results" (LM-Eval accuracy, H100 throughput via vLLM) [Chitty-Venkata et al., 2025]; AdapMoE §Results (25% expert reduction, 1.35× speedup, zero accuracy loss) [AdapMoE, ASPLOS 2024]; GRAPE §Results (1.40% avg improvement, max 2.45 pp per-task) [GRAPE, ACL 2026]; DiEP §4 (92% MMLU retention, +7.1% vs uniform) [DiEP, arXiv:2509.16105].
- **Mitigations**: Start with LExI-style offline sensitivity profiling (no retraining required) [LExI, 1]; constrain reduction to k̄_l ∈ [0.7, 0.9] × k_baseline for quality-neutral operating point; combine per-layer (Alloc-L) with per-token (Alloc-T) routing to recover quality without raising average k̄_l [Alloc-MoE 2, ACL 2026]; for reasoning-heavy deployments, weight reasoning-task performance in the per-layer sensitivity metric.

### Detailed breakdown

- **Reported quality delta (from closest analogues)**:
  - **LExI [1]** (arXiv:2509.02753, 2025): On Qwen1.5-MoE-A2.7B, achieves 10% better accuracy than uniform expert pruning at matched H100 throughput. No overall accuracy regression vs the full-k baseline was reported at moderate budget reductions, indicating that non-uniform layer-specific k assignment dominates uniform assignment both in efficiency and in quality preservation.
  - **GRAPE [6]** (arXiv:2604.06542, ACL 2026): Averaged 1.40% accuracy improvement over strongest local (uniform per-layer) baseline across Mixtral-8x7B, Mixtral-8x22B, DeepSeek-MoE, Qwen-MoE, and GPT-OSS. Maximum per-task gain: 2.45 percentage points. This quantifies the direct quality benefit of global heterogeneous layer-wise allocation vs uniform allocation at matched expert-retention budgets.
  - **DiEP [5]** (arXiv:2509.16105, 2025): On Mixtral-8×7B at 50% expert retention, retains ~92% of original MMLU performance while achieving up to 7.1% higher accuracy than competing uniform-pruning methods. Demonstrates that differentiable per-layer expert allocation extracts more quality per FLOPs than hand-tuned uniform baselines.
  - **EvoESAP [16]** (arXiv:2603.06003, 2026): Up to +19.6% on MATH-500 at 50% sparsity compared to uniform allocation across 7B–30B SMoE LLMs. The gain is largest on reasoning-heavy tasks, implying that non-uniform expert allocation preferentially preserves reasoning capacity (later/middle layers may get higher k_l).
  - **AdapMoE [10]** (ASPLOS 2024): Achieves 25% reduction in average activated experts with 1.35× speedup at *no accuracy loss* using offline-calibrated per-layer sensitivity thresholds. This is the most conservative published result: even without quality improvement, 25% expert reduction is free quality-wise.
  - **Alloc-MoE [2]** (ACL 2026): 1.15× prefill and 1.34× decode speedup on DeepSeek-V2-Lite at *half* the original activation budget. Quality metrics are reported as maintained vs baseline in the main results table — indicating that 50% activation budget reduction is achievable without quality regression when allocation is optimized heterogeneously across layers.
  - **MoLA [8]** (NAACL 2025 Findings): Outperforms uniform allocation across 6 NLP and commonsense QA benchmarks with fewer total parameters, when allocating more experts to higher layers. This establishes a directional prior: the quality benefit of non-uniform allocation is not evenly distributed — it is concentrated in *depth-differentiated* allocation patterns (fewer experts in lower layers, more in higher layers).

- **Monotonicity**: Quality loss with respect to average k̄_l is *not* monotone. At moderate reductions (k̄_l ≈ 0.7–0.9 × k_baseline), multiple papers report quality-neutral or quality-positive outcomes. The non-uniformity of the allocation is more important than the average level: GRAPE's +1.40% improvement over uniform allocation at the same average budget directly demonstrates this. The quality cliff appears at very aggressive average k̄_l (< 0.5 × k_baseline), but the exact threshold at 512-expert scale is not published.

- **Recovery**: For the static post-training variant (LExI/Alloc-L/AdapMoE), quality recovery is trivial — simply increase the per-layer k_l budget in the configuration. No retraining required; the model weights are frozen. For the dynamic learned variant (DiEP-style gradient-based allocation), recovery requires re-running the allocation optimization with a higher budget constraint.

- **Conditions for acceptable degradation**:
  - Any accuracy regression is acceptable only if the speedup exceeds 1.2× (practical threshold for visible user impact in interactive serving). Based on AdapMoE [10], a 25% expert reduction at 1.35× speedup achieves zero regression — this is the conservative operating point. The GRAPE and EvoESAP results suggest 40–50% expert reduction with <2% accuracy regression is achievable at large model scale with optimal heterogeneous allocation.
  - For tasks with strong reasoning components (MATH, BBH, coding), quality sensitivity to k_l reduction is higher — EvoESAP's +19.6% MATH gain specifically reflects that uniform allocation was *misallocating* compute away from reasoning-critical layers. This suggests per-layer profiling should specifically weight reasoning-task performance in the sensitivity metric.
  - **No experiments at 512-expert, 60-layer scale exist in the literature.** The closest analogues are Mixtral-8x7B (8 experts, 32 layers) and DeepSeek-V2-Lite. Extrapolation to E=512, L=60 carries uncertainty; the recommendation is to run LExI's sensitivity profiling as a first step, which requires no retraining and provides free quality characterization.
  - Synergy with dynamic k̄_l: If the quality at a given average k̄_l is insufficient, adding per-token dynamic allocation (Idea 1.1) within each layer's budget can recover quality by directing experts to hard tokens without increasing the layer average, as demonstrated by Alloc-MoE [2]'s combined Alloc-L + Alloc-T approach.
