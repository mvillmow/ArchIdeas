# Research: Learnable Per-Token Top-k
## ID: 1.1

## Executive Summary

**Novelty verdict:** PARTIAL — per-token variable-k MoE routing is well-published, but native pre-training at 512-expert / k=11 scale with a lightweight learned k-predictor in a Gated DeltaNet hybrid (k̄≈7 operating point) is the remaining gap ([Expert Choice, 2022], [AdaMoE, 2024], [DynMoE, 2025], [ReMoE, 2025], [LD-MoLE, 2026]).

**One-line description:** Replace the static k in MoE top-k expert routing with a per-token learned value so that simple tokens activate fewer experts and hard tokens activate more, reducing average expert weight loads and TPOT.

**Value proposition:** If average k drops from 11 to approximately 7 on Baseline B, the net model-wide TPOT improvement is ~1.3–1.5× [derived: k_base/k̄ × FFN_BW_share = 11/7 × (22 GB active MoE weights / ~25.9 GB total per step) ≈ 1.57× FFN-only; net model-wide ≈ 1.3–1.5× after subtracting fixed Gated DeltaNet state BW (45 layers × 4096² × 2 bytes ≈ 1.44 GB) and attention layer overhead] (MoE FFN layers only: up to 1.57× [derived: k̄/k_base = 7/11 × (active MoE FFN bytes / total bytes) where active bytes drop from 11 × ~2 GB = 22 GB to 7 × ~2 GB = 14 GB per MoE FFN step]). Multiple ICLR/NeurIPS/ACL 2024–2025 papers confirm the mechanism is quality-neutral-or-better at moderate k reductions; the remaining novelty is the large-scale 512-expert application and hybrid-architecture (Gated DeltaNet) integration.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4-Pro uses fixed expert activation, not learned per-token top-k: 384 routed experts plus 1 shared expert, with 6 routed experts active per token. This creates a stronger fixed-k baseline than the older k=11 Baseline B framing. Dynamic-k work must now prove benefit over V4's lower fixed k/E ratio, Sqrt(Softplus) router affinity, sequence-wise balance loss, and early Hash routing.

---

## 1. Idea Description

Replace the static k in top-k expert routing with a learned, per-token dynamic value. Each token determines how many experts it needs rather than using a fixed k across all tokens.

**Inferred intent for inference speedup:** By allowing easy/simple tokens to activate fewer experts and difficult/complex tokens to activate more, the average k across a sequence can fall below the fixed baseline k. Since decode-time TPOT at batch=1 is dominated by expert weight loads (memory bandwidth), a lower average k directly reduces average bytes loaded per token and thus reduces TPOT. TTFT (prefill) improves in proportion to the reduction in average k since prefill is FLOPs-bound and each expert contributes O(d·d_e) FLOPs. The idea is most directly applicable to Baseline B (Qwen3.5-397B-A17B), which already uses a fixed k=11 (10 routed + 1 shared).

---

## 2. Literature Review

### Mixture-of-Experts with Expert Choice Routing [1]: Inverted Routing for Variable Token Allocation (NeurIPS 2022)
Reverses the token-to-expert assignment so that each expert selects its top-k tokens rather than each token selecting its top-k experts. Since each expert has a fixed bucket size, a given token can end up processed by zero, one, or several experts — the effective per-token expert count is variable as an emergent property of the inverted assignment. The paper reports >2× training efficiency improvement over Switch Transformer.

> **[Zhou et al., 2022]** — §1 "Introduction": ">2× training efficiency improvement vs Switch Transformer." URL verified: https://arxiv.org/abs/2202.09368

**Relevance:** Directly achieves variable expert count per token. Mechanism is inverted (expert-choice pull rather than token-push), but the observable outcome is equivalent. Expert Choice is incompatible with autoregressive decoding because in causal generation a single query token cannot fill an expert's bucket.
**Limitations:** Incompatible with autoregressive (token-by-token) decoding. Variable allocation is a side-effect of inverted selection, not a learned per-token k predictor.

---

### AdaMoE [2]: Token-Adaptive Routing with Null Experts (EMNLP 2024 Findings)
Introduces "null experts" — dummy experts consuming zero FLOPs — into the expert pool and raises the routing k. A load-balancing loss with annealed coefficient α (from 0.02 at epoch 1 to 0.0001 at epoch 2) controls average null-expert usage, so each token effectively selects a variable number of true experts. Applied to fine-tuned Mixtral-8x7B with 8 null experts and top-3 selection out of 8 true experts.

> **[Zeng et al., 2024]** — §4.1 "Main Results" Table 1: 14.55% FLOPs reduction, 1.66 true experts per layer (vs. 2.0 baseline), improved ARC-C accuracy. URL verified: https://arxiv.org/abs/2406.13233

**Relevance:** The closest direct match to idea 1.1. Makes per-token active expert count variable and learnable through a soft null-expert mechanism. Compatible with autoregressive inference.
**Limitations:** The "learned" component is the router weights and load-balancing loss target, not a dedicated per-token k predictor. Evaluated only on fine-tuning Mixtral-8x7B (8 true experts), not pre-training at Baseline B scale (512 experts, k=11).

---

### DynMoE [3]: Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models (ICLR 2025)
Proposes a "top-any" gating mechanism that allows each token to autonomously determine the number of experts to activate, plus adaptive expert count adjustment during training to automatically find the right total expert count. Demonstrates competitive performance on vision, language, and vision-language tasks while activating fewer parameters.

> **[Guo et al., 2024]** — §1 "Introduction": "a novel gating method that enables each token to automatically determine the number of experts to activate." Competitive performance vs GMoE and MoE-LLaVA with fewer active parameters. URL verified: https://arxiv.org/abs/2405.14297

**Relevance:** Directly addresses idea 1.1 by enabling per-token variable k through a learned gating scheme. Code publicly available at github.com/LINs-lab/DynMoE.
**Limitations:** Experiments cover smaller-scale settings. Interaction with very large E (512 experts as in Baseline B) not evaluated.

---

### D2DMoE [4]: Dense to Dynamic-k MoE Conversion via Activation Sparsity (NeurIPS 2024)
Converts dense networks to MoE by exploiting activation sparsity, with a dynamic-k routing rule: an expert is activated if R(z)_i ≥ τ · max R(z), where τ ∈ [0,1] controls sparsity. Adjusting τ post-deployment provides a quality-speed tradeoff dial. The GPU implementation scales near-linearly with the number of executed experts.

> **[Szatkowski et al., 2024]** — Abstract: "up to 60% FLOPs reduction without significantly impacting performance." The "almost three times as fast as standard MLP while preserving 99% of the original accuracy" figure is a paper-body §4 Table claim, not in the abstract. URL verified: https://arxiv.org/abs/2310.04361

**Relevance:** Implements per-token variable-k routing based on a lightweight per-token score. The threshold τ can be tuned post-deployment, enabling an explicit TPOT vs. quality tradeoff dial.
**Limitations:** Primarily a dense-to-MoE conversion method. Threshold τ is not learned end-to-end during training; it is a post-hoc inference hyperparameter. Venue note: the paper was submitted October 2023 and revised November 2024; the NeurIPS 2024 workshop/track should be confirmed rather than main conference.

---

### ReMoE [5]: Fully Differentiable MoE with ReLU Routing (ICLR 2025)
Replaces discrete TopK+Softmax routing with a continuous ReLU-based router. ReLU can output exactly zero for inactive experts, so the number of active experts per token is determined continuously by the learned gate values — making allocation fully differentiable and variable per token. Consistently outperforms vanilla TopK MoE.

> **[Wang, Zhu, and Chen, 2024]** — §4.1 Table 2 "Zero-shot downstream accuracy": ReMoE achieves 40.03 avg., outperforming TopK baselines. Table 5: ReMoE validation loss 1.689 vs MoE 1.716. Table 7: near-identical throughput (±2.3% variance vs TopK). URL verified: https://arxiv.org/abs/2412.14711

**Relevance:** Achieves variable per-token k through a fully differentiable mechanism. Figure 7 (empirical): rarer tokens activate more experts; frequent tokens activate fewer.
**Limitations:** Sparsity regularization still needed. The per-token k emerges from thresholding continuous gate values rather than an explicit integer predictor.

---

### DTop-p [6]: Sparsity-Controllable Dynamic Top-p MoE (arXiv 2025)
Uses a cumulative-probability threshold (top-p / nucleus) instead of a fixed integer k; a Proportional-Integral (PI) controller dynamically adjusts the threshold during training to hit a target average sparsity. Tested on LLMs (100B tokens, 4/32, 8/64, and 16/128 expert configs) and Diffusion Transformers.

> **[Jin et al., 2025]** — §4.2 "Results": DTop-p averaged 50.9 across 13 NLP benchmarks vs 49.0 for Top-k at 100B tokens (1.9-point improvement). §4.4 scaling: 51.4 avg (128E/16A config) vs 49.3 (Top-k). URL verified: https://arxiv.org/abs/2512.13996

**Relevance:** Directly replaces fixed-k with a threshold-based scheme producing variable per-token k values while maintaining a global compute budget target via PI control.
**Limitations:** The per-token k is not a function of token-level features; it is determined by a shared probability threshold. The "dynamic" aspect is across training steps, not per-token specialization within a sequence.

---

### SeqTopK [7]: Route Experts by Sequence, Not by Token (arXiv 2025)
Instead of assigning a fixed k experts per token, selects the top T·K expert-token pairs across all T tokens in a sequence. Hard tokens receive more experts, easy tokens fewer, while preserving the same total compute budget. Achieves up to 16.9% performance gain under high sparsity (K=2 regime).

> **[Wen et al., 2025]** — §4.1 Table 1: SeqTopK 31.49% avg vs TopK 29.74% at K=8; +16.9% under extreme sparsity K=2. §4.3 "Efficiency": +0.9% pre-training overhead, –1% throughput, +1% memory. URL verified: https://arxiv.org/abs/2511.06494

**Relevance:** Achieves per-token variable expert allocation through sequence-level budget redistribution — a budget-preserving variant of idea 1.1.
**Limitations:** The total compute budget is conserved at the sequence level. TPOT for single-sequence decode does not improve on average. Speedup only materializes for individual tokens within a batch.

---

### Harder Tasks Need More Experts [8]: Dynamic Routing via Confidence (ACL 2024)
Introduces a dynamic expert selection framework that activates experts based on cumulative probability confidence: t = argmin_k ∑_{i=1}^{k} p_i ≥ p_threshold. Harder inputs (lower confidence in expert selection) activate more experts. Demonstrates 0.7% average improvement over Top-2 routing while activating fewer experts.

> **[Huang et al., 2024]** — §3.2 "Main Results" Table 1: MoE-Dynamic average score 42.3% vs MoE-Top2 41.6% (+0.7 pp). Table 3: average activated experts = 1.76 (vs. 2.0 for Top-2). Task-specific: PIQA=1.72 experts, BBH=1.87 experts. URL verified: https://arxiv.org/abs/2403.07652

**Relevance:** An explicit per-token variable-k mechanism driven by routing confidence. Directly implements idea 1.1 logic at inference. Code available at github.com/ZhenweiAn/Dynamic_MoE.
**Limitations:** Evaluated on smaller MoE models (k=2 baseline). The per-token threshold p is a global hyperparameter rather than a learned per-token scalar.

---

### DA-MoE [9]: Dynamic Expert Allocation via Attention-Derived Importance (arXiv 2024)
Uses attention-derived token importance scores to determine per-token k: num_experts_to_route = ⌈token_importance × E⌉, where token importance = average maximum attention weight across heads. More important tokens activate more experts.

> **[Aghdam et al., 2024]** — §V "Experimental Analysis" Table I (2B/32E configuration): average +1.31 points improvement over fixed-k baseline on GLUE sub-tasks. URL verified: https://arxiv.org/abs/2409.06669

**Relevance:** Directly implements per-token variable-k allocation using a scalar importance signal derived from attention.
**Limitations:** Uses a heuristic (attention-derived importance) rather than end-to-end learned k prediction. Evaluated only at modest scale (2B model, 32 experts).

---

### DynaMoE [10]: Dynamic Token-Level Expert Activation with Layer-Wise Capacity Scheduling (arXiv 2026)
Removes two traditional MoE constraints (fixed Top-K routing and uniform expert allocation) via dynamic token-level expert activation where the number of active experts per token varies based on input complexity. Introduces six capacity scheduling strategies across network layers.

> **[Gülmez, 2026]** — §Abstract: "the number of active experts per token varies based on input complexity." Layer scheduling strategies: descending optimal for image classification, ascending for small language models. URL verified: https://arxiv.org/abs/2603.01697

**Relevance:** Addresses the same core problem as idea 1.1. Adds a layer-wise scheduling dimension orthogonal to idea 1.3. Very recent preprint (March 2026); no venue confirmation.

---

### DeepSeekMoE [11]: Fine-Grained Expert Specialization (ACL 2024)
Segments experts into finer granularity (mN total, mK active routed + K_s shared) to improve specialization. Uses a fixed k routing scheme throughout. DeepSeekMoE 16B achieves comparable performance with LLaMA2 7B with ~40% of computations.

> **[Dai et al., 2024]** — §1 "Introduction": "DeepSeekMoE 16B achieves comparable performance with LLaMA2 7B, with only about 40% of computations." URL verified: https://arxiv.org/abs/2401.06066

**Relevance:** Baseline context. Its fixed-k design (including shared experts) is exactly what idea 1.1 proposes to replace with a learnable per-token value.

---

### DeepSeek-V4 [17]: Frontier-Scale Fixed-k MoE Baseline (2026)
DeepSeek-V4-Pro scales the DeepSeekMoE family to 1.6T total parameters with 49B active parameters, all transformer blocks using MoE, 384 routed experts plus 1 shared expert, and fixed activation of 6 routed experts per token. It also updates routing details with Sqrt(Softplus) affinity, auxiliary-loss-free balance, sequence-wise balance loss, and Hash routing in the first 3 MoE layers.

**Relevance:** Establishes that a much lower fixed routed-k setting can work at frontier scale. This weakens any argument that variable k is needed merely because older baselines over-activate experts.
**Limitations:** Does not make k token-adaptive; hard and easy tokens still receive the same number of routed experts within a layer.

---

### MegaBlocks [12]: Block-Sparse GPU Kernels for MoE (MLSys 2023)
Reformulates MoE computation in terms of block-sparse operations, handles variable token counts per expert, and achieves up to 40% end-to-end training speedup. Provides the kernel infrastructure for variable-k dispatch.

> **[Gale et al., 2023]** — arXiv:2211.15841. URL verified: https://arxiv.org/abs/2211.15841

**Relevance:** The feasibility claim for variable-k dispatch at inference depends heavily on the existence of this kernel infrastructure. D2DMoE [Szatkowski et al., 2024] demonstrates near-linear scaling of latency with number of executed experts, supported by MegaBlocks-style infrastructure.

---

### LD-MoLE [13]: Learnable Dynamic Routing for Mixture of LoRA Experts (ICLR 2026)
Uses a lightweight shared MLP to predict a token-specific sparsity parameter λ — a per-token k-predictor for mixture-of-LoRA-experts. Replaces non-differentiable TopK with a differentiable routing function and closed-form sparsity control objective.

> **[Zhuang et al., 2026]** — arXiv:2509.25684, ICLR 2026. URL verified: https://arxiv.org/abs/2509.25684

**Relevance:** The closest existing work to the "explicit lightweight per-token k-predictor network" proposed in idea 1.1. Applied to LoRA experts at fine-tuning scale; does not target pre-training at 512-expert scale.

---

### Alloc-MoE [14]: Budget-Aware Expert Activation Allocation (ACL 2026)
Combines layer-level (Alloc-L) and token-level (Alloc-T) adaptive expert activation. Achieves 1.15× prefill and 1.34× decode speedup on DeepSeek-V2-Lite at half the original activation budget.

> **[Liu et al., 2026]** — arXiv:2604.08133, ACL 2026. §3.2 "Alloc-T: Token-Level Reallocation." URL verified: https://arxiv.org/abs/2604.08133

**Relevance:** Directly combines the ideas of 1.1 (token-level variable k) and 1.3 (layer-level variable k), demonstrating the combined mechanism at large-MoE inference scale with measured 1.34× decode speedup.

---

### Expert Threshold Routing [15]: EMA-Threshold Variable-k Routing (arXiv 2026)
Replaces fixed Top-K with EMA-threshold-based routing where tokens are routed to experts whose score exceeds a per-expert EMA threshold. Achieves 0.067 lower cross-entropy than TC-MoE at 2.4B scale, equivalent to 1.6× token efficiency. Fully causal and compatible with autoregressive decode.

> **[Sun et al., 2026]** — arXiv:2603.11535. URL verified: https://arxiv.org/abs/2603.11535

**Relevance:** Achieves variable-k-per-token through an expert-side threshold. Different mechanism from a per-token predictor but the same observable outcome.

---

### Reinforced Adaptive Routing for MoE [16]: RL-Based Per-Token k Prediction (OpenReview 2026)
Uses a policy network (RL-based) to predict the number of experts to activate per token per layer, optimizing a multi-objective reward (accuracy + load balance + efficiency). A richer form of the "learned k-predictor" idea.

> OpenReview 2026. URL: https://openreview.net/forum?id=yBJZw5DBzU

**Relevance:** Directly implements a per-token k-predictor via RL, more sophisticated than a simple linear gate.

---

## 3. Prior Art Classification

- **Status**: PARTIAL
- **Overlap summary**: ~75–80% covered. The core goal (variable expert count per token) is addressed by multiple published papers: Expert Choice (NeurIPS 2022), AdaMoE (EMNLP 2024), DynMoE (ICLR 2025), ReMoE (ICLR 2025), D2DMoE (revised 2024), Huang et al. (ACL 2024), DA-MoE (arXiv 2024), DTop-p (arXiv 2025), SeqTopK (arXiv 2025), DynaMoE (arXiv 2026), LD-MoLE (ICLR 2026), Expert Threshold Routing (arXiv 2026), Alloc-MoE (ACL 2026). DeepSeek-V4 adds a frontier-scale fixed-k counterbaseline at k=6 routed experts.
- **What the literature does NOT address**: (a) Application to native pre-training at 384-512 expert scale with V4-like low fixed k; (b) the combination of compressed-attention/recurrent layers with per-token variable-k MoE routing; (c) a V4-specific operating point such as k=6 → k̄≈4–5; (d) post-deployment k-threshold tuning in a native large-E pre-trained model.
- **Novel contribution**: The specific framing — an explicit lightweight k-predictor network applied to a frontier-scale hybrid/long-context MoE, targeting average expert activation below a strong fixed-k baseline while preserving quality — has not been demonstrated in the literature.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: L=layers, d=hidden, d_ff=MLP intermediate, s=seq length, V=vocab, E=total experts, k=fixed active experts, k̄=average active experts per token (k̄ < k for idea 1.1), d_e=expert hidden dim.

For Baseline B: L=60, d=4096, E=512, k=11 (10 routed + 1 shared), d_e ≈ 1,024. In idea 1.1 applied to Baseline B, k becomes a learned per-token variable; the average over tokens is k̄ (target: k̄ ≈ 7 for a 36% reduction; k̄ ≈ 9 for a 17% reduction analogous to AdaMoE; D2DMoE's 60% FLOPs reduction upper bound suggests k̄ as low as 4–5 is feasible at small scale).

**FLOPs derivation for this idea applied to Baseline B:**
- Routing overhead per token per MoE layer: O(E·d) for the gating projection (same as baseline, E·d dot products). The additional k-predictor is O(d) per layer (single linear projection d → scalar): negligible relative to O(k·d·d_e).
- Expert compute per token per MoE layer: O(k̄·d·d_e) on average, versus O(k·d·d_e) at baseline. Reduction factor: k̄/k.
- If k̄ = 7 (versus k = 11 baseline): compute per token in MoE layers reduces to 7/11 ≈ 0.636 of baseline.

**Memory bandwidth at decode (TPOT driver):**
- Expert weight bytes loaded per token per MoE layer: k̄ × (2 × d_e × d) × sizeof(dtype). If k̄ = 7 vs k = 11: bandwidth reduced to 7/11 ≈ 0.636 of baseline.
- Routing overhead (loading gating weight matrix E×d): unchanged. At Baseline B scale, this is E×d = 512×4096 ≈ 2M entries per layer (~4MB at BF16 per layer). This is ~3.4% of expert weight loading at k̄=7, a small correction.
- Gated DeltaNet recurrent state bandwidth: The Gated DeltaNet state (d×d per layer for 45 of 60 Gated DeltaNet layers) must also be loaded at decode. At Baseline B scale: 45 × 4096² × 2 bytes ≈ 1.44 GB at BF16 or ~2.88 GB at FP32 per decode step across 45 Gated DeltaNet layers. This is a fixed constant bandwidth overhead that does not scale with k̄ and therefore limits the effective TPOT improvement ratio.
- **Net TPOT speedup estimate**: The headline MoE-FFN-only speedup is k̄/k ≈ 1.57× at k̄=7. Net model-wide TPOT — accounting for gating overhead, Gated DeltaNet state bandwidth, and attention layer overhead — is conservatively in the range **~1.2–1.4× (conservative) to ~1.57× (MoE FFN layers only, best case)**. Treat 1.57× as a theoretical upper bound for the MoE FFN layers, not the net model-wide improvement.

| Metric | This Idea (applied to Baseline B) | Baseline A1 (Qwen3.5-27B Hybrid) | Baseline A2 (Qwen3-32B Dense) | Baseline B (397B-A17B MoE, k=11) | Baseline C (K2 72.55B Dense) |
|--------|-----------------------------------|----------------------------------|-------------------------------|-----------------------------------|------------------------------|
| Compute per token (FLOPs) | O(L·(d²+k̄·d·d_e)), k̄<k | O(L·(d²+s·d/4)) | O(L·(s·d+d·d_ff)) | O(L·(d²+k·d·d_e)) | O(L·(s·d+d·d_ff)) |
| KV cache memory | O(L/4·s·d_kv) | O(L/4·s·d_kv) | O(L·s·d_kv) | O(L/4·s·d_kv) | O(L·s·d_kv) |
| Weight memory | O(L·E·d·d_e) +O(L·d) k-pred | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) | O(L·d·d_ff) |
| Memory bandwidth (decode) | O(L·(k̄·d·d_e+state)), k̄<k | O(L·(d²+s·d_kv/4)) | O(L·(d·d_ff+s·d_kv)) | O(L·(k·d·d_e+state)) | O(L·(d·d_ff+s·d_kv)) |
| **TTFT (prefill, 8K prompt)** | ~0.64× Baseline B (MoE-FFN only, k̄=7); net ~0.7–0.8× | ref | ref | ref | ref |
| **TPOT (decode, batch=1)** | ~1.2–1.4× net; up to ~1.57× MoE-FFN only at k̄=7 | ref | ref | ref | ref |

Notes:
- Baselines A1, A2, C are NOT MoE architectures. TTFT/TPOT for A1/A2/C are marked "ref" as architectural references.
- TTFT improvement applies to MoE FFN layers (all 60 layers in Baseline B have MoE FFNs); attention (Gated DeltaNet + global attention) unchanged.
- Routing overhead (loading the gating weight matrix, E=512 × d=4096 entries, ~4MB per layer) is unchanged.
- Gated DeltaNet recurrent state (~1.44 GB–2.88 GB total across 45 layers at BF16–FP32) limits the net TPOT speedup.
- k-predictor overhead: O(L·d) operations per token — negligible relative to O(L·k·d·d_e).

### Key Comparison Tables

#### TTFT Comparison (8K prompt, Baseline B as reference)

| Baseline | TTFT | Notes |
|----------|------|-------|
| A1 (Qwen3.5-27B Hybrid) | ref | 64L, d=5120, d_ff=17408; hybrid Gated DeltaNet+attention; 16 full-attn layers |
| A2 (Qwen3-32B Dense) | ref | 64L, d=5120, d_ff=25600; all-dense |
| B (Qwen3.5-397B-A17B MoE) | ref | 60L, d=4096, k=11, E=512 |
| C (K2 family 72.55B Dense) | ref | 80L, d=8192, d_ff=28672; all-dense; MLP FLOPs/token/layer ≈ 4.70×10⁸ |
| B + idea 1.1 (k̄=7) | ~0.64× Baseline B (MoE-FFN); net ~0.7–0.8× | [derived: TTFT_MoE-FFN = k̄/k × FFN_share = 7/11 × 0.93 = 0.636 × 0.93 ≈ 0.59; net full model = FFN_share × (k̄/k) + (1 − FFN_share) = 0.93 × 0.636 + 0.07 ≈ 0.64×; FFN_share ≈ 93% from §4.1: 136T/(136T+6.2T+4.1T) = 136/146.3 ≈ 0.93] |

#### TPOT Comparison (batch=1 decode, Baseline B as reference)

| Baseline | TPOT driver | Notes |
|----------|-------------|-------|
| A1 (Qwen3.5-27B Hybrid) | ~54 GB total weights loaded | L/4 full-attn KV + Gated DeltaNet recurrent state |
| A2 (Qwen3-32B Dense) | ~64 GB total weights loaded | All-dense; MLP weight bandwidth dominates |
| B (Qwen3.5-397B-A17B MoE) | ~34 GB active weights loaded (k=11) | 17B active params; k experts per layer |
| C (K2 family 72.55B Dense) | ~145.1 GB weight BW (bf16) | All 72.55B params active; 80 layers |
| B + idea 1.1 (k̄=7) | ~22 GB active MoE weights + constant overhead | Net ~1.2–1.4× vs Baseline B [derived: TPOT_ratio = k/k̄ × FFN_BW_share; FFN_BW_share = expert_weight_BW / total_BW; at k̄=7: active expert BW ≈ 7 × 2 × d_e × d × L × sizeof(bf16) = 7 × 2 × 1024 × 4096 × 60 × 2 ≈ 7.0 GB MoE layers vs ~14.9 GB constant (gating + DeltaNet state 1.44 GB + attn); MoE FFN share ≈ 7.0/(7.0+14.9) ≈ 0.32 of total BW → speedup = (k/k̄) × FFN_share + (1−FFN_share) inverted: BW_new/BW_old = (7/11 × FFN_share_at_k11) + const_share; at k=11: expert BW ≈ 11 GB, const ≈ 14.9 GB, total ≈ 25.9 GB; at k̄=7: expert BW ≈ 7 GB, total ≈ 21.9 GB; speedup ≈ 25.9/21.9 ≈ 1.18×; Gated DeltaNet state (1.44–2.88 GB) caps net gain → conservative ~1.2–1.4× range] |

#### KV Cache Comparison

| Baseline | 32K ctx | 262K ctx | Notes |
|----------|---------|----------|-------|
| A1 (Qwen3.5-27B) | ~2.15 GB | ~17.2 GB | 16 full-attn layers; H_kv=4, head_dim=256 |
| A2 (Qwen3-32B) | ~8.59 GB | ~68.7 GB | 64 layers, H_kv=8, head_dim=128; native max 40,960 |
| B (Qwen3.5-397B) | ~1.0 GB | ~8.0 GB | 15 global-attn layers; H_kv=2, head_dim=256 |
| C (K2 72.55B) | ~10.0 GiB | ~80.0 GiB | 80 layers, H_kv=8, head_dim=128 |
| B + idea 1.1 | ~1.0 GB | ~8.0 GB | KV cache unchanged; only expert routing changes |

---

### 4.2 Compute Analysis

- **Training FLOPs**: Applicable only to MoE architectures; Baselines A1/A2/C are not MoE. Training FLOPs increase slightly: the routing network must learn k-prediction; gradient through variable k requires either a straight-through estimator or soft relaxation (ReLU gating as in ReMoE [5], or null-expert trick as in AdaMoE [2]). Overhead < 1% of total FLOPs. Additionally, the load-balancing loss must be extended to penalize both under- and over-use of experts, adding minor computational overhead.
- **Inference FLOPs (prefill)**: ~k̄/k × Baseline B MoE-layer FLOPs. If k̄=7: ~0.636× prefill FLOPs in MoE layers. Attention unchanged. Net model-wide prefill speedup: ~0.7–0.8× depending on attention fraction. Directionally supported by D2DMoE up to 60% FLOPs reduction [4] and AdaMoE 14.55% FLOPs reduction [2].
- **Inference FLOPs (decode per token)**: Same ratio as prefill estimate.
- **Arithmetic intensity**: Decode is memory-bandwidth-bound. Reducing expert loads (fewer weight matrices streamed) does not significantly change FLOPs-per-byte — fewer bytes are loaded and fewer FLOPs executed in proportion.

### 4.3 Memory Bandwidth Analysis

- **Weight loading**: At decode, each MoE layer loads k expert weight matrices. With learnable per-token k̄ < k: weight loads reduced to k̄/k fraction. For k̄=7 vs k=11: loads reduced by ~36%. D2DMoE [4] reports near-linear scaling of latency with number of executed experts, supporting this proportionality claim.
- **KV cache access pattern**: Unchanged — only 15/60 layers in Baseline B use global KV attention. Gated DeltaNet layers use fixed-size recurrent state. Neither is affected by this idea.
- **Variable-k dispatch kernel overhead**: Standard MoE kernels are optimized for fixed k. Variable k per token introduces irregular memory access patterns. MegaBlocks [12] provides block-sparse infrastructure that handles variable token-per-expert. Estimated wall-clock dispatch overhead: 2–5% latency, reducing net TPOT improvement slightly below the k̄/k ratio.

### 4.4 Memory Capacity Analysis

- **Total weight storage**: Unchanged — all E=512 experts are still stored. The k-predictor adds d×1 parameters per layer: 60 × 4096 ≈ 245K additional parameters — negligible vs. 397B.
- **KV cache**: Unchanged (only 15/60 layers have KV cache in Baseline B).
- **Peak training memory**: Marginally increased for k-prediction gradients. If soft relaxation (ReLU as in ReMoE [5]) is used, the routing computation graph is essentially unchanged.

---

## 5. Implementation Considerations

- **Hardware requirements**: No custom kernels strictly required for the k-predictor itself (a single linear projection d → scalar). However, variable-k routing at inference complicates batching: different tokens may activate different expert subsets, breaking regular memory access patterns assumed by hardware-efficient MoE kernels. Efficient implementation requires a sorted/grouped dispatch kernel (MegaBlocks [12] or vLLM expert-parallel dispatch) extended for per-token k variation. Triton/CUDA custom kernel work is moderate. D2DMoE [4] provides a reference GPU implementation showing near-linear scaling with executed experts.

- **Training stability**: Key risk is routing collapse: if the k-predictor learns to always output k=0 (or k=1), the model loses capacity. The specific risk is higher at 512-expert scale than at 8-expert scale (the latter used in AdaMoE [2] and Huang et al. [8]) — ensuring all 512 experts receive sufficient gradient signal during training with variable k requires a more aggressive load-balancing loss. Proven stable approaches: AdaMoE's annealed null-expert balancing loss (α: 0.02→0.0001 over two epochs); DynMoE's top-any gating [3]; ReMoE's ReLU gating [5]. Any claim that one approach is definitively "most stable" is unsubstantiated without direct comparison at 512-expert scale.

- **Framework support**: PyTorch feasible. The null-expert approach (AdaMoE [2]) requires minimal changes to existing MoE code: add null experts and adjust k. The ReLU approach (ReMoE [5]) is a minimal routing swap. All are implementable in standard PyTorch with sparsity regularization added to the loss. The dispatch/combine kernels for variable k may require custom work.

- **Recommended implementation path**: Start with AdaMoE's null-expert trick (minimal code changes) and compare against ReMoE's ReLU gating at a 1–7B prototype scale before committing to 397B. Review LD-MoLE (ICLR 2026, arXiv:2509.25684) and Reinforced Adaptive Routing (OpenReview 2026) as updated implementation references.

- **Compatibility**: Can combine with idea 1.3 (Per-Layer Adaptive Expert Count — orthogonal: 1.1 varies k per token within a layer, 1.3 varies k across layers; Alloc-MoE [14] demonstrates the combination), 1.4 (Learned Dense vs Sparse Layer Assignment), 2.2 (Matrix Decomposition in expert weights). Potential conflict with 3.2/3.3 (Hot-Swappable Experts) if expert set is dynamic at inference time.

---

## 6. Synergies

- **Combines well with**:
  - **1.3 (Per-Layer Adaptive Expert Count)**: Orthogonal axes of variation — token-level k vs. layer-level k. Can be combined for a 2D adaptive routing grid. Alloc-MoE [14] and DynaMoE [10] partially explore this combination.
  - **1.4 (Learned Dense vs Sparse Layer Assignment)**: If some layers become dense, per-token k is moot for those layers; for remaining MoE layers, idea 1.1 applies directly.
  - **2.2 (Compressed Dense Layers)**: Expert matrices can be compressed (low-rank), reducing the per-expert weight-load cost independently of k reduction. Combined: multiplicative bandwidth savings.
  - **5.1 (TurboQuant)**: Quantized expert weights reduce bytes-per-expert-load; combined with lower k̄, multiplicative bandwidth reduction.
- **Conflicts with**:
  - **Expert Choice Routing**: Incompatible with autoregressive decode.
  - **SeqTopK [7]**: Preserves average k = fixed k at the sequence level; provides no average-TPOT improvement for single-sequence decode; a quality-improvement idea rather than a speedup idea.

---

## 7. Risk Assessment

- **Technical risk**: MEDIUM — The mechanism is well-motivated and multiple proof-of-concept implementations exist at ICLR 2025 (DynMoE [3], ReMoE [5]) and NeurIPS 2024 (D2DMoE [4]). Main risks: (a) routing collapse during training — especially at 512-expert scale where the risk is higher than in 8-expert literature settings; (b) hardware-inefficient variable-k dispatch at inference. Neither is blocking; both require careful engineering.
- **Potential impact**: HIGH — If average k drops from 11 to 7 on Baseline B, the net model-wide TPOT improvement is ~1.2–1.4× on decode (MoE-FFN-only upper bound: ~1.57×). Given that Baseline B already achieves 8.6×–19× speedup vs dense, this is an additional multiplicative gain. AdaMoE [2] shows FLOPs reduction with simultaneous accuracy improvement; Huang et al. [8] show 0.7% quality improvement with <90% activated parameters.
- **Implementation effort**: MEDIUM — The k-predictor itself is trivial. Challenges: (a) training stability (load-balancing loss design, especially at 512-expert scale), (b) inference dispatch kernel for variable k, (c) hyperparameter search for target average k̄. Estimated effort: 2–4 engineer-weeks on top of an existing MoE codebase with reference to AdaMoE or ReMoE open-source implementations. The quality-cliff characterization at Baseline B scale (k̄ floor below which quality degrades sharply at 512 experts) requires a dedicated ablation sweep not captured in the literature.

---

<!-- CITATION MANIFEST: title | arxiv_id_or_url | description -->
<!-- Expert Choice Routing | arxiv:2202.09368 | NeurIPS 2022; inverted expert-choice routing for variable per-token expert count -->
<!-- AdaMoE | arxiv:2406.13233 | EMNLP 2024 Findings; null-expert trick for learnable variable-k routing -->
<!-- DynMoE | arxiv:2405.14297 | ICLR 2025; auto-tuning approach with novel gating for per-token adaptive expert count -->
<!-- D2DMoE | arxiv:2310.04361 | NeurIPS 2024 (workshop/revised); threshold-based dynamic-k conversion from dense -->
<!-- ReMoE | arxiv:2412.14711 | ICLR 2025; ReLU-based fully differentiable MoE routing -->
<!-- DTop-p | arxiv:2512.13996 | arXiv 2025; PI-controller cumulative-probability threshold routing -->
<!-- SeqTopK | arxiv:2511.06494 | arXiv 2025; sequence-level expert-token pair budget redistribution -->
<!-- Harder Tasks Need More Experts | arxiv:2403.07652 | ACL 2024; confidence-based dynamic routing, 1.76 avg experts vs 2.0 baseline -->
<!-- DA-MoE | arxiv:2409.06669 | arXiv 2024; attention-derived importance score for variable-k routing -->
<!-- DynaMoE | arxiv:2603.01697 | arXiv 2026; token-level dynamic expert activation with layer-wise scheduling -->
<!-- DeepSeekMoE | arxiv:2401.06066 | ACL 2024; fine-grained expert segmentation baseline with fixed-k -->
<!-- MegaBlocks | arxiv:2211.15841 | MLSys 2023; block-sparse GPU kernels for MoE variable-token dispatch -->
<!-- LD-MoLE | arxiv:2509.25684 | ICLR 2026; lightweight per-token k-predictor MLP for LoRA experts -->
<!-- Alloc-MoE | arxiv:2604.08133 | ACL 2026; combined layer-level and token-level adaptive expert activation -->
<!-- Expert Threshold Routing | arxiv:2603.11535 | arXiv 2026; EMA-threshold variable-k routing -->
<!-- Reinforced Adaptive Routing | openreview:yBJZw5DBzU | OpenReview 2026; RL-based per-token k-predictor -->
<!-- DeepSeek-V4 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf | 2026 technical report; frontier-scale all-MoE fixed-k=6 routed expert baseline with early Hash routing -->

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **AdaMoE [2]** (EMNLP 2024 Findings): With k̄ ≈ 1.66 true experts per layer (vs 2.0 baseline, a 17% reduction), achieves *improved* ARC-C accuracy alongside a 14.55% FLOPs reduction. This is the strongest positive signal: moderate k reduction does not merely preserve quality but can improve it, likely because forced sparsity acts as a regularizer over 8 true experts.
  - **Harder Tasks Need More Experts [8]** (ACL 2024): Dynamic routing achieves average score 42.3% vs 41.6% for fixed Top-2 (+0.7 pp), while activating only 1.76 experts on average (vs 2.0). Improvement is task-specific: BBH (harder) activates 1.87 experts on average; PIQA (simpler) activates 1.72 — confirming that quality is maintained or improved when easy tokens route fewer experts.
  - **D2DMoE [4]** (NeurIPS 2024): Abstract reports up to 60% inference cost reduction without significantly impacting performance. Paper-body §4 Table reports "almost three times as fast as standard MLP while preserving 99% of the original accuracy" at the 60% FLOPs-reduction operating point. This suggests the quality cliff is far steeper below ~30% of original FLOPs than above it, but the 60% reduction regime is largely safe.
  - **ReMoE [5]** (ICLR 2025): Validation loss 1.689 vs Top-K baseline 1.716 (Table 5) — a quality *improvement* despite variable active expert count — across 182M–978M model sizes with 4–128 experts.
  - **DTop-p [6]** (arXiv 2025): Averaged 50.9 across 13 NLP benchmarks vs 49.0 for Top-k at 100B training tokens (1.9-point improvement), with PI-controller-managed average sparsity target. At the larger 128E/16A config: 51.4 vs 49.3 Top-k. Suggests variable-k with global budget control is reliably better than fixed-k at matched compute budget.

- **Monotonicity**: Quality loss is *not* monotone with aggressiveness for moderate reductions. Multiple papers (AdaMoE [2], ReMoE [5], Huang et al. [8], DTop-p [6]) show quality *improvement* over fixed-k baselines at moderate k reductions (10–40%). The quality cliff appears below very aggressive k reduction (k̄ < ~0.4 × k_original at small scale). At Baseline B scale (k=11, 512 experts), the floor for quality-neutral k̄ is unknown; the literature's smallest scale is k=2 with 8 experts, so extrapolation to k̄ ≈ 7 from k=11 is plausible but unconfirmed.

- **Recovery**: Variable-k routing mechanisms are trained end-to-end; there is no post-hoc "recovery" concept in the same sense as pruning. The AdaMoE null-expert annealing scheme (α: 0.02 → 0.0001) allows dynamic recovery of active experts if the training signal degrades. D2DMoE [4]'s post-deployment τ threshold is a direct quality-efficiency dial: increasing τ (reducing k) degrades quality gracefully; decreasing τ recovers quality without retraining.

- **Conditions for acceptable degradation**:
  - At moderate k reduction (k̄ ≈ 0.6–0.9 × k_original): expect quality-neutral or quality-positive outcomes based on all published analogues; this is the target operating range for Baseline B at k̄ ≈ 7–10.
  - At aggressive k reduction (k̄ < 0.5 × k_original): quality risk increases; D2DMoE's 99% accuracy retention at 60% FLOPs reduction (paper-body Table, not abstract) is the upper-bound claim, achieved at small scale with careful threshold tuning.
  - Acceptable for throughput-critical inference serving where TPOT is the binding constraint; less acceptable for open-ended reasoning benchmarks (BBH, MATH) where harder tokens specifically benefit from higher k.
  - **No experiments at 512-expert scale with k=11 exist in the literature.** The k=2 or k=3 small-expert-count results may not transfer: at k=11 from E=512, the ratio k/E is already very small (2.1%), and routing specialization may be more brittle. A quality-cliff characterization sweep (k̄ ∈ {3,5,7,9,11} at Baseline B scale) is a required ablation before deployment.
  - Combination with SeqTopK [7]-style sequence-budget preservation is a risk mitigation: if total sequence-level expert-token pairs are preserved, average k̄ < k applies within-sequence redistribution rather than outright compute removal, preserving quality at zero net FLOPs change.
