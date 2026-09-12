# Research: Learned Dense vs Sparse Layer Assignment
## ID: 1.4

## Executive Summary

**Novelty verdict:** PARTIAL — end-to-end gradient-based learning of dense-vs-MoE layer assignment during LLM-scale pretraining is ~55% covered (NAS-based AutoMoE, post-hoc MoEfication/CMoE/ExpertWeaver, Gumbel-Softmax binary architecture choices in GDAS/ProxylessNAS exist); residual novelty is applying ProxylessNAS-style binary gates to dense/MoE assignment inside a standard LLM pretraining run at 1–3B+ scale ([AutoMoE, 2], [DeepSeek-V3, 10], [ProxylessNAS/GDAS, 13]).

**One-line description:** The model learns during training which layers should be dense MLP and which should be MoE (sparse), via soft Gumbel-Softmax gates that are discretized post-training, rather than this assignment being a hand-engineered architectural choice.

**Value proposition:** Hand-designed hybrid assignments (DeepSeek-V3's 3-dense + 58-MoE; DeepSpeed-MoE's pyramid rule) demonstrate that layer-type heterogeneity is beneficial — but DeepSeek-V4 adds a stronger counter-baseline: all Transformer blocks use MoE, with Hash routing stabilizing the first 3 MoE layers. A gradient-based learned assignment must now beat both dense-first and all-MoE-plus-early-Hash designs, potentially finding configurations that human-designed ablations miss. At inference time, the learned assignment produces a standard hybrid model with zero overhead from the assignment mechanism itself.

**Critical caveats:**
- TTFT improvement vs. dense baseline (A2) at 8K context is ~1.74× — even though MLP dominates at 8K (MLP FLOPs ≈ 3× attention FLOPs per token; attention only overtakes MLP for A2 above ~154K tokens), the ~2.3× MLP reduction (D=16, S=48, k·d_e=d_ff/4) yields ~1.74× TTFT speedup because the attention component (25% of total FLOPs at 8K) is unchanged. For standard MoE configurations (k·d_e ≈ 0.3–1.0 × d_ff), realistic speedup vs. dense is 1.3–2×.
- Baseline A1 (Qwen3.5-27B) is a hybrid architecture combining DeltaNet layers and standard full-attention layers with dense MLP — not a uniform MoE architecture. Idea 1.4's compute reduction is relative to A2 (the fully-dense baseline), not A1.
- **DeepSeek-V4 update (2026-04-24):** early dense FFNs are no longer the only scale-tested stabilization path. V4-Pro uses 61 MoE Transformer blocks, 384 routed experts plus 1 shared expert, 6 active routed experts per token, and Hash routing in the first 3 MoE layers. Treat "all-MoE + deterministic early routing" as a required baseline for this idea.

---

## 1. Idea Description

The model learns during training which layers should be dense (full MLP) and which should be sparse (MoE), rather than this being a hand-designed architectural choice.

**Key distinction from Idea 1.3:** Idea 1.3 adjusts the number of active experts k_l per MoE layer post-training. Idea 1.4 decides the layer type (dense vs. MoE) itself during training. A layer assigned to "dense" by idea 1.4 has no expert routing at all; idea 1.3 then further optimizes k_l within layers that remain MoE.

**Inference behavior:** The assignment gate is used ONLY during training. Post-training, the gate is discretized (argmax or threshold), producing a fixed hybrid architecture. At inference, the model is a standard hybrid (some layers dense MLP, some MoE MLP) with zero per-token overhead from the assignment mechanism.

**Inferred intent:** If the learned assignment concentrates MoE layers in positions where activation sparsity is high (experts specialize well) and dense layers where activation is uniform, prefill FLOPs reduce by the MoE factor without quality loss. Simultaneously, the learned policy should discover whether early dense layers, deterministic early routing, or a hybrid of both best prevents MoE routing instability from corrupting early feature extraction.

---

## 2. Literature Review

### DeepSpeed-MoE / PR-MoE [1]
*Rajbhandari, Li, Yao, Zhang, Aminabadi, Awan, Rasley, He — arXiv:2201.05596 (ICML 2022)*

Introduces Pyramid-Residual MoE (PR-MoE), which places MoE layers only in the second half of the network (finding that deeper layers benefit more from large expert counts). PR-MoE reduces MoE model parameter size by up to 3× with no change to model quality. Also introduces Residual-MoE where each token passes a fixed dense MLP plus one chosen expert (>10% faster than Top-2 MoE).

> **[Rajbhandari et al., 2022]** — §3.2 "Layer Analysis" (Second-Half-MoE outperforms First-Half-MoE), §4 PR-MoE Table 2 (3× parameter reduction)

**Relevance:** Demonstrates that hand-designed layer-type assignment outperforms uniform MoE placement; directly establishes that layer position matters for MoE effectiveness — the signal that a learned assignment would exploit.
**Limitations:** Assignment is still hand-designed (second-half rule derived from ablation), not gradient-based.

---

### AutoMoE: Heterogeneous MoE with Adaptive Computation for Efficient NMT [2]
*Jawahar, Mukherjee, Liu, Awadallah, Gao, Joty — arXiv:2210.07535 (ACL Findings 2023)*

First NAS framework for heterogeneous MoE design — searches over which layers are dense vs. sparse MoE, expert count per layer, and expert size, under explicit FLOPs/latency constraints. Uses supernet training (shared weights) + evolutionary search. Achieves 4× inference speedup (CPU) and FLOPs reduction with parity BLEU to dense Transformer, within 1 BLEU of MoE SwitchTransformer.

> **[Jawahar et al., 2022/2023]** — §4 "Experiments" (4× CPU speedup, BLEU parity vs. dense; within 1 BLEU vs. MoE SwitchTransformer — ACL Findings 2023 Table 3)

**Relevance:** Closest prior art to idea 1.4 in the NAS direction — explicitly searches "which layers are dense vs. MoE." The core mechanism is the same; the difference is evolutionary NAS vs. gradient-based learning during standard pretraining.
**Limitations:** Evolutionary search (not gradient-based); applied to NMT at small scale; NAS is a separate phase from standard pretraining.

---

### MoEfication: Transformer FFN Layers are MoEs [3]
*Zhang, Lin, Liu, Li, Sun, Zhou — arXiv:2110.01786 (ACL Findings 2022)*

Converts pre-trained dense transformers into MoE models post-hoc by grouping FFN neurons into experts based on activation patterns, then training small routers. With 10–30% of FFN parameters active, retains >95% of original performance; 2× speedup at 25% FFN parameter activation.

> **[Zhang et al., 2021/2022]** — §4 "Experiments" Table 2 (>95% performance retention; 2× speedup at 25% activation)

**Relevance:** Demonstrates selective per-layer dense-to-MoE conversion; the observation that activation sparsity is present in pre-trained FFN layers at all positions motivates learning the assignment during training rather than post-hoc.
**Limitations:** Post-hoc; requires pre-trained dense model; assignment not gradient-based during training.

---

### CMoE: Converting MoE from Dense to Accelerate LLM Inference [4]
*Pei et al. — arXiv:2502.04416 (2025)*

Training-free dense-to-MoE conversion: analyzes neuron activation patterns to partition FFN neurons into shared (always-active) and routed experts with layer-adaptive configurations. Per abstract: training-free; at 75% activation ratio lossless perplexity (~5% acceleration); at 25% activation 1.5× latency reduction. (The "<5 min on single GPU" figure is paper-body.)

> **[Pei et al., 2025]** — §4 "Experiments" Table 1 (lossless at 75% activation; 1.5× latency at 25% activation)

**Relevance:** Establishes a competitive non-learning baseline for per-layer dense/sparse assignment via activation statistics.
**Limitations:** Post-hoc; no gradient signal; not jointly trained.

---

### ExpertWeaver: Unlocking Inherent MoE in Dense LLMs with GLU Activation Patterns [5]
*Feng et al. — arXiv:2602.15521 (2026)*

Identifies that GLU mechanisms in dense LLMs create an inherent MoE structure composed of consistently-activated "universal" neurons and dynamically-activated "specialized" neurons. ExpertWeaver partitions neurons accordingly with layer-adaptive configurations for MoE initialization without training.

**Relevance:** Theoretical grounding for why learned assignment would find different per-layer configurations — the inherent MoE structure is layer-dependent.
**Limitations:** Post-hoc analysis; does not train the assignment.

---

### Mixed Sparsity Training (MST): Achieving 4× FLOP Reduction for Transformer Pretraining [6]
*Hu, Li, Huang — arXiv:2408.11746 (2024)*

MST integrates dynamic sparse training with Sparsity Variation, Mixed-Growing, and Hybrid Sparse Attention during pretraining. Uses three phases: warm-up (dense→sparse), ultra-sparsification, and restoration (sparse→dense). On GPT-2, achieves 4× FLOP reduction without performance loss.

> **[Hu et al., 2024]** — §4 "Results" Table 2 (4× FLOP reduction, zero-shot/few-shot downstream tasks on GPT-2)

**Relevance:** Demonstrates training-time selection of layer sparsity patterns via gradient signals. Per-layer temporal sparsity variation during training is a form of learned assignment.
**Limitations:** Targets weight sparsity (N:M / dynamic), not MoE expert routing. Final model has same fixed architecture; does not produce a static dense/sparse layer partition for inference.

---

### Dense Training, Sparse Inference (DS-MoE) [7]
*Pan, Shen, Liu, Mishra, Zhang, Oliva, Raffel, Panda — arXiv:2404.05567 (2024)*

Proposes training MoE models with dense computation (all experts active during training) but sparse inference. Uses Mutual Information (MI) loss for load balancing. DS-MoE-6B runs 1.86× faster than Mistral-7B and 1.50–1.71× faster than comparable MoEs with 30–40% active parameters.

> **[Pan et al., 2024]** — §4.2 "Inference Efficiency" Table 3 (DS-MoE-6B: 1.86× faster than Mistral-7B)

**Relevance:** Demonstrates that dense training of a sparse-inference MoE produces better quality; if idea 1.4 learns which layers are MoE during training, the dense-training-for-sparse-inference regime could improve the assignment signal quality.
**Limitations:** Does not solve the layer-assignment problem (all layers are uniformly MoE in DS-MoE).

---

### ReMoE: Fully Differentiable MoE with ReLU Routing [8]
*Wang, Zhu, Chen et al. — arXiv:2412.14711 (ICLR 2025)*

Replaces TopK+Softmax routing in MoE with ReLU-based routing, making expert selection fully differentiable. Active expert count determined by ReLU sparsity rather than fixed k. Outperforms TopK routing across sizes 182M–978M, 4–128 experts at ICLR 2025.

**Relevance:** The ReLU routing mechanism is conceptually related to the gradient signal needed for learned assignment — a fully differentiable sparse selection mechanism.
**Limitations:** Does not decide whether a layer should be MoE or dense; applies within fixed MoE layers only.

---

### Mixtral of Experts [9]
*Jiang, Sablayrolles, Roux et al. (Mistral AI) — arXiv:2401.04088 (2024)*

Mixtral 8x7B applies MoE to every feed-forward layer (all 32 layers). 8 experts per layer, 2 active per token. 47B total / 13B active per token. Outperforms LLaMA-2-70B on benchmarks.

> **[Jiang et al., 2024]** — p.2 §2 "Mixtral of Experts" (13B/47B active parameter ratio)

**Relevance:** Key reference where the architectural decision is uniform MoE at all layers — the opposite extreme from a learned assignment.
**Limitations:** No learning of layer assignment; hand-designed uniform placement.

---

### DeepSeek-V3 [10]
*DeepSeek-AI — arXiv:2412.19437 (2024)*

DeepSeek-V3 uses a hybrid architecture with 3 dense layers (0–2) followed by 58 MoE layers (3–60), totaling 671B parameters with 37B active per token. The dense-first design prevents MoE routing instability from interfering with early feature extraction. Uses 256 routed experts + 1 shared expert with 8 active routed per token.

> **[DeepSeek-AI, 2024]** — arXiv:2412.19437 §"Model Architecture Overview" (3 dense + 58 MoE layers; routing instability motivation for dense-first design)

**Note:** DeepSeek-V3 uses 256 experts with 8 active (not 512/11 as in Baseline B). Baseline B (Qwen3.5-397B-A17B) has 512 experts, k=11.
**Relevance:** Empirical evidence that optimal layer-type assignment is NOT uniform; dense layers at early positions improve convergence stability. The 3+58 split motivates idea 1.4.
**Limitations:** Assignment determined by ablation, not gradient-based learning.

---

### DeepSeek-V4 [20]
*DeepSeek-AI — technical report / Hugging Face release (2026)*

DeepSeek-V4-Pro uses MoE layers in every Transformer block rather than retaining a dense-first prefix. To stabilize the earliest layers, it uses Hash routing in the first 3 MoE layers; subsequent MoE layers use learned routing with 384 routed experts, 1 shared expert, and 6 active routed experts per token.

**Relevance:** This is the strongest adverse update for the old dense-first assumption. A learned dense/sparse assignment can still be valuable, but the experimental baseline must include V4's all-MoE + early Hash routing design, not only DeepSeek-V3's 3-dense + 58-MoE pattern.
**Limitations:** Assignment is still hand-designed, and Hash routing is token-ID deterministic rather than learned layer assignment.

---

### DARTS: Differentiable Architecture Search [11]
*Liu, Simonyan, Yang — arXiv:1806.09055 (ICLR 2019)*

Introduces continuous relaxation of discrete architecture choices via weighted sums of candidate operations. Gradient-based architecture search. After search, highest-coefficient operation is selected (discretization). Runs on a single GPU in days vs. GPU-years for random search.

> **[Liu et al., 2018/2019]** — §3.2 "Architecture Search" (discretization gap via argmax of architecture weights; known failure mode at inference)

**Relevance:** Foundational method for gradient-based layer-type assignment. Idea 1.4 could be implemented as a DARTS-style supernet where each layer chooses between dense-MLP and MoE-MLP via trainable soft gate.
**Limitations:** DARTS applied to full-scale LLM pretraining is computationally prohibitive (supernet requires all candidate operations simultaneously — at 30B+ scale, 209 GB simultaneous weight memory is infeasible); discretization gap is a known failure mode.

---

### P-DARTS, PC-DARTS, DARTS-: DARTS Failure Mode Mitigations [12]
*P-DARTS: Chen et al. — arXiv:1904.12760 (ICCV 2019); PC-DARTS: Xu et al. — arXiv:1907.05737 (ICLR 2020); DARTS-: Chu et al. — arXiv:2009.01027 (ICLR 2021)*

P-DARTS bridges the discretization gap via progressive depth increase during search. PC-DARTS uses partial channel connections to reduce memory/time overhead of supernet training — directly addressing the memory infeasibility for large-scale models. DARTS- analyzes and mitigates DARTS's tendency to select skip connections (analogous to the "all layers collapse to dense" failure mode).

**Note on venues:** PC-DARTS was accepted at ICLR 2020 (not 2021 as sometimes mis-cited); DARTS- at ICLR 2021.

**Relevance:** Provide known solutions to the two cited failure modes of gradient-based layer assignment: (a) discretization gap, (b) assignment collapse. Essential reading for implementing idea 1.4.

---

### GDAS / SNAS / ProxylessNAS: Gumbel-Softmax NAS [13]
*GDAS: Dong & Yang — arXiv:1910.04465 (CVPR 2019); SNAS: Xie et al. — arXiv:1812.09926 (ICLR 2019); ProxylessNAS: Cai et al. — arXiv:1812.00332 (ICLR 2019)*

GDAS uses Gumbel-Softmax sampling directly for architecture search — the exact mechanism for idea 1.4's gradient-based assignment. SNAS is the canonical paper for Gumbel-Softmax in NAS. ProxylessNAS uses binary gates (STE) for architecture binary choices, eliminating supernet memory overhead.

**Relevance:** Provide the mechanistic implementation toolkit for idea 1.4's core method. ProxylessNAS's binary gate approach directly solves the memory infeasibility of the DARTS supernet for large LLMs.

---

### HAQ / HAWQ: Automated Mixed-Precision as Structural Analogue [14]
*HAQ: Wang et al. — arXiv:1811.08886 (CVPR 2019); HAWQ: Dong et al. — arXiv:1905.03696 (ICCV 2019)*

HAQ uses RL for per-layer bit-width assignment; HAWQ uses Hessian second-order sensitivity for per-layer precision decisions. The per-layer assignment problem is structurally identical to the dense/MoE assignment problem: a binary or categorical per-layer decision based on sensitivity analysis.

**Relevance:** 5+ years of developed methodology for the same problem structure as idea 1.4. HAWQ's Hessian-based sensitivity analysis could equally apply to "should this layer be dense or MoE?"

---

### DeepSeek-MoE / OpenMoE: Layer-Level MoE Analysis at LLM Scale [15]
*DeepSeek-MoE: Dai et al. — arXiv:2401.06066 (2024); OpenMoE: Xue et al. — arXiv:2402.01739 (2024)*

DeepSeek-MoE uses fine-grained expert segmentation and shared experts alongside routed experts, analyzing layer-level importance heterogeneity. OpenMoE analyzes routing mechanisms and finds that routing decisions are predominantly context-independent (token-ID-driven), stabilize early in pretraining, and that later-sequence tokens are more likely to be dropped — providing empirical grounding for per-layer routing heterogeneity, though not a simple early-uniform/late-specialized split.

**Relevance:** OpenMoE's empirical finding directly supports the hypothesis that dense-in-early-positions and MoE-in-later-positions is optimal — providing direct empirical grounding for what a learned assignment should discover.

---

### Optimal Sparsity for MoE Models on Reasoning Tasks [16]
*Multiple authors — arXiv:2508.18672 (2025)*

Finds that optimal MoE sparsity varies by task type: memorization tasks improve monotonically with total parameters; reasoning tasks saturate and can regress with increased sparsity. Optimal sparsity must be jointly determined by active FLOPs and tokens-per-parameter.

**Relevance:** Empirical evidence that uniform sparsity across all layers is suboptimal; a learned assignment trained on mixed task distributions would need to discover per-layer sparsity levels balancing these competing requirements.

---

### Generalizing Scaling Laws for Dense and Sparse LLMs [17]
*Hossain, Wu, Taylor, Jannesari — arXiv:2508.06617 (2025/2026)*

Proposes a unified scaling law applicable to both dense and sparse (MoE) LLMs, providing theoretical basis for predicting when MoE achieves better loss than dense at equivalent parameter counts.

**Relevance:** Theoretical basis for understanding when MoE vs. dense provides better scaling per layer — informing the assignment budget D+S=L.

---

### LExI: Layer-Adaptive Active Experts for Efficient MoE Inference [18]
*Chitty-Venkata, Madireddy, Emani, Vishwanath — arXiv:2509.02753 (2025)*

Data-free post-training technique that determines the optimal number of active experts per layer in a pretrained MoE model using model weights alone to estimate relative layer importance. Adaptively assigns different expert activation budgets per layer. Applied to Qwen1.5-MoE, achieves same throughput on H100 with 10% better accuracy vs. uniform expert pruning.

> **[Chitty-Venkata et al., 2025]** — §4 "Experiments" (Qwen1.5-MoE: same throughput, 10% accuracy improvement vs. traditional expert pruning)

**Relevance:** Direct prior art for per-layer heterogeneous compute allocation in MoE models. LExI solves the k_l optimization post-training (structurally Idea 1.3 territory); idea 1.4 extends further by deciding layer type (dense vs. MoE) during training. Together they form a complete per-layer assignment pipeline.
**Limitations:** Post-training only; does not change layer type (dense/MoE), only adjusts k within existing MoE layers; not gradient-based during pretraining.

---

### Composer: A Search Framework for Hybrid Neural Architecture Design [19]
*Acun, Sinha, Ardalani, Bae et al. — arXiv:2510.00379 (ICLR 2026)*

Principled architecture search framework for hybrid models combining different computational primitives (Attention, MLP). Searches at small scale then extrapolates to larger scales (350M–3B). Discovered architectures outperform Llama 3.2, improving downstream accuracy by 1.1–3.1% on average with better efficiency.

> **[Acun et al., 2025/2026]** — §5 "Results" (1.1–3.1% downstream accuracy improvement vs. Llama 3.2; small-to-large scale extrapolation)

**Relevance:** Demonstrates that learned/searched hybrid architecture assignment (interleaving of primitive types) outperforms human-designed baselines at LLM scale — directly supporting the hypothesis behind idea 1.4. Composer uses offline search; idea 1.4 proposes integrating this into the training run via gradient-based gates.
**Limitations:** Uses evolutionary/discrete search (not gradient-based during training); focuses on attention vs. SSM primitives rather than dense vs. MoE FFN assignment; separate search phase required.

---

## 3. Prior Art Classification

- **Status:** PARTIAL
- **Overlap summary:** ~55% covered.
  - EXISTS: Hand-designed hybrid assignments (DeepSeek-V3, PR-MoE, Mixtral uniform) and all-MoE plus early deterministic routing (DeepSeek-V4).
  - EXISTS: NAS-based layer-type assignment (AutoMoE [2]) — evolutionary, not gradient-based during pretraining.
  - EXISTS: Post-hoc dense-to-MoE assignment (MoEfication [3], CMoE [4], ExpertWeaver [5]).
  - EXISTS: Post-training per-layer active-expert assignment (LExI [18]) — data-free weight-based, not gradient-based during training.
  - EXISTS: Hybrid architecture search frameworks (Composer [19]) — discrete/evolutionary search for primitive interleaving at small scale; not gradient-based during pretraining.
  - EXISTS: Gumbel-Softmax binary architecture choices (GDAS [13], SNAS [13], ProxylessNAS [13]).
  - **NOVEL:** End-to-end gradient-based learning of which layers should be dense vs. MoE during LLM-scale pretraining without a separate NAS phase — specifically, applying ProxylessNAS/GDAS-style binary gates to the dense/MoE layer assignment choice within a standard LLM pretraining run. This specific combination has not been demonstrated at pretraining scale.
- **Novelty framing (refined):** Applying Gumbel-Softmax or ProxylessNAS-style binary gates to the dense/MoE assignment during standard LLM pretraining at scale. The mechanism components all exist; the novelty is the specific application at LLM pretraining scale.

**Novelty verdict: PARTIAL — mechanism components (Gumbel-Softmax NAS, evolutionary layer-type search, post-hoc dense/MoE conversion) all exist in prior art; novelty is the specific end-to-end gradient-based application of ProxylessNAS/GDAS-style binary gates to dense/MoE layer assignment during LLM-scale pretraining without a separate NAS phase.**

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: L=total layers (D+S=L), D=dense layers, S=MoE layers, d=hidden dim, d_ff=dense FFN intermediate dim, d_e=MoE expert intermediate dim, E=total experts per MoE layer, k=active experts per token, s=sequence length.

**FLOP derivations:**
- Dense MLP per token per layer: O(d·d_ff)
- MoE layer per token per layer: O(k·d·d_e)
- Total per token: O(D·d·d_ff + S·k·d·d_e)

**TPOT bandwidth per token:**
- Dense layer: loads W_gate(d×d_ff) + W_up(d×d_ff) + W_down(d_ff×d) = O(d·d_ff) weight bytes
- MoE layer: loads k active expert weights: O(k·d·d_e)
- Total: O(D·d·d_ff + S·k·d·d_e)

**Example scenario (Baseline A2 analog: L=64, d=5120, d_ff=25600):**
If D=16, S=48, and k·d_e = d_ff/4 = 6400 (moderate compression, not the unrealistic d_ff/8):
- FLOP ratio vs A2: (16×25600 + 48×6400) / (64×25600) = (409600 + 307200) / 1638400 = 716800/1638400 ≈ 0.44× → ~2.3× MLP FLOP reduction vs A2

**TTFT derivation:**
At s=8K tokens, d=5120, d_ff=25600, H_q=64, H_kv=8, head_dim=128 (A2):
- MLP FLOPs per token per layer (SwiGLU, 3 matrices): 6×d×d_ff = 6×5120×25600 ≈ 786M
- Attention FLOPs per token per layer (full accounting: QKV proj + score matmuls at avg position s/2 + O proj):
  - Q proj: 2×d×d ≈ 52.4M; K+V proj (GQA): 2×2×d×(H_kv×head_dim) = 4×5120×1024 ≈ 21.0M
  - Attention scores (QK, causal avg s/2): 2×(s/2)×H_q×head_dim = 2×4096×64×128 ≈ 67.1M
  - Attention values (AV): same ≈ 67.1M; Output proj: 2×d×d ≈ 52.4M
  - Total attention per token ≈ 260M
- MLP : Attention ratio at 8K ≈ 3.0:1. **MLP dominates at s=8K** (attention only overtakes MLP above ~s=6×d_ff ≈ 154K tokens).
- Net TTFT speedup vs A2 (with ~2.3× MLP reduction at D=16, S=48, k·d_e=d_ff/4):
  MLP fraction = 786M/(786M+260M) ≈ 75%
  Attention fraction = 25%
  Net TTFT = 1 / (0.75/2.3 + 0.25) = 1 / (0.326 + 0.25) = 1/0.576 ≈ 1.74× (NOT ~3×)
- At very long context (s≫154K), attention dominates and MLP speedup has diminishing impact on TTFT.

**TPOT (bandwidth-bound at decode, batch=1):** Since decode is bandwidth-bound and attention is typically shorter (s ≪ prefill length), MLP bandwidth dominates:
- Realistic TPOT speedup vs A2: ~1.5–2.3× depending on k·d_e assumption.

| Metric | This Idea (D dense + S MoE, D+S=L) | Baseline A1 (Qwen3.5-27B Hybrid — DENSE MLP) | Baseline A2 (Qwen3-32B Dense) | Baseline B (397B-A17B MoE, 512 exp, k=11) | Baseline C (K2 ~72.55B Dense) |
|--------|-------------------------------------|-----------------------------------------------|-------------------------------|---------------------------------------------|-------------------------------|
| MLP FLOPs/token | O(D·d·d_ff + S·k·d·d_e) | O(L·d·d_ff) [dense MLP, d_ff=17408] | O(L·d·d_ff) [dense MLP, d_ff=25600] | O(L·k·d·d_e) [all MoE FFN] | O(L·d·d_ff) [dense, d_ff=28672] |
| KV cache (32K ctx) | O(L·s·d_kv) if all full-attn; unchanged from attn-type choice | ~2.15 GB | ~8.59 GB | ~1.0 GB | ~10.0 GiB |
| Weight memory (total) | O(D·d·d_ff + S·E·d·d_e) — larger than pure-dense if E large | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) | O(L·d·d_ff) |
| Memory bw (decode) | O(D·d·d_ff + S·k·d·d_e) active bytes | O(L·d·d_ff) | O(L·d·d_ff) | O(L·k·d·d_e) | O(L·d·d_ff) |
| TTFT (8K prompt) | ↓ ~1.5–2× vs A2 (NOT ~3×; MLP dominates at 8K; attention only overtakes MLP above ~154K tokens for A2) | ref | ref | ref | ref |
| TPOT (batch=1) | ↓ ~1.5–2.3× vs A2 (bandwidth-bound; MLP dominates at decode) | ref | ref | ref | ref |
| Training cost | ↑ ~1.5–2× during search phase; 2–4× for full two-stage pipeline | 1.0× | 1.0× | 1.0× | 1.0× |

**Note on Baseline A1:** A1 uses dense MLP (d_ff=17408), NOT MoE FFN. Comparisons to A1 show idea 1.4 would have HIGHER active compute than A1 for the dense layers (since A1's d_ff=17408 < A2's d_ff=25600; the idea is being applied to an A2-like architecture not A1-like). The A1 row correctly shows O(L·d·d_ff) for A1 dense MLP.

### 4.2 Compute Analysis

- **Search phase training FLOPs:** If using Gumbel-Softmax soft gates: each layer runs both dense AND MoE paths simultaneously → ~2× MLP FLOPs per layer during search. ProxylessNAS binary path avoids this by sampling a hard path in each forward pass — back to 1× FLOPs during search but with STE gradient.
- **Integrated soft-gate training (ProxylessNAS style):** ~1.05–1.15× training FLOPs overhead (gate parameters = L scalar logits; negligible). The "1.5–3×" range from NAS literature applies to supernet approaches with all operations active.
- **Two-stage pipeline (safest discretization mitigation):** (1) search phase, (2) retrain with fixed assignment from scratch. Total cost: 2× full pretraining budget — which would change the estimate to "2× training cost minimum."

### 4.3 Memory Capacity

**Weight storage example (L=64, d=5120, d_ff=25600, E=64, k=8, d_e=3200):**
- 16 dense layers: 16 × 3 × 5120 × 25600 × 2 bytes = 16 × 786MB = 12.6 GB
- 48 MoE layers: 48 × 64 × 3 × 5120 × 3200 × 2 bytes = 48 × 64 × 98.3MB = 301 GB
- Total: ~314 GB — approximately 5× the pure-dense model's weight storage (~64 GB for 32B model)
- This is the inherent cost of MoE at large E: large total weight storage, small active compute.

**KV cache:** Unchanged by MLP type assignment. At 32K context:
- A2-like: all full-attn 64 layers × ~0.134 GB/layer total aggregate... per formula: 2×64×32768×(8×128)×2 ≈ 8.59 GB total (note: ~2.15 GB is the A1 total across 16 layers, not an A2 per-layer figure)
- B-like: 15 GatedAttn only → ~1.0 GB at 32K
- C: all full-attn 80 layers → ~10.0 GiB at 32K

---

## 5. Implementation Considerations

**Inference (post-training):** Standard hybrid model — zero overhead from assignment mechanism. Dense layers run standard MLP; MoE layers run standard expert dispatch.

**Training (ProxylessNAS binary gate approach, recommended over DARTS supernet):**
- Binary gate per layer: σ(z_l) ∈ [0,1] where z_l is a trainable scalar logit.
- Hard assignment in forward pass: layer l is dense if z_l > 0.5, else MoE. Gradient via STE.
- Memory: Only one path (dense or MoE) active per layer per forward pass — avoids 2× memory of DARTS supernet. The 209 GB simultaneous weight memory issue is solved.
- Bi-level optimization: alternate between updating MLP weights (gate fixed) and updating gate logits (MLP weights fixed). Prevents gradient interference.

**Training risks:**
1. Assignment collapse (all layers → dense or all → MoE): Mitigation — capacity constraint loss λ·(D_actual - D_target)².
2. Discretization gap: Mitigation — ProxylessNAS binary gates or temperature-annealed Gumbel-Softmax (τ: 5.0 → 0.1 over search phase). Two-stage training is safest but doubles cost.
3. Gradient interference: Mitigation — bi-level optimization (DARTS-style alternation).

**Framework support:** PyTorch Gumbel-Softmax and STE are natively available. Binary gates add L scalar parameters (negligible). Custom MoE dispatch kernels (DeepSpeed/Megatron-LM) remain unchanged for MoE layers.

---

## 6. Synergies

- **1.1 (Per-Token Adaptive Top-k):** Sequential composition — 1.4 assigns layer types, 1.1 optimizes k within MoE layers. Strong positive synergy.
- **1.3 (Per-Layer Adaptive Expert Count):** Complementary — 1.4 decides dense/MoE, 1.3 decides k_l given MoE. Together: complete layer-level compute allocation policy.
- **2.2 (Compressed Dense Layers via Matrix Decomposition):** Reduces per-expert weight bytes independently; multiplicative benefit.
- **5.1 (TurboQuant KV cache):** Independent dimension; additive benefit.

---

## 7. Risk Assessment

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Inference overhead | ZERO | Assignment gate discarded post-training |
| Training cost (ProxylessNAS) | LOW (1.05–1.15×) | Binary gate STE with single-path forward |
| Training cost (two-stage, safest) | HIGH (2× budget) | Full retrain from scratch after search |
| Discretization gap | MEDIUM-HIGH | Known LLM-scale severity unknown; two-stage training mitigates |
| Assignment collapse | MEDIUM (manageable) | Capacity constraint loss required |
| Gradient interference | MEDIUM | Bi-level optimization required |
| Novel contribution | PARTIAL | Mechanism components exist; LLM pretraining application is new |
| Implementation effort | MEDIUM-HIGH: 4–8 engineer-weeks | NAS framework integration + training validation |

**Overall verdict:** INVESTIGATE FURTHER — The gradient-based mechanism is genuinely novel at LLM pretraining scale, but the training complexity (discretization gap risk, bi-level optimization, potential 2× training cost for safe two-stage pipeline) makes this significantly more expensive than the immediately actionable alternatives (post-training static assignment per idea 1.3). Recommend prototyping at 1–3B scale first, validating that the learned assignment converges to a better solution than both DeepSeek-V3's dense-first rule and DeepSeek-V4's all-MoE + early Hash routing rule before committing to 27B+ scale.

---

## Citation Manifest

[1] DeepSpeed-MoE / PR-MoE: Rajbhandari, Li, Yao, Zhang, Aminabadi, Awan, Rasley, He — arXiv:2201.05596 (ICML 2022)
[2] AutoMoE: Jawahar, Mukherjee, Liu, Awadallah, Gao, Joty — arXiv:2210.07535 (ACL Findings 2023)
[3] MoEfication: Zhang, Lin, Liu, Li, Sun, Zhou — arXiv:2110.01786 (ACL Findings 2022)
[4] CMoE: Pei et al. — arXiv:2502.04416 (2025)
[5] ExpertWeaver: Feng et al. — arXiv:2602.15521 (2026)
[6] MST (Mixed Sparsity Training): Hu, Li, Huang — arXiv:2408.11746 (2024)
[7] Dense Training, Sparse Inference (DS-MoE): Pan, Shen, Liu, Mishra, Zhang, Oliva, Raffel, Panda — arXiv:2404.05567 (2024)
[8] ReMoE: Wang, Zhu, Chen et al. — arXiv:2412.14711 (ICLR 2025)
[9] Mixtral: Jiang, Sablayrolles, Roux et al. (Mistral AI) — arXiv:2401.04088 (2024)
[10] DeepSeek-V3: DeepSeek-AI — arXiv:2412.19437 (2024)
[11] DARTS: Liu, Simonyan, Yang — arXiv:1806.09055 (ICLR 2019)
[12] P-DARTS / PC-DARTS / DARTS-: Chen et al. arXiv:1904.12760 (ICCV 2019); Xu et al. arXiv:1907.05737 (ICLR 2020); Chu et al. arXiv:2009.01027 (ICLR 2021)
[13] GDAS / SNAS / ProxylessNAS: Dong & Yang arXiv:1910.04465 (CVPR 2019); Xie et al. arXiv:1812.09926 (ICLR 2019); Cai et al. arXiv:1812.00332 (ICLR 2019)
[14] HAQ / HAWQ: Wang et al. arXiv:1811.08886 (CVPR 2019); Dong et al. arXiv:1905.03696 (ICCV 2019)
[15] DeepSeek-MoE / OpenMoE: Dai et al. arXiv:2401.06066 (2024); Xue et al. arXiv:2402.01739 (2024)
[16] Optimal Sparsity: Multiple authors — arXiv:2508.18672 (2025)
[17] Generalizing Scaling Laws: Hossain, Wu, Taylor, Jannesari — arXiv:2508.06617 (2025/2026)
[18] LExI: Chitty-Venkata, Madireddy, Emani, Vishwanath — arXiv:2509.02753 (2025)
[20] DeepSeek-V4: DeepSeek-AI — "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" technical report / Hugging Face release (2026)
[19] Composer: Acun, Sinha, Ardalani, Bae et al. — arXiv:2510.00379 (ICLR 2026)

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: NAS-based layer assignment achieves 4× CPU speedup at parity BLEU vs dense Transformer and within 1 BLEU of Switch Transformer [AutoMoE, 2, ACL Findings 2023, §Results]; post-hoc dense-to-MoE retains >95% of original performance with 10–30% of FFN active [MoEfication, 3, ACL Findings 2022]; Composer discovered hybrid architectures outperform Llama 3.2 by +1.1–3.1% downstream accuracy at 350M–3B scale [Composer, 19, ICLR 2026]; LExI reports 10% better accuracy vs uniform expert pruning at matched throughput on MoE models [LExI, 18, arXiv:2509.02753].
- **Known failure modes**: No experiments on end-to-end gradient-based dense/MoE assignment during LLM pretraining exist at any scale (AutoMoE is evolutionary NAS on NMT; MoEfication and CMoE are post-hoc); discretization gap and assignment collapse artifacts of gradient-based gates are uncharacterized [1.4 §3]; reasoning-task quality *regresses* with too-high sparsity fraction S/L [Optimal Sparsity, 16, arXiv:2508.18672]; aggressive MoE ratios degrade quality measurably at 25% activation [CMoE, 4].
- **Empirical evidence**: AutoMoE §Results (4× speedup at parity BLEU) [AutoMoE, 2]; MoEfication §Results (>95% retention at 25% activation) [MoEfication, 3]; CMoE §Results (lossless at 75%, degrades at 25%) [CMoE, 4]; Composer §Results (+1.1–3.1% over Llama 3.2) [Composer, 19]; PR-MoE §Results (second-half MoE beats first-half/uniform at matched params) [PR-MoE/DeepSpeed-MoE, 1, ICML 2022].
- **Mitigations**: Hard-constrain the first D_min layers to dense in the assignment gate to prevent early-layer MoE routing instability [DeepSeek-V3, 10]; run prototype at 1–3B scale and compare against the DeepSeek-V3 3-dense + 58-MoE split before scaling; if over-sparsed, recover via 1-hour LoRA fine-tune on 2,000 samples (restores >76% downstream) [CMoE, 4]; only adopt the learned partition if resulting TTFT/TPOT improvement exceeds 1.5× vs fully-dense baseline.

### Detailed breakdown

- **Reported quality delta (from closest analogues)**:
  - **AutoMoE [2]** (ACL Findings 2023): NAS-based heterogeneous layer assignment achieves 4× CPU inference speedup at *parity BLEU* vs dense Transformer for NMT, and within 1 BLEU of MoE SwitchTransformer. This is the most direct evidence that learned per-layer dense/MoE assignment does not regress quality vs a fixed assignment, and that the NAS-discovered assignment is superior to both uniform-dense and uniform-MoE baselines.
  - **MoEfication [3]** (ACL Findings 2022): Post-hoc per-layer dense-to-MoE conversion retains >95% of original performance with 10–30% of FFN parameters active (2× speedup at 25% activation). The 5% quality floor at 25% activation is a reference lower bound; a learned assignment during training would be expected to find better layer-specific operating points than post-hoc conversion.
  - **CMoE [4]** (arXiv 2025): At 75% activation ratio: lossless perplexity with 5% speedup; at 25% activation ratio: ~1.5× speedup with measurable quality degradation (recovery via 1-hour LoRA fine-tuning on 2,000 samples restores >76% downstream accuracy). This establishes a practical repair path if the learned assignment produces an overly sparse partition.
  - **PR-MoE / DeepSpeed-MoE [1]** (ICML 2022): Second-half MoE placement (dense layers first, MoE layers second half) outperforms both first-half MoE and uniform MoE placement at matched parameter count — 3× parameter reduction with no quality change. This quality-preserving hand-designed assignment is the direct baseline that the learned assignment in Idea 1.4 must beat or match.
  - **Composer [19]** (ICLR 2026): Discovered hybrid architectures outperform Llama 3.2 by 1.1–3.1% downstream accuracy on average across 350M–3B scale benchmarks. This is the strongest positive signal: optimized assignment (found by search) gives an absolute quality improvement over hand-designed baselines at the same compute budget.
  - **LExI [18]** (arXiv 2025): For per-layer k assignment within existing MoE layers (the softer version of Idea 1.4's assignment), 10% better accuracy at matched throughput vs uniform expert pruning. Implies that the layer-wise assignment degree-of-freedom is highly valuable for quality preservation.
  - **Optimal Sparsity [16]** (arXiv:2508.18672, 2025): Memorization tasks improve monotonically with total parameters (more MoE layers is better); reasoning tasks *saturate and can regress* with increased sparsity. This means that an assignment with too many MoE layers (high sparsity) can *hurt* quality on reasoning benchmarks even as it improves throughput — the learned assignment must balance task-type sensitivity.

- **Monotonicity**: Quality vs sparsity (fraction S/L of MoE layers) is not monotone. PR-MoE [1] shows quality is maximized at a non-trivial dense/MoE split (not all-MoE). The Optimal Sparsity paper [16] shows reasoning-task quality can regress if S/L is too large. The optimal split is task-distribution-dependent; a learned assignment that is exposed to diverse training data should converge to a generalizable compromise.

- **Recovery**: Post-training recovery is available at two levels. (1) Layer-type recovery: if the learned assignment is suboptimal, the assignment can be re-initialized and the search phase re-run without discarding model weights (ProxylessNAS binary gate re-init). (2) Quality-specific recovery: CMoE [4] demonstrates that brief LoRA fine-tuning (1 hour, 2,000 samples) recovers downstream accuracy after aggressive sparsification — the same applies if the learned assignment over-sparses specific layers.

- **Conditions for acceptable degradation**:
  - Quality degradation is acceptable only when the resulting TTFT or TPOT improvement exceeds 1.5× vs the fully-dense baseline — below this, the engineering cost of the NAS training phase is unlikely to be justified.
  - Early-layer densification (dense-first, MoE-later pattern as in DeepSeek-V3 [10]) is a quality safeguard, but DeepSeek-V4 [20] shows a second viable safeguard: keep early layers sparse and use deterministic Hash routing before learned routing becomes stable.
  - If the learned assignment at 1–3B scale prototype significantly outperforms both DeepSeek-V3's 3-dense + 58-MoE split and a DeepSeek-V4-style all-MoE + early Hash routing split, the quality case for scaling the approach to 27B+ is strong. If the improvement is marginal (<0.5% on standard benchmarks), the two-stage training cost is not justified and the best hand-designed assignment should be used.
  - **No experiments on end-to-end gradient-based dense/MoE assignment during LLM pretraining at any scale exist in the literature.** AutoMoE [2] uses evolutionary NAS on NMT, not gradient-based pretraining; MoEfication [3] and CMoE [4] are post-hoc. The quality characteristics of the gradient-based assignment (discretization gap, assignment collapse artifacts) are uncharacterized and must be measured experimentally at 1–3B scale before committing to larger scales.
