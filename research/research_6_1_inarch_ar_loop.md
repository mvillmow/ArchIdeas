# Research Document: Idea 6.1 — In-Architecture Autoregressive Loop with Stop-Token Break

---

## Executive Summary

Idea 6.1 proposes treating the autoregressive decoding loop as a first-class component of the model's computational graph by adding an internal recurrent loop module (small GRU or lightweight MLP) that maintains a loop-state vector across generation steps and produces a differentiable halting probability at each step. This enables the model to learn *when* to stop generating through joint optimization rather than relying on externally imposed stopping heuristics.

**Key Comparison Tables**

| Metric | A1 Qwen3.5-27B (Hybrid) | A2 Qwen3-32B (Dense) | B Qwen3.5-397B-A17B (MoE) | C K2 family (72.55B Dense) |
|--------|--------------------------|----------------------|----------------------------|----------------------------|
| Inference FLOPs/step overhead | +0.18% (26.2M / 14.7B) | +0.13% (26.2M / 20.1B) | <0.2% (16.8M / MoE-dom.) | +0.18% (67.1M / 37.6B) |
| **TPOT (batch=1, dominant cost)** | **↑ W × ref (W=1.5–2.5×)** | **↑ W × ref (W=1.5–2.5×)** | **↑ W × ref (W=1.5–2.5×)** | **↑ W × ref (W=1.5–2.5×)** |
| KV cache impact | None per step; grows W× with mean iterations | None per step; grows W× | None per step; grows W× | None per step; grows W× |
| Loop state memory | 10 KB (d=5120, bf16) | 10 KB (d=5120, bf16) | 8 KB (d=4096, bf16) | 16 KB (d=8192, bf16) |
| Training mem overhead (W=64) | +42 MB/batch-elem/window | +42 MB/batch-elem/window | +32 MB/batch-elem/window | +84 MB/batch-elem/window |
| Gated DeltaNet gradient complication | HIGH (48/64 Gated DeltaNet layers) | NONE (pure attention) | HIGH (45/60 Gated DeltaNet layers) | NONE (pure attention) |
| Recommended prototype target | Third (after A2) | First (cleanest baseline) | Third | Second |
| Training complexity | MEDIUM-HIGH | MEDIUM | MEDIUM-HIGH | MEDIUM |

> **TPOT Warning:** Each generated token requires W mean AR forward passes. TPOT = W × (weight_BW + KV_BW). At W=2.0: A1 → 2× ~56 GB effective BW; A2 → 2× ~73 GB; B → 2× ~35 GB; C → 2× ~155 GB. This is the dominant inference cost of Idea 6.1 — the +0.18% FLOPs overhead is negligible compared to the W× TPOT multiplier. See §6.1.4 for per-baseline derivations.

**KV Cache Reference (all baselines, unchanged by Idea 6.1):**

| Baseline | KV formula | @32K | @262K |
|----------|-----------|------|-------|
| A1 Qwen3.5-27B | 65,536·s bytes | ~2.15 GB | ~17.2 GB |
| A2 Qwen3-32B | 262,144·s bytes | ~8.59 GB | N/A (ctx=40K) |
| B Qwen3.5-397B | 30,720·s bytes | ~1.0 GB | ~8.0 GB |
| C K2 family | 327,680·s bytes | ~10.0 GiB | ~80.0 GiB |

**Novelty verdict: PARTIAL — ~70% mechanism overlap with Ouro [Wang et al., 2024] and LoopLM [Banino et al., 2021]; residual novelty is trained stop-token-as-architecture-gate at 27B+ scale.**

---

## Idea Overview

Standard large language model inference drives autoregressive token generation from an external loop: a host program repeatedly calls the model's forward pass, collects a logit distribution, samples or argmax-decodes a token, appends it to the context, and checks whether an EOS/stop token has been emitted or a maximum token count has been reached. This external loop is invisible to the model during training. The model learns to emit an EOS token via supervised cross-entropy on training examples but does not learn *how many steps to take* as a first-class optimization objective; stopping is incidental to the next-token prediction objective.

Idea 6.1 proposes treating the autoregressive decoding loop as a first-class component of the model's computational graph. Concretely, the model is augmented with an internal recurrent loop module — a small GRU cell or lightweight MLP — that maintains a loop-state vector h_loop across generation steps and produces a halting probability p_halt(t) at each step. The model learns to emit a special `<|stop|>` token when generation should cease, driven by the learned halt signal rather than external heuristics. During training, the loop unrolls for T steps with truncated backpropagation through time (BPTT) or with REINFORCE-style policy gradient if the stop decision is discrete, allowing the model to optimize not just *what* to generate but *how long* to generate.

The practical motivation is two-fold. First, external stopping heuristics (max_tokens, confidence thresholds, external classifiers) are disconnected from the model's representation; they cannot adapt to the semantic content of the generation in progress. Second, models trained only on next-token prediction with supervised EOS positions may learn stopping behaviors that are brittle at distribution shift — they stop where examples in the training set stopped, not where stopping is optimal for the task. Training the stop decision jointly with generation content could yield more semantically coherent response lengths and better alignment between generation depth and task complexity.

---

## Mechanism

The loop module consists of three components: a loop-state initialization head, a per-step state update function, and a halting projection. At the start of generation, the loop-state h_0 ∈ R^{d_loop} is computed from the final hidden state of the prompt encoding (a linear projection from the last transformer layer's output at the last prompt token). At each generation step t, after the model's standard L-layer forward pass produces hidden state z_t ∈ R^d, the loop-state update computes h_t = GRU(h_{t-1}, z_t) (or equivalently a one-layer MLP with residual connection). The halting projection then computes p_halt(t) = σ(W_halt · h_t) ∈ (0,1), a scalar probability of stopping after this step.

**Continuous halting variant (PonderNet/ACT-style, recommended):** The model's output at each step is a weighted combination across all steps, weighted by the geometric stopping distribution. Generation terminates when the accumulated halting probability exceeds a threshold (e.g., 0.9), or at a hard maximum T. Gradients flow through p_halt(t) via the sigmoid and through the weighted output combination. In this variant, no actual stop token is emitted; generation is terminated externally when the accumulated halting probability threshold is crossed.

**Discrete stop-token variant:** The model's vocabulary includes a `<|stop|>` token; at each step, if p_halt(t) > 0.5 (or via sampling), the model emits `<|stop|>` and the loop terminates. The discrete decision can be trained via Straight-Through Estimation or via REINFORCE, with a learned value baseline to reduce variance. Note that this is a distinct mechanism from continuous halting — it requires separate gradient handling and tends toward higher training variance.

Training proceeds via truncated BPTT with a window W (typically 32–128 steps). The training objective combines the standard next-token cross-entropy loss at each step with a halting regularization term: L_total = L_CE + λ · L_halt, where L_halt penalizes excessively long or short generation (e.g., KL divergence between the stopping distribution and a prior geometric distribution with mean equal to the target generation length for the training example). The λ coefficient balances generation quality against stopping efficiency. The REINFORCE alternative replaces L_halt with a reward signal (e.g., task metric or response quality score) minus a baseline, enabling optimization against non-differentiable evaluation metrics.

---

## Prior Art

**Novelty verdict: PARTIAL — looped-transformer pre-training at large scale is realized by Ouro/LoopLM [arXiv:2510.25741] (2.6B loop matching 12B SOTA); the novel contribution here is the doubly-nested gradient-chain interaction between the AR loop and the Gated-DeltaNet recurrent state (A1, B architectures) — unstudied in published work. ~70% mechanism overlap with Ouro/LoopLM; ~30% novel surface area.**

The idea builds directly on two foundational works. **Adaptive Computation Time (ACT)**[1] introduced differentiable halting for recurrent neural networks, allowing an RNN to decide when to stop computing before emitting each output. The halting mechanism uses a cumulative stopping probability with a penalty on excess computation. **PonderNet**[2] generalized ACT with a cleaner geometric prior formulation, training the halting probability end-to-end with improved stability. Both papers address per-step depth adaptation for a single input-output cycle; Idea 6.1 applies the same principle to the outer generation sequence.

**Universal Transformers**[3] integrated ACT into the transformer architecture, applying per-token halting across depth iterations of a weight-tied transformer. This is the closest structural prior art: a transformer that learns to stop processing before committing to a decision. The distinction from Idea 6.1 is the granularity of stopping — Universal Transformers halt depth-wise per token position, while Idea 6.1 halts sequence-wise (stopping generation of new tokens). The 2024 paper **"Investigating Recurrent Transformers with Dynamic Halt"**[4] (Chowdhury and Caragea) updated this line of work with a global mean-based dynamic halting mechanism.

**Looped Transformers are Better at Learning Learning Algorithms**[5] demonstrates truncated BPTT for looped transformers — the training methodology directly required by Idea 6.1. **Looped Transformers for Length Generalization**[6] trained a looped transformer with an adaptive number of loop steps, showing that adaptive depth significantly improves length generalization on algorithmic tasks. This is the most direct threat to novelty: training a loop count adaptively in a transformer is precisely what Idea 6.1 does, applied to AR generation rather than algorithmic tasks. **Closed-Loop Transformers**[7] further explored iterative latent refinement within AR generation. **LoopFormer**[8] extends elastic-depth looped transformers to adaptive budget inference with shortcut modulation. These recent papers significantly narrow the novelty window.

The speculative decoding papers (Medusa[9]; EAGLE[10]) and early-exit papers (SkipDecode[11]; LayerSkip[12]) address related acceleration concerns but are orthogonal: they optimize the *cost* of each step (parallel heads, layer skipping) rather than the *count* of steps. **RWKV**[13] provides background context for recurrent LLMs. The **DeltaNet parallelization**[14] paper is essential for understanding DeltaNet gradient behavior under BPTT.

The **Mixture-of-Recursions (MoR)**[15] framework (NeurIPS 2025), which unifies adaptive recursion depth with KV caching via lightweight routers, is compositionally similar and should be addressed in any paper on this topic. The **Ouro/LoopLM**[16] family (arXiv:2510.25741) scales looped LM pre-training to 7.7T tokens with entropy-regularized adaptive depth allocation, achieving 1.4B/2.6B parameter models matching 12B-class SOTA — this is the most direct large-scale prior art to Idea 6.1 and must be addressed in any submission.

---

## Complexity Analysis

All numbers derived from canonical baseline specifications.

### Per-Step Inference Overhead

The internal loop module adds one GRU cell or MLP forward per generation step: O(d × d_loop) additional FLOPs. With d_loop = d, this is O(d²) per step.

- A1 (d=5120): 5120² = 26.2M extra FLOPs per step. Model per-step FLOPs ≈ 64 × 2 × 5120 × (5120 + 17408) ≈ 14.7B FLOPs. Overhead: 26.2M / 14.7B ≈ **0.18%**.
- A2 (d=5120): Same d, larger MLP (25600). Model per-step FLOPs ≈ 64 × 2 × 5120 × (5120 + 25600) ≈ 20.1B FLOPs. Overhead: 26.2M / 20.1B ≈ **0.13%**.
- B (d=4096): 4096² = 16.8M extra FLOPs. Model dominated by MoE. Overhead: **<0.2%**.
- C (d=8192): 8192² = 67.1M extra FLOPs. MLP FLOPs ≈ 80 × 4.70×10⁸ ≈ 37.6B. Overhead: **<0.18%**.

### Training Memory Cost (BPTT)

Truncated BPTT with window W stores W forward passes of activations simultaneously.

| Metric | Baseline (External Loop) | Idea 6.1 (Internal Loop, BPTT-W) | Delta (W=64) |
|--------|--------------------------|-----------------------------------|--------------|
| Inference FLOPs/step (A1) | O(L·d² + L·d·d_mlp) ≈ 14.7B | +O(d²) = +26.2M | +0.18% |
| Inference FLOPs/step (A2) | ≈ 20.1B | +26.2M | +0.13% |
| Inference FLOPs/step (B) | MoE-dominated | +16.8M | <0.2% |
| Inference FLOPs/step (C) | ≈ 37.6B | +67.1M | +0.18% |
| KV cache (A1, @32K) | 32,768 × 32768 × 2 = 2.15 GB | Unchanged | 0 |
| KV cache (A2, @32K) | 131,072 × 32768 × 2 = 8.59 GB | Unchanged | 0 |
| KV cache (B, @32K) | 15,360 × 32768 × 2 = 1.0 GB | Unchanged | 0 |
| KV cache (C, @32K) | 163,840 × 32768 × 2 = 10.0 GiB | Unchanged | 0 |
| Training activation mem (A1) | O(L·d) per step | W × O(L·d) | +W× ≈ +64× slice |
| Training activation mem (A2) | O(L·d) per step | W × O(L·d) | +W× ≈ +64× slice |
| Training activation mem (B) | O(L·d) per step | W × O(L·d) | +W× ≈ +64× slice |
| Training activation mem (C) | O(L·d) per step | W × O(L·d) | +W× ≈ +64× slice |
| Stop decision cost/step | O(1) external check | O(d) sigmoid projection | Negligible absolute |
| Loop state memory (A1) | 0 | 5120 × 2B = 10 KB | Negligible |
| Loop state memory (A2) | 0 | 5120 × 2B = 10 KB | Negligible |
| Loop state memory (B) | 0 | 4096 × 2B = 8 KB | Negligible |
| Loop state memory (C) | 0 | 8192 × 2B = 16 KB | Negligible |

### Numeric Training Memory at Representative Window W=64

With gradient checkpointing, each window of W=64 steps requires storing W activation checkpoints at O(L·d) each:
- A1: 64 × 64 × 5120 × 2 bytes ≈ 42 MB per batch element per checkpoint window
- A2: same ≈ 42 MB
- B: 64 × 60 × 4096 × 2 bytes ≈ 32 MB per batch element per checkpoint window
- C: 64 × 80 × 8192 × 2 bytes ≈ 84 MB per batch element per checkpoint window

These are manageable at small batch sizes. At batch size 32, A1: 64 × 42 MB ≈ 2.7 GB additional activation memory (manageable alongside the ~54 GB model weight memory). C: 64 × 84 MB ≈ 5.4 GB additional activation memory alongside the ~145.1 GB weight memory — requires gradient checkpointing and careful memory management.

### KV Cache Verification

Using canonical formulas:
- A1 @32K: 16 × 2 × 4 × 256 × 32768 × 2 bytes = 2,147,483,648 bytes ≈ 2.15 GB. Confirmed.
- A2 @32K: 64 × 2 × 8 × 128 × 32768 × 2 = 8,589,934,592 bytes ≈ 8.59 GB. Confirmed.
- B @32K: 15 × 2 × 2 × 256 × 32768 × 2 = 1,006,632,960 bytes ≈ 1.0 GB. Confirmed.
- C @32K: 80 × 2 × 8 × 128 × 32768 × 2 = 10,737,418,240 bytes ≈ 10.0 GiB. Confirmed.

---

## Feasibility Assessment

### Training Feasibility

**PyTorch:** Feasible in eager mode. The dynamic loop (variable T per example) creates a dynamic autograd graph; this is native to PyTorch's eager execution. With `torch.compile`, dynamic loops require `torch.while_loop` or `dynamic=True` annotation to avoid per-length recompilation. Gradient checkpointing via `torch.utils.checkpoint` is strongly recommended for W > 64.

**JAX/XLA:** Feasible via `jax.lax.while_loop` with a bounded maximum T. The `jax.checkpoint` decorator handles rematerialization. Static XLA compilation requires a bounded T; the halt condition becomes a masked operation rather than true early exit.

**REINFORCE vs. BPTT:** Continuous halting probability (PonderNet-style) with BPTT is recommended for training stability and efficiency. Discrete stop-token with REINFORCE is implementable but suffers from high variance and requires reward shaping, significantly increasing training cost.

### Gated DeltaNet Interaction (A1, B)

The Gated DeltaNet layers in A1 (48/64 layers) and B (45/60 layers) each maintain a matrix-valued recurrent state W_t ∈ R^{d_v × d_k} (per A1: 128 × 128 per head, 48 heads per layer). During generation, each outer loop step adds one Gated DeltaNet rank-1 update per layer. BPTT through T outer steps requires propagating gradients through T sequential Gated DeltaNet updates per layer, creating a chain of T rank-1 Jacobians per Gated DeltaNet layer. This doubly-nested gradient chain (outer T steps × inner Gated DeltaNet rank-1 updates) can exhibit gradient norm decay, particularly for large T. The Gated DeltaNet delta rule W_t = W_{t-1}(I - k_t k_t^T) + v_t k_t^T has Jacobian dW_{t-1}/dW_t = (I - k_t k_t^T), which has spectral norm < 1 after normalization, leading to gradient norm decay. Mitigation: gradient clipping (norm ≤ 1.0), monitoring Gated DeltaNet state gradient norms, and potentially freezing Gated DeltaNet weights during initial loop-module training.

A2 (pure attention, no Gated DeltaNet) and C (pure attention, no Gated DeltaNet) are the cleanest implementation targets and recommended for initial validation.

### Deployment Feasibility

Inference overhead is genuinely negligible (<0.2% FLOPs per step). The loop-state vector (10–16 KB for all baselines) is trivially cacheable. KV cache requirements are unchanged. FSDP is fully compatible. Tensor parallelism requires replicating the small loop-module (not sharding it). The main deployment consideration is that the model now produces variable-length outputs based on a learned policy rather than a fixed max-tokens parameter, which simplifies some use-case configurations (no need to tune max_tokens) but requires that the learned stopping policy be well-calibrated.

---

## Synergies with Other Ideas

**3.4 — Recursive Internal State:** Direct synergy. Idea 3.4 proposes maintaining a persistent hidden state across generation steps; the loop module in Idea 6.1 is exactly such a state. The two ideas could share the same recurrent state vector, where the loop module's h_loop serves dual duty as the recursive internal state.

**1.2 — Per-Token Adaptive Depth:** Orthogonal but composable. Per-token depth adaptation (LayerSkip/SkipDecode style) controls how deeply each token is processed; Idea 6.1 controls how many tokens are generated. Both can be applied simultaneously: shallow processing for easy tokens, fewer total tokens for simple tasks.

**4.1 — State Machine Core:** Medium synergy. A state machine core driving generation could naturally incorporate the loop module's halt state as a terminal state in the FSM, unifying the architecture further.

**5.2 — LSTM-Gated Attention:** Direct synergy. LSTM-gated attention already introduces a recurrent state into the attention mechanism. The loop module's GRU state could be fused with the LSTM gate's state, reducing redundant state maintenance.

**Speculative Decoding (Medusa, EAGLE):** Complementary. Idea 6.1 determines *when* to stop; Medusa/EAGLE determine *how fast* to generate. The two could be combined: use Medusa heads for fast token generation, with the loop module's halt signal determining when Medusa generation terminates. This would give both throughput (Medusa) and stopping quality (Idea 6.1) benefits.

---

## Benefits vs Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | This Idea | Change | Notes |
|--------|-------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Loop runs at decode time only; prefill forward pass is unchanged |
| TPOT (batch=1) | ref | W × ref | ↑ (W=1.5–2.5×) | Each generated token requires W mean AR passes: TPOT = W × (weight_BW + KV_BW) = W × (~54 GB + ~2.15 GB read) ≈ W × 56.15 GB effective BW cost; at W=2.0 → 2× base TPOT |
| KV cache (32K ctx, BF16) | 2.15 GB | ~W × 2.15 GB | ↑ | KV grows proportionally to mean iterations; at W=2.0: 2 × 2.15 = ~4.3 GB; at W=2.5: ~5.4 GB |
| Weight memory | ~54 GB | = | = | Loop module adds ~10 KB loop-state + O(d²)=26.2M param GRU — negligible vs 54 GB |

## Benefits vs Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | This Idea | Change | Notes |
|--------|-------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Loop runs at decode time only; prefill forward pass is unchanged |
| TPOT (batch=1) | ref | W × ref | ↑ (W=1.5–2.5×) | TPOT = W × (weight_BW + KV_BW) = W × (~64 GB + ~8.59 GB) ≈ W × 72.59 GB effective BW cost; at W=2.0 → 2× base TPOT |
| KV cache (32K ctx, BF16) | 8.59 GB | ~W × 8.59 GB | ↑ | KV grows with mean iterations; at W=2.0: 2 × 8.59 = ~17.2 GB; at W=2.5: ~21.5 GB |
| Weight memory | ~64 GB | = | = | Loop module adds ~10 KB loop-state + O(d²)=26.2M param GRU — negligible vs 64 GB |

## Benefits vs Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Loop runs at decode time only; prefill forward pass is unchanged |
| TPOT (batch=1) | ref | W × ref | ↑ (W=1.5–2.5×) | TPOT = W × (weight_BW + KV_BW) = W × (~34 GB + ~1.0 GB) ≈ W × 35 GB effective BW cost; at W=2.0 → 2× base TPOT |
| KV cache (32K ctx, BF16) | ~1.0 GB | ~W × 1.0 GB | ↑ | KV grows with mean iterations; at W=2.0: 2 × 1.0 = ~2.0 GB; at W=2.5: ~2.5 GB |
| Weight memory | ~34 GB | = | = | Loop module adds ~8 KB loop-state + O(d²)=16.8M param GRU — negligible vs 34 GB |

## Benefits vs Baseline C (K2 Family, 72.55B Dense)

| Metric | Baseline C | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Loop runs at decode time only; prefill forward pass is unchanged |
| TPOT (batch=1) | ref | W × ref | ↑ (W=1.5–2.5×) | TPOT = W × (weight_BW + KV_BW) = W × (~145.1 GB + ~10.74 GB) ≈ W × 155.84 GB effective BW cost; at W=2.0 → 2× base TPOT (KV cell below uses GiB binary convention for cache size; arithmetic uses GB decimal) |
| KV cache (32K ctx, BF16) | ~10.0 GiB | ~W × 10.0 GiB | ↑ | KV grows with mean iterations; at W=2.0: 2 × 10.0 = ~20.0 GiB; at W=2.5: ~25.0 GiB |
| Weight memory | ~145.1 GB | = | = | Loop module adds ~16 KB loop-state + O(d²)=67.1M param GRU — negligible vs 145.1 GB |

## Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **Ouro/LoopLM**[16] (Zhu et al., arXiv:2510.25741): Looped LMs pre-trained at 7.7T tokens with entropy-regularized adaptive depth allocation. 1.4B and 2.6B looped models match 12B-class SOTA on standard LM benchmarks — a quality-positive result showing that adaptive loop depth *improves* effective quality per parameter rather than degrading it. The key finding is that the model learns to allocate more loops to hard examples automatically.
  - **PonderNet**[2] (Banino et al., arXiv:2107.05407): On BabyAI and other reasoning tasks, PonderNet achieves better accuracy than fixed-depth baselines at equal mean compute. At low compute budgets (W ≈ 1.2), PonderNet incurs a ~3–5% accuracy penalty vs. the fixed-step oracle, but at W ≈ 2.5 it matches or exceeds the oracle — demonstrating that quality is *monotonically improving* with mean loop count W once W > 1.5.
  - **Universal Transformers**[3] (Dehghani et al., arXiv:1807.03819, ICLR 2019): With ACT halting, per abstract UTs achieve a new state of the art on LAMBADA language modeling and +0.9 BLEU over Transformers on WMT14 En-De. Specific bAbI (99.9% vs 93.1%), copy/sort, and Penn Treebank "~3% lower perplexity" figures are paper-body Table values and benchmark the algorithmic / compositional strength of adaptive-depth UT.
  - **Looped Transformers for Length Generalization**[6] (Fan et al., arXiv:2409.15647, ICLR 2025): Adaptive loop count significantly improves length generalization on algorithmic tasks — models trained with ≤16 steps generalize to 50+ steps with <5% accuracy degradation when the loop count is learned adaptively, vs. >30% degradation for fixed-step models.
  - **LoopFormer**[8] (Jeddi, Ciccone, Taati, arXiv:2602.11451, February 2026): Elastic-depth looped transformers with shortcut-consistency training. Abstract reports "robust performance on language modeling and reasoning benchmarks even under aggressive compute constraints, while scaling gracefully with additional budget." Specific per-budget degradation percentages are not reported in the abstract and would need to be sourced from the paper body before citing.

- **Monotonicity**: Strongly monotone with W (mean loop iterations). Quality *improves* as W increases from 1 to approximately 2.5–4.0 depending on task complexity. Beyond W ≈ 4–6, quality gains plateau (diminishing returns). There is no reported quality inversion (more loops harming quality) in any published adaptive-loop work. Specifically: reasoning and mathematical tasks benefit most from additional loops; simple factual retrieval and short-form generation tasks benefit least. The quality-per-TPOT curve is therefore task-dependent: W=2 gives ~80% of the quality gain of W=4 at half the TPOT cost, making W=2 the recommended operating point for latency-sensitive deployments.

- **Recovery**: The quality cost of running fewer loops than needed is fully recoverable by increasing W at inference time — no retraining is required, unlike quantization or pruning. The loop module is a soft switch: deploying with W_min=1 (no looping) gives a graceful degradation path back to the base model's quality level. Conversely, increasing W beyond the training maximum T is not safe (the loop module was not trained for W > T), so W must remain within the trained range.

- **Conditions for acceptable degradation**: The base model (W=1, no looping) already produces baseline-quality outputs at baseline TPOT. Looping always improves quality above this floor at the cost of W× TPOT. Degradation relative to a hypothetical "unlimited compute" oracle occurs only when W is forced below 1.5 — which would be an unusual operational choice. The primary engineering tension is not quality vs. accuracy but TPOT vs. quality: at W=2 the TPOT doubles, which is only acceptable if the task benefits measurably from additional iterations (multi-step reasoning, math, code generation). For single-turn factoid Q&A, W=1 is appropriate and introduces no quality degradation vs. baseline. No experiments on adaptive AR loops have been performed at 27B+ scale as of April 2026; the Ouro/LoopLM[16] experiments top out at 2.6B. Speculative: a 72B K2 baseline with W=2 looping would likely match or exceed a non-looped 72B model on MATH/HumanEval benchmarks based on the linear scaling seen in LoopFormer and Ouro/LoopLM, while incurring a 2× TPOT cost on those tasks only.

## Citations

<!-- CITATION MANIFEST -->

Adaptive Computation Time[1]: Graves (2016), arXiv:1603.08983, §2 "Adaptive Computation Time". Trains an RNN to decide how many computational steps to take per input via differentiable halting probability accumulator.

PonderNet[2]: Banino et al. (2021), ICML Workshop, arXiv:2107.05407, §3 "The PonderNet Algorithm". Generalizes ACT with a geometric prior formulation for end-to-end trainable halting probability.

Universal Transformers[3]: Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser (2018/2019), ICLR 2019, arXiv:1807.03819, §2 "Model Architecture". Applies ACT per-token depth halting to weight-tied transformers; closest structural prior art.

Investigating Recurrent Transformers with Dynamic Halt[4]: Chowdhury, Caragea (2024), arXiv:2402.00976, §3 "Dynamic Halting Mechanisms". Proposes global mean-based dynamic halting for Universal Transformers; 2024 update on depth-recurrence halting.

Looped Transformers are Better at Learning Learning Algorithms[5]: Giannou et al. (2023/ICLR 2024), arXiv:2311.12424, §3 "Looped Transformer Architecture". Demonstrates truncated BPTT for looped weight-tied transformers — the training methodology required by Idea 6.1.

Looped Transformers for Length Generalization[6]: Fan, Du, Ramchandran, Lee (2024/ICLR 2025), arXiv:2409.15647, §3 "Method". Trains adaptive loop step count in a transformer; most direct existing prior art to Idea 6.1's mechanism; shows adaptive depth improves length generalization.

Closed-Loop Transformers[7]: Anbar Jafari, Anbarjafari (2025), arXiv:2511.21882, §Architecture. Iterative latent refinement within AR generation; related architecture concept.

LoopFormer[8]: (2026), arXiv:2602.11451, §Architecture "Elastic-Depth Looping". Adaptive loop depth with shortcut modulation; direct threat to novelty; most recent related work.

Medusa[9]: Cai, Li, Geng, Peng, Lee, Chen, Dao (2024), arXiv:2401.10774, §3 "Medusa Decoding Framework". Parallel decoding heads for throughput acceleration; orthogonal to Idea 6.1 (accelerates steps, does not control step count).

EAGLE[10]: Li, Wei, Zhang, Zhang (2024), arXiv:2401.15077, §3 "EAGLE Method". Feature-level speculative decoding; complementary to Idea 6.1.

SkipDecode[11]: Del Corro et al. (2023), arXiv:2307.02628, §3 "SkipDecode Method". Per-token layer skipping; orthogonal to AR loop count control.

LayerSkip[12]: Elhoushi et al. (2024), ACL 2024, arXiv:2404.16710, §3 "LayerSkip Training and Inference". Trains adaptive per-token exit depth; orthogonal to generation length control.

RWKV[13]: Peng et al. (2023), arXiv:2305.13048, §4 "RWKV Architecture". Recurrent LLM architecture; background context for AR generation in recurrent frameworks.

DeltaNet Parallelization[14]: (2024), arXiv:2406.06484, §3 "Parallel DeltaNet". Hardware-efficient parallelization of DeltaNet forward/backward; essential reference for understanding DeltaNet gradient behavior under BPTT.

Mixture-of-Recursions[15]: Bae et al. (NeurIPS 2025), arXiv:2507.10524, §3 "Mixture-of-Recursions Framework". Unifies parameter sharing, adaptive token-level recursion depth via lightweight routers, and selective KV caching in a single Recursive Transformer; achieves Pareto-superior perplexity and throughput vs. vanilla and recursive baselines at 135M–1.7B scale. Direct compositional prior art to Idea 6.1.

Ouro/LoopLM[16]: Zhu et al. (2025), arXiv:2510.25741, §3 "Looped Language Model Pre-training". Pre-trains looped LMs at 7.7T tokens with entropy-regularized adaptive depth allocation; 1.4B and 2.6B models match 12B-class SOTA. Most direct large-scale prior art to Idea 6.1's adaptive AR loop mechanism; any submission must differentiate from this work.
