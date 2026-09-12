# Research: LoRA Everywhere (Full-Model Core + Adapter Split)
## ID: 4.3

## Executive Summary

**Novelty verdict:** PARTIAL — LoRA-everywhere as a training-time parameterization W = W_core + A·B is ~60% covered at the 4.3-C pure-A·B variant by CoLA (pre-training parameterization of all linear layers up to 7B) [CoLA, 6]; GaLore/GaLore 2 cover gradient-level low-rank while preserving full-rank weights [GaLore, 3; GaLore 2, 4]; SLTrain covers low-rank + sparse from-scratch pretraining [SLTrain, 10]; residual novelty is (a) shared W_core across layers *combined with* per-layer A·B and (b) purely linear W = W_core + A·B without bottleneck nonlinearity at native pretraining scale above 7B — neither demonstrated in a single paper ([CoLA, 6], [GaLore 2, 4], [SLTrain, NeurIPS 2024, 10]).

Idea 4.3 proposes parameterizing every weight matrix in the model as W = W_core + A·B from the start of pre-training. The review identifies **three distinct realizations** that must be analyzed separately:

- **4.3-A (Training-only merge):** W = W_core + A·B merged post-training → identical to dense. Only training-time optimizer memory reduction (~50%). Zero inference benefit.
- **4.3-B (AB retained + W_core merge bias):** Identical FLOP count to dense. No benefit over 4.3-A.
- **4.3-C (Pure A·B at inference, no W_core):** 2× weight memory reduction, ~1.64× TPOT improvement at batch=1 decode. TTFT is ~2× FASTER (not slower). This is the only variant with inference benefits.

Quality parity demonstrated up to 7B scale with nonlinear bottleneck (CoLA, Liu et al., 2025); at 27B+ scale, the nonlinear bottleneck is load-bearing and uniform rank treatment risks catastrophic perplexity degradation on V-projection and MLP down-projections.

**Scale-validation status:** Quality parity for pure A·B inference (Case 4.3-C) with nonlinear bottleneck is validated at ≤7B scale (CoLA, Liu et al., 2025). At 27B+ scale, the nonlinear bottleneck is load-bearing and uniform-rank treatment risks catastrophic perplexity degradation on V-projection and MLP down-projections.
**Recommendation: Prototype on A2 (dense) at ≤7B before any 27–32B commitment.**


## Key Comparison Tables

### vs. Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | 4.3-A (merge) | 4.3-C (pure A·B, r=d/4) |
|--------|------------|--------------|------------------------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | = (identical) | ↓ ~0.5× (halved weight matmul FLOPs) |
| Memory bandwidth (decode) | O(L·(d²+s·d_kv/4)) | = (identical) | ↓ ~2× weight bytes (2× faster TPOT) |
| KV cache (32K ctx) | ~2.15 GB | = | = |
| Weight memory | O(L·d·d_ff) | = | ↓ ~2× at r=d/4 |
| Training cost | 1.0× | ~0.60–0.70× | ~0.60–0.70× |
| TTFT (8K prompt) | ref | ~1.0× | **~0.5× (2× faster)** |
| TPOT (batch=1) | ref | ~1.0× | ~1.64× faster (bandwidth-bound) |

### vs. Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | 4.3-A (merge) | 4.3-C (pure A·B, r=d/4) |
|--------|------------|--------------|------------------------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | = | ↓ ~0.5× weight matmul FLOPs |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | = | ↓ ~2× weight bytes |
| KV cache (32K ctx) | ~8.59 GB (GQA 8:1) | = | = |
| Weight memory | ~64 GB bf16 | = | ~32 GB (enabling single 40GB A100 deployment) |
| Training cost | 1.0× | ~0.60–0.70× | ~0.60–0.70× |
| TTFT (8K prompt) | ref | ~1.0× | **~0.5× (2× faster)** |
| TPOT (batch=1) | ref | ~1.0× | ~1.64× faster |

### vs. Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | 4.3-A (merge) | 4.3-C (pure A·B, r=d_e/4) |
|--------|-----------|--------------|---------------------------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | = | ↓ ~0.5× per activated expert |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | = | ↓ ~2× expert weight bytes |
| KV cache (32K ctx) | ~1.0 GB (15 full-attn, 32Q/2KV, hd=256) | = | = |
| Expert weight memory | O(L·E·d·d_e) | = | ↓ ~2× at r=d_e/4 |
| TPOT (batch=1) | ref | ~1.0× | ~1.5–1.64× faster (expert weight bytes halved) |

### vs. Baseline C (K2 family, 72.55B dense Llama-arch)

| Metric | Baseline C (K2) | 4.3-A (merge) | 4.3-C (pure A·B, r=d/4) |
|--------|----------------|--------------|------------------------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)), L=80 | = | ↓ ~0.5× weight matmul |
| Memory bandwidth | ~145.1 GB weight BW | = | ~72.5 GB (2× reduction) |
| KV cache (32K ctx) | ~10.0 GiB | = | = |
| KV cache (262K ctx) | ~80.0 GiB | = | = |
| Weight memory | ~72.55B params | = | ~36B params at r=d/4 |
| MLP FLOPs/token/layer | ~4.70 × 10⁸ | = | ~2.35 × 10⁸ (pure A·B) |
| TTFT (8K prompt) | ref | ~1.0× | **~0.5× (2× faster)** |
| TPOT (batch=1) | ref | ~1.0× | ~1.64× faster |

---

## 1. Idea Description

Every weight matrix in the model is parameterized as W = W_core + A·B from the start of pre-training, yielding training-time optimizer memory savings and — in the 4.3-C variant (pure W = A·B, no W_core, kept unmerged at inference) — approximately 2× weight memory reduction and ~1.64× inference throughput improvement.

**Three realizations must be distinguished:**

1. **4.3-A (Training-only reparameterization):** W = W_core + A·B during training; merged to a single matrix W_full = W_core + A·B at inference. Inference is identical to dense baseline. Only benefit: ~50% optimizer state reduction during training (A, B are smaller than W).

2. **4.3-B (AB retained with merged bias):** W_core merged as a bias; A and B retained at inference as separate matrices. Forward pass: W_core·x + A·(B·x) = same FLOPs as W_full·x. No inference benefit.

3. **4.3-C (Pure factorization, no W_core):** W = A·B from scratch; A and B retained unmerged at inference. Forward pass: A·(B·x) — two smaller matmuls. At r=d/4: 2 × (d × d/4 × d) = d²/2 operations vs. d² baseline — **2× fewer weight bytes, ~1.64× TPOT improvement** (validated by CoLA at 7B scale). **TTFT ~2× faster** (prefill is also compute-bound on these weight matmuls at long context).

---

## 2. Literature Review

### LoRA (2022, ICLR)
Foundational W = W_core + A·B structure. Designed for fine-tuning (W_core frozen from pretrained checkpoint). The "LoRA Everywhere" pre-training scenario generalizes this to all weights trained jointly from scratch.

LoRA: Low-Rank Adaptation of Large Language Models[1]: ICLR 2022, arXiv:2106.09685, §"Introduction", §4.2 "Optimal Rank r?"

### Intrinsic Dimensionality (2021, ACL)
Provides theoretical grounding: weight updates during fine-tuning live in a very low-dimensional subspace. Motivated LoRA's design. Implies from-scratch training may also be doable in a low-rank subspace, though this does not follow directly.

Intrinsic Dimensionality Explains Effectiveness of LM Fine-Tuning[2]: ACL 2021, arXiv:2012.13255, §4.1 "Measuring Intrinsic Dimensionality"

### GaLore (2024, ICML)
Projects gradient (not weight) onto a low-rank subspace during optimization. Enables pre-training LLaMA 1B and 7B from scratch with weights remaining full-rank at inference. Key distinction: GaLore shows training-memory benefit without committing weights to a low-rank inference structure. Training benefit (memory reduction ~50%) can be achieved without inference penalty.

GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection[3]: ICML 2024, arXiv:2403.03507, §"Introduction" (Table 1: memory comparison)

### GaLore 2 (2025, arXiv)
Extends GaLore to large-scale pre-training (Llama 7B, 500B tokens). Weights remain full-rank. Confirms low-rank gradient dynamics stable at 7B+ scale.

GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection[4]: arXiv:2504.20437, §"Abstract"

### LoRA+ (2024, ICML)
Setting identical learning rates for A and B is suboptimal. B requires higher learning rate by model-width-dependent ratio. Up to 2× fine-tuning speedup at equal compute. Critical for pre-training: learning rate schedule and A:B ratio must be carefully designed.

LoRA+: Efficient Low Rank Adaptation of Large Models[5]: ICML 2024, arXiv:2402.12354, §"Abstract"

### CoLA (2025, arXiv)
**Most direct implementation of idea 4.3-C in the pre-training context.** Parameterizes all linear layers as h′=B·σ(A·x) with nonlinear bottleneck (σ). Per abstract: "reduces the computing cost by 2× and improves training throughput by 1.86× while maintaining full-rank level performance"; resulting LLMs are "2× smaller, enabling faster inference with lower memory cost." Paper-body Tables 6 and 10 report 1.64× inference throughput and 1.67× memory reduction at r=d/4 on specific model sizes. Validation limited to 7B scale.

CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation[6]: arXiv:2502.10940 (2025), §3 "CoLA Architecture", §Tables 6 and 10

### WeLore (ICML 2025)
Non-uniform rank tolerance: Q/K projections tolerate ≥90% rank reduction; V and MLP down-projections resist compression. Uniform 50% rank reduction yields perplexity 1836.62 vs 11.87 for non-uniform allocation on LLaMA-2 7B — catastrophic degradation from uniform treatment. Non-uniform rank allocation is essential.

From GaLore to WeLore: How Low-Rank Weights Non-uniformly Emerge from Low-Rank Gradients[7]: ICML 2025, arXiv:2407.11239, §"Rank Tolerance Analysis" (Table: uniform 1836.62 vs non-uniform 11.87 PPL at 50% effective rank reduction)

### Nuclear Norm Regularization Risk (2025)
Weight decay applied to the A·B factored form implicitly induces nuclear norm regularization, biasing solutions toward low rank and potentially causing rank collapse in attention layers. Training LoRA with weight decay is equivalent to nuclear norm regularization in full fine-tuning.

LoRA Training Provably Converges to a Low-Rank Global Minimum or It Fails Loudly[8]: arXiv:2502.09376 (2025), §"Implicit Regularization and Weight Decay"

### Intruder Dimensions (2024)
LoRA-structured training may introduce "intruder dimensions" — new, high-ranking singular vectors absent in full training — potentially degrading multi-task robustness and modeling of the pre-training distribution. Partially correctable post-hoc by scaling down the associated singular values.

LoRA vs Full Fine-tuning: An Illusion of Equivalence[9]: arXiv:2410.21228 (2024), Shuttleworth et al., §"Intruder Dimensions" and §"Causal Intervention"

### SLTrain (NeurIPS 2024)
Parameterizes weights as W = sparse + A·B from scratch at pretraining, training both components jointly. Achieves quality comparable to full-rank pre-training on LLaMA up to 7B while reducing parameter size by 42–45% and memory by up to 73% (with quantization). Directly related to idea 4.3: extends pure-factorization pretraining with a sparse residual that substitutes for W_core, recovering quality without a full-rank W_core. Key distinction: sparse component restores expressivity that pure low-rank (4.3-C) must otherwise obtain from the nonlinear bottleneck.

SLTrain: A Sparse Plus Low-Rank Approach for Parameter and Memory Efficient Pretraining[10]: NeurIPS 2024, arXiv:2406.02214, §"Main Results" (Table 2: LLaMA 7B perplexity)

### LoRA-the-Explorer / LTE (2024)
Trains neural networks from scratch using parallel low-rank adapters via a bi-level optimization algorithm. Periodically merges multiple parallel low-rank heads back to main weights, enabling large-scale synchronization-efficient distributed pretraining. Competitive with standard pre-training on vision transformers. Directly relevant as a pretraining-from-scratch low-rank parameterization approach.

Training Neural Networks from Scratch with Parallel Low-Rank Adapters[11]: arXiv:2402.16828 (2024), Huh et al., §"Abstract" and §"Experiments"

---

## 3. Prior Art Classification

**Novelty verdict: PARTIAL — CoLA [Chen et al., arXiv:2502.18066] realizes pre-training LoRA-everywhere at ≤7B scale; the novel surface is quality parity of pure low-rank (4.3-C variant: W = A·B with no dense W_core) with a dense baseline at 27B+ parameter scale — unconfirmed in published work. ~60% component overlap; ~40% novel validation regime.**

### 3.1 Prior Art Gap

CoLA[6] realizes "LoRA Everywhere" as pre-training parameterization for all linear layers at up to 7B scale, but inserts a nonlinearity between A and B and has no shared W_core across layers. The specific novelties of idea 4.3 — shared W_core and purely linear W=W_core+A·B without bottleneck nonlinearity at native pre-training scale — remain untested above 7B parameters.

---

## 4. Technical Analysis

### TTFT Analysis

For pure A·B inference (4.3-C), r=d/4:
- Full-rank dense: 2·s·d² FLOPs per layer (matrix multiply)
- Pure A·B: x·A = 2·s·d·(d/4) + (result)·B = 2·s·(d/4)·d = total 4·s·d·(d/4) = s·d² — **half the FLOPs**
- TTFT at 8K prefill: **~0.5× = 2× FASTER**

Case 4.3-B (W_core present + unmerged A·B) has MORE bytes to load and MORE FLOPs, yielding no TTFT improvement.

### Key Parameters (Baseline A2: d=5120, d_ff=25600)
- r=d/4 = 1280 for A2 attention layers (d×d matrices)
- r=d_ff/4 = 6400 for MLP projections (rectangular matrices)

> **Note on rank scale:** These ranks (r=1280, r=6400) are far above the 4–256 range typical in the LoRA *adaptation* literature. That range is appropriate for fine-tuning because pretrained weights already span a rich subspace and updates live in a much smaller intrinsic subspace. In the *pre-training* regime (idea 4.3), W=A·B must express the full gradient space from random initialization — there is no pretrained subspace to leverage, so the rank must be large enough to avoid bottlenecking learning capacity. Setting r=d/4 is [derived: weight bytes per matrix = 2×d×r×sizeof(bf16) vs d²×sizeof(bf16) for dense; ratio = 2r/d = 2×(d/4)/d = 0.5 → exactly 2× weight memory reduction; parameter count per factored layer = 2×d×(d/4) = d²/2 vs d² dense, preserving sufficient expressivity for quality parity (validated empirically by CoLA[6] at r≈d/4)]; it halves weight bytes while keeping the effective parameter count high enough to match dense training quality. These ranks should not be compared to adaptation-LoRA ranks without this context.

- At r=d/4, pure A·B weight bytes = 2 × d × (d/4) × 2 = d²/2 bytes per matrix (vs. d² for dense)

### Non-Uniform Rank Is Essential
WeLore[7] establishes that uniform rank reduction is catastrophic at the perplexity level (1836 vs 11.87 PPL at 50% uniform reduction on LLaMA-2 7B). Non-uniform rank assignment per weight type is required:
- Q/K projections: tolerate ≥90% rank reduction (low intrinsic rank in trained weights)
- V projections: resist compression (critical for sequence representation)
- MLP down-projections: resist compression

---

## 5. Implementation Considerations

- **What you gain (4.3-C):** ~2× weight memory reduction, ~1.64× TPOT throughput improvement, ~0.5× TTFT improvement (2× faster prefill), ~50% training optimizer state reduction.
- **Training-only benefit (4.3-A):** GaLore[3] provides equivalent training memory savings without committing weights to a low-rank inference structure — making 4.3-A's training benefit less compelling.
- **Rank allocation:** Non-uniform rank is essential — V-proj/MLP-down require ≥90% effective rank; Q/K tolerate aggressive reduction.
- **Optimizer tuning:** Apply LoRA+ A:B learning-rate ratio for 2× training speedup at equal compute.

---

## 6. Synergies

- **With GaLore / GaLore 2:** GaLore provides gradient-level low-rank projection with full-rank weights preserved; useful as a fallback for variant 4.3-A when inference-time weight compression is not required.
- **With LoRA+:** A:B learning-rate asymmetry complements any pre-training LoRA parameterization; 2× speedup at equal compute.
- **With SLTrain:** W = sparse + A·B (SLTrain) can be viewed as 4.3 augmented with a sparse residual — the sparse component supplies expressivity that pure A·B in variant 4.3-C must otherwise recover through nonlinearity.
- **With Idea 1.2 (early exit):** Per-layer TPOT gains compound only when a decode-time early-exit mechanism avoids re-loading shared weights across all L layers.

---

## 7. Risk Assessment

- **Quality cliff (uniform rank):** Catastrophic perplexity degradation on V-projection and MLP down-projections (WeLore: 1836 vs 11.87 PPL). Mitigation: non-uniform rank allocation per weight type.
- **Nuclear norm / rank collapse:** Weight decay on LoRA parameterization is equivalent to nuclear norm regularization, biasing toward rank collapse in attention layers. Mitigation: tune weight decay separately for A, B, and W_core.
- **Intruder dimensions:** LoRA-structured training may introduce singular vectors absent from full training, degrading multi-task robustness. Mitigation: post-hoc singular-value intervention.
- **Quality parity at scale (4.3-C):** Unconfirmed at 27B+ scale. The nonlinear bottleneck in CoLA may be load-bearing; removing it (as 4.3-C pure-linear requires) is untested beyond 7B.
- **Implementation effort:** LOW-MEDIUM — standard PyTorch parameterization; adapter tensor management for per-layer A/B.

---

## 8. Accuracy / Quality Tradeoff

- **Expected quality delta vs baseline**: CoLA achieves on-par perplexity (34.04 vs 34.06 at 60M; 15.52 vs 15.56 at 1B) with a nonlinear bottleneck at r=d/4 [CoLA, 6, arXiv:2502.10940]; uniform 50% rank reduction collapses catastrophically (1836.62 vs 11.87 PPL on LLaMA-2 7B) without non-uniform rank allocation [WeLore, 7, ICML 2025]; SLTrain with W = sparse + A·B achieves on-par quality with full-rank LLaMA 7B pretraining [SLTrain, 10, NeurIPS 2024]; LoRA+ enables 2× speedup at equal compute via A:B learning-rate ratio [LoRA+, 5, ICML 2024].
- **Known failure modes**: Quality cliff is concentrated in specific weight types — V-proj and MLP down-proj require ≥90% effective rank; Q/K tolerate aggressive reduction [WeLore, 7]; removing the CoLA nonlinear bottleneck (as 4.3-C pure linear requires) is untested above 7B parameters and may reintroduce the rank-collapse behavior LoRA+weight-decay is known to cause [LoRA Provable Convergence, 8, arXiv:2502.09376]; LoRA-structured training introduces high-ranking "intruder dimensions" absent from full training, degrading multi-task robustness [LoRA vs Full Fine-tuning, 9, arXiv:2410.21228]; 4.3-A and 4.3-B yield no inference benefit — only 4.3-C delivers TPOT/TTFT gains.
- **Empirical evidence**: CoLA §Results (PPL parity at 60M and 1B; 1.64× inference throughput; 1.67× inference memory reduction at r=d/4) [CoLA, 6]; WeLore Table (uniform 50% rank: 1836.62 vs 11.87 PPL; non-uniform: recovers) [WeLore, 7]; SLTrain §Results (73% memory reduction with quantization, on-par with LLaMA 7B) [SLTrain, 10]; Shuttleworth et al. §Analysis (intruder dimensions correctable via post-hoc singular-value intervention) [LoRA vs Full, 9].
- **Mitigations**: Use non-uniform rank — allocate higher r to V-proj/MLP-down, lower r to Q/K [WeLore, 7]; apply LoRA+ A:B learning-rate ratio for 2× training speedup [LoRA+, 5]; prefer CoLA's nonlinear bottleneck (σ between A and B) if pushing beyond 7B scale until pure-linear 4.3-C is validated at that scale; post-hoc correct intruder dimensions via singular-value intervention [LoRA vs Full, 9]; fall back to GaLore / GaLore 2 (gradient-level low-rank projection with full-rank weights) if inference compromise is unacceptable [GaLore, 3; GaLore 2, 4].

---

<!-- CITATION MANIFEST -->

[1] LoRA: Low-Rank Adaptation of Large Language Models: Hu et al. (ICLR 2022). arXiv:2106.09685. Foundational W=W_core+A·B structure for fine-tuning. Basis for 4.3's training-time parameterization.

[2] Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning: Aghajanyan et al. (ACL 2021). arXiv:2012.13255. Theoretical grounding: fine-tuning updates live in low-dimensional subspace. Motivated LoRA.

[3] GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection: Zhao et al. (ICML 2024). arXiv:2403.03507. Gradient-level low-rank projection preserving full-rank weights. Better training-memory option than 4.3-A without inference compromise.

[4] GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection: Su, Gu et al. (2025). arXiv:2504.20437. Extends GaLore to 7B scale, 500B tokens, full-rank weights throughout.

[5] LoRA+: Efficient Low Rank Adaptation of Large Models: Hayou, Ghosh, Yu (ICML 2024). arXiv:2402.12354. A:B learning rate ratio is critical. B requires higher LR by model-width-dependent ratio. 2× speedup at equal compute.

[6] CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation: Liu et al. (2025). arXiv:2502.10940. Parameterizes linear layers as h′=B·σ(A·x). Abstract reports 2× computing-cost reduction and 1.86× training throughput at full-rank parity; Tables 6/10 paper-body report 1.64× inference throughput and 1.67× inference memory reduction at r=d/4. Closest published realization of 4.3-C.

[7] From GaLore to WeLore: How Low-Rank Weights Non-uniformly Emerge from Low-Rank Gradients: Jaiswal et al. (ICML 2025). arXiv:2407.11239. Uniform 50% rank reduction: 1836.62 vs 11.87 PPL on LLaMA-2 7B. Non-uniform rank allocation essential — Q/K tolerate ≥90% effective rank reduction; V and MLP down-projections do not.

[8] LoRA Training Provably Converges to a Low-Rank Global Minimum or It Fails Loudly: (2025). arXiv:2502.09376. Training LoRA with weight decay is equivalent to nuclear norm regularization in full fine-tuning, strongly biasing toward low rank and risking rank collapse.

[9] LoRA vs Full Fine-tuning: An Illusion of Equivalence: Shuttleworth, Andreas, Torralba, Sharma (2024). arXiv:2410.21228. LoRA-structured training introduces new high-ranking singular vectors ("intruder dimensions") absent from full training, degrading multi-task robustness and pre-training distribution modeling. Partially correctable by post-hoc singular value intervention.

[10] SLTrain: A Sparse Plus Low-Rank Approach for Parameter and Memory Efficient Pretraining: Han et al. (NeurIPS 2024). arXiv:2406.02214. W = sparse + A·B from pretraining achieves quality on par with full-rank LLaMA 7B pre-training. Sparse residual provides the expressivity that pure A·B (4.3-C) must recover via nonlinear bottleneck. Up to 73% memory reduction with quantization.

[11] Training Neural Networks from Scratch with Parallel Low-Rank Adapters: Huh, Cheung, Bernstein, Isola, Agrawal (2024). arXiv:2402.16828. Bi-level optimization for pretraining with parallel low-rank heads, periodically merged. Competitive with standard pre-training on vision transformers. Extends LoRA-style pretraining to from-scratch training at scale.
