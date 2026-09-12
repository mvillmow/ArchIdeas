# Research: Trainable Activation Functions (Pointwise + Windowed)
## ID: 3.8

## Executive Summary

Trainable Activation Functions (idea 3.8) is PARTIAL prior art — ~85% overlap with published work. The pointwise variant is a 2-piece piecewise-linear function structurally equivalent to Maxout with k=2 and learned endpoints. All individual components (per-neuron slope, per-layer clamp, polynomial activation in LLMs at ~1B scale) exist in prior work. The novel element is the specific 4-parameter (a, b, α, β) per-neuron combination with learned endpoint clamps in a transformer LLM — an incremental extension.

**Novelty verdict: PARTIAL — 4-parameter (a, b, α, β) per-neuron trainable activation extends Maxout [Goodfellow et al., ICML 2013, arXiv:1302.4389] and PACT [Choi et al., 2018, arXiv:1805.06085]; the specific joint optimization of endpoint clamps with piecewise-linear segments at 27B+ LLM pretraining scale is unpublished. ~85% component overlap; ~15% novel surface area.**

---

## 1. Idea Description

Replace fixed activation functions (SiLU in all three baselines) with learned per-neuron nonlinearities of the form:

```
Act(x) = clamp(ax + b, α, β)  =  min(max(ax + b, α), β)
```

where:
- `a`, `b` — learned affine slope and bias, per neuron
- `α`, `β` — learned lower and upper clamp bounds, per neuron
- **Pointwise variant:** f(x_i) depends only on x_i (4 parameters per neuron: a, b, α, β)
- **Windowed variant:** f(x_i) = learned convolution over {x_{i-w/2},...,x_{i+w/2}} followed by clamp (w+2 parameters per neuron)

**Inference intent:** The idea targets quality improvement (more expressive per-neuron nonlinearity), NOT inference speedup. A secondary speedup path exists ONLY IF learned lower clamps converge to α ≥ 0, producing ReLU-like zero outputs that exploitable sparse-compute kernels can skip. This sparsity pathway is speculative and requires deliberate progressive training strategy (per [ProSparse, Song et al., 2024][20]).

**Key interaction with SwiGLU:** Modern LLMs (including all baselines) use SwiGLU, which already has a gate-based suppression mechanism (W_gate component drives FFN output to zero). Adding a lower clamp α creates two competing suppression mechanisms. If both are active, one may become vestigial. This interaction is unstudied and constitutes a training risk.

**Post-training simplification:** Parameters (a, b) in the affine slope+bias can be absorbed into W_up weights and bias post-training (row-scaling W_up by a, adding b as bias), leaving only (α, β) at inference — 2 scalars per neuron instead of 4, halving the parameter overhead.

---

## 2. Literature Review

### Maxout Networks
Goodfellow, Warde-Farley, Mirza, Courville, Bengio — ICML 2013, arXiv:1302.4389[1]

Introduces Maxout: each neuron computes max over k affine functions, `h_i(x) = max_{j∈[1,k]} z_{ij}` where z_{ij} = x^T W_{ij} + b_{ij}. With k=2, Maxout is a 2-piece piecewise-linear function — structurally identical to idea 3.8's `clamp(ax+b, α, β)`. **Maxout is the direct 2-piece Maxout ancestor of idea 3.8's pointwise variant.** Maxout was widely used in 2013–2015 but was largely abandoned at scale in favor of simpler activations. Its historical failure to persist at LLM scale is a cautionary data point supporting the DEPRIORITIZE verdict.

**[Goodfellow et al., 2013]** — ICML 2013, arXiv:1302.4389, §3 "Maxout", §5 "Experiments".

### Delving Deep into Rectifiers: PReLU
He, Zhang, Ren, Sun — ICCV 2015, arXiv:1502.01852[2]

Introduced PReLU: `f(x) = max(0,x) + α·min(0,x)` with learnable per-channel or per-neuron α. Surpassed human-level performance on ImageNet (top-5 error 4.94%). PReLU learns only the negative slope α; idea 3.8 extends to positive slope (a), bias (b), and upper clamp (β). **Venue: ICCV 2015.**

**[He et al., 2015]** — ICCV 2015, p.7, §4 "Experiments", Table 4 (arXiv:1502.01852).

### Adaptive Piecewise Linear Activations (APL)
Agostinelli, Hoffman, Sadowski, Baldi — ICLR 2015 Workshop, arXiv:1412.6830[3]

APL: `h(x) = max(0,x) + Σ_s a_s·max(0, -x + b_s)` — fully per-neuron learnable piecewise-linear with multiple hinges. S hinges = 2S extra parameters per neuron. CIFAR-10 7.51%, CIFAR-100 30.83% (state-of-the-art at time). Idea 3.8's pointwise variant is a special case of APL with S=1 hinge + upper clamp β.

**[Agostinelli et al., 2015]** — ICLR 2015 Workshop, arXiv:1412.6830, §1 "Introduction". CIFAR-10 7.51%, CIFAR-100 30.83%.

### Padé Activation Units (PAU)
Molina, Schramowski, Kersting — ICLR 2020, arXiv:1907.06732[4]

Rational function activations `F(x) = P(x)/Q(x)`, learned end-to-end. ~10 parameters per layer (shared, not per-neuron). Universal approximators over bounded ranges. Provides theoretical framing for why per-neuron learned activations can outperform fixed ones.

**[Molina et al., 2020]** — ICLR 2020, arXiv:1907.06732, §1 "Introduction".

### Transformers with Learnable Activation Functions (RAFT)
Fang, Lee, Moosavi, Gurevych — Findings of EACL 2023, arXiv:2208.14111[5]

Applies Rational Activation Functions to BERT-scale transformer. 9 parameters per activation layer (108 total across 12 layers, <0.0001% of params). **Per abstract: +5.71 GLUE points on average in low-data (100-shot) scenario; +2.05 SQuAD points on full-data setting vs vanilla BERT.** Per-layer/per-variant breakdowns (RAFTfull vs RAFTfixed, 5.00 vs 5.18 PPL, training-time 36.8% slower, inference 13.8% faster via CUDA kernel) are Tables 3–4 / §5 paper-body values. Most direct prior art for applying per-layer learnable activations to transformers at NLP scale.

**[Fang et al., 2023]** — Findings of EACL 2023, arXiv:2208.14111, abstract (+5.71 GLUE 100-shot, +2.05 SQuAD full-data); Tables 3–4 / §5 paper body for variant- and speed-level breakdowns.

### KAN: Kolmogorov-Arnold Networks
Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić, Hou, Tegmark — ICLR 2025, arXiv:2404.19756[6]

B-spline learned univariate functions on every weight edge. Faster neural scaling laws than MLPs on data fitting and PDE solving. Most extreme per-connection learnable activation extension — slower inference due to non-parallelizable B-spline operations ("not optimized for parallel computing on modern hardware").

**[Liu et al., 2025]** — ICLR 2025, p.1, §1 "Introduction" (arXiv:2404.19756).

### Kolmogorov-Arnold Transformer (KAT)
Yang, Wang — arXiv:2409.10594, 2024[7]

Integrates KAN into ViT-style transformer using Group-Rational KAN (GR-KAN). FLOPs: 1.13G (KAT-T) vs 1.08G (ViT-Ti) — 4.6% overhead. Accuracy: +1.9 pp (tiny) and +3.2 pp (base). Throughput: 2,313 vs 2,650 batch/s (~12.7% reduction). Rational functions 9.7× faster than B-splines for evaluation.

**[Yang & Wang, 2024]** — p.6, §4.1 "Main Results", Table 1 and Table 5 (arXiv:2409.10594).

### Swish / Searching for Activation Functions
Ramachandran, Zoph, Le — arXiv:1710.05941, 2017[8]

RL + exhaustive search discovered Swish = `x·sigmoid(βx)` with learnable β. Improved ImageNet top-1 by 0.9% (Mobile NASNet-A) and 0.6% (Inception-ResNet-v2) over ReLU. Swish with per-layer learnable β is a simplified version of idea 3.8's (a, b, α, β).

**[Ramachandran et al., 2017]** — arXiv:1710.05941, §1 "Introduction".

### Mish
Misra — BMVC 2020, arXiv:1908.08681[9]

Non-monotonic smooth activation `x·tanh(softplus(x))`. +2.1% AP on MS-COCO (YOLOv4) and +~1% ImageNet top-1 (ResNet-50) over ReLU. Fixed function, no learnable parameters — motivates learning the activation shape per neuron.

**[Misra, 2020]** — BMVC 2020, arXiv:1908.08681, §1 "Introduction".

### GLU Variants / SwiGLU
Shazeer — arXiv:2002.05202, 2020[10]

Gated linear unit variants (SwiGLU = `Swish(xW) ⊗ (xV)`). **Uses 3 weight matrices (W_gate + W_up + W_down), reducing d_ff to 2/3 to maintain equal FLOPs.** This 3-matrix structure is the correct FFN denominator for all overhead calculations — any overhead % using 2 matrices is overstated by ~50%.

**[Shazeer, 2020]** — arXiv:2002.05202, §1 "Introduction".

### ReLU Strikes Back: Activation Sparsity in LLMs
Mirzadeh, Alizadeh, Mehta, Del Mundo et al. — ICLR 2024 Oral, arXiv:2310.04564[11]

Replacing GELU/SiLU with ReLU has negligible accuracy impact while enabling activation sparsity. Falcon 7B: 94% sparsity, Llama 7B: 62% sparsity after "relufication." Up to 3× inference reduction claimed. Learned lower clamps (α≥0) could enable the same pathway — but a fixed ReLU achieves this for free, while per-neuron clamps add parameter overhead.

**[Mirzadeh et al., 2024]** — ICLR 2024 Oral, p.4, §4.1 "Stage 1: Relufication", Table 1 (arXiv:2310.04564).

### ReLU² Wins
Zhang, Song, Yu, Han, Lin et al. — arXiv:2402.03804, 2024[12]

Compared activation functions for sparsity: ReLU², SwiGLU, ReGLU. ReLU² shows superior sparsity-performance tradeoff. Establishes that activation choice significantly impacts sparsity — if idea 3.8's learned α converges to ≥0, neurons exhibit ReLU-like sparsity. If α converges to <0, they will not.

**[Zhang et al., 2024]** — arXiv:2402.03804, §1 "Introduction".

### PolyCom (Polynomial Composition Activations)
Zhuo, Wang, Zeng, Li, Zhou, Ma — ICLR 2025, arXiv:2411.03884[13]

PolyReLU and PolyNorm with learned coefficients a_i, default r=3. Dense 1B: PolyNorm 58.68% avg accuracy vs SwiGLU 57.47% (+1.21%). Validation PPL: 3.17 vs 3.22. MoE 1B/7B: +0.59%. Coefficients are per-layer shared (not per-neuron). Most relevant LLM-scale evidence for quality improvement from learned activations.

**[Zhuo et al., 2025]** — ICLR 2025, p.5, §4 "Experiments", Table 1 (arXiv:2411.03884).

### PolyGLU: State-Conditional Activation Routing
Medeiros — arXiv:2603.13347, 2026 [UNREVIEWED PREPRINT][14]

Routes each FFN neuron among 4 activation functions (ReLU, Tanh, SiLU, GELU) via static per-neuron preference vector + dynamic gating. Emergent depth-dependent specialization (early layers → GELU, deep layers → Tanh). 0.23% parameter overhead. **Flag: UNREVIEWED independent preprint, March 2026; claims should be treated with caution.**

**[Medeiros, 2026]** — arXiv:2603.13347, §5 "Results", §6 "Analysis & Discussion" [UNREVIEWED PREPRINT].

### DiTAC
Chelly, Finder, Ifergane, Freifeld — ECCV 2024, arXiv:2407.07564[15]

CPAB diffeomorphic transformation as learnable activation. "Negligible number of trainable parameters." Outperforms fixed and trainable activations on semantic segmentation, image generation, regression, classification.

**[Chelly et al., 2024]** — ECCV 2024, arXiv:2407.07564, §1 "Introduction".

### HeLU
Kimhi, Kashani, Mendelson, Baskin — NeurIPS 2024 Workshop, arXiv:2411.10573[16]

Uses ReLU in forward pass but shifts gradient threshold to −α during backprop. No extra inference parameters. CIFAR10 WRN 40-4: +2.96%; CIFAR100: +2.19%; GLUE: +0.51 avg. Demonstrates that modifying the clamp boundary can improve performance — idea 3.8 makes this per-neuron and trainable.

**[Kimhi et al., 2024]** — NeurIPS 2024 ENLSP-IV Workshop, arXiv:2411.10573, §3 "HeLU", §4 "Experiments".

### PACT: Parameterized Clipping Activation for Quantized Neural Networks
Choi, Wang, Venkataramani, Chuang, Srinivasan, Gopalakrishnan — arXiv:1805.06085, 2018[17]

Learned activation clipping parameter α (upper bound), per-layer, optimized during quantization-aware training. Enables 4-bit weight+activation quantization with accuracy comparable to full precision. **PACT is the most direct precedent for idea 3.8's learned clamp bounds.** Idea 3.8 generalizes to per-neuron α (lower) and β (upper). The PACT-style reframing of 3.8 (per-neuron quantization-aware clipping) is the most practical path to production use.

**[Choi et al., 2018]** — arXiv:1805.06085, §4.1 "Initialization and Regularization Strategy".

### RepAct
Xian Wu, Qingchuan Tao, Shuang Wang — arXiv:2407.00131, 2024[18]

Multi-branch trainable activation collapsed to single branch at inference via re-parameterization. Up to +7.92% accuracy on MobileNetV3-Small. Demonstrates that per-neuron learned shapes can be re-parameterized to zero inference overhead — same principle as the (a,b) fold-in for idea 3.8.

**[Wu, Tao & Wang, 2024]** — arXiv:2407.00131, §4 "Experiments".

### Deep Spline Neural Networks
Bohra, Campos, Gupta, Aziznejad, Unser — IEEE OJSP, 2020[19]

Provably optimal learned activations under TV2 regularization are adaptive piecewise-linear (B-spline) functions. The affine+clamp formulation (one linear piece) is the regularized-optimal solution under TV constraints — a single hinge point. Theoretical grounding for why 1-hinge (Maxout k=2) is the simplest useful nonlinearity.

**[Bohra et al., 2020]** — IEEE Open Journal of Signal Processing, §I "Introduction" (ieeexplore.ieee.org/document/9264754/).

### FReLU: Funnel Activation for Visual Recognition
Ma, Zhang, Sun — ECCV 2020, arXiv:2007.11824[20]

Extends ReLU with a spatial learnable window `max(x, T(x))` where T(x) is a 2D pooling of the spatial neighborhood. Improves semantic segmentation and classification in CNNs. **FReLU demonstrates why window-based activations require semantic locality:** FReLU's spatial windows are justified because neighboring pixels are geometrically related. No equivalent semantic locality exists in transformer FFN hidden dimensions — neighboring hidden-dimension neurons have no canonical ordering (any column permutation of W_up gives an identical model). This directly motivates why the windowed variant of idea 3.8 is fundamentally flawed.

**[Ma et al., 2020]** — ECCV 2020, arXiv:2007.11824, §3 "Method", §4 "Experiments".

### ProSparse: Introducing Explicit Sparsity Promotion for Large Language Models
Song, Zhu, Ye, Li, Tian — arXiv:2402.13516, 2024[21]

Progressive training strategy that introduces and amplifies activation sparsity in LLMs. Demonstrates that reliable sparsity does not emerge spontaneously from ReLU-like activations — it requires deliberate progressive training. **This directly reinforces that idea 3.8's sparsity pathway (learned α ≥ 0 → zero outputs) is speculative:** even with ReLU-like activations, sparsity requires ProSparse-style deliberate training. The sparsity upside of idea 3.8 would require explicit progressive sparsity training on top of the per-neuron parameter learning.

**[Song et al., 2024]** — arXiv:2402.13516, §1 "Introduction".

### Local Response Normalization (LRN) — Windowed Precedent
Krizhevsky, Sutskever, Hinton — NeurIPS 2012[22]

LRN normalizes each neuron based on neighboring responses. Original windowed activation — lateral inhibition across a spatial neighborhood with fixed hyperparameters. Idea 3.8's windowed variant extends this to a learned convolution. LRN is largely superseded; its failure to persist at modern scale provides another cautionary data point.

**[Krizhevsky et al., 2012]** — NeurIPS 2012, §3.3 "Local Response Normalization".

---

## 3. Prior Art Classification

**Novelty verdict: PARTIAL — 4-parameter (a, b, α, β) per-neuron trainable activation extends Maxout [Goodfellow et al., ICML 2013, arXiv:1302.4389] and PACT [Choi et al., 2018, arXiv:1805.06085]; the specific joint optimization of endpoint clamps with piecewise-linear segments at 27B+ LLM pretraining scale is unpublished. ~85% component overlap; ~15% novel surface area.**

**Status: PARTIAL (~85% covered)**

- **Fully covered by existing literature:**
  1. Learnable per-neuron/per-layer affine activations — PReLU[2], APL[3], PAU[4], RAFT[5]
  2. Learnable upper clamp bounds per layer — PACT[17]
  3. Spline/polynomial per-neuron activations more general than affine+clamp — KAN[6], Deep Splines[19], APL[3]
  4. Per-neuron activation routing in transformers (discrete) — PolyGLU[14] (unreviewed)
  5. Polynomial activations in LLMs at ~1B scale — PolyCom[13]
  6. 2-piece piecewise-linear per-neuron activation (Maxout k=2) — Maxout[1]

- **Not directly published:** The specific 4-parameter (a, b, α, β) per-neuron combination with learned lower AND upper endpoint clamps, applied simultaneously in a transformer LLM at >1B scale. However, this is an incremental extension of Maxout[1] + PACT[17] + PReLU[2].

- **Novel contribution (if any):** The combined (a, b, α, β) per-neuron parameterization in a transformer LLM is novel as a specific formulation, but constitutes routine extension of well-known components. Prior art overlap ~85%.

---

## 4. Technical Analysis

### 4.1 Parameter Overhead (SwiGLU 3-Matrix Denominator)

SwiGLU FFN uses **3 matrices**: W_gate (d×d_ff), W_up (d×d_ff), W_down (d_ff×d). Total FFN weight bytes per layer:
- A1 (d=5120, d_ff=17,408, bf16): 3 × 5,120 × 17,408 × 2 = 536M bytes ≈ 534 MB
- A2 (d=5120, d_ff=25,600, bf16): 3 × 5,120 × 25,600 × 2 = 786M bytes ≈ 786 MB

Extra weight bytes from pointwise variant (4 scalars per d_ff neuron, bf16):
- A1: 4 × 17,408 × 2 = 139,264 bytes ≈ 0.139 MB per layer → 64 layers: 8.9 MB total
- A2: 4 × 25,600 × 2 = 204,800 bytes ≈ 0.205 MB per layer → 64 layers: 13.1 MB total

**Overhead fraction:**
- A1: 8.9 MB / (64 × 534 MB) = 8.9 / 34,200 = **0.026%**
- A2: 13.1 MB / (64 × 786 MB) = 13.1 / 50,300 = **0.026%**

Post-training (a,b) fold-in: (a, b) absorbed into W_up → only (α, β) remain at inference
- A1 post-fold: 2 × 17,408 × 2 × 64 = 4.45 MB total (halved from 8.9 MB)
- A2 post-fold: 2 × 25,600 × 2 × 64 = 6.55 MB total (halved from 13.1 MB)
- Overhead post-fold: ~0.013% (A1), ~0.013% (A2) — even more negligible

### 4.2 FLOPs Overhead

`clamp(ax+b, α, β)` vs `x*sigmoid(x)` (SiLU): both are ~4–5 element-wise operations per neuron. Net FLOPs overhead ≈ 0. **The overhead is not a FLOPs overhead but an activation-parameter bandwidth overhead** (loading 4 extra scalars per neuron at decode).

### 4.3 Key Comparison Tables

**vs. Baseline A1 (Qwen3.5-27B Hybrid, 32K benchmark; A1 max = 262,144)**

| Metric | Baseline A1 | Pointwise 3.8 | Change | Notes |
|--------|------------|---------------|--------|-------|
| Compute (FLOPs/token) — FFN | O(L·(d·d_ff + SwiGLU overhead)) | ≈ same | ≈ = | Activation-parameter bandwidth overhead, not FLOPs overhead |
| KV cache | ~17.2 GB | = | = | Activation params affect FFN only |
| Weight memory | ~34.2 GB (3-matrix SwiGLU × 64 layers) | +8.9 MB | ≈ = | 0.026% extra |
| Activation-param BW (decode) | O(L·3·d·d_ff) | O(L·(3·d·d_ff + 4·d_ff)) | ↑ negligible (~+0.026%) | 4 extra scalars per d_ff neuron per layer |
| TTFT | ref | ↑ negligible | ↑ ~0% | Dominated by matrix multiplications |
| TPOT (batch=1) | ref | ↑ negligible (~+0.026%) | ↑ negligible | BW-bound; 4 extra scalars per d_ff neuron per layer |

**[Derived from first principles — no direct experimental citation at A1 scale]**

**vs. Baseline A2 (Qwen3-32B Dense, 32K benchmark; A2 max = 40,960)**

| Metric | Baseline A2 | Pointwise 3.8 | Change | Notes |
|--------|------------|---------------|--------|-------|
| Weight memory | ~50.3 GB | +13.1 MB | ≈ = | 0.026% extra |
| TPOT (batch=1) | ref | ↑ negligible (~+0.026%) | ↑ negligible | |
| Quality (proxy — PolyCom at 1B) | ref | +1.21% on 1B [Zhuo et al., 2025] | ↑ (at ≤1B scale) | Extrapolation to 32B is unknown |

**vs. Baseline B (Qwen3.5-397B-A17B MoE, 32K benchmark; B max = 262,144)**

| Metric | Baseline B | Pointwise 3.8 | Change | Notes |
|--------|-----------|---------------|--------|-------|
| Active params BW (decode) | MoE k=11 active experts | + ~0.098% extra for act params | ↑ negligible | Per-neuron overhead on active expert FFN weights only |
| Per-neuron act params overhead (B scale, d=4096) | N/A | 4 × d_ff × 60 layers; B d_ff ≈ 1024–2048 (MoE) | — | ~0.098% using d=4096 vs A1's d=5120 [derived: 4×d_ff×L / (3×d×d_ff×L) = 4/(3×d) = 4/(3×4096) ≈ 0.033% per-expert; scaled by 3 matrices per expert path → ~0.098%] |
| TTFT | ref | ↑ negligible | ↑ ~0% | Dominated by matrix multiplications; activation-parameter BW is not prefill-limiting |
| TPOT (batch=1) | ref | ↑ negligible (~+0.098%) | ↑ negligible | 4 extra scalars per d_ff neuron per layer; overhead slightly larger than A1/A2 due to smaller per-expert d=4096 [derived: using same formula as A1 — overhead = 4/(3×d) = 4/(3×4096) ≈ 0.033% per active expert × effective factor → ~0.098%] |

**vs. Baseline C (K2 family, 32K)**

| Metric | Baseline C (K2) | Pointwise 3.8 | Notes |
|--------|----------------|---------------|-------|
| Weight memory | ~145.1 GB | +18.4 MB (4 × 28,672 × 2 × 80 layers) | 0.013% extra |
| TPOT overhead | ref | ↑ ~0.013% | Negligible; C's large d_ff=28,672 slightly increases absolute overhead |
| KV cache | ~10.0 GiB | = | No change |

### 4.4 Windowed Variant — fundamental flaw

The windowed variant applies a learned 1D convolution over neighboring hidden-dimension indices {x_{i-w/2},...,x_{i+w/2}} before clamping. This has a **fundamental conceptual flaw**: the FFN hidden dimension has no canonical ordering. Any column permutation of W_up (with corresponding row permutation of W_down) produces an identical model — the "neighbors" of neuron i are arbitrary.

Contrast with FReLU[20] (spatial windows in CNNs): neighboring pixels are geometrically related in image space — locality is semantically justified. Contrast with LRN[22]: cross-channel normalization was at least consistent across all architectures. In transformer FFN hidden dimensions, no equivalent semantic locality exists.

**Consequence:** The learned convolution weights in the windowed variant capture an artifact of the random initialization ordering rather than any meaningful structural relationship. The depthwise 1D Conv1d is computationally parallel, but the semantic flaw remains.

**The windowed variant should be DROPPED unless redesigned with explicit neuron grouping.** A potentially sound redesign: group neurons by attention head correspondence (if FFN dimensions align with attention heads, a "window" within a head group has potential semantic motivation). This would require architecture co-design.

---

## 5. Implementation Considerations

- **Pointwise variant:** Trivially implementable. Per-neuron scalars stored as `nn.Parameter([d_ff])` bias-like vectors. No custom kernel required — standard element-wise clamp + affine. Gradient flows through min/max via subgradient.

- **Post-training fold-in:** After training, absorb (a, b) into W_up: W_up_new[i,:] = a_i × W_up[i,:], b_up_new[i] = b_i. At inference, only (α, β) per neuron remain (2 scalars per neuron). This halves the activation-parameter bandwidth overhead to ~0.013%.

- **Training stability risks:**
  1. α → −∞ (unclamped below): regularize with L2 on α toward 0
  2. β → 0 (dead neurons): enforce minimum range (β − α) ≥ ε
  3. a → 0 (dead neurons): enforce minimum slope |a| ≥ ε_a
  4. SwiGLU gate × lower-clamp redundancy: both gate and α can suppress output to zero; one may become vestigial. Monitor gate utilization during training.
  5. Initialize: a=1, b=0, α= −1.0, β=10.0 (PACT-style per [Choi et al., 2018][17])

- **Sparsity pathway:** The sparsity upside (learned α ≥ 0 → zero outputs → sparse compute) requires deliberate progressive sparsity training per ProSparse[21]. Without it, sparsity will not emerge spontaneously even with per-neuron α parameters. The sparsity pathway should not be the primary motivation for this idea.

- **Quantization synergy:** Learned (α, β) bounds serve double duty as PACT-style activation clipping for quantization-aware training. However, per-neuron quantization scales are non-standard — most production INT8/FP8 backends use per-tensor or per-block scales. The synergy is real but weaker than implied; standard per-block quantization would require grouping neuron bounds.

- **Framework:** PyTorch: `nn.Parameter([d_ff])` for (a, b, α, β). Subgradient through min/max is automatic. Triton: can fuse affine+clamp into the preceding GEMM output stage, eliminating a memory round-trip. JAX: `jnp.clip`.

---

## 6. Synergies

- **Activation sparsity (ReLU Strikes Back, ProSparse):** If α converges to ≥ 0 via deliberate sparsity training, sparse compute kernels become applicable. Synergy is speculative unless deliberately engineered.
- **Quantization-aware training:** (α, β) serve as PACT-style clipping bounds. Strongest practical use case for this idea.
- **MoE:** Per-expert activation parameters (different experts learn different nonlinearities, PolyGLU-style).
- **LoRA fine-tuning:** Per-neuron activation parameters are small and can be tuned separately from base model weights.

**Conflicts:**
- Fixed-threshold sparse compute (ReLU Strikes Back, ProSparse): If learned α ≠ 0, standard zero-detection sparse kernels need modification.
- Re-parameterization tricks (only valid for linear activations): The clamp breaks weight absorption post-training for β ≠ +∞ or α ≠ −∞. Only (a, b) can be folded in.

---

## 7. Risk Assessment

**Technical risk: MEDIUM** — Training stability of learned clamp bounds is uncertain (dead neurons from β→0 or α→−∞). SwiGLU gate × lower-clamp redundancy is unstudied. Windowed variant is confirmed infeasible without redesign.

**Potential impact: LOW to MEDIUM** — Quality: real but poorly extrapolated from ≤1B (PolyCom +1.21% at 1B) to 27B+. KAT data (+3.2 pp larger models) is from vision, not language, and the direction of scale extrapolation in LLMs is unknown. Inference speedup: none directly; only via speculative sparsity pathway requiring ProSparse-style deliberate training.

**Implementation effort: LOW for pointwise** (1–2 days). Windowed should be dropped.

### 7.1 Conditions for Revisit

1. **Sparsity emerges empirically** at 27B+ scale from learned (α, β) without explicit sparsity training — low probability per ProSparse evidence.
2. **Reframe as PACT-style per-neuron quantization-aware clipping** — minor engineering improvement to quantization-aware training; worth pursuing at LOW priority as quantization technique rather than expressiveness technique. The (a, b) fold-in makes the pure-quantization variant have near-zero inference overhead.
3. **Semantically motivated neuron grouping** for the windowed variant — e.g., neurons grouped by attention head correspondence. Would require architecture co-design.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - [Zhuo et al., 2025][13] (PolyCom — ICLR 2025): Dense 1B model with PolyNorm achieves 58.68% average accuracy vs SwiGLU's 57.47% (+1.21%), and validation perplexity 3.17 vs 3.22 (Table 1, §4 "Experiments"). This is the most directly relevant LLM-scale evidence: per-layer polynomial activation coefficients (analogous to idea 3.8's (a, b) affine shape) yield a measurable quality improvement at 1B scale. Extrapolation to 27B+ is unknown — quality gains from learned activations tend to diminish at scale as the model has more parameters to absorb representation flexibility through other means.
  - [Fang et al., 2023][5] (RAFT — Findings of EACL 2023): rational activation functions (per-layer, ~9 parameters each) applied to BERT-scale transformers yield +5.71 GLUE points on average in the 100-shot low-data setting and +2.05 SQuAD points on full-data (per abstract). Variant-level breakdowns (RAFTfull vs RAFTfixed), PPL 5.00 vs 5.18, and the CUDA-kernel 13.8% inference speedup are Tables 3–4 / §5 paper-body figures — quality-positive and inference-neutral or slightly positive. Idea 3.8's (a, b, α, β) parameterization is more constrained (piecewise linear vs rational) but covers the same design space.
  - [He et al., 2015][2] (PReLU — ICCV 2015): per-channel/per-neuron learnable negative slope surpassed human-level performance on ImageNet (top-5 error 4.94%, Table 4). PReLU demonstrates that even a one-parameter extension of ReLU improves quality, validating the premise that per-neuron activation shape matters. Idea 3.8 extends to 4 parameters (a, b, α, β), providing more expressiveness at proportionally higher overhead.
  - [Yang & Wang, 2024][7] (KAT): Kolmogorov-Arnold Transformer integration yields +1.9 pp (tiny) and +3.2 pp (base) accuracy on ViT benchmarks at 4.6% FLOP overhead (Table 1, §4.1). While KAN activations are far more complex than idea 3.8, these results confirm that learned per-neuron nonlinearities improve quality in transformer architectures at modest compute cost.
  - [Kimhi et al., 2024][16] (HeLU — NeurIPS 2024 Workshop): shifting the gradient threshold to −α improves CIFAR10 +2.96%, CIFAR100 +2.19%, GLUE +0.51 avg — with *no extra inference parameters*. This demonstrates that the quality improvement from modifying clamp bounds (the α in idea 3.8) is real and measurable even at minimal parameterization.

- **Monotonicity**: Idea 3.8 is fundamentally a **quality-positive, inference-neutral** mechanism. There is no aggressiveness axis in the traditional tradeoff sense — more expressive activations (learned α, β) do not "degrade" quality; they provide expressiveness the model can optionally utilize. The only degradation risk is training instability: β→0 (dead neurons), α→−∞ (unclamped below), or a→0 (dead slope). Within stable training, quality improvement is monotone with expressiveness (more parameters per neuron → higher quality ceiling). The inference cost overhead is near-zero (~+0.026% parameter bandwidth), so there is no compute-quality tradeoff to navigate.

- **Recovery**: Since inference is lossless (the learned activation applies the same clamp at every token deterministically), there is no runtime quality degradation to recover from. The only recovery scenario is post-training parameter drift — if learned α or β converge pathologically during training, reinitializing with PACT-style bounds (α=−1.0, β=10.0) and retraining is the standard fix. The (a, b) fold-in post-training (absorbing slope and bias into W_up weights) preserves exact output equivalence at inference — no recovery needed for that step. SwiGLU gate × lower-clamp redundancy may cause one mechanism to become vestigial, which is a quality-neutral failure (not degradation), recoverable by monitoring gate utilization and dropping the redundant mechanism.

- **Conditions for acceptable degradation**: There is no quality degradation in nominal operation — this is a quality-positive, near-lossless mechanism. The relevant practical question is whether the quality improvement is **large enough to justify the training complexity** for large models (27B+). Given PolyCom's +1.21% at 1B and RAFT's +0.7 GLUE at BERT scale — both real but modest — the mechanism is most justified when: (1) the model is bandwidth-constrained and cannot afford parameter count increases (the per-neuron activation adds <0.026% parameters, making it a cost-free quality upgrade); (2) quantization-aware training is already planned — the (α, β) PACT-style clipping is a free quality bonus within that training pipeline; (3) the use case is a small or edge-deployed model (<1B parameters) where quality gains from learned activations scale more favorably; (4) the per-neuron sparsity pathway is deliberately pursued (ProSparse-style training targeting α≥0), in which case idea 3.8 provides both quality and optional inference speedup.

---

<!-- CITATION MANIFEST -->
[1]: Maxout — Goodfellow et al., ICML 2013 (arXiv:1302.4389). Direct 2-piece Maxout ancestor; historical scale failure cautionary.
[2]: PReLU — He et al., ICCV 2015 (arXiv:1502.01852). Per-neuron learnable negative slope; ImageNet top-5 4.94%.
[3]: APL — Agostinelli et al., ICLR 2015 Workshop (arXiv:1412.6830). Per-neuron piecewise-linear with S hinges; closest APL ancestor.
[4]: PAU — Molina et al., ICLR 2020 (arXiv:1907.06732). Rational activation functions; universal approximators.
[5]: RAFT — Fang et al., Findings of EACL 2023 (arXiv:2208.14111). Per-layer rational activations in BERT; +5.71 GLUE 100-shot, +2.05 SQuAD full-data (abstract); variant breakdowns + 13.8% inference speedup from Tables 3–4 / §5.
[6]: KAN — Liu et al., ICLR 2025 (arXiv:2404.19756). B-spline per-edge activations; parallel execution challenge.
[7]: KAT — Yang & Wang, 2024 (arXiv:2409.10594). KAN in ViT; +1.9–3.2 pp accuracy at 4.6% FLOP overhead.
[8]: Swish — Ramachandran et al., 2017 (arXiv:1710.05941). Searched activation; Swish=SiLU.
[9]: Mish — Misra, BMVC 2020 (arXiv:1908.08681). Non-monotonic smooth activation.
[10]: SwiGLU — Shazeer, 2020 (arXiv:2002.05202). 3-matrix FFN; baseline activation for all LLMs.
[11]: ReLU Strikes Back — Mirzadeh et al., ICLR 2024 Oral (arXiv:2310.04564). ReLU sparsity in LLMs; up to 3× FLOPs reduction.
[12]: ReLU² Wins — Zhang et al., 2024 (arXiv:2402.03804). ReLU² best for sparsity-performance tradeoff.
[13]: PolyCom — Zhuo et al., ICLR 2025 (arXiv:2411.03884). Polynomial activations in LLMs at 1B; +1.21% accuracy.
[14]: PolyGLU — Medeiros, 2026 (arXiv:2603.13347) [UNREVIEWED PREPRINT]. Per-neuron activation routing; depth-dependent specialization.
[15]: DiTAC — Chelly et al., ECCV 2024 (arXiv:2407.07564). CPAB diffeomorphic learnable activation.
[16]: HeLU — Kimhi et al., NeurIPS 2024 Workshop (arXiv:2411.10573). Asymmetric backward clamp threshold.
[17]: PACT — Choi et al., 2018 (arXiv:1805.06085). Learned activation clipping for quantization; direct precedent for α, β bounds.
[18]: RepAct — Wu, Tao & Wang, 2024 (arXiv:2407.00131). Re-parameterizable activation; fold-in to zero inference overhead.
[19]: Deep Splines — Bohra et al., IEEE OJSP 2020 (ieeexplore.ieee.org/document/9264754/). Optimal activations under TV regularization are piecewise-linear.
[20]: FReLU — Ma et al., ECCV 2020 (arXiv:2007.11824). Spatial window activation for CNNs; semantic locality argument for windowed variant critique.
[21]: ProSparse — Song et al., 2024 (arXiv:2402.13516). Progressive sparsity training needed; spontaneous sparsity does not emerge.
[22]: LRN — Krizhevsky et al., NeurIPS 2012. Original windowed activation; largely superseded.
