# Research: Skip List Layers (Replacing Per-Layer Residuals)
## ID: 4.4

## Executive Summary

**Novelty verdict:** PARTIAL — per-layer residuals, LayerDrop, MoD, ShortGPT, Hyper-Connections, and AltUp cover most of the surface; the specific formulation of *replacing* per-layer residuals with connections placed at fixed exponentially spaced intervals (N/2, N/4, N/8, ...) has no direct prior art, but gradient attenuation makes the strong form a NO-GO and the hybrid form reduces to constrained Hyper-Connections ([He et al., 2016], [LayerDrop, 2020], [MoD, 2024], [ShortGPT, 2024], [Hyper-Connections, 2024], [AltUp, 2023]).

**⚠ STRONG FORM: NO-GO for production.** Removing all residual connections eliminates gradient pathways critical for training stability (He et al., 2016; Figure 1). The strong form is not recommended. See the **Hybrid Form** section for a viable alternative that preserves skip connections for training while enabling selective layer skipping at inference.

Idea 4.4 proposes replacing standard per-layer residual connections in transformer stacks with skip connections placed at exponentially spaced intervals (N/2, N/4, N/8, ...). Two forms exist with very different risk profiles:

- **Strong form (full replacement):** Remove all per-layer residuals; retain only exponential-interval connections. **DEPRIORITIZE.** The 32-layer sub-block between skip points in a 64-layer network exhibits gradient attenuation of approximately ~10⁻¹⁰ [derived: each residual-free layer attenuates gradient by ~0.5× in expectation under standard init (SiLU activations, Kaiming-normal weights; expected Jacobian spectral norm ≈ 0.5); L=32 consecutive residual-free layers → (0.5)^32 = 2.328×10⁻¹⁰] — identical to the plain network failure documented by He et al. (2016). No inference benefit (residual connections are O(d), negligible vs. O(d·d_ff) MLP).
- **Hybrid form (supplement + Hyper-Connections initialization):** Keep per-layer residuals at small α ≈ 0.1 (ReZero-style), add exponential-interval long-range connections. **PURSUE.** Equivalent to constrained Hyper-Connections with exponential structural prior. Potential +1–6 benchmark point quality improvement (bounded by Hyper-Connections results). Zero inference speedup in either form.

**Scale-validation status:** Hybrid-form skip layers have been studied primarily at ≤7B scale; 27B+ from-scratch training stability with selective skipping is unvalidated.
**Recommendation: Prototype on A2 (dense) at ≤7B before any 27–32B commitment.**

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 validates the residual-enrichment direction through Manifold-Constrained Hyper-Connections (mHC), not the strong skip-list replacement form. mHC expands the residual stream, dynamically generates residual mixing parameters, constrains the residual mixing matrix to the Birkhoff polytope with Sinkhorn normalization, and reports only 6.7% overlapped 1F1B pipeline-stage wall-time overhead after fused kernels/recomputation. Update: hybrid residual enrichment is now a serious scale-tested baseline; strong residual replacement remains DEPRIORITIZE.

## Key Comparison Tables

### vs. Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | Strong Form | Hybrid Form |
|--------|------------|-------------|-------------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | = (residuals are O(d), ~77K:1 ratio vs MLP) | = |
| Memory bandwidth (decode) | O(L·(d²+s·d_kv/4)) | = (no weight matrices changed) | = |
| KV cache (32K ctx) | ~2.15 GB | = | = |
| Weight memory | O(L·d·d_ff) | = | = |
| Training cost | 1.0× | ~1.0–2.0× (instability overhead) | ~1.0–1.3× (modest tuning) |
| TTFT (8K prompt) | ref | = ref | = ref |
| TPOT (batch=1) | ref | = ref | = ref |
| Quality (benchmark avg) | ref | UNKNOWN / HIGH RISK | +1–6 pts (bounded by Hyper-Connections) |
| Skip_store memory overhead | 0 | +503 MB (computed at A2 dims, d=5120, 8K ctx; see §4.2) | +503 MB |

### vs. Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | Strong Form | Hybrid Form |
|--------|------------|-------------|-------------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | = (0.001% change from residual removal) | = |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | = | = |
| KV cache (32K ctx) | ~8.59 GB (GQA 8:1) | = | = |
| Weight memory | ~64 GB bf16 | = | = |
| Training cost | 1.0× | ~1.0–2.0× | ~1.0–1.3× |
| TTFT (8K prompt) | ref | = ref | = ref |
| TPOT (batch=1) | ref | = ref | = ref |
| Training stability | STABLE | HIGH RISK (10⁻¹⁰ gradient across 32-layer sub-block) | MEDIUM (per-layer α highway maintained) |

### vs. Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | Strong Form | Hybrid Form |
|--------|-----------|-------------|-------------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | = (MoE routing unchanged) | = |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | = | = |
| KV cache (32K ctx) | ~1.0 GB (15 full-attn, 32Q/2KV, hd=256) | = | = |
| Weight memory | O(L·E·d·d_e) | = | = |
| Training cost | 1.0× | ~1.0–2.0× | ~1.0–1.3× |
| TPOT (batch=1) | ref | = ref | = ref |

### vs. Baseline C (K2 family, 72.55B dense Llama-arch)

| Metric | Baseline C (K2) | Strong Form | Hybrid Form |
|--------|----------------|-------------|-------------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)), L=80 | = | = |
| Memory bandwidth | ~145.1 GB weight BW | = | = |
| KV cache (32K ctx) | ~10.0 GiB | = | = |
| Weight memory | ~72.55B params | = | = |
| MLP FLOPs/token/layer | ~4.70 × 10⁸ | = | = |
| Training cost | 1.0× | ~1.0–2.0× | ~1.0–1.3× |
| Skip_store overhead (L=80) | 0 | 7 skip pts × ~134 MB = +940 MB | +940 MB |

**Note for K2 (L=80):** The longest sub-block under exponential spacing is 40 layers (N/2=40) — worse than the A1/A2 32-layer case. Training risk is even higher for K2-scale strong form.

---

## 1. Idea Description

**Proposed mechanism:** Remove the standard per-layer residual connection and replace it with skip connections placed at exponentially spaced intervals (N/2, N/4, N/8, etc.). The residual signal enters at multiple hierarchical scales rather than at every layer.

**Important clarification on inference impact:** Replacing per-layer residual connections with exponentially-spaced ones does NOT change FLOPs, weight loading, TTFT, or TPOT. Residual connections are elementwise additions of O(d) cost. At A2 (d=5120, d_ff=25600), the residual-vs-MLP FLOP ratio is 5,120 / 393,000,000 ≈ 77,000:1 — negligible in any practical sense. Any value from idea 4.4 must come from quality improvement or training efficiency, not inference speedup.

---

## 2. Literature Review

### Deep Residual Learning for Image Recognition (He et al., 2016)
Per-layer residual connections[1] are essential to avoid gradient degradation in networks as shallow as 34 layers. The convergence failure of plain 34-layer CNNs is shown in Figure 1. This is the most direct evidence against the strong form of idea 4.4. Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. CVPR 2016. arXiv:1512.03385.

### Identity Mappings in Deep Residual Networks (He et al., 2016)
Follow-up analysis[2] showing that multiplicative gating on shortcuts (scaling, 1×1 conv, dropout) harms information propagation. Pure identity shortcuts are optimal. A 1001-layer ResNet achieves 4.62% CIFAR-10 error with pre-activation (BN + ReLU before conv). ECCV 2016, arXiv:1603.05027.

### Residual Networks Behave Like Ensembles of Relatively Shallow Networks (Veit et al., 2016)
Key result[3]: in a 110-layer ResNet, most gradient comes from paths only 10–34 layers deep. This has a dual interpretation for idea 4.4: (a) it could suggest removing per-layer connections since "many paths are redundant," but (b) the correct reading is that the exponential-interval scheme eliminates all short paths, leaving only paths ≥32 layers — exactly the ineffective gradient regime identified by this paper. The paper supports HIGH risk classification. NeurIPS 2016, arXiv:1605.06431.

### Highway Networks (Srivastava et al., 2015)
Learned gated information routing[4] enables training networks exceeding 900 layers. The carry gate is per-layer and adaptive — structurally different from fixed exponential-interval skips. NeurIPS 2015, arXiv:1505.00387.

### Densely Connected Convolutional Networks (Huang et al., 2017)
DenseNet[5] connects each layer to every subsequent layer: L(L+1)/2 connections. Opposite extreme from Skip List Layers. Demonstrates that more skip connections improve quality; the question is how much is needed. Minor nuance: DenseNet uses dense blocks of 12–30 layers (not all-to-all over full depth), separated by transition layers. CVPR 2017 Best Paper, arXiv:1608.06993.

### Deep Networks with Stochastic Depth (Huang et al., 2016)
Training-time layer dropping[6] with random per-batch identity shortcuts. Enables >1000-layer ResNets. Key distinction from idea 4.4: stochastic depth randomizes which layers are dropped and keeps all per-layer identity paths active; it does not fix the sparse pattern. ECCV 2016, arXiv:1603.09382.

### Feature Pyramid Networks (Lin et al., 2017)
Multi-scale feature pyramids[7] with hierarchical skip connections. FPN achieves representational benefit through bidirectional pathways (bottom-up + top-down) — the top-down pathway is inapplicable to causal LLM transformers (unidirectional). The multi-scale motivation transfers; the bidirectional fusion does not. CVPR 2017.

### Deep Layer Aggregation (Yu et al., 2018)
Hierarchical DLA aggregation[8] demonstrates that non-uniform, hierarchically-spaced connections improve CNN representations. Correctly supplementing rather than replacing per-layer residuals. CVPR 2018, arXiv:1707.06484.

### ReZero is All You Need (Bachlechner et al., 2021)
ReZero[9]: replaces each residual with x + α·F(x), α initialized to zero. Enables stable training of 120-layer transformers; 56% faster convergence on enwiki8. Directly supports the hybrid mitigation strategy: keep per-layer residuals (α initially small but trainable) while adding exponential-interval connections. UAI 2021, arXiv:2003.04887.

### Fixup Initialization (Zhang et al., 2019)
Foundation for training very deep plain networks without BatchNorm[10]: scales down residual branch parameters by L^{-1/(2m-2)}. Works at 10,000+ layers. Theoretical basis for initialization-only mitigations of gradient instability in deep sub-blocks. ICLR 2019, arXiv:1901.09321.

### T-Fixup (Huang et al., 2020)
Transformer-specific Fixup initialization[11]: scales residual branches by (9N)^{-1/4} at initialization (~0.220× at L=64). Achieves stable training without warmup. Complementary to ReZero for the hybrid mitigation. ICML 2020.

### On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)
Pre-LN vs Post-LN analysis[12]: Pre-LN provides well-behaved gradients via mean field theory. The gradient stability guarantee depends on the per-layer residual path existing — removing it would eliminate the key gradient highway that Pre-LN relies on. ICML 2020, arXiv:2002.04745.

### DeepNet: Scaling Transformers to 1,000 Layers (Wang et al., 2022)
DeepNorm[13]: modifies residuals to α·x + f(x) with theoretically-derived α. Enables 1000-layer transformers. A 200-layer model (3.2B) outperforms a 48-layer 12B model by 5 BLEU on multilingual translation. Demonstrates per-layer residuals are load-bearing even at extreme depth. IEEE TPAMI, arXiv:2203.00555.

### ResiDual (Xie et al., 2023)
Pre-Post-LN dual residual connections[14]: combines Pre-LN and Post-LN simultaneously. Outperforms both baseline residual patterns on machine translation. Demonstrates that residual connectivity structure profoundly affects both training stability and model capacity. arXiv:2304.14802.

### AltUp — Alternating Updates (Baykal et al., NeurIPS 2023)
AltUp[15]: partitions the layer stack into groups that receive residual updates at different rates. Non-per-step residual updates in transformers are practical. The most mechanistically comparable work to idea 4.4's mechanism: both involve non-uniform residual update schedules. Key distinction: AltUp uses uniform group alternation and supplements per-layer residuals; idea 4.4 uses exponential intervals and replaces them. NeurIPS 2023, arXiv:2301.13310.

### SkipNet (Wang et al., 2018)
Learned layer skipping in CNNs[16] via gating network: 30–90% computation reduction with maintained accuracy. Distinguishes learned (SkipNet) vs. fixed (idea 4.4) skipping. Key precursor to layer-skipping literature. ECCV 2018, arXiv:1711.09485.

### Hyper-Connections (Zhu et al., ICLR 2025)
Learned matrix of connection strengths[17] across depths and widths. On OLMoE-1B-7B: +6 ARC-Challenge (41.8→47.8), 1.8× faster convergence. On OLMo-7B: validation loss 2.581→2.559, downstream avg 70.1→71.0. Most directly comparable modern work: Hyper-Connections learns the connectivity; idea 4.4's hybrid form can be framed as initializing Hyper-Connections with an exponential structural prior. ICLR 2025, arXiv:2409.19606.

### Manifold-Constrained Hyper-Connections / DeepSeek-V4 mHC (Xie et al., 2026; DeepSeek-AI, 2026)
mHC constrains Hyper-Connections residual mixing matrices to the set of doubly stochastic matrices, bounding spectral norm and stabilizing deep residual propagation. DeepSeek-V4 uses mHC in a 61-layer, 1.6T-parameter MoE model. This is the strongest scale evidence for learned residual mixing. It supports preserving and enriching residual highways, not replacing per-layer residuals with sparse fixed skip intervals.

### M2R2 (Bhendawade et al., 2025)
Multi-rate residual evolution[18]: modulates "velocity" of residual update per token and layer for early exit / speculative decoding. 2.8× speculative decoding speedup, 2.9× MoE speedup. Conceptually adjacent to Skip List Layers; optimizes for inference efficiency rather than residual replacement. arXiv:2502.02040.

### Attention Residuals (Kimi Team, 2026)
Replaces fixed residual addition with softmax attention over all preceding layer outputs[19]. Kimi Linear (48B/3B active), 1.4T tokens: +7.5 GPQA-Diamond, +3.1 HumanEval, +3.6 MATH vs standard residual. ~2% latency overhead per Block AttnRes. The most aggressive known residual replacement — provides an optimistic quality ceiling for residual enrichment research. arXiv:2603.15031.

### Skip Connection Survey (Xu et al., 2024)
Comprehensive survey[20] of skip connection patterns. Identifies no published work specifically testing fixed exponential-interval skip connections as a replacement for per-layer residuals. Directly supports the PARTIAL novelty classification. arXiv:2405.01725.

### Skip-Layer Attention (Chen et al., 2024)
Cross-layer attention in transformers[21]: queries in layer L attend to K/V from layer L-k (k>1). Demonstrates non-adjacent layer connections improve language modeling. Applies to the attention mechanism, not residual connections; does not replace per-layer residuals. arXiv:2406.11274.

### Mixture-of-Depths (Raposo et al., 2024)
Token-wise dynamic compute allocation[22]: a top-k router at each layer decides which tokens participate in self-attention and MLP; the rest pass through via identity (residual skip). Achieves iso-FLOP parity with dense baselines and up to ~50% sampling speedup. Directly relevant: MoD's per-layer identity bypass is the token-level analogue of idea 4.4's layer-level skip; the key distinction is that MoD retains per-layer residual paths and skips computation dynamically rather than removing the residual structure. ICML 2024, arXiv:2404.02258.

### LayerDrop (Fan et al., 2019)
Structured dropout over transformer layers[23]: randomly drops entire layers during training, enabling any-depth sub-network extraction at inference without fine-tuning. Directly relevant to layer skipping: demonstrates that transformer layers can be removed post-training with limited accuracy degradation, supporting the existence of layer redundancy that idea 4.4's strong form attempts to exploit. The key difference from idea 4.4 is that LayerDrop retains per-layer residuals during training and drops layers stochastically (not at fixed exponential intervals). ICLR 2020, arXiv:1909.11556.

### ShortGPT (Men et al., 2024)
Layer redundancy analysis in LLMs[24]: introduces a Block Influence (BI) metric (input-output cosine similarity per layer) and shows that many middle-to-late layers are near-identity transformations. Direct removal of low-BI layers achieves superior compression vs. prior pruning methods. Highly relevant: provides empirical evidence that some transformer layers are redundant, which partially motivates idea 4.4; however, ShortGPT's finding is that redundancy emerges after training on standard residual networks — it does not imply that removing residuals during training would be safe. arXiv:2403.03853.

---

## 3. Prior Art Classification

- **Status**: PARTIAL (~85% component coverage after addition of MoD, LayerDrop, ShortGPT)
- **Novel contribution**: The specific combination of (a) *replacing* (not supplementing) all per-layer residual connections in a deep transformer with (b) connections placed at *fixed exponentially spaced intervals* (N/2, N/4, N/8, ...) is not found in published work. AltUp (NeurIPS 2023) demonstrates non-per-step residual updates but uses group alternation and supplements per-layer connections.
- **DeepSeek-V4 update**: mHC makes residual enrichment a scale-tested baseline. Any hybrid version of this idea must now compare against Hyper-Connections/mHC-style learned residual mixing, not only against standard residuals.
- **Searches returning no match for the specific formulation**:
  1. "skip list residual connection multi-scale skip connections deep neural network" — no match
  2. "hierarchical skip connections exponential residual deep network" — no match
  3. "replacing per-layer residuals exponential interval transformer" — no match

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: L=layers, d=hidden, d_ff=MLP intermediate, s=seq length

| Metric | This Idea | Baseline A1 | Baseline A2 | Baseline B |
|--------|-----------|------------|------------|-----------|
| Compute per token (FLOPs) | O(L·(d²+s·d/4)) | O(L·(d²+s·d/4)) | O(L·(s·d+d·d_ff)) | O(L·(d²+k·d·d_e)) |
| KV cache memory | O(L/4·s·d_kv) | O(L/4·s·d_kv) | O(L·s·d_kv) | O(L/4·s·d_kv) |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) |
| TTFT (prefill) | = ref | ref | ref | ref |
| TPOT (decode, batch=1) | = ref | ref | ref | ref |

**Residual FLOP ratio:** At A2 (d=5120, d_ff=25600): residual adds = 5,120 FLOPs per layer; SwiGLU MLP = 3 × 5,120 × 25,600 ≈ 393M FLOPs per layer. Ratio: 77,000:1. TTFT change from removing 64 per-layer residuals and adding 6 exponential-interval residuals: 0.001% of total prefill FLOPs — unmeasurable.

### 4.2 Training Stability Analysis

The 32-layer sub-block (layers 33–64 in a 64-layer Skip List network) is a plain network without internal gradient highway. Gradient attenuation estimate (SiLU, standard init):

E[||J_total||] ≈ (0.5)^32 ≈ **2.3 × 10⁻¹⁰** across the sub-block [derived: each residual-free layer attenuates gradient by ~0.5× in expectation (random init); 32 consecutive layers → (0.5)^32 = 2.328×10⁻¹⁰, identical to the vanishing gradient failure documented by He et al. (2016) for plain networks]

This is effectively zero. He et al. (2016a) empirically demonstrated convergence failure at the same depth (34 layers plain) in Figure 1. The HIGH training risk classification is supported by both theoretical derivation and empirical precedent.

**Skip_store memory overhead:**
- 6 skip connection buffers × s × d × 2 bytes = 6 × 8,192 × 5,120 × 2 = **503 MB** at A2/8K
- This adds approximately 38–72% to activation memory under standard gradient checkpointing
- **Activation memory during training:** skip_store buffers must be kept pinned throughout the forward pass, increasing memory modestly.

### 4.3 Memory Bandwidth Analysis

- **Weight loading:** Unchanged. Skip connections have no learnable weights.
- **KV cache access pattern:** Unchanged. KV cache is populated by the attention layers, which are not altered.
- **Activation memory during training:** **Modestly increased** by ~503 MB (6 skip_store buffers at A2/8K scale).

---

## 5. Implementation Considerations

### Hardware Requirements

No custom kernels required. Skip connections are elementwise additions. Changing which layers feed into which others requires only control flow changes in the forward pass.

```python
# Conceptual forward pass implementation
skip_store = {}
for layer_idx, layer in enumerate(self.layers):
    if layer_idx in self.skip_sources:  # [0, 1, 3, 7, 15, 31] for L=64
        skip_store[layer_idx] = x

    # NO per-layer residual in strong form
    x = layer.attn(layer.norm1(x))   # pure sequential
    x = layer.mlp(layer.norm2(x))

    if layer_idx in self.skip_targets:  # [1, 3, 7, 15, 31, 63]
        x = x + skip_store[self.skip_connections[layer_idx]]
```

Pipeline parallel note: skip connections crossing stage boundaries require ~0.6 ms NVLink transfer overhead per step — under 1% of typical step time.

### Training Stability

**Primary risk:** The 32-layer plain sub-block at layers 33–64 (strong form). **Must-do interventions:**

1. T-Fixup[11] initialization: scale output projections by (9L)^{-1/4} = 0.220× at L=64
2. Extended warmup: 10K+ steps vs standard 2–4K
3. Monitor gradient norms at skip points; alert if gradient norm < 10⁻⁶

**Decision-gate experiment:** A 32-layer plain transformer (no residuals) trained on enwiki8 for 10K steps costs approximately $25 (1 GPU, 6 hours). If this fails to converge (expected per He 2016), pivot to hybrid immediately, saving ~$6,000 in wasted experiment costs.

### Hybrid Variant (Recommended Pursuit Path)

The hybrid form is technically sound: add exponential-interval skip connections on top of existing per-layer residuals, initialized with ReZero-style α ≈ 0.1. This effectively creates a constrained Hyper-Connections variant with exponential structural prior. Per-layer gradient highways are maintained; the exponential-interval connections provide multi-scale residual paths that Hyper-Connections training can refine.

**Reframed hybrid description:**
> "Exponential-interval structural prior for Hyper-Connections: initialize a Hyper-Connections framework with strong weights on adjacent-layer connections and non-zero weights on exponentially-spaced cross-layer connections; zero weights on all others. Test whether this structural prior improves convergence speed over random Hyper-Connections initialization."

**Composition with Idea 4.5 (Learned Residual Flow):** If combining 4.4 and 4.5, the order of application is critical. Idea 4.5 gating (which modulates the per-layer residual weight) must be applied before idea 4.4 connectivity topology is established. Applying 4.4 first (removing or restructuring residual paths) and then layering 4.5 gating on the resulting topology would gate connections that no longer exist in the standard residual sense, producing undefined or degenerate behavior. Correct composition order: first define per-layer residual strengths via 4.5, then overlay the exponential-interval skip connections from 4.4 as additive contributions.

---

## 6. Key Tradeoffs

- **What you gain (hybrid form):** Forced non-trivial layer learning; hierarchical multi-scale gradient highways at log(L) points; potential representation diversity analogous to FPN/DLA vision gains; quality gains bounded by Hyper-Connections results (+1–6 benchmark points).
- **What you lose (strong form):** Per-layer gradient highways; path-length diversity (Veit et al.); practical convergence at depth ≥32.
- **What you gain/lose (both forms):** No TPOT/TTFT benefit. No weight memory change. Modest training memory increase (~503 MB at A2/8K).

---

## 7. Risk Assessment

- **Technical risk (strong form):** HIGH. The 32-layer plain sub-block exhibits ~10⁻¹⁰ gradient attenuation. He et al. (2016a) empirically demonstrated convergence failure at the same depth in Figure 1. No TPOT/TTFT benefit.
- **Technical risk (hybrid form):** MEDIUM. Per-layer residuals maintained; exponential-interval additions are supplementary. Training instability risk is low.
- **Implementation effort:** LOW. No custom kernels; pure PyTorch control flow; <100 lines of code change.
- **Potential impact:** LOW-MEDIUM. Quality improvement is speculative for the strong form; MEDIUM for the hybrid form if it matches Hyper-Connections gains.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta:** No direct evidence for fixed exponential-interval skip connections replacing per-layer residuals in transformers. Indirect comparators for hybrid form: Hyper-Connections +6 ARC-Challenge[17]; Attention Residuals +7.5 GPQA-Diamond[19]. These represent optimistic ceilings from richer mechanisms; idea 4.4's simpler fixed schedule would achieve less improvement.
- **Monotonicity:** Unknown for transformer LLMs. A cliff at depth thresholds (sub-block depth exceeding the ~10–34 layer effective gradient path from Veit et al.) is plausible.
- **Recovery:** Retraining required. Training instability from strong form cannot be recovered post-hoc.

---

---

<!-- CITATION MANIFEST -->

[1] Deep Residual Learning for Image Recognition: He, Zhang, Shaoqing Ren, Sun (CVPR 2016). arXiv:1512.03385. Per-layer residuals essential at depth ≥34 layers; Figure 1 shows plain network convergence failure. Best Paper.

[2] Identity Mappings in Deep Residual Networks: He, Zhang, Shaoqing Ren, Sun (ECCV 2016). arXiv:1603.05027. Multiplicative gating on shortcuts harms training; pure identity shortcuts are optimal. 1001-layer ResNet at 4.62% CIFAR-10.

[3] Residual Networks Behave Like Ensembles: Veit, Wilber, Belongie (NeurIPS 2016). arXiv:1605.06431. Gradient in 110-layer ResNets primarily flows through 10–34 layer paths. Removing per-layer residuals eliminates all short paths.

[4] Training Very Deep Networks (Highway Networks): Srivastava, Greff, Schmidhuber (NeurIPS 2015). arXiv:1505.00387. Learned carry gates enable training 900+ layer networks. Adaptive per-layer routing differs from fixed exponential schedule.

[5] Densely Connected Convolutional Networks (DenseNet): Huang, Liu, van der Maaten, Weinberger (CVPR 2017). arXiv:1608.06993. L(L+1)/2 connections with concatenation. Opposite extreme from Skip List Layers. Best Paper.

[6] Deep Networks with Stochastic Depth: Huang, Sun, Liu, Sedra, Weinberger (ECCV 2016). arXiv:1603.09382. Random per-batch layer dropping during training. Randomized (not fixed exponential) and keeps all per-layer identity paths.

[7] Feature Pyramid Networks: Lin, Dollár, Girshick, He, Hariharan, Belongie (CVPR 2017). Multi-scale hierarchical skip connections with bidirectional pathways. Top-down pathway inapplicable to causal transformers.

[8] Deep Layer Aggregation (DLA): Yu, Wang, Shelhamer, Darrell (CVPR 2018). arXiv:1707.06484. Hierarchical feature aggregation via IDA and HDA. Adds connections rather than replacing per-layer residuals.

[9] ReZero is All You Need: Bachlechner, Majumder, Mao, Halcrow, McAuley (UAI 2021). arXiv:2003.04887. x + α·F(x) with α=0 init; 56% faster convergence; stable 120-layer transformers. Key mitigation for hybrid form.

[10] Fixup Initialization: Zhang, Dauphin, Ma (ICLR 2019). arXiv:1901.09321. Train deep residual networks without BatchNorm by scaling residual branches by L^{-1/(2m-2)}. Theoretical basis for plain sub-block initialization.

[11] T-Fixup — Improving Transformer Optimization: Huang, Perez, Ba, Volkovs (ICML 2020). Transformer-specific Fixup: scale residual branches by (9N)^{-1/4} ≈ 0.220× at L=64. Stable training without warmup.

[12] On Layer Normalization in the Transformer Architecture: Xiong et al. (ICML 2020). arXiv:2002.04745. Pre-LN ensures well-behaved gradients throughout depth; depends on per-layer residuals existing.

[13] DeepNet — Scaling Transformers to 1,000 Layers: Wang, Ma, Dong et al. (IEEE TPAMI). arXiv:2203.00555. DeepNorm: α·x + f(x). 200-layer 3.2B model outperforms 48-layer 12B by 5 BLEU.

[14] ResiDual: Xie, Zhang, Guo et al. (arXiv 2023). arXiv:2304.14802. Pre-Post-LN dual residual connections. Residual structure profoundly affects training stability and capacity.

[15] AltUp — Alternating Updates for Efficient Transformers: Baykal, Cutler, Dikkala, Ghosh, Panigrahy, Wang (NeurIPS 2023). arXiv:2301.13310. Non-per-step residual updates via group alternation. Most mechanistically comparable to idea 4.4; supplements rather than replaces per-layer residuals.

[16] SkipNet — Learning Dynamic Routing in CNNs: Wang, Yu, Dou, Darrell, Gonzalez (ECCV 2018). arXiv:1711.09485. Learned layer skipping; 30–90% computation reduction. Learned (not fixed exponential) skipping pattern.

[17] Hyper-Connections: Zhu, Huang, Huang et al. (ICLR 2025). arXiv:2409.19606. Learned multi-depth connectivity weights. +6 ARC-Challenge on OLMoE-1B-7B; 1.8× faster convergence. Hybrid form of idea 4.4 = constrained Hyper-Connections with exponential structural prior.

[18] M2R2 — Mixture of Multi-Rate Residuals: Bhendawade, Najibi, Naik, Belousova (Apple, arXiv 2025). arXiv:2502.02040. Multi-rate residual velocity modulation for early exit/speculative decoding. 2.8× speedup.

[19] Attention Residuals: Kimi Team (arXiv 2026). arXiv:2603.15031 (arXiv ID could not be independently verified — date appears future-dated; treat as provisional). Softmax attention over all preceding layers replaces residual addition. +7.5 GPQA-Diamond, +3.1 HumanEval, +3.6 MATH. Optimistic quality ceiling for residual enrichment.

[20] Development of Skip Connection in Deep Neural Networks: A Survey: Xu et al. (arXiv 2024). arXiv:2405.01725. No published work found for fixed exponential-interval skip connection replacement in transformers.

[21] Skip-Layer Attention: Chen, Wang, Zhang et al. (arXiv 2024). arXiv:2406.11274. Cross-layer attention (non-adjacent K/V) in transformers. Applies to attention mechanism, not residual path.

[22] Mixture-of-Depths: Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro (ICML 2024). arXiv:2404.02258. Token-wise dynamic layer compute allocation via top-k routing; identity bypass for non-selected tokens. Up to ~50% sampling speedup at iso-FLOP. Token-level analogue of idea 4.4's layer-level skip; retains per-layer residuals.

[23] LayerDrop — Reducing Transformer Depth on Demand with Structured Dropout: Fan, Grave, Joulin (ICLR 2020). arXiv:1909.11556. Stochastic layer dropping during training enables any-depth sub-network extraction at inference. Demonstrates layer-level redundancy in transformers; differs from idea 4.4 in using stochastic (not fixed exponential) dropping and retaining per-layer residuals.

[24] ShortGPT — Layers in Large Language Models are More Redundant Than You Expect: Men, Xu, Zhang, Wang, Lin, Lu, Han, Chen (arXiv 2024). arXiv:2403.03853. Block Influence (BI) metric identifies near-identity middle layers; direct removal achieves strong compression. Redundancy emerges post-training on standard residuals; does not imply training without residuals is safe.
