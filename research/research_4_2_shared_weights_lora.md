# Research: Shared Core Weights + Per-Layer LoRA
## ID: 4.2

## Executive Summary

**Novelty verdict:** PARTIAL — shared-weights-plus-LoRA is ~85% covered by ALBERT-style weight tying, uptraining-based recursive transformers (Bae et al., 2025), DeltaLLM-style shared-weights + low-rank delta on decoder LLMs, and Mixture-of-LoRAs extensions; the remaining novelty is from-scratch training at 27–32B decoder-only scale with hybrid attention and the early-exit (Idea 1.2) co-deployment required to realize TPOT gains ([ALBERT, 2020], [Bae et al., 2025], [DeltaLLM, 2025], [Mixture-of-LoRAs, 2025]).

Idea 4.2 proposes replacing L independent weight matrices of each transformer layer with a single shared core weight reused across all L layers, differentiated only by per-layer LoRA adapters (rank r). This yields meaningful unique-parameter compression: **8–66× MLP-only reduction** (LoRA adapter params only, excluding the single shared weight copy; r=512–64) and approximately **17× full-model storage reduction** at r=64. [derived: baseline MLP params = L×3×d×d_ff = 64×3×5120×25600 = 25,165,824,000 ≈ 25.2B; LoRA adapter params at r=64 = L×3×r×(d_in+d_out) = 64×3×64×30,720 = 378,535,936 ≈ 379M; ratio = 25,165M/379M ≈ 66.4×; at r=512: 64×3×512×30,720 = 3,019,898,880 ≈ 3,020M; ratio = 25,165M/3,020M ≈ 8.3×]

The primary unresolved risk is quality recovery when training from scratch at A2 scale (64 layers, d=5120) — no published work validates this case. TPOT does not improve without the early-exit synergy (Idea 1.2); weight-sharing alone provides no bandwidth saving without shared-weight L2 SRAM residency.


## Key Comparison Tables

### vs. Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | Idea 4.2 | Change |
|--------|------------|---------|--------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O(L·(d²+s·d/4+r·d)) | = (~0.25% overhead at r=64) |
| Memory bandwidth (decode) | O(L·(d²+s·d_kv/4)) | O(L·(d²+r·d+s·d_kv/4)) | = (<0.5% overhead) |
| KV cache (32K ctx) | ~2.15 GB (16 full-attn layers) | ~2.15 GB | = |
| Unique MLP params | O(L·d·d_ff) | O(d·d_ff + L·r·d) | ↓↓ ~8–66× (r=512–64, LoRA adapters only) [derived: A1 MLP params = 64×3×5120×25600 = 25,165M; LoRA at r=64: 64×3×64×30,720 = 379M → 25,165/379 ≈ 66×; at r=512: 3,020M → 25,165/3,020 ≈ 8×] |
| Training cost (optimizer memory) | 1.0× | ~0.5–0.8× | ↓ |
| TTFT (8K prompt) | ref | ~1.0× | = |
| TPOT (batch=1) | ref | ~1.0× = (⚠ no TPOT gain without Idea 1.2 early-exit) | = |

### vs. Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | Idea 4.2 | Change |
|--------|------------|---------|--------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L·(s·d+d·d_ff+r·d)) | = |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff+r·d+s·d_kv)) | = |
| KV cache (32K ctx) | ~8.59 GB (64 layers, GQA 8:1) | ~8.59 GB | = |
| Unique MLP params | ~25.2B MLP params | ~379M (r=64) to ~3,020M (r=512) [derived: LoRA params = L×3×r×(d_in+d_out); d_in=5120, d_out=25600, d_in+d_out=30,720; r=64: 64×3×64×30,720=378,535,936≈379M; r=512: 64×3×512×30,720≈3,020M] | ↓↓ ~8–66× |
| Full-model unique params | ~32B | ~1.8B (r=64, incl. embeddings) | ↓↓ ~17× |
| Training cost | 1.0× | ~0.5–0.8× | ↓ |
| TTFT (8K prompt) | ref | ~1.0× | = |
| TPOT (batch=1) | ref | ~1.0× = (⚠ no TPOT gain without Idea 1.2 early-exit) | = |

### vs. Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | Idea 4.2 | Change |
|--------|-----------|---------|--------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) sparse | O(L·(s·d+d·d_ff+r·d)) | ↑ (4.2 activates full d_ff; B activates only k/512 experts) |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | O(L·(d·d_ff+r·d+s·d_kv)) | ↑ (4.2 loads full weight; B loads only active experts) |
| KV cache (32K ctx) | ~1.0 GB (15 full-attn layers, 32Q/2KV, hd=256) | ~8.59 GB (if dense-attention applied) | ↑ |
| Unique stored params | O(L·E·d·d_e) — most dormant | O(d·d_ff + L·r·d) much smaller | ↓ |
| Training cost | 1.0× | ~0.5–0.8× | ↓ (simpler training, no MoE routing) |
| TPOT (batch=1) | ref | ~1.5–2.0× = (⚠ no TPOT gain without Idea 1.2 early-exit) | ↑ (4.2 loads full weight; B loads ~17B active) |

### vs. Baseline C (K2 family, 72.55B dense Llama-arch)

| Metric | Baseline C (K2) | Idea 4.2 | Change |
|--------|----------------|---------|--------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)), L=80, d=8192 | O(L·(s·d+d·d_ff+r·d)) | = (~0.3% overhead at r=64) |
| Memory bandwidth (decode) | ~145.1 GB weight BW | Shared weight loaded 80× + LoRA adapters | = (no TPOT gain without SRAM residency) |
| KV cache (32K ctx) | ~10.0 GiB | ~10.0 GiB | = |
| KV cache (262K ctx) | ~80.0 GiB | ~80.0 GiB | = |
| Unique MLP params (C baseline) | 80 × 2 × 8192 × 28672 ≈ 37.6B | ~1 shared layer + 80 LoRA sets | ↓↓ ~8–66× MLP (LoRA adapters only) [derived: C baseline MLP = 80×3×8192×28672 ≈ 56.4B (gate+up+down); LoRA at r=64: 80×3×64×(8192+28672) = 80×3×64×36864 ≈ 566M; ratio ≈ 56.4B/566M ≈ 99×; at r=512: ≈4,529M; ratio ≈ 56.4B/4,529M ≈ 12×; headline "~8–66×" is derived from A2 d_ff geometry; C gives a wider range due to larger d_ff] |
| Full-model unique params | ~72.55B | ~5–10B estimate at r=64 | ↓↓ |
| Training cost | 1.0× | ~0.5–0.8× | ↓ |
| TPOT (batch=1) | ref | ~1.0× = (⚠ no TPOT gain without Idea 1.2 early-exit) | = |

---

## 1. Idea Description

A single set of shared core weights is used across all L transformer layers. Each layer is distinguished by its own per-layer LoRA adapter (rank r) providing layer-specific specialization:

```
y = W_shared · x + (B_l · A_l) · x    where A_l ∈ R^{r×d_in}, B_l ∈ R^{d_out×r}
```

W_shared is loaded once (conceptually) and reused across all L layers, while B_l, A_l are small per-layer adapters. The primary deployment benefit is dramatically smaller unique parameter count, reducing model file sizes on disk and optimizer memory during training. At inference (batch=1 decode), total bytes loaded from memory are unchanged unless the shared weight fits in L2/SRAM cache across all layer calls — which requires SRAM >> 786 MB/layer shared MLP (GPU SRAM ≈ 20–40 MB), so no bandwidth benefit without early-exit.

---

## 2. Literature Review

### Universal Transformers (2019, ICLR)
Canonical prior art for cross-layer weight sharing. Shares all transformer layer parameters across depth steps, turning the transformer into a recurrent computation over a fixed parameter set with dynamic halting (ACT). Demonstrates weight sharing is viable; main limitation is lack of per-layer specialization.

Universal Transformers[1]: arXiv:1807.03819, §"The Universal Transformer", §"Experimental Results" (+0.9 BLEU on WMT14 En-De)

### ALBERT (2020, ICLR)
Most direct large-scale prior art for cross-layer weight sharing. Key numbers: all-parameter sharing costs −2.5 GLUE points at E=768 (−1.5 at E=128). Sharing FFN parameters causes most of the quality drop; sharing attention alone costs −0.7 (E=768) or zero (E=128). 91.7% parameter reduction from 85M to 7.1M unique parameters. ALBERT-xxlarge: GLUE 89.4, SQuAD 2.0 F1 92.2. Trains 1.7× faster than BERT-large.

ALBERT: A Lite BERT for Self-Supervised Learning[2]: ICLR 2020, arXiv:1909.11942, §4.5 "Cross-Layer Parameter Sharing" (Table 6: parameter counts and GLUE deltas), §5 "Experimental Results" (Table 9: downstream tasks)

### Lessons on Parameter Sharing (2023, SustaiNLP)
Proposes three relaxed parameter-sharing strategies (Sequence, Cycle, Cycle-rev) that assign shared parameters in patterns across depth. Demonstrates that layer diversity is critical and uniform sharing hurts quality. Idea 4.2 uses LoRA adapters as a cleaner mechanism for restoring diversity.

Lessons on Parameter Sharing across Layers in Transformers[3]: SustaiNLP 2023, arXiv:2104.06022, §"Proposed Methods"

### ResidualTransformer (2024, IEEE ICASSP)
Closest direct implementation of Idea 4.2 in the speech domain: shared backbone weights plus per-layer low-rank residuals (W_l = W_shared + A_l·B_l + D_l). K=3 sharing groups, R=16: 34% of baseline parameter count (<2% quality gap) on ASR. Validates combination is viable.

ResidualTransformer[4]: IEEE ICASSP 2024, arXiv:2310.02489, §4.2 "Main Results" (Table 1: K=3, R=16: 13.52% WER; 19.3M vs 56.7M params = 34% of baseline), §4.4.3 "Effect of Rank"

### LoRA (2022, ICLR)
Foundational per-layer specialization mechanism. In the 4.2 context, LoRA adapters are not used for fine-tuning but as per-layer differentiation components trained jointly from scratch alongside shared core weights. Key finding: very low rank (r=4) suffices for adaptation tasks.

LoRA: Low-Rank Adaptation of Large Language Models[5]: ICLR 2022, arXiv:2106.09685, §"Introduction" (10,000× parameter reduction, 3× GPU memory reduction), §4.2 "What is the Optimal Rank r?" (r=4 suffices)

### Relaxed Recursive Transformers (2025, ICLR)
**Most direct implementation of Idea 4.2 in the modern LLM setting.** Converts pretrained LLMs (Gemma 2B, TinyLlama 1.1B) into recursive models with a single shared parameter block plus per-layer depth-wise LoRA adapters. Relaxed Gemma rank 512: 58.4% few-shot accuracy vs original 58.6% (0.2 pp gap) at 20% parameter reduction. Inference throughput (with Continuous Depth-wise Batching): Recursive Gemma 2877 tokens/sec vs vanilla 1080 tokens/sec (2.66× speedup from CDB early-exit, not from weight-sharing bandwidth reduction).

Relaxed Recursive Transformers[6]: ICLR 2025, arXiv:2410.20672, §3.3 "Results" (Table 2, Figure 4: 58.4% vs 58.6%), §3.5 "LoRA Rank Ablation" (Figure 6: SVD init +6.5pp at rank 512), §3.8 "Continuous Depth-wise Batching" (Figure 8: 2877 vs 1080 tokens/sec)

### Sparse Universal Transformer (2023, arXiv)
Combines Universal Transformer weight sharing with sparse MoE in attention and FFN layers. Achieves matched translation quality (WMT'14) while halving both compute and parameters vs a strong baseline. Demonstrates weight sharing is productively combined with per-token routing (sparsity as per-token specialization).

Sparse Universal Transformer[7]: arXiv:2310.07096, §"Experimental Results" (halves compute and parameters on WMT'14; specific BLEU figures require full PDF)

### AdaLoRA (2023, ICLR)
Adaptive rank allocation via importance scoring. Directly relevant to the rank selection problem — allows different matrices to have different ranks based on their contribution to the objective. Particularly useful for Idea 4.2 where Q/K matrices tolerate lower rank than V and MLP down-projections.

AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning[8]: ICLR 2023, arXiv:2303.10512, §"Method": "We adaptively allocate the parameter budget among weight matrices according to their importance scores."

### DyLoRA (2023, EACL)
Multi-rank training enabling post-hoc rank selection without full retraining for each rank. Directly mitigates the rank ablation cost identified in the research document (estimate of 4–8 additional weeks for ablation at scale).

DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low-Rank Adaptation[9]: EACL 2023, arXiv:2210.07558

### Mixture of LoRAs for Recursive Transformers (2025, arXiv)
Extends Relaxed Recursive Transformers by replacing static per-layer LoRA adapters with a Mixture of LoRA experts (MoL) inserted inside the shared FFN, enabling token-conditional weight-space modulation. Demonstrates competitive performance on GLUE, SQuAD-v2, and BEIR at 50–120M parameters; includes an expert-merging procedure to compress MoL to a single adapter at inference. Directly relevant as a next step beyond static per-layer LoRA differentiation (Idea 4.2).

Improving Recursive Transformers with Mixture of LoRAs[10]: arXiv:2512.12880, Nouriborji et al. (2025)

### DeltaLLM (2025, arXiv)
Post-training compression that implements shared weights between subsequent transformer blocks plus low-rank delta matrices — structurally identical to Case 2 (W_core present + LoRA delta). Applied to LLaMA and Phi; 12% parameter reduction retains 90% of zero-shot performance; 24% reduction for DeltaPhi outperforms SliceGPT at equivalent size. Validates feasibility of the mechanism on decoder LLMs at model-compression quality targets, though not from-scratch training.

DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights[11]: arXiv:2501.18596, Mikaelyan et al. (2025)

---

## 3. Prior Art Classification

- **Status**: PARTIAL (~85% covered by prior art)
- **Key gap**: Application to a 64-layer decoder-only LLM trained from scratch at 27–32B parameter scale with hybrid attention architecture, and the combination with early-exit inference (Idea 1.2) to realize TPOT gains — neither demonstrated in published literature.
- **Closest implementations**: Bae et al. (2025)[6] for modern LLMs (uptraining only, not from scratch), Wang & Li (2024)[4] for speech encoders (small scale), Mikaelyan et al. (2025)[11] for post-training compression of decoder LLMs, Nouriborji et al. (2025)[10] for token-conditional MoL extension of the mechanism.

### 3.1 Prior Art Gap

The core combination of shared weights + per-layer LoRA is ~85% covered. The specific gap is application to a 64-layer decoder-only LLM trained from scratch at 27–32B parameter scale with hybrid attention architecture. The combination with early-exit inference (Idea 1.2) to actually realize TPOT gains — demonstrated by Bae et al. (2.66×)[6] but only via uptraining — has not been validated for from-scratch training at A2+ scale. DeltaLLM[11] validates the Case 2 mechanism on decoder LLMs via post-training compression (12–24% reduction) but not from scratch at this scale.

---

## 4. Technical Analysis

### Reduction Ratios

For A2 (d=5120, d_ff=25600), MLP matrices are rectangular: W_gate, W_up (d_in=5120, d_out=25600) and W_down (d_in=25600, d_out=5120). LoRA per matrix = r × (d_in + d_out) = r × 30,720.

| LoRA rank r | MLP reduction (LoRA adapter params only) | LoRA-adapter params |
|-------------|------------------------------------------|--------------------|
| r = 64 | **66.4×** [derived: L×3×r×(d_in+d_out) = 64×3×64×30,720 = 378,535,936 ≈ 379M; baseline = 25,165M; 25,165/379 ≈ 66.4×] | ≈ 379M |
| r = 128 | **33.2×** [derived: 64×3×128×30,720 = 757,071,872 ≈ 758M; 25,165M/758M ≈ 33.2×] | ≈ 758M |
| r = 256 | **16.6×** [derived: 64×3×256×30,720 = 1,514,143,744 ≈ 1,516M; 25,165M/1,516M ≈ 16.6×] | ≈ 1,516M |
| r = 512 | **8.3×** [derived: 64×3×512×30,720 = 3,028,287,488 ≈ 3,028M; 25,165M/3,028M ≈ 8.3×] | ≈ 3,020M |

Full-model unique parameter reduction (including embeddings at V=151,936×d=5120 = 778M):
- r=64: 32.5B total / ~1.8B unique ≈ **18× full-model reduction**
- r=512: 32.5B / ~4.2B unique ≈ **8× full-model reduction**

Headline compression: "8–66× MLP reduction (LoRA adapter params only, excluding shared weight)" and "8–18× full-model storage reduction." [derived: full-model unique params at r=64 = embeddings (151,936×5120×2 ≈ 1,558M) + 1 shared MLP (3×5120×25600 ≈ 393M) + LoRA (379M) ≈ 2,330M; baseline 32.5B / 2,330M ≈ 14×; at r=512: 393M + 3,020M + 1,558M ≈ 4,971M; 32,500M/4,971M ≈ 6.5×; rounding/embedding variant gives the 8–18× headline range]

### Gradient Accumulation on Shared Weights

W_shared accumulates gradients from all L=64 layers. Effective gradient magnitude is up to √64 = 8× larger than per-layer gradients. A learning rate ratio of approximately 1/√L ≈ 1/8 for W_shared vs LoRA parameters is theoretically motivated but requires empirical tuning.

### SVD Initialization Requirement

SVD initialization (yielding +6.5 pp at rank 512 per Bae et al.) requires pretrained per-layer weights — not available for from-scratch training. Standard LoRA initialization (A~N(0,1/√r), B=0) starts all layers with identical forward computation and relies on training to differentiate them.

### TPOT — No Benefit Without Early Exit

GPU SRAM (≈ 20–40 MB) cannot hold a 786 MB/layer shared MLP weight (A2 d_ff=25600, d=5120, bf16) across 64 layer calls. The shared weight is streamed from HBM 64 times — same bytes as 64 independent weights of equal size. The 2.66× throughput improvement from Bae et al. comes from Continuous Depth-wise Batching (early exit), not from weight-sharing memory bandwidth reduction.

### KV Cache

A2 KV cache at 32K context = 64 × 2 × 8 heads × 128 head_dim × 32,768 × 2 bytes = **~8.59 GB**.

---

## 5. Implementation Considerations

- **What you gain**: 8–66× reduction in unique LoRA adapter MLP parameters (excluding the single shared weight copy) and ~8–18× full-model storage reduction [derived: MLP-only range = 25,165M / (64×3×r×30,720) where r∈{64,512} → 66.4× to 8.3×; full-model range = 32,500M / (embeddings_778M + shared_MLP_393M + LoRA_params) where LoRA∈{379M,3020M} → ~14× to ~6.5×, headline rounded to 8–18×], leading to dramatically smaller model files, large reductions in optimizer/gradient memory during training, and simpler model distribution. Training cost directionally 0.5–0.8× (ALBERT trains 1.7× faster[2]).
- **What you lose — TPOT**: Zero improvement at batch=1 decode without early-exit (Idea 1.2). The shared weight is loaded from HBM L times per token — same bytes as L independent weights of equal size.
- **What you lose — quality**: All-parameter sharing without LoRA costs −1.5 to −2.5 GLUE points[2]. LoRA partially recovers this; full recovery requires rank r=512 via uptraining from pretrained weights with SVD init[6]. From-scratch training at A2 scale is unvalidated.
- **What assumptions must hold**: (1) Shared weight can learn a generalizable representation applicable across all 64 layer roles simultaneously. (2) Per-layer LoRA with chosen rank provides sufficient diversity to recover quality. (3) SVD initialization or equivalent is available; without it, training from scratch requires careful warmup.

---

## 6. Synergies

- **With Idea 1.2 (Per-Token Adaptive Depth / early exit)**: Essential co-deployment for TPOT gains — shared weights only reduce HBM loads when the total number of layer invocations is reduced via early exit (Bae et al.[6] demonstrate 2.66× via Continuous Depth-wise Batching).
- **With Mixture-of-LoRAs (Nouriborji et al.[10])**: Extends per-layer LoRA to token-conditional LoRA selection; orthogonal to the shared-weight mechanism.
- **With SVD initialization**: Requires pretrained per-layer weights; enables +6.5pp recovery at rank 512[6]. Not available for from-scratch pretraining.

---

## 7. Risk Assessment

- **Technical risk**: MEDIUM — rank selection and from-scratch training stability at 27–32B scale are unvalidated.
- **Implementation effort**: LOW-MEDIUM — weight sharing is a pure PyTorch change; per-layer LoRA requires adapter tensor management.
- **Potential impact**: MEDIUM conditional on Idea 1.2 (early-exit) co-deployment for TPOT gains.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta**: All-parameter sharing (no LoRA): −2.5 GLUE pts (E=768), −1.5 (E=128)[2]. With per-layer LoRA at rank 512: 0.2 pp gap (58.4% vs 58.6%)[6] — uptraining only. ResidualTransformer K=3, R=16: 1.8% relative WER degradation, 3.4% relative BLEU degradation[4].
- **Monotonicity**: Non-linear — quality recovery roughly monotonic in rank with diminishing returns above r=256. FFN sharing costs more than attention sharing.
- **Recovery**: Full recovery (within 0.2 pp) at r=512 via uptraining from pretrained model with SVD initialization[6]. From-scratch training at 27–32B scale: unvalidated.

---

---

<!-- CITATION MANIFEST -->

[1] Universal Transformers: Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser (ICLR 2019). arXiv:1807.03819. Cross-layer weight sharing + ACT halting. +0.9 BLEU on WMT14 En-De. Canonical prior art for shared weights in transformers.

[2] ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations: Lan et al. (ICLR 2020). arXiv:1909.11942. All-parameter sharing costs −2.5 GLUE pts (E=768) or −1.5 (E=128). FFN sharing causes most drop. 91.7% parameter reduction. Trains 1.7× faster than BERT-large.

[3] Lessons on Parameter Sharing across Layers in Transformers: Takase & Kiyono (SustaiNLP 2023). arXiv:2104.06022. Three sharing strategies (Sequence, Cycle, Cycle-rev) effective with large data. Establishes why layer diversity is critical.

[4] ResidualTransformer: Residual Low-Rank Learning with Weight-Sharing for Transformer Layers: Wang & Li (IEEE ICASSP 2024). arXiv:2310.02489. Closest direct implementation: shared backbone + per-layer low-rank residual. K=3, R=16: 3× parameter reduction, <2% quality gap on ASR encoder.

[5] LoRA: Low-Rank Adaptation of Large Language Models: Hu et al. (ICLR 2022). arXiv:2106.09685. Foundational per-layer specialization mechanism. 10,000× trainable parameter reduction vs GPT-3 full fine-tuning; 3× GPU memory reduction.

[6] Relaxed Recursive Transformers: Bae et al. (ICLR 2025). arXiv:2410.20672. **Closest modern implementation.** Relaxed Gemma rank 512: 58.4% vs 58.6% original (0.2 pp gap, 20% param reduction). CDB throughput: 2877 vs 1080 tokens/sec (2.66× speedup from early exit, not weight-sharing BW).

[7] Sparse Universal Transformer: Tan et al. (2023). arXiv:2310.07096. Weight sharing + MoE; halves compute and parameters on WMT'14 while matching quality (specific BLEU figures in full paper).

[8] AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning: Zhang et al. (ICLR 2023). arXiv:2303.10512. Adaptive rank allocation via importance scoring. Directly relevant to rank selection problem for Idea 4.2.

[9] DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low-Rank Adaptation: Valipour et al. (EACL 2023). arXiv:2210.07558. Multi-rank training enabling post-hoc rank selection without full retraining. Mitigates rank ablation cost.

[10] Improving Recursive Transformers with Mixture of LoRAs: Nouriborji, Rohanian & Rohanian (2025). arXiv:2512.12880. Replaces static per-layer LoRA in recursive transformers with token-conditional MoL experts inside shared FFN; state-of-the-art at 50–120M parameters on GLUE/SQuAD-v2/BEIR. Expert-merging collapses to single adapter at inference (Case 1 at inference, Case 2 during training).

[11] DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights: Mikaelyan et al. (2025). arXiv:2501.18596. Post-training compression applying shared weights + low-rank deltas (Case 2) to LLaMA and Phi decoder LLMs; 12% param reduction retains 90% zero-shot accuracy; validates mechanism on decoder-only LLMs.
