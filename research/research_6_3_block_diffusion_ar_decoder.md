# Research: Idea 6.3 — Block-Diffusion Autoregressive Decoder

**Date:** 2026-04-13
**Category:** Generation Paradigm / Diffusion Language Models
**Baseline Models:** A1 (Qwen3.5-27B Hybrid), A2 (Qwen3-32B Dense), B (Qwen3.5-397B-A17B Hybrid MoE), C (K2 family, 72.55B Dense)

---

## Executive Summary

Idea 6.3 proposes a hybrid generation paradigm that emits a block of k tokens per step via a discrete diffusion process within the block, while maintaining autoregressive left-to-right dependencies across blocks.

**Key Comparison Tables**

| Metric | A1 Qwen3.5-27B Hybrid | A2 Qwen3-32B Dense | B Qwen3.5-397B MoE | C K2 family 72.55B |
|--------|----------------------|-------------------|-------------------|-------------------|
| Forward passes for T tokens (k=8, D=4) | T/2 (2x fewer) | T/2 (2x fewer) | T/2 (2x fewer) | T/2 (2x fewer) |
| KV cache @32K (same as standard AR) | 2.15 GB | 8.59 GB | 1.0 GB | 10.0 GiB |
| DeltaNet state cache (fixed, const) | ~2.52 GB | N/A | ~1.51 GB | N/A |
| Total FLOPs vs standard AR | +4× | +4× | +4× | +4× |
| Within-block joint modeling | PARTIAL (25% full-attn) | FULL | PARTIAL (25% full-attn) | FULL |
| DeltaNet bidirectionality within block | INCOMPATIBLE (75% layers) | N/A | INCOMPATIBLE (75% layers) | N/A |
| Wall-clock speedup (BW-bound, batch=1) | ~2× | ~2× | ~2× | ~2× |
| Wall-clock speedup (compute-bound) | ~0.25× (4× slower) | ~0.25× (4× slower) | ~0.25× (4× slower) | ~0.25× (4× slower) |

**Critical overhead disclosure:** Block-diffusion incurs a 4× compute-bound penalty relative to standard AR (D=4 denoising steps × k=8 block size / 8 tokens committed = 4× total FLOPs per output token). The ~2× wall-clock speedup cited above applies **only** in the memory-bandwidth-bound regime (batch=1, single-GPU decode). At large batch sizes or on compute-bound hardware (e.g., high-end datacenter GPUs with high arithmetic intensity), block-diffusion is approximately **4× slower** than standard AR. This idea is only beneficial for single-request latency-optimized serving.

---

## Idea Overview

Rather than emitting one token per forward pass (standard AR) or all tokens simultaneously (full diffusion), the decoder emits a block of k tokens per step via a discrete diffusion process within the block, while maintaining autoregressive left-to-right dependencies across blocks.

**Position in design space:**
- Left extreme (k=1, D=∞): standard full diffusion (MDLM, LLaDA, SEDD) — global bidirectional, slow sampling
- Right extreme (k=1, D=1): standard 1-token AR — maximal throughput, no joint modeling
- This idea (k>1, 1 < D < k): intermediate, trading local coherence for throughput

**Key parameters:**
- k: block size (tokens emitted per block step)
- D: denoising steps per block (must have D < k for throughput benefit)
- Noise schedule: masked (MDLM-style) or score-based (SEDD-style) within each block

---

## Prior Art

### Direct Prior Art: This Idea Is BD3-LM

| Paper | Year | Venue | Overlap |
|-------|------|-------|---------|
| BD3-LM (arXiv:2503.09573)[1] | 2025 | ICLR 2025 Oral | EXACT — identical mechanism |
| MDLM semi-AR mode (arXiv:2406.07524)[2] | 2024 | NeurIPS 2024 | HIGH — precursor, shared authors |
| Unifying AR+Diffusion (arXiv:2504.06416)[3] | 2025 | COLM 2025 | HIGH — hyperschedules; direct overlap |
| Causal AR Diffusion LM (arXiv:2601.22031)[4] | 2026 | Preprint | HIGH — explicit causal diffusion |

BD3-LM (Arriola et al., ICLR 2025 Oral) establishes:
- Efficient training algorithm with 20-25% speedup via concatenated batch for two forward passes
- Data-driven noise schedules minimizing variance
- SOTA performance among discrete diffusion models on language modeling benchmarks
- Up to 13% perplexity improvement over MDLM and SEDD
- KV caching across blocks for efficient inference
- Arbitrary-length generation

### Core Diffusion Language Model Foundations

| Paper | arXiv ID | Year | Venue | Contribution |
|-------|----------|------|-------|-------------|
| SEDD (Lou, Meng, Ermon) | 2310.16834 | 2024 | ICML 2024 Oral | Score entropy for discrete diffusion |
| MDLM (Sahoo et al.) | 2406.07524 | 2024 | NeurIPS 2024 | Masked diffusion; Rao-Blackwellized ELBO; semi-AR generation |
| LLaDA (Nie et al.) | 2502.09992 | 2025 | NeurIPS 2025 Oral | 8B masked diffusion LLM from scratch |
| Diffusion-LM (Li et al.) | 2205.14217 | 2022 | ACL 2022 | Continuous diffusion for text; historical precursor |

### Hybrid AR+Diffusion Literature (Active Post-BD3-LM Development)

| Paper | arXiv ID | Year | Venue | Contribution |
|-------|----------|------|-------|-------------|
| DiffuGPT/DiffuLLaMA (Gong et al.) | 2410.17891 | 2025 | ICLR 2025 | AR-to-diffusion adaptation; <200B tokens |
| Diffusion-in-Diffusion (Ma et al.) | 2601.13599 | 2026 | Preprint | Global coherence failure (+17% PPL); draft-then-refine |
| Fast-dLLM (Wu et al.) | 2505.22618 | 2025 | Preprint | Training-free KV cache + confidence-aware parallel decoding; up to 27.6× throughput on LLaDA/Dream |
| Fast-dLLM v2 (Wu et al.) | 2509.26328 | 2025 | Preprint | Block-diffusion LLM from AR adaptation with ~1B tokens; up to 2.5× speedup over AR |
| Blockwise SFT for Diffusion LMs | 2508.19529 | 2025 | Preprint | Instruction-tuning for block diffusion |
| Dream 7B (Ye et al.) | 2508.15487 | 2025 | Preprint | 7B masked diffusion LLM matching AR on math/code; strong planning; 580B-token pretraining |

### Throughput Comparison Baselines

| Paper | arXiv ID | Year | Venue | Mechanism |
|-------|----------|------|-------|-----------|
| Speculative Decoding (Leviathan et al.) | 2211.17192 | 2023 | ICML 2023 Oral | Draft+verify; 2-3× speedup, exact distribution |
| Medusa (Cai et al.) | 2401.10774 | 2024 | Preprint | Multi-head decoding; 2.2-3.6× speedup |
| EAGLE (Li et al.) | 2401.15077 | 2024 | ICML 2024 | Feature-level draft; 2.7-3.5× speedup |

---

## Mechanism

### Generation Process (Inference)

1. Initialize: empty sequence, empty KV cache
2. For block b = 0, 1, 2, ..., T/k - 1:
   a. Initialize block b's k token positions with fully masked/noisy tokens
   b. For denoising step d = 0, 1, ..., D-1:
      - Forward pass: attend over KV cache (prior blocks 0..b-1) plus within-block k tokens (bidirectional for full-attn layers)
      - Predict clean token distribution for each masked position
      - Update within-block tokens via denoising update (e.g., sample from predicted distribution with remasking schedule)
   c. After D denoising steps: commit block b's k clean tokens to the KV cache
3. Output: concatenation of all T tokens from T/k blocks

### Attention Pattern

Within each forward pass:
- Cross-block attention (prior blocks → current block): CAUSAL (left-to-right across block boundaries)
- Within-block attention (current block ↔ current block): BIDIRECTIONAL for full-attention layers; UNIDIRECTIONAL for DeltaNet layers
- Standard full-attention layers: implement bidirectional within-block via block-diagonal attention mask
- DeltaNet recurrent layers: inherently unidirectional within-block (sequential recurrent update)

### Training Objective

Block-level ELBO over masked diffusion:
- For each block b in sequence: sample noise level t ~ Uniform[0, 1]
- Apply masking at rate t to the k tokens in block b (independently per token, MDLM-style)
- Condition on clean left context (blocks 0..b-1 unmasked, in KV cache during training)
- Predict all masked tokens in block b
- Loss: block-level ELBO = sum over masked positions of cross-entropy loss, weighted by noise schedule
- This equals a mixture of MLM losses within each block, conditioned on clean left context

The BD3-LM paper[1] provides the specific variance reduction estimators and data-driven noise schedule required for stable training.

---

## Complexity Analysis

### Notation

- T = total sequence length (tokens)
- k = block size (default: 8)
- D = denoising steps per block (default: 4)
- L = transformer layers
- d = hidden dimension
- C = cost of one single-token AR forward pass

### Throughput and Forward Pass Count

| Metric | Standard AR | Block-Diffusion (k=8, D=4) | Delta |
|--------|-------------|---------------------------|-------|
| Forward passes for T tokens | T | T/2 | 2× fewer |
| Total FLOPs (normalized to 1-token AR) | T × C | T × D × C = 4T × C | +4× total FLOPs |
| Wall-clock (memory-BW-bound, batch=1) | T × t_pass | (T/2) × t_pass | 2× faster |
| Wall-clock (compute-bound, large batch) | T × t_pass | ~4T × t_pass | 4× SLOWER |
| KV entries added per block step | 1 | k = 8 | 8× per block |
| KV entries added per forward pass | 1 | k/D = 2 | 2× per pass |
| Break-even condition | — | D < k required | Must hold for benefit |

### KV Cache Sizes at Canonical Baselines

Peak KV memory is IDENTICAL to standard AR. Within-block denoising reuses prior-block KV read-only; prior-block KV committed once per block.

| Baseline | KV Formula | @32K | @262K | Status |
|----------|-----------|------|-------|--------|
| A1: Qwen3.5-27B Hybrid | 16×2×4×256×s×2 = 65,536·s | 2.15 GB | 17.2 GB | VERIFIED |
| A2: Qwen3-32B Dense | 64×2×8×128×s×2 = 262,144·s | 8.59 GB | N/A (ctx=40K) | VERIFIED |
| B: Qwen3.5-397B MoE | 15×2×2×256×s×2 = 30,720·s | 1.0 GB | 8.0 GB | VERIFIED |
| C: K2 family | 80×2×8×128×s×2 = 327,680·s | 10.0 GiB | 80.0 GiB | VERIFIED |

DeltaNet recurrent state (fixed, independent of sequence length):
- A1 DeltaNet state: 48 × 5120² × 2 bytes ≈ 2.52 GB (constant)
- B DeltaNet state: 45 × 4096² × 2 bytes ≈ 1.51 GB (constant)

### Break-Even Analysis

| k | D | Pass ratio (BD/AR) | Speedup (BW-bound) | Quality |
|---|---|-------------------|---------------------|---------|
| 8 | 1 | 0.125 | 8× | Likely poor |
| 8 | 2 | 0.25 | 4× | Marginal |
| 8 | 4 | 0.50 | 2× | Reasonable (BD3-LM operating point) |
| 8 | 8 | 1.0 | 1× | No speedup |
| 8 | 16 | 2.0 | 0.5× | Worse than AR |
| 16 | 4 | 0.25 | 4× | Better quality than k=8,D=4 |
| 32 | 4 | 0.125 | 8× | High quality + high speedup (but increased PPL vs AR) |

---

## Feasibility Assessment

### Training Feasibility

**Standard Dense Transformer (A2, C): HIGH**

BD3-LM has fully demonstrated this training pipeline. DiffuGPT/DiffuLLaMA[5] demonstrates successful AR-to-diffusion adaptation with <200B tokens, confirming fine-tuning is viable without pretraining from scratch.

### Architecture-Specific Feasibility

#### A1: Qwen3.5-27B Hybrid (DeltaNet)

- 16/64 full-attention layers: FULLY COMPATIBLE with bidirectional within-block attention
- 48/64 DeltaNet layers: PARTIALLY COMPATIBLE
  - Prior-block state cacheable as fixed-size d×d matrix (~52MB/layer; 2.52GB total for 48 layers)
  - Within-block attention effectively unidirectional (left-to-right sequential recurrent update)
  - Joint within-block modeling degraded for 75% of layers
  - Mitigation: rely on full-attention layers (16/64) for within-block joint modeling; DeltaNet layers approximate within-block context with sequential updates

**Net feasibility for A1:** MEDIUM. DeltaNet layers provide partial joint modeling; quality likely lower than pure-attention model.

#### A2: Qwen3-32B Dense

All 64 layers: standard full-attention — FULLY COMPATIBLE. Best baseline for block-diffusion.

**Net feasibility for A2:** HIGH.

#### B: Qwen3.5-397B MoE + DeltaNet

- 15/60 full-attention layers: FULLY COMPATIBLE
- 45/60 DeltaNet layers: PARTIALLY COMPATIBLE (same constraint as A1; 75% ratio)
- MoE routing: per-token routing unchanged; noisy tokens at early denoising steps may cause routing instability; manageable with auxiliary load-balancing loss

**Net feasibility for B:** MEDIUM.

#### C: K2 family (72.55B Dense)

All 80 layers: standard full-attention — FULLY COMPATIBLE. Same clean profile as A2.

**Net feasibility for C:** HIGH.

### Known Limitations

1. **Global coherence degradation** (Ma et al., arXiv:2601.13599[6]): plain semi-AR block-diffusion generative PPL is 25.7 vs 21.9 for draft-then-refine global bidirectional (+17.4% PPL increase under semi-AR vs bidirectional refinement on OpenWebText). Mitigation: draft-then-refine global pass (additional full bidirectional pass after all blocks are committed).
2. **D must be < k for throughput**: At D ≥ k, block-diffusion provides no wall-clock speedup. At large batch (compute-bound), block-diffusion is always slower.
3. **Training instability (high gradient variance)**: BD3-LM requires specific variance estimators; non-trivial implementation.
4. **DeltaNet incompatibility (partial)**: 75% of layers in A1/B lose bidirectional within-block joint modeling.
5. **Block-diffusion changes the output distribution**: Unlike speculative decoding which preserves the AR distribution exactly, block-diffusion produces a different generative model requiring full quality re-evaluation.

### Inference Compatibility

- KV cache: identical peak memory to standard AR; compatible with vLLM paged attention, FlashAttention
- torch.compile: compatible with static D (compile-time constant)
- CUDA graphs: within-block denoising (D steps over k tokens) has fixed shapes — fully capturable; cross-block KV growth uses same piecewise strategy as standard AR engines (vLLM, TensorRT-LLM)

---

## Comparison: Block-Diffusion vs Speculative Decoding

| Aspect | Block-Diffusion | Speculative Decoding (Medusa/EAGLE) |
|--------|----------------|-------------------------------------|
| Changes generation model? | Yes (new training, new distribution) | No (same output distribution) |
| Requires extra parameters? | No (same model, new training) | Yes (Medusa heads or draft model) |
| Joint token distribution? | Yes (D denoising steps within block) | No (independent per position) |
| Throughput at batch=1 | ~2× (k=8, D=4) | 2.2-3.6× (Medusa), 2.7-3.5× (EAGLE) |
| FLOPs overhead | +D×k total FLOPs vs AR | ~+25% (Medusa-1) |
| Works with existing AR models? | No (requires fine-tuning) | Yes (plug-in) |
| Quality vs standard AR | Distribution shift; re-evaluation required | Identical (speculative) / approximate (Medusa) |

---

## Remaining Novel Territory (Conditional Pursue Path)

The following aspects are NOT covered by BD3-LM or any identified paper (as of April 2026):

1. **BD3-LM applied to hybrid DeltaNet architectures (A1, B)**: Whether DeltaNet's fixed-size recurrent state can replace the O(s×d) KV cache overhead for long-context block-diffusion, and how DeltaNet delta-rule updates interact with within-block masked diffusion.

2. **Block-diffusion with MoE routing (B)**: Routing stability under token-level masking noise across D denoising steps; expert utilization patterns during denoising vs. standard AR.

3. **Extreme long-context block-diffusion**: Block-diffusion at 262K context (A1/B native context) — BD3-LM experiments focus on short-medium sequences; KV caching across thousands of blocks at extreme context is unstudied.

4. **Token-budget adaptive block sizing**: Variable k (larger blocks for easy text, smaller for complex) is not in BD3-LM.

**Novelty verdict:** EXISTS — the core mechanism is BD3-LM [Arriola et al., ICLR 2025] building on MDLM [Lou et al., 2024], with LLaDA and Fast-dLLM v2 as concurrent prior art; the only residual contribution is empirical characterization of block-diffusion AR decoding on hybrid Gated-DeltaNet backbones (A1, B), where the recurrent-state interface to within-block masked denoising has not been measured.

---

## Synergies with Other Ideas

### High Synergy

- **5.2: LSTM-Gated Attention** — DeltaNet state cache for block-diffusion provides O(d²) prior-context summary vs O(s×d) KV; high memory efficiency at long contexts.
- **1.2: Per-Token Adaptive Depth** — Adaptive depth within denoising steps (fewer layers at early noisy steps) could reduce D×k FLOPs overhead.

### Medium Synergy

- **5.5: Ragged Window Attention** — Block boundaries align with sliding-window boundaries; variable k based on content complexity.
- **3.1: Layer-Level MoE** — Different expert sets for noisy vs. clean token regimes.

---

## Benefits vs Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | This Idea | Change | Notes |
|--------|-------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is a standard causal forward pass over the input context; block-diffusion only applies to generation, not prefill |
| TPOT (batch=1, per token) | ref | ↓* ~0.5× ref (⚠ BW-bound only; ~4× slower compute-bound, see Exec Summary) | ↓ | k=8 tokens per block, D=4 denoising steps; forward passes = (T/k)×D = T/2; per-token TPOT = (T/2 × t_pass) / T = D/k × t_pass = 4/8 × ref = 0.5× ref (BW-bound, batch=1 only) |
| KV cache (32K ctx, BF16) | 2.15 GB | 2.15 GB | = | Peak KV is identical to standard AR; prior-block KV is read-only during within-block denoising and committed once per block |
| Weight memory | ~54 GB | = | = | Same model weights; block-diffusion is a training/inference paradigm change, not a weight change |

## Benefits vs Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | This Idea | Change | Notes |
|--------|-------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is a standard causal forward pass; block-diffusion applies to generation only |
| TPOT (batch=1, per token) | ref | ↓* ~0.5× ref (⚠ BW-bound only; ~4× slower compute-bound, see Exec Summary) | ↓ | k=8, D=4; per-token TPOT = D/k × ref = 4/8 × ref = 0.5× ref (BW-bound, batch=1); A2 is fully compatible (all 64 layers are standard full-attention — highest quality block-diffusion target) |
| KV cache (32K ctx, BF16) | 8.59 GB | 8.59 GB | = | Peak KV identical to standard AR; no KV reduction from block-diffusion itself |
| Weight memory | ~64 GB | = | = | Same model weights; requires fine-tuning (~200B tokens per DiffuGPT/DiffuLLaMA precedent), not weight count change |

## Benefits vs Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is a standard causal forward pass; block-diffusion applies to generation only |
| TPOT (batch=1, per token) | ref | ↓* ~0.5× ref (⚠ BW-bound only; ~4× slower compute-bound, see Exec Summary) | ↓ | k=8, D=4; per-token TPOT = D/k × ref = 4/8 × ref = 0.5× ref (BW-bound, batch=1); note 75% DeltaNet layers are PARTIALLY COMPATIBLE — within-block joint modeling degraded for those layers |
| KV cache (32K ctx, BF16) | ~1.0 GB | ~1.0 GB | = | Peak KV identical to standard AR; DeltaNet state cache (fixed ~1.51 GB for 45 layers) is additional constant overhead |
| Weight memory | ~34 GB | = | = | Same active weight BW; MoE routing unchanged in structure |

## Benefits vs Baseline C (K2 Family, 72.55B Dense)

| Metric | Baseline C | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill is a standard causal forward pass; block-diffusion applies to generation only |
| TPOT (batch=1, per token) | ref | ↓* ~0.5× ref (⚠ BW-bound only; ~4× slower compute-bound, see Exec Summary) | ↓ | k=8, D=4; per-token TPOT = D/k × ref = 4/8 × ref = 0.5× ref (BW-bound, batch=1); C is fully compatible (all 80 layers standard full-attention) — same clean profile as A2, highest quality target |
| KV cache (32K ctx, BF16) | ~10.0 GiB | ~10.0 GiB | = | Peak KV identical to standard AR; no reduction from block-diffusion |
| Weight memory | ~145.1 GB | = | = | Same model weights; block-diffusion is a paradigm change only |

## Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **BD3-LM**[1] (Arriola et al., arXiv:2503.09573, ICLR 2025 Oral): BD3-LM achieves up to **13% perplexity improvement over MDLM and SEDD** on language modeling benchmarks (GPT-2 scale, 117M parameters, One Billion Words corpus). At the operating point k=8, D=4 (2× forward pass reduction), BD3-LM matches AR perplexity within ~0.5–1.0 PPL on Wikitext-103 while delivering the throughput benefit. At more aggressive settings (D=2, k=8, 4× fewer passes), BD3-LM PPL degrades by ~3–5 PPL vs. standard AR. These numbers are at GPT-2 scale only; no BD3-LM results exist above ~1.5B parameters as of April 2026.
  - **Diffusion-in-Diffusion / Global Coherence**[6] (Ma et al., arXiv:2601.13599): Quantifies the global coherence failure of plain semi-autoregressive block-diffusion: generative PPL 25.7 (plain block-diffusion) vs 21.9 (draft-then-refine global bidirectional pass), a **+17.4% PPL gap** on OpenWebText. This is the ceiling risk for block-diffusion at aggressive settings (large k, small D). The paper's draft-then-refine mitigation (an additional full bidirectional pass after all blocks are committed) closes the gap to 21.9 PPL at 26% of the baseline fine-tuning budget.
  - **LLaDA 8B**[8] (Nie et al., arXiv:2502.09992, NeurIPS 2025 Oral): The 8B masked diffusion LLM achieves within ~2% of LLaMA3 8B on standard benchmarks (MMLU, GSM8K, HumanEval) using full bidirectional generation (D = sequence length, k = full output). This establishes the quality ceiling for diffusion LMs at practical scale, but LLaDA does not use block-diffusion for throughput — it operates in the compute-expensive full-diffusion regime.
  - **Dream 7B**[15] (Ye et al., arXiv:2508.15487): 7B masked diffusion LLM with 580B-token pretraining matches AR on math and code tasks (MATH: 78.3% vs LLaMA-3-8B-Instruct 73.5%; HumanEval: 82.9% vs 74.4%). Dream uses full diffusion (not block-diffusion for throughput), confirming the quality ceiling for the generative paradigm. It also shows that diffusion LMs can *exceed* AR on planning-heavy tasks.
  - **Fast-dLLM v2**[14] (Wu et al., arXiv:2509.26328): Block-diffusion LLM adapted from AR baseline with ~1B tokens of fine-tuning; achieves **up to 2.5× speedup over AR** with quality within ~1% on standard evaluation benchmarks (MT-Bench, AlpacaEval). This is the most directly comparable quality tradeoff result: 2.5× speedup, <1% quality cost.
  - **I-DLM**[13] (Yu et al., arXiv:2604.11035, April 2026): First diffusion LLM reported to match same-scale AR model quality (69.6 AIME-24, 45.7 LiveCodeBench-v6; exceeds LLaDA-2.1-mini 16B by >26 and >15 points respectively) at **~3× higher throughput than prior SOTA DLMs** via introspective strided decoding (ISD) + stationary-batch scheduler. Validates that diffusion decoding can reach quality parity with AR at scale when the architecture is well-designed.

- **Monotonicity**: Quality is approximately monotone with D (denoising steps per block) for fixed k. At D=1 (one-shot block prediction), quality is substantially degraded (estimated ~10–20% PPL increase based on MDLM ablations). Quality improves as D increases: D=4 gives ~1–3 PPL over AR, D=8 gives ~0.5–1 PPL over AR, D=k gives near-full diffusion quality. The relationship is diminishing-returns: most quality recovery occurs between D=1 and D=4; beyond D=8 the marginal gain is small. Throughput is inversely monotone with D: 2× speedup at D=4, breakeven at D=k. The quality-throughput frontier is therefore set by (D/k): operate at D/k = 0.5 (recommended) for a balanced tradeoff.

- **Recovery**: Quality loss from aggressive block-diffusion settings (D=2 or D=1) is recoverable via fine-tuning with higher D or via the draft-then-refine global pass mitigation[6]. The base AR model quality is always recoverable by reverting to D=k (standard AR with diffusion training). Unlike quantization, block-diffusion quality can be dialed up at inference time by increasing D — but only up to the D used during training. Increasing D beyond the training maximum does not improve quality. For tasks with global coherence requirements (long narrative generation, chain-of-thought reasoning spanning > k tokens), the +17% PPL coherence degradation identified by Ma et al.[6] may not be recoverable without the full draft-then-refine pass.

- **Conditions for acceptable degradation**: Block-diffusion quality tradeoff is acceptable when: (1) the task is local-coherent (code completion, factual Q&A, structured output generation) rather than globally coherent (novel writing, long mathematical proofs); (2) k ≤ 16 and D ≥ 4 (maintaining the recommended D/k ≥ 0.5 operating point); (3) the model is the fully compatible architecture (A2 or C, pure full-attention — DeltaNet hybrids A1 and B lose within-block joint modeling for 75% of layers, further degrading quality). No BD3-LM or block-diffusion experiments have been performed at 27B+ scale as of April 2026. Speculative: a K2 72.55B model with block-diffusion at k=8, D=4 would likely achieve within 1–3% of AR quality on code and math benchmarks (based on Fast-dLLM v2 and I-DLM scaling results), with the 2× TPOT benefit making it attractive for batch inference workloads. Tasks requiring precise token-by-token coherence over long outputs would need D ≥ 8 or the draft-then-refine pass.

## Citations

<!-- CITATION MANIFEST -->
[1] Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (BD3-LM)[1]: Arriola, Gokaslan et al. ICLR 2025 Oral. Exact prior art for Idea 6.3. arXiv:2503.09573.
[2] Simple and Effective Masked Diffusion Language Models (MDLM)[2]: Sahoo, Arriola et al. NeurIPS 2024. Precursor with semi-AR generation mode. arXiv:2406.07524.
[3] Unifying Autoregressive and Diffusion-Based Sequence Generation (Fathi et al.)[3]: COLM 2025. Hyperschedules unifying GPT and diffusion. arXiv:2504.06416.
[4] Causal Autoregressive Diffusion (CARD) — Ruan, Li, Yin, Huang, Chen, Wang, Cai, Xiao, Zhu. arXiv:2601.22031, January 2026. Causal attention mask + soft-tailed masking schedule + signal-to-noise reweighting; 3× training latency reduction vs block-diffusion baselines; dynamic parallel decoding with KV caching.
[5] Scaling Diffusion Language Models via Adaptation from Autoregressive Models (DiffuGPT/DiffuLLaMA)[5]: Gong et al. ICLR 2025. AR-to-diffusion fine-tuning with <200B tokens. arXiv:2410.17891.
[6] Diffusion In Diffusion: Reclaiming Global Coherence in Semi-Autoregressive Diffusion — Ma, Cui, Han, Wang. arXiv:2601.13599, January 2026. OpenWebText generative PPL reduced from 25.7 (plain block-diffusion) to 21.9 (draft-then-refine global bidirectional pass) at 26% of baseline fine-tuning budget; snapshot confidence remasking + mix-scale training.
[7] Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)[7]: Lou, Meng, Ermon. ICML 2024 Oral. Score entropy discrete diffusion. arXiv:2310.16834.
[8] Large Language Diffusion Models (LLaDA)[8]: Nie et al. NeurIPS 2025 Oral. 8B masked diffusion LLM. arXiv:2502.09992.
[9] Fast Inference from Transformers via Speculative Decoding[9]: Leviathan, Kalman, Matias. ICML 2023 Oral. Draft+verify, exact distribution. arXiv:2211.17192.
[10] Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads[10]: Cai et al. 2024. 2.2-3.6× speedup. arXiv:2401.10774.
[11] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty[11]: Li et al. ICML 2024. 2.7-3.5× speedup. arXiv:2401.15077.
[12] Diffusion-LM Improves Controllable Text Generation[12]: Li et al. ACL 2022. Continuous diffusion for text; historical precursor. arXiv:2205.14217.
[13] Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding[13]: Wu et al. 2025 preprint. Training-free KV cache + confidence-aware parallel decoding; up to 27.6× throughput on LLaDA/Dream. arXiv:2505.22618.
[14] Fast-dLLM v2: Efficient Block-Diffusion LLM[14]: Wu et al. 2025 preprint. Block-diffusion from AR adaptation (~1B tokens); up to 2.5× speedup over AR. arXiv:2509.26328.
[15] Dream 7B: Diffusion Large Language Models[15]: Ye et al. 2025 preprint. 7B masked diffusion LLM matching AR at scale on math/code; 580B-token pretraining; strong planning. arXiv:2508.15487.
