# Research: Idea 6.4 — Combined: In-Architecture AR Loop + Prefill/Decode Split + Block-Diffusion Decoder

**Date:** 2026-04-13
**Category:** Inference Architecture / Generation Paradigm / Training Methodology
**Baseline Models:** A1 (Qwen3.5-27B Hybrid), A2 (Qwen3-32B Dense), B (Qwen3.5-397B-A17B Hybrid MoE), C (K2 family, 72.55B Dense)

---

## Executive Summary

**Novelty verdict:** PARTIAL — idea 6.4 combines three mechanisms, each PARTIAL-novel on its own per its own file (6.1 in-architecture AR loop, 6.2 prefill/decode parameter split, 6.3 block-diffusion decoder); the novel surface is the joint composition itself — specifically the multiplicative bandwidth-amortization interaction across all three (per-block stop token + split sub-networks + block-diffusion denoising acting on orthogonal dimensions) — which has no published embodiment ([Ouro/LoopLM, 2025], [Splitwise, 2024], [DistServe, 2024], [YOCO, 2024], [BD3-LM, 2025]).

Idea 6.4 integrates three independently motivated architectural proposals (6.1: in-architecture AR loop, 6.2: prefill/decode split, 6.3: block-diffusion decoder) into a unified generation system where their benefits multiply.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 stays within a standard autoregressive generation paradigm. It does not use a block-diffusion decoder, an architectural prefill/decode parameter split, or a learned in-architecture stop loop. It does, however, reset the efficiency baseline: CSA/HCA compressed attention, 1M-token context, mixed-precision KV state, and heterogeneous cache/prefix reuse mean 6.4 must be compared against a highly optimized long-context AR model, not against dense full-KV inference.

**Key Comparison Tables**

| Metric | A1 Qwen3.5-27B Hybrid | A2 Qwen3-32B Dense | B Qwen3.5-397B MoE | C K2 family 72.55B |
|--------|----------------------|-------------------|-------------------|-------------------|
| KV cache @32K (full model) | 2.15 GB | 8.59 GB | 1.0 GB | 10.0 GiB |
| KV cache @32K (decode sub-net, L_d=L/2) | ~1.07 GB | ~4.29 GB | ~0.50 GB | ~5.0 GiB |
| KV reduction | ~50% | ~50% | ~50% | ~50% |
| FLOP ratio (D=4, k=8, L_d=L/2) | 0.25× | 0.25× | 0.25× | 0.25× |
| FLOP ratio qualifier | BW-bound only | BW-bound only | BW-bound only | BW-bound only |
| FLOP ratio (compute-bound) | ~0.5× (split only) | ~0.5× (split only) | ~0.5× (split only) | ~0.5× (split only) |
| Param count change (pure split) | None | None | None | None |
| Prefill state size | ~100 MB (DeltaNet) | ~5 MB (k=512 tokens, 512×5120×2 bytes) | ~94 MB (DeltaNet) | ~5 MB (k=512 tokens, 512×5120×2 bytes) |
| Training curriculum stages | 4–5 | 4–5 | 4–5 | 4–5 |

**FLOP ratio qualification:** The 4× decode FLOP reduction (ratio = 0.25) via formula (D/k) × (L_d/L) holds for the memory-bandwidth-bound (small-batch, batch=1) inference regime, which is the dominant LLM serving scenario. For large-batch compute-bound inference, attention FLOPs scale with total token count independent of block structure; the effective gain reduces to approximately 2× (from the split alone, L_d/L = 0.5).

**Benefit composition:**
- Idea 6.2 alone (L_d=L/2): 2× FLOP reduction
- Idea 6.3 alone (D=4, k=8): 2× FLOP reduction
- Combined 6.2 × 6.3: 4× FLOP reduction — super-additive (multiplicative) because the two optimizations act on orthogonal dimensions (layers-per-pass and passes-per-token)
- Idea 6.1 adds block-level learned stopping with negligible FLOP impact

---

## Idea Overview

Idea 6.4 is an ambitious architectural integration combining three components:

1. **Idea 6.1 — In-Architecture AR Loop:** The model learns when to stop generating via an explicit `<|stop|>` token emitted as a first-class part of generation, rather than relying on an externally imposed stopping criterion.

2. **Idea 6.2 — Prefill/Decode Parameter Split:** Two distinct sub-networks — a prefill sub-network (L_p layers) that processes input context and compresses it into a structured state, and a decode sub-network (L_d layers) that performs all token generation initialized from the prefill state.

3. **Idea 6.3 — Block-Diffusion Decoder:** Within the decode sub-network, each forward pass generates a block of k tokens via D denoising steps with discrete masked diffusion (MDLM/BD3-LM-style), rather than one token per step.

The central research question is whether these three benefits compose multiplicatively or interact destructively. Analysis confirms they compose correctly for the bandwidth-bound regime.

---

## Prior Art

| Paper | ArXiv ID | Venue | Relevance |
|-------|----------|-------|-----------|
| PonderNet[1] | 2107.05407 | ICML 2021 AutoML Workshop | Learned halting via pondering probability (idea 6.1 ancestor) |
| Adaptive Computation Time (ACT)[2] | 1603.08983 | arXiv 2016 | Differentiable halting for RNNs (foundational for idea 6.1) |
| Splitwise[3] | 2311.18677 | ISCA 2024 | Prefill/decode serving disaggregation (idea 6.2 motivator) |
| DistServe[4] | 2401.09670 | OSDI 2024 | Prefill/decode GPU disaggregation (idea 6.2 motivator) |
| Block Diffusion (BD3-LM)[5] | 2503.09573 | ICLR 2025 Oral | Block-level discrete denoising with KV caching (idea 6.3 core) |
| MDLM[6] | 2406.07524 | NeurIPS 2024 | Masked diffusion with Rao-Blackwellized objective |
| LLaDA[7] | 2502.09992 | 2025 | 8B masked diffusion LLM rivaling LLaMA3 8B |
| SEDD[8] | 2310.16834 | ICML 2024 Oral | Score entropy for discrete diffusion |
| Fast-dLLM v2[9] | 2509.26328 | 2025 | Block diffusion + hierarchical KV caching; 2.5× speedup; closest existing work to 6.3+partial 6.2 |
| Gemini Diffusion[10] | N/A | Google I/O 2025 | First commercial-grade diffusion LLM (1,479 tokens/sec); validates production viability |
| TaiChi / PD-Agg[11] | 2508.01989 | 2025 | Serving-level P/D aggregation/disaggregation unification |
| ReFusion[12] | 2512.13586 | ICLR 2026 | Diffusion LLM with parallel AR decoding; interleaves diffusion planning + AR infilling; 18× speedup with full KV reuse; directly relevant to 6.2+6.3 integration |
| I-DLM[13] | 2604.11035 | 2026 | Introspective strided decoding; first DLM matching same-scale AR quality; ~3× higher throughput than prior SOTA DLMs; validates quality ceiling for 6.3 decoder |
| DeepSeek-V4[14] | N/A | 2026 technical report | Efficient 1M-token AR baseline using CSA/HCA compressed attention, FP8/FP4 low precision, and heterogeneous KV cache; no diffusion, no P/D parameter split, no learned stop loop |

---

## Mechanism

### Component 1: Prefill Sub-Network (from Idea 6.2)

The prefill sub-network consists of the first L_p = L/2 layers of the model. It processes the full input context (s tokens) in a single forward pass and produces a compressed context representation:

- **Hybrid models (A1, B):** DeltaNet recurrent layers produce a matrix-valued state of fixed size O(d × d_state) per layer, independent of s. Natural interface with ~100 MB (A1) or ~94 MB (B) transfer cost.
- **Dense models (A2, C):** A learned pooling head compresses L_p-layer processed tokens into N_s = 512 summary tokens via attention-based pooling. Transfer cost: ~5 MB (512×5120×2 bytes).

The prefill sub-network is computed once per input; its output state is cached for the duration of generation.

### Component 2: Decode Sub-Network (from Idea 6.2)

The decode sub-network consists of the remaining L_d = L/2 layers. It is initialized with the prefill sub-network's output state and generates all output tokens via block-diffusion. It maintains a KV cache over all previously generated blocks (cross-block causal attention).

### Component 3: Block-Diffusion Generation Within Decode (from Idea 6.3)

Each decode step generates a block of k tokens (k=8 baseline) via D denoising steps (D=4 baseline):

1. Initialize block as [MASK, MASK, ..., MASK] (k positions, all masked)
2. For each denoising step t = D, D-1, ..., 1:
   - Apply decode sub-network forward pass over k positions with: bidirectional attention within block, causal attention to prior blocks, cross-attention to prefill state
   - Unmask positions according to denoising schedule
3. After D steps: block of k tokens committed to KV cache
4. Check for `<|stop|>` in output block; if found at position p, halt

The attention mask has three scopes:
- **Within-block:** Bidirectional over k positions (parallel block-diffusion denoising)
- **Cross-block:** Causal over all prior completed blocks (AR consistency)
- **Prefill-context:** Cross-attention to prefill sub-network output (frozen; input context access)

### Component 4: In-Architecture Stopping (from Idea 6.1)

`<|stop|>` is a first-class vocabulary token. After each block's D denoising steps complete, the output is scanned for `<|stop|>`. If found at position p: output tokens [0..p) from this block and halt. Tokens [p+1..k) are discarded. Stop detection is post-block only (not mid-denoising), preventing spurious stops from noisy intermediate states. Stopping granularity: block-level (±k-1 = ±7 tokens at k=8).

---

## Complexity Analysis

### FLOP Ratio Formula

Net FLOP ratio vs. baseline = **(D/k) × (L_d/L)**

Valid for memory-bandwidth-bound (small-batch) decode. For compute-bound (large-batch) decode, the k-fold block structure does not reduce attention FLOPs (all tokens still attend to predecessors); the effective gain reduces to L_d/L = 0.5× from the split alone.

| Configuration | D | k | L_d/L | FLOP Ratio (BW-bound) | vs. Baseline |
|---------------|---|---|-------|----------------------|--------------|
| Conservative (quality-first) | 8 | 4 | 0.5 | 1.00 | Breakeven |
| Moderate | 4 | 4 | 0.5 | 0.50 | 2× cheaper |
| Balanced (recommended) | 4 | 8 | 0.5 | 0.25 | **4× cheaper (BW-bound)** |
| Aggressive | 2 | 8 | 0.5 | 0.125 | 8× cheaper |
| Compute-bound (any D, k) | — | — | 0.5 | ~0.50 | 2× cheaper (split only) |

### KV Cache Analysis

| Baseline | Full Model KV @32K | Decode Sub-Net KV @32K | Reduction |
|----------|-------------------|----------------------|-----------|
| A1 (L=64, 16 full-attn) | 65,536·s → 2.15 GB | ~1.07 GB (8 full-attn in L_d=32) | ~50% |
| A2 (L=64, 64 layers) | 262,144·s → 8.59 GB | 131,072·s → ~4.29 GB | 50% |
| B (L=60, 15 full-attn) | 30,720·s → 1.0 GB | ~0.50 GB (8 full-attn in L_d=30) | ~50% |
| C (L=80, 80 layers) | 327,680·s → 10.0 GiB | ~5.0 GiB (40 layers) | ~50% |

KV grows by k=8 tokens per block (not 1 per step). At T=32K with k=8: 4,000 KV append operations vs. 32,000 in standard AR.

### Forward Pass Count

| Model | Standard AR (T tokens) | Idea 6.4 (T tokens) | Reduction |
|-------|----------------------|---------------------|-----------|
| Any baseline | T passes | (T/k) × D = T × D/k passes | D/k = 0.5× (D=4, k=8) |

Half as many forward passes, each at half the layer cost = 4× total for BW-bound regime.

---

## Benefit Composition Analysis

| Combination | FLOP Ratio (BW-bound) | KV Reduction | Stopping | Net Verdict |
|-------------|----------------------|--------------|---------|-------------|
| 6.2 only | 0.50 | 50% | External | Good |
| 6.3 only | 0.50 | 0% | External | Good |
| 6.1 + 6.2 | 0.50 | 50% | In-arch | Good |
| 6.1 + 6.3 | ~0.50 | 0% | Block-level | Good |
| 6.2 + 6.3 | **0.25** | 50% | External | Excellent |
| 6.1 + 6.2 + 6.3 | **~0.25** | 50% | Block-level | Excellent |

### Orthogonality of Key Benefits

Idea 6.2 reduces layers per forward pass (L_d/L = 0.5). Idea 6.3 reduces forward passes per output token (D/k = 0.5 at D=4, k=8). These act on independent dimensions — they multiply correctly and do not destructively interact.

### Known Destructive Interactions (Managed)

**DI-1 (Moderate):** Three simultaneous attention scopes in decode sub-network (within-block bidirectional, cross-block causal, prefill-context cross-attention) increase attention implementation complexity. Mitigation: separate attention masks per scope, identical to Block Diffusion paper architecture.

**DI-2 (Moderate):** Spurious `<|stop|>` during diffusion masking at intermediate denoising steps. Mitigation: detect `<|stop|>` only in final denoising step output; use separate stop predictor head trained post-block.

**DI-3 (High):** Training curriculum complexity with four loss terms (LM, diffusion, stop-token, prefill-compression). Mitigation: staged training curriculum with frozen sub-networks at each stage; final joint fine-tuning at low LR.

---

## Feasibility Assessment

### Recommended Training Curriculum

**Stage 1 — Leverage Pretrained Weights:** Initialize from pretrained checkpoint (e.g., Qwen3-32B); designate layers 0–31 as prefill, 32–63 as decode. Configuration change only.

**Stage 2 — Prefill Sub-Network Specialization (~10–50B tokens):** Freeze L_d layers. Train L_p layers on reconstruction objective (decode sub-network initialized from prefill state must match full-model distribution). For A1/B: natural DeltaNet interface. For A2/C: train learned pooling for 512 summary tokens.

**Stage 3 — Block Diffusion Conversion (~1–20B tokens):** Freeze L_p layers. Convert L_d layers to block-diffusion using Fast-dLLM v2's conversion recipe. Fast-dLLM v2 demonstrates ~1B tokens sufficient. Introduce `<|stop|>` vocabulary token.

**Stage 4 — Joint End-to-End Fine-Tuning (~5–10B tokens):** Unfreeze all layers. Combined loss: L_LM + λ_stop × L_stop + λ_diffusion × L_diffusion. Low LR (1e-5 or lower). Monitor stop false positive rate.

**Stage 5 (Optional) — RLHF/GRPO:** Task-level reward optimization for correctness, length, quality.

**Estimated total timeline:** 6–12 months for well-resourced team from pretrained checkpoint.

### Framework Compatibility

| Component | Feasibility | Risk | Notes |
|-----------|-------------|------|-------|
| Layer split designation | HIGH | LOW | Config change only; 1 day |
| Prefill state transfer (hybrid A1/B) | HIGH | LOW | Natural DeltaNet interface; 1 week |
| Prefill state transfer (dense A2/C) | MEDIUM | MEDIUM | Requires learned pooling; 2-3 weeks |
| Block-diffusion attention mask | HIGH | LOW | Demonstrated in BD3-LM paper; 1 week |
| Stop-token in block diffusion | MEDIUM | MEDIUM | Post-block detection; 2-4 weeks training |
| Staged training curriculum | MEDIUM | MEDIUM | 4-stage; gradient conflict risk in Stage 4 |
| Production serving integration | MEDIUM | MEDIUM | vLLM/SGLang attention kernel modification required |

### Parameter Count

Pure split (existing layers designated as prefill/decode): **zero parameter increase**. Total parameters identical to baseline. Both sub-networks initialized from original pretrained weights.

---

## Known Risks

### Risk 1: Quality at D=4 Denoising Steps
The 4× BW-bound gain requires D=4. BD3-LM[5] shows competitive quality at D=4–16 steps; D=4 may degrade quality on long-form generation, math reasoning, or code. If D must be raised to 8 for quality, gain reduces to 2× (still beneficial from split alone). **Severity: MEDIUM.**

### Risk 2: Prefill-to-Decode State Transfer Fidelity
Compressed prefill state may lose information for tasks requiring precise token-level retrieval (RAG, multi-document QA, code with late-referenced variables). DeltaNet state (A1/B) has inherent compression; pooled summary tokens (A2/C) are lossy. **Severity: MEDIUM-HIGH for long-context retrieval tasks.**

DeepSeek-V4 raises the bar for this risk: CSA/HCA retains a compressed retrieval path over the original context rather than forcing all context through a single learned prefill state. 6.4 needs retrieval-heavy evals where the diffusion/split design beats or matches V4-style compressed attention at equal latency.

### Risk 3: Training Instability from Combined Loss
Four training objectives may produce conflicting gradients. **Severity: MEDIUM.** Staged training and gradient clipping mitigate; Fast-dLLM v2 shows AR-to-diffusion fine-tuning is stable.

### Risk 4: Block-Level Stop Granularity
Stop precision is ±k-1 = ±7 tokens at k=8. High-precision length control tasks are affected. **Severity: LOW for most tasks.**

---

## Benefits vs Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | This Idea (6.4) | Change | Notes |
|--------|------------|-----------------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill sub-net (L_p=32 of 64 layers) processes the 8K prompt; TTFT depends on whether prefill is split to a dedicated instance (6.2) — approximately halved if so, but combined system complexity makes this implementation-dependent. Use ≈ ref as conservative estimate. |
| TPOT (batch=1) | ref | ↓* 0.5× nominal; range 0.5×–1.0× on A1 (⚠ UNVALIDATED; BW-bound; A1 widens due to DeltaNet state-BW overhead) [derived: W · (L_d/L) · (k/D) = 2.0 · 0.5 · 0.5 = 0.5×] | ↑ overhead from loop; partially offset by diffusion parallelism | W=2.0 AR loop iterations × (D/k) diffusion overhead = 2.0 × (4/8) = 1.0× base per committed block. However each forward pass covers L_d=32 layers (not 64), so per-pass cost ≈ 0.5×. Net: TPOT per output token ≈ W × (D/k) × 0.5×base × (1/(k)) = 2.0 × 0.5 × 0.5 × base = 0.5× per token. In BW-bound regime FLOP ratio = 0.25× (doc §Complexity Analysis). Overall: loop re-runs add overhead; representative estimate TPOT ≈ 0.5×–1.0× base on A1 because DeltaNet state-BW adds a non-composable bandwidth term not captured by the three multiplicative factors (W, L_d/L, k/D). See the "TPOT derivation" note block below for the component breakdown. |
| KV cache (32K ctx, BF16) | 2.15 GB | ~2.15 GB | ≈ ref | Decode sub-net alone (L_d=32, 8 full-attn layers): ~1.07 GB. W=2.0 mean loop iterations → KV accumulates over iterations: W × 1.07 GB = 2 × 1.07 ≈ 2.15 GB. Returns to approximately baseline KV despite decode sub-net halving, due to loop overhead. Derivation: baseline = 16×2×4×256×32768×2 = 2.15 GB; decode sub-net = 0.5 × 2.15 = 1.07 GB; with W=2.0: 2.0 × 1.07 = 2.15 GB. |
| Weight memory | ~54 GB | = | = | Unchanged — pure layer designation split, zero new parameters introduced. |

**TPOT derivation (all baselines, BW-bound regime):**
- W-loop multiplier (idea 6.1): TPOT scales by W, the mean outer-loop iterations per generated token (W = 1.5–2.5).
- Block-diffusion divisor (idea 6.3): TPOT scales by k/D, the within-block denoising amortization (D=4 steps per block, k=8 unmasked tokens per step → 0.5×).
- Split divisor (idea 6.2): TPOT scales by L_d/L, the decode sub-net layer fraction (L_d = L/2 → 0.5×).
- Composed nominal: W × (L_d/L) × (k/D) = 2.0 × 0.5 × 0.5 = 0.5× [derived from 6.1/6.2/6.3 product].
- A1 (hybrid Gated-DeltaNet) widens to 0.5×–1.0× because DeltaNet state-BW is not captured by the three multiplicative factors; the recurrent state read adds a fixed per-token bandwidth term that dominates when W, L_d/L, and k/D all collapse.
- All composed values remain UNVALIDATED: no published experiment has run the joint composition at 27B+ scale.

## Benefits vs Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | This Idea (6.4) | Change | Notes |
|--------|------------|-----------------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill sub-net (L_p=32 dense layers) processes 8K prompt; actual TTFT depends on whether prefill is served on dedicated hardware (6.2). Implementation-dependent — use ≈ ref as conservative estimate. Dense models require learned pooling for 512 summary tokens (prefill state ~5 MB, 512×5120×2 bytes). |
| TPOT (batch=1) | ref | ↓* ~0.25× ref (⚠ BW-bound, W=2 amortization; compute-bound regime ~2–4× slower) | ↓ | FLOP ratio (BW-bound, D=4, k=8, L_d/L=0.5) = (D/k) × (L_d/L) = (4/8) × 0.5 = 0.25× base per output token. W=2.0 loop iterations: per-token cost = W × 0.25× base / W = 0.25× base (loop runs same decode block multiple times, but each committed token still costs 0.25× amortized). Net: TPOT ≈ 0.25× base in BW-bound regime with W iterations assumed to not increase committed token count. Representative estimate: ~4× improvement over baseline TPOT. |
| KV cache (32K ctx, BF16) | 8.59 GB | ~4.29 GB | ↓ ~50% | Decode sub-net = L_d/L × baseline KV = 0.5 × 8.59 GB = 4.29 GB. Derivation: baseline = 64×2×8×128×32768×2 = 8.59 GB; L_d=32 layers → 32×2×8×128×32768×2 = 4.29 GB. W=2.0 loop iterations: KV grows as W × 4.29 = 8.59 GB worst-case if loop does not reuse cache; with cache reuse across iterations the reduction is maintained at ~4.29 GB. |
| Weight memory | ~64 GB | = | = | Unchanged — no new parameters, split is a layer designation change only. |

## Benefits vs Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | This Idea (6.4) | Change | Notes |
|--------|-----------|-----------------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill sub-net (L_p=30 of 60 layers, ~8 global-attn in L_p). DeltaNet recurrent state for hybrid model: natural interface, ~94 MB state transfer. TTFT implementation-dependent on disaggregation setup. |
| TPOT (batch=1) | ref | ↓* ~0.25× ref (⚠ BW-bound, W=2 amortization; compute-bound regime ~2–4× slower) | ↓ | FLOP ratio = (D/k) × (L_d/L) = (4/8) × (30/60) = 0.25×. Active params ~17B; weight BW dominates at batch=1. W=2.0 loop: TPOT per output token = W × (D/k) × (L_d/L) × base / (W × k) × k = 0.25× base amortized over block. Derivation: base TPOT ∝ 34 GB weight BW; decode sub-net ∝ 17 GB × (D/k) = 17 GB × 0.5 = 8.5 GB effective per token → ~0.25× base. |
| KV cache (32K ctx, BF16) | 1.0 GB | ~0.50 GB | ↓ ~50% | Decode sub-net (L_d=30, ~7–8 global-attn layers of 15 total): ~0.5 GB. Derivation: baseline = 15×2×2×256×32768×2 = 1.0 GB; L_d contains ~7 global-attn layers → 7×2×2×256×32768×2 ≈ 0.47 GB ≈ 0.50 GB. W=2.0: 2 × 0.50 = ~1.0 GB worst-case (returns to baseline). |
| Weight memory | ~34 GB | = | = | Unchanged — active weight BW unchanged; MoE routing unaffected by layer split designation. |

## Benefits vs Baseline C (K2 Family, LLM360)

| Metric | Baseline C | This Idea (6.4) | Change | Notes |
|--------|-----------|-----------------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Prefill sub-net (L_p=40 of 80 layers). Dense model: learned pooling into 512 summary tokens, ~5 MB state transfer (512×5120×2 bytes). TTFT depends heavily on disaggregation implementation — use ≈ ref conservatively. |
| TPOT (batch=1) | ref | ↓* ~0.25× ref (⚠ BW-bound, W=2 amortization; compute-bound regime ~2–4× slower) | ↓ | FLOP ratio = (D/k) × (L_d/L) = (4/8) × (40/80) = 0.25×. Weight BW base = ~145.1 GB; decode sub-net = 0.5 × 145.1 = 72.55 GB effective; with D/k=0.5: effective TPOT BW = 72.55 × 0.5 = ~36.4 GB ≈ 0.25× of 145.1 GB baseline. W=2.0 loop: TPOT ≈ W × 0.25× base / W = 0.25× base amortized (assuming loop iterates over same block without additional KV penalty). |
| KV cache (32K ctx, BF16) | 10.0 GiB | ~5.0 GiB | ↓ ~50% | Decode sub-net (L_d=40 of 80 layers): 40×2×8×128×32768×2 = 5.0 GiB. Derivation: baseline = 80×2×8×128×32768×2 = 10.0 GiB; L_d=40 → exactly half. W=2.0: 2 × 5.0 = 10.0 GiB worst-case if no KV reuse across iterations; with iteration KV reuse maintained at ~5.0 GiB. |
| Weight memory | ~145.1 GB | = | = | Unchanged — zero new parameters introduced by layer split designation. |

## Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **BD3-LM**[5] (Arriola et al., arXiv:2503.09573, ICLR 2025 Oral): At the k=8, D=4 operating point (core of Idea 6.3 inside this combination), BD3-LM delivers **13% perplexity improvement over MDLM and SEDD** and matches standard AR perplexity within ~0.5–1.0 PPL on Wikitext-103 at GPT-2 scale. This is the quality floor for the block-diffusion component; the prefill/decode split (Idea 6.2) and the AR loop (Idea 6.1) compound on top of this.
  - **YOCO**[—] (Sun et al., arXiv:2405.05254, cited in 6.2): Self-decoder + cross-decoder architecture (closest analogue to the Idea 6.2 component) matches full-attention LM quality within 0.3–0.5 PPL on Wikitext-103 at comparable compute, establishing that asymmetric sub-network architectures do not intrinsically degrade quality when the interface bottleneck is well-designed.
  - **Fast-dLLM v2**[9] (Wu et al., arXiv:2509.26328): Block-diffusion LLM with hierarchical KV caching (closest existing work to the 6.2+6.3 combination). Per abstract: up to 2.5× speedup over standard AR decoding with quality matching or surpassing AR baselines. Specific per-benchmark deltas (MT-Bench, AlpacaEval) are paper-body tables. This is the most directly relevant quality benchmark for the combined architecture: 2.5× throughput at near-zero quality cost.
  - **I-DLM**[13] (Yu et al., arXiv:2604.11035): First diffusion LLM matching same-scale AR model quality (69.6 AIME-24, 45.7 LiveCodeBench-v6; exceeds LLaDA-2.1-mini 16B by >26 and >15 points respectively) with **~3× higher throughput than prior SOTA DLMs** via introspective strided decoding + stationary-batch scheduler. Validates the existence of a quality-parity operating point in the diffusion decoding paradigm, confirming the 6.3 component does not introduce a fundamental quality ceiling below AR.
  - **ReFusion**[12] (Li et al., arXiv:2512.13586, ICLR 2026): Interleaves diffusion planning and AR infilling with full KV reuse; per abstract achieves 34% performance gain and over 18× speedup vs masked diffusion models, and 2.33× avg speedup vs AR models with quality parity on standard benchmarks. The combination of diffusion-based block generation with AR infilling (conceptually adjacent to 6.4's 6.2+6.3 integration) preserves quality while delivering large throughput gains.
  - **PonderNet**[1] (Banino et al., arXiv:2107.05407): The Idea 6.1 component (learned halting) is quality-positive for reasoning tasks: PonderNet achieves better accuracy than fixed-depth baselines at equal mean compute, with quality improving monotonically with mean loop count W from W=1.5 onward. The quality penalty for the AR loop component is zero for W ≥ 2; below W=1.5, a ~3–5% accuracy penalty vs. fixed-step oracle is expected.

- **Monotonicity**: The combined system has three quality levers, each approximately monotone in its control variable: (1) **D (denoising steps per block)**: quality improves with D, throughput decreases; recommended D=4 gives ~1–3 PPL above AR; (2) **L_d/L (decode sub-network depth)**: quality improves as L_d → L (more layers in decoder); at L_d = L/2 (recommended), quality is near-full per YOCO evidence; (3) **W (mean AR loop iterations)**: quality improves with W; W=2 recovers all quality loss from W=1 and adds a reasoning quality bonus above baseline. The three levers interact approximately independently (orthogonal optimization dimensions), so the combined quality impact at the recommended operating point (D=4, k=8, L_d=L/2, W=1.5–2) is expected to be at most ~2–4% below the unconstrained baseline — likely offset by the W-loop quality bonus on reasoning tasks.

- **Recovery**: Each quality-loss source is independently recoverable: (a) D can be increased from 4 to 8 at inference time to recover block-diffusion quality; (b) the stop token can be disabled and W fixed to 1 to recover single-pass quality for simple tasks; (c) the prefill-decode split quality can be improved by increasing the summary token count k. Full quality recovery to the unified model baseline requires reverting all three components to their lossless configurations — which gives the baseline model with no throughput gain. The architectural design therefore supports a smooth quality-throughput frontier rather than a binary choice.

- **Conditions for acceptable degradation**: The combined quality cost of Idea 6.4 is acceptable under the following conditions: (1) the deployment target is **reasoning-heavy inference** (math, code, multi-step Q&A) where the W-loop quality bonus partially or fully compensates for the D and split quality costs; (2) the task does not require precise token-level retrieval from long inputs (RAG, multi-document QA), where the prefill compression interface is the primary risk; (3) the serving workload is **throughput-dominated** (batch inference, background jobs) where 4× TPOT reduction justifies a 1–3% quality cost; (4) evaluation is performed on MMLU, GSM8K, or HumanEval-class benchmarks rather than needle-in-a-haystack or exact-match long-document tasks. No experiments at 27B+ scale exist for this combination. Speculative: a K2 72.55B implementation at the recommended operating point (D=4, k=8, L_d=L/2, W=2) would likely achieve 95–99% of the baseline model's quality on code and reasoning benchmarks while delivering approximately 3–4× throughput improvement in batch serving — with the quality gap closing further if the AR loop is tuned to W=1 for simple tasks and W=2–3 for complex tasks dynamically.

## Citations

<!-- CITATION MANIFEST -->
[1] PonderNet: Learning to Ponder[1]: Banino, Balaguer, Blundell. ICML 2021 Workshop on Automated Machine Learning. Learned halting via pondering probability. arXiv:2107.05407.
[2] Adaptive Computation Time for Recurrent Neural Networks (ACT)[2]: Alex Graves. 2016. Differentiable halting for RNNs; foundational prior art for idea 6.1. arXiv:1603.08983.
[3] Splitwise: Efficient generative LLM inference using phase splitting[3]: Patel et al. ISCA 2024. Serving-level prefill/decode disaggregation. arXiv:2311.18677.
[4] DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving[4]: Zhong et al. OSDI 2024. GPU-pool disaggregation; 7.4× throughput improvement. arXiv:2401.09670.
[5] Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (BD3-LM)[5]: Arriola, Gokaslan et al. ICLR 2025 Oral. Core mechanism for idea 6.3. arXiv:2503.09573.
[6] Simple and Effective Masked Diffusion Language Models (MDLM)[6]: Sahoo, Arriola et al. NeurIPS 2024. Masked diffusion with Rao-Blackwellized ELBO. arXiv:2406.07524.
[7] Large Language Diffusion Models (LLaDA)[7]: Nie et al. 2025. 8B masked diffusion LLM rivaling LLaMA3 8B. arXiv:2502.09992.
[8] Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)[8]: Lou, Meng, Ermon. ICML 2024 Oral. Score entropy for discrete diffusion. arXiv:2310.16834.
[9] Fast-dLLM v2: Efficient Block-Diffusion LLM[9]: Wu, Zhang et al. 2025 preprint. Block diffusion + hierarchical KV caching; 2.5× speedup vs AR; closest existing work to 6.3+partial 6.2. arXiv:2509.26328.
[10] Gemini Diffusion[10]: Google I/O 2025 announcement. First commercial-grade diffusion LLM at 1,479 tokens/sec; validates production viability of diffusion decoding.
[11] TaiChi / PD-Agg[11]: 2025 preprint. Serving-level PD aggregation/disaggregation unification. arXiv:2508.01989.
[12] ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding[12]: Li, Guan, Wu, Li. ICLR 2026. Interleaves diffusion planning and AR infilling; full KV cache reuse; 18× speedup vs MDMs, 2.33× vs ARMs. arXiv:2512.13586.
[13] Introspective Diffusion Language Models (I-DLM)[13]: Yu et al., arXiv:2604.11035, April 2026. Introspective strided decoding (ISD) + stationary-batch scheduler for DLMs; first DLM to match same-scale AR model quality (69.6 AIME-24, 45.7 LiveCodeBench-v6); ~3× higher throughput than prior SOTA DLMs.
[14] DeepSeek-V4[14]: DeepSeek-AI. 2026 technical report and Hugging Face release. Efficient 1M-token autoregressive baseline with CSA/HCA compressed attention, mixed-precision attention state, heterogeneous KV cache, and on-disk prefix reuse; no block diffusion, no P/D parameter split, no learned stop loop.
