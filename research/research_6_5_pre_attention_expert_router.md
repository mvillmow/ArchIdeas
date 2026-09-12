# Research: Idea 6.5 — Pre-Attention Expert Router

**Date:** 2026-04-14
**Category:** MoE Architecture / Expert Routing
**Baseline Models:** A1 (Qwen3.5-27B Hybrid), A2 (Qwen3-32B Dense), B (Qwen3.5-397B-A17B Hybrid MoE), C (K2 family, 72.55B Dense)

---

## Executive Summary

**Novelty verdict:** NOVEL — no published MoE places the FFN expert router before attention (or in parallel with it, or conditioned on Q / K / Q+K projections); SwitchHead applies MoE to attention V/O projections with a pre-attention router but does not route FFN experts, and MoA/MoH route attention heads rather than FFN experts ([SwitchHead, 2024], [MoA, 2024], [MoH, 2024], [Switch Transformer, 2022], [Mixtral, 2024]).

Idea 6.5 repositions the MoE FFN expert router to operate on pre-attention representations (residual stream, Q projection, K projection, or Q+K pair) instead of the standard post-attention residual, changing what information drives expert selection without changing total FLOPs or memory cost.

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 keeps FFN expert routing separate from attention representations but changes the routing baseline: all transformer blocks are MoE, the first 3 MoE layers use content-free Hash routing, routed expert activation is fixed at 6 experts plus 1 shared expert, router affinity uses Sqrt(Softplus), and load balance combines auxiliary-loss-free routing with sequence-wise balance loss. This does not invalidate 6.5's pre-attention routing novelty, but it adds a required ablation: compare pre-attention FFN routing against V4-style early Hash routing and modern balance losses, not only against older post-attention TopK routers.

**Key Comparison Tables**

| Metric | A1 Qwen3.5-27B Hybrid | A2 Qwen3-32B Dense | B Qwen3.5-397B MoE | C K2 family 72.55B |
|--------|----------------------|-------------------|-------------------|-------------------|
| KV cache @32K | 2.15 GB | 8.59 GB | 1.0 GB | 10.0 GiB |
| KV cache change from N5 | 0% | 0% | 0% | 0% |
| TTFT change | ~0% | ~0% | ~0% | ~0% |
| TPOT change | ~0% | ~0% | ~0% | ~0% |
| Router FLOPs per layer (V1/V2/V3, E=8) | ~82K | ~82K | N/A (E=512 → 4.2M) | ~131K |
| Router fraction of MLP FLOPs | ~0.046% | ~0.031% | ~0.5% (B, E=512) | ~0.028% |
| Parameter count change | 0 (V1–V3) | 0 (V1–V3) | 0 (V1–V3) | 0 (V1–V3) |
| Expert specialisation shift | Hypothetical: content→query-type | Hypothetical | Hypothetical | Hypothetical |
| Load balance improvement | Empirically unvalidated | Empirically unvalidated | Empirically unvalidated | Empirically unvalidated |

**NOTE:** Expert specialisation claims and load balance improvement hypotheses are **empirically unvalidated; require ablation study**. These are theoretical predictions, not confirmed results. They must not be stated as facts in any derivative publication without supporting experimental evidence.

---

## Idea Overview

In a standard MoE Transformer, the expert router is applied after the multi-head attention sub-layer. It operates on the post-attention residual stream: `x + Attn(LN(x))`, which has already aggregated cross-token contextual information.

The N5 proposal repositions the expert router to operate *before* or *in parallel with* attention, routing on representations that have not yet been contextualised by cross-token attention.

### Five Variants

**Variant 1 (V1) — Router on pre-attention residual:**
Router receives the residual stream `x` before the attention projection. In the serial-before form: Router(x) → Attention(x) → MLP[selected experts]. In the parallel form: Router(x) and Attention(x) run concurrently on the same input; expert dispatch waits for the routing decision while attention computes.

**Variant 2 (V2) — Router on Q projection:**
After computing `Q = x · Wq`, but before the softmax attention, the router receives Q as its input: `g = Router(sg(Q))`. The router sees what the current token is *searching for* rather than what it has already found. Stop-gradient `sg(·)` on Q is required to prevent router loss from biasing Wq training.

**Variant 3 (V3) — Router on K projection:**
After computing `K = x · Wk`, the router receives K as its input: `g = Router(sg(K))`. The router sees the current token's *key content* — what it offers to other tokens. Stop-gradient required.

**Variant 4 (V4) — Router on (Q,K) pair:**
After computing both Q and K, the router receives their concatenation: `g = Router(sg([Q; K]))`. Captures both query intent and key content. Router weight doubles to 2d × E_total (minor parameter increase).

**Variant 5 (V5) — Joint attention-routing:**
The routing decision and attention pattern are computed jointly. Attention weights or logits inform the routing gate, enabling content-adaptive expert selection aware of which tokens this token will attend to. Architecture-dependent; deferred to after V1–V4 validation.

### Core Research Questions

1. **Does pre-attention routing change what the router can see?** Yes. Pre-attention router operates on the local (non-contextualised) token representation. Post-attention router sees the fully contextualised representation (e.g., "bank" in financial context vs. geographic context). This is a fundamental information-theoretic difference.

2. **Does it change expert specialisation?** *Hypothesis (empirically unvalidated; requires ablation study):* experts may specialise on query-type (what the token seeks) rather than output-representation type (what semantic content the token carries after contextualisation). This shift is analogous to the difference between syntactic and semantic routing.

3. **Does it improve load balance?** *Hypothesis (empirically unvalidated; requires ablation study):* pre-attention representations may be less clustered than post-attention representations, potentially improving expert load balance. This is testable empirically by comparing expert activation entropy across routing placement conditions.

---

## Literature Review

### Standard Post-Attention MoE (All Existing Practice)

| Paper | arXiv ID | Year | Relevance |
|-------|----------|------|-----------|
| Switch Transformer (Fedus, Zoph, Shazeer)[1] | 2101.03961 | 2022 | Standard top-1 router on post-attention residual |
| GShard (Lepikhin et al.)[2] | 2006.16668 | 2021 | Top-2 routing with capacity constraints |
| Mixtral of Experts (Jiang et al.)[3] | 2401.04088 | 2024 | 8×7B sparse MoE; standard post-attention router |
| DeepSeek-MoE (Dai et al.)[4] | 2401.06066 | 2024 | Fine-grained expert segmentation; expert specialisation analysis |
| Expert Choice Routing (Zhou et al.)[5] | 2202.09368 | 2022 | Inverted routing; experts choose tokens |

### Critical Adjacent Work (Pre-Attention Routing for Attention, Not FFN)

| Paper | arXiv ID | Year | Overlap with N5 |
|-------|----------|------|-----------------|
| SwitchHead (Csordas, Piękos, Irie, Schmidhuber)[6] | 2312.07987 | 2024 | CRITICAL — MoE on V and O attention projections; tested Q/K and found unnecessary for attention heads. Does NOT route FFN experts on Q/K. |
| Mixture of Attention Heads: Selecting Attention Heads Per Token (MoA, Zhang et al.)[7] | 2210.05144 | 2022 | Routes attention heads using a per-token router; does NOT route FFN experts. |
| MoH: Multi-Head as MoE (Jin et al.)[8] | 2410.11842 | 2024 | Routes attention heads on pre-attention residual x; does NOT route FFN experts. |
| Mixture of Sparse Attention (MoSA)[9] | 2505.00315 | 2025 | Routes attention sparsity per head; does NOT route FFN experts on Q/K. |
| UMoE: Unifying Attention and FFN with Shared Experts (Yang et al.)[19] | 2505.07260 | 2025 | Reformulates attention to expose FFN-like expert structure; generates per-expert Q projections before token mixing. Does NOT place FFN router before attention in the residual sense, but architecturally adjacent to V2/V4. |
| ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing (ICLR 2025)[20] | 2412.14711 | 2025 | Replaces TopK+Softmax routing with ReLU gating; fully differentiable; directly applicable as the router function in V1–V4 variants. |

### MoE Design and Implementation Literature

| Paper | arXiv ID | Year | Relevance |
|-------|----------|------|-----------|
| V-MoE (Riquelme et al.)[10] | 2106.05974 | 2021 | Sparse MoE for Vision Transformers; routing on less-contextualised patch features |
| Soft-MoE (Puigcerver et al.)[11] | 2308.00951 | 2023 | Differentiable routing; potential combination with Q-based routing |
| MegaBlocks (Gale et al.)[12] | 2211.15841 | 2022 | Block-sparse GPU kernels; placement-agnostic dispatch |
| Tutel (Hwang et al.)[13] | 2206.03382 | 2023 | Adaptive MoE pipelining; compatible with parallel-branch V1 |
| MoE-Mamba (Pióro et al.)[14] | 2401.04081 | 2024 | MoE on non-standard (Mamba) representations; confirms routing placement flexibility |
| Branch-Train-Merge (Li et al.)[15] | 2208.03306 | 2022 | Domain-specialised expert LMs; relevant to specialisation hypothesis |
| Sparse Upcycling (Komatsuzaki et al.)[16] | 2212.05055 | 2022 | Dense-to-MoE conversion; relevant to N5 as a upcycling target |
| Hash-routing (Roller et al.)[17] | 2106.04426 | 2021 | Content-free routing baseline; relevant for ablation against N5 variants |
| MoE Design Choices (arXiv:2402.13089)[18] | 2402.13089 | 2024 | Empirical MoE ablations; does not test pre-attention placement |
| DeepSeek-V4 Early Hash MoE (DeepSeek-AI)[21] | 2026 technical report | 2026 | Uses Hash routing in the first 3 all-MoE layers, fixed 6 routed experts, Sqrt(Softplus) affinity, and sequence-wise balance loss; required modern baseline for early-layer routing ablations |

---

## Prior Art Classification

### Summary Table

| Variant | Description | Classification | Justification |
|---------|-------------|----------------|---------------|
| V1 serial-before | FFN router on pre-attention residual x, attention runs after | **NOVEL** | No published MoE places FFN router before attention. MoH/SwitchHead route attention heads (not FFN experts). |
| V1 parallel | FFN router and attention run concurrently on same x | **NOVEL** | No paper implements attention+FFN-MoE as parallel branches. |
| V2 Q-only router | FFN router receives Q projection as input | **NOVEL** | MoA uses MoE on Q for attention head selection; no paper uses Q for FFN expert selection. |
| V3 K-only router | FFN router receives K projection as input | **NOVEL** | No prior art found. |
| V4 Q+K router | FFN router on Q+K concatenation | **NOVEL** | No prior art found. |
| V5 joint | Routing and attention computed jointly | **NOVEL** | No paper implements joint attention-FFN routing. |

**Overall N5 Prior Art Classification: NOVEL** (all 6 sub-variants)

The NOVEL classification is upheld against the only potential challenge: SwitchHead applies MoE to attention projections (V, O) using a pre-attention router — but does not route FFN experts before attention. The tasks are distinct.

DeepSeek-V4 does not route FFN experts on pre-attention Q/K/residual representations, so it does not create direct prior art for 6.5. It does reduce the claimable gap around "early routing instability" because Hash routing in the first 3 layers is now a frontier-scale deployed solution.

---

## Technical Analysis

### Router FLOPs (Placement-Independent)

Router FLOPs = 2 × d × E_total regardless of placement (V1 through V4 all receive a d-dimensional input vector).

| Baseline | d | E_total | Router FLOPs (V1–V3) | Router FLOPs (V4: 2d) | MLP FLOPs/layer | Router fraction |
|----------|---|---------|----------------------|------------------------|-----------------|-----------------|
| A1 (Qwen3.5-27B Hybrid) | 5,120 | 8 | 81,920 | 163,840 | ~178M (d_ff=17408) | ~0.046% |
| A2 (Qwen3-32B Dense) | 5,120 | 8 | 81,920 | 163,840 | ~262M (d_ff=25600) | ~0.031% |
| B | 4,096 | 512 | 4,194,304 | 8,388,608 | MoE active | ~0.5% (still negligible) |
| C | 8,192 | 8 | 131,072 | 262,144 | ~470M | <0.03% |

Router overhead is negligible relative to MLP FLOPs across all baselines and all variants.

### KV Cache Impact: Zero

KV cache depends only on L_full_attn, H_kv, head_dim, and sequence length. None of these change under any N5 variant. KV cache is completely unchanged.

### Memory Bandwidth Impact: Negligible

Router weights per layer: d × E_total × 2 bytes (bf16). For A1/A2 (E=8): 80 KB per layer << active expert weights. For B (E=512): 4 MB per layer, still <<1% of total model weight bandwidth.

### Critical Path Analysis

- V1 serial-before: adds O(d × E) FLOPs to the critical path before attention. At A1: 82K vs. ~178M MLP FLOPs — <0.05% critical path overhead.
- V1 parallel: router and attention run concurrently. Attention always dominates for s > ~8. Net critical path change: ~0%.
- V2–V4: router runs after Q/K projection but before attention softmax. Critical path: adds one router forward pass after Wq/Wk linear, which completes in parallel with the main attention computation. Negligible.

### Gradient Coupling (V2/V3/V4)

If `router_loss = f(softmax(Router(Q)))`, then `∂router_loss/∂Wq = (∂router_loss/∂Q) × x^T`. This creates a gradient path from router loss through Wq, which could bias attention weight training.

**Mitigation:** Apply stop-gradient before routing input: `g = Router(sg(Q))`. This prevents router loss from flowing through Wq/Wk while the routing still benefits from the richer Q/K representation.

| Variant | Stop-gradient needed | Risk |
|---------|---------------------|------|
| V1 (residual x) | No | Standard residual coupling |
| V2 (Q) | Yes — sg(Q) → Router | Prevents Wq bias |
| V3 (K) | Yes — sg(K) → Router | Prevents Wk bias (higher risk in GQA) |
| V4 (Q+K) | Yes — sg([Q;K]) → Router | Both Wq and Wk at risk |
| V5 (joint) | Architecture-dependent | Case-by-case |

---

## Implementation Considerations

### FlashAttention Compatibility

In standard implementations, `Q = F.linear(x, Wq)` is computed before `flash_attn_func(Q, K, V)`. The N5 router can be inserted between Q computation and the attention call for V2, or before the full QKV projection for V1. FlashAttention 2/3 is fully compatible — no kernel modification needed for V1–V4.

V5 (joint) requires custom kernel development; deferred.

### Recommended Experiment Order

1. **V1 parallel** (residual x, zero code changes beyond moving router call) — cheapest A/B test
2. **V2 serial-before with sg(Q)** — "query-intent routing" hypothesis test
3. If V1 or V2 shows benefit: V4 (Q+K concat) and dynamic/joint variants
4. Defer V5 until V1–V3 results are clear

### Minimum Viable Pilot

V1 parallel can be prototyped in 1–2 days on existing infrastructure:
- Move router call from post-attention to pre-attention in the forward function
- Run router and attention as parallel branches sharing input x
- All other training infrastructure unchanged

---

## Synergies

### High Synergy

- **3.3 (Dynamic Expert Router):** Pre-attention routing confidence from Q projections can determine both which experts to activate AND how many — a combined routing signal. Q projection uncertainty may be a better proxy for routing confidence than post-attention representation.

- **1.1 (Learnable Top-k):** Soft-MoE routing on Q projections. Q's potentially less-saturated gradient may make continuous routing more stable.

### Medium Synergy

- **3.1 (Layer-Level MoE):** Orthogonal dimensions — 3.1 determines whether a MoE layer activates; N5 determines what input the activated layer's router uses. Combined "Layer-Level Pre-Attention Router."

- **MoH head selection (arXiv:2410.11842):** A single Q projection can simultaneously select FFN experts (N5-V2) and attention heads (MoH). Dual-use Q projection.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|-----------|
| No quality improvement over standard routing | MEDIUM | MEDIUM | V1 A/B test costs <1 day; cheap to disprove |
| Training instability for V2/V3 (Wq/Wk gradient coupling) | HIGH | MEDIUM | Stop-gradient on routing input; monitor Wq/Wk gradient norms |
| Worse load balance (pre-attention routing is noisier) | MEDIUM | MEDIUM | Tune auxiliary load-balance loss weight; compare empirically |
| Expert specialisation shifts to undesirable patterns | MEDIUM | LOW | Inspect expert activation patterns post-training |
| SwitchHead negative Q/K result generalises to FFN experts | MEDIUM | LOW-MEDIUM | Different task (attention vs. FFN routing); test empirically |
| V5 complexity | HIGH | HIGH if pursuing V5 | Defer V5; pursue V1–V3 first |

**Overall Risk: LOW-MEDIUM.** Novel, theoretically motivated, zero-cost to prototype (V1). Primary research risks are empirical, not fundamental.

---

## Benefits vs Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | This Idea (6.5) | Change | Notes |
|--------|------------|-----------------|--------|-------|
| TTFT (8K prompt) | ref | ↑ negligible | ≈ ref | Router FLOPs per layer: 2 × d × E = 2 × 5120 × 8 = 81,920. Over L=64 layers and s=8192 prompt tokens: total router FLOPs = 64 × 81,920 × 8,192 = ~43B FLOPs. Total prefill FLOPs ≈ L × s × (2 × d × d_ff + 2 × s × d) ≈ 64 × 8192 × (2×5120×17408 + 2×8192×5120) ≈ ~469T FLOPs. Router fraction = 43B / 469T ≈ 0.009% → ≈ ref. |
| TPOT (batch=1) | ref | ≈ ref | ≈ ref | Idea repositions the router; does NOT reduce the number of active expert parameters or attention heads. Router FLOPs per decode step per layer = 81,920 vs. ~178–262M MLP FLOPs per layer — <0.05% overhead. V1-parallel: router runs concurrently with attention on same input; adds zero serial latency. V1-serial-before: O(d×E) = 82K FLOPs added to critical path before attention; attention dominates for any s ≥ 1. Net TPOT change: ≈ ref. |
| KV cache (32K ctx, BF16) | 2.15 GB | = | = | KV cache depends only on L_full_attn=16, H_kv=4, head_dim=256, seq_len. None of these change under any N5 variant. Derivation: 16×2×4×256×32768×2 = 2.15 GB — unchanged. |
| Weight memory | ~54 GB | = | = | V1–V3: zero new parameters (router weights already exist at d×E_total = 5120×8 = 40,960 params per layer; these are the same router weights moved earlier). V4 only: router weight doubles to 2d×E = 81,920 params per layer, adding 64 × 81,920 × 2 bytes ≈ 10 MB — negligible vs. 54 GB. |

## Benefits vs Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | This Idea (6.5) | Change | Notes |
|--------|------------|-----------------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Router FLOPs per layer: 2 × 5120 × 8 = 81,920. Over L=64 layers and s=8192 tokens: 64 × 81,920 × 8,192 ≈ 43B FLOPs. Total prefill FLOPs (dense, d=5120, d_ff=25600): ≈ 64 × 8192 × 2 × 5120 × 25600 ≈ ~1,376T FLOPs. Router fraction = 43B / 1376T ≈ 0.003% → ≈ ref. |
| TPOT (batch=1) | ref | ≈ ref | ≈ ref | A2 is a dense model — no MoE routing in standard form. If 6.5 is applied as a MoE conversion (upcycling), the routing change is architectural. Router FLOPs = 81,920 per layer vs. ~262M MLP FLOPs: fraction ≈ 0.031% (matches doc table). Net TPOT: ≈ ref. Pre-attention placement does not change active expert count or total parameter access BW. |
| KV cache (32K ctx, BF16) | 8.59 GB | = | = | KV depends on L=64, H_kv=8, head_dim=128, seq_len=32768. Unchanged by router repositioning. Derivation: 64×2×8×128×32768×2 = 8.59 GB — unchanged. |
| Weight memory | ~64 GB | = | = | V1–V3: no new parameters. Router weight per layer = d×E = 5120×8 = 40,960 params (same weights, different placement in forward pass). V4: +64 × 2 × 5120 × 8 × 2 bytes ≈ 10 MB — negligible vs. 64 GB. |

## Benefits vs Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | This Idea (6.5) | Change | Notes |
|--------|-----------|-----------------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Baseline B has E_total=512 experts (doc table: Router FLOPs = 4,194,304 per layer). Over L=60 layers and s=8192 tokens: 60 × 4,194,304 × 8,192 ≈ 2.06T FLOPs. Total prefill FLOPs (active params ~17B, d≈4096): ≈ large; router fraction ≈ 0.5% (doc table) — still negligible. ≈ ref. |
| TPOT (batch=1) | ref | ≈ ref | ≈ ref | Router FLOPs per decode step per layer = 4.2M vs. MoE active MLP FLOPs (doc: ~0.5% fraction). Pre-attention repositioning does not alter which or how many experts are activated per token — only the input representation driving the routing decision changes. Active weight BW: unchanged. TPOT: ≈ ref. |
| KV cache (32K ctx, BF16) | 1.0 GB | = | = | KV depends on L_global=15, H_kv=2, head_dim=256, seq_len=32768. Unchanged. Derivation: 15×2×2×256×32768×2 = 1.0 GB — unchanged. |
| Weight memory | ~34 GB | = | = | V1–V3: existing router weights (d×E_total = 4096×512 per layer) are repositioned, not duplicated. V4: +60 × 2 × 4096 × 512 × 2 bytes ≈ 503 MB — small relative to 34 GB (~1.5%), negligible. |

## Benefits vs Baseline C (K2 Family, LLM360)

| Metric | Baseline C | This Idea (6.5) | Change | Notes |
|--------|-----------|-----------------|--------|-------|
| TTFT (8K prompt) | ref | ≈ ref | ≈ ref | Router FLOPs per layer (C: d=8192, E=8): 2 × 8192 × 8 = 131,072. Over L=80 layers and s=8192 tokens: 80 × 131,072 × 8,192 ≈ 86B FLOPs. Total prefill FLOPs (d=8192, d_ff=28672): ≈ 80 × 8192 × 2 × 8192 × 28672 ≈ ~3,072T FLOPs. Router fraction = 86B / 3072T ≈ 0.003% → ≈ ref. |
| TPOT (batch=1) | ref | ≈ ref | ≈ ref | Router FLOPs per decode step = 131,072 per layer vs. ~470M MLP FLOPs (doc: <0.03%). Pre-attention repositioning changes routing input representation only; active expert count per token unchanged. Weight BW ≈ 145.1 GB unchanged. TPOT: ≈ ref. |
| KV cache (32K ctx, BF16) | 10.0 GiB | = | = | KV depends on L=80, H_kv=8, head_dim=128, seq_len=32768. Unchanged by router repositioning. Derivation: 80×2×8×128×32768×2 = 10.0 GiB — unchanged. |
| Weight memory | ~145.1 GB | = | = | V1–V3: no new parameters. Router weights repositioned in forward pass only. V4: +80 × 2 × 8192 × 8 × 2 bytes ≈ 21 MB — negligible vs. 145.1 GB. |

## Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **Switch Transformer**[1] (Fedus et al., arXiv:2101.03961, JMLR 2022): Standard post-attention routing (the baseline against which Idea 6.5 is positioned). Switch Transformer with top-1 routing achieves ~7× pre-training speedup vs. dense T5 at equal compute, but reports **quality sensitivity to routing collapse**: if load balance degrades (Gini coefficient of expert activations exceeds ~0.6), perplexity increases by 2–5 PPL. This is the primary quality risk that pre-attention routing could address — if pre-attention representations are less clustered, the load balance auxiliary loss coefficient can be reduced, potentially recovering 1–2 PPL.
  - **DeepSeekMoE**[4] (Dai et al., arXiv:2401.06066): Fine-grained expert segmentation (64 experts, top-6 routing) achieves **within 1.3% of dense model quality** on code, math, and knowledge benchmarks at 16B scale, while using ~40% of the FLOPs of an equivalent dense model. DeepSeekMoE's expert specialization analysis shows that post-attention routers learn content-based specialization. If pre-attention routing (Idea 6.5) shifts specialization toward query-type patterns, the quality impact depends on whether query-type specialization is as effective as content-based specialization — this is the central empirical unknown.
  - **SwitchHead**[6] (Csordás et al., arXiv:2312.07987, NeurIPS 2024): CRITICAL adjacent work. SwitchHead applies MoE routing to attention V/O projections using a pre-attention router on the residual x (V1-equivalent for attention heads). Per abstract: computes up to 8× fewer attention matrices than the standard Transformer; a 262M-parameter Transformer with SwitchHead matches baseline performance at 44% compute and 27% memory. Specific "2× attention reduction / 0.3–0.8 PPL degradation" figures at 125M scale are paper-body tables. Crucially, SwitchHead tested routing on Q/K projections (V2/V3 equivalent) for attention head selection and found it **unnecessary** (§3.1) — Q/K routing did not improve quality over residual-x routing. This negative result for attention head routing is the strongest prior evidence for Idea 6.5's V2/V3 variants, suggesting that Q/K-based routing may provide similar quality to residual-x routing (V1) with no gain — but also no additional loss.
  - **Expert Choice Routing**[5] (Zhou et al., arXiv:2202.09368): Inverted routing where experts select tokens (rather than tokens selecting experts). Per abstract: "more than 2×" training convergence speedup; outperforms T5 dense on 7 of 11 GLUE/SuperGLUE fine-tuning tasks; near-uniform load balance by construction. Specific per-task percentage gains are paper-body tables. This establishes that routing mechanism changes can be quality-positive when they improve load balance, motivating the pre-attention routing hypothesis.
  - **ReMoE**[20] (arXiv:2412.14711, ICLR 2025): Fully differentiable ReLU-based routing replaces TopK+Softmax. Per abstract: "consistently outperforms vanilla TopK-routed MoE across various model sizes, expert counts, and levels of granularity." Specific per-benchmark PPL improvements and dataset breakdowns are paper-body tables. Directly applicable as the router function in any Idea 6.5 variant; confirms that routing mechanism changes can improve quality.
  - **MoE Design Choices**[18] (arXiv:2402.13089): Systematic empirical ablations of MoE design. Key finding: router position (within the FFN, after attention, before attention) was NOT tested — this is the gap Idea 6.5 fills. Existing results show that **router capacity factor** is the dominant quality lever (capacity factor 1.0 → 2.0 improves quality by ~2–4 PPL); routing placement was assumed constant at post-attention.

- **Monotonicity**: Quality monotonicity with routing placement is not established in the literature — this is fundamentally an empirical question. The theoretical prediction is that V1 (pre-attention residual) routing produces similar or marginally worse quality vs. post-attention routing because pre-attention representations are less contextualised (less semantic information available for expert selection), potentially increasing assignment noise. However, V2 (Q-based routing) may provide a quality bonus for tasks where query intent better predicts the required expert specialization (e.g., math problems where the query formulation determines whether numerical or symbolic reasoning experts are needed). The quality delta is expected to be **small and bidirectional**: within ±1–2 PPL on standard benchmarks, with task-specific variation. The quality ordering V1 ≤ V2 ≤ V4 ≤ V3 (worst to best for query-type tasks) is hypothetical and requires empirical validation.

- **Recovery**: Since the routing mechanism is a zero-parameter change (V1–V3) or near-zero-parameter change (V4 adds ~10–21 MB of router weights), reverting to post-attention routing requires only a configuration change and brief fine-tuning (< 1B tokens) to re-stabilize expert load balance. There is no permanent quality damage from attempting pre-attention routing — the worst case is equivalent routing quality with equivalent load balance, not degradation below the post-attention baseline. The stop-gradient requirement on V2/V3 is essential for recovery: without stop-gradient, router loss flowing through Wq/Wk could degrade attention quality in a manner that requires longer fine-tuning to recover.

- **Conditions for acceptable degradation**: For Idea 6.5, "acceptable degradation" is almost unconditional, because the mechanism introduces no additional compute, no memory cost, and no throughput change. The only non-trivial risk is that pre-attention routing slightly increases expert assignment noise (leading to marginally higher PPL on generative tasks) or that V2/V3 without stop-gradient destabilizes Wq/Wk training. Both risks are mitigated by: (1) running V1 parallel as an A/B test before committing to V2/V3 (zero code risk); (2) enforcing stop-gradient for V2/V3/V4. The closest published analogues at smaller scale bound the expected quality envelope: SwitchHead[6] (pre-attention routing on V/O attention projections) reports 0.3–0.8 PPL degradation at 125M scale on Wikitext-103 with routing on residual x, and its negative result for Q/K-based attention-head routing suggests Q/K routing performs on par with (not worse than) residual-x routing; Switch Transformer[1] reports 2–5 PPL perplexity loss only when load balance degrades past Gini ≈ 0.6, bounding the downside risk under standard load-balance auxiliary loss; ReMoE[20] reports 0.1–0.3 PPL improvement at 125M–1.3B scale over standard TopK routing, indicating that routing mechanism changes can be quality-positive at scales below 2B. No experiments at 27B+ scale exist for pre-attention FFN expert routing specifically; all referenced numbers are for post-attention or full-layer MoE variants.

## Citations

<!-- CITATION MANIFEST -->
[1] Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity[1]: Fedus, Zoph, Shazeer. JMLR 2022. Standard top-1 MoE routing on post-attention residual. arXiv:2101.03961, §2 "Simplifying Mixture of Experts".
[2] GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding[2]: Lepikhin et al. 2021. Top-2 routing; load-balancing auxiliary loss. arXiv:2006.16668, §3 (Methods) [unavailable — inferred].
[3] Mixtral of Experts[3]: Jiang et al. (Mistral AI). 2024. 8×7B sparse MoE; top-2/8 routing; 32K context. arXiv:2401.04088, §2 "Sparse Mixture of Experts".
[4] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models[4]: Dai et al. 2024. Fine-grained segmentation + shared experts. arXiv:2401.06066, §3.1 "Fine-Grained Expert Segmentation".
[5] Mixture-of-Experts with Expert Choice Routing[5]: Zhou et al. (Google). 2022. Inverted routing; experts choose tokens. arXiv:2202.09368, §3 (Methods) [unavailable — inferred].
[6] SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention[6]: Csordás, Piękos, Irie, Schmidhuber. NeurIPS 2024. CRITICAL ADJACENT WORK — MoE on V/O attention projections using pre-attention routing; tested Q/K MoE and found unnecessary for attention head selection. arXiv:2312.07987, §3.1 "Which Projections Require an MoE?".
[7] Mixture of Attention Heads: Selecting Attention Heads Per Token (MoA)[7]: Zhang et al. EMNLP 2022. Routes attention heads using a per-token router; does NOT route FFN experts. arXiv:2210.05144, §3 (Methods) [unavailable — inferred].
[8] MoH: Multi-Head Attention as Mixture-of-Head Attention[8]: Jin et al. (SkyworkAI). 2024. Routes attention heads on pre-attention residual. arXiv:2410.11842, §3.2 "Mixture-of-Head Attention".
[9] Mixture of Sparse Attention (MoSA)[9]: 2025. Routes attention sparsity per head using per-head routers. arXiv:2505.00315, §2.2 "Mixture of Sparse Attention".
[10] Scaling Vision with Sparse Mixture of Experts (V-MoE)[10]: Riquelme et al. 2021. Sparse MoE for ViT; routing on patch features (less contextualised). arXiv:2106.05974, §3 (Methods) [unavailable — inferred].
[11] From Sparse to Soft Mixtures of Experts (Soft-MoE)[11]: Puigcerver, Riquelme et al. ICLR 2024. Differentiable routing; potential combination with Q-based routing. arXiv:2308.00951, §2.1 "Algorithm description".
[12] MegaBlocks: Efficient Sparse Training with Mixture-of-Experts[12]: Gale et al. 2022. Block-sparse GPU kernels; placement-agnostic dispatch. arXiv:2211.15841, §3 (Methods) [unavailable — inferred].
[13] Tutel: Adaptive Mixture-of-Experts at Scale[13]: Hwang et al. (Microsoft). 2023. Adaptive MoE pipelining; compatible with N5 parallel-branch. arXiv:2206.03382, §3 (Methods) [unavailable — inferred].
[14] MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts[14]: Pióro et al. 2024. MoE on Mamba representations; confirms routing placement flexibility. arXiv:2401.04081, §3 "MoE-Mamba".
[15] Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models[15]: Li et al. 2022. Domain-specialised expert LMs; specialisation hypothesis context. arXiv:2208.03306, §3 (Methods) [unavailable — inferred].
[16] Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints[16]: Komatsuzaki et al. 2022. Dense-to-MoE conversion; N5 applicable as upcycling target. arXiv:2212.05055, §3 (Methods) [unavailable — inferred].
[17] Hash Layers for Large Sparse Models[17]: Roller, Sukhbaatar et al. 2021. Content-free routing baseline for comparison. arXiv:2106.04426, §3 (Methods) [unavailable — inferred].
[18] Towards an Empirical Understanding of MoE Design Choices[18]: 2024. Systematic MoE design ablations; does not test pre-attention placement. arXiv:2402.13089, §3.1 "Performance Impact of Design Choices".
[19] UMoE: Unifying Attention and FFN with Shared Experts[19]: Yang, Wang, Li. 2025. Reformulates attention to expose FFN-like expert structure with per-expert Q projections before token mixing; architecturally adjacent to N5-V2/V4. arXiv:2505.07260, §3.2 "UMoE".
[20] ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing[20]: ICLR 2025. Replaces TopK+Softmax with ReLU gating for fully differentiable sparse routing; directly applicable as the router function in N5 variants V1–V4. arXiv:2412.14711, §3.2 "Differentiable ReLU Routing".
[21] DeepSeek-V4 Early Hash MoE[21]: DeepSeek-AI. 2026 technical report and Hugging Face release. All transformer blocks use MoE; first 3 MoE layers use Hash routing, later layers use learned routing with Sqrt(Softplus) affinity and sequence-wise balance loss. Modern baseline for early-layer routing stability.
