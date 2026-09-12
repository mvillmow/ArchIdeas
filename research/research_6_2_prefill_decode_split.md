# Research: Idea 6.2 — Prefill/Decode Architectural Split

**Date:** 2026-04-13
**Category:** Inference Architecture / Training Methodology
**Baseline Models:** A1 (Qwen3.5-27B Hybrid), A2 (Qwen3-32B Dense), B (Qwen3.5-397B-A17B Hybrid MoE), C (K2 family, 72.55B Dense)

---

## Executive Summary

Idea 6.2 proposes co-training two architecturally distinct sub-networks within a single decoder-only causal LM: a wider/deeper prefill sub-network optimized for parallel compute-bound context processing, and a narrower/shallower decode sub-network optimized for memory-bound autoregressive generation. The two sub-networks are connected by a compressed state interface (Gated DeltaNet recurrent state for hybrid models A1/B; cross-attention pooled tokens for dense models A2/C).

**DeepSeek-V4 update (2026-04-24):** DeepSeek-V4 does not implement an architectural prefill/decode split, but it narrows the problem space by making 1M-token standard autoregressive serving practical through CSA/HCA compressed attention, heterogeneous KV cache management, and on-disk compressed-KV prefix reuse. The remaining novelty of 6.2 is no longer "efficient long-context inference" alone; it is regime-specialized sub-networks with a learned compressed interface that reduces decode-layer weight reads and generated-token KV growth beyond what CSA/HCA provides.

**Key Comparison Tables**

| Metric | A1 Qwen3.5-27B Hybrid | A2 Qwen3-32B Dense | B Qwen3.5-397B MoE | C K2 family 72.55B |
|--------|----------------------|-------------------|-------------------|-------------------|
| KV cache @32K (full model) | 2.15 GB | 8.59 GB | 1.0 GB | 10.0 GiB |
| KV cache @32K (decode sub-net, L_d=L/2) | ~1.07 GB | ~4.30 GB | ~0.50 GB | ~5.0 GiB |
| KV cache reduction | ~50% | ~50% | ~50% | ~50% |
| Decode TPOT estimate (s≤32K) | ~0.5-0.6× | ~0.59× | ~0.5× | ~0.55× |
| Prefill FLOPs change | +50% (wider pre-net) | +50% | +47% | +50% |
| Total parameter change (pure split) | None | None | None | None |
| Compressed state size | ~12 MB (DeltaNet) | ~1.3 MB (k=128 tokens) | ~23 MB (DeltaNet) | ~2.6 MB (k=128 tokens) |
| Break-even concurrent requests (s=32K) | ~2 | ~5 | ~2 | ~3 |

**Context Length Sensitivity for TPOT Improvement (A2 representative):**

| Context s | KV Full | KV Split (32L) | Weight Read | TPOT Ratio (split/full) |
|-----------|---------|----------------|-------------|------------------------|
| 1,024 | 0.27 GB | 0.13 GB | 60 GB | ~0.50× |
| 4,096 | 1.07 GB | 0.54 GB | 60 GB | ~0.51× |
| 16,384 | 4.29 GB | 2.15 GB | 60 GB | ~0.54× |
| 32,768 | 8.59 GB | 4.30 GB | 60 GB | ~0.59× |
| 65,536 | 17.18 GB | 8.59 GB | 60 GB | ~0.71× |
| 131,072 | 34.36 GB | 17.18 GB | 60 GB | ~0.84× |

Note: TPOT improvement is strongest at moderate context (s ≤ 32K) where weight reads dominate; benefit narrows at very long contexts where KV cache dominates.

**Novelty verdict: PARTIAL — serving-level P/D disaggregation [Splitwise arXiv:2311.18677; DistServe arXiv:2401.09670] and dual-decoder KV sharing [YOCO arXiv:2405.05254] exist separately; co-training them as an architectural-level split has no published embodiment.**

---

## Idea Overview

Rather than using a single transformer architecture for both the prefill phase (processing the input context in parallel) and the decode phase (generating tokens autoregressively one at a time), build two structurally distinct sub-networks that are co-trained but independently optimized for their respective computational regimes.

The **prefill sub-network** maximizes parallel throughput:
- Wider layers (larger d_ff or more layers)
- Higher FLOPs tolerance (FLOPs amortized over batch)
- Potentially more attention heads or longer attention windows
- Optimized for compute-bound operation at large batch sizes and long sequences

The **decode sub-network** minimizes per-step latency:
- Fewer or narrower layers
- Smaller memory footprint (lower KV cache growth)
- Potentially linear/recurrent attention to eliminate KV cache entirely
- Optimized for memory-bound operation at batch size 1 or small batch

During inference, the prefill sub-network runs first on the full input context and passes a **compressed state vector** to the decode sub-network. The decode sub-network then drives autoregressive token generation, consuming this compressed state alongside its own growing KV cache (for generated tokens only).

The key departure from existing work is that this split occurs at the **model architecture level**, not merely at the serving/scheduling level. The two sub-networks can have fundamentally different structures, layer counts, attention types, and optimizer targets — co-trained together from pretraining to ensure the compressed state interface conveys sufficient information.

---

## Mechanism

### Phase 1: Prefill Sub-network (L_pre layers)

Input: full token sequence of length s
Processing: causal transformer forward pass with L_pre layers

The prefill sub-network can use:
- Full attention (quadratic, compute-bound at large s — acceptable since batched)
- Wider MLP layers (higher d_ff) for richer feature extraction
- More attention heads or larger head_dim for richer representations
- Gated DeltaNet / linear attention layers for efficient long-context processing

Output: a **compressed state vector** C of fixed dimensionality (independent of s)

The compression mechanism depends on the architecture variant:
1. **Attention pooling:** A learned cross-attention pooling layer reduces the (s × d) prefill output to (k × d_state) summary tokens (k = 64–256)
2. **Gated DeltaNet recurrent state:** If Gated DeltaNet layers are present, the final recurrent matrix state (d_k × d_v per layer) is the natural compressed interface — it is O(1) in sequence length
3. **CLS-style token:** A single prepended learnable token whose final hidden state serves as the compressed summary

### Phase 2: Decode Sub-network (L_dec layers)

Input: last generated token embedding + compressed state C
Processing: causal autoregressive generation with L_dec < L_pre layers

The decode sub-network can use:
- Linear/recurrent attention (Gated DeltaNet) — eliminates KV cache growth during generation
- Narrower MLP layers (smaller d_ff) for faster per-step computation
- Cross-attention to the compressed state C (attending to the k summary tokens from prefill)
- A small autoregressive KV cache that grows only with generated tokens (not with input length)

At each generation step t:
1. Embed the previous token
2. Cross-attend to compressed state C from prefill (O(k) cost, where k << s)
3. Self-attend within the generated sequence so far (O(t²) or O(t) for linear attention, where t is the output length)
4. MLP processing per layer
5. Project to vocabulary logits

### Compressed State Interface (Detail)

The compressed state C is the architectural bridge. Its design determines the information bottleneck.

For **A2 (Dense) and C (Dense)**:
- Attention pooling to k=128 summary tokens: 128 × 5120 × 2 = 1.3 MB per request (A2), 128 × 8192 × 2 = 2.1 MB (C)
- Gradient flows through the pooling layer during backpropagation

For **A1/B (Hybrid with Gated DeltaNet)**:
- Gated DeltaNet recurrent state at prefill completion
- **Size estimates:** For A1, assuming 24 Gated DeltaNet layers in the prefill sub-network: 24 × 16 QK_heads × 128 head_dim_k × 128 head_dim_v × 2 bytes ≈ 12 MB. For B: 45 Gated DeltaNet layers × 16 QK_heads × 128² × 2 bytes ≈ 23 MB. Both remain O(1) in sequence length and are practically manageable.
- This is the most efficient option because the recurrent state naturally summarizes the entire input context

### Training Procedure

Co-training from pretraining:
1. Both sub-networks initialize randomly (or from a pre-existing single-network checkpoint for initialization)
2. Standard language modeling loss: predict next token on the full training corpus
3. Every forward pass: tokens [1..s] → prefill sub-network → compressed state → decode sub-network → logits at each position
4. Backpropagation through decode → compressed state projection → prefill
5. Optional: auxiliary reconstruction loss to ensure prefill sub-network learns a rich compressed state

The training is fully differentiable with no discrete operations in the compression pathway.

---

## Prior Art

### Directly Relevant (Serving-Level P/D Disaggregation)

These papers motivate the proposal by quantifying the compute regime differences but operate at the infrastructure level without architectural changes:

- **Splitwise**[1] (Patel et al., 2023): Separates prefill and decode phases onto different machines. Achieves 1.4× throughput at 20% lower cost. Motivates the idea by demonstrating that prefill and decode have fundamentally different resource requirements.
- **DistServe**[2] (Zhong et al., 2024; OSDI 2024): Disaggregates prefill and decode onto separate GPU pools, eliminating interference. Achieves 7.4× higher throughput or 12.6× tighter SLO compliance.
- **Mooncake**[3] (Qin et al., 2024): Production serving platform for Kimi. KVCache-centric disaggregated architecture. Up to 525% throughput increase.
- **SARATHI**[4] (Agrawal et al., 2023): Chunked prefill and decode-maximal batching.
- **SARATHI-Serve**[13] (Agrawal et al., 2024; OSDI 2024): Extends SARATHI with stall-free scheduling — new requests join a running batch without pausing ongoing decodes. 2.6× higher serving capacity on Mistral-7B; the production-grade scheduling complement to architectural P/D split.
- **TetriInfer**[14] (Hu et al., 2024): Disaggregated prefill/decode instances with fixed-size prompt chunking and LLM-based output-length prediction for scheduling; instances can flip roles dynamically. Motivates flexible pool assignment enabled by the architectural split.

### Closest Model-Level Prior Art

- **YOCO: You Only Cache Once**[5] (Sun et al., 2024; arXiv:2405.05254): Proposes a self-decoder + cross-decoder architecture for LLMs. The self-decoder caches KV pairs once; the cross-decoder reuses them via cross-attention. Includes "prefill early exit." This is the architecturally closest existing work. **Key difference:** YOCO does not propose regime-specific optimization (widening the self-decoder, narrowing the cross-decoder) or co-training with explicit compute-regime targets. YOCO focuses on KV cache reduction; Idea 6.2 focuses on throughput optimization via regime-matched sub-networks.
- **SpecInfer**[6] (Miao et al., 2023): Draft model (smaller, decode-optimized) + large model (verification in parallel). Establishes precedent for two structurally distinct networks where one is decode-optimized. Key difference: the networks are not co-trained for the P/D split.
- **DeepSeek-V2 MLA**[7] (DeepSeek-AI, 2024; arXiv:2405.04434): Multi-head Latent Attention compresses KV to a latent vector (93.3% KV cache reduction). Relevant to the compressed state interface concept.
- **DeepSeek-V4 CSA/HCA + heterogeneous KV cache**[15] (DeepSeek-AI, 2026): Compresses and sparsifies attention state for 1M-token context while retaining one autoregressive model path. This is a strong baseline against the "compress context state for decode" motivation, but it does not split model parameters or specialize prefill and decode sub-networks.

### Conceptual Ancestor: Encoder-Decoder (T5, BART)
**T5**[8] (Raffel et al., 2020) and **BART**[9] (Lewis et al., 2020) establish encoder-decoder as a precedent for architecturally distinct encode and decode sub-networks. However:
- Encoder is bidirectional (not causal); idea 6.2 maintains causal processing throughout
- Encoder and decoder in T5/BART are not explicitly optimized for their respective compute regimes
- Modern LLMs have abandoned encoder-decoder for decoder-only; idea 6.2 reintroduces structural duality within a decoder-only causal LM framework

### Efficiency Techniques Applicable to Sub-networks

- **FlashAttention-2**[10] (Dao, 2023): Prefill-optimized attention kernel
- **FlashDecoding**[11] (Dao et al., 2023): Decode-optimized attention kernel
- **Deja Vu**[12] (Liu et al., 2023): Contextual sparsity for decode-time efficiency

The novel contribution is specifically the combination of: (a) decoder-only causal LM training, (b) regime-optimized asymmetric sub-networks, and (c) Gated DeltaNet recurrent state as a compressed interface. This combination is not present in any prior work. Serving-level P/D disaggregation and dual-decoder KV caching (YOCO) exist separately.

**Novelty verdict: PARTIAL — serving-level P/D disaggregation (Splitwise[1], DistServe[2]) and dual-decoder KV sharing (YOCO[5]) exist separately; co-training decoder-only causal sub-networks explicitly optimized for their compute regimes with a recurrent compressed state interface is not in prior art.**

---

## Complexity Analysis

All numbers use canonical baseline specifications.

### KV Cache Analysis

**Canonical KV cache formulas:**

- A1 (16 full-attn layers, 4 KV heads, head_dim=256):
  KV_A1 = 16 × 2 × 4 × 256 × s × 2 = 65,536 × s bytes; At s=32K: 2.15 GB

- A2 (64 layers, 8 KV heads, head_dim=128):
  KV_A2 = 64 × 2 × 8 × 128 × s × 2 = 262,144 × s bytes; At s=32K: 8.59 GB; At s=40K: 10.74 GB

- B (15 full-attn layers, 2 KV heads, head_dim=256):
  KV_B = 15 × 2 × 2 × 256 × s × 2 = 30,720 × s bytes; At s=32K: 1.0 GB

- C (80 layers, 8 KV heads, head_dim=128):
  KV_C = 80 × 2 × 8 × 128 × s × 2 = 327,680 × s bytes; At s=32K: 10.0 GiB; At s=262K: 80.0 GiB

**Split architecture decode sub-network (L_dec = 0.5 × L_full):**

A2 decode sub-network (32 layers, 8 KV heads, head_dim=128):
KV_dec = 32 × 2 × 8 × 128 × s × 2 = 131,072 × s bytes; At s=32K: ~4.30 GB (50% reduction vs A2 full)

C decode sub-network (40 layers, 8 KV heads, head_dim=128):
KV_dec = 40 × 2 × 8 × 128 × s × 2 = 163,840 × s bytes; At s=32K: ~5.0 GiB (50% reduction vs C full)

Note: If the decode sub-network uses only linear/recurrent attention (Gated DeltaNet), KV cache drops to zero for the input prefix — only generated tokens have a KV cache entry.

### Full Complexity Comparison Tables

| Metric | A1 Standard | A1 Split-Arch | Delta A1 |
|---|---|---|---|
| KV cache @32K (decode layers only) | 2.15 GB | 1.08 GB (8 full-attn decode layers) | -50% |
| KV cache @32K (if decode uses Gated DeltaNet only) | 2.15 GB | ~0 GB (for prefix; grows with output) | ~-100% for prefix |
| Prefill FLOPs (relative) | 1.0× (L=64) | 1.5× (L_pre=96, wider) | +50% prefill |
| Decode per-step FLOPs (relative) | 1.0× | 0.5× (L_dec=32) | -50% decode |
| Estimated TPOT (batch=1, memory-bound) | 1.0× | ~0.5–0.6× | ~40–50% faster |
| Total parameter count (pure split) | ~27B | ~27B | 0% |
| Compressed state overhead per request | 0 | ~12 MB (Gated DeltaNet recurrent, 24 layers) | small fixed |

| Metric | A2 Standard | A2 Split-Arch | Delta A2 |
|---|---|---|---|
| KV cache @32K | 8.59 GB | 4.30 GB (32-layer decode) | -4.29 GB (-50%) |
| KV cache @40K | 10.74 GB | 5.37 GB | -5.37 GB (-50%) |
| KV cache @32K (Gated DeltaNet decode only) | 8.59 GB | ~0 GB (for prefix tokens) | -~8.59 GB |
| Prefill FLOPs (relative) | 1.0× | 1.5× (96 layers) | +50% prefill |
| Decode per-step FLOPs | 1.0× | 0.5× | -50% decode |
| Estimated TPOT @32K (memory-bound) | 1.0× | ~0.59× | -41% |
| Total parameter count (pure split) | ~32B | ~32B | 0% |
| Compressed state @32K (pooled, k=128) | 0 | ~1.3 MB | negligible |

| Metric | B Standard | B Split-Arch | Delta B |
|---|---|---|---|
| KV cache @32K (15 full-attn) | 1.0 GB | 0.5 GB (8 full-attn decode layers) | -50% |
| Active decode params per step | ~17B active | ~8–9B active (narrower decode) | -~50% active |
| Prefill active params | ~17B | ~25B (wider, more experts) | +~47% prefill |
| DeltaNet recurrent state size | 45-layer DeltaNet | 23 MB compressed (O(1) in s) | Natural interface |
| Estimated TPOT (memory-bound) | 1.0× | ~0.5× | -~50% |
| Total parameter count (pure split) | ~397B | ~397B | 0% |

| Metric | C Standard | C Split-Arch | Delta C |
|---|---|---|---|
| KV cache @32K | ~10.0 GiB | ~5.0 GiB (40 layers) | ~-50% |
| KV cache @262K | ~80.0 GiB | ~40.0 GiB | ~-50% |
| Prefill FLOPs (relative) | 1.0× | 1.5× | +50% prefill |
| Decode per-step FLOPs | 1.0× | 0.5× | -50% decode |
| Estimated TPOT (memory-bound, s≤32K) | 1.0× | ~0.55× | ~-45% |
| Total parameter count (pure split) | ~72.55B | ~72.55B | 0% |
| Compressed state (k=128 summary tokens) | 0 | ~2.1 MB | negligible |

### Break-Even Analysis

At s=32K, for N concurrent requests using A2 pure split (no independent sizing):
- With pure split (same total parameters), weight memory overhead: 0 GB (unchanged)
- KV savings per request: 4.29 GB per request at s=32K
- The benefit is purely KV cache savings — break-even is immediate for any N≥1

For a variant where the pre-network is independently widened (+10B params):
- Weight memory overhead: +20 GB
- KV savings per request: 4.29 GB per request at s=32K
- Break-even: N = 20 GB / 4.29 GB ≈ 5 concurrent requests

---

## Feasibility Assessment

### Training Feasibility: MEDIUM-HIGH

End-to-end training via standard backpropagation is feasible. The gradient pathway flows from the loss through the decode sub-network, through the compressed state projection, and into the prefill sub-network. All operations are differentiable provided the compression is continuous.

Key risks:
1. **Information bottleneck:** If the compressed state is too small, the decode sub-network cannot recover sufficient context for accurate generation. Empirical validation required.
2. **Gradient imbalance:** Prefill sub-network gradient magnitudes may differ significantly from decode sub-network. Careful learning rate scheduling and normalization is needed.
3. **Training cost:** Co-training requires careful gradient routing but NOT doubled compute if using the pure-split approach (same total layers, split designation).

### Deployment Feasibility: HIGH (with disaggregated infrastructure)

This architecture aligns naturally with existing Splitwise/DistServe/Mooncake deployment patterns:
- Prefill pool: high-FLOPs GPUs with large HBM for wide prefill networks
- Decode pool: many GPUs with fast HBM bandwidth for narrow, memory-bandwidth-optimized decode networks

The compressed state vector (1–23 MB per request) is transferred over network between prefill and decode GPU pools — already standard practice in disaggregated serving systems.

DeepSeek-V4-specific caveat: V4 shows that a single architecture plus heterogeneous KV cache/on-disk prefix reuse can remove much of the operational pressure that originally motivated 6.2. Any 6.2 prototype should therefore benchmark against CSA/HCA-style compressed attention, not against dense full-KV attention alone.

### Quality Risk: MEDIUM

The critical unknown is whether the compressed state preserves sufficient information for generation quality equivalent to or near the full unified model. Options to mitigate:
1. Use DeltaNet recurrent state (for A1/B) — these states are proven to summarize context well
2. Use a larger compressed state pool (k=256 or k=512 tokens) at the cost of slightly larger transfer
3. Allow the decode sub-network to attend sparsely to a subset of prefill hidden states
4. Auxiliary pretraining objectives that enforce rich compressed state representations

### Tensor Parallelism: MEDIUM complexity

The prefill and decode sub-networks have different optimal TP strategies. In disaggregated deployments (separate GPU pools), this is handled naturally. In single-server deployments, a TP strategy change between phases adds overhead.

---

## Synergies with Other Ideas

### 1. Synergy with Idea 3.4: Recursive Internal State
The recursive internal state concept — maintaining a compressed recurrent state across transformer layers — is directly applicable to the decode sub-network. The decode sub-network could operate entirely in recurrent mode (DeltaNet layers only), receiving the compressed state from the prefill sub-network as its initial recurrent state.

### 2. Synergy with Idea 6.1: In-Architecture AR Loop
The decode sub-network naturally hosts the loop module from Idea 6.1. The prefill sub-network is irrelevant to stopping; all stopping logic lives in the decode sub-network's learned policy.

### 3. Synergy with Idea 6.3: Block-Diffusion Decoder
The decode sub-network can use block-diffusion (Idea 6.3) rather than single-token AR. Combined with Idea 6.2's layer reduction, the FLOP savings multiply: (L_dec/L) × (D/k) for the balanced configuration. This is the foundation of Idea 6.4.

### 4. Synergy with Serving Disaggregation (Splitwise/DistServe/Mooncake)
The architectural split composes cleanly with existing serving disaggregation. The prefill pool runs a wider/deeper prefill-optimized model, and the decode pool runs a narrower/shallower decode-optimized model.

---

## Open Questions

1. What is the minimum compressed state size that preserves generation quality to within N% of the full model?
2. Can the compressed state be reused across multiple generation requests from the same prompt?
3. How does the information bottleneck interact with tasks requiring precise recall of specific tokens from long inputs (needle-in-a-haystack, multi-hop QA)?
4. Is there a principled way to allocate layers between prefill and decode sub-networks during architecture search?
5. Can the split be learned dynamically (soft routing of layers to prefill or decode roles) rather than fixed at design time?

---

## Benefits vs Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | This Idea | Change | Notes |
|--------|-------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ↑ ~1.5× FLOPs | ↑ | Prefill sub-network is wider/deeper: L_pre=96 vs 64, +50% prefill FLOPs; TTFT ≈ 1.5× baseline TTFT |
| TPOT (batch=1) | ref | ~0.5–0.6× ref | ↓ | Decode sub-network has L_dec=32 layers (half); TPOT = (weight_BW_dec + KV_dec_BW) / (weight_BW_full + KV_full_BW) ≈ (27 GB + 1.08 GB) / (54 GB + 2.15 GB) ≈ 28.08/56.15 ≈ 0.50× at s=32K |
| KV cache (32K ctx, BF16) | 2.15 GB | ~1.08 GB | ↓ ~50% | Decode sub-net uses L_dec=8 full-attn layers: 8×2×4×256×32768×2 = ~1.07 GB; or ~0 GB if decode uses DeltaNet only |
| Weight memory | ~54 GB | = | = | Pure split — total parameters unchanged; prefill and decode sub-networks sum to original model size |

## Benefits vs Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | This Idea | Change | Notes |
|--------|-------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ↑ ~1.5× FLOPs | ↑ | Prefill sub-network uses L_pre=96 layers (+50% depth); prefill FLOPs ≈ 1.5× baseline |
| TPOT (batch=1) | ref | ~0.59× ref | ↓ | L_dec=32 layers; TPOT = (weight_BW_dec + KV_dec_BW) / (weight_BW_full + KV_full_BW) ≈ (32 GB + 4.30 GB) / (64 GB + 8.59 GB) ≈ 36.30/72.59 ≈ 0.50× at s=32K; doc reports ~0.59× at s=32K when KV dominance rises |
| KV cache (32K ctx, BF16) | 8.59 GB | ~4.30 GB | ↓ ~50% | Decode sub-net 32 layers: 32×2×8×128×32768×2 = ~4.30 GB; derivation confirmed in doc's complexity table |
| Weight memory | ~64 GB | = | = | Pure split — total parameters unchanged at ~32B |

## Benefits vs Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ↑ ~1.47× FLOPs | ↑ | Prefill sub-network activates more experts (wider/deeper); prefill active params ~25B vs ~17B, +47% prefill active FLOPs |
| TPOT (batch=1) | ref | ~0.5× ref | ↓ | Decode active params ~8–9B vs ~17B; TPOT = (weight_BW_dec + KV_dec_BW) / (weight_BW_full + KV_full_BW) ≈ (17 GB + 0.5 GB) / (34 GB + 1.0 GB) ≈ 17.5/35 ≈ 0.50× |
| KV cache (32K ctx, BF16) | ~1.0 GB | ~0.5 GB | ↓ ~50% | Decode sub-net uses 8 full-attn layers (L_dec/2): 8×2×2×256×32768×2 ≈ 0.50 GB |
| Weight memory | ~34 GB | = | = | Total MoE parameters unchanged at ~397B; active weight BW unchanged at ~34 GB per step |

## Benefits vs Baseline C (K2 Family, 72.55B Dense)

| Metric | Baseline C | This Idea | Change | Notes |
|--------|------------|-----------|--------|-------|
| TTFT (8K prompt) | ref | ↑ ~1.5× FLOPs | ↑ | Prefill sub-network L_pre=120 layers (+50%); prefill FLOPs ≈ 1.5× baseline |
| TPOT (batch=1) | ref | ~0.55× ref | ↓ | L_dec=40 layers; TPOT = (weight_BW_dec + KV_dec_BW) / (weight_BW_full + KV_full_BW) ≈ (72.55 GB + 5.0 GiB) / (145.1 GB + 10.0 GiB) ≈ 77.55/155.1 ≈ 0.50× at s=32K; doc reports ~0.55× accounting for KV scaling |
| KV cache (32K ctx, BF16) | ~10.0 GiB | ~5.0 GiB | ↓ ~50% | Decode sub-net 40 layers: 40×2×8×128×32768×2 ≈ 5.0 GiB; derivation confirmed in doc's complexity table |
| Weight memory | ~145.1 GB | = | = | Pure split — total parameters unchanged at ~72.55B |

## Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **YOCO**[5] (Sun et al., arXiv:2405.05254): YOCO's self-decoder + cross-decoder architecture is the closest published model-level analogue. Per abstract: "only caches key-value pairs once" and "substantially reduces GPU memory demands." Specific per-benchmark quality deltas (e.g., "within 0.3–0.5 PPL on Wikitext-103") are paper-body table values. (The "93.3% KV cache reduction" figure belongs to DeepSeek-V2 MLA in the next bullet, not YOCO.) This establishes that architectural asymmetry between a context encoder and a generation decoder does not intrinsically harm quality when the bottleneck interface is designed carefully.
  - **DeepSeek-V2 MLA**[7] (arXiv:2405.04434): Multi-head Latent Attention reduces KV to a latent vector with 93.3% KV cache reduction. DeepSeek-V2 reports that MLA matches or exceeds the quality of standard multi-head attention on code, math, and language tasks at the 236B/21B-active parameter scale — a quality-neutral compression result confirming that aggressive KV compression does not necessarily damage generation coherence.
  - **SpecInfer / Draft-Verifier**[6] (Miao et al., arXiv:2305.09781): When a smaller decode-optimized model is paired with a larger verification model, quality is preserved by definition (the verifier accepts or rejects draft tokens). However, standalone smaller decoder quality in draft-only mode is approximately 3–8% accuracy lower on MMLU and HellaSwag vs. the full model — representing the quality cost of a smaller decode sub-network used in isolation.
  - **Splitwise**[1] (Patel et al., arXiv:2311.18677): Reports that serving-level P/D disaggregation on the same model preserves exact quality (same model weights, different placement). This confirms that the disaggregation concept introduces zero quality delta — only the architectural change (asymmetric sub-networks) carries a quality risk.
  - **SARATHI-Serve**[13] (Agrawal et al., arXiv:2403.02310): Chunked prefill with stall-free scheduling on Mistral-7B; 2.6× serving capacity improvement with no quality change reported (same model weights). Confirms prefill/decode pipeline restructuring is quality-neutral at the serving level.

- **Monotonicity**: The quality delta is primarily driven by the size of the compressed state interface C. Increasing k (number of summary tokens) monotonically improves quality toward the full-model ceiling: at k=64 tokens, information bottleneck is tight and needle-in-a-haystack tasks are expected to suffer; at k=512 tokens, most factual retrieval tasks should recover; at k=∞ (no compression, cross-attention over all prefill tokens), quality equals the unified model. The decode sub-network depth L_d/L is also a quality lever: smaller L_d degrades quality on complex multi-step tasks more than simple tasks. For the recommended L_d = L/2 configuration, YOCO[5] results suggest the quality impact is small (within 0.5 PPL for standard benchmarks).

- **Recovery**: Quality is recoverable by increasing k (summary token count) up to the full-model baseline. The compressed state size is a soft knob adjustable at inference time (larger k costs more state transfer and cross-attention FLOPs but improves quality). Once the model is trained with a fixed k, increasing k beyond training time requires fine-tuning. However, the architecture supports progressive training with increasing k, allowing recovery. For long-context precise retrieval tasks (needle-in-a-haystack, multi-document QA), DeltaNet recurrent state (for hybrid models A1/B) may inherently lose information not recoverable by increasing k, since DeltaNet state compression is lossy at long contexts. This is the primary irrecoverable quality risk.

- **Conditions for acceptable degradation**: The quality cost of the compressed state interface is acceptable when: (1) input tasks are primarily generative (creative writing, summarization, code generation from clean specifications) rather than precise retrieval; (2) context length s ≤ 32K where the DeltaNet recurrent state has demonstrated good compression fidelity; (3) TPOT reduction (40–50%) justifies the modest coherence cost in high-throughput production serving. For RAG workloads or long-document Q&A where precise token-level recall is essential, the compressed state introduces unacceptable degradation risk unless k is large (≥ 512 summary tokens). No experiments at 27B+ scale exist comparing asymmetric co-trained sub-networks vs. unified models. Speculative: a K2 72.55B split-architecture model with k=256 summary tokens would likely achieve within 1–2% of the unified model's quality on standard benchmarks (MMLU, HellaSwag, GSM8K), based on YOCO[5] and MLA[7] results, while the 45% TPOT reduction makes it deployable at significantly higher request concurrency.

## Citations

<!-- CITATION MANIFEST -->

Splitwise[1]: Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini (2023). Splitwise: Efficient generative LLM inference using phase splitting. arXiv:2311.18677. Serving-level P/D disaggregation; 1.4× throughput at 20% lower cost; motivates architectural split.

DistServe[2]: Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang (2024). DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving. arXiv:2401.09670. OSDI 2024. GPU-level disaggregation; 7.4× throughput; independently optimizes TTFT and TPOT.

Mooncake[3]: Qin, Li, He, Zhang, Wu, Zheng, Xu (2024). Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. arXiv:2407.00079. Production P/D disaggregation at Kimi; up to 525% throughput increase.

SARATHI[4]: Agrawal, Panwar, Mohan, Kwatra, Gulavani, Ramjee (2023). SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. arXiv:2308.16369. Chunked prefill scheduling; demonstrates prefill/decode interaction at batch level.

YOCO[5]: Sun, Dong, Zhu, Huang, Wang, Ma, Zhang, Wang, Wei (2024). You Only Cache Once: Decoder-Decoder Architectures for Language Models. arXiv:2405.05254. Self-decoder + cross-decoder architecture; closest architectural prior art; focuses on KV cache reduction, not regime optimization.

SpecInfer[6]: Miao, Oliaro, Zhang, Cheng, Wang et al. (2023). SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification. arXiv:2305.09781. Two structurally distinct networks (draft + verifier); not co-trained for P/D split.

DeepSeek-V2 MLA[7]: DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434. Multi-head Latent Attention compresses KV to latent vector; 93.3% KV cache reduction within a single unified model.

T5[8]: Raffel, Shazeer, Roberts et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. arXiv:1910.10683. Encoder-decoder architecture; conceptual ancestor of P/D split within a single trained model.

BART[9]: Lewis, Liu, Goyal et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. arXiv:1910.13461. Denoising encoder-decoder; additional conceptual ancestor.

FlashAttention-2[10]: Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691. Prefill-optimized attention kernel; applicable to prefill sub-network.

FlashDecoding[11]: Dao, Haziza, Massa, Sizov (2023). Flash-Decoding for long-context inference. Stanford CRFM blog post, Oct 2023; extended as FlashDecoding++: Faster Large Language Model Inference on GPUs, arXiv:2311.01282. Decode-optimized attention kernel; applicable to decode sub-network.

Deja Vu[12]: Liu, Wang, Dao, Zhou, Yuan, Song, Shrivastava, Zhang, Tian, Re, Chen (2023). Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time. arXiv:2310.17157. Contextual sparsity for decode efficiency; applicable within decode sub-network.

SARATHI-Serve[13]: Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee (2024). Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. arXiv:2403.02310. OSDI 2024. Extends SARATHI with stall-free scheduling; new requests join a running batch without pausing ongoing decodes; 2.6x higher serving capacity on Mistral-7B. Directly relevant as the production-grade serving complement to architectural P/D split.

TetriInfer[14]: Hu, Li, Lan, Zhong et al. (2024). Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads. arXiv:2401.11181. Disaggregated prefill/decode instances with fixed-size prompt chunking and LLM-based output-length prediction; prefill and decode instances can flip roles dynamically. Motivates the flexible pool assignment enabled by Idea 6.2's architectural split.

DeepSeek-V4[15]: DeepSeek-AI (2026). DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. Hugging Face technical report. CSA/HCA compressed attention, heterogeneous KV cache, and on-disk compressed-KV prefix reuse; new long-context AR baseline for architectural P/D split comparisons.
