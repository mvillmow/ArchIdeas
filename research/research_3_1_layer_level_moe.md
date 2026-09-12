# Research: Layer-Level MoE (Full Block Routing)
## ID: 3.1

## 1. Idea Description

**From arch_research_ideas.md (Section 3, idea 3.1):**

> The MoE router wraps entire transformer blocks (attention + FFN + norm) rather than just the feed-forward layer. The router selects which complete architectural block to execute per token.

**Key distinction:** Standard MoE routes between expert FFNs within a single layer while sharing one attention mechanism. Idea 3.1 routes between complete transformer blocks — each expert is a full block (attention + FFN + norms). This fundamentally changes the KV cache implications: if each block has its own attention, routing selects which attention pattern is applied to a token, and k expert blocks × their own KV entries can multiply KV cache size by up to k×.

**Inferred inference-speedup intent:** By routing each token through only k of E complete blocks (rather than through every layer), the compute per forward pass becomes approximately k/E × total, while the total parameter count grows by E×. This mirrors MoE FFN scaling at the coarser block granularity — enabling different tokens to apply different attention + FFN configurations, potentially allowing greater specialization than FFN-only MoE at the cost of more complex KV cache management.

**Key parameter space for analysis:** L=64 layers, E=8 experts, k=2 active, stride P (1 of P layers is a full-block MoE position), d=5120, d_ff=25600. Note: d_ff=25600 is the geometry of **Baseline A2 (Qwen3-32B Dense)**; Baseline A1 (Qwen3.5-27B Hybrid) has a different MLP geometry (d_ff=17408, derived from its SwiGLU intermediate size). The parameter-space analysis below uses A2 geometry as the primary proxy; A1-specific figures are called out explicitly where they differ.

---

## 2. Executive Summary

**Novelty verdict:** PARTIAL — component elements (FFN MoE, head-level MoE, binary full-block routing, block-level KV sparsity) are individually published, but no paper combines E independently parameterized full transformer blocks (own W_Q/W_K/W_V/W_O/W_FFN) with per-token top-k routing in AR generation while analyzing the k-fold KV-cache multiplication and sparse-history problems ([Switch Transformer, 2022], [Mixtral, 2024], [MoA, 2024], [SwitchHead, 2024], [MoD, 2024]).

Layer-level full-block routing (idea 3.1) is a novel architectural concept with no direct prior instantiation in the literature. The core novelty claim — that routing entire transformer blocks (not just FFN experts) enables per-token selection of distinct attention patterns — is confirmed as unaddressed by any published work. No paper combines (1) E independently parameterized full transformer blocks each with own W_Q, W_K, W_V, W_O, W_FFN, (2) a per-token router selecting k of E at each layer position in autoregressive generation, (3) analysis of the resulting k-fold KV cache multiplication problem, and (4) characterization of sparse expert-specific attention histories as a quality degradation mechanism.

The idea's primary blocker is not merely KV cache *size* inflation (k× at routed layers, mitigable via MLA-style compression) but a more fundamental issue: **KV cache semantics**. When expert e processes token t, it can only attend to prior positions where expert e was also active — approximately k/E ≈ 25% of history under balanced routing. This structurally alters attention semantics in an uncharacterized way.

Three KV strategies exist, ranked by novelty/feasibility trade-off: (A) dense all-expert KV (correct semantics, E× overhead — impractical), (B) shared KV + per-expert FFN/value (preserves semantics, eliminates novelty — reduces to SwitchHead + FFN MoE), and (C) per-expert MLA latent KV (preserves novelty, reduces storage, sparse-history problem remains open). Strategy C is the recommended investigation path. The TPOT crossover point — where weight-load savings from routing exceed KV read overhead — is approximately 334K tokens [derived: weight_saving = (1−k/E) × block_BW = (1−2/8) × 912 MB = 684 MB; KV_overhead = (k−1) × d_kv_bytes × s = 2048 × s bytes; break-even: s = 684×10⁶ / 2048 ≈ 334,000 tokens; see §5.1 for full derivation] for Qwen3-32B geometry at k=2, E=8, meaning TPOT *improves* at typical deployment sequence lengths.

---

## 3. Literature Review

### Mixture of Depths: Dynamically Allocating Compute in Transformer-Based Language Models
Raposo et al., 2024 — arXiv:2404.02258, §3 "Mixture of Depths"[1]

Proposes Mixture-of-Depths (MoD), where a top-k router per layer dynamically selects which tokens execute a full transformer block (self-attention + MLP) and which bypass via residual skip. Expert-choice routing assigns a fixed fraction of tokens to the full computation. Models match baseline performance for equivalent FLOPs, with up to 50% wall-clock speedup during post-training sampling. This is the closest prior published work to idea 3.1: a binary (execute vs. skip) form of full-block routing. The key gap is that MoD has only two "experts" and no analysis of E distinct parameterized blocks with independent attention.

### SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention
Csordás et al., 2024 — NeurIPS, arXiv:2312.07987, §4 "SwitchHead"[2]

Applies MoE to the value and output projections of attention. A 262M-parameter SwitchHead model matches baseline perplexity with 44% of compute and 27% of memory. The "SwitchAll" variant combines MoE attention with MoE feedforward layers. Demonstrates attention-layer MoE is feasible and quality-preserving. KV implications: SwitchHead's sparse attention projections reduce the number of attended heads; notably, it shares K/V across experts, which is precisely the Strategy B mitigation. Idea 3.1 goes further by routing the entire block as a unit with independent KV projections.

### Mixture of Attention Heads: Selecting Attention Heads Per Token (MoA)
Zhang et al., 2022 — EMNLP, arXiv:2210.05144, §3 "Mixture-of-Attention-Heads"[3]

Combines multi-head attention with MoE: each head has its own parameters, and a router selects k of H heads per token. Demonstrates per-token attention-head routing is learnable and practical on MT and MLM tasks. Idea 3.1 generalizes MoA by routing entire blocks (attention + FFN together), not just attention heads. No KV cache analysis for per-token head routing.

### MoH: Multi-Head Attention as Mixture-of-Head Attention
Jin et al., 2025 — ICML 2025, arXiv:2410.11842, §3 "MoH Formulation"[4]

Reformulates multi-head attention so head outputs are weighted sums (MoE-style) instead of equal sums. MoH-LLaMA3-8B achieves 64.0% average accuracy on 14 benchmarks vs 61.6% for LLaMA3-8B, using only 75% of attention heads. Directly demonstrates improved quality-per-FLOPs over dense multi-head attention via routing.

### UMoE: Unifying Attention and FFN with Shared Experts
Yang et al., 2025 — NeurIPS 2025 Spotlight, arXiv:2505.07260[5]

Reformulates attention to expose an FFN-like structure, enabling a unified MoE parameterization across both attention and FFN layers. The most direct existing approach to unifying attention + FFN under a single MoE routing framework. UMoE shares parameters between attention and FFN components rather than instantiating E independent full blocks. Relevance: architecturally close to idea 3.1 but does not demonstrate the full novelty of E independently parameterized complete blocks with independent attention weights.

### Multi-Head Mixture-of-Experts (MH-MoE)
Wu et al., 2024 — NeurIPS 2024, arXiv:2404.15045[6]

Splits each input token into multiple sub-tokens assigned to different experts simultaneously, processed in parallel, then reintegrated. Addresses the low expert-activation problem in standard sparse MoE. Demonstrates that routing multiple sub-token representations to different experts provides better expert utilization than standard top-k MoE. Applied to FFN experts; attention is shared.

### Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer (UoE)
Yang et al., 2025 — arXiv:2503.02495[7]

Decomposes both MLP and attention blocks into groups using tensor parallelism, with hierarchical routing making correlated joint selections for both components. Applies MoE routing to both attention and MLP simultaneously — hierarchically close to idea 3.1 — but expert decomposition is based on partitioning existing weight matrices (tensor parallelism grouping), not on instantiating E independent full-block copies.

### Mixture-of-Modules: Reinventing Transformers as Dynamic Assemblies of Modules (MoM)
Gong et al., 2024 — EMNLP 2024, ACL Anthology:2024.emnlp-main.1164[22]

Constructs transformer forward passes by iteratively routing each token through independently parameterized attention and FFN modules drawn from shared pools, where two separate routers (one for attention, one for FFN) dynamically select which modules to apply at each step. Demonstrates that removing 50% of attention modules and 25% of FFN modules post-training preserves performance, and achieves GPT-2 (774M) parity with 16% fewer TFLOPs and 42% less memory. The most architecturally proximal published work to idea 3.1: independently parameterized attention and FFN modules routed per token, constructing virtual computation paths of variable depth and composition. The key gap vs. idea 3.1 is that MoM routes attention and FFN independently with separate routers (not as coupled full-block units), and operates on top of a base transformer architecture rather than replacing all layers with MoE blocks.

### Layerwise Recurrent Router for Mixture-of-Experts (RMoE)
Qiu et al., 2025 — ICLR 2025, arXiv:2408.06793, §3 "Layerwise Recurrent Router"[8]

Introduces a GRU-based router that propagates routing state across consecutive MoE layers, enabling cross-layer routing correlation. Consistently outperforms independent per-layer routers on language modeling. Directly relevant to idea 3.1: full-block selection at layer L should coordinate with selection at L+1, and the residual stream dependency makes this even more critical than in FFN-only MoE.

### Mixture of Nested Experts: Adaptive Processing of Visual Tokens (MoNE)
Jain et al., 2024 — NeurIPS, arXiv:2407.19985, §4 "MoNE Architecture"[9]

Routes visual tokens to experts of varying computational size (nested matryoshka-style subsets). Achieves equivalent performance to baseline ViT while reducing inference compute by more than 2× on ImageNet-21K and video benchmarks. Full-block routing (at ViT layer granularity) achieves 2× inference speedup. Demonstrates block-level per-token routing at scale, though in a vision (non-autoregressive) setting with nested subsets rather than E fully independent blocks.

### Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing (MoSA)
Piękos et al., 2025 — arXiv:2505.00315 (arXiv preprint)[10]

Replaces dense attention heads with multiple sparse-attention heads via expert-choice routing over tokens. Drastically reduces KV cache size (specific reduction rate not quantified in paper), reduces training memory, and achieves O(k² + T) per-head complexity. Directly addresses per-head sparse attention routing with KV cache size implications. Routing is to token positions within a single head rather than to which expert block to execute.

### MoBA: Mixture of Block Attention for Long-Context LLMs
Lu et al., 2025 — arXiv:2502.13189, §3 "Mixture of Block Attention"[11]

Applies MoE principles to the attention mechanism at block/chunk level. Tokens route attention queries to context blocks (contiguous KV chunks) rather than attending globally. Deployed in production supporting Kimi's long-context requests. The closest prior to block-level routing in attention, though it routes queries to KV context blocks (sparsity over token positions) rather than to expert attention parameter sets.

### PiKV: KV Cache Management System for Mixture of Experts
Liu et al., 2025 — arXiv:2508.06526 (arXiv preprint)[12]

Addresses KV cache management specific to MoE architectures with expert-sharded distributed KV layout, sparse expert routing, and adaptive stream scheduling with activity-based eviction. Specific speedup and memory-reduction figures (2.2× / 65%) are paper-body values, not stated in the abstract. Directly applicable framework for idea 3.1's per-expert KV problem, though designed for FFN-MoE with shared attention.

### SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained Transformer Scaling
Vejendla, 2025 — EMNLP 2025, arXiv:2510.04286[13]

Routes contiguous slices of a token's hidden vector to different FFN experts. Achieves 1.7× inference speedup with improved load balancing. Demonstrates sub-token routing granularity, complementary to idea 3.1's full-block routing; applies only to FFN experts with shared attention.

### DeepSeek-V3 Technical Report
DeepSeek-AI, 2024 — arXiv:2412.19437, §3 "Architecture"[14]

State-of-the-art MoE LLM with 671B total parameters (37B active per token), using DeepSeekMoE (fine-grained FFN experts) combined with Multi-Head Latent Attention (MLA) for KV cache compression. MLA compresses KV to a low-dimensional latent vector, reducing KV cache size dramatically while maintaining attention quality. DeepSeek-V3's design choice to keep attention dense and shared (via MLA) while routing only FFN experts is precisely the gap that idea 3.1 addresses. MLA is directly applicable to per-expert KV compression (Mitigation A).

### Mixture-of-Experts with Expert Choice Routing
Zhou et al., 2022 — NeurIPS, arXiv:2202.09368, §4 "Expert Choice Routing"[15]

Introduces expert-choice routing where each expert selects its top-k tokens rather than tokens selecting experts. Eliminates token dropping and load imbalance. Achieves >2× training convergence speedup and ~20% step-time reduction vs token-choice routing on 8B/64E models. Directly applicable to full-block routing — each full-block expert selects its token quota, guaranteeing load balance. With expert-choice, all experts receive gradient per batch.

### Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
Shazeer et al., 2017 — ICLR, arXiv:1701.06538, §2 "The Sparsely-Gated Mixture-of-Experts Layer"[16]

Foundational sparse MoE paper. Introduces learned top-k gating with noise for load balancing, applied to LSTM language models achieving up to 30× capacity increase at comparable compute. Establishes the gating and routing formalism that all subsequent MoE work builds on. Required foundational citation for any MoE research.

### Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
Fedus et al., 2022 — JMLR, arXiv:2101.03961[17]

First large-scale demonstration of sparse MoE in a Transformer LLM. Canonical scale reference for sparse MoE routing. Required foundational citation.

### Mixtral of Experts
Jiang et al., 2024 — arXiv:2401.04088, §2.1 "Sparse Mixture of Experts"[18]

Canonical k=2, E=8 sparse MoE production LLM (Mistral AI). The k/E values analyzed throughout idea 3.1 (k=2, E=8) match Mixtral's canonical configuration. Most widely deployed production sparse MoE LLM prior to DeepSeek-V3.

### Jamba: A Hybrid Transformer-Mamba Language Model
Lieber et al., 2024 — arXiv:2403.19887[19]

52B production hybrid model (AI21 Labs) mixing Mamba-SSM and Transformer blocks with MoE, using architecturally distinct block types in the same model. The closest production analog to idea 3.1's heterogeneous expert concept. Demonstrates that mixed architectural block types can be trained together stably at scale — directly validating the premise that distinct block architectures can coexist in one model.

### DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
DeepSeek-AI, 2024 — arXiv:2405.04434[20]

Original paper introducing Multi-Head Latent Attention (MLA), which compresses KV cache via low-rank latent projection. MLA is the primary candidate mitigation (Strategy C) for idea 3.1's KV inflation problem. Required citation for any analysis that references MLA.

### Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning
Rosenbaum et al., 2018 — ICLR, arXiv:1711.01239[21]

Formal precursor to idea 3.1's routing concept. Learns to route inputs through sequences of function modules via a separate routing policy network. Idea 3.1 is an instantiation of modular network routing applied at the transformer block granularity. Required citation for conceptual completeness.

---

## 4. Prior Art Classification

**Status:** PARTIAL (~55% covered). The component elements of idea 3.1 are individually well-studied:
- MoE applied to FFN layers: fully covered (Switch Transformer[17], Mixtral[18], DeepSeek-V3[14], Expert Choice[15])
- MoE applied to attention heads/projections: covered (MoA[3], MoH[4], SwitchHead[2])
- Unified attention+FFN routing: partially covered (UMoE[5], UoE[7], MoM[22])
- Binary full-block routing (execute vs. skip): covered (MoD[1])
- Block-level attention sparsity over KV positions: covered (MoBA[11])
- KV cache management for MoE: covered (PiKV[12])
- Full-block routing in vision: covered (MoNE[9])

**Novel contribution (confirmed):** No paper combines all four of: (1) E independently parameterized full transformer blocks each with own W_Q, W_K, W_V, W_O, W_FFN; (2) per-token router selecting k of E at each layer position in autoregressive generation; (3) resulting k-fold KV cache multiplication problem or its mitigations; (4) sparse expert-specific attention histories as a quality degradation mechanism unique to autoregressive full-block MoE.

---

## 5. Technical Analysis

### 5.1 Theoretical Complexity

**Variables:** L = total layer positions; E = total full-block experts; k = active experts per token per layer (k << E); d = hidden dim; d_ff = MLP intermediate dim; s = sequence length; d_kv = KV dim per head × n_kv_heads; P = stride (1 of P layers is full-block MoE).

**Per-layer compute (full block MoE, k active blocks):**
- Each active block: O(d² + s·d + d·d_ff) per token (precise form: O(4d² + 2sd + 3d·d_ff))
- k blocks per token per layer: O(k·(d² + s·d + d·d_ff))
- Router FLOPs: O(d×E) per layer — ~0.01% of block FLOPs, negligible
- The simplified form O(k·(s·d + d·d_ff)) is valid for long sequences (s >> d); for chat-length sequences where s < d, the d² attention projection term contributes ~20% additional compute

**KV cache (full-block MoE, critical analysis):**
Each of E full-block experts has its own W_K, W_V projections. Expert e at position t generates K_e[t], V_e[t]. At a later decode step, expert e can only attend to positions where it was previously active: approximately k/E ≈ 25% of history under load-balanced routing. This produces **sparse expert-specific attention histories** — a fundamentally different failure mode from KV cache size inflation.

- Expected-case KV per full-block layer (load-balanced routing): O(k × s × d_kv)
- Worst-case KV per full-block layer (heterogeneous routing): O(E × s × d_kv)
- Combined over L layers: O(L × s × d_kv × (1 + (k-1)/P)) expected; O(L × s × d_kv × (1 + (E-1)/P)) worst-case

**Weight storage per full-block MoE layer:**

For Qwen3-32B geometry (d=5120, d_kv≈1024, d_ff=25600):
- Q + O projections: 2d² = 2 × 5120² = 52.4M params per block
- K + V projections: 2 × d × d_kv = 10.5M params per block
- FFN (SwiGLU): 3 × d × d_ff = 393.2M params per block
- **Total per block: 456.1M params** (the proxy d×d_ff + d×d_kv = 136.2M understates by 3.35×)

At E=8, P=4 (every 4th layer is full-block MoE), weight storage is **~9.4× Baseline A2**. A full E=8, P=1 configuration at Qwen3-32B geometry approaches ~302B parameters — requiring multi-node expert parallelism.

**TPOT crossover sequence length:**
- Weight saving per full-block layer per decode step: (1 - k/E) × 912 MB ≈ 684 MB (for k=2, E=8, Qwen3-32B)
- KV overhead per full-block layer: (k-1) × d_kv × s × 2 bytes = 2048 × s bytes
- **Break-even: s_crossover ≈ 334,000 tokens** [derived: set weight_saving = KV_overhead; (1 − k/E) × block_weight_bytes = (k−1) × d_kv × s_crossover × 2; block_weight_bytes = 456.1M params × 2 bytes = 912 MB (Qwen3-32B geometry, §5.1); weight_saving = (1 − 2/8) × 912 MB = 0.75 × 912 MB = 684 MB; KV_overhead per full-block layer = (k−1) × d_kv_bytes × s = (2−1) × (1024 heads × 2 bytes) × s = 2048 × s bytes; solve: 684 × 10⁶ = 2048 × s_crossover → s_crossover = 684×10⁶ / 2048 ≈ 334,000 tokens]
- Below ~334K tokens (all typical chat/code/reasoning deployments), weight-load savings exceed KV read overhead — full-block routing **improves** TPOT. The conservative "unknown" characterization in earlier analysis understates the benefit at typical context lengths.

### 5.2 Comparison Table — Key Comparison Tables

**vs. Baseline A1 (Qwen3.5-27B Hybrid)**

| Metric | Baseline A1 | Idea 3.1 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O(L·k·(s·d+d·d_ff)) at full-block layers | ↑ or ↓ by P,k,E | A1 uses cheap DeltaNet for 3/4 layers; full-block experts at all positions may be more expensive |
| KV cache (32K ctx, bf16) | ~2.15 GB (16 full-attn layers only) | ~4.3 GB at P=4,k=2 (2× A1) | ↑ 2× at P=4,k=2 | A1 KV only on 1/4 layers (DeltaNet layers: zero KV); full-block MoE layers add KV |
| Weight memory | O(L·d·d_ff) | O(L/P·E·full_block + ...) | ↑ ~3.7× at E=8,P=4 vs A1 | Uses d_ff=17408 of A1 |
| TTFT (8K prompt) | ref | ~(k/E)× at full-block positions | ↓ conditional on P ≤ 4 | **Critical caveat:** A1 uses DeltaNet for ~3/4 of its layers, giving O(n) prefill complexity on those layers. Replacing a DeltaNet position with a full-attention expert block changes that layer's prefill from O(n·d) to O(n²·d_head) — a quadratic penalty that compounds at context lengths beyond ~8K tokens. At 32K context the per-layer prefill cost at a full-attention expert block is ~16× higher than the DeltaNet it replaces; at 128K context, ~256× higher. The "↓ conditional" improvement holds only if full-block positions remain sparse enough (P large) that the few O(n²) layers are outweighed by O(n) savings elsewhere. For P=1 (every layer is a full-block MoE), TTFT vs A1 is almost certainly worse at context > 8K due to this quadratic blowup. |
| TPOT (batch=1) | ref | Improves for seq < ~334K tokens | ↓ for typical contexts | Crossover at ~334K tokens |

**vs. Baseline A2 (Qwen3-32B Dense)**

| Metric | Baseline A2 | Idea 3.1 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L·k/E·(s·d+d·d_ff)) at P=1 | ↓ E/k × | At k=2, E=8: 0.25× FLOPs at full-block positions |
| KV cache (32K ctx, bf16) | ~8.59 GB | ~10.75 GB at P=4,k=2; ~17.2 GB at P=1,k=2 | ↑ 1.25×–2× | P=4,k=2: overhead factor 1+(k-1)/P=1.25×; P=1: 2× |
| Weight memory | ~64 GB bf16 | ~602 GB at E=8,P=1 (corrected 9.4×); ~170 GB at E=8,P=4 | ↑ ~2.65×–9.4× | Proxy undercount corrected to include Q/O projections |
| TTFT (8K prompt) | ref | ~0.25× at P=1,k=2,E=8 | ↓ 4× | Significant prefill speedup at full routing |
| TPOT (batch=1, s<334K) | ref | Improved (weight savings > KV overhead) | ↓ modest | Crossover at ~334K tokens per first-principles derivation |

**vs. Baseline B (Qwen3.5-397B-A17B MoE)**

| Metric | Baseline B | Idea 3.1 | Change | Notes |
|--------|-----------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | O(L·k·(s·d+d·d_ff)) at full-block positions | ↑ if d_ff >> d_e | B uses fine-grained small experts + DeltaNet for 75% of layers |
| KV cache (32K ctx, bf16) | ~1.0 GB (15/60 GatedAttn layers) | ~10.75 GB at P=4 | ↑ 10× vs B | B avoids KV on 75% of layers via DeltaNet; idea 3.1 adds KV at all full-block positions |
| Weight memory | ~397B total, ~17B active | ↑ toward E×full-block scale | ↑ significantly | Full-block experts larger than fine-grained FFN experts in B |
| TTFT (8K prompt) | ref | ~comparable | = | Both reduce FLOPs via k-of-E routing; comparable active expert compute |
| TPOT (batch=1) | ref | Worse for long context | ↑ worse | B's DeltaNet (75% layers) avoids KV entirely; idea 3.1 at full-block positions adds KV that B never incurs |

**vs. Baseline C (K2 family, 72.55B dense Llama-arch)**

| Metric | Baseline C (K2) | Idea 3.1 | Change | Notes |
|--------|----------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)), L=80, d=8192, d_ff=28672 | O(L·k/E·full_block) at P=1 | ↓ at k/E = 0.25 | K2 is much larger (72B dense); idea 3.1 at 32B-equivalent with E=8,k=2 uses ~0.25× K2 FLOPs at full-block positions |
| KV cache (32K ctx, bf16) | ~10.0 GiB (10.74 GB) (80 layers, GQA 8KV) | ~10.75 GB at P=4,k=2 (A2-geometry) | ≈ comparable | K2 KV at 32K ≈ 10.74 GB decimal (10.0 GiB binary); idea 3.1 at standard geometry slightly above |
| KV cache (262K ctx, bf16) | ~80.0 GiB (85.9 GB) | ~86 GB at P=4,k=2 (A2-geometry) | ≈ comparable | Both require multi-GPU for 262K context |
| Weight memory | ~145.1 GB bf16 | ~170–602 GB at E=8 | ↑ 1.2×–4× | Depends strongly on P; P=4 most practical |
| MLP FLOPs per token/layer | ~4.70 × 10⁸ | ~k/E × (4.70 × 10⁸) at routing positions | ↓ k/E × | Idea 3.1 at comparable geometry: k/E active FLOPs per layer at full-block positions |
| Params (active) | ~72.55B | ~7–15B active at k/E=0.1–0.2 (A2-geometry) | ↓ significantly if routing | Routing k/E = 0.25 yields ~8B active params per forward pass at Qwen3-32B geometry |

---

## 6. Implementation Considerations

**Hardware requirements:** Routing to one of E full blocks per token is equivalent to the existing sparse MoE dispatch problem, but with larger expert blocks. Standard MoE dispatch kernels can be adapted. The critical new kernel requirement is **per-expert KV cache management**: each active expert block must write KV entries to its own cache slot for the current token and read its own KV history for prior tokens. Current inference systems (vLLM, TGI) do not natively support per-expert KV indexing. Custom CUDA/Triton kernels for expert-partitioned KV read/write are required.

Engineering estimates:
- Training prototype (PyTorch): 2–4 weeks with custom KV management
- Training at scale (DeepSpeed/Megatron multi-node): several person-months
- Production inference (vLLM/TGI + custom FlashAttention variant): 3–6 person-months

**KV semantics: three strategies:**

| Strategy | KV Semantics | KV Cache Size | Expert Novelty | Feasibility |
|----------|-------------|--------------|---------------|-------------|
| A: Dense KV (all experts cache all positions) | Correct (full causal) | E× (~68.8 GB at 32K, E=8) | Preserved | LOW — too expensive |
| B: Shared KV, per-expert FFN+value projection | Correct (full causal, shared KV) | 1× (standard ~8.59 GB) | Reduced to SwitchHead+FFN MoE | HIGH — already demonstrated |
| C: Per-expert MLA latent KV | Sparse expert-specific history | k× reduced by MLA factor | Preserved | MEDIUM — novel engineering, open quality question |

Strategy B is feasible but eliminates the core novelty ("distinct attention patterns per token"). Strategy C preserves novelty but leaves the sparse history quality problem open.

**Training stability risks:**

| Risk | Severity | Mitigation | Status |
|------|----------|------------|--------|
| Token load imbalance across experts | MEDIUM | Expert Choice routing (Zhou et al., 2022)[15] | SOLVED — proven at scale |
| Attention diversity collapse (experts learn same pattern) | MEDIUM | No established solution | OPEN PROBLEM |
| Sparse expert history (quality degradation) | HIGH | Token-consistent routing (loses flexibility) | OPEN PROBLEM |
| Copy-initialization vs. random | LOW | Copy-init from dense + diversity regularization | MANAGEABLE |
| E× weight storage (multi-GPU) | HIGH | Tensor/expert parallelism | EFFECTIVE |

---

## 7. Synergies

- **1.1 (Learnable Per-Token Top-k):** adapts k (active block count) per token, reducing compute on "easy" tokens
- **1.2 (Per-Token Adaptive Depth):** complementary — idea 3.1 controls block type per layer, 1.2 controls layer count
- **5.1 (TurboQuant at KV Interface):** compresses the k× KV overhead from full-block experts, directly addressing the key bottleneck
- **1.6 (Learned Layer Type):** the E expert blocks could be of different architectural types (attention, linear attention, SSM), combining into a heterogeneous full-block MoE
- **4.2 (Shared Core Weights + Per-Layer LoRA):** within each expert, core weights shared with per-expert LoRA adapters differentiating them — reduces E× weight growth to 1× core + E × LoRA

**Conflicts:**
- **4.3 (LoRA Everywhere):** sharing base weights across all expert blocks reduces between-expert differentiation
- Standard GQA: assumes shared KV heads across Q heads — conflicts with per-expert KV projections at the inter-expert level

---

## Risk Assessment

**Technical risk: HIGH** — The KV cache multiplication (k× per full-block layer) is a fundamental architectural constraint with no established mitigation that fully preserves quality. The sparse expert KV history problem is a novel, uncharacterized failure mode potentially more severe than size inflation. Routing collapse for full-block experts is higher-risk than for FFN experts (more degrees of freedom to collapse into a dominant pattern).

**Potential impact: HIGH** — If the sparse history semantics are characterized and mitigated (via Strategy C + MLA per-expert or via token-consistent routing), full-block routing enables per-token selection of distinct attention patterns — a qualitative capability gain beyond FFN-only MoE. At k/E = 0.25 active experts, TTFT reduces by ~4× and TPOT improves for all deployments under ~334K sequence length.

**Implementation effort: HIGH** — Custom KV indexing per expert block, expert-dispatch kernels covering attention layers, routing stability infrastructure, and evaluation at scale. None of these components exist in current open-source frameworks in the required combined form.

**Recommended three-stage experimental path:**
1. **Stage 1 — Semantics validation (1–2 weeks, < 1B params):** Compare (a) dense baseline, (b) FFN-only MoE, (c) Strategy B (shared KV + per-expert FFN+value), (d) Strategy C (per-expert MLA KV). Measure perplexity, long-range dependency tasks, attention diversity across experts. **Purpose:** characterize whether sparse expert-specific attention histories cause measurable quality degradation before infrastructure investment.
2. **Stage 2 — KV infrastructure prototype (4–8 weeks):** Implement per-expert KV cache indexing in PyTorch with custom FlashAttention-style kernel for expert-masked KV access.
3. **Stage 3 — Scale validation (GPU budget TBD):** Train 7B-equivalent full-block MoE with Expert Choice routing and best-performing KV strategy from Stage 1. Compare against SwitchHead + FFN-MoE as the "combination of existing techniques" baseline.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **Mixture of Depths (Raposo et al., 2024)[1]**: Binary full-block routing (execute vs. skip) matches baseline performance at equivalent FLOPs, with up to 50% wall-clock speedup during post-training sampling. This is the closest published quality baseline for idea 3.1: a two-expert (execute or bypass) version of full-block routing shows zero measured quality degradation when FLOPs are held constant. However, MoD has only two fixed "experts" with no between-expert diversity — the quality implications of E=8 independently parameterized blocks with different attention patterns are not addressed.
  - **MoH (Jin et al., ICML 2025)[4]**: MoH-LLaMA3-8B achieves 64.0% average accuracy on 14 benchmarks vs. 61.6% for the dense LLaMA3-8B baseline using only 75% of attention heads (+2.4 points). This demonstrates that routing among independent attention head parameterizations (a component of idea 3.1's full-block routing) can *improve* quality by concentrating compute in more informative heads — routing does not necessarily degrade attention quality.
  - **SwitchHead (Csordás et al., NeurIPS 2024)[2]**: A 262M-parameter SwitchHead model matches baseline perplexity with 44% of compute and 27% of memory, using sparse value/output projections in attention. The "SwitchAll" variant (MoE attention + MoE FFN) also matches baseline perplexity. This establishes that combined attention+FFN routing can preserve perplexity at matched compute — the key quality question for idea 3.1 is whether the sparse expert-specific KV history problem creates quality degradation *beyond* the matched-compute baseline.
  - **MoM (Gong et al., EMNLP 2024)[22]**: Independently routed attention and FFN modules (the closest architectural analog to idea 3.1) achieves GPT-2 (774M) parity with 16% fewer TFLOPs and 42% less memory. Removing 50% of attention modules and 25% of FFN modules post-training preserves performance. This provides the strongest existing quality baseline for the idea: independent module routing of both attention and FFN simultaneously incurs negligible quality cost when routing is done at train time.
  - **Expert Choice Routing (Zhou et al., NeurIPS 2022)[15]**: Expert-choice routing achieves >2× training convergence speedup and ~20% step-time reduction vs. token-choice routing on 8B/64E models with matched final quality. The routing mechanism itself does not degrade quality relative to token-choice when load balance is guaranteed — relevant because load-balanced full-block routing should not introduce quality costs from routing mechanics alone.

- **Monotonicity**: Quality degradation from the sparse expert-specific KV history problem is **expected to be non-monotone and task-dependent** rather than a simple function of k or E. Tasks requiring long-range cross-position dependencies (multi-hop reasoning, long-context QA) are expected to degrade most severely as k/E (the fraction of history each expert attends to) decreases. Tasks dominated by local context (code generation with short scopes, single-sentence completion) may show near-zero degradation. The MoD result[1] (zero degradation at binary routing) suggests that with only two experts the history sparsity is mild enough to be undetected, but at E=8 with k=2 each expert attends to only 25% of history under balanced routing.

- **Recovery**: Two recovery mechanisms are available, each with different quality-efficiency tradeoffs:
  - **Token-consistent routing** (routing the same token always to the same expert): eliminates sparse history entirely by ensuring each expert sees a complete causal history for its assigned tokens. Full causal attention quality is preserved. However, this sacrifices per-token routing flexibility — different tokens at the same position always use the same expert, reducing specialization.
  - **Strategy B (shared KV, per-expert FFN + value projection)**: As analyzed in §5.2, Strategy B preserves full causal attention semantics with zero KV overhead above the standard baseline. Quality is expected to match or exceed standard MoE (comparable to SwitchHead[2] results). The tradeoff is that Strategy B reduces the idea's novelty to "SwitchHead + FFN-MoE" — well-studied territory.
  - **Per-expert MLA latent KV (Strategy C)**: Partially recovers quality by compressing history into a low-dimensional latent that all past tokens contribute to. DeepSeek-V2[20] demonstrates that MLA preserves quality with 5–13× KV compression. Whether the expert-specific sparse history is adequately captured in a shared MLA latent remains an open research question.

- **Conditions for acceptable degradation**:
  - Full-block routing with Strategy B (shared KV) is acceptable for any quality-sensitive deployment — this is the MoM[22] / SwitchHead[2] / UMoE[5] regime with demonstrated parity.
  - Full-block routing with Strategy C (per-expert MLA) requires empirical validation at the 7B scale (Stage 1 of the recommended experimental path) before deployment in quality-sensitive contexts.
  - The sparse history quality degradation is most acceptable for **tasks dominated by recent local context** (code completion, short-context chat, summarization of the current window) where long-range cross-position dependencies are rare.
  - The degradation is least acceptable for **multi-hop factual reasoning, long-context document QA, and any task where a token at position T requires information from a specific earlier position T-k** that may have been assigned to a different expert and thus be invisible to the current expert's KV history.
  - Quality risk is also increased during the early training phase when routing is not yet stable — MoD[1] addresses this with a 2-stage training protocol (train router after base model converges); a similar protocol is recommended for idea 3.1 to prevent routing collapse during the period when expert diversity is being established.

<!-- CITATION MANIFEST -->
[1]: Mixture of Depths — Raposo et al., 2024 (arXiv:2404.02258). Top-k per-layer token routing where selected tokens execute full block, others bypass via residual skip; matches baseline performance for equivalent FLOPs at up to 50% wall-clock speedup during sampling.
[2]: SwitchHead — Csordás et al., 2024 (NeurIPS, arXiv:2312.07987). MoE applied to value/output attention projections; 44% compute, 27% memory at matched perplexity for 262M model.
[3]: MoA — Zhang et al., 2022 (EMNLP, arXiv:2210.05144). Per-token routing of k of H attention heads; demonstrates attention-layer MoE feasibility on MT and MLM.
[4]: MoH — Jin et al., 2025 (ICML 2025, arXiv:2410.11842). Weighted-sum head routing; MoH-LLaMA3-8B 64.0% vs 61.6% avg accuracy at 75% heads used.
[5]: UMoE — Yang et al., 2025 (NeurIPS 2025 Spotlight, arXiv:2505.07260). Unified MoE parameterization across attention and FFN via shared expert structure.
[6]: MH-MoE — Wu et al., 2024 (NeurIPS 2024, arXiv:2404.15045). Sub-token decomposition to multiple expert routes; improved expert utilization over standard top-k MoE.
[7]: UoE — Yang et al., 2025 (arXiv:2503.02495). Hierarchical routing over tensor-parallel-decomposed attention and MLP block components.
[8]: RMoE — Qiu et al., 2025 (ICLR 2025, arXiv:2408.06793). GRU-based router propagating routing state across consecutive MoE layers; consistently outperforms independent per-layer routing.
[9]: MoNE — Jain et al., 2024 (NeurIPS, arXiv:2407.19985). Nested-expert routing over ViT tokens; 2× inference speedup at equivalent accuracy on ImageNet-21K.
[10]: MoSA — Piękos et al., 2025 (arXiv:2505.00315, arXiv preprint). Per-head sparse attention routing; drastic KV cache reduction (specific percentage not reported); O(k² + T) per-head complexity.
[11]: MoBA — Lu et al., 2025 (arXiv:2502.13189). MoE-style routing of attention queries to KV context chunks; deployed in Kimi long-context production.
[12]: PiKV — Liu et al., 2025 (arXiv:2508.06526, arXiv preprint). Expert-sharded KV layout + activity-based eviction for MoE KV management; abstract describes the framework qualitatively (specific speedup/memory figures are paper-body values).
[13]: SliceMoE — Vejendla, 2025 (EMNLP 2025, arXiv:2510.04286). Sub-token slice routing to FFN experts; 1.7× speedup with improved load balancing.
[14]: DeepSeek-V3 — DeepSeek-AI, 2024 (arXiv:2412.19437). 671B/37B active MoE with MLA for KV compression and auxiliary-loss-free load balancing; establishes shared-attention MoE as SOTA.
[15]: Expert Choice — Zhou et al., 2022 (NeurIPS, arXiv:2202.09368). Expert-selects-top-k-tokens routing; eliminates token dropping; >2× convergence speedup.
[16]: Sparsely-Gated MoE — Shazeer et al., 2017 (ICLR, arXiv:1701.06538). Foundational sparse MoE; noisy top-k gating for load balance; 30× capacity increase at comparable compute in LSTM LMs.
[17]: Switch Transformer — Fedus et al., 2022 (JMLR, arXiv:2101.03961). First large-scale sparse MoE Transformer; canonical scale reference for routing in LLMs.
[18]: Mixtral — Jiang et al., 2024 (arXiv:2401.04088). Canonical k=2, E=8 production sparse MoE LLM; provides the k/E configuration reference for idea 3.1's analysis.
[19]: Jamba — Lieber et al., 2024 (arXiv:2403.19887). 52B hybrid SSM+Transformer+MoE; demonstrates that architecturally distinct block types train stably at production scale.
[20]: DeepSeek-V2 — DeepSeek-AI, 2024 (arXiv:2405.04434). Original MLA paper; KV compression via low-rank latent projection; primary mitigation candidate (Strategy C) for idea 3.1's KV inflation.
[21]: Routing Networks — Rosenbaum et al., 2018 (ICLR, arXiv:1711.01239). Modular network precursor; per-input routing among function modules via a separate policy network; theoretical foundation for block-level routing in transformers.
[22]: MoM — Gong et al., 2024 (EMNLP 2024, ACL:2024.emnlp-main.1164). Per-token routing of independently parameterized attention and FFN modules via separate routers; achieves GPT-2 (774M) parity with 16% fewer TFLOPs and 42% less memory; closest published analog to idea 3.1's coupled-block routing concept.
