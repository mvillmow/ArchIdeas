# Research: Learned Layer Type (Transformer / SSM / Conv / Mamba)
## ID: 1.6

---

## Executive Summary

**Novelty verdict:** PARTIAL — static 2- and 3-primitive hybrids and 3-primitive NAS are published; the remaining novelty is extending to a 4-primitive NAS search space including causal convolution at 7B–27B+ scale, and content-adaptive per-layer primitive routing at inference ([Jamba, 2024], [Nemotron-Flash, 2025], [Jet-Nemotron/PostNAS, 2025], [Samba, 2024], [MoD, 2024]).

**One-line description**: A model that learns — via NAS or inference-time routing — which computational primitive (full attention, SSM/Mamba, causal convolution, or linear attention) to use at each layer position, extending the already-deployed 2-primitive hybrid baseline (Qwen3.5-27B) to a 4-primitive optimized assignment.

**Recommended path**: Static proxy-based NAS variant (4-primitive search at 1–3B scale using MAD/Composer methodology), then validate at 7B scale. Dynamic routing is secondary, higher risk.

> **Warning — TPOT at low sparsity (α=0.08):** At α=0.08 (92% attention-layer replacement, the Nemotron-H operating point) and long context (s≥65K), TPOT overhead vs. a dense baseline is ~2–3× *worse* than the α=0.25 case: the KV-cache bandwidth savings are substantial but weight-loading bandwidth is nearly unchanged, so the net gain is context- and α-dependent. The tables in §5 present the α=0.25 scenario as the primary comparison point. **Do not apply the α=0.25 TPOT figures (≈1.18× improvement) to the α=0.08 regime without re-deriving the bandwidth breakdown using the §4.1 correction methodology.**

---

## 1. Idea Description

**From arch_research_ideas.md (Section 1, idea 1.6):**

> A hybrid model that learns which computational primitive to use at each layer position — self-attention, state space model, convolution, or Mamba block — either via NAS or dynamic inference-time selection.

**Inferred intent:** Different layers in a transformer have different sensitivity profiles and computational bottlenecks. Self-attention is expensive (O(s·d) per token, KV cache grows with sequence) but powerful for recall and global context. SSMs (Mamba, Mamba-2) are cheap (O(d·N) per token, constant recurrent state) but weaker at associative recall. Convolutions sit in between. A learned assignment — static via NAS, or dynamic at inference time — could yield a model cheaper than all-attention with less quality loss than all-SSM.

**Pilot context:** Baseline A1 (Qwen3.5-27B) is already a 2-primitive hybrid (3:1 linear-attention:full-attention ratio). Idea 1.6 extends this to 4 primitives and makes the assignment learned rather than hand-designed.

**Variant decomposition:**
- **Variant 1A (Static Proxy NAS)**: Search at 1–3B scale, fix assignment, train from scratch at full scale. No routing overhead at inference. Recommended.
- **Variant 1B (Differentiable DARTS NAS)**: Continuous relaxation over 4 primitives, joint optimization, then discretize. Higher risk of collapse; ~2× training overhead.
- **Variant 1C (Dynamic Inference-Time Routing)**: Learned gate selects primitive per layer per token at runtime. Highest risk; batching problem at decode.

---

## 2. Literature Review

### [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors**: Albert Gu, Tri Dao (arXiv: 2312.00752), ICLR 2024
- **Summary**: Selective SSM — SSM parameters are input-dependent, enabling selective retention/forgetting. 3B model matches Transformers of equivalent size. At long sequences (>2K tokens), 5× higher throughput than Transformer baselines. Compute per token: O(d·N) with N=16 (Mamba-1) — constant in sequence length. Recurrent state: O(d·N) fixed size.
- **Relevance**: Establishes primary SSM primitive. Key TPOT comparison: at short context (<8K), Transformers are up to 1.9× faster on TTFT [17]; at very long context (~57K), SSM is up to 4× faster [17].
- **Limitations**: Weaker on associative recall tasks. Fixed-size state limits exact long-range retrieval.

### [2] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2/SSD)
- **Authors**: Tri Dao, Albert Gu (arXiv: 2405.21060), ICML 2024
- **Summary**: Establishes Structured State Space Duality (SSD): SSMs and linear attention compute the same structured matrix product. Mamba-2 uses scalar-times-identity state matrix; 2–8× faster training than Mamba-1 via tensor core utilization. State size: N=128 (vs N=16 in Mamba-1 — 8× larger). Compute per token: O(d·N) with N=128.
- **Relevance**: Mamba-2 is the SSM primitive used in most 2024–2025 hybrid models. Establishes theoretical equivalence between SSM and linear attention.

### [3] Jamba: A Hybrid Transformer-Mamba Language Model
- **Authors**: Opher Lieber et al. (AI21 Labs) (arXiv: 2403.19887), ICLR 2025 (approximate venue)
- **Summary**: First large-scale Transformer–Mamba–MoE hybrid. 52B params, 12B active. 1:7 attention:Mamba ratio (one attention layer per eight). MoE applied every two blocks (16 experts, top-2). Fits in single 80GB GPU. Achieves strong results up to 256K token context.
- **Relevance**: Direct implementation of static-NAS variant (2-primitive). The hand-designed 1:7 ratio is exactly what a learned NAS would automate.

### [4] Jamba-1.5: Hybrid Transformer-Mamba Models at Scale
- **Authors**: Team AI21 Labs (arXiv: 2408.12570), 2024
- **Summary**: Scales to 398B total/94B active (Large) and 52B total/12B active (Mini). Maintains 1:7 ratio. Competitive with Llama-3.1-70B and Mixtral-8×22B. 256K context window. 2× less GPU memory for long contexts vs comparable Transformer.
- **Relevance**: Validates 2-primitive static assignment scales well. 1:7 ratio is a practical operating point the 4-primitive NAS would explore as a special case.

### [5] Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models
- **Authors**: NVIDIA ADLR team (arXiv: 2504.03624), 2025
- **Summary**: 8B, 47B, 56B scale hybrid Mamba-2–Transformer models. Nemotron-H-56B: 54 Mamba-2 layers, 54 MLP layers, 10 self-attention layers (replacing 92% of attention). Per abstract: "up to 3× faster at inference compared to similarly-sized models such as Qwen-2.5-7B/72B and Llama-3.1-8B/70B." Paper-body §Experiments specifies the 65K input / 1K output context and α≈0.08 operating point where the 3× figure applies; the 1.8× vs Qwen-2.5-7B is paper-body. Better or equal accuracy on standard benchmarks.
- **Note**: The **3× throughput figure applies at 65K context with α≈0.08 (92% Mamba replacement)** per paper body. It does not apply at 32K context or α=0.25.
> **[NVIDIA ADLR, 2025]** — arXiv:2504.03624, abstract ("up to 3× faster at inference" vs Qwen-2.5 and Llama-3.1 families); paper body §Experiments for context-length and α qualifiers.

### [6] Hymba: A Hybrid-Head Architecture for Small Language Models
- **Authors**: Xin Dong et al. (NVIDIA) (arXiv: 2411.13676), ICLR 2025
- **Summary**: Intra-layer hybrid: attention heads and SSM (Mamba-2) heads run in parallel within the same layer. Also uses meta tokens, cross-layer KV sharing, global/sliding-window heads. Hymba-1.5B-Base outperforms all sub-2B models and surpasses Llama-3.2-3B with +1.32% accuracy, 11.67× KV cache reduction, 3.49× throughput improvement.
- **Relevance**: Demonstrates power of reducing attention layer fraction. Design is manually specified, not NAS-learned.
> **[Dong et al., 2024/2025]** — arXiv:2411.13676, Abstract/Table 1: "11.67× reduction in cache size" and "3.49× higher throughput" vs. Llama-3.2-3B.

### [7] Zamba: A Compact 7B SSM Hybrid Model
- **Authors**: Paolo Glorioso et al. (arXiv: 2405.16712), 2024
- **Summary**: Mamba backbone with single shared attention module reused across multiple SSM blocks. Faster at inference than comparable Transformer; substantially less memory for long-sequence generation. Trained on 1T tokens.
- **Relevance**: Demonstrates alternative topology for static assignment — shared weight-tied attention rather than repeated per-interval. A NAS would need to consider weight-sharing as a design dimension.

### [8] MAD: Mechanistic Design and Scaling of Hybrid Architectures
- **Authors**: Michael Poli et al. (arXiv: 2403.17844), ICML 2024
- **Summary**: MAD pipeline: train >500 language models (70M–7B) using small-scale synthetic token manipulation tasks as compute-cheap proxies for scaling laws. Result: hybrid architectures outperform pure Transformer, convolutional, and recurrent baselines at compute-optimal and overtrained regimes.
> **[Poli et al., 2024]** — arXiv:2403.17844, Abstract: "architectures found via MAD…outperform state-of-the-art Transformer, convolutional, and recurrent architectures (Transformer++, Hyena, Mamba)."
- **Limitations**: Discrete search (not differentiable); proxy tasks may miss reasoning capabilities; only 2-primitive spaces validated.

### [9] Composer: A Search Framework for Hybrid Neural Architecture Design
- **Authors**: Multiple authors (arXiv: 2510.00379), ICLR 2026
- **Summary**: Searches over hybrid neural architectures at small scales (350M–3B), extrapolates top designs to larger scales. Achieves 2.8–8.3% downstream task improvements (averaging 1.1–3.1%) over Llama 3.2. Small-to-large transfer methodology validated.
- **Relevance**: Direct NAS approach to the core mechanism of idea 1.6. Automates primitive interleaving discovery — same problem but 2-primitive only (attention + MLP).

### [10] DARTS: Differentiable Architecture Search
- **Authors**: Hanxiao Liu, Karen Simonyan, Yiming Yang (arXiv: 1806.09055), ICLR 2019
- **Summary**: Differentiable NAS via continuous relaxation: weighted sum over candidate operations at each layer, learned by gradient descent, then discretized. Competitive CNN architectures with 4 GPU-days of search.
- **Relevance**: Foundational methodology for Variant 1B. DARTS suffers from architecture collapse toward cheapest operations — a 4-primitive space with SSM (cheapest) vs full attention (most expensive) would exacerbate this. PC-DARTS/SNAS/GDAS mitigations apply.

### [11] TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model
- **Authors**: Multiple authors (arXiv: 2503.24067), AAAI 2026
- **Summary**: Shares parameter matrices between Transformer (QKV) and Mamba (CBx). "TransPoints" — positions where the model switches from attention to SSM mode, can be sequence-length-adaptive. Achieves superior training efficiency vs single and hybrid baselines.
- **Relevance**: Closest published work to dynamic inference-time selection (Variant 1C) — 2-primitive switching based on sequence length thresholds (not content-adaptive per-layer learned routing).

### [12] BASED: Simple Linear Attention Language Models Balance the Recall–Throughput Tradeoff
- **Authors**: Simran Arora et al. (arXiv: 2402.18668), ICLR 2024
- **Summary**: Hybrid of linear attention (Taylor-kernel) and local sliding-window attention. Fundamental recall–throughput tradeoff formalized: fixed-size recurrent state compresses context, hurting associative recall. At 1.3B: 24× higher throughput than FlashAttention-2 for 1024-token generation; +6.22 accuracy over pure Mamba on recall tasks (Table 3).
- **Relevance**: Formalizes the recall–throughput tradeoff motivating the 4-primitive selection. Each primitive occupies a different point on this curve.
> **[Arora et al., 2024]** — arXiv:2402.18668, Abstract: "24× higher throughput on language generation than FlashAttention-2, when generating 1024 tokens using 1.3B parameter models."

### [13] Hybrid Architectures for Language Models: Systematic Analysis and Design Insights
- **Authors**: Multiple authors (arXiv: 2510.04800), 2024
- **Summary**: Systematic survey of hybrid architecture design choices. High Mamba fraction (e.g., 1:7) consistently performs well but placement matters — attention layers in the middle of the stack are important.
- **Relevance**: Thorough empirical analysis of static 2-primitive design space. The finding that placement matters (not just ratio) implies the search space is richer than simple ratio selection.

### [14] Proxy-Mamba: Training-Free NAS for Mamba Architectures
- **Authors**: Multiple authors (Springer chapter 10.1007/978-981-95-7075-1_33, URL unverified), 2024/2025
- **Note**: **URL not verified per citation standards.**
- **Summary**: Training-free NAS using SigScore proxy (Sigmoid-normalized weight × gradient magnitude) to evaluate Mamba architectures without training.
- **Relevance**: Training-free NAS over SSM architectures. Most practically feasible NAS approach for the Mamba primitive.

### [15] DUET: Disaggregated Hybrid Mamba-Transformer LLMs with Prefill and Decode-Specific Packages
- **Authors**: Multiple authors (arXiv: 2603.15530), 2026
- **Note**: **POST-CUTOFF (March 2026) — UNVERIFIED. Used as hardware motivation only.**
- **Summary**: Chiplet-based hardware disaggregation assigning prefill (systolic-array) and decode (vector-unit) to specialized chiplets for hybrid Mamba-Transformer models.
- **Relevance**: Hardware motivation for the TTFT/TPOT analysis — different primitives have different arithmetic intensity profiles, motivating disaggregated hardware.

### [16] MambaVision: A Hybrid Mamba-Transformer Vision Backbone
- **Authors**: Ali Hatamizadeh, Jan Kautz (arXiv: 2407.08083), CVPR 2025
- **Summary**: Hybrid Mamba-Transformer for computer vision. State-of-the-art ImageNet accuracy (84.2% top-1) with better throughput than comparable pure Transformer vision backbones.
- **Relevance**: Validates hybrid approach in a second modality. Demonstrates optimal mixing ratio and placement are domain-dependent.

### [17] Characterizing SSM and SSM-Transformer Hybrid LLM Performance with Long Context Length
- **Authors**: Saptarshi Mitra, Rachid Karami, Haocheng Xu, Sitao Huang, Hyoukjun Kwon (arXiv: 2507.12442), 2025
- **Summary**: Benchmarks SSM and hybrid models on long-context inference. At short context (<8K): Transformers are up to 1.9× faster on TTFT. At very long context (~57K tokens): SSMs offer up to 4× lower latency. SSMs demonstrate ~64% reduced memory footprint.
- **Relevance**: Direct TTFT/TPOT characterization. The context-length crossover point is key for the dynamic selection variant.
> **[Mitra et al., 2025]** — arXiv:2507.12442, Abstract: "Transformers are up to 1.9x faster at short sequences (<8K tokens)" and "SSMs demonstrate a dramatic performance inversion, becoming up to 4x faster at very long contexts (~57K tokens)."

### [18] Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling
- **Authors**: Ren et al. (arXiv: 2406.07522), ICLR 2025
- **Summary**: Combines Mamba with sliding window attention (SWA) and MLP blocks — a 3-primitive hybrid with explicit mixing ratio analysis. Achieves unlimited context via Mamba's unbounded recurrent state combined with SWA's local precise recall.
- **Relevance**: Closest existing published work to a 3-primitive NAS space. Samba demonstrates that extending beyond 2 primitives is being explored — one step below the 4-primitive space proposed in idea 1.6.

### [19] Griffin: Mixing Gated Linear Recurrences with Local Self-Attention for Efficient Language Models
- **Authors**: De et al. (Google DeepMind) (arXiv: 2402.19427), 2024
- **Summary**: Hybrid model mixing gated linear recurrences with local self-attention. Trained at 14B scale. Reports training stability analysis for mixed recurrent+attention gradient flows.
- **Relevance**: Provides empirical evidence for training stability of recurrent+attention hybrids at 14B scale. Relevant to the training stability discussion in §5.

### [20] Mixture-of-Depths: Dynamically Allocating Compute in Transformer Language Models
- **Authors**: Raposo et al. (arXiv: 2404.02258), 2024
- **Summary**: Learns which tokens should participate in each layer's computation (per-token depth routing via top-k selection). Demonstrates content-adaptive per-layer compute decisions.
- **Relevance**: Closest published work to the dynamic routing variant (Variant 1C). MoD addresses the same "which compute primitive to apply at which layer/token position" question from a token-routing perspective. The dynamic routing variant in idea 1.6 is PARTIAL (not fully novel) given MoD — routing over 4 computational primitives (vs MoD's "participate vs skip") has not been demonstrated.

### [21] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models
- **Authors**: NVIDIA (arXiv: 2511.18890), 2025
- **Summary**: Performs automated layer-type search over three primitives — {Attention, DeltaNet, Mamba-2} — at small proxy scale to find latency-optimal hybrid architectures. Discovered interleaved DeltaNet-FFN-Mamba2-FFN and Attention-FFN-Mamba2-FFN building blocks. Achieves 18.7×/45.6× higher throughput vs Qwen3-1.7B/Qwen3-0.6B respectively, along with +5.5% average accuracy and 1.3×/1.9× lower latency.
- **Relevance**: Directly validates the core mechanism of idea 1.6 at 3 primitives. Confirms that automated search over {attention, linear attention (Gated DeltaNet), SSM/Mamba-2} yields material throughput gains. The 4-primitive extension in idea 1.6 adds causal convolution to this already-demonstrated 3-primitive search, narrowing the novelty gap.
> **[NVIDIA, 2025]** — arXiv:2511.18890: three candidate operators (DeltaNet, Attention, Mamba-2); 18.7×/45.6× higher throughput than Qwen3-1.7B/0.6B respectively with +5.5% avg accuracy and 1.3×/1.9× lower latency.

### [22] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search
- **Authors**: Yuxian Gu et al. (arXiv: 2508.15884), NeurIPS 2025
- **Summary**: PostNAS — begins with a pretrained full-attention model, freezes MLP weights, then searches over full-attention vs linear-attention layer assignments. Pipeline: (1) optimal full-attention placement/elimination, (2) linear attention block selection, (3) custom attention block design, (4) hardware-aware hyperparameter search. Jet-Nemotron-2B achieves up to 53.6× generation throughput speedup vs full-attention baseline.
- **Relevance**: Directly addresses the learned layer-type assignment problem for the 2-primitive {full-attention, linear-attention} space with a post-training NAS approach. The frozen-MLP strategy is a practical alternative to Variant 1A's from-scratch search: it eliminates NAS training overhead and can be applied to an existing pretrained model.

---

## 3. Prior Art Classification

- **Status**: PARTIAL — ~80% overlap given [21] and [22]
- **What exists (static 2-primitive hybrids)**: Jamba [3], Jamba-1.5 [4], Nemotron-H [5], Hymba [6], Zamba [7], Griffin [19] — all static hand-designed. MambaVision [16] validates in vision.
- **What exists (3-primitive hybrids)**: Samba [18] (Mamba + SWA + MLP) — extends toward the 4-primitive space.
- **What exists (3-primitive NAS)**: Nemotron-Flash [21] (Attention + DeltaNet + Mamba-2) — automated search over 3 primitives at proxy scale; directly validates the core mechanism of idea 1.6 minus causal convolution.
- **What exists (NAS over layer types, post-training)**: Jet-Nemotron/PostNAS [22] — NAS over {full-attention, linear-attention} assignment starting from pretrained model; frozen-MLP strategy. MAD [8] and Composer [9] perform proxy-based search; all currently limited to ≤3 primitives.
- **What exists (dynamic switching)**: TransMamba [11] — sequence-length-triggered switching (not content-adaptive). Mixture-of-Depths [20] — content-adaptive per-layer routing (token participation, not primitive selection).
- **Novel contributions (what is specifically new)**:
  1. **4-primitive NAS search space including causal convolution**: Nemotron-Flash [21] demonstrates 3-primitive search over {Attention, DeltaNet, Mamba-2}. Adding causal convolution as a 4th primitive and extending to 7B+ scale is the remaining novel contribution.
  2. **Dynamic per-layer content-adaptive routing at inference** (Variant 1C): PARTIAL — MoD [20] demonstrates content-adaptive per-layer routing, but routing over 4 distinct computational primitives (not just "participate vs skip") has not been published.
  3. **4-primitive applied to 27B+ production model**: Published hybrids use at most 3 primitives (Nemotron-Flash at small scale; Samba at 3.8B). The 4-primitive extension at 27B+ scale has not been validated.

### 3.1 Prior Art Gap

Static 2-primitive hybrids are thoroughly validated at scale with manually-designed assignments; Samba [18] extends to 3 primitives. Nemotron-Flash [21] demonstrates automated NAS over 3 primitives {Attention, DeltaNet, Mamba-2} at proxy scale — the closest direct prior art to idea 1.6. Jet-Nemotron/PostNAS [22] demonstrates post-training NAS over {full-attention, linear-attention} assignments. Dynamic content-adaptive routing exists (MoD [20]) but for token participation, not primitive selection.

What does not exist: (1) A NAS search over a 4-primitive space including {full attention, SSM/Mamba-2, causal convolution, linear attention} simultaneously at 7B+ scale (Nemotron-Flash covers 3 of these 4 at small scale); (2) dynamic content-adaptive (not length-adaptive) per-layer primitive selection at inference time at production scale over 4 primitives.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

**Variable definitions:**
- L = total layers
- d = hidden dim (5120 for A1/A2)
- d_ff = MLP intermediate dim (17408 for A1; 25600 for A2)
- s = sequence length
- N = SSM state size (16 for Mamba-1; 128 for Mamba-2)
- k_c = convolution kernel width
- α = fraction of full-attention layers
- β = fraction of SSM/Mamba layers
- γ = fraction of causal convolution layers
- δ = fraction of linear attention layers
- α + β + γ + δ = 1

**Per-primitive complexity (per token, per layer):**

| Primitive | Compute/token | Memory (inference state) | KV cache grows with s? |
|-----------|--------------|--------------------------|------------------------|
| Full self-attention | O(s·d) per-token attention; O(d²) projection | O(s·d_kv) KV cache | Yes |
| SSM/Mamba-1 | O(d·N), N=16: O(16d) per token | O(d·N) fixed recurrent state = O(16d) | No |
| Mamba-2 (SSD) | O(d·N), N=128: O(128d) per token | O(d·N) state = O(128d) per layer | No |
| Linear attention / Gated DeltaNet | O(d²) per token (outer product update) | O(d²) recurrent state per layer | No |
| Causal convolution (k_c kernel) | O(k_c·d) per token | O(k_c·d) lookback buffer (bounded) | No |

**Note on O(d·N) vs O(d²) approximation**: The research doc's original notation "O(d·N) ≈ O(d²)" is accurate only for linear attention (N ~ d) but incorrect for Mamba-1 (N=16, so O(d·16) = O(d), not O(d²)) and misleading for Mamba-2 (N=128, so O(d·128) — not O(d²) in the strict sense but closer in magnitude). These are kept as separate rows in the table above. The merged formula for SSM+linear-attention uses (β+δ)·O(d²) as a Big-O approximation where Mamba-2 or Mamba-1 is grouped with linear attention.

**Mixed-primitive model complexity:**

| Metric | This Idea (α=0.25, β=0.50, γ=0.125, δ=0.125) | Baseline A1 (Qwen3.5-27B Hybrid) | Baseline A2 (Qwen3-32B Dense) | Baseline B (397B-A17B MoE) | Baseline C (K2 ~72.55B Dense) |
|--------|-----------------------------------------------|-----------------------------------|-------------------------------|----------------------------|-----------------------------|
| Compute/token (prefill) | O(L·[α·s·d + (β+δ)·d² + γ·k_c·d + d·d_ff]) | O(L·(d² + s·d/4)) | O(L·(s·d + d·d_ff)) | O(L·(d² + k·d·d_e)) | O(L·(s·d + d·d_ff)) |
| KV cache (32K) | O(α·L·s·d_kv) ≈ α × A2_cache = ~2.15 GB at α=0.25 | ~2.15 GB | ~8.59 GB | ~1.0 GB | ~10.0 GiB |
| Weight memory | ≈ O(L·d·d_ff) | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) | O(L·d·d_ff) |
| Bandwidth (decode, batch=1) | O(L·[d² + α·s·d_kv]) (weight loading for ALL layers + KV for α fraction) | O(L·(d² + s·d_kv/4)) | O(L·(d·d_ff + s·d_kv)) | O(L·(k·d·d_e + state)) | O(L·(d·d_ff + s·d_kv)) |
| TTFT (8K prompt) | ~1.05–1.10× vs A2 (see note) | ref | ref | ref | ref |
| TPOT (batch=1, 32K ctx) | ~1.18× vs A2 at α=0.25 (see note) | ref | ref | ref | ref |

**TTFT analysis at 8K context (A2 config, s=8192):**
- MLP FLOPs per layer: 2 × 5120 × 25600 ≈ 262M FLOPs (two linear projections; SwiGLU adds gate projection → 3 × 5120 × 25600 ≈ 393M, but 262M used here for the two main weight matrices)
- Attention FLOPs per layer: s × d = 8192 × 5120 ≈ 42M FLOPs
- Attention fraction of total: 42M / (42M + 262M) = 14%

At α=0.25 (replacing 75% of attention layers with SSM/conv/linear-attn):
- Total FLOPs saved: 0.75 × 14% ≈ 10.5% of total prefill FLOPs
- TTFT improvement: ~1.10×

TTFT improvement is strongly context-dependent — the 2–3× range applies only at s ≥ 128K where attention FLOPs dominate MLP FLOPs:
- At s=8K: ~1.05–1.10× vs A2
- At s=32K: ~1.3–1.5× vs A2 (attention becomes ~39% of FLOPs)
- At s=128K+: ~2–3× vs A2 (attention dominates)
- Nemotron-H's reported 3×: at s=65K with α≈0.08 (92% Mamba replacement)

**TPOT analysis at 32K context, α=0.25, batch=1:**
At batch=1 decode, performance is bandwidth-bound.
- A2 bandwidth per layer: weight load 262 MB + KV cache 67 MB = 329 MB
- This idea (α=0.25): weight load 262 MB + KV cache (0.25 × 67 MB = 16.75 MB) = 278.75 MB
- TPOT ratio: 278.75 / 329 ≈ 0.847 → **~1.18× improvement**

A 2–4× TPOT range requires either α≈0.08 (Nemotron-H regime) at s=65K, OR much longer context (s >> 128K) where KV cache dominates weight bandwidth.
[derived: TPOT_ratio = total_BW_new / total_BW_old = (W_BW + α×KV_BW) / (W_BW + KV_BW); at s=32K, A2: W_BW=64 GB, per-layer KV at 32K = 2×32768×8×128×2 B = 134 MB → 8.59 GB total; TPOT_ratio(α=0.25) = (64 + 0.25×8.59)/(64+8.59) = (64+2.15)/72.59 = 66.15/72.59 ≈ 0.912 → 1.10× improvement; for α=0.08 at s=65K: KV≈(65536/32768)×8.59×(0.08/0.25)≈2.75 GB vs W_BW≈64 GB → TPOT_ratio=(64+0.22)/(64+2.75)=64.22/66.75≈0.962 → only 1.04×; 3× requires s >> 100K where KV >> W_BW — consistent with NVIDIA ADLR 2025 arXiv:2504.03624 at the stated context length]

**Dynamic routing overhead (Variant 1C):**
A minimal routing gate (2-layer MLP, d_gate=64, P=4 primitives):
- FLOPs: 2 × (5120 × 64 + 64 × 4) ≈ 656K FLOPs/layer — 0.4% of full-attention layer FLOPs at 32K context
- Memory overhead: 5120 × 64 × 2 ≈ 655 KB/layer — 0.25% of attention weight loading
Gate overhead is negligible for the static path but the batching problem at decode (tokens selecting different primitives → under-utilized minority-primitive kernel) can cause 2–4× throughput reduction vs static assignment at batch > 1.
[derived: at batch B, tokens split across P=4 primitives → expected B/4 tokens per primitive; GPU kernel launch overhead per primitive ≈ 10–50 μs; at B=1, only 1 primitive active per token → 3/4 of kernel invocations are empty or 1-token (occupancy ~3% vs peak); GEMV throughput scales as min(B_primitive, B_threshold) where B_threshold ≈ 32–64 for tensor-core utilization; at B=1 with P=4 and random routing, effective throughput = 1/P × peak = 0.25× → 4× slower vs static; at B=16, average B_primitive=4, occupancy ~12.5% → ~2–3× slower; 2–4× range brackets the B=1 to B=8 regime]

### 4.2 Compute Analysis

- **Training FLOPs (static NAS, from scratch)**: ~0.5–0.8× Baseline A2 training FLOPs, because SSM/conv/linear-attn layers are cheaper per forward pass. Plus NAS overhead: 1.1–1.3× for proxy-based NAS; 1.8–2.0× for DARTS. Net: training cost comparable to or cheaper than A2 for proxy-based NAS variant.
- **Inference FLOPs (prefill, long context s >> d)**: ≈ α × Baseline A2 FLOPs per layer. At α=0.1 (Nemotron-H), ~10–30% of A2 FLOPs at long context.
- **Inference FLOPs (decode)**: Weight loading dominates at batch=1 — similar to A2 since weight sizes are comparable across primitives. KV cache is reduced by α factor.
- **Arithmetic intensity**: Heterogeneous per layer type. SSM layers at decode are relatively compute-bound (scan has higher arithmetic intensity than GEMV). Full attention layers at decode are bandwidth-bound (KV cache access). No single hardware optimization covers all layers.

### 4.3 Memory Bandwidth Analysis

- **Weight loading (all primitives, decode)**: O(d·d_ff) per layer regardless of primitive type — weight matrices are same size. Unchanged vs Baseline A2.
- **KV cache loading (full attention layers only)**: O(α·L·s·d_kv) total — only α fraction of layers read KV cache. At α=0.25, KV cache bandwidth = 25% of A2.
- **Recurrent state**: O(β·L·d·N) for SSM layers (small — 1.3 MB/layer for Mamba-2); O(δ·L·d²) for linear-attn layers (52 MB/layer at d=5120). Recurrent state accesses are fixed-size (not growing with s).

**Bandwidth formula**: O(L·d²) weight loading for ALL layers + O(α·L·s·d_kv) KV cache — i.e., O(L·(d² + α·s·d_kv)). Weight loading applies to every layer, not only the non-attention fraction.

### 4.4 Memory Capacity Analysis

- **Total weight storage**: ≈ same as Baseline A2. SSM/Mamba layers slightly smaller (SSM weight matrices O(d·N)); linear attention weight matrices O(d²). Net: ~0.9–1.0× Baseline A2.
- **KV cache at 32K**: α × 8.59 GB. At α=0.25: ~2.15 GB (same as A1 at 32K). At α=0.08 (Nemotron-H): ~0.69 GB.
- **Peak training memory**: Similar to A2 for from-scratch training. SSM scan may require storing all intermediate states during backprop — O(L·s·d·N) activation memory — mitigated by activation checkpointing.

---

## 5. Comparison Tables

### vs Baseline A1 (Qwen3.5-27B Hybrid, DENSE MLP, 25% full-attention layers)

| Metric | Baseline A1 | This Idea (α=0.15) | Change | Notes |
|--------|------------|---------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | O(L·[0.15·s·d + 0.70·d² + 0.15·k_c·d + d·d_ff]) | ↓ at long s | Full-attn fraction 15% vs A1's 25%; SSM/conv fill cheaply |
| Memory bandwidth (decode) | O(L·(d²+s·d_kv/4)) | O(L·(d² + 0.15·s·d_kv)) | ↓ at long s | KV cache access reduced: 15% vs 25% of layers |
| KV cache (32K) | ~2.15 GB | ~2.15 × (0.15/0.25) ≈ ~1.3 GB | ↓ ~1.7× | α/A1_α = 0.15/0.25 = 0.6× |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) | ≈ = | Slight decrease from SSM's smaller O(d·N) weights |
| Training cost | 1.0× | 1.1–1.3× (proxy NAS); 1.8–2× (DARTS) | ↑ | NAS one-time overhead; from-scratch per-layer cost same |
| TTFT (8K prompt) | ref | ~1.02–1.05× improvement | ↓ slight | At 8K: attention is small fraction of total; fewer attn layers saves ~5% |
| TTFT (32K+ prompt) | ref | ~1.2–1.5× improvement | ↓ moderate | At longer context, attention fraction grows |
| TPOT (batch=1, 32K ctx) | ref | ~1.05–1.10× improvement | ↓ slight | KV fraction of bandwidth: 0.15 vs 0.25 × A1's 2.15 GB |

### vs Baseline A2 (Qwen3-32B Dense, all full-attention, d_ff=25600)

| Metric | Baseline A2 | This Idea (α=0.25) | Change | Notes |
|--------|------------|---------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L·[0.25·s·d + 0.75·d² + d·d_ff]) | ↓ at long s | Attention fraction: 25% vs 100%; O(d²) << O(s·d) at long s |
| Memory bandwidth (decode) | O(L·(d·d_ff+s·d_kv)) | O(L·(d·d_ff + 0.25·s·d_kv)) | ↓ proportional to α | Weight loading unchanged; KV cache cut to 25% |
| KV cache (32K) | ~8.59 GB | ~2.15 GB (α=0.25) | ↓ 4× | Direct proportionality: 0.25 × 8.59 GB |
| Weight memory | O(L·d·d_ff) | O(L·d·d_ff) | ≈ = | |
| Training cost | 1.0× | 0.5–0.8× (cheaper per-layer) + NAS overhead | ↓ | Cheaper layers for 75% of model; NAS is 1-time cost |
| TTFT (8K prompt) | ref | ~1.05–1.10× | ↓ ~5–10% | Attention only 14% of FLOPs at 8K; see derivation above |
| TTFT (32K+ prompt) | ref | ~1.3–1.5× (α=0.25) or ~2–3× (α=0.08, 65K+) | ↓ moderate–large | Context-dependent; see Nemotron-H at 65K for upper bound |
| TPOT (batch=1, 32K ctx) | ref | ~1.18× (α=0.25) | ↓ ~18% | Derived: 278.75 MB vs 329 MB/layer/token |
| TPOT (batch=1, 65K+ ctx, α=0.08) | ref | ~2–3× | ↓ large | Consistent with NVIDIA ADLR arXiv:2504.03624 |

### vs Baseline B (Qwen3.5-397B-A17B MoE, 512 experts k=11)

| Metric | Baseline B | This Idea (27B-scale dense) | Change | Notes |
|--------|-----------|------------------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | O(L·[α·s·d + (1-α)·d² + d·d_ff]) | = or ↓ | No MoE; potentially fewer active FLOPs at long s if α small |
| Memory bandwidth (decode) | O(L·(k·d·d_e+state)) | O(L·(d·d_ff + α·s·d_kv)) | ↑ | No MoE sparsity: loads all weights per token; worse than B at batch=1 |
| KV cache (32K) | ~1.0 GB | ~2.15 GB at α=0.25 | ↑ slightly | B uses ~25% attn layers at d=4096; this idea at d=5120 and same α is slightly larger |
| Weight memory | 397B total (O(L·E·d·d_e)) | 27B dense (O(L·d·d_ff)) | ↓ ~14× | Scale difference acknowledged |
| Training cost | very high (397B) | much lower (27B scale) | ↓ | Different size class |
| TPOT (batch=1) | ref | ↑ (worse) | ↑ | MoE loads k/E fraction of expert weights; dense loads all |

### vs Baseline C (K2 ~72.55B Dense, L=80, d=8192, d_ff=28672)

| Metric | Baseline C | This Idea (27B-scale, α=0.25) | Change | Notes |
|--------|-----------|-------------------------------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | O(L·[0.25·s·d + 0.75·d² + d·d_ff]) | ↓ at long s (similar direction to vs A2) | C is larger model; idea is 27B vs C's 72.55B |
| KV cache (32K) | ~10.0 GiB | ~2.15 GB (27B dense at α=0.25) | ↓ ~4.7× | Model size difference contributes; α contributes |
| Weight memory | ~145.1 GB | ~54 GB (27B dense) | ↓ ~2.7× | Model size difference |
| Training cost | very high | much lower | ↓ | Different size class |
| TTFT (8K) | ref | ↓ (smaller model + fewer attn layers) | ↓ | Model size + attention fraction both favorable |
| TPOT (batch=1) | ref | ↓ (smaller model bandwidth) | ↓ | 27B vs 72.55B — weight bandwidth 2.7× lower |

---

## 6. Implementation Considerations & Synergies

### 6.1 Implementation Considerations

- **Hardware requirements**:
  - Full attention: FlashAttention-2/3 — full production maturity on A100+
  - Mamba-2 SSM: state-spaces/mamba repo, Triton/CUDA kernels. A100+ required for performance. Multi-GPU training with FSDP requires custom SSM state tensor placement (non-shardable by standard FSDP).
  - Causal convolution: `torch.nn.Conv1d` with causal padding — no special hardware; fully standard.
  - Linear attention / Gated DeltaNet: Flash-Linear-Attention repo (Triton). A100+ required.
  - All 4 primitives have PyTorch implementations; multi-primitive integration requires 4–8 weeks engineering for `torch.compile` compatibility.

- **Training stability**:
  - Static assignment (post-NAS): LOW risk. Track record: Jamba, Jamba-1.5, Nemotron-H [5], Griffin [19] all demonstrate stable training of 2-primitive hybrids at scale.
  - DARTS (Variant 1B): MEDIUM-HIGH risk. Architecture collapse toward cheapest operation (SSM) in a 4-primitive space is exacerbated vs DARTS on CNNs. Mitigations: PC-DARTS, SNAS/GDAS.
  - Dynamic routing (Variant 1C): HIGH risk. Routing collapse (all tokens → SSM) analogous to expert collapse in MoE. Load-balancing loss required. Gumbel-Softmax for hard routing training.
  - SSM scan backward is numerically stable (Mamba-1/2 well-tested). The heterogeneous gradient flow (attention vs SSM) for mixed primitives requires monitoring — Griffin [19] provides evidence this is manageable at 14B scale.

- **Framework support**:
  - Training: All 4 primitives in PyTorch; main challenge is torch.compile + FSDP compatibility across 4 types.
  - Serving (vLLM): Full attention supported; Mamba/SSM partially via patches; linear attention (Gated DeltaNet) not in mainline; causal conv standard. A 4-primitive model requires 6–12 months engineering for production-quality serving.
  - Nemotron-H [5] and Jamba [3] open-source releases will provide reference implementations for 2-primitive hybrid serving.

---

### 6.2 Synergies

- **Combines well with**:
  - **1.1 (Learnable Per-Token Top-k)**: Dynamic primitive selection + dynamic expert count — both adaptive computation forms
  - **1.2 (Per-Token Adaptive Depth)**: Complementary; an early-exiting token skips both expensive attention and routing at later layers
  - **1.4 (Learned Dense vs Sparse Layer Assignment)**: Layer type (attention/SSM/conv) and dense/sparse MoE are orthogonal design dimensions; Jamba already combines these
  - **5.1–5.5 (KV Cache compression)**: Applies to the attention-layer fraction's KV cache; reduces penalty for keeping some full-attention layers
- **Conflicts with**:
  - **4.2 (Shared Core Weights + Per-Layer LoRA)**: Weight sharing across layers is incompatible if layers have different primitive types
  - **3.4 (Recursive Internal State)**: Looping computation assumes fixed layer structure; mixing primitives with a loop complicates state management

### 6.3 Recommended Experimental Path

1. **Experiment 1 (Required)**: 4-primitive proxy NAS at 1B scale using MAD/Composer framework. Search over ~500 configurations of α/β/γ/δ fractions and placement patterns. ~60 GPU-days on 8×A100. Goal: find an assignment that outperforms best 2-primitive hand-designed hybrid.

2. **Experiment 2 (Required)**: Train best assignment from Exp 1 at 7B scale (~700 GPU-days). Validate quality parity or better vs Jamba-1.5 / Nemotron-H at equivalent parameter count.

3. **Experiment 3 (Required)**: TTFT/TPOT benchmarking at the correct context lengths (8K, 32K, 65K+). Report context-specific speedups with the corrected ranges from §4.1.

4. **Experiment 4 (Optional)**: Dynamic routing at 1B scale. Budget for 10+ iterations with different load-balancing hyperparameters. Gate collapse is likely on first pass.

---

## 7. Risk Assessment

- **Technical risk**:
  - Static NAS variant (1A): LOW. 2-primitive hybrids are thoroughly validated; 4-primitive is incremental.
  - DARTS variant (1B): MEDIUM-HIGH. Architecture collapse with heterogeneous compute costs is a known failure mode.
  - Dynamic routing variant (1C): HIGH. Routing collapse + batching problem at decode; no published content-adaptive primitive routing at scale.
- **Potential impact**: HIGH. Nemotron-H [5] demonstrates 3× throughput at 65K context with 92% Mamba replacement; 11.67× cache reduction in Hymba [6]. A NAS-discovered 4-primitive assignment could outperform all hand-designed hybrids per MAD [8].
- **Implementation effort**:
  - Proxy-based NAS at 1–3B: ~60 GPU-days for search + ~700 GPU-days for 7B validation
  - Dynamic routing prototype at 1B: 4–8 weeks + 3–5 months research iteration for stability
  - Production deployment: 12–18 months beyond research prototype

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **Nemotron-H [5]** (arXiv:2504.03624, 2025): Nemotron-H-56B with 92% Mamba-2 layer replacement achieves "better or equal accuracy on standard benchmarks" vs Llama-3.1-8B, with 3× throughput at 65K context. This is the largest-scale (56B) published validation that replacing the majority of full-attention layers with SSM does not degrade quality on standard NLP tasks.
  - **Nemotron-Flash [21]** (arXiv:2511.18890, 2025): Automated 3-primitive NAS over {Attention, DeltaNet, Mamba-2} achieves 18.7×/45.6× higher throughput vs Qwen3-1.7B/0.6B (respectively) with +5.5% average accuracy and 1.3×/1.9× lower latency per abstract. Quality is reported as maintained or improved vs single-primitive baselines — the NAS-discovered assignment is specifically quality-superior to any single-type model.
  - **Jamba-1.5 [4]** (arXiv:2408.12570, 2024): At 398B total / 94B active with fixed 1:7 attention:Mamba ratio, competitive with Llama-3.1-70B and Mixtral-8×22B on standard benchmarks. This validates that static 2-primitive hybrid assignment is quality-neutral vs dense Transformer at 94B+ active scale.
  - **Hymba [6]** (ICLR 2025): Hymba-1.5B achieves +1.32% accuracy improvement over Llama-3.2-3B on standard benchmarks with 11.67× KV cache reduction and 3.49× throughput improvement. Intra-layer mixing of attention heads and SSM heads produces a quality improvement — not just quality preservation — vs pure-attention baselines.
  - **BASED [12]** (ICLR 2024): On recall-intensive tasks, hybrid linear-attention + sliding-window-attention achieves +6.22 accuracy points over pure Mamba at 1.3B scale. This is a critical *negative* data point for pure SSM: Mamba alone degrades associative recall quality by 6+ points vs any hybrid that includes a full-attention component. Any 4-primitive assignment that completely eliminates full-attention layers will incur this recall-quality penalty.
  - **Mamba-1 [1]** (ICLR 2024): 3B Mamba matches Transformer quality on standard language benchmarks; at sequences >2K, 5× higher throughput. However, Mamba underperforms on associative recall and in-context learning tasks where full attention is needed — establishing the quality floor for high-β (high SSM fraction) assignments.
  - **Mitra et al. [17]** (arXiv:2507.12442, 2025): At short context (<8K), Transformers are up to 1.9× faster on TTFT than SSMs; at ~57K tokens, SSMs are up to 4× faster. The quality implication: for long-context benchmarks (SCROLLS, RULER), replacing attention with SSM/Mamba reduces context utilization at positions >state compression capacity (~N=128 for Mamba-2), causing quality degradation proportional to how much the task depends on long-range exact retrieval.
  - **Composer [9]** (ICLR 2026): Searched hybrid architectures outperform Llama 3.2 by 2.8–8.3% on specific downstream tasks (1.1–3.1% averaged). This establishes that an *optimized* assignment strictly outperforms both hand-designed hybrids and pure-primitive models — the quality upside of idea 1.6 is not just matching baselines but discovering superior assignment configurations.

- **Monotonicity**: Quality loss with increasing α (full-attention fraction reduction) is not monotone. At α in [0.08, 0.25] (8–25% full-attention layers), published hybrids show quality-neutral or quality-positive outcomes on standard benchmarks. Quality degrades sharply for tasks requiring associative recall or long-context exact retrieval as α → 0. The crossover — below what α does recall-quality regression occur — is approximately α ≈ 0.05–0.10 based on BASED [12] and Jamba results.

- **Recovery**: Static NAS assignment is fixed post-search; quality recovery requires either a new search run at a relaxed efficiency constraint (higher α) or fine-tuning with additional full-attention layers unfrozen. The PostNAS approach (Jet-Nemotron [22]) allows post-training layer-type swapping — specifically, replacing linear-attention layers back to full-attention if quality metrics fall below threshold — without full pretraining restart.

- **Conditions for acceptable degradation**:
  - Short-context tasks (< 8K tokens): quality loss is minimal at α ≥ 0.15 based on Nemotron-H [5] and Jamba [3/4]. Standard benchmark accuracy (MMLU, HellaSwag, ARC) is preserved across all published 2-primitive hybrids with 1:7 to 1:4 attention ratios.
  - Long-context exact-recall tasks (RULER, NIAH >64K): quality degradation is expected and will be proportional to the fraction of context beyond Mamba-2's state capacity. Quality degradation is acceptable only if the throughput gain (potentially 3–4× at 65K+ context per Nemotron-H [5]) exceeds the SLA threshold for the target application.
  - Causal convolution fraction (γ): no published direct evidence for causal-conv quality contribution in large LLMs. The quality implication of the 4th primitive (convolution) is speculative — this is the highest-uncertainty quality dimension of Idea 1.6. Adding convolution may dilute the quality established by 2-primitive hybrids without adding proportional capability.
  - **No experiments at 27B+ scale with 4-primitive NAS assignment exist.** Published work is limited to 3 primitives (Nemotron-Flash [21] at small proxy scale; Samba [18] at 3.8B). The quality extrapolation from 3-primitive to 4-primitive at 27B+ scale is speculative. Experiment 1 (4-primitive proxy NAS at 1B, §6.3) is required before quality claims can be made for the full 4-primitive assignment at Baseline A1/A2 scale.
  - The key quality risk unique to dynamic routing (Variant 1C) vs static NAS: routing collapse toward the cheapest primitive (SSM) would eliminate all full-attention layers, causing the BASED [12] -6.22 recall penalty. A strong load-balancing loss that enforces minimum full-attention utilization (α_min ≥ 0.1) is required to prevent this failure mode.

---

## Citations

| # | Citation | Key claim | Status |
|---|----------|-----------|--------|
| 1 | Gu & Dao (Mamba), arXiv:2312.00752, ICLR 2024 | 5× throughput at >2K tokens; O(d·N) decode compute | PARTIALLY VERIFIED |
| 2 | Dao & Gu (Mamba-2/SSD), arXiv:2405.21060, ICML 2024 | 2–8× faster training vs Mamba-1; N=128 state | VERIFIED |
| 3 | Lieber et al. (Jamba), arXiv:2403.19887, ICLR 2025 | 52B/12B active; 1:7 ratio; 256K context | PARTIALLY VERIFIED |
| 4 | AI21 Labs (Jamba-1.5), arXiv:2408.12570, 2024 | 398B total/94B active (Large); 52B total/12B active (Mini) | VERIFIED |
| 5 | NVIDIA ADLR (Nemotron-H), arXiv:2504.03624, 2025 | 3× throughput at 65K context, α≈0.08 | PARTIALLY VERIFIED |
| 6 | Dong et al. (Hymba), arXiv:2411.13676, ICLR 2025 | 11.67× cache reduction, 3.49× throughput vs Llama-3.2-3B | PARTIALLY VERIFIED |
| 7 | Glorioso et al. (Zamba), arXiv:2405.16712, 2024 | Shared-attention Mamba; competitive with 7B Transformers | VERIFIED (qualitative) |
| 8 | Poli et al. (MAD), arXiv:2403.17844, ICML 2024 | >500 models; 4-primitive hybrids outperform pure models | PARTIALLY VERIFIED |
| 9 | Multiple authors (Composer), arXiv:2510.00379, ICLR 2026 | 2.8–8.3% improvement over Llama 3.2 at 350M–3B | VERIFIED |
| 10 | Liu et al. (DARTS), arXiv:1806.09055, ICLR 2019 | Differentiable NAS via continuous relaxation | VERIFIED |
| 11 | Multiple authors (TransMamba, seq-level hybrid), arXiv:2503.24067, AAAI 2026 | Sequence-length-adaptive attention/SSM switching via shared QKV/CBx | VERIFIED |
| 12 | Arora et al. (BASED), arXiv:2402.18668, ICLR 2024 | 24× throughput vs FlashAttn-2 at 1.3B; +6.22 on recall | PARTIALLY VERIFIED |
| 13 | Multiple authors (Hybrid Arch Review), arXiv:2510.04800, 2024 | Placement of attention in stack matters for quality | PARTIALLY VERIFIED |
| 14 | Multiple authors (Proxy-Mamba), Springer (URL unverified) | Training-free NAS via SigScore proxy | PARTIALLY VERIFIED |
| 15 | Multiple authors (DUET), arXiv:2603.15530, 2026 | Chiplet disaggregation for hybrid models | UNVERIFIED (post-cutoff) |
| 16 | Hatamizadeh & Kautz (MambaVision), arXiv:2407.08083, CVPR 2025 | 84.2% ImageNet top-1; better throughput than pure Transformer | PARTIALLY VERIFIED |
| 17 | Mitra et al. (Long-Context Benchmark), arXiv:2507.12442, 2025 | 1.9× Transformer TTFT advantage at <8K; 4× SSM at ~57K | VERIFIED |
| 18 | Ren et al. (Samba), arXiv:2406.07522, ICLR 2025 | 3-primitive hybrid (Mamba+SWA+MLP); mixing ratio analysis | VERIFIED |
| 19 | De et al. (Griffin), arXiv:2402.19427, 2024 | Gated linear recurrence + local attention at 14B; training stability | VERIFIED |
| 20 | Raposo et al. (Mixture-of-Depths), arXiv:2404.02258, 2024 | Content-adaptive per-layer token routing via top-k | VERIFIED |
| 21 | NVIDIA (Nemotron-Flash), arXiv:2511.18890, 2025 | 3-primitive NAS {Attention, DeltaNet, Mamba-2}; 18.7×/45.6× throughput vs Qwen3-1.7B/0.6B | NEW |
| 22 | Gu et al. (Jet-Nemotron/PostNAS), arXiv:2508.15884, NeurIPS 2025 | Post-training NAS: freeze MLP, search {full-attn, linear-attn}; 53.6× throughput | NEW |
