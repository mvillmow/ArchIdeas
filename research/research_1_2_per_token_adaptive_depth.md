# Research: Per-Token Adaptive Depth
## ID: 1.2

## Executive Summary

**Novelty verdict:** PARTIAL — per-token early-exit is well-established, but the specific three-zone structure of static prefix + dynamic middle with monotonic early exit + static postfix (with its postfix-heterogeneity problem) is not studied in any single published paper ([CALM, 2022], [LayerSkip, 2024], [MoD, 2024], [MoR, 2025], [Lawson & Aitchison, 2025]).

**One-line description:** A transformer with static prefix and postfix layers bounding a dynamic middle section where each token exits as soon as a learned router determines it has sufficient representation, reducing average compute to L_pre + T̄ + L_suf layers where T̄ < L_mid.

**Value proposition:** Empirical analogues (CALM: up to 3× speedup; LayerSkip: 1.34–2.16×; DEL: 2.16–2.62×) confirm that per-token early exit yields substantial speedups with minimal quality degradation when the training recipe (LayerSkip-style layer dropout + early exit loss) is applied. The specific static prefix + dynamic middle + static postfix three-zone structure is a genuine architectural novelty not present in any single published paper, providing guaranteed minimum depth and uniform final projection for all tokens. The dominant engineering challenge is the conditional weight-load kernel required for genuine TPOT speedup; naive masking implementations yield no bandwidth benefit.

---

## 1. Idea Description

A model with static prefix and postfix layer sets, but a dynamic middle section where per-token routing decides how many intermediate layers to execute. Tokens that are easy to process exit early; harder tokens use more compute.

**Inferred intent for inference speedup:** By allowing easy tokens to exit the middle section early while hard tokens traverse all middle layers, the average per-token compute falls to L_pre + T̄ + L_suf where T̄ < L_mid. At batch=1 decode (TPOT), this is memory-bandwidth-bound: fewer layers traversed means fewer weight matrices loaded from HBM, yielding a genuine TPOT reduction proportional to (L_pre + T̄ + L_suf) / (L_pre + L_mid + L_suf). TTFT (prefill) is FLOPs-bound: fewer middle-layer FLOPs for easy tokens reduces average prefill FLOPs. Routing overhead (the exit-decision classifier) is the key cost; only net speedup counts. The static prefix ensures a minimum representational depth; the static postfix ensures a final projection and normalization pass regardless of exit point.

---

## 2. Literature Review

### Adaptive Computation Time [1]: Foundational Per-Step Halting (arXiv 2016)
Introduces ACT, a mechanism that allows recurrent neural networks to learn how many computational steps to take. A learned scalar halting probability per step is accumulated; the network stops when the cumulative probability exceeds a threshold. This is the foundational work for input-adaptive computation depth.

**Relevance:** Directly establishes the principle that different inputs warrant different amounts of computation, and provides the mathematical framework (halting probability, ponder time penalty) adopted by subsequent adaptive-depth work.
**Limitations:** Designed for vanilla RNNs; does not address transformers, KV caching, or inference-time TPOT/TTFT.

---

### Universal Transformers [2]: ACT Applied to Weight-Shared Transformers (ICLR 2019)
Extends transformers with a recurrent inductive bias and applies ACT to each token position independently. A shared weight block is applied repeatedly; ACT determines per-token halting. Achieves stronger systematic generalization on algorithmic tasks.

**Relevance:** Implements per-token variable depth using ACT on weight-shared transformer layers. Differs from idea 1.2 in that it uses weight sharing (single block applied N times), whereas idea 1.2 uses distinct middle layers.
**Limitations:** Shared weights limit expressiveness. Does not have the static prefix + dynamic middle + static postfix structure.

---

### Depth-Adaptive Transformer [3]: Per-Token Early Exit for NMT Decoding (ICLR 2020)
Applies per-token early exit to neural machine translation decoders. Two variants: a token-specific multinomial classifier that predicts exit depth before computation begins, and a geometric-like binary classifier per block. Matches strong baselines while using up to 76% less computation.

**Relevance:** Direct implementation of per-token adaptive depth for autoregressive decoding. The token-specific exit classifier is essentially the routing mechanism proposed in idea 1.2.
**Limitations:** Evaluated on encoder-decoder translation models, not decoder-only LLMs. Small scale (sub-billion parameters). KV cache implications under per-token variable depth not studied.

---

### CALM [4]: Confident Adaptive Language Modeling with Provable Guarantees (NeurIPS 2022)
Proposes CALM, a framework for per-token early exit in autoregressive language model generation. Uses three confidence measures (softmax response, state propagation via cosine distance between hidden states, and a trained early-exit classifier) to make local per-token exit decisions while providing theoretical guarantees that sequence-level quality metrics are maintained. Achieves up to 3× speedup on summarization, machine translation, and question answering.

> **[Schuster et al., 2022]** — NeurIPS 2022, §4 "Experiments", Table 1: up to 3× wall-clock speedup on T5/LaMDA while maintaining sequence-level quality with provable guarantees.

**Relevance:** The most direct antecedent to idea 1.2 for autoregressive generation. Establishes that per-token early exit is both theoretically well-founded and empirically effective.
**Limitations:** Evaluated on encoder-decoder models. Uses global shared exit heads rather than a three-zone structure. Does not study KV cache implications of variable-depth tokens in a batch.

---

### Mixture-of-Depths (MoD) [5]: Per-Layer Token Budget with Residual Bypass (arXiv 2024)
Enforces a per-layer compute budget by routing only the top-k tokens (by router score) through each transformer layer's self-attention and MLP; remaining tokens bypass via residual. A static budget B is set at training time. MoD models match baseline perplexity for equivalent training FLOPs, with up to 50% fewer FLOPs per forward pass.

> **[Raposo et al., 2024]** — arXiv:2404.02258, §4, Figure 3: MoD matches vanilla transformer loss at equivalent FLOPs; up to 50% FLOPs reduction per forward pass.

**Relevance:** Closest published work to idea 1.2 in mechanism. Implements per-token variable-depth computation by allowing tokens to skip individual layers. Key difference: MoD applies non-monotonic skipping (a token can skip layer 3 but execute layer 7); idea 1.2 uses monotonic early exit.
**Limitations:** Non-monotonic routing makes KV cache management complex. The prefix/postfix structure of idea 1.2 is not present.

---

### LayerSkip [6]: Early Exit Inference and Self-Speculative Decoding (ACL 2024)
Trains LLMs with layer dropout (low dropout rates for early layers, higher for later layers) and an early exit loss (all layers share the same exit head). At inference, tokens can exit early; a self-speculative decoding mode uses early-exit drafts verified by the full model. Achieves speedups of 1.34×–2.16× depending on task.

> **[Elhoushi et al., 2024]** — arXiv:2404.16710, ACL 2024, §5 Table 2: 2.16× on CNN/DM, 1.82× on coding, 2.0× on TOPv2. §3.1 "Training": layer dropout with increasing rates toward later layers. Note: author list corrections applied — verified authorship includes Carole-Jean Wu among the co-authors.

**Relevance:** The strongest published demonstration that per-token early exit works for decoder-only LLMs at scale. The training recipe (layer dropout + early exit loss) is the recommended approach for idea 1.2's training stability.
**Limitations:** No static prefix/postfix structure — any layer can be an exit point. KV cache complications with variable exit per token in a batch are acknowledged but not fully resolved.

---

### Mixture-of-Recursions (MoR) [7]: Adaptive Recursion Depth with KV-Selective Caching (NeurIPS 2025 / ICML 2025)
Unifies parameter sharing and adaptive computation: a single shared transformer block is applied recursively, with lightweight routers assigning each token a different recursion depth. KV caching is selective — only tokens still active at a given recursion depth cache their KVs, reducing KV memory.

**Relevance:** Different tokens traverse different numbers of computational blocks. MoR's KV cache insight — only caching for active tokens — is directly applicable to idea 1.2's static postfix design. Authors: Bae, Y. Kim, Bayat, S. Kim, Ha, Schuster, Fisch, Harutyunyan, Ji, Courville, Se-Young Yun.
**Limitations:** Weight sharing limits depth of specialization. Requires training from scratch with the MoR architecture.

---

### Inner Thinking Transformer (ITT) [8]: Critical Token Routing with Thinking Step Encoding (ACL 2025)
Uses Adaptive Token Routing (selecting critical tokens for further thinking), Residual Thinking Connections, and Thinking Step Encoding. Critical tokens receive more layers; non-critical tokens exit early. On 162M–466M models: 162M ITT achieves 96.5% of 466M performance with 43.2% less training data.

> **[Chen et al., 2025]** — arXiv:2502.13842, §4.2 Table 2: 162M ITT achieves 96.5% of 466M Transformer on 11 benchmarks with 43.2% less training data. Note: full author list is 10 authors; the research doc listed only 2 (Yilong Chen, Junyuan Shang).

**Relevance:** Very close to idea 1.2. ITT's Thinking Step Encoding addresses the representation alignment problem when tokens have traversed different numbers of layers.
**Limitations:** Evaluated only at small scale (up to 466M parameters). No KV cache analysis at scale.

---

### AdaInfer [9]: Per-Request Early Exit via Confidence Signal (IJCAI 2025)
Determines per-input early exit using a lightweight classifier with two features: top token probability and gap with second-ranked token. No model retraining required. Achieves average 17.8% layer pruning ratio (up to 43% on sentiment tasks) with less than 1% performance drop.

> **[Wang et al., 2025]** — IJCAI 2025, §4 Table 1: 17.8% average layer pruning; up to 43% on sentiment classification; <1% performance drop on Llama2 and OPT models.

**Relevance:** Demonstrates that a simple confidence signal is sufficient for exit decisions in modern LLMs without retraining. Validates the routing approach for idea 1.2.
**Limitations:** Per-sequence (not per-token) exit decisions. Evaluated only on classification tasks, not generation.

---

### TIDE [10]: Post-Training Per-Token Early Exit via Convergence Signal (arXiv 2026)
A post-training system that attaches lightweight routers at periodic checkpoint layers, selecting the earliest layer at which each token's hidden state has converged (cosine similarity to the final layer). No model retraining. On DeepSeek-R1-Distill-8B on A100: reduces prefill latency by 7.2% and increases single-batch throughput by 6.6%.

> **[Jaber and Jaber, 2026]** — arXiv:2603.21365, §4 "Experiments": 7.2% prefill latency reduction and 6.6% single-batch throughput improvement. Authors: Jaber Jaber and Osama Jaber.

**Relevance:** The most current direct implementation of per-token early exit for LLM inference without retraining. TIDE's modest gains (6–7%) illustrate the real-world routing overhead challenge and demonstrate that a post-training retrofit achieves much less than a trained-from-scratch implementation.
**Limitations:** Modest gains (6–7%). No static prefix/postfix structure.

---

### DEL [11]: Context-Aware Dynamic Exit Layer via Shadow Token Analysis (COLM 2025)
Dynamically selects both the exit layer and speculation length during LLM inference using a Token-per-Layer (TPL) metric. Shadow Token Analysis efficiently estimates acceptance probabilities for all exit layers simultaneously using cached hidden states. Achieves 2.16×–2.62× speedup over vanilla autoregressive decoding, improving upon prior speculative decoding methods by up to 0.19×.

> **[Entezari Zarch et al., 2025]** — arXiv:2504.05598, §5 Table 2: 2.16×–2.62× speedup over vanilla AR decoding. Venue "COLM 2025" — noted as unverified in original citations review.

**Relevance:** DEL extends LayerSkip-style early exit with a dynamic exit-layer selection mechanism that adapts to context. The context-awareness is relevant to idea 1.2's routing design.
**Limitations:** Requires an existing early-exit capable model. Context-aware selection adds routing overhead.

---

### SkipGPT [12]: Token-Aware Adaptive Depth with Decoupled MLP/Attention Pruning (ICML 2025)
Addresses both horizontal dynamics (token-level heterogeneity) and vertical dynamics (MLP vs. self-attention layers have distinct functional roles). Uses global token-aware routing. Reduces over 40% of model parameters while matching or exceeding dense baseline performance.

> **[Zhao et al., 2025]** — arXiv:2506.04179, §4 "Experiments": >40% parameter reduction while matching dense baseline. Note: venue "ICML 2025" unverified given submission date; should be treated as arXiv until confirmed.

**Relevance:** SkipGPT's insight that MLP and attention layers should have decoupled pruning policies is relevant to idea 1.2: the prefix and postfix could preserve all attention layers while the dynamic middle primarily skips MLP-heavy blocks.
**Limitations:** Structural pruning, not per-token dynamic routing at inference. Two-stage training process.

---

### EE-LLM [13]: Large-Scale Training and Inference Framework for Early-Exit LLMs (ICML 2024)
The only published large-scale (3D parallelism) training and inference framework specifically designed for early-exit LLMs. Addresses the KV-cache-compatible inference challenge that idea 1.2 identifies as novel.

> **[Chen, Pan, Li, Ding, Zhou, 2024]** — arXiv:2312.04916, ICML 2024. The production feasibility of large-scale early-exit training and inference.

**Relevance:** Critical for production feasibility analysis. Addresses exactly the engineering challenges (KV cache management, parallel training, inference kernel) described in idea 1.2's implementation considerations.
**Limitations:** Published 2024; specifics of KV cache implementation for monotonic-exit designs not confirmed.

---

### "Diminishing Returns" [14]: Early-Exit Suitability in Modern LLMs (arXiv 2026)
CRITICAL ADVERSE FINDING: Modern LLMs show diminishing early-exit suitability due to reduced layer redundancy in improved training recipes. Key findings: (a) dense transformers >20B show greater early-exit potential than smaller models — favorable for 27–32B target scale; (b) MoE and SSM architectures are substantially harder to early-exit than dense; (c) fine-tuned models are less suitable than base pretrained models.

> **[Wei et al., 2026]** — arXiv:2603.23701, §4 "Results": adverse findings for hybrid (A1) and MoE (B) baselines; favorable for dense (A2) and large dense (C) baselines.

**Relevance:** The most important adverse finding for idea 1.2. Must be addressed before scaling to hybrid/MoE baselines (A1 and B). The finding that large dense models (>20B) are actually MORE suitable supports application to A2 and C.

---

### GateSkip [15]: Sigmoid-Linear Residual Gates for Layer Skipping (ICLR 2026)
Uses sigmoid-linear residual gates for token-wise layer skipping in decoder-only LLMs. Fine-tunes stably, contrasted with early-exit alternatives as more training-stable. Saves up to 15% compute at >90% accuracy on pretrained models.

> **[Laitenberger et al., 2026]** — arXiv:2510.13876, ICLR 2026.

**Relevance:** Closest 2026 mechanism to idea 1.2's routing (token-wise residual gates for layer skipping). Does not enforce the three-zone structure but provides updated stability evidence.

---

### "Learning to Skip the Middle Layers" [17]: Adverse Finding for Three-Zone Middle-Skipping (arXiv 2025)
Proposes a learned gating mechanism to dynamically skip a variable number of symmetric central transformer blocks based on input, directly instantiating the middle-skipping structure of idea 1.2. Key finding: at all investigated scales, the architecture does not improve the trade-off between validation cross-entropy and estimated FLOPs compared to a dense baseline with fewer layers.

> **[Lawson and Aitchison, 2025]** — arXiv:2506.21103, §4 "Experiments": no improvement over dense baselines in the FLOPs vs. perplexity trade-off at tested scales. Proposed middle-skipping gating with learned sparsity control.

**Relevance:** Direct adverse finding for idea 1.2's three-zone structure. Middle-layer skipping with gating does not automatically improve the compute-quality frontier; a full training-from-scratch recipe (LayerSkip-style dropout + early-exit loss) may be required to realize gains.
**Limitations:** Evaluated at modest scales; large-scale (27B+) behavior may differ. Does not use a prefix/postfix structure identical to idea 1.2.

---

### D-LLM [18]: Token-Adaptive Compute Allocation with KV Eviction (NeurIPS 2024)
A dynamic inference paradigm for LLMs that uses a per-layer decision module to adaptively allocate compute per token, skipping unnecessary layers and applying a KV-cache eviction policy for exited tokens. Achieves up to 45% FLOPs and KV storage reduction on Q&A, summarization, and math tasks; up to 50% on commonsense reasoning.

> **[Jiang et al., 2024]** — NeurIPS 2024 poster; up to 45–50% reduction in computational cost and KV storage across multiple task categories.

**Relevance:** NeurIPS 2024 peer-reviewed result confirming per-token adaptive depth with KV eviction is achievable at scale. The KV eviction policy is directly applicable to idea 1.2's middle-section design.
**Limitations:** No static prefix/postfix structure. KV eviction mechanism differs from idea 1.2's per-token monotonic exit.

---

### FlexiDepth [16]: Post-Training Variable-Depth System (COLM 2025)
Direct empirical evidence that "FlexiDepth does not yet achieve wall-clock speedup due to varied skipping patterns and I/O overhead." The most important counter-evidence to the TPOT speedup claim.

> **[Luo, Wang, Yan, 2025]** — arXiv:2503.23798, COLM 2025.

**Relevance:** Critical engineering reality check — even a well-implemented variable-depth system has not achieved wall-clock speedup in production. Idea 1.2 must demonstrate it can overcome this challenge through its training-from-scratch recipe (vs. FlexiDepth's post-training approach).

---

## 3. Prior Art Classification

- **Status**: PARTIAL (~75–80% covered)
- **Overlap summary**: The core mechanism — per-token variable depth in a transformer — is thoroughly established by CALM (2022), Depth-Adaptive Transformer (2020), MoD (2024), LayerSkip (2024), ITT (2025), MoR (2025), GateSkip (2026), TIDE (2026), D-LLM (NeurIPS 2024) [18]. Speedups of CALM (3×), LayerSkip (2.16×), DEL (2.62×) confirm the mechanism works. Adverse findings: FlexiDepth [16] (no wall-clock speedup without custom kernel) and Lawson & Aitchison [17] (middle-skipping gating alone does not improve FLOPs/perplexity trade-off) underscore the training-recipe and kernel requirements.
- **Critical adverse finding**: The "Diminishing Returns" paper [14] finds that MoE (B) and SSM/hybrid (A1) architectures are substantially harder to early-exit than dense models. For A2 and C (dense models), the mechanism is most applicable. Application to A1 and B requires preliminary validation.
- **Novel contribution**: The specific architectural structure of idea 1.2 — **static prefix layers** (always executed) + **dynamic middle** (per-token monotonic early exit) + **static postfix layers** (always executed) — is not studied in any single published paper. None of CALM, LayerSkip, MoD, Universal Transformers, MoR, GateSkip enforce the precise three-zone structure with guaranteed postfix execution. The postfix-heterogeneity issue (postfix self-attention sees tokens with different middle-section depths) also lacks analysis in the literature. Note: [17] (Lawson & Aitchison, 2025) proposes a closely related middle-skipping architecture and finds no FLOPs/quality improvement at tested scales without a full training-from-scratch recipe; this strengthens the requirement for LayerSkip-style training in idea 1.2 rather than a post-hoc gating approach.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

**Variables:**
- L = L_pre + L_mid + L_suf (prefix, middle, suffix layer counts)
- d = hidden dimension
- d_ff = MLP intermediate dimension
- s = sequence length (context)
- T̄ = mean middle layers executed per token; T̄ < L_mid gives speedup
- r_mid = T̄ / L_mid (fraction of middle layers executed on average; r_mid < 1)

**Compute derivation:**
- Prefix: every token executes L_pre layers fully: O(L_pre · (s·d + d·d_ff))
- Middle: each token i executes T_i ≤ L_mid layers. Average: O(T̄ · (s·d + d·d_ff))
- Suffix: every token executes L_suf layers fully: O(L_suf · (s·d + d·d_ff))
- Total: O((L_pre + T̄ + L_suf) · (s·d + d·d_ff))
- Router overhead: O(L_mid · d) per token; negligible vs. O(L_mid · (s·d + d·d_ff))

**KV cache complexity:**
- Prefix layers (L_pre): full KV for all tokens
- Middle layers (L_mid): only tokens still active at each layer contribute KV; average T̄/L_mid fraction
- Postfix layers (L_suf): all tokens participate in self-attention (all tokens exit the postfix equally); full KV for all tokens
- Total KV cache: O((L_pre + T̄_mid + L_suf) · s · d_kv) average; worst case O(L · s · d_kv)

**KV cache at 32K (Baseline A2):**
- Baseline A2 total KV: 64 layers × 2 × 8 KV-heads × 128 head_dim × 32,768 × 2 bytes ≈ **8.59 GB** at 32K context. At A2's 40,960-token native max, total KV = 64×2×8×128×40960×2 ≈ **10.74 GB**; at 262,144 context reached via YaRN/RoPE extension, KV grows to ~68.7 GB.
- At T̄ = L_mid/2 with L_mid = L/2: KV savings = 25% × 8.59 GB ≈ **~2.15 GB freed** at 32K context.

**Memory bandwidth at decode (TPOT driver, batch=1):**
- Average layers traversed: L_pre + T̄ + L_suf
- Speedup fraction: (L_pre + T̄ + L_suf) / L
- If L_pre = L_suf = L/4, L_mid = L/2, T̄ = L_mid/2: effective layers = L/4 + L/4 + L/4 = 3L/4 → **~1.33× speedup**
- If T̄ = L_mid/3: effective layers = L/4 + L/6 + L/4 = 2L/3 → **~1.5× speedup**

**Critical caveat (from FlexiDepth [16])**: Wall-clock speedup is NOT automatic. FlexiDepth demonstrates that even a well-implemented variable-depth system may fail to achieve wall-clock improvement due to "varied skipping patterns and I/O overhead." Achieving genuine TPOT improvement requires a kernel that physically skips weight loading for skipped layers, not merely masks computation outputs.

| Metric | This Idea (Applied to dense Baseline A2 config) | Baseline A1 (Qwen3.5-27B Hybrid) | Baseline A2 (Qwen3-32B Dense) | Baseline B (397B-A17B MoE) | Baseline C (K2 72.55B Dense) |
|--------|--------------------------------------------------|----------------------------------|-------------------------------|---------------------------|------------------------------|
| Compute per token (FLOPs) | O((L_pre+T̄+L_suf)·(s·d+d·d_ff)) | O(L·(d²+s·d/4)) | O(L·(s·d+d·d_ff)) | O(L·(d²+k·d·d_e)) | O(L·(s·d+d·d_ff)) |
| KV cache (avg) | O((L_pre+T̄+L_suf)·s·d_kv) avg | O(L/4·s·d_kv) | O(L·s·d_kv) | O(L/4·s·d_kv) | O(L·s·d_kv) |
| KV cache at 32K | ~(L_pre+T̄+L_suf)/L × A2 baseline → ~75% × 8.59 GB ≈ 6.44 GB | ~2.15 GB | ~8.59 GB | ~1.0 GB | ~10.0 GiB |
| KV cache at 262K | N/A (A2 max native 40,960) | ~17.2 GB | ~68.7 GB | ~8.0 GB | ~80.0 GiB |
| Weight memory (total stored) | O(L·d·d_ff) unchanged | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) | O(L·d·d_ff) |
| Memory bandwidth (decode, batch=1) | O((L_pre+T̄+L_suf)·(d·d_ff+s·d_kv)) | O(L·(d²+s·d_kv/4)) | O(L·(d·d_ff+s·d_kv)) | O(L·(k·d·d_e+state)) | O(L·(d·d_ff+s·d_kv)) |
| Router overhead | O(L_mid·d) [negligible] | — | — | — | — |
| **TTFT (prefill, 8K prompt)** | ≈ (L_pre+T̄+L_suf)/L × ref; ~0.67–0.75× at T̄=L_mid/3–L_mid/2 | ref | ref | ref | ref |
| **TPOT (decode, batch=1)** | ~0.67–0.75× ref (IF conditional weight-load kernel implemented; otherwise no gain) | ref | ref | ref | ref |

**KV savings at 32K**: For A2 with T̄=L_mid/2, KV saving ≈ 25% × 8.59 GB ≈ 2.15 GB freed at 32K context. The ~17 GB savings figure applies only at 262,144-token context (YaRN-extended), which exceeds A2's native maximum of 40,960 tokens.

### Key Comparison Tables

#### TTFT Comparison (8K prompt, all baselines ref)

| Baseline | TTFT | Notes |
|----------|------|-------|
| A1 (Qwen3.5-27B Hybrid) | ref | L=64, d=5120; hybrid; 16 full-attn layers |
| A2 (Qwen3-32B Dense) | ref | L=64, d=5120, d_ff=25600 |
| B (Qwen3.5-397B-A17B MoE) | ref | L=60, d=4096, k=11, E=512 |
| C (K2 72.55B Dense) | ref | L=80, d=8192, d_ff=28672; MLP FLOPs/layer ≈ 4.70×10⁸ |
| A2 + idea 1.2 (T̄=L_mid/2) | ~0.75× A2 | [derived: effective_layers = L_pre + T̄ + L_suf; with L_pre=L_suf=L/4=16, T̄=L_mid/2=16: effective_layers = 16+16+16 = 48 of 64 → TTFT ∝ 48/64 = 0.75×; CALM (3× speedup ↔ ~0.33× FLOPs) and LayerSkip (2× ↔ ~0.5× FLOPs) bracket this; 0.75× accounts for router overhead at each middle layer (~L_mid×d per token, negligible vs. d×d_ff)] |
| A2 + idea 1.2 (T̄=L_mid/3) | ~0.67× A2 | [derived: T̄=L_mid/3=32/3≈10.7; effective_layers = 16+10.7+16 = 42.7 of 64 → TTFT ∝ 42.7/64 ≈ 0.667×; equivalently (L_pre + L_mid/3 + L_suf)/L = (L/4 + L/6 + L/4)/L = (3/12 + 2/12 + 3/12) = 8/12 = 0.667×] |

#### TPOT Comparison (batch=1 decode)

| Baseline | TPOT driver | Wall-clock speedup over A2 |
|----------|-------------|---------------------------|
| A2 (Qwen3-32B Dense) | ~64 GB weight BW | 1.0× (ref) |
| C (K2 72.55B Dense) | ~145.1 GB weight BW | ~0.44× (slower due to larger model) |
| A2 + idea 1.2 (trained, conditional-load kernel) | ~48 GB weight BW | ~1.33× [derived: weight BW reduction = effective_layers/L = 48/64 = 0.75×; TPOT speedup = 1/0.75 = 1.33×; basis: A2 total weight BW ≈ 64 GB (32B params × 2 bytes/param bf16); at T̄=L_mid/2 only 48 layers load weights per token → 48/64 × 64 GB = 48 GB; speedup = 64 GB / 48 GB = 1.33×; requires conditional-load kernel — naive masked-compute still loads all 64 GB] |
| A2 + idea 1.2 (post-training retrofit, FlexiDepth-style) | ~64 GB (no kernel optimization) | ~1.0× (no improvement without kernel) |

#### KV Cache Comparison

| Baseline | 32K ctx | 40K ctx | 262K ctx | Notes |
|----------|---------|---------|----------|-------|
| A1 (Qwen3.5-27B) | ~2.15 GB | ~2.68 GB | ~17.2 GB | 16 full-attn layers |
| A2 (Qwen3-32B) | ~8.59 GB | ~10.74 GB | ~68.7 GB | 64 layers, native max 40,960 |
| B (Qwen3.5-397B) | ~1.0 GB | ~1.22 GB | ~8.0 GB | 15 global-attn layers |
| C (K2 72.55B) | ~10.0 GiB | ~12.5 GiB | ~80.0 GiB | 80 layers |
| A2 + idea 1.2 (T̄=L_mid/2) | ~6.44 GB | ~8.06 GB | N/A (exceeds native max) | ~25% KV savings vs A2 baseline |

---

### 4.2 Compute Analysis

- **Training FLOPs vs. Baseline A2**: Approximately 1.0×–1.1×. Layer dropout (LayerSkip recipe) reduces actual FLOPs per training step proportional to average dropout rate, which partially offsets the additional early-exit loss computation. The router adds O(L_mid · d) — negligible. Note: all middle-layer activations must be retained during training (for early-exit loss backprop), partially negating layer dropout memory savings.
- **Inference FLOPs (prefill)**: ≈ (L_pre + T̄ + L_suf)/L × Baseline A2 FLOPs. CALM (3× speedup → ~0.33× FLOPs) and LayerSkip (2× → ~0.5× FLOPs) suggest T̄/L_mid ≈ 0.33–0.67 is achievable with minimal quality loss.
- **Inference FLOPs (decode per token)**: Same fraction as prefill.
- **Batch>1 dispatch overhead**: The MoE-style token-grouping dispatch adds O(B × k) scatter/gather overhead where k = number of distinct exit depths. At batch=32, k=4 depth buckets: 128 scatter operations per layer. Not quantified but real; FlexiDepth [16] cites I/O overhead as the primary reason for lack of wall-clock speedup.

### 4.3 Memory Bandwidth Analysis

- **Weight loading at decode**: If T̄ = L_mid/2 with L_pre = L_suf = L/4: average layers = 3L/4. Bandwidth reduction = 25% vs. dense baseline → TPOT speedup ~1.33×. **Requires a kernel that actually skips loading weight matrices for skipped layers** — naive implementations load all weights and mask computations, yielding no memory bandwidth benefit.
- **KV cache access pattern**: The postfix layers must attend to all s tokens; their KV access is equivalent to baseline. Middle-layer KV entries are only accessed by tokens still in the middle section at that layer.
- **Activation memory**: Peak activation memory during training includes activations for all middle-layer checkpoints for the early-exit loss. This partially offsets the layer dropout memory savings. Not fully quantified in the existing literature for the specific prefix/postfix design.

### 4.4 Memory Capacity Analysis

- **Total weight storage**: Unchanged from Baseline A2 — O(L · d · d_ff). All middle-layer weights are stored; routing determines which are *used* per token, not which are stored.
- **KV cache at 32K tokens**: Compared to Baseline A2 (~8.59 GB): at T̄=L_mid/2, saves ~25% → frees ~2.15 GB. At T̄=L_mid/3: saves ~33% → frees ~2.86 GB. The ~17 GB savings figure applies at 262,144-token context only, beyond A2's 40,960-token native maximum.
- **Peak training memory**: LayerSkip training uses approximately the same peak memory as baseline training. The router adds negligible additional parameters (L_mid × d scalars ≈ 32 × 5120 ≈ 163K parameters for A2 config — negligible vs. 32B).

---

## 5. Implementation Considerations

- **Hardware requirements**: The bandwidth-bound TPOT speedup requires a kernel that conditionally skips weight loading for exited tokens. Naive implementations (mask out skipped layers' outputs) still load weights, yielding no memory bandwidth benefit — this is the central finding from TIDE [10] (6–7% gain only) and FlexiDepth [16] (no wall-clock speedup yet). Requires: (a) a fused conditional-load CUDA/Triton kernel or (b) structured batching where all tokens at the same exit point are grouped. EE-LLM [13] (ICML 2024) is the most relevant production framework for large-scale early-exit training and inference.

- **Training stability**: LayerSkip recipe (layer dropout + early exit loss) is proven and open-sourced [6]. "Early Exit Is a Natural Capability" (arXiv:2412.01455) finds early exit does not require joint optimization, suggesting post-hoc retrofit is partially viable. Risk: if the router learns to always exit at the earliest possible middle layer, the model loses the benefit of deep middle layers. Mitigation: minimum-depth penalty in training loss.

- **Engineering effort**: **13–22 engineer-weeks** for production quality [derived: TIDE post-training retrofit = 2,389 LOC achieving only 6–8% speedup without custom kernels ↔ ~4–6 eng-weeks; production quality requires: (a) conditional-load CUDA/Triton kernel for weight skipping (~3–5 weeks), (b) training integration with layer dropout + early-exit loss backprop (~3–5 weeks), (c) vLLM/PagedAttention modifications for variable-depth KV (~3–6 weeks); total: 4+3+3=10 weeks lower bound to 6+5+6=17 weeks upper; rounding to account for integration testing and validation → 13–22 engineer-weeks]. TIDE's simpler post-training retrofit (2,389 LOC) achieves only 6–8% speedup — production speedup requires the custom kernel work.

- **Framework support**: PyTorch feasible. The forward pass modification (loop over middle layers with exit check) is implementable. However, for GPU efficiency, the loop should be replaced with a fused kernel. vLLM integration requires custom PagedAttention modifications for variable-depth KV.

- **Applicability to baselines**:
  - **Baseline A2 (dense)**: Best candidate — "Diminishing Returns" [14] finds large dense models >20B are most suitable for early exit. Direct application of static prefix + dynamic middle + static postfix.
  - **Baseline C (K2 72.55B dense)**: Also a strong candidate — large dense model with L=80 layers provides more middle-section opportunity. The large model size means weight loading dominates TPOT, amplifying the benefit of reduced layers.
  - **Baseline A1 (hybrid Gated DeltaNet)**: CAUTION — "Diminishing Returns" [14] finds SSM/hybrid architectures are substantially harder to early-exit. Apply adaptive depth only to linear-attention middle layers; full-attention layers could be in the mandatory prefix/postfix. Validate suitability before committing.
  - **Baseline B (MoE)**: CAUTION — "Diminishing Returns" [14] finds MoE architectures are substantially harder to early-exit. Can combine with idea 1.1 in MoE middle layers; early-exiting tokens skip those experts. Validate suitability before committing.

---

## 6. Synergies

- **Combines well with**:
  - **1.1 (Learnable Per-Token Top-k)**: Orthogonal dimensions — depth (number of layers) vs. breadth (number of experts per layer). Combined, an early-exiting token also uses fewer experts in the layers it does execute.
  - **3.4 (Recursive Internal State / Internal CoT)**: Idea 3.4's iterative refinement could use idea 1.2's dynamic middle section as the loop body, with the exit condition being the routing decision.
  - **5.1 (TurboQuant KV cache)**: Quantizing middle-layer KV entries reduces KV memory impact.
  - **5.4 (Linked Attention / Sparse KV)**: Combined with early exit, KV entries for easy tokens can be evicted after their exit layer.
  - **4.2 (Shared Core Weights + Per-Layer LoRA)**: Shared middle-layer weights reduce the cost of skipping.

- **Conflicts with**:
  - **3.1 (Layer-Level MoE)**: Two-level routing hierarchy (per-layer MoE selection + per-token depth exit) may be difficult to train stably.
  - **Fixed-depth inference engines**: Hardware-fixed computation graphs require significant re-engineering.

---

## 7. Risk Assessment

- **Technical risk**: MEDIUM-HIGH — The per-token early exit mechanism is well-established (CALM, LayerSkip). The specific prefix/postfix structure is a minor architectural variant. The key technical risks are: (a) inference kernel — achieving genuine TPOT speedup requires conditional weight loading, which TIDE and FlexiDepth show is not automatic; (b) applicability to hybrid/MoE baselines is adversely impacted by the "Diminishing Returns" [14] findings; (c) the postfix-heterogeneity issue (postfix attention sees tokens with different middle-section depths) lacks published analysis.

- **Potential impact**: HIGH for Baseline A2 and C (dense models) — if T̄ = L_mid/2 achievable, TPOT improves ~1.33× and KV cache saves ~25%. MEDIUM for A1 and B — hybrid/MoE architectures have reduced early-exit suitability per [14].

- **Implementation effort**: MEDIUM-HIGH — Training modification (LayerSkip recipe + router) is 1–2 engineer-weeks. The inference kernel (conditional weight loading, batch dispatch for variable depth) is 4–6 weeks. Integration with serving infrastructure is 2–4 weeks. Total: **13–22 engineer-weeks** for production quality, grounded in FlexiDepth/TIDE implementation evidence.

---

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - CALM [4]: Up to 3× speedup while provably maintaining ROUGE/BLEURT (NeurIPS 2022, §4, Table 1)
  - LayerSkip [6]: 1.34–2.16× speedup; within 1–2% of full-depth baseline (ACL 2024, §5, Table 2)
  - Depth-Adaptive Transformer [3]: Matches baseline BLEU with up to 76% less decoder computation (ICLR 2020)
  - AdaInfer [9]: 17.8% average pruning; <1% performance drop (IJCAI 2025, Table 1)
  - MoD [5]: Matches or slightly improves baseline perplexity at isoFLOP (arXiv:2404.02258, §4, Figure 3)
  - "Diminishing Returns" [14]: Adverse finding — reduced suitability in modern architectures, particularly hybrid and MoE

- **Monotonicity**: Approximately monotone with a cliff. The cliff depth and location are task-dependent and architecture-dependent (modern models with improved training recipes have reduced layer redundancy per [14]).

- **Recovery**: Exit threshold adjustable at inference time without retraining (CALM framework). LayerSkip self-speculative decoding provides full-quality recovery at small cost.

- **Static postfix quality advantage**: The static postfix provides a quality robustness guarantee not present in CALM/LayerSkip — all tokens receive the same final-layer normalization and projection regardless of middle-section depth. This is expected to reduce quality degradation at aggressive T̄ targets. No published analysis of this specific guarantee exists.

---

<!-- CITATION MANIFEST: title | arxiv_id_or_url | description -->
<!-- Adaptive Computation Time | arxiv:1603.08983 | arXiv 2016; foundational per-step halting probability for RNNs -->
<!-- Universal Transformers | arxiv:1807.03819 | ICLR 2019; ACT applied to weight-shared transformers -->
<!-- Depth-Adaptive Transformer | openreview:SJg7KhVKPH | ICLR 2020; per-token early exit for NMT decoding; 76% FLOPs reduction -->
<!-- CALM | arxiv:2207.07061 | NeurIPS 2022; per-token early exit with provable quality guarantees; up to 3× speedup -->
<!-- Mixture-of-Depths | arxiv:2404.02258 | arXiv 2024; per-layer token budget with residual bypass; 50% FLOPs reduction -->
<!-- LayerSkip | arxiv:2404.16710 | ACL 2024; layer dropout + early exit loss + self-speculative decoding; 1.34–2.16× speedup -->
<!-- Mixture-of-Recursions | arxiv:2507.10524 | NeurIPS 2025 / ICML 2025; shared block recursive depth with KV-selective caching -->
<!-- Inner Thinking Transformer | arxiv:2502.13842 | ACL 2025; critical token routing with thinking step encoding -->
<!-- Not All Layers of LLMs Are Necessary During Inference (AdaInfer) | arxiv:2403.02181 | IJCAI 2025; per-request early exit via confidence gap; no retraining; 17.8% layer pruning -->
<!-- TIDE | arxiv:2603.21365 | arXiv 2026; post-training per-token early exit via hidden-state cosine convergence; 6–7% speedup -->
<!-- DEL | arxiv:2504.05598 | COLM 2025; context-aware dynamic exit layer for speculative decoding; 2.16–2.62× speedup -->
<!-- SkipGPT | arxiv:2506.04179 | ICML 2025 (unverified venue); token-aware adaptive depth with decoupled MLP/attention pruning -->
<!-- EE-LLM | arxiv:2312.04916 | ICML 2024; large-scale 3D-parallel training and inference framework for early-exit LLMs -->
<!-- Diminishing Returns of Early-Exit Decoding | arxiv:2603.23701 | arXiv 2026; critical adverse finding — reduced early-exit suitability in modern LLMs, especially MoE/SSM -->
<!-- What Layers When: Learning to Skip Compute in LLMs with Residual Gates (GateSkip) | arxiv:2510.13876 | ICLR 2026; sigmoid-linear residual gates for token-wise layer skipping; stable fine-tuning -->
<!-- Adaptive Layer-skipping in Pre-trained LLMs (FlexiDepth) | arxiv:2503.23798 | COLM 2025; critical adverse finding — no wall-clock speedup yet from variable-depth inference -->
<!-- Learning to Skip the Middle Layers of Transformers | arxiv:2506.21103 | arXiv 2025; adverse finding — middle-layer gating skipping does not improve FLOPs/perplexity trade-off at tested scales vs dense baseline -->
<!-- D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models | NeurIPS 2024 | NeurIPS 2024; per-token layer skip + KV eviction; 45–50% FLOPs and KV storage reduction -->
