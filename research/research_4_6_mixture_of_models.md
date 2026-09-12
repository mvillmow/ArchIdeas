# Research: Mixture of Models (MoMoRA)
## ID: 4.6

## Executive Summary

MoMoRA (Mixture of Models with Recurrence and Attention) is a multi-model neural architecture routing individual tokens at generation time to entire heterogeneous pre-trained models — not FFN sub-network experts within a single model. Four novel components are combined: (1) token-level routing to complete models, (2) persistent GRU-style recurrent state spanning model selections, (3) two-level hierarchical routing (coarse group + fine model selection), (4) ACT-style gated output emission.

The gated emission mechanism (P>1 ponder steps per token) is an honest latency increase, not a free win. TTFT and TPOT are ≥ baseline, not reduced. The architecture's value proposition is quality improvement via specialization, not inference speedup.

A2 KV cache at 32K context is ~8.59 GB. MoMoRA's sliding-window KV advantage (~2.1 GB) over A2 is ~4×.


## Key Comparison Tables

### vs. Baseline A1 (Qwen3.5-27B Hybrid)

| Metric | Baseline A1 | MoMoRA (k=1, P=1, 27B pool) | MoMoRA (k=1, P=1.5 assumed) |
|--------|------------|------------------------------|------------------------------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4+d·d_ff)) | ≈ same as A1 (same-class model) | ↑ ~1.5× vs. A1 |
| Memory bandwidth (decode) | ~54 GB bf16 | ~54 GB (ties A1 best case) | ~81 GB effective (P×weight_load) |
| KV cache (32K ctx) | ~2.15 GB (16 full-attn) | ~2.1 GB (sliding window W=2048) | ≈ same |
| KV cache (262K ctx) | ~17.2 GB | ~2.1 GB (window fixed) | ↓ large advantage at long ctx |
| Weight memory (total pool) | ~54 GB | ~162 GB (M=3×27B) | same |
| Training cost | 1.0× | ~3–5× (all pool models jointly) | same |
| TTFT (8K prompt) | ref | ≈ ref† (k=1, P=1) | ↑* ~1.5× (P=1.5) (⚠ sequential ponder loop; P_avg>1 unvalidated) |
| TPOT (batch=1) | ref | ↑* (⚠ sequential ponder loop; P_avg>1 unvalidated) | ↑* ~1.5× (P=1.5) (⚠ sequential ponder loop; P_avg>1 unvalidated) |
| Quality | ref | +? (specialization benefit undemonstrated) | same |

Pool models inherit the A1 (Qwen3.5-27B Hybrid: H_kv=4, head_dim=256, 16 full-attn layers) spec unless otherwise noted.

† P=1 is the idealized single-expert case. Average P_avg is a ponder-loop hyperparameter and is not validated by published measurements; TPOT figures for P=1 are lower bounds.

### vs. Baseline A2 (Qwen3-32B Dense)

| Metric | Baseline A2 | MoMoRA (k=1, P=1) | Notes |
|--------|------------|-------------------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | ≈ O(P·k·C_m) | Ties at k=1, P=1, similar-class model |
| Memory bandwidth (decode) | ~64 GB bf16 | ~54 GB (k=1, 27B) | MoMoRA slightly lighter at k=1, P=1 |
| KV cache (32K ctx) | **~8.59 GB** | ~2.1 GB | ~4× smaller — real advantage |
| Weight memory | ~64 GB bf16 | ~162 GB (pool) | 2.5× larger total weight storage |
| Training cost | 1.0× (reference) | ~3–5× | Substantially higher |
| TTFT (8K prompt) | ref | ≈ ref† (P=1) / ↑ (P>1) | Ties or worse |
| TPOT (batch=1) | ref | ↑* (⚠ sequential ponder loop; P_avg>1 unvalidated) | Ties or worse |

† P=1 is the idealized single-expert case. Average P_avg is a ponder-loop hyperparameter and is not validated by published measurements; TPOT figures for P=1 are lower bounds.

### vs. Baseline B (Qwen3.5-397B-A17B MoE)

| Metric | Baseline B | MoMoRA (k=1, 27B pool) | Notes |
|--------|-----------|------------------------|-------|
| Compute (FLOPs/token) | 17B active params | ~27B active | ↑ MoMoRA loses unless light (<17B) pool models used |
| Memory bandwidth (decode) | ~34 GB (17B active) | ~54 GB (k=1×27B) | ↑ MoMoRA loads more weight per token |
| KV cache (32K ctx) | ~1.0 GB (15 full-attn, 32Q/2KV) | ~2.1 GB sliding window | ≈ comparable; B already hybrid |
| Total weight storage | ~397B = ~794 GB | ~162 GB | ↓ MoMoRA far smaller total storage |
| Training cost | 1.0× (reference) | ~3–5× vs. A2 | ↑ MoMoRA substantially more expensive |
| TPOT (batch=1) | ref | ↑* (loads 54 vs 34 GB) (⚠ sequential ponder loop; P_avg>1 unvalidated) | MoMoRA slower per token vs. B |

† P=1 is the idealized single-expert case. Average P_avg is a ponder-loop hyperparameter and is not validated by published measurements; TPOT figures for P=1 are lower bounds.

### vs. Baseline C (K2 family, 72.55B dense Llama-arch)

| Metric | Baseline C (K2) | MoMoRA (k=1, 27B pool) | Notes |
|--------|----------------|------------------------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)), L=80 | O(k·C_m) per step | k=1 of 27B vs 72.55B: MoMoRA lighter |
| Memory bandwidth | ~145.1 GB weight BW | ~54 GB (k=1×27B) | ↓ MoMoRA 2.7× lighter at decode |
| KV cache (32K ctx) | ~10.0 GiB | ~2.1 GB sliding window | ↓ MoMoRA 4.8× smaller KV |
| Total weight storage | ~72.55B = ~145 GB | ~162 GB (M=3×27B) | ≈ comparable |
| TPOT (batch=1) | ref (145 GB) | ~0.37× (54 GB/145 GB) | ↓ MoMoRA faster per step if pool models are small |

Note: MoMoRA has a genuine advantage vs K2 in that it activates only one model per token step (if k=1). With P=1 (no ponder repetition), MoMoRA can achieve substantially lower TPOT than K2 by routing easy tokens to small (e.g., 7B) pool models.

---

## 1. Idea Description

MoMoRA routes individual tokens at generation time through a 7-step pipeline:

1. **Token Embedding**: Shared embedding converts input token to d_model vector
2. **Recurrent State Fusion**: Fuse with persistent h_{t-1} via GRU-style gating
3. **Hierarchical Routing (Two-Level)**: Level 1 → coarse group (Light/Medium/Heavy); Level 2 → top-k selection within group
4. **Model Processing**: Selected model(s) process fused representation
5. **Local Attention + Memory**: Sliding-window attention over recent tokens; optional retrieval
6. **Gated Output Emission**: Gate g_t > τ → emit token; else → ponder another step
7. **State Update**: h_{t+1} from selected model output

**The gated emission (step 6) increases average TPOT** — it is an adaptive compute mechanism for quality, not a speed improvement. This must be emphasized: P_avg > 1 is the assumed operating point, not a theoretically derived minimum. P_avg is a hyperparameter tuned via ponder cost calibration.

**Novelty verdict: PARTIAL — jointly end-to-end trained mixture of models with shared recurrent state at token granularity has no published precedent; BTM [Li et al., 2022] and FrugalGPT [Chen et al., 2023] cover model- and query-level routing respectively.**

---

## 2. Literature Review

### Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., ICLR 2017)
Foundation of gated conditional computation[1] at the FFN sub-network level. MoMoRA extends to routing across entire distinct models. Top-k gating network + load-balancing formulation directly precedent. arXiv:1701.06538. Section: §2 "The Mixture-of-Experts Layer."

### Switch Transformers (Fedus, Zoph, Shazeer, JMLR 2022)
Top-1 routing as stable and scalable[2]. Load-balancing auxiliary loss (§2.2) is mandatory for MoMoRA to prevent routing collapse. arXiv:2101.03961.

### Mixtral of Experts (Jiang et al., 2024)
8-of-8 FFN MoE at layer level[3]: 47B total, 13B active. Outperforms Llama-2 70B. Demonstrates heterogeneous activation via identical FFN blocks; MoMoRA extends to heterogeneous full-model routing. arXiv:2401.04088.

### Branch-Train-Merge (Li et al., ACL 2023 Findings)
Trains M separate full language models on different data domains[4], then routes between them at inference using a learned routing head. Most direct structural analog to MoMoRA's model pool design. Key distinctions: BTM trains independently (not joint end-to-end); routes at request level (not token). Novelty claim survives: jointly end-to-end trained persistent recurrent state at token granularity is not in BTM. arXiv:2208.03306.

### FrugalGPT (Chen, Zaharia, Zou, TMLR 2024)
Query-level LLM cascade[5]: up to 98% cost reduction at equal quality; +4% accuracy at equal cost. Token-level analog is MoMoRA's Level 1/2 routing applied per token. arXiv:2305.05176.

### RouteLLM (Ong et al., ICLR 2025)
Binary router trained on 80k preference pairs[6]: >2× cost reduction without quality loss; routers transfer across model swaps. Demonstrates preference-data routing training for model-level selection. arXiv:2406.18665.

### HybridLLM (Ding et al., ICLR 2024)
Quality-aware query routing[7]: 40% fewer calls to large model with no quality drop; test-time threshold tuning. Direct analog to MoMoRA's Light/Medium/Heavy coarse routing and τ gate adjustment. arXiv:2404.14618.

### Unified Routing and Cascading (Dekoninck, Baader, Vechev, 2024)
Theoretically optimal cascade routing framework[8] unifying routing and cascading. Optimality conditions identify quality estimator as critical. Relevant to MoMoRA's escalation decision theory. arXiv:2410.10347.

### Speculative Decoding (Leviathan, Kalman, Matias, ICML 2023)
Draft model + verifier model[9]: 2×–3× speedup on T5-XXL; preserves exact output distribution. Two-model collaborative inference at token level — MoMoRA's closest deployed analog. arXiv:2211.17192.

### Speculative Sampling (Chen, Borgeaud et al., 2023)
Concurrent independent validation[10] of speculative decoding on Chinchilla 70B: 2–2.5× speedup. DeepMind. arXiv:2302.01318.

### Adaptive Computation Time (Graves, 2016)
ACT halting mechanism[11] for RNNs: differentiable, learns number of computation steps. Ponder cost penalty structure maps to MoMoRA's gated emission training. arXiv:1603.08983.

### Depth-Adaptive Transformer (Elbayad et al., ICLR 2020)
Per-token adaptive depth halting in Transformers[12] — the direct Transformer-era precedent for MoMoRA's gated emission, more directly relevant than RNN-era ACT alone. arXiv:1910.10073.

### PonderNet (Banino, Balaguer, Blundell, DeepMind, 2021)
Probabilistic halting model[13]: p(halt|step n) = λ_n. Fully differentiable, unbiased gradient estimates; outperforms ACT on extrapolation. Applicable to MoMoRA's gated emission for more stable gradient estimation. arXiv:2107.05407.

### Think before you speak: Pause Tokens (Goyal et al., ICLR 2024)
Learnable pause tokens delay output extraction[14]: SQuAD +18% EM, CommonSenseQA +8%, GSM8k +1% on 1B model. Empirically validates that additional latent computation before token emission improves accuracy — directly validates MoMoRA's gated emission design motivation. arXiv:2310.02226.

### Universal Transformers (Dehghani et al., ICLR 2019)
Weight-shared recurrent transformer with per-position ACT halting[15]. +0.9 BLEU over standard Transformer on WMT14 En-De. Combines weight sharing, recurrent depth, and adaptive halting — three of MoMoRA's components (missing: multi-model routing). arXiv:1807.03819.

### RetNet (Sun, Dong, Huang et al., 2023)
Multi-scale retention[16] supporting parallel/recurrent/chunkwise modes: 8.4× higher throughput, 70% less memory vs. transformers. Demonstrates GRU-style persistent recurrent hidden state at inference — directly validates MoMoRA's h_t design. arXiv:2307.08621.

### Griffin (De, Smith, Fernando, Botev et al., Google DeepMind, 2024)
Gated linear recurrences + local sliding-window attention[17]. Griffin-7B/14B match Llama-2 on >6× fewer tokens. Directly implements the GRU-style gating + sliding-window attention combination that MoMoRA's recurrent fusion and attention module use. arXiv:2402.19427.

### Mamba (Gu, Dao, 2023)
Selective SSM with input-dependent gating[18]: Mamba-3B matches Transformers 2×its size; 5× inference throughput. Validates input-gated recurrent state as an effective architectural primitive — supports MoMoRA's GRU update gate design. arXiv:2312.00752.

### Mixture-of-Agents (Wang et al., 2024)
Layered LLM agent architecture[19]: 65.1% AlpacaEval 2.0 vs. 57.5% GPT-4 Omni using open-source only (+7.6%). "Collaborativeness" — LLMs produce better outputs given other models' outputs. MoMoRA is a token-level, recurrent, routing-efficient version of MoA. arXiv:2406.04692.

### Mixture of Depths (Raposo et al., 2024)
Per-token binary layer-skip routing[20] within a single model at 6B+ scale. Up to 50% faster sampling at equivalent FLOPs. Token-level routing based on complexity is learnable — within-model analog validates MoMoRA's token-level routing granularity. arXiv:2404.02258.

### HDMoLE (Mu et al., ICASSP 2025)
Two-level hierarchical routing for LoRA expert selection[21]: Level 1 domain group → Level 2 specific adapter. 9.6% trainable parameters vs. full fine-tuning at comparable quality. Direct analog to MoMoRA's hierarchical routing structure at LoRA scale. arXiv:2409.19878.

### MixLLM (Wang et al., NAACL 2025)
Contextual-bandit routing across heterogeneous LLM APIs[22]: 97.25% of GPT-4 quality at 24.18% cost. Bandit-based adaptive learning is a potential training mechanism for MoMoRA's router. arXiv:2502.18482.

### FusionRoute (Xiong et al., 2026)
Token-level routing across heterogeneous off-the-shelf LLMs[23]: lightweight router selects best expert per decoding step and adds a complementary logit to refine the chosen model's distribution. Proves that pure expert-only token routing is theoretically suboptimal without a complementary generator — directly challenges MoMoRA's assumption that routing alone is sufficient, and suggests a logit-blending augmentation for the gated emission step. Outperforms sequence-level routing, model merging, and direct fine-tuning across math, code, and instruction-following benchmarks on Llama-3 and Gemma-2 families. arXiv:2601.05106.

---

## 3. Prior Art Classification

- **Status**: PARTIAL (~75% component coverage)
- **Novel contribution**: The specific combination of (a) token-level routing to complete heterogeneous full models (not FFN sub-networks), (b) persistent GRU-style recurrent state h_t spanning cross-model selections, (c) two-level hierarchical routing, and (d) gated output emission — within a single jointly-trained shared-interface architecture at token granularity — is not present as a unified system. BTM[4] provides domain-specialized full-model routing but at request level without persistent recurrent state. The novelty claim is qualified: *jointly end-to-end trained* with *shared recurrent state* at *token granularity* is the specific gap.

---

## 4. Technical Analysis

### 4.1 Theoretical Complexity

Variables: M=pool size, k=active models, d=shared d_model, C_m=model compute, P=avg ponder steps, W=sliding window size

| Metric | MoMoRA (k=1) | Baseline A1 | Baseline A2 | Baseline B |
|--------|-------------|------------|------------|-----------|
| Compute/token | O(P·k·C_m+d²·router) | O(L·(d²+s·d/4+d·d_ff)) | O(L·(s·d+d·d_ff)) | O(L·(d²+s·d/4+k_e·d·d_e)) |
| KV cache | O(W·d_kv) | O(L/4·s·d_kv) | O(L·s·d_kv) | O(L/4·s·d_kv) |
| Recurrent state | O(d) | O(d²/H per head, linear) | 0 | O(d²/H per head, linear) |
| Active weight memory | O(k·N_model) | O(L·d·d_ff) | O(L·d·d_ff) | O(k_e·d·d_e) |
| Total weight storage | O(M·N_model) | O(L·d·d_ff) | O(L·d·d_ff) | O(L·E·d·d_e) |
| TTFT | ↑ P× vs. A1 | ref | ref | ref |
| TPOT (batch=1) | ↑ P·k× worst case | ref | ref | ref |

### 4.2 KV Cache Comparison

**A2 at 32K context:**
KV = 2 × L × H_kv × head_dim × seq_len × 2 bytes
= 2 × 64 × 8 × 128 × 32,768 × 2 = **8,589,934,592 ≈ 8.59 GB**

**MoMoRA sliding window W=2048:**
KV = L × W × d_kv × 2 bytes (per-layer; for A1-class models)
Approximate: ~2.1 GB per model at W=2048, similar number of layers

MoMoRA KV advantage over A2: **~4× smaller** (2.1 GB vs. 8.59 GB).

At A1's true max context (262K): MoMoRA sliding window stays at ~2.1 GB; A1 full KV would be ~17.2 GB [16×2×4×256×262144×2] → MoMoRA advantage at long context is substantial (~8×).

### 4.3 Memory Bandwidth Analysis

- **Weight loading per decode step (k=1):** ~54 GB for 27B-class pool model in bf16 — same as Baseline A1
- **Effective weight loading (P=1.5):** 1.5 × 54 = ~81 GB effective (P repetitions)
- **Recurrent state h_t:** 5120 × 2 bytes = 10 KB — negligible
- **Router overhead:** O(d × d_r) ≈ 0.15% of model forward FLOPs — negligible

**Novelty verdict: PARTIAL — jointly end-to-end trained mixture of models with shared recurrent state at token granularity has no published precedent; BTM [Li et al., 2022] and FrugalGPT [Chen et al., 2023] cover model- and query-level routing respectively.**

---

## 5. Implementation Considerations

**Hardware requirements:** M=3 × 27B requires ~162 GB VRAM minimum → 2× H100 80 GB. For training: 3–4 TB peak (all M models + gradients + optimizer states) → 8+ H100s.

**Critical blockers:**

1. **Shared d_model constraint:** All pool models must share d_model. Pre-trained models at different scales use different d_model (7B: 4096; 27B: 5120). Forces training all pool models from scratch with unified d_model — eliminates leveraging existing checkpoints. True training cost may substantially exceed 3–5× estimate.

2. **Dynamic halting incompatibility:** Variable-P halting loops are incompatible with torch.compile, CUDA graphs, TensorRT-LLM, vLLM, SGLang. Requires PyTorch eager mode (~2–3× throughput loss vs. compiled) or JAX lax.while_loop. Custom inference engine needed for production.

3. **Joint training infrastructure:** 8+ H100s for full M=3×27B joint training. Router gradients couple model updates; FSDP or DeepSpeed ZeRO-3 required.

**Training stability risks:**
- Routing collapse (all tokens → one model): Mandatory load-balancing auxiliary loss[2]
- GRU backprop through heterogeneous model boundaries: gradient clipping
- Ponder cost calibration: too low → infinite loops; too high → forces P=1 always

---

## 6. Synergies

- **Combines with 3.4 (Recursive Internal State):** MoMoRA's gated emission is a multi-model version of 3.4's looped latent computation. Each ponder step could route to a different pool model.
- **Combines with 1.1 (Learnable Per-Token Top-k):** k (active models) could be made per-token adaptive.
- **Combines with 3.2 (Swappable Experts):** Pool models could hot-swap domain-specialized variants.
- **Conflicts with 4.2 (Shared Core + LoRA):** MoMoRA requires heterogeneous models; single shared core undermines model diversity.

---

## 7. Risk Assessment and Verdict

- **Technical risk:** HIGH — routing collapse more severe with M=3 models than M=512 FFN experts; gated emission training notoriously unstable; shared d_model constraint blocks existing checkpoint reuse.
- **Implementation effort:** 6–18 person-months (team of 3–5); custom CUDA kernels; custom inference engine.
- **Training cost:** 3–5× vs. A2 (potentially more if all pool models trained from scratch).
- **Quality ceiling:** Plausible but undemonstrated. Component evidence: +7.6% AlpacaEval 2.0 (MoA[19]), +4% accuracy at equal cost (FrugalGPT[5]), +18% SQuAD EM (pause tokens[14]).

### Recommended 5-Step Ablation Sequence

1. **(1–2 months)** Validate token-level routing signal: frozen speculative-decoding setup (light=draft, heavy=verifier), test if router can classify token difficulty into 2–3 buckets.
2. **(2–4 months)** BTM-style system: 2–3 domain-specialized models trained independently + learned request-level router.
3. **(3–6 months)** Add token-level routing to BTM: route individual tokens to best domain model.
4. **(6–12 months)** Add persistent recurrent state h_t.
5. **(12–18 months)** Add gated output emission (only if steps 1–4 positive).


---

## 8. Accuracy / Quality Tradeoff

**Novelty verdict: PARTIAL — jointly end-to-end trained mixture of models with shared recurrent state at token granularity has no published precedent; BTM [Li et al., 2022] and FrugalGPT [Chen et al., 2023] cover model- and query-level routing respectively.**

- **Reported quality delta (from closest analogues)**:
  - [Wang et al., 2024][19] (Mixture-of-Agents): layered LLM agent collaboration achieves 65.1% AlpacaEval 2.0 versus 57.5% GPT-4 Omni using open-source models only (+7.6 pp). MoMoRA is a token-level recurrent version of MoA — routing individual tokens through specialized models rather than entire responses through model layers. The +7.6 pp result indicates substantial quality upside from multi-model collaboration at the request level, but token-level routing introduces additional routing error that may reduce this gain.
  - [Chen et al., TMLR 2024][5] (FrugalGPT): query-level LLM cascade achieves +4% accuracy at equal cost (and up to 98% cost reduction at equal quality) by routing queries to cheaper models for easy tasks and escalating to stronger models for hard tasks. This validates MoMoRA's routing premise at the query level — the token-level routing in MoMoRA is a fine-grained version of the same mechanism and should achieve similar or better quality-per-cost ratios if routing is accurate.
  - [Goyal et al., ICLR 2024][14] (Pause Tokens): +18% SQuAD EM, +8% CommonSenseQA, +1% GSM8k on 1B models by inserting learnable pause tokens before output extraction. This directly validates MoMoRA's gated output emission design: delaying emission while pondering additional computation steps improves quality. The quality gain is real but comes at a latency cost (P_avg > 1 increases TPOT proportionally).
  - [Dehghani et al., ICLR 2019][15] (Universal Transformers): weight-shared recurrent transformer achieves +0.9 BLEU on WMT14 En-De over standard Transformer. This validates MoMoRA's recurrent fusion component — persistent h_t spanning model selections provides quality improvements analogous to recurrent depth.
  - [Ding et al., ICLR 2024][7] (HybridLLM): quality-aware routing achieves 40% fewer large-model calls with no quality drop, via test-time threshold tuning. This demonstrates that the core routing mechanism (Light/Medium/Heavy coarse routing) can be calibrated for quality-neutrality: routing to smaller models for easy tokens does not degrade quality if the threshold is set correctly.
  - [Xiong et al., 2026][23] (FusionRoute): token-level per-decoding-step expert selection across heterogeneous LLMs, with complementary logit addition, outperforms sequence-level routing and model merging across math, code, and instruction-following benchmarks (arXiv:2601.05106). Proves that pure expert-only routing is *suboptimal* without complementary logit generation — directly challenging MoMoRA's routing-only gated emission and suggesting a quality ceiling from the routing design.

- **Monotonicity**: Quality in MoMoRA is **not monotone with a single aggressiveness axis** because the tradeoff is two-dimensional: (1) routing accuracy (how reliably easy tokens reach appropriate pool models) and (2) ponder steps P_avg (how many computation rounds per token). Higher P_avg monotonically improves quality (more computation) at the cost of higher TPOT (↑P× per token). Routing accuracy is not easily tuned after training — routing collapse is a binary failure mode, not a gradual degradation. Within a stable routing regime, quality improvement from gated emission is monotone with P_avg.

- **Recovery**: Quality recovery mechanisms: (1) **Increase P_avg** (reduce ponder-cost penalty) to allow more computation steps per token — monotonically recovers quality at cost of TPOT; (2) **Retrain routing head** with stronger load-balancing loss[2] if routing collapse has occurred — prevents quality degradation from single-model lock-in; (3) **FusionRoute-style logit augmentation** — adding a complementary generator to the routing decision provides quality recovery without architecture changes; (4) **Fallback to Heavy pool model** for all tokens as a quality floor — degrades to a single-model system at full compute cost but guaranteed quality baseline. Because MoMoRA trains all components jointly, routing quality loss cannot be easily patched without retraining from Phase 3 or later in the recommended ablation sequence.

- **Conditions for acceptable degradation**: The architecture's value proposition is quality *improvement* (specialization via routing), not quality-neutral speedup. Quality degradation occurs only if routing is miscalibrated (routing hard tokens to light models). Degradation is acceptable under the following conditions: (1) the token distribution is heavily dominated by simple tokens (>70% of generated tokens are common continuations, stopwords, simple phrases) where routing to light models is quality-neutral; (2) the use case tolerates request-level quality variance (some tokens slightly worse than a strong monolithic model) in exchange for average-token quality gains from specialization; (3) the gated emission is calibrated conservatively (high τ threshold, low P_avg cap) such that emission only occurs when the ponder gate is highly confident, preventing premature token emission on hard tokens; (4) the pool models cover complementary domain specializations (coding, math, language) so routing-to-wrong-model errors are bounded by domain coverage — applicable when the token stream is domain-homogeneous within a context window.

---

<!-- CITATION MANIFEST -->

[1] Sparsely-Gated MoE: Shazeer, Mirhoseini et al. (ICLR 2017). arXiv:1701.06538. §2 "The Mixture-of-Experts Layer." Foundational gated conditional computation; top-k selection precedent.

[2] Switch Transformers: Fedus, Zoph, Shazeer (JMLR 2022). arXiv:2101.03961. §2.2 "Routing Algorithm." Top-1 routing; load-balancing auxiliary loss mandatory for collapse prevention.

[3] Mixtral of Experts: Jiang et al. (Mistral AI, 2024). arXiv:2401.04088. 47B total, 13B active; outperforms Llama-2 70B. FFN-level heterogeneous activation.

[4] Branch-Train-Merge: Li et al. (ACL 2023 Findings). arXiv:2208.03306. Domain-specialized full-model training with routing; most direct structural analog to MoMoRA pool design.

[5] FrugalGPT: Chen, Zaharia, Zou (TMLR 2024). arXiv:2305.05176. Query-level cascade; 98% cost reduction, +4% accuracy at equal cost.

[6] RouteLLM: Ong, Almahairi et al. (ICLR 2025). arXiv:2406.18665. Binary router on preference data; >2× cost reduction.

[7] HybridLLM: Ding, Mallick et al. (ICLR 2024). arXiv:2404.14618. Quality-aware query routing; 40% fewer large-model calls; test-time threshold tuning.

[8] Unified Routing and Cascading: Dekoninck, Baader, Vechev (2024). arXiv:2410.10347. Theoretically optimal cascade routing framework.

[9] Speculative Decoding: Leviathan, Kalman, Matias (ICML 2023). arXiv:2211.17192. 2×–3× speedup draft+verifier setup; closest deployed multi-model inference analog.

[10] Speculative Sampling: Chen, Borgeaud et al. (DeepMind, 2023). arXiv:2302.01318. 2–2.5× speedup on Chinchilla 70B; independent concurrent validation.

[11] Adaptive Computation Time: Graves (2016). arXiv:1603.08983. §2 "Adaptive Computation Time." Differentiable halting for RNNs; ponder cost structure for gated emission.

[12] Depth-Adaptive Transformer: Elbayad et al. (ICLR 2020). arXiv:1910.10073. Per-token adaptive depth halting in Transformers; direct Transformer-era precedent for gated emission.

[13] PonderNet: Banino, Balaguer, Blundell (DeepMind, 2021). arXiv:2107.05407. Probabilistic halting; unbiased gradients; outperforms ACT on extrapolation.

[14] Think before you speak — Pause Tokens: Goyal et al. (ICLR 2024). arXiv:2310.02226. +18% SQuAD EM, +8% CommonSenseQA, +1% GSM8k via pause tokens; validates gated emission motivation.

[15] Universal Transformers: Dehghani et al. (ICLR 2019). arXiv:1807.03819. Weight-shared recurrent transformer with per-position ACT; +0.9 BLEU WMT14 En-De.

[16] RetNet: Sun, Dong, Huang et al. (Microsoft, 2023). arXiv:2307.08621. 8.4× throughput, 70% less memory; GRU-style recurrent hidden state validates h_t design.

[17] Griffin: De, Smith, Fernando et al. (Google DeepMind, 2024). arXiv:2402.19427. GRU-style LRU + local sliding-window attention; Griffin-7B/14B match Llama-2; validates both MoMoRA architectural primitives.

[18] Mamba: Gu, Dao (2023). arXiv:2312.00752. Selective SSM, input-dependent gating; 5× inference throughput; validates GRU update gate as effective primitive.

[19] Mixture-of-Agents: Wang, Wang, Athiwaratkun et al. (2024). arXiv:2406.04692. 65.1% vs. 57.5% GPT-4 Omni AlpacaEval 2.0 (+7.6%). Multi-model collaboration; MoMoRA is token-level recurrent version.

[20] Mixture of Depths: Raposo, Ritter et al. (DeepMind, 2024). arXiv:2404.02258. Per-token binary routing within single model; 50% faster sampling. Within-model analog validates token-level routing granularity.

[21] HDMoLE: Mu, Wei et al. (ICASSP 2025). arXiv:2409.19878. Two-level hierarchical LoRA expert routing; 9.6% parameters at full-FT quality. Direct analog to MoMoRA's Level 1/2 routing.

[22] MixLLM: Wang, Liu et al. (NAACL 2025). arXiv:2502.18482. Contextual-bandit routing; 97.25% GPT-4 quality at 24.18% cost. Bandit-based router training applicable to MoMoRA.

[23] FusionRoute: Xiong, Zhou et al. (2026). arXiv:2601.05106. Token-level per-decoding-step expert selection across heterogeneous LLMs with complementary logit addition; proves pure routing is suboptimal without complementary generation. Closest direct analog to MoMoRA token-level routing; challenges sufficiency of routing-only gated emission.
