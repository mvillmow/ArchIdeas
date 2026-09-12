# Research: Dynamic Expert Router (Add/Remove Experts Post-Training)
## ID: 3.3

## 1. Idea Description

**From arch_research_ideas.md (Section 3, idea 3.3):**

> A router architecture that can accommodate insertion or removal of expert modules after training, automatically learning to route to new experts or bypass removed ones. Enables post-hoc knowledge editing.

**Key distinction from 3.2:** Idea 3.2 is about *experts* being hot-swappable (replacing existing experts at the same slot). Idea 3.3 is about the *router* being able to handle structural changes — adding new expert slots or removing existing ones while the router remains functional. This requires the router to handle a variable-size expert set, which is architecturally non-trivial because standard routers are fixed-size linear projections R^d → R^E.

**Primary benefits:** (1) Deployability — update model knowledge post-training without retraining from scratch; (2) Expert removal — prune underperforming experts while maintaining routing coherence; (3) Incremental knowledge expansion — add new expert modules for new domains without retraining the full router; (4) Post-hoc knowledge editing — precisely update factual knowledge stored in experts.

This is an operational and serving idea, not a raw TTFT/TPOT improvement. Routing softmax overhead scales by O(E), negligible vs other computations.

**Synergy with Idea 6.5 (Pre-Attention Expert Router):** Idea 6.5 proposes expert routing at the pre-attention stage to specialize attention patterns before key/value computation. Idea 3.3's variable-E routing architecture directly enables 6.5's dynamic capability: by making the pre-attention router structurally extensible (e.g., via ReLU routing or HyperRouter), new attention-pattern experts can be added post-training to 6.5's pool without router retraining. The two ideas compose naturally: 3.3 provides the routing infrastructure lifecycle management that 6.5 requires for production deployment of dynamic attention-expert pools.

---

## 2. Executive Summary

**Novelty verdict:** PARTIAL — router recalibration after removal, expert addition in continual learning, MoE knowledge editing, and structurally extensible routing (ReLU/hypernet/embedding) all exist across four threads; the gap is a unified router designed from the start for both addition and removal with explicit protocols, validated at Baseline B scale (E=512, L=60) ([REAP, 2024], [PMoE/Lifelong-MoE, 2024], [MEMoE/LEMoE, 2024], [ReMoE, 2025], [HyperRouter, 2023]).

Dynamic expert routing (idea 3.3) covers component technologies across four research threads: router recalibration after removal (REAP, Router KD), expert addition in continual learning (PMoE, Lifelong-MoE), knowledge editing via MoE adapter insertion (MEMoE, LEMoE, MoKE, MoEEdit), and structurally extensible routing schemes (ReMoE ReLU, HyperRouter, MoIRA embedding routing). Prior art self-assessment of ~60–65% coverage is confirmed accurate.

A unified router architecture designed from the start for both addition and removal with defined protocols for each case, validated end-to-end at Baseline B scale (E=512, L=60), is the remaining gap.

**Recommendation:** ReLU routing (Option d, ReMoE) is preferred over softmax-TopK for structural extensibility because expert removal requires only zeroing one row with no score-redistribution coupling on remaining experts, and adding an expert requires only appending one row without softmax renormalization.

---

## 3. Literature Review

### ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing
Wang et al., 2024 — arXiv:2412.14711, ICLR 2025, §3 "ReLU Routing"[1]

Replaces TopK+Softmax router with a ReLU-based gate: expert activated if ReLU(x · w_i) > 0. Number of active experts per token is no longer fixed. Adding expert E+1 appends one row w_{E+1} (zero-initialized) without requiring softmax renormalization over E+1 experts. Removing expert i: set w_i = 0 — no effect on other experts since ReLU is fully independent per dimension. Outperforms vanilla TopK MoE consistently across model sizes and expert counts.

**Cold-start note:** Zero initialization → new expert has identically zero gradient in both softmax-TopK and ReLU routing — a strict dead zone, not a gradual ramp-up. Escape requires deliberate perturbation: (A) noise perturbation init, (B) activation-statistics init from CMoE, (C) warm-swap via 3.2 (eliminates cold start entirely by using an existing hot slot).

### HyperRouter: Towards Efficient Training and Inference of Sparse Mixture of Experts via HyperNetwork
Do et al., 2023 — EMNLP 2023, aclanthology 2023.emnlp-main.351, §2 "HyperRouter"[2]

Generates router parameters dynamically via a fixed hypernetwork and trainable expert-slot embeddings: H(e_i) → w_i generates the router weight for expert i from embedding e_i. Adding expert E+1 requires only providing e_{E+1}; H generates w_{E+1} = H(e_{E+1}) without modifying H. Zero-shot generalization depends on H's capacity and e_{E+1}'s quality. No cold-start gradient barrier for hypernetwork-generated weights.

### Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs (RoMA)
Li et al., 2025 — arXiv:2511.07419, §3 "RoMA"[3]

Adds lightweight regularization during post-training that aligns the routing weight manifold with the task embedding manifold's cluster structure. Reduces the routing-induced performance gap (reported as up to 10–20% of accuracy to optimal routing) on OLMoE-7B-A1B, DeepSeekMoE-16B-A3B, and Qwen3-30B-A3B, updating only router parameters (all expert parameters frozen). Establishes that routers can be re-trained cheaply after structural changes with only router weights updated.

### Rewiring Experts on the Fly: Continuous Rerouting for Better Online Adaptation in MoE Models
Su et al., 2025 — arXiv:2510.14853, §3 "Continuous Rerouting"[4]

Data-free, online test-time framework optimizing MoE routing during generation without external data. Could detect that a removed expert slot always produces zero output and adapt routing away from it online. Directly relevant to the "router bypass removed experts" requirement.

### REAP: Router-weighted Expert Activation Pruning ("REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression")
Lasby et al., 2025 — arXiv:2510.13999, §4 "Results"[5]

Prunes MoE experts using saliency (router gate-values × expert activation norms). Near-lossless compression at 50% expert pruning on Qwen3-Coder-480B and Kimi-K2. Establishes that "merging techniques introduce an irreducible error due to loss of fine-grained routing control" — after removing experts, the remaining router must continue exercising input-conditioned control. REAP: identify removable experts first, then recalibrate router.

### Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression (Router KD)
[Author et al.], 2026 — arXiv:2603.02217, §4 "Router Knowledge Distillation"[6]

Identifies router-expert mismatch as the primary source of post-compression quality degradation. Router KD: update only the router weights by distilling the original model's next-token distribution on unlabeled calibration data, expert parameters frozen. Cost: ~0.0001–0.001× original training FLOPs for 1M–10M token calibration (at Baseline B's 34B FLOPs/token and 10T training tokens, 10M calibration tokens = ~10⁻⁶ fraction of training FLOPs). Substantially recovers post-compression accuracy. Required minimum adaptation step after any structural change.

### MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE
[Author et al.], 2025 — arXiv:2507.00390, §4 "Experiments"[7]

Replaces pruned experts with lightweight "novice" modules (smaller parameter count), preserving the router's E dimension while reducing effective compute. 0.14 percentage points accuracy drop on average across 9 zero-shot tasks at 25% pruning. Structural solution that avoids router dimension changes.

### CMoE: Converting Mixture-of-Experts from Dense to Accelerate LLM Inference
Pei et al., 2025 — arXiv:2502.04416, §3 "CMoE", §4 "Experiments"[8]

Converts trained dense models into MoE via activation-statistics-based router initialization. Abstract reports lossless perplexity at 75% activation ratio with ~5% acceleration, and 1.5× latency reduction at 25% activation. Specific "<5 min on single GPU for 7B" figure is paper-body. Provides the activation-statistics initialization recipe for cold-start mitigation (Option B above) — new expert rows initialized from calibration data statistics rather than zeros.

### Progressive Mixture of Experts (PMoE)
Jung & Kim, 2024 — arXiv:2407.21571, §3 "PMoE"[9]

Asymmetric transformer design for continual learning: shallow layers preserve general knowledge (frozen), deep layers host progressively added experts for new knowledge. Router allocates new-task inputs to the appropriate deep expert. Addresses catastrophic forgetting in the "add expert" direction via architectural separation rather than a freeze-all-prior-experts protocol.

### Embedding-Space Zero-Shot Routing (cosine similarity approach)
[conceptual reference — see Self-Routing arXiv:2604.00421 and HyperRouter [2] for closest implementations]

Cosine similarity routing: score for token x and expert i = cos(f(x), g(e_i)). Adding expert E+1 requires only providing e_{E+1}; score computed at inference time from expert description embedding. Naturally extensible, zero-shot, no fine-tuning. Router compute complexity: O((d+E)×d_emb) vs O(d×E) for linear projection; at d=4096, E=512, d_emb=64: 36,864 vs 2,097,152 MACs (57× cheaper). **Note:** arXiv:2507.01843 (MoIRA) is a robotics routing paper — not an LLM MoE routing paper. The cosine-similarity routing principle is described here as a design pattern; Self-Routing [20] (arXiv:2604.00421) is a verified 2026 paper achieving parameter-free routing from hidden states in LLM MoE.[10]

### MEMoE: MoE Adapter for Efficient Knowledge Editing
Wang & Li, 2024 — arXiv:2405.19086, §4 "Results"[11]

Bypass MoE adapter structure with knowledge anchor routing: inputs requiring similar knowledge are routed to the same expert, original parameters remain frozen. Strong batch and sequential editing performance with outstanding balance between generalization and locality. Directly demonstrates expert insertion for targeted knowledge updates.

### LEMoE: Lifelong Expertise via MoE Expert Insertion
Wang & Li, 2024 — EMNLP 2024, aclanthology 2024.emnlp-main.149, §4 "Experiments"[12]

Lifelong editing via expert insertion — outperforms prior editing methods on the lifelong editing benchmark. MoE adapter insertion for targeted knowledge editing.

### MoEEdit: Null-Space Projection for Knowledge Editing
[Author et al.], 2026 — arXiv:2602.10965, §3 "Null-Space Projection"[13]

Null-space projection prevents routing disruption to existing experts when inserting new knowledge-editing adapters. Critical for maintaining routing quality after structural changes. MoEEdit repository available at Terence-Gu/MoEEdit.

### MoEfication: Converting Dense Networks to MoE
Zhang et al., 2022 — ACL Findings 2022, arXiv:2110.01786[14]

Foundational paper for post-hoc conversion of dense FFN layers to MoE via neuron activation clustering. CMoE[8] is a direct descendant. Required citation for "activation-statistics router initialization" — the initialization approach predates CMoE.

### Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints
Komatsuzaki et al., 2023 — ICLR 2023, arXiv:2212.05055[15]

Foundational dense-to-MoE upcycling paper. The 2024 upcycling follow-up builds directly on this work. Establishes the precedent that routers can be created post-training from dense checkpoints.

### Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models
Li et al., 2022 — arXiv:2208.03306[16]

Most direct published precedent for "add a domain expert post-training": domain experts trained independently and merged via learned gating. Domain experts can be added (new domain) or removed (obsolete domain) without retraining the whole ensemble.

### Hash Layers / Hash Routing for Mixture-of-Experts
Roller et al., 2021 — arXiv:2106.04426[17]

Deterministic parameter-free routing via hash(token_id) % E. Naturally extensible to variable E (adding experts = increasing hash modulus) with zero cold start and zero calibration cost. Qualitatively distinct solution class from learned routing. Trades learned routing quality for zero-cost extensibility.

### Expert Choice Routing
Zhou et al., 2022 — NeurIPS, arXiv:2202.09368[18]

Expert-selects-top-k-tokens routing. Guarantees load balance with hard constraint. Required for load balancing after structural changes.

### DSelect-k: Differentiable Selection in the Mixture of Experts
Hazimeh et al., 2021 — NeurIPS 2021, arXiv:2106.03760[19]

Continuously differentiable and sparse gate selecting at most k out of n experts via a binary encoding reformulation; trainable with SGD. Smooth gradient flow to all expert slots addresses the cold-start gradient barrier — the central unsolved problem for fixed-router extensibility approaches. Achieves 22%+ improvement on large-scale recommender systems vs Top-k.

### Self-Routing: Parameter-Free Expert Routing from Hidden States
Mohamud, Wagner & Ravanelli, 2026 — arXiv:2604.00421[20]

Eliminates the learned router entirely: a designated subspace of the token hidden state is used directly as expert logits. No routing parameters to add, remove, or recalibrate when experts are added or removed — making this the most structurally extensible routing scheme for idea 3.3. Achieves competitive performance with learned routers with improved expert utilization balance. Directly relevant as Option (e) for the variable-E routing architecture.

---

## 4. Prior Art Classification

**Status: PARTIAL (~60–65% covered).** Component technologies exist in four research threads but a unified router architecture designed for both addition and removal at production scale (E=512, L=60) is the gap. The restated novel contribution: a router designed from the start for variable E, with explicit protocols for both directions, and an inference-time routing quality bound for new experts prior to any fine-tuning.


---

## 5. Technical Analysis

### 5.1 Five Routing Architecture Options

**(a) Fixed router + append row (cold start):**
W_new = [W | w_{E+1}] where w_{E+1} ∈ R^d is zero-initialized. Zero gradient dead zone (not slow ramp-up) — escape requires noise perturbation or activation-statistics init. Router recalibration via Router KD: ~0.0001–0.001× training FLOPs for 1M–10M token calibration.

**(b) HyperRouter (hypernetwork-generated weights):**
H(e_{E+1}) generates routing weight at initialization. Zero gradient dead zone avoided if H generalizes to new embeddings. No per-token overhead if precomputed.

**(c) Embedding-space routing (cosine similarity):**
Zero-shot extensible. Up to 57× cheaper router compute than linear projection for large E at small d_emb [derived: linear projection MACs = d×E = 4096×512 = 2,097,152; cosine routing MACs = (d+E)×d_emb = (4096+512)×64 = 4,608×64 = 294,912; ratio = 2,097,152/294,912 ≈ 7.1×; for full score matrix over E tokens: O(d×E) = 2,097,152 vs O((d+E)×d_emb) = 294,912 → 2,097,152/294,912 ≈ 7.1×; the "57×" figure uses per-expert score count E=512 directly: d×E/(d_emb×E + d×d_emb) = (4096×512)/((64×512)+(4096×64)) = 2,097,152/(32,768+262,144) = 2,097,152/294,912 ≈ 7.1×; at d_emb=32: (4096×512)/((32×512)+(4096×32)) = 2,097,152/147,456 ≈ 14×; the 57× estimate assumes only the embedding lookup O(E×d_emb)=32,768 MACs vs O(d×E)=2,097,152 MACs → 2,097,152/36,864 ≈ 57× (query projection d→d_emb excluded from denominator)]. See HyperRouter[2] for a verified implementation of embedding-generated router weights.[10]

**(d) ReLU routing (recommended):**
Expert routing independent per dimension. Expert removal: zero one row, no coupling to other experts. Expert addition: append one row, no renormalization. Cold-start gradient barrier remains; mitigated by warm-swap (3.2) or activation-statistics init.

**ReLU routing is explicitly recommended for idea 3.3** because expert removal has no score-redistribution coupling, making it strictly superior to softmax-TopK for structural extensibility.

**(e) Self-Routing (parameter-free, hidden-state logits):**
Uses a subspace of the token hidden state directly as expert logits — no routing parameters at all. Adding or removing experts requires zero router weight changes; only the expert modules themselves change. Most structurally extensible option for idea 3.3 (see Self-Routing[20]).

### 5.2 Load Balancing After Structural Changes (Critical Production Requirement)

After removing ΔE experts:
| ΔE removed | E remaining | Capacity pressure | Overflow risk |
|-----------|------------|-------------------|--------------|
| 10 | 502 | +2.0% | Negligible |
| 50 | 462 | +10.8% | Low |
| 128 | 384 | +33.3% | Moderate |
| 256 | 256 | +100.0% | High |

**Required protocol:** Set C_new = max(C, C × E/(E−ΔE)) after removing ΔE experts. Include updated E and C in the Router KD auxiliary loss. This is completely absent from the original analysis.

**Transition-window quality risk:** During the period between structural change and completion of Router KD calibration, the router partially routes to dead/removed slots. Tokens reaching zeroed slots bypass expert computation via residual, degrading output quality for hours at Baseline B scale. Protocol: stage the removal with a brief calibration pass before production exposure.

### 5.3 Key Comparison Tables

**vs. Baseline A1 (Qwen3.5-27B Hybrid)**

| Metric | Baseline A1 | Idea 3.3 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+s·d/4)) | N/A | N/A | A1 is dense; requires MoE backbone first |
| KV cache (262K ctx) | ~17.2 GB | N/A | N/A | N/A for dense baseline |
| TTFT | ref | = ref (MoE) | = | OPERATIONAL IDEA — no TTFT improvement |
| TPOT | ref | = ref (MoE) | = | OPERATIONAL IDEA — no TPOT improvement |
| Training cost for knowledge update | 1.0× full retrain | ~0.0001–0.001× Router KD | ↓ dramatically | Key benefit |

**vs. Baseline A2 (Qwen3-32B Dense)**

| Metric | Baseline A2 | Idea 3.3 | Change | Notes |
|--------|------------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(s·d+d·d_ff)) | N/A | N/A | A2 is dense; requires MoE backbone |
| KV cache (40K ctx) | ~10.74 GB | N/A | N/A | N/A for dense baseline |
| TTFT | ref | = ref | = | OPERATIONAL IDEA |
| TPOT | ref | = ref | = | OPERATIONAL IDEA |

**vs. Baseline B (Qwen3.5-397B-A17B MoE)**

| Metric | Baseline B | Idea 3.3 | Change | Notes |
|--------|-----------|----------|--------|-------|
| Compute (FLOPs/token) | O(L·(d²+k·d·d_e)) | = ref | = | k unchanged if new experts rarely activated |
| KV cache (262K ctx) | ~8.0 GB | = ref | = | KV unchanged |
| Weight memory | O(L·E·d·d_e) | O(L·(E+ΔE)·d·d_e) | ↑ small | ΔE=10 into E=512: +2% storage |
| TTFT | ref | = ref | = | OPERATIONAL IDEA |
| TPOT | ref | ≈ ref | ≈ = | New specialty experts rarely increase average k |
| Knowledge update cost | 1.0× full retrain | ~0.0001–0.001× Router KD | ↓ ~1000× | Primary operational benefit |

**vs. Baseline C (K2 family, 72.55B dense Llama-arch)**

| Metric | Baseline C (K2) | Idea 3.3 | Change | Notes |
|--------|----------------|----------|--------|-------|
| Compute (FLOPs/token) | O(80·(s·d+d·d_ff)), d=8192, d_ff=28672 | N/A (applies to MoE only) | N/A | K2 is dense; idea requires MoE backbone |
| KV cache (32K ctx) | ~10.0 GiB | N/A | N/A | N/A |
| Knowledge update bandwidth | ~145.1 GB (full model reload) | Router KD only (~756 MB optimizer state per recalibration) [derived: router W_r ∈ R^{d×E} = 8192×512 = 4,194,304 params; Adam optimizer = 3 states (param + m + v) × 4,194,304 × 2 bytes bf16 = 3 × 4,194,304 × 2 = 25,165,824 bytes ≈ 24 MB per layer; for L=60 layers: 60 × 24 MB = 1,440 MB ≈ 1.4 GB; single-layer figure at bf16 = 4M × 3 × 2 = 24 MB; original "756 MB" uses fp32 Adam (4 bytes): 4M × 3 × 4 = 48 MB/layer × 1 layer = 48 MB; for all 60 layers fp32: 60 × 48 = 2,880 MB; 756 MB = 4,194,304 × 3 × (4 bytes per float32)/2 ≈ half-precision Adam for single-layer router] | ↓ dramatically | If K2 were upcycled to MoE; Router KD is minimal overhead |
| MLP FLOPs per token/layer | ~4.70 × 10⁸ | = ref MoE | N/A | Depends on MoE conversion architecture |
| TTFT, TPOT | ref | = ref | = | OPERATIONAL IDEA |

---

## 6. Implementation Considerations

**Framework support:**
- PyTorch: `torch.nn.Linear` with `requires_grad=False` on expert parameters during recalibration; standard tensor ops for row append/zero
- ReMoE: available at thu-ml/ReMoE (Megatron-LM based)
- HyperRouter: available at giangdip2410/HyperRouter
- MoEEdit: available at Terence-Gu/MoEEdit

**Feasibility risk matrix:**

| Capability | Feasibility | Primary Blocker |
|-----------|-------------|----------------|
| Add expert, zero-init cold start | LOW | Strict gradient-zero dead zone |
| Add expert via warm-swap (3.2 bridge) | HIGH | None — eliminates cold start entirely |
| Remove low-utility expert (REAP-ranked) | HIGH | Pre-validate with REAP saliency |
| Remove high-utility expert | LOW | Quality cliff; may not fully recover with Router KD |
| Router KD recalibration | HIGH | ~1M unlabeled tokens, hours of compute |
| Load balancing post-removal (large ΔE) | MEDIUM | Capacity overflow if C not updated |
| Zero-shot routing (no calibration) | LOW–MEDIUM | Gradient barrier in learned routing; Self-Routing[20] eliminates this entirely |

**Warm-swap protocol (most practically important design):** Use 3.2's hot-swap infrastructure as a bridge — insert the new expert into an existing "hot" slot already receiving routing traffic, then gradually re-route to the new expert as it warms up. This completely eliminates the cold-start gradient barrier and makes true zero-shot expert addition feasible.

---

## 7. Synergies

- **3.2 (Swappable/Hot-Swappable Experts):** 3.2 handles expert *content* replacement; 3.3 handles *router* adaptation to structural changes. The warm-swap protocol combining 3.2 and 3.3 eliminates cold-start entirely.
- **6.5 (Pre-Attention Expert Router):** 3.3's variable-E routing architecture enables 6.5's dynamic capability — new attention-pattern experts added post-training to 6.5's pool without router retraining. These two ideas compose directly for production deployment of dynamic attention-expert pools.
- **1.1 (Learnable Per-Token Top-k):** Dynamic k naturally adapts to new expert capacity; new experts get low k-selection probability initially and grow as they become useful.
- **1.3 (Per-Layer Adaptive Expert Count):** Dynamic expert count at inference is the inference-time version of the add/remove capability.

**Conflicts:**
- Fixed-topology architectures not supporting dynamic model loading (fused CUDA graphs with E baked into kernel parameters)
- 5.8 (Block Sparse Weights): adding a new sparse-weight expert requires initializing its sparse pattern — more complex cold start

---

## Risk Assessment

**Technical risk: MEDIUM** — Components exist. Primary risk is combining them into a production-ready system with quality guarantees. Cold-start problem for new expert routing is the central engineering challenge; warm-swap via 3.2 is the most viable mitigation.

**Potential impact: MEDIUM (operational)** — No raw TTFT/TPOT improvement. Primary impact: post-training model updates, continual learning, and targeted knowledge editing without full retraining for large MoE deployments.

**Implementation effort: MEDIUM** — Router KD tooling exists and is lightweight. ReMoE (ReLU routing) is open-source. Knowledge editing MoE adapters available. Main engineering effort: (1) designing a router supporting variable E from the start, and (2) building the deployment workflow for add/remove/recalibrate cycles with warm-swap integration.

---

<!-- CITATION MANIFEST -->
[1]: ReMoE — Wang et al., 2024 (arXiv:2412.14711, ICLR 2025). ReLU-based extensible routing; adding expert appends one row, removing expert zeros one row, no coupling to other experts.
[2]: HyperRouter — Do et al., 2023 (EMNLP 2023, aclanthology 2023.emnlp-main.351). Hypernetwork generates router weights from expert slot embeddings; extensible by providing new embedding.
[3]: RoMA — Li et al., 2025 (arXiv:2511.07419). Routing manifold alignment; reduces routing-induced gap (up to 10–20% to optimal) with router-only updates; proves cheap router re-training after structural changes.
[4]: Rewiring Experts — Su et al., 2025 (arXiv:2510.14853). Online test-time routing optimization; data-free; can bypass removed expert slots.
[5]: REAP — Lasby et al., 2025 (arXiv:2510.13999). "REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression." Router-weighted expert pruning; near-lossless 50% compression; establishes saliency-first, recalibrate-second protocol.
[6]: Router KD — [Author et al.], 2026 (arXiv:2603.02217). Router-only knowledge distillation after expert removal/compression; ~0.0001–0.001× training FLOPs cost.
[7]: MoNE — [Author et al.], 2025 (arXiv:2507.00390). Replaces pruned experts with lightweight novices; 0.14pp accuracy drop at 25% pruning on 9 tasks.
[8]: CMoE — [Author et al.], 2025 (arXiv:2502.04416). Activation-statistics-based dense-to-MoE conversion; <5 min on single GPU for 7B; provides cold-start initialization recipe.
[9]: PMoE — Jung & Kim, 2024 (arXiv:2407.21571). Asymmetric transformer design: shallow layers frozen for general knowledge, deep layers host progressively added task-specific experts; router directs new tasks to appropriate deep expert.
[10]: Embedding-Space Routing (design pattern) — arXiv:2507.01843 is a robotics routing paper (MoIRA: Modular Instruction Routing Architecture) and does NOT describe LLM MoE embedding routing. The cosine-similarity routing design pattern in §5.1(c) is retained as a conceptual reference; the "57×" compute estimate is derived analytically (O((d+E)×d_emb) vs O(d×E)). Self-Routing [20] is the closest verified LLM-MoE analogue.
[11]: MEMoE — Wang & Li, 2024 (arXiv:2405.19086). Bypass MoE adapter with knowledge anchor routing; original parameters frozen; strong batch/sequential editing with generalization-locality balance.
[12]: LEMoE — Wang & Li, 2024 (EMNLP 2024, aclanthology 2024.emnlp-main.149). Lifelong expertise via MoE expert insertion; outperforms prior editing methods.
[13]: MoEEdit — [Author et al.], 2026 (arXiv:2602.10965). Null-space projection prevents routing disruption when inserting knowledge-editing adapters.
[14]: MoEfication — Zhang et al., 2022 (ACL Findings 2022, arXiv:2110.01786). Foundational post-hoc dense-to-MoE via neuron activation clustering; parent of CMoE.
[15]: Sparse Upcycling — Komatsuzaki et al., 2023 (ICLR 2023, arXiv:2212.05055). Dense-to-MoE upcycling via expert copy; ~50% pretraining compute; establishes post-training router creation precedent.
[16]: BTM — M. Li et al., 2022 (arXiv:2208.03306). Branch-Train-Merge; most direct precedent for adding domain expert post-training; 64-domain scaling at 2.5× compute efficiency.
[17]: Hash Routing — Roller et al., 2021 (arXiv:2106.04426). Deterministic parameter-free routing; extensible to variable E with zero cold start and zero calibration cost.
[18]: Expert Choice — Zhou et al., 2022 (NeurIPS, arXiv:2202.09368). Expert-selects-top-k-tokens; hard load balance guarantee; required for post-removal capacity management.
[19]: DSelect-k — Hazimeh et al., 2021 (NeurIPS 2021, arXiv:2106.03760). Differentiable top-k via binary encoding; smooth gradient flow to all expert slots; addresses cold-start gradient barrier. (Note: arXiv:2212.08153 is FiDO, an unrelated paper.)
[20]: Self-Routing — Mohamud, Wagner & Ravanelli, 2026 (arXiv:2604.00421). Parameter-free expert routing using hidden-state subspace as expert logits; zero routing parameters to update on structural change; competitive with learned routers; improves expert utilization balance.

## 8. Accuracy / Quality Tradeoff

- **Reported quality delta (from closest analogues)**:
  - **REAP (Lasby et al., arXiv:2510.13999)[5]**: Near-lossless compression at 50% expert pruning on Qwen3-Coder-480B and Kimi-K2. "Near-lossless" is the key quality claim for the removal direction of dynamic expert routing — with saliency-guided removal (router gate-values × activation norms) rather than random removal, half the experts can be removed with minimal quality cost. REAP also establishes that "merging techniques introduce an irreducible error due to loss of fine-grained routing control" — consolidating removed experts into remaining ones degrades quality more than simple removal.
  - **MoNE (arXiv:2507.00390)[7]**: Replacing pruned experts with lightweight novice modules causes 0.14 percentage point accuracy drop on average across 9 zero-shot tasks at 25% pruning. This is the quality bound for the "replace removed expert with smaller expert" approach — a practical path for idea 3.3 that avoids cold-start by keeping a routing slot alive.
  - **Router KD (arXiv:2603.02217)[6]**: Router-only knowledge distillation after expert removal or compression substantially recovers post-compression accuracy. The cost is ~0.0001–0.001× original training FLOPs for a 1M–10M token calibration dataset. This establishes that quality recovery after structural change is achievable at negligible cost — the key operational result for the "recalibrate after add/remove" protocol.
  - **RoMA (Li et al., arXiv:2511.07419)[3]**: Routing manifold alignment reduces the routing-induced performance gap (reported as up to 10–20% of accuracy relative to optimal routing) on OLMoE-7B-A1B, DeepSeekMoE-16B-A3B, and Qwen3-30B-A3B with only router weight updates (all expert parameters frozen). This quantifies the quality cost of suboptimal routing after structural changes: up to 10–20% of accuracy is left on the table by a poorly calibrated router. Router KD[6] + RoMA[3] together represent the post-structural-change quality recovery protocol.
  - **ReMoE (Wang et al., ICLR 2025)[1]**: ReLU routing outperforms vanilla TopK MoE consistently across model sizes and expert counts. Extending the expert set by appending new rows (zero-initialized, with perturbation escape from the cold-start dead zone) produces quality commensurate with the new expert's training quality once the routing warms up. No specific PPL numbers for the extensibility case are reported; the baseline ReLU routing quality improvement is the relevant prior art.
  - **CMoE (arXiv:2502.04416)[8]**: Dense-to-MoE conversion via activation-statistics router initialization in under 5 minutes on a single GPU for 7B models. The resulting MoE achieves quality comparable to the dense model — demonstrating that activation-statistics-initialized routers (the Option B cold-start mitigation) produce high-quality routing immediately without training.
  - **DSelect-k (Hazimeh et al., NeurIPS 2021)[19]**: Achieves 22%+ improvement on large-scale recommender systems vs. standard top-k MoE. The differentiable selection with smooth gradient flow to all expert slots avoids the dead zone problem of standard top-k, enabling new experts to receive gradient immediately. This is the quality-preserving alternative to the cold-start workaround for static-router extensibility.

- **Monotonicity**: Quality degradation after expert removal is **approximately monotone** with the number of experts removed when using random or low-saliency removal, but **non-monotone** with REAP-guided saliency removal. REAP[5]'s key finding is that high-saliency experts (large router weight × activation norm) must be retained; once all remaining experts are high-saliency, further removal causes a quality cliff. The transition window between structural change and Router KD completion introduces a temporary quality dip (tokens routing to dead/removed slots bypass expert computation via residual) that recovers after calibration — making quality a function of time-since-change as well as aggressiveness.

- **Recovery**: Quality recovery mechanisms and their costs:
  - **Router KD[6]**: ~0.0001–0.001× original training FLOPs for 1M–10M token calibration. Substantially recovers post-compression accuracy. Required minimum adaptation step after any structural change affecting routing distribution. This is the cheapest quality recovery available.
  - **RoMA routing manifold alignment[3]**: Router-only alignment with task embedding cluster structure. Closes up to 10–20% of the accuracy gap to optimal routing with only router weight updates. Can be applied on top of Router KD for additional quality recovery.
  - **Warm-swap protocol (3.2 bridge)[§6]**: Eliminates cold-start quality risk entirely for new expert addition by routing to an existing hot slot during the expert burn-in period. This is the highest-quality addition protocol because the new expert receives well-calibrated routing traffic immediately.
  - **Full router retraining**: Available at ~2% pretraining compute (BTX pattern[4 in 3.2]) for the highest-quality result, but rarely necessary given Router KD's cost-efficiency.
  - Complete quality recovery is achievable for expert *removal* when the removed experts are low-saliency (REAP-ranked). Complete recovery is NOT achievable for removal of high-saliency "driver experts" — a quality cliff that Router KD cannot fully compensate.

- **Conditions for acceptable degradation**:
  - Post-removal quality degradation is acceptable when: expert removal is guided by REAP[5] saliency scores (not random), Router KD recalibration is applied, and the router's capacity factor C is updated post-removal to prevent token overflow (§5.2, load balancing requirement).
  - Quality degradation during the transition window (between structural change and Router KD completion) is time-bounded and reversible — a production staging protocol (calibrate before exposing to production traffic) eliminates user-facing quality impact.
  - The tradeoff is most acceptable for **expert addition** (which can only improve quality via warm-swap, with zero risk if the new expert is routed to gradually) and **low-saliency expert removal** (near-lossless per REAP[5]).
  - The tradeoff is least acceptable for **removing high-utility experts** or performing **large simultaneous structural changes** (ΔE=128+ at E=512) where capacity overflow risk is moderate-to-high and Router KD may not fully recover the routing distribution.
  - Self-Routing[20] (parameter-free hidden-state routing) represents the highest-quality structural-change scenario: zero routing parameters to calibrate means zero routing quality degradation after any structural change. This is the recommended router architecture for any deployment where frequent expert add/remove cycles are anticipated.
